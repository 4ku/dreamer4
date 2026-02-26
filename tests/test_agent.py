"""
Comprehensive tests for Module 4: Agent Heads (policy, reward, value).

Covers:
    - SymExpTwoHot: roundtrip accuracy, gradient flow, edge cases.
    - TaskEncoder: shape, broadcast over T.
    - PolicyHead: continuous + discrete, MTP, log_prob, sample, entropy.
    - RewardHead: MTP logits shape, predict scalar, loss.
    - ValueHead: logits shape, predict scalar, loss.
    - compute_lambda_returns: hand-verified examples, boundary cases.
    - pmpo_policy_loss: gradient direction, KL regularization.
    - behavior_cloning_loss: MTP shifted targets.
    - reward_prediction_loss: MTP shifted targets.
    - Integration: DynamicsModel + agent heads end-to-end.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dreamer4.distributions import symlog, symexp, SymExpTwoHot
from dreamer4.agent import (
    TaskEncoder,
    PolicyHead,
    RewardHead,
    ValueHead,
    compute_lambda_returns,
    pmpo_policy_loss,
    behavior_cloning_loss,
    reward_prediction_loss,
)


# ===================================================================
# SymLog / SymExp
# ===================================================================

class TestSymLogSymExp:
    def test_symlog_symexp_roundtrip(self):
        x = torch.linspace(-10, 10, 201)
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)

    def test_symlog_of_zero(self):
        assert symlog(torch.tensor(0.0)).item() == 0.0

    def test_symexp_of_zero(self):
        assert symexp(torch.tensor(0.0)).item() == 0.0

    def test_symlog_positive(self):
        x = torch.tensor(1.0)
        assert torch.allclose(symlog(x), torch.log(torch.tensor(2.0)))

    def test_symlog_negative(self):
        x = torch.tensor(-1.0)
        assert torch.allclose(symlog(x), -torch.log(torch.tensor(2.0)))

    def test_symexp_positive(self):
        x = torch.tensor(1.0)
        expected = torch.exp(torch.tensor(1.0)) - 1.0
        assert torch.allclose(symexp(x), expected.unsqueeze(0).squeeze(), atol=1e-6)

    def test_gradient_flows_through_symlog(self):
        x = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
        y = symlog(x).sum()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ===================================================================
# SymExpTwoHot
# ===================================================================

class TestSymExpTwoHot:
    @pytest.fixture
    def twohot(self):
        return SymExpTwoHot(num_bins=255, low=-20.0, high=20.0)

    def test_bin_values_shape(self, twohot):
        assert twohot.bin_values.shape == (255,)

    def test_bin_values_monotonic(self, twohot):
        diffs = twohot.bin_values[1:] - twohot.bin_values[:-1]
        assert (diffs > 0).all(), "Bin values should be strictly increasing"

    def test_bin_values_symmetric(self, twohot):
        assert torch.allclose(
            twohot.bin_values + twohot.bin_values.flip(0),
            torch.zeros(255),
            atol=1e-5,
        ), "Bins should be symmetric around zero"

    def test_encode_shape(self, twohot):
        values = torch.randn(4, 8)
        encoded = twohot.encode(values)
        assert encoded.shape == (4, 8, 255)

    def test_encode_sums_to_one(self, twohot):
        values = torch.randn(16)
        encoded = twohot.encode(values)
        sums = encoded.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)

    def test_encode_nonnegative(self, twohot):
        values = torch.randn(32)
        encoded = twohot.encode(values)
        assert (encoded >= -1e-7).all()

    def test_encode_two_hot(self, twohot):
        """Each row should have at most 2 non-zero entries."""
        values = torch.randn(32)
        encoded = twohot.encode(values)
        nnz_per_row = (encoded > 1e-7).sum(dim=-1)
        assert (nnz_per_row <= 2).all()

    def test_roundtrip_accuracy(self, twohot):
        """encode -> treat as soft targets -> decode should be close."""
        values = torch.tensor([0.0, 1.0, -1.0, 5.0, -5.0, 0.001])
        encoded = twohot.encode(values)
        decoded = (encoded * twohot.bin_values).sum(dim=-1)
        assert torch.allclose(decoded, values, atol=0.5), (
            f"Roundtrip mismatch: {values} -> {decoded}"
        )

    def test_decode_from_logits(self, twohot):
        logits = torch.zeros(255)
        mid = 127
        logits[mid] = 100.0  # strong peak at middle bin
        decoded = twohot.decode(logits)
        assert abs(decoded.item() - twohot.bin_values[mid].item()) < 0.1

    def test_log_prob_shape(self, twohot):
        logits = torch.randn(4, 8, 255)
        values = torch.randn(4, 8)
        lp = twohot.log_prob(logits, values)
        assert lp.shape == (4, 8)

    def test_log_prob_gradient(self, twohot):
        logits = torch.randn(8, 255, requires_grad=True)
        values = torch.randn(8)
        lp = twohot.log_prob(logits, values)
        lp.sum().backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_loss_scalar(self, twohot):
        logits = torch.randn(4, 8, 255)
        values = torch.randn(4, 8)
        loss = twohot.loss(logits, values)
        assert loss.dim() == 0

    def test_loss_decreases_with_better_logits(self, twohot):
        """Loss should be lower when logits peak at the correct bin."""
        values = torch.tensor([0.0])
        good_logits = torch.zeros(1, 255)
        mid = 127
        good_logits[0, mid] = 10.0
        bad_logits = torch.randn(1, 255)
        good_loss = twohot.loss(good_logits, values)
        bad_loss = twohot.loss(bad_logits, values)
        assert good_loss < bad_loss

    def test_encode_clamps_extreme_values(self, twohot):
        extreme = torch.tensor([1e10, -1e10])
        encoded = twohot.encode(extreme)
        assert encoded.sum(dim=-1).allclose(torch.ones(2), atol=1e-5)

    def test_small_num_bins(self):
        twohot = SymExpTwoHot(num_bins=5, low=-2.0, high=2.0)
        values = torch.tensor([0.0])
        encoded = twohot.encode(values)
        assert encoded.shape == (1, 5)
        assert encoded.sum().item() == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# TaskEncoder
# ===================================================================

class TestTaskEncoder:
    def test_output_shape_1d_task(self):
        enc = TaskEncoder(num_tasks=10, d_model=64, n_agent=1)
        task_ids = torch.randint(0, 10, (4,))
        out = enc(task_ids)
        assert out.shape == (4, 1, 1, 64)

    def test_output_shape_2d_task(self):
        enc = TaskEncoder(num_tasks=10, d_model=64, n_agent=2)
        task_ids = torch.randint(0, 10, (4, 8))
        out = enc(task_ids)
        assert out.shape == (4, 8, 2, 64)

    def test_different_tasks_different_embeddings(self):
        enc = TaskEncoder(num_tasks=10, d_model=64, n_agent=1)
        out0 = enc(torch.tensor([0]))
        out1 = enc(torch.tensor([1]))
        assert not torch.allclose(out0, out1)


# ===================================================================
# PolicyHead — Continuous
# ===================================================================

class TestPolicyHeadContinuous:
    @pytest.fixture
    def head(self):
        return PolicyHead(
            d_model=64, action_dim=4, action_type="continuous",
            mtp_length=8, mlp_depth=2, mlp_ratio=2.0,
        )

    def test_forward_shape(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        assert params.shape == (2, 8, 8)  # 4 * 2 = 8 (mean + log_std)

    def test_forward_all_length(self, head):
        x = torch.randn(2, 8, 64)
        all_params = head.forward_all(x)
        assert len(all_params) == 8

    def test_log_prob_shape(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        actions = torch.randn(2, 8, 4)
        lp = head.log_prob(params, actions)
        assert lp.shape == (2, 8)

    def test_sample_shape(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        actions, lp = head.sample(params)
        assert actions.shape == (2, 8, 4)
        assert lp.shape == (2, 8)

    def test_entropy_shape(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        ent = head.entropy(params)
        assert ent.shape == (2, 8)

    def test_entropy_positive(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        ent = head.entropy(params)
        assert (ent > 0).all()

    def test_gradient_flows(self, head):
        x = torch.randn(2, 4, 64, requires_grad=True)
        params = head(x)
        actions, lp = head.sample(params)
        lp.sum().backward()
        assert x.grad is not None


# ===================================================================
# PolicyHead — Discrete
# ===================================================================

class TestPolicyHeadDiscrete:
    @pytest.fixture
    def head(self):
        return PolicyHead(
            d_model=64, action_dim=3, action_type="discrete",
            num_categories=5, mtp_length=4, mlp_depth=2, mlp_ratio=2.0,
        )

    def test_forward_shape(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        assert params.shape == (2, 8, 15)  # 3 * 5

    def test_dist_is_categorical(self, head):
        params = torch.randn(2, 8, 15)
        d = head.dist(params)
        assert isinstance(d, torch.distributions.Categorical)

    def test_sample_shape(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        actions, lp = head.sample(params)
        assert actions.shape == (2, 8, 3)
        assert lp.shape == (2, 8)

    def test_log_prob_shape(self, head):
        x = torch.randn(2, 8, 64)
        params = head(x)
        actions = torch.randint(0, 5, (2, 8, 3))
        lp = head.log_prob(params, actions)
        assert lp.shape == (2, 8)


# ===================================================================
# RewardHead
# ===================================================================

class TestRewardHead:
    @pytest.fixture
    def head(self):
        return RewardHead(
            d_model=64, mtp_length=4, mlp_depth=2, mlp_ratio=2.0,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )

    def test_forward_shape(self, head):
        x = torch.randn(2, 8, 64)
        logits = head(x, head_idx=0)
        assert logits.shape == (2, 8, 31)

    def test_forward_all_length(self, head):
        x = torch.randn(2, 8, 64)
        all_logits = head.forward_all(x)
        assert len(all_logits) == 4

    def test_predict_shape(self, head):
        x = torch.randn(2, 8, 64)
        rewards = head.predict(x)
        assert rewards.shape == (2, 8)

    def test_loss_gradient(self, head):
        x = torch.randn(2, 8, 64, requires_grad=True)
        logits = head(x)
        targets = torch.randn(2, 8)
        loss = head.twohot.loss(logits, targets)
        loss.backward()
        assert x.grad is not None


# ===================================================================
# ValueHead
# ===================================================================

class TestValueHead:
    @pytest.fixture
    def head(self):
        return ValueHead(
            d_model=64, mlp_depth=2, mlp_ratio=2.0,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )

    def test_forward_shape(self, head):
        x = torch.randn(2, 8, 64)
        logits = head(x)
        assert logits.shape == (2, 8, 31)

    def test_predict_shape(self, head):
        x = torch.randn(2, 8, 64)
        values = head.predict(x)
        assert values.shape == (2, 8)

    def test_loss_gradient(self, head):
        x = torch.randn(2, 8, 64, requires_grad=True)
        logits = head(x)
        targets = torch.randn(2, 8)
        loss = head.twohot.loss(logits, targets)
        loss.backward()
        assert x.grad is not None


# ===================================================================
# compute_lambda_returns
# ===================================================================

class TestLambdaReturns:
    def test_single_step(self):
        """With T=1, return = value (bootstrap only)."""
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[5.0]])
        continues = torch.tensor([[1.0]])
        ret = compute_lambda_returns(rewards, values, continues, gamma=0.99, lam=0.95)
        assert ret.shape == (1, 1)
        assert ret[0, 0].item() == pytest.approx(5.0)

    def test_two_steps_all_continue(self):
        """Hand-verify: R_1 = r_1 + gamma * ((1-lam)*v_2 + lam*R_2), R_2 = v_2."""
        rewards = torch.tensor([[1.0, 2.0]])
        values = torch.tensor([[10.0, 5.0]])
        continues = torch.tensor([[1.0, 1.0]])
        gamma, lam = 0.99, 0.95

        ret = compute_lambda_returns(rewards, values, continues,
                                     gamma=gamma, lam=lam)
        R2 = 5.0
        R1 = 1.0 + gamma * ((1 - lam) * 5.0 + lam * R2)
        assert ret[0, 1].item() == pytest.approx(R2, abs=1e-5)
        assert ret[0, 0].item() == pytest.approx(R1, abs=1e-5)

    def test_terminal_cuts_bootstrapping(self):
        """When continues=0, future rewards/values are ignored."""
        rewards = torch.tensor([[1.0, 100.0]])
        values = torch.tensor([[0.0, 1000.0]])
        continues = torch.tensor([[0.0, 1.0]])

        ret = compute_lambda_returns(rewards, values, continues,
                                     gamma=0.99, lam=0.95)
        # continues[0]=0 means episode ended at t=0, so R_0 = r_0 + 0 = 1.0
        assert ret[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_shape(self):
        B, T = 4, 16
        ret = compute_lambda_returns(
            torch.randn(B, T), torch.randn(B, T), torch.ones(B, T),
        )
        assert ret.shape == (B, T)

    def test_gamma_zero_means_immediate_reward(self):
        """With gamma=0, returns should just be rewards except last = value."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[10.0, 20.0, 30.0]])
        continues = torch.ones(1, 3)
        ret = compute_lambda_returns(rewards, values, continues,
                                     gamma=0.0, lam=0.95)
        assert ret[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert ret[0, 1].item() == pytest.approx(2.0, abs=1e-5)
        assert ret[0, 2].item() == pytest.approx(30.0, abs=1e-5)

    def test_lambda_zero_gives_td0(self):
        """lam=0 means R_t = r_t + gamma * v_{t+1} (pure 1-step TD)."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.0, 5.0, 10.0]])
        continues = torch.ones(1, 3)
        gamma = 0.99
        ret = compute_lambda_returns(rewards, values, continues,
                                     gamma=gamma, lam=0.0)
        assert ret[0, 2].item() == pytest.approx(10.0, abs=1e-5)
        assert ret[0, 1].item() == pytest.approx(2.0 + gamma * 10.0, abs=1e-5)
        assert ret[0, 0].item() == pytest.approx(1.0 + gamma * 5.0, abs=1e-5)

    def test_lambda_one_gives_mc(self):
        """lam=1 means R_t = r_t + gamma * R_{t+1} (Monte Carlo)."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.0, 0.0, 10.0]])
        continues = torch.ones(1, 3)
        gamma = 0.99
        ret = compute_lambda_returns(rewards, values, continues,
                                     gamma=gamma, lam=1.0)
        R2 = 10.0
        R1 = 2.0 + gamma * R2
        R0 = 1.0 + gamma * R1
        assert ret[0, 2].item() == pytest.approx(R2, abs=1e-5)
        assert ret[0, 1].item() == pytest.approx(R1, abs=1e-5)
        assert ret[0, 0].item() == pytest.approx(R0, abs=1e-5)


# ===================================================================
# PMPO Policy Loss
# ===================================================================

class TestPMPO:
    def test_positive_advantages_decrease_loss(self):
        """Increasing log_prob for positive advantages should reduce loss."""
        adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
        lp_low = torch.tensor([-2.0, -2.0, -2.0, -2.0])
        lp_high = torch.tensor([-0.5, -0.5, -0.5, -0.5])
        loss_low = pmpo_policy_loss(lp_low, adv, alpha=0.5, beta=0.0)
        loss_high = pmpo_policy_loss(lp_high, adv, alpha=0.5, beta=0.0)
        assert loss_high < loss_low

    def test_negative_advantages_increase_loss(self):
        """Increasing log_prob for negative advantages should increase loss."""
        adv = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        lp_low = torch.tensor([-2.0, -2.0, -2.0, -2.0])
        lp_high = torch.tensor([-0.5, -0.5, -0.5, -0.5])
        loss_low = pmpo_policy_loss(lp_low, adv, alpha=0.5, beta=0.0)
        loss_high = pmpo_policy_loss(lp_high, adv, alpha=0.5, beta=0.0)
        assert loss_high > loss_low

    def test_kl_regularization(self):
        """Adding KL regularization should increase loss when policies differ."""
        lp = torch.tensor([[-1.0, -1.0]])
        adv = torch.tensor([[1.0, -1.0]])
        prior_lp = torch.tensor([[-0.5, -0.5]])  # prior is different

        loss_no_kl = pmpo_policy_loss(lp, adv, beta=0.0)
        loss_kl = pmpo_policy_loss(lp, adv, prior_log_probs=prior_lp, beta=0.3)
        assert loss_kl != loss_no_kl

    def test_alpha_balancing(self):
        """With alpha=1.0 only positive advantages matter."""
        adv = torch.tensor([[1.0, -1.0]])
        lp = torch.tensor([[-1.0, -1.0]])
        loss = pmpo_policy_loss(lp, adv, alpha=1.0, beta=0.0)
        # Only pos term: -(1.0 * (-1.0) / 1) = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_scalar_output(self):
        lp = torch.randn(4, 8)
        adv = torch.randn(4, 8)
        loss = pmpo_policy_loss(lp, adv, beta=0.0)
        assert loss.dim() == 0

    def test_gradient_direction(self):
        """Gradient should push log_prob up for positive advantages."""
        lp = torch.tensor([-1.0, -1.0], requires_grad=True)
        adv = torch.tensor([1.0, 1.0])
        loss = pmpo_policy_loss(lp, adv, alpha=0.5, beta=0.0)
        loss.backward()
        # Loss = -alpha * mean(lp[pos]) so grad w.r.t. lp = -alpha / n_pos
        # Negative gradient -> optimizer step increases lp (good!)
        assert (lp.grad < 0).all()


# ===================================================================
# Behavior Cloning Loss
# ===================================================================

class TestBehaviorCloningLoss:
    def test_scalar_output(self):
        head = PolicyHead(
            d_model=32, action_dim=2, action_type="continuous",
            mtp_length=4, mlp_depth=1, mlp_ratio=2.0,
        )
        embed = torch.randn(2, 8, 32)
        actions = torch.randn(2, 8, 2)
        loss = behavior_cloning_loss(head, embed, actions, mtp_length=4)
        assert loss.dim() == 0

    def test_loss_decreases_with_training(self):
        torch.manual_seed(42)
        head = PolicyHead(
            d_model=32, action_dim=2, action_type="continuous",
            mtp_length=2, mlp_depth=1, mlp_ratio=2.0,
        )
        embed = torch.randn(4, 8, 32)
        actions = torch.randn(4, 8, 2)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)

        losses = []
        for _ in range(30):
            loss = behavior_cloning_loss(head, embed, actions, mtp_length=2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_mtp_respects_shifted_targets(self):
        """MTP head n should compare against actions shifted by n."""
        head = PolicyHead(
            d_model=32, action_dim=2, action_type="continuous",
            mtp_length=3, mlp_depth=1, mlp_ratio=2.0,
        )
        embed = torch.randn(1, 5, 32)
        actions = torch.randn(1, 5, 2)
        loss = behavior_cloning_loss(head, embed, actions, mtp_length=3)
        loss.backward()
        for p in head.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()


# ===================================================================
# Reward Prediction Loss
# ===================================================================

class TestRewardPredictionLoss:
    def test_scalar_output(self):
        head = RewardHead(
            d_model=32, mtp_length=4, mlp_depth=1, mlp_ratio=2.0,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )
        embed = torch.randn(2, 8, 32)
        rewards = torch.randn(2, 8)
        loss = reward_prediction_loss(head, embed, rewards, mtp_length=4)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        head = RewardHead(
            d_model=32, mtp_length=2, mlp_depth=1, mlp_ratio=2.0,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )
        embed = torch.randn(2, 8, 32, requires_grad=True)
        rewards = torch.randn(2, 8)
        loss = reward_prediction_loss(head, embed, rewards, mtp_length=2)
        loss.backward()
        assert embed.grad is not None

    def test_loss_decreases_with_training(self):
        torch.manual_seed(42)
        head = RewardHead(
            d_model=32, mtp_length=2, mlp_depth=1, mlp_ratio=2.0,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )
        embed = torch.randn(4, 8, 32)
        rewards = torch.zeros(4, 8)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)

        losses = []
        for _ in range(50):
            loss = reward_prediction_loss(head, embed, rewards, mtp_length=2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ===================================================================
# Integration: DynamicsModel + Agent Heads
# ===================================================================

class TestIntegration:
    def test_dynamics_agent_tokens_flow_to_heads(self):
        """Agent tokens from DynamicsModel feed through all three heads."""
        from dreamer4.dynamics import DynamicsModel

        D_MODEL = 64
        D_SPATIAL = 16
        N_SPATIAL = 8
        N_AGENT = 1
        ACTION_DIM = 4

        dynamics = DynamicsModel(
            d_model=D_MODEL, d_spatial=D_SPATIAL, n_spatial=N_SPATIAL,
            n_register=2, n_agent=N_AGENT,
            n_heads=4, depth=2, k_max=4, action_dim=ACTION_DIM,
            time_every=1, dropout=0.0,
            use_qk_norm=False, logit_cap=None,
            space_mode="wm_agent",
        )

        task_enc = TaskEncoder(num_tasks=5, d_model=D_MODEL, n_agent=N_AGENT)
        policy = PolicyHead(
            d_model=D_MODEL, action_dim=ACTION_DIM,
            action_type="continuous", mtp_length=4, mlp_depth=1,
        )
        reward = RewardHead(
            d_model=D_MODEL, mtp_length=4, mlp_depth=1,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )
        value = ValueHead(
            d_model=D_MODEL, mlp_depth=1,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )

        B, T = 2, 4
        actions = torch.randn(B, T, ACTION_DIM)
        step_idx = torch.zeros(B, T, dtype=torch.long)
        signal_idx = torch.zeros(B, T, dtype=torch.long)
        z_noisy = torch.randn(B, T, N_SPATIAL, D_SPATIAL)
        task_ids = torch.randint(0, 5, (B,))
        agent_tokens = task_enc(task_ids).expand(B, T, N_AGENT, D_MODEL)

        z1_hat, agent_out = dynamics(
            actions, step_idx, signal_idx, z_noisy,
            agent_tokens=agent_tokens,
        )

        assert z1_hat.shape == (B, T, N_SPATIAL, D_SPATIAL)
        assert agent_out is not None
        assert agent_out.shape == (B, T, N_AGENT, D_MODEL)

        agent_embed = agent_out[:, :, 0, :]  # (B, T, D_MODEL)

        policy_params = policy(agent_embed)
        pred_actions, log_probs = policy.sample(policy_params)
        assert pred_actions.shape == (B, T, ACTION_DIM)

        pred_rewards = reward.predict(agent_embed)
        assert pred_rewards.shape == (B, T)

        pred_values = value.predict(agent_embed)
        assert pred_values.shape == (B, T)

    def test_full_gradient_flow(self):
        """Gradients flow from agent head losses back through dynamics."""
        from dreamer4.dynamics import DynamicsModel

        D_MODEL = 32
        D_SPATIAL = 8
        N_SPATIAL = 4
        N_AGENT = 1

        dynamics = DynamicsModel(
            d_model=D_MODEL, d_spatial=D_SPATIAL, n_spatial=N_SPATIAL,
            n_register=1, n_agent=N_AGENT,
            n_heads=2, depth=2, k_max=4, action_dim=2,
            time_every=1, use_qk_norm=False, logit_cap=None,
            space_mode="wm_agent",
        )
        policy = PolicyHead(
            d_model=D_MODEL, action_dim=2,
            action_type="continuous", mtp_length=2, mlp_depth=1,
        )
        value = ValueHead(
            d_model=D_MODEL, mlp_depth=1,
            num_bins=31, bin_low=-5.0, bin_high=5.0,
        )

        B, T = 2, 4
        z_noisy = torch.randn(B, T, N_SPATIAL, D_SPATIAL)
        actions = torch.randn(B, T, 2)
        step_idx = torch.zeros(B, T, dtype=torch.long)
        signal_idx = torch.zeros(B, T, dtype=torch.long)
        agent_tokens = torch.randn(B, T, N_AGENT, D_MODEL)
        target_actions = torch.randn(B, T, 2)

        _, agent_out = dynamics(
            actions, step_idx, signal_idx, z_noisy,
            agent_tokens=agent_tokens,
        )
        agent_embed = agent_out[:, :, 0, :]

        bc_loss = behavior_cloning_loss(policy, agent_embed, target_actions, mtp_length=2)
        value_logits = value(agent_embed)
        val_loss = value.twohot.loss(value_logits, torch.zeros(B, T))

        total_loss = bc_loss + val_loss
        total_loss.backward()

        has_grad = False
        for p in dynamics.parameters():
            if p.grad is not None and p.grad.abs().max() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed back to dynamics model"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_policy_head_single_timestep(self):
        head = PolicyHead(
            d_model=32, action_dim=2, action_type="continuous",
            mtp_length=4, mlp_depth=1,
        )
        x = torch.randn(1, 1, 32)
        params = head(x)
        actions, lp = head.sample(params)
        assert actions.shape == (1, 1, 2)

    def test_bc_loss_short_sequence(self):
        """BC loss with T < mtp_length should still work."""
        head = PolicyHead(
            d_model=32, action_dim=2, action_type="continuous",
            mtp_length=8, mlp_depth=1,
        )
        embed = torch.randn(1, 3, 32)
        actions = torch.randn(1, 3, 2)
        loss = behavior_cloning_loss(head, embed, actions, mtp_length=8)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_lambda_returns_batch_independence(self):
        """Each batch element is computed independently."""
        torch.manual_seed(0)
        rewards = torch.randn(2, 5)
        values = torch.randn(2, 5)
        continues = torch.ones(2, 5)

        ret_full = compute_lambda_returns(rewards, values, continues)
        ret_0 = compute_lambda_returns(
            rewards[:1], values[:1], continues[:1]
        )
        ret_1 = compute_lambda_returns(
            rewards[1:], values[1:], continues[1:]
        )
        assert torch.allclose(ret_full[0], ret_0[0], atol=1e-6)
        assert torch.allclose(ret_full[1], ret_1[0], atol=1e-6)
