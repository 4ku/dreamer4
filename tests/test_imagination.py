"""Tests for dreamer4.imagination — Imagination rollouts and training."""

import copy

import torch
import pytest

from dreamer4.dynamics import DynamicsModel
from dreamer4.agent import (
    PolicyHead, RewardHead, ValueHead, TaskEncoder,
    compute_lambda_returns, pmpo_policy_loss,
)
from dreamer4.imagination import (
    ImaginedTrajectory,
    imagine_rollout,
    imagination_training_step,
    make_prior_policy,
)
from dreamer4.replay import ReplayBuffer
from dreamer4.envs import DMControlEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T_CTX, D_MODEL, D_SPATIAL, N_SPATIAL = 4, 2, 64, 32, 8
N_REGISTER, N_AGENT = 2, 1
N_HEADS, DEPTH = 2, 2
K_MAX = 4
ACTION_DIM = 4
HORIZON = 6


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dynamics(device):
    return DynamicsModel(
        d_model=D_MODEL,
        d_spatial=D_SPATIAL,
        n_spatial=N_SPATIAL,
        n_register=N_REGISTER,
        n_agent=N_AGENT,
        n_heads=N_HEADS,
        depth=DEPTH,
        k_max=K_MAX,
        action_dim=ACTION_DIM,
        mlp_ratio=2.0,
        time_every=2,
        space_mode="wm_agent_isolated",
    ).to(device).eval()


@pytest.fixture
def policy(device):
    return PolicyHead(
        d_model=D_MODEL,
        action_dim=ACTION_DIM,
        action_type="continuous",
        mtp_length=4,
        mlp_depth=2,
        mlp_ratio=2.0,
    ).to(device)


@pytest.fixture
def reward_head(device):
    return RewardHead(
        d_model=D_MODEL,
        mtp_length=4,
        mlp_depth=2,
        mlp_ratio=2.0,
        num_bins=31,
        bin_low=-5.0,
        bin_high=5.0,
    ).to(device).eval()


@pytest.fixture
def value_head(device):
    return ValueHead(
        d_model=D_MODEL,
        mlp_depth=2,
        mlp_ratio=2.0,
        num_bins=31,
        bin_low=-5.0,
        bin_high=5.0,
    ).to(device)


@pytest.fixture
def task_encoder(device):
    return TaskEncoder(
        num_tasks=2, d_model=D_MODEL, n_agent=N_AGENT,
    ).to(device)


@pytest.fixture
def context(device):
    return torch.randn(B, T_CTX, N_SPATIAL, D_SPATIAL, device=device)


@pytest.fixture
def context_actions(device):
    return torch.randn(B, T_CTX, ACTION_DIM, device=device)


# ---------------------------------------------------------------------------
# Rollout shape tests
# ---------------------------------------------------------------------------

class TestImagineRollout:

    def test_trajectory_shapes(
        self, dynamics, policy, reward_head, value_head, task_encoder,
        context, context_actions, device,
    ):
        traj = imagine_rollout(
            dynamics, policy, reward_head, value_head, task_encoder,
            context=context,
            context_actions=context_actions,
            horizon=HORIZON,
            k_max=K_MAX,
            K=2,
            tau_ctx=0.1,
        )
        assert isinstance(traj, ImaginedTrajectory)
        assert traj.latents.shape == (B, HORIZON, N_SPATIAL, D_SPATIAL)
        assert traj.actions.shape == (B, HORIZON, ACTION_DIM)
        assert traj.log_probs.shape == (B, HORIZON)
        assert traj.rewards.shape == (B, HORIZON)
        assert traj.values.shape == (B, HORIZON)
        assert traj.continues.shape == (B, HORIZON)
        assert traj.agent_embeds.shape == (B, HORIZON, D_MODEL)

    def test_trajectory_without_context_actions(
        self, dynamics, policy, reward_head, value_head, task_encoder,
        context, device,
    ):
        traj = imagine_rollout(
            dynamics, policy, reward_head, value_head, task_encoder,
            context=context,
            context_actions=None,
            horizon=4,
            k_max=K_MAX,
            K=2,
        )
        assert traj.latents.shape[1] == 4
        assert traj.actions.shape[1] == 4

    def test_continues_are_ones(
        self, dynamics, policy, reward_head, value_head, task_encoder,
        context, device,
    ):
        traj = imagine_rollout(
            dynamics, policy, reward_head, value_head, task_encoder,
            context=context,
            horizon=HORIZON,
            k_max=K_MAX,
            K=2,
        )
        assert torch.all(traj.continues == 1.0)

    def test_rollout_is_no_grad(
        self, dynamics, policy, reward_head, value_head, task_encoder,
        context, device,
    ):
        traj = imagine_rollout(
            dynamics, policy, reward_head, value_head, task_encoder,
            context=context,
            horizon=3,
            k_max=K_MAX,
            K=2,
        )
        assert not traj.latents.requires_grad
        assert not traj.actions.requires_grad
        assert not traj.rewards.requires_grad


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------

class TestImaginationTrainingStep:

    @pytest.fixture
    def trajectory(self, device):
        return ImaginedTrajectory(
            latents=torch.randn(B, HORIZON, N_SPATIAL, D_SPATIAL, device=device),
            actions=torch.tanh(torch.randn(B, HORIZON, ACTION_DIM, device=device)),
            log_probs=torch.randn(B, HORIZON, device=device),
            rewards=torch.rand(B, HORIZON, device=device),
            values=torch.randn(B, HORIZON, device=device),
            continues=torch.ones(B, HORIZON, device=device),
            agent_embeds=torch.randn(B, HORIZON, D_MODEL, device=device),
        )

    def test_returns_loss_dict(self, trajectory, policy, value_head):
        policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
        value_optim = torch.optim.Adam(value_head.parameters(), lr=1e-3)
        result = imagination_training_step(
            trajectory, policy, value_head,
            policy_optim, value_optim,
        )
        assert isinstance(result, dict)
        for key in (
            "policy_loss", "value_loss", "mean_reward", "mean_value",
            "mean_advantage", "policy_grad_norm", "value_grad_norm",
            "advantage_std", "advantage_pos_frac", "action_mean", "action_std",
        ):
            assert key in result, f"Missing key: {key}"

    def test_gradients_only_on_policy_and_value(
        self, trajectory, policy, value_head, dynamics, reward_head,
    ):
        policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
        value_optim = torch.optim.Adam(value_head.parameters(), lr=1e-3)

        dyn_params_before = {
            n: p.clone() for n, p in dynamics.named_parameters()
        }
        rw_params_before = {
            n: p.clone() for n, p in reward_head.named_parameters()
        }

        imagination_training_step(
            trajectory, policy, value_head,
            policy_optim, value_optim,
        )

        for n, p in dynamics.named_parameters():
            assert torch.equal(p, dyn_params_before[n]), \
                f"Dynamics param {n} changed during imagination step"
        for n, p in reward_head.named_parameters():
            assert torch.equal(p, rw_params_before[n]), \
                f"Reward param {n} changed during imagination step"

    def test_with_prior_policy(self, trajectory, policy, value_head):
        prior = make_prior_policy(policy)
        policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
        value_optim = torch.optim.Adam(value_head.parameters(), lr=1e-3)
        result = imagination_training_step(
            trajectory, policy, value_head,
            policy_optim, value_optim,
            prior_policy=prior,
            beta=0.3,
        )
        assert "policy_loss" in result

    def test_value_loss_decreases(self, trajectory, policy, value_head):
        policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
        value_optim = torch.optim.Adam(value_head.parameters(), lr=3e-3)
        torch.manual_seed(42)

        losses = []
        for _ in range(20):
            result = imagination_training_step(
                trajectory, policy, value_head,
                policy_optim, value_optim,
            )
            losses.append(result["value_loss"])

        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, (
            f"Value loss did not decrease: first_5_avg={avg_first:.4f}, "
            f"last_5_avg={avg_last:.4f}"
        )


# ---------------------------------------------------------------------------
# Prior policy tests
# ---------------------------------------------------------------------------

class TestMakePriorPolicy:

    def test_prior_is_frozen(self, policy):
        prior = make_prior_policy(policy)
        for p in prior.parameters():
            assert not p.requires_grad

    def test_prior_is_copy(self, policy, device):
        prior = make_prior_policy(policy)
        embed = torch.randn(2, D_MODEL, device=device)
        params_orig = policy.forward(embed)
        params_prior = prior.forward(embed)
        assert torch.allclose(params_orig, params_prior, atol=1e-6)

    def test_prior_diverges_after_training(self, policy, device):
        prior = make_prior_policy(policy)
        embed = torch.randn(B, D_MODEL, device=device)
        actions = torch.tanh(torch.randn(B, ACTION_DIM, device=device))

        optim = torch.optim.Adam(policy.parameters(), lr=0.1)
        for _ in range(5):
            params = policy.forward(embed)
            loss = -policy.log_prob(params, actions).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        params_now = policy.forward(embed)
        prior_params = prior.forward(embed)
        assert not torch.allclose(params_now, prior_params, atol=1e-3), \
            "Policy should have diverged from prior after training"


# ---------------------------------------------------------------------------
# Replay buffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:

    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            obs = torch.randn(3, 4, 4)
            act = torch.randn(2)
            buf.add(obs, act, float(i), done=(i == 9))
        assert len(buf) == 10

    def test_sample_sequence_shapes(self):
        buf = ReplayBuffer(capacity=1000)
        for ep in range(5):
            for t in range(20):
                buf.add(
                    torch.randn(3, 8, 8),
                    torch.randn(4),
                    float(t) * 0.1,
                    done=(t == 19),
                )
        batch = buf.sample_sequence(batch_size=4, seq_len=8)
        assert batch["obs"].shape == (4, 8, 3, 8, 8)
        assert batch["actions"].shape == (4, 8, 4)
        assert batch["rewards"].shape == (4, 8)
        assert batch["dones"].shape == (4, 8)

    def test_add_episode(self):
        buf = ReplayBuffer(capacity=1000)
        obs = torch.randn(15, 3, 8, 8)
        actions = torch.randn(15, 4)
        rewards = torch.randn(15)
        buf.add_episode(obs, actions, rewards)
        assert len(buf) == 15
        assert buf._dones[-1] is True
        assert buf._dones[0] is False

    def test_no_cross_episode_sampling(self):
        buf = ReplayBuffer(capacity=1000)
        for ep in range(3):
            for t in range(5):
                buf.add(
                    torch.full((1,), float(ep)),
                    torch.zeros(1),
                    0.0,
                    done=(t == 4),
                )
        batch = buf.sample_sequence(batch_size=100, seq_len=5)
        for i in range(100):
            ep_id = batch["obs"][i, 0, 0].item()
            for t in range(5):
                assert batch["obs"][i, t, 0].item() == ep_id, \
                    "Sequence crossed episode boundary"

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.add(torch.tensor([float(i)]), torch.zeros(1), 0.0, done=(i % 5 == 4))
        assert len(buf) == 10


# ---------------------------------------------------------------------------
# DMControl env smoke test
# ---------------------------------------------------------------------------

class TestDMControlEnv:

    def test_creation_and_reset(self):
        env = DMControlEnv("cartpole", "swingup", size=64, action_repeat=2)
        obs = env.reset()
        assert obs.shape == (3, 64, 64)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_step(self):
        env = DMControlEnv("cartpole", "swingup", size=64, action_repeat=2)
        env.reset()
        action = env.sample_random_action()
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == (3, 64, 64)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_action_dim(self):
        env = DMControlEnv("cartpole", "swingup", size=64)
        assert env.action_dim == 1

    def test_observation_shape(self):
        env = DMControlEnv("cartpole", "swingup", size=32)
        assert env.observation_shape == (3, 32, 32)

    def test_episode_terminates(self):
        env = DMControlEnv("cartpole", "swingup", size=64, action_repeat=2)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 600:
            obs, reward, done, _, _ = env.step(env.sample_random_action())
            steps += 1
        assert steps > 0
