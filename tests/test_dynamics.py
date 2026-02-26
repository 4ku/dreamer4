"""Tests for dreamer4.dynamics — Dynamics model and Shortcut Forcing."""

import math

import torch
import pytest

from dreamer4.dynamics import (
    ActionEncoder,
    ShortcutSignalEncoder,
    DynamicsModel,
    corrupt_representations,
    ramp_weight,
    sample_flow_schedule,
    sample_bootstrap_schedule,
    shortcut_forcing_loss,
    sample_one_timestep,
    sample_sequence,
    _log2_int,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, D_MODEL, D_SPATIAL, N_SPATIAL = 2, 4, 64, 32, 8
N_REGISTER, N_AGENT = 2, 1
N_HEADS, DEPTH = 2, 2
K_MAX = 8
ACTION_DIM = 6


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
    ).to(device)


@pytest.fixture
def dynamics_no_agent(device):
    return DynamicsModel(
        d_model=D_MODEL,
        d_spatial=D_SPATIAL,
        n_spatial=N_SPATIAL,
        n_register=N_REGISTER,
        n_agent=0,
        n_heads=N_HEADS,
        depth=DEPTH,
        k_max=K_MAX,
        action_dim=ACTION_DIM,
        mlp_ratio=2.0,
        time_every=2,
        space_mode="wm_agent_isolated",
    ).to(device)


# ---------------------------------------------------------------------------
# 1. ActionEncoder
# ---------------------------------------------------------------------------

class TestActionEncoder:
    def test_shape_with_actions(self, device):
        enc = ActionEncoder(d_model=D_MODEL, action_dim=ACTION_DIM).to(device)
        actions = torch.randn(B, T, ACTION_DIM, device=device)
        out = enc(actions)
        assert out.shape == (B, T, 1, D_MODEL)

    def test_shape_without_actions(self, device):
        enc = ActionEncoder(d_model=D_MODEL, action_dim=ACTION_DIM).to(device)
        out = enc(None, batch_time_shape=(B, T))
        assert out.shape == (B, T, 1, D_MODEL)

    def test_none_requires_batch_time_shape(self, device):
        enc = ActionEncoder(d_model=D_MODEL, action_dim=ACTION_DIM).to(device)
        with pytest.raises(AssertionError):
            enc(None)

    def test_actions_clamped(self, device):
        enc = ActionEncoder(d_model=D_MODEL, action_dim=ACTION_DIM).to(device)
        big_actions = torch.ones(B, T, ACTION_DIM, device=device) * 10.0
        out = enc(big_actions)
        assert out.shape == (B, T, 1, D_MODEL)
        assert torch.isfinite(out).all()

    def test_unlabeled_is_constant_across_batch(self, device):
        enc = ActionEncoder(d_model=D_MODEL, action_dim=ACTION_DIM).to(device)
        out = enc(None, batch_time_shape=(3, T))
        assert torch.allclose(out[0], out[1])
        assert torch.allclose(out[1], out[2])


# ---------------------------------------------------------------------------
# 2. ShortcutSignalEncoder
# ---------------------------------------------------------------------------

class TestShortcutSignalEncoder:
    def test_shape(self, device):
        enc = ShortcutSignalEncoder(d_model=D_MODEL, k_max=K_MAX).to(device)
        step_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        signal_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        out = enc(step_idx, signal_idx)
        assert out.shape == (B, T, 1, D_MODEL)

    def test_different_inputs_different_outputs(self, device):
        enc = ShortcutSignalEncoder(d_model=D_MODEL, k_max=K_MAX).to(device)
        s0 = torch.zeros(1, 1, dtype=torch.long, device=device)
        s1 = torch.ones(1, 1, dtype=torch.long, device=device)
        out0 = enc(s0, s0)
        out1 = enc(s1, s1)
        assert not torch.allclose(out0, out1, atol=1e-6)

    def test_max_indices(self, device):
        enc = ShortcutSignalEncoder(d_model=D_MODEL, k_max=K_MAX).to(device)
        emax = int(math.log2(K_MAX))
        step_idx = torch.full((B, T), emax, dtype=torch.long, device=device)
        signal_idx = torch.full((B, T), K_MAX, dtype=torch.long, device=device)
        out = enc(step_idx, signal_idx)
        assert out.shape == (B, T, 1, D_MODEL)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. DynamicsModel forward shapes
# ---------------------------------------------------------------------------

class TestDynamicsModel:
    def test_forward_shapes(self, dynamics, device):
        actions = torch.randn(B, T, ACTION_DIM, device=device)
        step_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        signal_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        z_noisy = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)

        z1_hat, agent_out = dynamics(actions, step_idx, signal_idx, z_noisy)

        assert z1_hat.shape == (B, T, N_SPATIAL, D_SPATIAL)
        assert agent_out is not None
        assert agent_out.shape == (B, T, N_AGENT, D_MODEL)

    def test_forward_no_actions(self, dynamics, device):
        step_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        signal_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        z_noisy = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)

        z1_hat, agent_out = dynamics(None, step_idx, signal_idx, z_noisy)
        assert z1_hat.shape == (B, T, N_SPATIAL, D_SPATIAL)

    def test_forward_no_agent(self, dynamics_no_agent, device):
        actions = torch.randn(B, T, ACTION_DIM, device=device)
        step_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        signal_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        z_noisy = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)

        z1_hat, agent_out = dynamics_no_agent(actions, step_idx, signal_idx, z_noisy)
        assert z1_hat.shape == (B, T, N_SPATIAL, D_SPATIAL)
        assert agent_out is None

    def test_flow_head_zero_init(self, dynamics):
        assert (dynamics.flow_head.weight == 0).all()
        assert (dynamics.flow_head.bias == 0).all()

    def test_output_finite(self, dynamics, device):
        actions = torch.randn(B, T, ACTION_DIM, device=device)
        step_idx = torch.ones(B, T, dtype=torch.long, device=device)
        signal_idx = torch.ones(B, T, dtype=torch.long, device=device) * 2
        z_noisy = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)

        z1_hat, _ = dynamics(actions, step_idx, signal_idx, z_noisy)
        assert torch.isfinite(z1_hat).all()

    def test_layout_consistency(self, dynamics):
        expected_S = 1 + 1 + N_SPATIAL + N_REGISTER + N_AGENT
        assert dynamics.layout.total_tokens() == expected_S

    def test_param_count_positive(self, dynamics):
        count = sum(p.numel() for p in dynamics.parameters())
        assert count > 0


# ---------------------------------------------------------------------------
# 4. Schedule sampling
# ---------------------------------------------------------------------------

class TestScheduleSampling:
    def test_flow_schedule_shapes(self, device):
        d, step_idx, tau, sig_idx = sample_flow_schedule(B, T, K_MAX, device)
        assert d.shape == (B, T)
        assert step_idx.shape == (B, T)
        assert tau.shape == (B, T)
        assert sig_idx.shape == (B, T)

    def test_flow_schedule_d_is_dmin(self, device):
        d, step_idx, tau, sig_idx = sample_flow_schedule(B, T, K_MAX, device)
        d_min = 1.0 / K_MAX
        assert torch.allclose(d, torch.full_like(d, d_min))

    def test_flow_schedule_step_idx_is_emax(self, device):
        d, step_idx, tau, sig_idx = sample_flow_schedule(B, T, K_MAX, device)
        emax = int(math.log2(K_MAX))
        assert (step_idx == emax).all()

    def test_flow_schedule_tau_range(self, device):
        d, step_idx, tau, sig_idx = sample_flow_schedule(100, 10, K_MAX, device)
        assert (tau >= 0.0).all()
        assert (tau < 1.0).all()

    def test_bootstrap_schedule_shapes(self, device):
        d, step_idx, tau, sig_idx = sample_bootstrap_schedule(B, T, K_MAX, device)
        assert d.shape == (B, T)
        assert step_idx.shape == (B, T)
        assert tau.shape == (B, T)
        assert sig_idx.shape == (B, T)

    def test_bootstrap_step_excludes_dmin(self, device):
        emax = int(math.log2(K_MAX))
        for _ in range(10):
            d, step_idx, tau, sig_idx = sample_bootstrap_schedule(32, 8, K_MAX, device)
            assert (step_idx < emax).all(), "Bootstrap should never sample d_min"

    def test_bootstrap_d_is_power_of_two(self, device):
        d, step_idx, tau, sig_idx = sample_bootstrap_schedule(32, 8, K_MAX, device)
        inv_d = (1.0 / d).round().long()
        is_pow2 = (inv_d & (inv_d - 1)) == 0
        assert is_pow2.all()

    def test_bootstrap_tau_on_grid(self, device):
        d, step_idx, tau, sig_idx = sample_bootstrap_schedule(32, 8, K_MAX, device)
        remainder = tau % d
        assert torch.allclose(remainder, torch.zeros_like(remainder), atol=1e-6)

    def test_signal_idx_range(self, device):
        d, step_idx, tau, sig_idx = sample_flow_schedule(32, 8, K_MAX, device)
        assert (sig_idx >= 0).all()
        assert (sig_idx < K_MAX).all()

    def test_log2_int_rejects_non_power_of_2(self):
        with pytest.raises(AssertionError):
            _log2_int(7)

    def test_log2_int_accepts_power_of_2(self):
        assert _log2_int(1) == 0
        assert _log2_int(2) == 1
        assert _log2_int(8) == 3
        assert _log2_int(64) == 6


# ---------------------------------------------------------------------------
# 5. Corrupt representations
# ---------------------------------------------------------------------------

class TestCorruptRepresentations:
    def test_shape(self, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)
        tau = torch.rand(B, T, device=device)
        z_tilde, z0 = corrupt_representations(z1, tau)
        assert z_tilde.shape == z1.shape
        assert z0.shape == z1.shape

    def test_tau_zero_is_pure_noise(self, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)
        tau = torch.zeros(B, T, device=device)
        z_tilde, z0 = corrupt_representations(z1, tau)
        assert torch.allclose(z_tilde, z0, atol=1e-6)

    def test_tau_one_is_clean(self, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)
        tau = torch.ones(B, T, device=device)
        z_tilde, z0 = corrupt_representations(z1, tau)
        assert torch.allclose(z_tilde, z1, atol=1e-6)

    def test_interpolation_midpoint(self, device):
        z1 = torch.ones(B, T, N_SPATIAL, D_SPATIAL, device=device)
        tau = torch.full((B, T), 0.5, device=device)
        z_tilde, z0 = corrupt_representations(z1, tau)
        expected = 0.5 * z0 + 0.5 * z1
        assert torch.allclose(z_tilde, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Ramp weight
# ---------------------------------------------------------------------------

class TestRampWeight:
    def test_at_zero(self):
        w = ramp_weight(torch.tensor(0.0))
        assert torch.allclose(w, torch.tensor(0.1))

    def test_at_one(self):
        w = ramp_weight(torch.tensor(1.0))
        assert torch.allclose(w, torch.tensor(1.0))

    def test_monotonic(self, device):
        tau = torch.linspace(0, 1, 100, device=device)
        w = ramp_weight(tau)
        diffs = w[1:] - w[:-1]
        assert (diffs >= 0).all()

    def test_batch_shape(self, device):
        tau = torch.rand(B, T, device=device)
        w = ramp_weight(tau)
        assert w.shape == (B, T)


# ---------------------------------------------------------------------------
# 7. Shortcut forcing loss
# ---------------------------------------------------------------------------

class TestShortcutForcingLoss:
    def test_loss_is_finite(self, dynamics, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)
        actions = torch.randn(B, T, ACTION_DIM, device=device)

        loss, aux = shortcut_forcing_loss(
            dynamics, z1=z1, actions=actions, k_max=K_MAX,
            bootstrap_fraction=0.25,
        )
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_loss_returns_aux_keys(self, dynamics, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)

        loss, aux = shortcut_forcing_loss(
            dynamics, z1=z1, actions=None, k_max=K_MAX,
        )
        expected_keys = {"flow_mse", "boot_mse", "loss_flow", "loss_boot", "tau_mean"}
        assert expected_keys == set(aux.keys())

    def test_loss_without_actions(self, dynamics, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)

        loss, aux = shortcut_forcing_loss(
            dynamics, z1=z1, actions=None, k_max=K_MAX,
        )
        assert torch.isfinite(loss)

    def test_no_bootstrap(self, dynamics, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)

        loss, aux = shortcut_forcing_loss(
            dynamics, z1=z1, actions=None, k_max=K_MAX,
            bootstrap_fraction=0.0,
        )
        assert torch.isfinite(loss)
        assert aux["loss_boot"].item() == 0.0

    def test_gradient_flow(self, dynamics, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)
        actions = torch.randn(B, T, ACTION_DIM, device=device)

        loss, _ = shortcut_forcing_loss(
            dynamics, z1=z1, actions=actions, k_max=K_MAX,
            bootstrap_fraction=0.0,
        )
        loss.backward()

        has_grad = False
        for p in dynamics.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to dynamics parameters"

    def test_loss_positive(self, dynamics, device):
        z1 = torch.randn(B, T, N_SPATIAL, D_SPATIAL, device=device)
        loss, _ = shortcut_forcing_loss(
            dynamics, z1=z1, actions=None, k_max=K_MAX,
        )
        assert loss.item() > 0, "Loss should be positive for random data"


# ---------------------------------------------------------------------------
# 8. Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_sample_one_timestep_shape(self, dynamics, device):
        dynamics.eval()
        past = torch.randn(B, 3, N_SPATIAL, D_SPATIAL, device=device)
        out, a_out = sample_one_timestep(
            dynamics, past_packed=past, k_max=K_MAX, K=2,
        )
        assert out.shape == (B, N_SPATIAL, D_SPATIAL)
        if dynamics.n_agent > 0:
            assert a_out.shape == (B, dynamics.n_agent, dynamics.d_model)
        else:
            assert a_out is None

    def test_sample_one_timestep_with_actions(self, dynamics, device):
        dynamics.eval()
        past = torch.randn(B, 3, N_SPATIAL, D_SPATIAL, device=device)
        actions = torch.randn(B, 4, ACTION_DIM, device=device)
        out, _ = sample_one_timestep(
            dynamics, past_packed=past, k_max=K_MAX, K=2,
            actions=actions,
        )
        assert out.shape == (B, N_SPATIAL, D_SPATIAL)

    def test_sample_one_timestep_finite(self, dynamics, device):
        dynamics.eval()
        past = torch.randn(B, 2, N_SPATIAL, D_SPATIAL, device=device)
        out, _ = sample_one_timestep(
            dynamics, past_packed=past, k_max=K_MAX, K=2,
        )
        assert torch.isfinite(out).all()

    def test_sample_one_timestep_no_context(self, dynamics, device):
        dynamics.eval()
        past = torch.randn(B, 0, N_SPATIAL, D_SPATIAL, device=device)
        out, _ = sample_one_timestep(
            dynamics, past_packed=past, k_max=K_MAX, K=2,
        )
        assert out.shape == (B, N_SPATIAL, D_SPATIAL)

    def test_sample_sequence_shape(self, dynamics, device):
        dynamics.eval()
        context = torch.randn(B, 2, N_SPATIAL, D_SPATIAL, device=device)
        out, a_outs = sample_sequence(
            dynamics, context=context, horizon=3,
            k_max=K_MAX, K=2,
        )
        assert out.shape == (B, 5, N_SPATIAL, D_SPATIAL)
        if dynamics.n_agent > 0:
            assert len(a_outs) == 3
            assert a_outs[0].shape == (B, dynamics.n_agent, dynamics.d_model)
        else:
            assert a_outs is None

    def test_sample_sequence_context_preserved(self, dynamics, device):
        dynamics.eval()
        context = torch.randn(B, 2, N_SPATIAL, D_SPATIAL, device=device)
        out, _ = sample_sequence(
            dynamics, context=context, horizon=2,
            k_max=K_MAX, K=2, tau_ctx=0.0,
        )
        assert torch.allclose(out[:, :2], context)


# ---------------------------------------------------------------------------
# 9. Loss decreases on toy data
# ---------------------------------------------------------------------------

class TestTrainingBehavior:
    def test_loss_decreases(self, device):
        """Train the dynamics model for several steps on toy data and check loss trend."""
        torch.manual_seed(42)
        dyn = DynamicsModel(
            d_model=32,
            d_spatial=16,
            n_spatial=4,
            n_register=1,
            n_agent=0,
            n_heads=2,
            depth=2,
            k_max=4,
            action_dim=4,
            mlp_ratio=2.0,
            time_every=2,
        ).to(device)

        z1 = torch.randn(4, 6, 4, 16, device=device)
        actions = torch.randn(4, 6, 4, device=device)

        opt = torch.optim.Adam(dyn.parameters(), lr=3e-3)

        losses = []
        for i in range(20):
            torch.manual_seed(100 + i)
            opt.zero_grad()
            loss, _ = shortcut_forcing_loss(
                dyn, z1=z1, actions=actions, k_max=4,
                bootstrap_fraction=0.0,
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())

        first_3_avg = sum(losses[:3]) / 3
        last_3_avg = sum(losses[-3:]) / 3
        assert last_3_avg < first_3_avg, (
            f"Loss trend did not decrease: first_3_avg={first_3_avg:.6f}, "
            f"last_3_avg={last_3_avg:.6f}"
        )
