"""End-to-end integration test for the full Dreamer 4 pipeline.

Tests that all modules work together: config, tokenizer, dynamics,
agent heads, imagination, checkpointing, and logging.
"""

import torch
import pytest

from dreamer4.config import TrainConfig, config_from_dict
from dreamer4.checkpoint import save_checkpoint, load_checkpoint
from dreamer4.logging import MetricsLogger
from dreamer4.tokenizer import Encoder, Decoder, Tokenizer
from dreamer4.train import Dreamer4Agent


def _make_tiny_config(**overrides) -> TrainConfig:
    """Minimal config for fast integration testing."""
    defaults = dict(
        domain="cartpole",
        task="swingup",
        image_size=32,
        action_repeat=2,
        n_envs=1,
        time_limit=10,
        patch_size=8,
        channels=3,
        d_bottleneck=8,
        n_bottleneck=16,
        packing_k=4,
        tokenizer_lr=3e-4,
        d_model=32,
        d_spatial=32,
        n_spatial=4,
        n_register=1,
        n_agent=1,
        n_heads=1,
        depth=1,
        k_max=2,
        dynamics_lr=1e-4,
        mtp_length=2,
        policy_mlp_depth=1,
        policy_mlp_ratio=1.0,
        reward_num_bins=15,
        value_num_bins=15,
        agent_lr=3e-4,
        imagination_horizon=2,
        imagination_K=1,
        imagination_ctx_len=2,
        total_steps=20,
        prefill_steps=15,
        train_every=8,
        batch_size=1,
        seq_len=4,
        replay_capacity=5000,
        log_every=1,
        eval_every=20,
        checkpoint_every=1,
        keep_last_n=2,
        prior_update_interval=2,
        device="cpu",
    )
    defaults.update(overrides)
    return config_from_dict(defaults)


def _make_tiny_tokenizer(cfg: TrainConfig) -> Tokenizer:
    n_patches = (cfg.image_size // cfg.patch_size) ** 2
    patch_dim = cfg.patch_size ** 2 * cfg.channels
    enc = Encoder(
        patch_dim=patch_dim,
        d_model=cfg.d_model,
        n_latents=cfg.n_bottleneck,
        n_patches=n_patches,
        n_heads=cfg.n_heads,
        depth=1,
        d_bottleneck=cfg.d_bottleneck,
    )
    dec = Decoder(
        d_bottleneck=cfg.d_bottleneck,
        d_model=cfg.d_model,
        n_latents=cfg.n_bottleneck,
        n_patches=n_patches,
        patch_dim=patch_dim,
        n_heads=cfg.n_heads,
        depth=1,
    )
    return Tokenizer(enc, dec)


class TestEndToEnd:

    def test_agent_construction(self):
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        assert agent.action_dim >= 1
        assert agent.dynamics is not None
        assert agent.policy is not None
        assert agent.reward_head is not None
        assert agent.value_head is not None

    def test_agent_optimizers(self):
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        opts = agent.build_optimizers()
        assert "dynamics" in opts
        assert "agent" in opts
        assert "value" in opts
        assert "policy" in opts
        assert "tokenizer" not in opts

    def test_agent_with_tokenizer_has_tokenizer_optimizer(self):
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))
        opts = agent.build_optimizers()
        assert "tokenizer" in opts

    def test_world_model_step(self):
        """One WM training step: tokenizer + dynamics + agent heads."""
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))
        opts = agent.build_optimizers()

        B, T = 1, 4
        batch = {
            "obs": torch.rand(B, T, cfg.channels, cfg.image_size, cfg.image_size),
            "actions": torch.randn(B, T, agent.action_dim),
            "rewards": torch.randn(B, T),
        }
        metrics = agent.train_world_model_step(batch, opts)
        assert "dynamics_loss" in metrics
        assert "bc_loss" in metrics
        assert "reward_pred_loss" in metrics
        assert "tokenizer_loss" in metrics

    def test_imagine_and_train(self):
        """One imagination rollout + policy/value update."""
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))
        opts = agent.build_optimizers()

        B, T = 1, 4
        batch = {
            "obs": torch.rand(B, T, cfg.channels, cfg.image_size, cfg.image_size),
            "actions": torch.randn(B, T, agent.action_dim),
            "rewards": torch.randn(B, T),
        }
        metrics = agent.imagine_and_train(batch, opts)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

    def test_policy_action(self):
        """Agent produces an action from observation history."""
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))

        obs_history = [
            torch.rand(cfg.channels, cfg.image_size, cfg.image_size)
            for _ in range(3)
        ]
        act_history = [
            torch.randn(agent.action_dim) for _ in range(2)
        ]
        action = agent.policy_action(obs_history, act_history)
        assert action.shape == (agent.action_dim,)

    def test_checkpoint_save_and_load(self, tmp_path):
        """Save checkpoint, load into fresh agent, verify weights match."""
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))
        opts = agent.build_optimizers()

        # Do one training step so weights differ from init
        B, T = 1, 4
        batch = {
            "obs": torch.rand(B, T, cfg.channels, cfg.image_size, cfg.image_size),
            "actions": torch.randn(B, T, agent.action_dim),
            "rewards": torch.randn(B, T),
        }
        agent.train_world_model_step(batch, opts)

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(ckpt_path, agent, opts, train_step=42, env_steps=1000, config=cfg)

        fresh = Dreamer4Agent(cfg)
        fresh.set_tokenizer(_make_tiny_tokenizer(cfg))
        fresh_opts = fresh.build_optimizers()
        meta = load_checkpoint(ckpt_path, fresh, fresh_opts, map_location="cpu")

        assert meta["train_step"] == 42
        assert meta["env_steps"] == 1000
        assert meta["config"]["domain"] == "cartpole"

        for p_orig, p_fresh in zip(agent.parameters(), fresh.parameters()):
            assert torch.equal(p_orig, p_fresh), "Weights should match after load"

    def test_tensorboard_logging(self, tmp_path):
        """MetricsLogger writes TensorBoard events during training."""
        ml = MetricsLogger(tmp_path / "tb")

        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))
        opts = agent.build_optimizers()

        B, T = 1, 4
        batch = {
            "obs": torch.rand(B, T, cfg.channels, cfg.image_size, cfg.image_size),
            "actions": torch.randn(B, T, agent.action_dim),
            "rewards": torch.randn(B, T),
        }
        metrics = agent.train_world_model_step(batch, opts)
        ml.log_scalars(1, metrics)
        ml.close()

        event_files = list((tmp_path / "tb").glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_full_iteration_wm_then_imagine(self):
        """WM step followed by imagination step = one full Algorithm 1 iteration."""
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))
        opts = agent.build_optimizers()

        B, T = 1, 4
        batch = {
            "obs": torch.rand(B, T, cfg.channels, cfg.image_size, cfg.image_size),
            "actions": torch.randn(B, T, agent.action_dim),
            "rewards": torch.randn(B, T),
        }

        wm_metrics = agent.train_world_model_step(batch, opts)
        im_metrics = agent.imagine_and_train(batch, opts)

        all_keys = set(wm_metrics) | set(im_metrics)
        expected = {"dynamics_loss", "bc_loss", "reward_pred_loss",
                    "tokenizer_loss", "policy_loss", "value_loss"}
        assert expected.issubset(all_keys)

    def test_prior_policy_update(self):
        """Prior policy is created and updated."""
        cfg = _make_tiny_config()
        agent = Dreamer4Agent(cfg)
        assert agent.prior_policy is None

        agent.update_prior_policy()
        assert agent.prior_policy is not None

        old_params = [p.clone() for p in agent.prior_policy.parameters()]

        # Train policy
        agent.set_tokenizer(_make_tiny_tokenizer(cfg))
        opts = agent.build_optimizers()
        batch = {
            "obs": torch.rand(1, 4, cfg.channels, cfg.image_size, cfg.image_size),
            "actions": torch.randn(1, 4, agent.action_dim),
            "rewards": torch.randn(1, 4),
        }
        agent.imagine_and_train(batch, opts)

        agent.update_prior_policy()
        for p_old, p_new in zip(old_params, agent.prior_policy.parameters()):
            if not torch.equal(p_old, p_new):
                return  # at least one param changed
        pytest.fail("Prior policy should update after training")
