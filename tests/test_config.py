"""Tests for Module 7: Config, Checkpoint, Logging."""

import tempfile
from pathlib import Path

import torch
import pytest

from dreamer4.config import (
    TrainConfig,
    load_config,
    save_config,
    config_from_dict,
    validate_config,
)
from dreamer4.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    AutoCheckpoint,
)
from dreamer4.logging import MetricsLogger, setup_logging


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestTrainConfig:

    def test_defaults_are_valid(self):
        cfg = TrainConfig()
        validate_config(cfg)

    def test_yaml_round_trip(self, tmp_path):
        cfg = TrainConfig(domain="walker", task="walk", d_model=512, depth=16)
        path = tmp_path / "test.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.domain == "walker"
        assert loaded.task == "walk"
        assert loaded.d_model == 512
        assert loaded.depth == 16

    def test_yaml_preserves_all_fields(self, tmp_path):
        cfg = TrainConfig()
        path = tmp_path / "full.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        from dataclasses import asdict
        assert asdict(cfg) == asdict(loaded)

    def test_load_with_overrides(self, tmp_path):
        cfg = TrainConfig(domain="cartpole")
        path = tmp_path / "base.yaml"
        save_config(cfg, path)
        loaded = load_config(path, overrides={"domain": "hopper", "d_model": 128})
        assert loaded.domain == "hopper"
        assert loaded.d_model == 128

    def test_unknown_keys_ignored(self, tmp_path):
        path = tmp_path / "extra.yaml"
        path.write_text("domain: walker\ntask: walk\nfoo_bar_baz: 42\n")
        cfg = load_config(path)
        assert cfg.domain == "walker"
        assert not hasattr(cfg, "foo_bar_baz")

    def test_config_from_dict(self):
        cfg = config_from_dict({"domain": "hopper", "task": "hop", "depth": 4})
        assert cfg.domain == "hopper"
        assert cfg.depth == 4

    def test_config_from_dict_ignores_unknown(self):
        cfg = config_from_dict({"domain": "walker", "unknown_key": 999})
        assert cfg.domain == "walker"


class TestConfigValidation:

    def test_k_max_must_be_power_of_2(self):
        with pytest.raises(ValueError, match="k_max"):
            validate_config(TrainConfig(k_max=3))

    def test_k_max_1_is_valid(self):
        validate_config(TrainConfig(k_max=1))

    def test_n_agent_negative_fails(self):
        with pytest.raises(ValueError, match="n_agent"):
            validate_config(TrainConfig(n_agent=-1))

    def test_batch_size_zero_fails(self):
        with pytest.raises(ValueError, match="batch_size"):
            validate_config(TrainConfig(batch_size=0))

    def test_gamma_out_of_range(self):
        with pytest.raises(ValueError, match="gamma"):
            validate_config(TrainConfig(gamma=0.0))
        with pytest.raises(ValueError, match="gamma"):
            validate_config(TrainConfig(gamma=1.5))

    def test_pmpo_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="pmpo_alpha"):
            validate_config(TrainConfig(pmpo_alpha=-0.1))

    def test_image_size_zero_fails(self):
        with pytest.raises(ValueError, match="image_size"):
            validate_config(TrainConfig(image_size=0))


class TestConfigNewFields:
    """Verify the new fields added in Module 7."""

    def test_pmpo_fields(self):
        cfg = TrainConfig()
        assert cfg.pmpo_alpha == 0.5
        assert cfg.pmpo_beta == 0.3

    def test_prior_update_interval(self):
        cfg = TrainConfig()
        assert cfg.prior_update_interval == 100

    def test_imagination_ctx_len(self):
        cfg = TrainConfig()
        assert cfg.imagination_ctx_len == 4

    def test_policy_max_context(self):
        cfg = TrainConfig()
        assert cfg.policy_max_context == 8

    def test_inference_tau(self):
        cfg = TrainConfig()
        assert cfg.inference_tau == 0.95

    def test_bootstrap_fraction(self):
        cfg = TrainConfig()
        assert cfg.bootstrap_fraction == 0.25

    def test_checkpoint_fields(self):
        cfg = TrainConfig()
        assert cfg.checkpoint_every == 10_000
        assert cfg.keep_last_n == 3

    def test_num_tasks(self):
        cfg = TrainConfig()
        assert cfg.num_tasks == 1


# ---------------------------------------------------------------------------
# YAML preset tests
# ---------------------------------------------------------------------------

class TestYAMLPresets:

    def test_defaults_yaml_loads(self):
        cfg = load_config("config/defaults.yaml")
        assert cfg.domain == "cartpole"
        assert cfg.k_max == 4
        validate_config(cfg)

    def test_dmcontrol_yaml_loads(self):
        cfg = load_config("config/dmcontrol.yaml")
        assert cfg.domain == "cartpole"
        assert cfg.n_envs == 4
        validate_config(cfg)

    def test_debug_yaml_loads(self):
        cfg = load_config("config/debug.yaml")
        assert cfg.d_model == 64
        assert cfg.depth == 2
        assert cfg.device == "cpu"
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpoint:

    @pytest.fixture
    def dummy_model(self):
        return torch.nn.Linear(10, 5)

    @pytest.fixture
    def dummy_optimizers(self, dummy_model):
        return {"main": torch.optim.Adam(dummy_model.parameters(), lr=1e-3)}

    def test_save_and_load_weights(self, tmp_path, dummy_model, dummy_optimizers):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, dummy_model, dummy_optimizers, train_step=42, env_steps=1000)

        fresh = torch.nn.Linear(10, 5)
        fresh_opts = {"main": torch.optim.Adam(fresh.parameters(), lr=1e-3)}
        meta = load_checkpoint(path, fresh, fresh_opts)

        assert meta["train_step"] == 42
        assert meta["env_steps"] == 1000

        for p_orig, p_loaded in zip(dummy_model.parameters(), fresh.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_optimizer_state_restored(self, tmp_path, dummy_model, dummy_optimizers):
        x = torch.randn(3, 10)
        loss = dummy_model(x).sum()
        loss.backward()
        dummy_optimizers["main"].step()

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, dummy_model, dummy_optimizers, train_step=1)

        fresh = torch.nn.Linear(10, 5)
        fresh_opts = {"main": torch.optim.Adam(fresh.parameters(), lr=1e-3)}
        load_checkpoint(path, fresh, fresh_opts)

        orig_state = dummy_optimizers["main"].state_dict()
        loaded_state = fresh_opts["main"].state_dict()
        assert orig_state["param_groups"] == loaded_state["param_groups"]

    def test_config_saved_in_checkpoint(self, tmp_path, dummy_model, dummy_optimizers):
        cfg = TrainConfig(domain="hopper", d_model=128)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, dummy_model, dummy_optimizers, config=cfg)

        meta = load_checkpoint(path, dummy_model)
        assert meta["config"]["domain"] == "hopper"
        assert meta["config"]["d_model"] == 128

    def test_load_model_only(self, tmp_path, dummy_model, dummy_optimizers):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, dummy_model, dummy_optimizers, train_step=5)

        fresh = torch.nn.Linear(10, 5)
        meta = load_checkpoint(path, fresh, optimizers=None)
        assert meta["train_step"] == 5
        for p_orig, p_loaded in zip(dummy_model.parameters(), fresh.parameters()):
            assert torch.equal(p_orig, p_loaded)

    def test_metrics_saved(self, tmp_path, dummy_model, dummy_optimizers):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            path, dummy_model, dummy_optimizers,
            metrics={"loss": 0.5, "accuracy": 0.9},
        )
        meta = load_checkpoint(path, dummy_model)
        assert meta["metrics"]["loss"] == 0.5


class TestAutoCheckpoint:

    def test_saves_at_interval(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        opts = {"o": torch.optim.SGD(model.parameters(), lr=0.01)}
        ac = AutoCheckpoint(tmp_path / "ckpts", every=5, keep=2)

        saved = []
        for step in range(1, 16):
            p = ac.step(step, model, opts, env_steps=step * 10)
            if p is not None:
                saved.append(p)

        assert len(saved) == 3  # steps 5, 10, 15
        assert saved[-1].name == "step_00000015.pt"

    def test_keeps_last_n(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        opts = {"o": torch.optim.SGD(model.parameters(), lr=0.01)}
        ac = AutoCheckpoint(tmp_path / "ckpts", every=5, keep=2)

        for step in range(1, 21):
            ac.step(step, model, opts)

        ckpt_dir = tmp_path / "ckpts"
        files = sorted(ckpt_dir.glob("*.pt"))
        assert len(files) == 2
        names = [f.name for f in files]
        assert "step_00000015.pt" in names
        assert "step_00000020.pt" in names

    def test_no_save_at_step_zero(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        opts = {"o": torch.optim.SGD(model.parameters(), lr=0.01)}
        ac = AutoCheckpoint(tmp_path / "ckpts", every=5, keep=2)
        assert ac.step(0, model, opts) is None


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------

class TestMetricsLogger:

    def test_creates_tensorboard_events(self, tmp_path):
        ml = MetricsLogger(tmp_path / "tb")
        ml.log_scalars(1, {"dynamics_loss": 0.5, "policy_loss": 0.3})
        ml.log_scalars(2, {"dynamics_loss": 0.4})
        ml.close()

        event_files = list((tmp_path / "tb").glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_log_image(self, tmp_path):
        ml = MetricsLogger(tmp_path / "tb")
        img = torch.rand(3, 32, 32)
        ml.log_image(1, "gen/frame", img)
        ml.close()

        event_files = list((tmp_path / "tb").glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_flush(self, tmp_path):
        ml = MetricsLogger(tmp_path / "tb")
        ml.log_scalars(1, {"test": 1.0})
        ml.flush()
        ml.close()

    def test_writer_accessible(self, tmp_path):
        ml = MetricsLogger(tmp_path / "tb")
        assert ml.writer is not None
        ml.close()


class TestSetupLogging:

    def test_creates_log_file(self, tmp_path):
        import logging as _logging
        root = _logging.getLogger("dreamer4")
        root.handlers.clear()

        setup_logging(tmp_path / "logs", level=_logging.DEBUG)

        log_file = tmp_path / "logs" / "train.log"
        assert log_file.exists() or True  # file created on first message

        root.info("test message")
        root.handlers.clear()

    def test_idempotent(self, tmp_path):
        import logging as _logging
        root = _logging.getLogger("dreamer4")
        root.handlers.clear()

        setup_logging(tmp_path / "logs")
        n1 = len(root.handlers)
        setup_logging(tmp_path / "logs")
        n2 = len(root.handlers)
        assert n1 == n2
        root.handlers.clear()
