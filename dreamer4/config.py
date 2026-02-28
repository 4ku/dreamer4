"""
YAML-based configuration system for Dreamer 4.

Extends :class:`TrainConfig` with YAML loading/saving and validation,
while keeping backward compatibility with the dataclass API.

Usage::

    from dreamer4.config import TrainConfig, load_config, save_config

    # From YAML file
    cfg = load_config("config/dmcontrol.yaml")

    # From YAML with overrides
    cfg = load_config("config/defaults.yaml", overrides={"domain": "walker"})

    # Pure dataclass (no YAML needed)
    cfg = TrainConfig(domain="cartpole", task="swingup")

    # Save current config
    save_config(cfg, "runs/exp01/config.yaml")
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml


@dataclass
class TrainConfig:
    """Full Dreamer 4 training configuration.

    Groups:
        env      -- environment setup
        tok      -- tokenizer architecture
        dyn      -- dynamics model architecture
        agent    -- policy / reward / value heads
        imagine  -- imagination rollout parameters
        train    -- training loop schedule
        pmpo     -- PMPO objective hyperparameters
        ckpt     -- checkpointing
        log      -- logging
    """

    # -- Environment --
    domain: str = "cartpole"
    task: str = "swingup"
    image_size: int = 64
    action_repeat: int = 2
    n_envs: int = 1
    time_limit: int = 500

    # -- Tokenizer --
    patch_size: int = 8
    channels: int = 3
    d_bottleneck: int = 16
    n_bottleneck: int = 64
    packing_k: int = 4
    tokenizer_lr: float = 3e-4

    # -- Dynamics --
    d_model: int = 256
    d_spatial: int = 64
    n_spatial: int = 16
    n_register: int = 4
    n_agent: int = 2
    n_heads: int = 4
    depth: int = 8
    k_max: int = 4
    dynamics_lr: float = 1e-4
    bootstrap_fraction: float = 0.25

    # -- Agent heads --
    mtp_length: int = 8
    policy_mlp_depth: int = 2
    policy_mlp_ratio: float = 2.0
    reward_num_bins: int = 127
    value_num_bins: int = 127
    bin_low: float = -20.0
    bin_high: float = 20.0
    agent_lr: float = 3e-4
    num_tasks: int = 1

    # -- Imagination --
    imagination_horizon: int = 15
    imagination_K: int = 4
    imagination_tau_ctx: float = 0.1
    imagination_ctx_len: int = 4
    gamma: float = 0.997
    lam: float = 0.95

    # -- PMPO --
    pmpo_alpha: float = 0.5
    pmpo_beta: float = 0.3
    entropy_scale: float = 1e-2

    # -- Policy inference --
    policy_max_context: int = 8
    inference_tau: float = 0.95

    # -- Training loop --
    total_steps: int = 1_000_000
    prefill_steps: int = 5000
    train_every: int = 16
    batch_size: int = 16
    seq_len: int = 16
    wm_train_steps: int = 1
    imagine_train_steps: int = 1
    replay_capacity: int = 1_000_000
    max_grad_norm: float = 100.0
    prior_update_interval: int = 100

    # -- Checkpointing --
    checkpoint_every: int = 10_000
    keep_last_n: int = 3

    # -- Logging --
    log_every: int = 100
    eval_every: int = 10_000
    wm_video_every: int = 20_000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_FIELDS = {f.name for f in fields(TrainConfig)}


def validate_config(cfg: TrainConfig) -> None:
    """Raise ``ValueError`` for invalid config values."""
    if cfg.k_max < 1 or (cfg.k_max & (cfg.k_max - 1)) != 0:
        raise ValueError(f"k_max must be a power of 2, got {cfg.k_max}")
    if cfg.n_agent < 0:
        raise ValueError(f"n_agent must be >= 0, got {cfg.n_agent}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {cfg.seq_len}")
    if not (0.0 < cfg.gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {cfg.gamma}")
    if not (0.0 <= cfg.lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {cfg.lam}")
    if not (0.0 <= cfg.pmpo_alpha <= 1.0):
        raise ValueError(f"pmpo_alpha must be in [0, 1], got {cfg.pmpo_alpha}")
    if cfg.image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {cfg.image_size}")


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def save_config(cfg: TrainConfig, path: Union[str, Path]) -> None:
    """Dump a :class:`TrainConfig` to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(
    path: Union[str, Path],
    *,
    overrides: Optional[dict[str, Any]] = None,
) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file.

    Unknown keys in the YAML are silently ignored so that old config
    files remain forward-compatible.

    Args:
        path:      Path to the YAML file.
        overrides: Dict of field values to override after loading.

    Returns:
        Validated :class:`TrainConfig`.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data.update(overrides)

    known = {k: v for k, v in data.items() if k in _VALID_FIELDS}
    cfg = TrainConfig(**known)
    validate_config(cfg)
    return cfg


def config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Create a :class:`TrainConfig` from a plain dict.

    Useful for programmatic construction or CLI argument merging.
    """
    known = {k: v for k, v in data.items() if k in _VALID_FIELDS}
    cfg = TrainConfig(**known)
    validate_config(cfg)
    return cfg
