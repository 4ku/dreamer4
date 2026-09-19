"""
Typed configuration for the phase-1 training entrypoints (tokenizer and
dynamics).

One nested dataclass tree per trainer, three ways to set values (later wins):

    defaults  <  --config file.yaml  <  command-line flags

CLI flags are generated from the dataclasses: nested fields become dotted
flags (``--data.seq_len 8``, ``--model.d_model 512``), top-level fields plain
ones (``--out runs/x``). Booleans take an explicit value (``--optim.amp
false``). Unknown YAML keys are rejected — a typo should fail loudly, not
silently train the default.

The resolved config is saved to ``<out>/config.yaml`` by the trainer, and the
same dict is embedded in every checkpoint, so a run is always reproducible
from its artifacts.

The defaults below ARE the reference phase-1a tokenizer recipe: a
d256/depth-2 tokenizer, 16 latents x 16 dims, loss-normalized MSE +
LPIPS(up128) w0.2 (the paper's weighting), EMA 0.999, bf16 autocast.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

import yaml

C = TypeVar("C")


# ---------------------------------------------------------------------------
# Tokenizer training config
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """What to train on and how to slice it into clips."""

    path: str = ""              # dataset dir(s), comma-separated (see dreamer4.data)
    val_path: str = ""          # dedicated held-out dir; empty = split from path
    val_frac: float = 0.05      # episode fraction held out when val_path is empty
    seq_len: int = 4            # frames per training clip (T)
    batch_size: int = 64
    num_workers: int = 2        # DataLoader workers (0 = decode in main process)
    cameras: str = ""           # LeRobot only: restrict/order cameras (comma list)
    camera_layout: str = "hstack"   # LeRobot only: camera tiling hstack|vstack|grid
    proprio: str = "none"       # none|auto (dataset-stored, e.g. LeRobot
                                # observation.state), or a mode the dataset's
                                # adapter declares (dreamer4.data.PROPRIO_MODES)
    episode_cache: int = 999    # LeRobot: decoded episodes kept in memory per worker


@dataclass
class ModelConfig:
    """Tokenizer architecture (see dreamer4.models.tokenizer)."""

    patch_size: int = 4
    d_model: int = 256          # WIDTH is the quality lever (not depth)
    depth: int = 2              # encoder and decoder each
    n_heads: int = 4
    n_kv_heads: int = 2
    n_latents: int = 16         # bottleneck tokens per frame. 16 is the floor
                                # MEASURED on gridworld; richer scenes need more
    d_bottleneck: int = 16      # dims per bottleneck token (tanh, [-1, 1])
    time_every: int = 2         # time attention every N layers
    decoder_mode: str = "decoder_cross"  # 'decoder' collapses; keep decoder_cross
    mlp_ratio: float = 8 / 3
    logit_cap: float = 50.0
    mae_p_min: float = 0.0      # MAE masking prob range; 0/0 disables masking
    mae_p_max: float = 0.0


@dataclass
class LossConfig:
    """Reconstruction objective. Weights are RELATIVE when loss_norm is on."""

    # mse + 0.2 * LPIPS is the paper's weighting. The previous default here,
    # l1 + 2.0 * LPIPS, left the decoder drawing a sprite one cell off at fixed
    # grid positions (measured 2026-09-18, empty maze: player wrong in 8.2 % of
    # all layouts, goal in 19.8 %, 32.5 dB) and 10 000 more steps did not move
    # it; this loss: 0 % and 65 dB. Obstacles, from scratch: 60.8 vs 55.5 dB.
    recon: str = "mse"          # mse (paper) | l1
    recon_weight: float = 1.0
    perceptual_backbone: str = "lpips"  # none|lpips|dinov3|hybrid
    perceptual_weight: float = 0.2      # primary perceptual term weight (paper)
    perceptual_up: int = 128            # LPIPS upscale resolution
    lpips_net: str = "alex"
    dino_weight: float = 1.0            # DINOv3 weight in hybrid mode
    proprio_weight: float = 0.3         # 0.1-0.3 sweet spot; 1.0 starves pixels
    loss_norm: bool = True              # divide each term by its running RMS
    loss_norm_decay: float = 0.99
    loss_norm_floor: float = 0.2        # divisor floor as frac of peak RMS
                                        # (0.2 swept better than 0.05 and than
                                        #  no floor at all)


@dataclass
class OptimConfig:
    """Optimizer, schedule and train-time model averaging."""

    lr: float = 3e-4
    warmup: int = 500           # linear LR warmup steps
    lr_final: float = 0.0       # dynamics trainer: linearly decay `lr` to this; 0 =
                                #   constant after warmup (its config sets 5e-5)
    lr_decay_steps: int = 0     # dynamics trainer: WHERE that decay sits. > 0: over the
                                #   last `lr_decay_steps` steps BEFORE the bootstrap ramp
                                #   opens, so the first bootstrap row meets `lr_final`
                                #   already (its config sets 1000). 0: across the ramp
                                #   itself (diverges on the obstacle maze, 2026-09-18),
                                #   or over the post-warmup steps when there is no ramp
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    ema_decay: float = 0.999    # weight EMA used for eval/artifacts; 0 = off
    amp: bool = True            # bf16 autocast for the tokenizer forward
    accum_steps: int = 1        # gradient accumulation (for big frames)


@dataclass
class TokenizerTrainConfig:
    """Top-level config for ``dreamer4.train.train_tokenizer``."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    out: str = "runs/tokenizer"
    steps: int = 16000
    seed: int = 0
    device: str = "cuda"
    log_every: int = 50
    val_every: int = 500
    val_clips: int = 128        # fixed validation clips (deterministic)
    eval_batch: int = 32
    ckpt_every: int = 2000
    resume: bool = False        # continue from <out>/checkpoints/latest.pt
    init_from: str = ""         # warm-start weights from this checkpoint
    freeze_encoder: bool = False  # decoder-only fine-tune (latent space frozen)
    latent_noise_max: float = 0.0   # decoder robustness: max sigma of z-noise
    latent_noise_warmup_frac: float = 0.0  # hold noise at 0 for this frac of steps
    proprio_dropout: float = 0.0    # per-sample prob of -2 sentinel proprio input
    max_minutes: float = 0.0    # wall-clock budget; 0 = none
    tag: str = ""               # free-form label recorded in experiments.jsonl


# ---------------------------------------------------------------------------
# Dynamics training config
# ---------------------------------------------------------------------------
#
# The defaults ARE the reference world-model recipe, ONE run of 30 000 steps:
# clean-context objective with 1-step scheduled sampling (ramped to 0.7),
# ctx-noise band 0.05-0.15, image batches 0.15 taken from episode frame 0,
# joint proprio w0.3 -- and the K=1-maker, the bootstrap term, switched on for
# the last 20 %: fraction 0 until step 24 000, 0 -> 0.5 by 25 500, held. The LR
# walks 3e-4 -> 5e-5 over the 1 000 steps BEFORE the ramp opens
# (optim.lr_decay_steps), so the first bootstrap row already meets the low rate.
# Decaying across the ramp instead -- the earlier recipe, 42 000 steps with
# bootstrap_start_frac 0.55 / ramp 0.10 -- diverges on the obstacle maze
# ~1 300 steps into the ramp and never recovers (2026-09-18). Measured on the
# obstacle maze, H=39: latent MSE 3e-4 at K=1 and K=4, player error 0.03 / 0.07.
# `--objective.bootstrap_frac 0` trains without the term (K=4 only).
# See scripts/train_gridworld.sh for the end-to-end recipe.


@dataclass
class DynamicsDataConfig:
    """What to train the world model on. Episodes must carry actions."""

    path: str = ""              # dataset dir(s), comma-separated (see dreamer4.data)
    val_frac: float = 0.05      # episode fraction held out for validation
    seq_len: int = 4            # training window (frames); 4 is a SHARP optimum
    batch_size: int = 64
    proprio: str = "auto"       # auto (the adapter's default state), none, or an
                                # adapter's own mode -- see dreamer4.data.PROPRIO_MODES
    cameras: str = ""           # LeRobot only: restrict/order cameras
    camera_layout: str = "hstack"   # LeRobot only: camera tiling
    episode_cache: int = 64     # decoded episodes kept while pre-encoding


@dataclass
class DynamicsModelConfig:
    """World-model architecture (see dreamer4.models.dynamics.DynamicsModel)."""

    d_model: int = 256
    depth: int = 8
    n_heads: int = 4
    n_kv_heads: int = 2
    n_register: int = 4
    k_max: int = 4              # finest shortcut grid (d_min = 1/k_max)
    K: int = 4                  # denoising steps per frame at inference/eval
    time_every: int = 4         # time attention every N layers. d8/te4 gates
                                # as well as te2 with half the time layers:
                                # TOTAL depth sets the quality class.
    logit_cap: float = 50.0


@dataclass
class TokenizerRefConfig:
    """The frozen tokenizer that defines the latent space."""

    ckpt: str = ""              # train_tokenizer checkpoint (required)
    decoder_ckpt: str = ""      # optional robust-decoder ckpt (same encoder)
    history: int = 4            # sliding temporal window (1 = per-frame)
    pack_k: int = 1             # bottleneck packing (n_spatial = n_latents/pack_k)


@dataclass
class ObjectiveConfig:
    """The training objective (see dreamer4.train.dynamics_objectives)."""

    objective: str = "clean_context"    # clean_context | shortcut_forcing (paper eq 4/7)
    tau_ctx: float = 0.1                # inference-time context corruption level
    ctx_noise_min: float = 0.05         # context-noise band (robustness to
    ctx_noise_max: float = 0.15         #   imperfect rollout context)
    sched_sample_prob: float = 0.7      # scheduled sampling target prob (0.7 tightens
                                        #   seed spread ~7x vs 0.5)
    sched_warmup_frac: float = 0.4      # ramp sched prob 0 -> target over this frac
    image_batch_prob: float = 0.15      # prob of a T=1 no-context step (dream-from-
                                        #   scratch); 0.3 measured worse
    bootstrap_frac: float = 0.5         # TARGET bootstrap batch fraction; >0 adds the
                                        #   shortcut bootstrap term (makes K=1/2/4
                                        #   inference all legal); 0 = K=4 only
    bootstrap_start_frac: float = 0.8   # hold the fraction at 0 for this frac of the
                                        #   run before ramping (0 = target from step 1,
                                        #   what a warm-started fine-tune wants)
    bootstrap_ramp_frac: float = 0.05   # ramp 0 -> bootstrap_frac over this frac of the
                                        #   run, then hold to the end (0 = step change)
    boot_weight: float = 0.5            # relative weight of the bootstrap term
    proprio_weight: float = 0.3         # relative weight of the joint proprio term


@dataclass
class DynamicsEvalConfig:
    """Validation = the WINDOWED open-loop rollout (the only honest gate)."""

    ctx: int = 2                # real context frames before dreaming
    horizon: int = 10           # dreamed steps during training validation
    episodes: int = 64          # validation episodes per eval
    final_horizons: str = "10,39"   # comma list: the full gate after training
    final_Ks: str = "1,4"           # comma list: denoise-step counts to gate


@dataclass
class DynamicsTrainConfig:
    """Top-level config for ``dreamer4.train.train_dynamics``."""

    data: DynamicsDataConfig = field(default_factory=DynamicsDataConfig)
    model: DynamicsModelConfig = field(default_factory=DynamicsModelConfig)
    tokenizer: TokenizerRefConfig = field(default_factory=TokenizerRefConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(
        grad_clip=0.5, amp=False,       # the reference dynamics recipe is fp32
        lr_final=5e-5, lr_decay_steps=1000))   # low LR BEFORE the bootstrap ramp
    eval: DynamicsEvalConfig = field(default_factory=DynamicsEvalConfig)

    out: str = "runs/dynamics"
    steps: int = 30000          # 24 000 plain + 6 000 with the bootstrap term
    seed: int = 0
    device: str = "cuda"
    log_every: int = 50
    val_every: int = 2000
    ckpt_every: int = 4000
    resume: bool = False        # continue from <out>/checkpoints/latest.pt
    init_from: str = ""         # warm-start weights from this checkpoint
    max_minutes: float = 0.0    # wall-clock budget; 0 = none
    tag: str = ""               # free-form label recorded in experiments.jsonl


# ---------------------------------------------------------------------------
# dict <-> dataclass
# ---------------------------------------------------------------------------


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def _nested_config_type(f: dataclasses.Field) -> Optional[type]:
    """The dataclass type of a nested-config field, or None for leaf fields."""
    if f.default_factory is not dataclasses.MISSING:
        obj = f.default_factory()
        if dataclasses.is_dataclass(obj):
            return type(obj)
    return None


def config_from_dict(cls: Type[C], data: Dict[str, Any], _path: str = "") -> C:
    """Build ``cls`` from a nested dict, rejecting unknown keys."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} at "
                       f"'{_path or cls.__name__}' — valid: {sorted(fields)}")
    defaults = cls()
    kwargs: Dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            continue
        nested = _nested_config_type(f)
        if nested is not None:
            kwargs[name] = config_from_dict(nested, dict(data[name]), f"{_path}{name}.")
        else:
            kwargs[name] = _coerce(data[name], type(getattr(defaults, name)),
                                   f"{_path}{name}")
    return cls(**kwargs)


def _coerce(value: Any, target: type, path: str) -> Any:
    if target is str and value is None:
        return ""                      # YAML `key:` (empty) means "unset"
    if target is bool:
        if isinstance(value, bool):
            return value
        return _str2bool(str(value))
    if target is float and isinstance(value, (int, float)):
        return float(value)
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"config '{path}' expects int, got {value!r}")
        return value
    if target is str:
        return str(value)
    if not isinstance(value, target):
        raise TypeError(f"config '{path}' expects {target.__name__}, got {value!r}")
    return value


def _str2bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got '{s}'")


# ---------------------------------------------------------------------------
# YAML + CLI
# ---------------------------------------------------------------------------


def save_config(cfg: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))


def _add_dataclass_args(parser: argparse.ArgumentParser, cls: type,
                        prefix: str = "") -> None:
    for f in dataclasses.fields(cls):
        nested = _nested_config_type(f)
        if nested is not None:
            _add_dataclass_args(parser, nested, prefix=f"{prefix}{f.name}.")
            continue
        default = getattr(cls(), f.name)
        kind = type(default)
        converter = _str2bool if kind is bool else kind
        parser.add_argument(f"--{prefix}{f.name}", type=converter,
                            default=argparse.SUPPRESS, metavar=kind.__name__,
                            help=f"(default: {default!r})")


def parse_config(cls: Type[C], argv: Optional[Sequence[str]] = None,
                 description: str = "") -> C:
    """
    defaults < ``--config`` YAML < CLI flags  ->  a validated config object.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="YAML file with (partial) config overrides")
    _add_dataclass_args(parser, cls)
    ns = vars(parser.parse_args(argv))

    merged = config_to_dict(cls())
    config_path = ns.pop("config", None)
    if config_path:
        loaded = yaml.safe_load(Path(config_path).read_text()) or {}
        _deep_update(merged, loaded)
    for dotted, value in ns.items():
        node = merged
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return config_from_dict(cls, merged)


def _deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> None:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def parse_camera_list(cameras: str) -> Optional[List[str]]:
    """'' -> None; 'a,b' -> ['a', 'b'] (for DataConfig.cameras)."""
    names = [c.strip() for c in cameras.split(",") if c.strip()]
    return names or None
