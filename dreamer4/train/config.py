"""
Typed configuration for the training entrypoints.

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

Defaults below ARE the production gridworld recipe (train_recipe_16tok.sh
STEP 1, 2026-07-06): a d256/depth-2 tokenizer, 16 latents x 16 dims,
loss-normalized L1 + LPIPS(up128) w2, EMA 0.999, bf16 autocast.
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
    proprio: str = "none"       # none|player|player_goal (gridworld)|auto
                                # (dataset-stored, e.g. LeRobot observation.state)
    episode_cache: int = 999    # LeRobot: decoded episodes kept in memory per worker


@dataclass
class ModelConfig:
    """Tokenizer architecture (see dreamer4.models.tokenizer)."""

    patch_size: int = 4
    d_model: int = 256          # WIDTH is the quality lever (not depth)
    depth: int = 2              # encoder and decoder each
    n_heads: int = 4
    n_kv_heads: int = 2
    n_latents: int = 16         # bottleneck tokens per frame (16 = gridworld floor)
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

    recon: str = "l1"           # l1 (crisper on flat content) | mse (paper)
    recon_weight: float = 1.0
    perceptual_backbone: str = "lpips"  # none|lpips|dinov3|hybrid
    perceptual_weight: float = 2.0      # primary perceptual term weight
    perceptual_up: int = 128            # LPIPS upscale resolution
    lpips_net: str = "alex"
    dino_weight: float = 1.0            # DINOv3 weight in hybrid mode
    consistency_weight: float = 0.0     # re-encode(recon) ~ z; trains decoder only
    proprio_weight: float = 0.3         # 0.1-0.3 sweet spot; 1.0 starves pixels
    loss_norm: bool = True              # divide each term by its running RMS
    loss_norm_decay: float = 0.99
    loss_norm_floor: float = 0.2        # divisor floor as frac of peak RMS
                                        # (0.2 > 0.05 > naive — 2026-07-04 sweep)


@dataclass
class OptimConfig:
    """Optimizer, schedule and train-time model averaging."""

    lr: float = 3e-4
    warmup: int = 500           # linear LR warmup steps
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
    tag: str = ""               # free-form label for the experiment journal


# ---------------------------------------------------------------------------
# Dynamics training config
# ---------------------------------------------------------------------------
#
# Defaults ARE the production gridworld recipe STEP 3 (train_recipe_16tok.sh,
# 2026-07-06 + the 2026-07-20 image-batch-frame0 finding): clean-context
# objective with 1-step scheduled sampling (ramped to 0.7), ctx-noise band
# 0.05-0.15, image batches 0.15 from episode frame 0, joint proprio w0.3.
# STEP 4 (the K=1-maker) = same config + init_from STEP 3's checkpoint +
# objective.bootstrap_frac 0.5 + objective.sched_warmup_frac 0.3 +
# optim.lr 5e-5 + optim.warmup 200 + steps 36000.


@dataclass
class DynamicsDataConfig:
    """What to train the world model on. Episodes must carry actions."""

    path: str = ""              # dataset dir(s), comma-separated (see dreamer4.data)
    val_frac: float = 0.05      # episode fraction held out for validation
    seq_len: int = 4            # training window (frames); 4 is a SHARP optimum
    batch_size: int = 64
    proprio: str = "none"       # none|player (gridworld)|auto (dataset-stored)
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
    time_every: int = 4         # d8/te4: same gate class as te2 at half the
                                # time layers (2026-07-21 depth ladder; user
                                # default). TOTAL depth sets the class.
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
                                        #   scratch; KEEP 0.15 — 0.3 retracted)
    image_batch_frame0: bool = True     # image batches use episode frame 0 only
                                        #   (terminal frames taught sprite-dropping)
    bootstrap_frac: float = 0.0         # >0 adds the shortcut bootstrap term (STEP 4:
                                        #   0.5 — trains flexible inference K=1/2/4)
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
        grad_clip=0.5, amp=False))      # the production dynamics runs are fp32
    eval: DynamicsEvalConfig = field(default_factory=DynamicsEvalConfig)

    out: str = "runs/dynamics"
    steps: int = 24000
    seed: int = 0
    device: str = "cuda"
    log_every: int = 50
    val_every: int = 2000
    ckpt_every: int = 4000
    resume: bool = False        # continue from <out>/checkpoints/latest.pt
    init_from: str = ""         # warm-start weights (bootstrap-LATER fine-tune)
    max_minutes: float = 0.0    # wall-clock budget; 0 = none
    tag: str = ""               # free-form label for the experiment journal


# ---------------------------------------------------------------------------
# Agent training config (phase 2 — BC + reward + continue on agent tokens)
# ---------------------------------------------------------------------------
#
# The dynamics ARCHITECTURE is not configured here: the agent trainer
# requires ``--init_from`` (a train_dynamics checkpoint) and rebuilds that
# exact model with ``n_agent`` agent tokens in ``wm_agent`` mode. Defaults =
# the recipe: warm-start the whole transformer at the proven ft lr 5e-5,
# fresh heads at 3e-4, keep the phase-1 clean-context WM loss running
# (paper: "we continue to apply the video prediction loss"), BC only on
# low-noise non-sticky episodes (the paper's task-relevant 50% analog).


@dataclass
class AgentDataConfig:
    """What to finetune on. Episodes must carry actions AND rewards."""

    path: str = ""              # dataset dir(s), comma-separated
    val_frac: float = 0.05      # episode fraction held out for validation
    seq_len: int = 4            # forward window — MUST stay the WM's trained window
    batch_size: int = 64        # world-model (clean-context) batch
    agent_batch_size: int = 64  # agent-heads batch (the second forward)
    proprio: str = "player"     # must match the warm-start dynamics model
    bc_frac: float = 0.5        # agent-batch fraction drawn from episodes the
                                #   dataset says are worth imitating (paper's
                                #   50/50 relevant/uniform). NOT a BC-quality
                                #   knob — BC already ignores the rest via the
                                #   row weight. It buys the REWARD head states
                                #   an expert never visits, which is exactly
                                #   where a phase-3 dream wanders.
    end_frac: float = 0.25      # per-sample prob of pinning the window to the
                                #   episode END (terminal frames for the
                                #   reward heads)
    episode_cache: int = 64     # decoded episodes kept while pre-encoding


@dataclass
class AgentModelConfig:
    """Agent-token + head architecture (see dreamer4.models.agent)."""

    n_agent: int = 3            # agent tokens per timestep; slots: 0 policy,
                                #   1 reward, 2 value (phase 3)
    num_tasks: int = 1          # gridworld is single-task (constant id 0)
    mtp_length: int = 3         # MTP horizon; window 4 supports n=0..2
                                #   (paper L=8 needs long contexts)
    head_mlp_depth: int = 2     # hidden layers in each head MLP
    head_mlp_ratio: float = 2.0  # head hidden = d_model * this
    num_bins: int = 255         # SymExpTwoHot bins (reward/value heads)
    bin_low: float = -20.0      # two-hot range in SYMLOG space; +-20 covers
    bin_high: float = 20.0      #   +-e^20 (Minecraft-scale) — gridworld's
                                #   +-1 rewards decode cleaner on +-3


@dataclass
class AgentLossConfig:
    """Agent loss terms; RELATIVE weights (everything is RMS-normalized
    together with the WM terms — paper: no hand-tuned scales)."""

    bc_weight: float = 1.0
    reward_weight: float = 0.3      # near-trivial modality: small weight
                                    #   suffices (2026-07-04 lossnorm rule).
                                    #   Termination is DERIVED from this head
                                    #   (no continue head) — gated by term_f1.


@dataclass
class AgentOnlineEvalConfig:
    """Online policy evaluation in the real GridWorld-v0 env."""

    episodes: int = 300         # episodes per online eval
    seed: int = 1000000         # env layout seeds (disjoint from the dataset's)
    every: int = 0              # eval every N steps (0 = only after training)


@dataclass
class AgentTrainConfig:
    """Top-level config for ``dreamer4.train.train_agent`` (phase 2)."""

    data: AgentDataConfig = field(default_factory=AgentDataConfig)
    agent: AgentModelConfig = field(default_factory=AgentModelConfig)
    loss: AgentLossConfig = field(default_factory=AgentLossConfig)
    tokenizer: TokenizerRefConfig = field(default_factory=TokenizerRefConfig)
                                # empty ckpt = reuse the dynamics checkpoint's
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(
        lr=5e-5, warmup=200, grad_clip=0.5, amp=False))
    eval: DynamicsEvalConfig = field(default_factory=DynamicsEvalConfig)
    online: AgentOnlineEvalConfig = field(default_factory=AgentOnlineEvalConfig)

    out: str = "runs/agent"
    steps: int = 20000
    seed: int = 0
    device: str = "cuda"
    head_lr: float = 3e-4       # fresh heads (policy/reward/continue/task)
    log_every: int = 50
    val_every: int = 2000
    ckpt_every: int = 4000
    resume: bool = False
    init_from: str = ""         # REQUIRED: train_dynamics checkpoint to finetune
    max_minutes: float = 0.0
    tag: str = ""


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
