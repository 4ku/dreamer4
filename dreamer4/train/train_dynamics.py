"""
Phase 1b: train the Dreamer 4 dynamics model (world model) in a frozen
tokenizer's latent space.

In: a dataset whose episodes carry ACTIONS, plus the phase-1a tokenizer
checkpoint that defines the latent space. Out: a dynamics checkpoint that the
later phases load frozen (:mod:`dreamer4.train.train_heads`,
:mod:`dreamer4.train.train_pmpo`) together with its evaluation artifacts.

Dataset-agnostic: anything :func:`dreamer4.data.open_video_dataset` can read
that carries actions — gridworld shards (discrete moves as one-hot) or
LeRobot datasets (parquet action vectors; ``observation.state`` becomes the
joint proprio stream via ``--data.proprio auto``). Episodes are encoded ONCE
by the frozen tokenizer (sliding ``tokenizer.history`` window — the temporal
receptive field stays fixed while the world model rolls any horizon) and
cached; training then samples latent windows from that cache.

The objectives live in :mod:`dreamer4.train.dynamics_objectives`; validation
is the WINDOWED open-loop rollout (:mod:`dreamer4.dynamics_eval`) — the same
dream loop phase 3 imagines in, scored by the dataset's own domain hooks
(gridworld: sprite errors + the ≤1-cell gate).

Usage (the defaults ARE the full phase, one run of 30 000 steps):

    python -m dreamer4.train.train_dynamics \\
        --data.path data/gridworld_10k \\
        --tokenizer.ckpt runs/tok_x/checkpoints/latest.pt \\
        --out runs/dyn_x

    # what those defaults are: the bootstrap term that makes K=1 sampling
    # accurate is switched on for the last 20 % of the SAME run -- fraction 0
    # until step 24 000, 0 -> 0.5 by 25 500, held; the LR walks 3e-4 -> 5e-5
    # over the 1 000 steps BEFORE the ramp opens (decaying across the ramp
    # diverges on the obstacle maze) and the loss normalizer is re-seeded where
    # it opens. Spelled out:
    ... --steps 30000 --optim.lr_final 5e-5 --optim.lr_decay_steps 1000 \\
        --objective.bootstrap_frac 0.5 \\
        --objective.bootstrap_start_frac 0.8 --objective.bootstrap_ramp_frac 0.05

    # without the bootstrap term (K=4 only): --objective.bootstrap_frac 0
    # bootstrap fine-tune of an existing model: --init_from <ckpt> --optim.lr 5e-5
    #   --optim.lr_final 0 --objective.bootstrap_start_frac 0 --objective.bootstrap_ramp_frac 0.25

Artifacts under ``--out``: ``config.yaml``, ``tb/``, resumable
``checkpoints/latest.pt``, ``demo.gif``, ``final.json`` (the full
horizon × K gate table); one line appended to ``../experiments.jsonl``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dreamer4.data import EpisodeVideoDataset, open_video_dataset, split_train_val
from dreamer4.models.dynamics import DynamicsModel, align_actions
from dreamer4.dynamics_eval import gate_rollout, one_step_eval, save_demo_gif
from dreamer4.models.tokenizer import FrozenTokenizer
from dreamer4.train.common import (EMA, LossCombiner, atomic_torch_save,
                                   set_seed, setup_run_logging,
                                   swap_in_weights, warmup_lr)
from dreamer4.train.config import (DynamicsModelConfig, DynamicsTrainConfig,
                                   ObjectiveConfig, config_from_dict,
                                   config_to_dict, parse_camera_list,
                                   parse_config, save_config)
from dreamer4.train.dynamics_objectives import (clean_context_loss,
                                                shortcut_forcing_loss)
from dreamer4.train.train_tokenizer import (weights_from_checkpoint,
                                            make_recon_strip)

OBJECTIVES = ("clean_context", "shortcut_forcing")


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def bootstrap_frac_at(obj: ObjectiveConfig, step: int, total_steps: int) -> float:
    """
    Bootstrap batch fraction at ``step`` (1-based) — the K=1-maker's ramp.

    0 for the first ``bootstrap_start_frac`` of the run, then a monotone
    linear ramp to ``bootstrap_frac`` over ``bootstrap_ramp_frac`` of it,
    then held at the target to the end. Distilling two half-steps into one is
    only worth supervising once the flow term has a model to distill, so the
    term arrives late — but arriving late as a STEP shocks the loss balance,
    hence the ramp.

    Both fractions default to 0, which returns the target from step 1: the
    plain constant ``bootstrap_frac`` older configs and checkpoints carry.
    """
    start = obj.bootstrap_start_frac * total_steps
    ramp = obj.bootstrap_ramp_frac * total_steps
    if step >= start + ramp:
        return obj.bootstrap_frac
    if step <= start:
        return 0.0
    return obj.bootstrap_frac * (step - start) / ramp


def decayed_lr(optim, step: int, total_steps: int, *, decay_from: int = 0,
               decay_to: int = 0) -> float:
    """
    Linear warmup to ``lr``, then a linear decay to ``lr_final`` (0 = off).

    ``decay_from``/``decay_to`` place the decay on the bootstrap ramp, which is
    where the run changes regime: the two-phase recipe this schedule replaces
    dropped the LR 6x for its bootstrap phase, and holding the high rate
    through the ramp costs a long excursion the model then has to heal from
    (measured). Without a window the decay spans the post-warmup steps.
    """
    lr = warmup_lr(optim.lr, step, optim.warmup)
    if optim.lr_final <= 0.0:
        return lr
    windowed = decay_to > decay_from
    lo = decay_from if windowed else optim.warmup
    hi = decay_to if windowed else total_steps
    if step <= lo:
        return lr
    frac = min(1.0, (step - lo) / max(1, hi - lo))
    return optim.lr * (1.0 - frac) + optim.lr_final * frac


# ---------------------------------------------------------------------------
# Model building & checkpoints
# ---------------------------------------------------------------------------


def build_dynamics(model: DynamicsModelConfig, *, n_spatial: int,
                   d_spatial: int, action_dim: int,
                   d_proprio: Optional[int], max_T: int, n_agent: int = 0,
                   space_mode: str = "wm_agent") -> DynamicsModel:
    """
    Build the dynamics model for one tokenizer's latent space.

    Args:
        model:      Architecture hyper-parameters.
        n_spatial:  Latent tokens per frame (Nz).
        d_spatial:  Width of one latent token (Dz).
        action_dim: Width of the ALIGNED action vector — the dataset's action
                    dim PLUS the start flag (see ``align_actions``).
        d_proprio:  Joint proprio width, or None when the dataset has none.
        max_T:      Longest time axis the RoPE tables and KV cache must serve.
        n_agent:    Phase-1 pretraining uses the default 0: with no AGENT
                    segment the ``wm_agent`` mask degenerates to full
                    world-to-world mixing. Phases 2-3 rebuild the same
                    checkpointed architecture with ``n_agent > 0``, and
                    world-token outputs are unchanged by construction because
                    world rows never attend AGENT columns.
        space_mode: Attention mask mode of the backbone.

    Returns:
        An untrained ``DynamicsModel`` (on the default device; callers move it).
    """
    return DynamicsModel(
        d_model=model.d_model, d_spatial=d_spatial, n_spatial=n_spatial,
        n_register=model.n_register, n_agent=n_agent, n_heads=model.n_heads,
        n_kv_heads=model.n_kv_heads, depth=model.depth, k_max=model.k_max,
        action_dim=action_dim, mlp_ratio=8 / 3, time_every=model.time_every,
        use_qk_norm=True,
        logit_cap=(model.logit_cap if model.logit_cap > 0 else None),
        space_mode=space_mode, max_T=max_T, d_proprio=d_proprio)


def load_dynamics_checkpoint(path, *, device: str | torch.device = "cpu",
                             prefer_ema: bool = True):
    """
    Rebuild a dynamics model from a ``train_dynamics`` checkpoint.

    Args:
        path:       Checkpoint written by this trainer (phase 1b).
        device:     Where to place the rebuilt model.
        prefer_ema: Load the EMA weights when the checkpoint carries them.

    Returns:
        ``(model.eval(), checkpoint_dict)``. The checkpoint carries the full
        nested ``config``, ``latent_meta`` (n_spatial / d_spatial /
        action_dim / d_proprio / max_T / window) and ``tokenizer_ckpt``, the
        path of the tokenizer whose latent space it was trained in.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["latent_meta"]
    model_cfg = config_from_dict(DynamicsModelConfig, ckpt["config"]["model"])
    dyn = build_dynamics(model_cfg, n_spatial=meta["n_spatial"],
                         d_spatial=meta["d_spatial"],
                         action_dim=meta["action_dim"],
                         d_proprio=meta["d_proprio"], max_T=meta["max_T"])
    dyn.load_state_dict(weights_from_checkpoint(ckpt, prefer_ema))
    return dyn.to(device).eval(), ckpt


# ---------------------------------------------------------------------------
# Latent episode cache
# ---------------------------------------------------------------------------


def encode_episodes(tokenizer: FrozenTokenizer, dataset: EpisodeVideoDataset,
                    indices, *, keep_video: bool = False,
                    chunk: int = 512) -> List[dict]:
    """
    Encode whole episodes ONCE into cached latents (fp16, on device).

    Args:
        indices:    Episode indices to encode.
        keep_video: Also keep the raw uint8 frames (validation needs them to
                    score a decoded dream against the real pixels).
        chunk:      Frames encoded per tokenizer call.

    Returns:
        One dict per episode: {"z": (Ti, Nz, Dz) fp16, "actions": (Ti-1, Da)
        float32, "proprio"?: (Ti, d_p) float32, "video"?: (Ti, H, W, C) uint8,
        "rewards"?/"terminals"?: (Ti-1,) when the dataset records them,
        "meta"?: collector metadata (see ``episode_meta``), "bc_weight": how
        much this episode may be imitated (see ``bc_weight``)}.

    Long episodes are encoded in chunks carrying ``history-1`` frames of left
    context, which yields latents identical to a single pass: the tokenizer's
    sliding window never looks more than ``history`` frames back.

    Phase 2 imports this function for its own cache, which is why an entry
    carries rewards, terminals, metadata and ``bc_weight`` that this trainer
    never reads.
    """
    episodes = []
    overlap = tokenizer.history - 1
    for i in indices:
        n = dataset.episode_frames(int(i))
        clip = dataset.clip(int(i), 0, n)
        video = clip["video"]
        parts = []
        for s in range(0, n, chunk):
            lo = max(0, s - overlap)
            z = tokenizer.encode_frames(video[None, lo : s + chunk])
            parts.append(z[0, s - lo :].to(torch.float16))
        entry = {"z": torch.cat(parts, 0), "actions": clip["actions"]}
        for key in ("proprio", "rewards", "terminals"):
            if key in clip:
                entry[key] = clip[key]
        meta = dataset.episode_meta(int(i))
        if meta:
            entry["meta"] = meta
        entry["bc_weight"] = dataset.bc_weight(int(i))
        if keep_video:
            entry["video"] = video
        episodes.append(entry)
    return episodes


def sample_windows(episodes: List[dict], B: int, T: int,
                   rng: np.random.Generator, device: torch.device, *,
                   start_only: bool = False, skip_terminal: bool = False,
                   with_video: bool = False, replace: bool = True) -> Dict[str, torch.Tensor]:
    """
    Draw B random length-T windows from the latent cache.

    Args:
        episodes:   Cache built by :func:`encode_episodes`.
        B, T:       Windows to draw, and frames per window.
        start_only: Pin every window to frame 0 (the episode-start
                    distribution) instead of a uniform random offset.
        skip_terminal: Never let a window end on an episode's terminal frame
                    (the frame a terminal transition lands in). Single-frame
                    batches use it: in a terminal frame the player sprite can
                    be hidden, and generating such frames from scratch teaches
                    the model to drop sprites.
        with_video: Also return the raw frames of each window.
        replace:    Sample episodes with replacement. With False the batch is
                    capped at the number of eligible episodes, which keeps a
                    validation batch free of duplicates.

    Returns:
        {"z": (B, T, Nz, Dz) float32 on device, "actions": (B, T-1, Da),
        "proprio"?: (B, T, d_p), "video"?: (B, T, H, W, C) uint8 on cpu}.
    """
    def ends_terminal(e):
        return bool(skip_terminal and "terminals" in e and len(e["terminals"])
                    and e["terminals"][-1])

    eligible = [e for e in episodes if e["z"].shape[0] - ends_terminal(e) >= T]
    if not eligible:
        raise ValueError(f"no episode has >= {T} frames")
    if not replace and len(eligible) < B:
        B = len(eligible)
    picks = (rng.choice(len(eligible), size=B, replace=replace)
             if replace else rng.permutation(len(eligible))[:B])
    zs, acts, props, vids = [], [], [], []
    for j in picks:
        e = eligible[int(j)]
        n = e["z"].shape[0] - ends_terminal(e)
        s = 0 if start_only else int(rng.integers(0, n - T + 1))
        zs.append(e["z"][s : s + T])
        acts.append(e["actions"][s : s + T - 1])
        if "proprio" in e:
            props.append(e["proprio"][s : s + T])
        if with_video:
            vids.append(e["video"][s : s + T])
    out = {"z": torch.stack(zs).float().to(device),
           "actions": torch.from_numpy(np.stack(acts)).float().to(device)}
    if props:
        out["proprio"] = torch.from_numpy(np.stack(props)).to(device)
    if vids:
        out["video"] = torch.from_numpy(np.stack(vids))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: DynamicsTrainConfig) -> Dict[str, float]:
    """
    Run dynamics training to completion.

    Args:
        cfg: Fully resolved config (dataclass defaults < YAML < CLI flags).

    Returns:
        The contents of ``final.json``: the primary gate row (largest horizon,
        last K), the one-step probe, the full horizon x K gate table under
        ``"gate"``, plus step count, parameter count and wall-clock minutes.
    """
    if cfg.objective.objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logging(out)
    # Resolve to an ABSOLUTE path before anything records it: this path is embedded in every
    # checkpoint and reopened much later by WorldModel. Left relative it resolves against
    # whatever directory the READER happens to run in, so an evaluation started one directory up
    # dies on FileNotFoundError.
    if cfg.tokenizer.ckpt:
        cfg.tokenizer.ckpt = str(Path(cfg.tokenizer.ckpt).resolve())
    save_config(cfg, out / "config.yaml")
    writer = SummaryWriter(log_dir=str(out / "tb"))

    # ---- frozen tokenizer + data ----------------------------------------
    if not cfg.tokenizer.ckpt:
        raise SystemExit("--tokenizer.ckpt is required (a train_tokenizer "
                         "checkpoint defining the latent space)")
    tokenizer = FrozenTokenizer(
        cfg.tokenizer.ckpt, history=cfg.tokenizer.history,
        pack_k=cfg.tokenizer.pack_k,
        decoder_ckpt=(cfg.tokenizer.decoder_ckpt or None), device=device)
    log.info(f"tokenizer: {cfg.tokenizer.ckpt} | latents "
             f"{tokenizer.n_spatial}x{tokenizer.d_spatial} "
             f"history={tokenizer.history} frame "
             f"{tokenizer.H}x{tokenizer.W}x{tokenizer.C}")

    dataset = open_video_dataset(
        cfg.data.path, proprio=cfg.data.proprio, actions=True,
        camera_layout=cfg.data.camera_layout,
        cameras=parse_camera_list(cfg.data.cameras),
        episode_cache=cfg.data.episode_cache)
    if dataset.action_dim is None:
        raise SystemExit(f"{cfg.data.path} carries no actions — the dynamics "
                         "model is action-conditioned")
    d_proprio = dataset.proprio_dim
    train_eps, val_eps = split_train_val(len(dataset),
                                         val_frac=cfg.data.val_frac,
                                         seed=cfg.seed, min_val=64)
    t_enc = time.time()
    train_cache = encode_episodes(tokenizer, dataset, train_eps)
    val_cache = encode_episodes(tokenizer, dataset, val_eps, keep_video=True)
    n_frames = sum(e["z"].shape[0] for e in train_cache)
    log.info(f"device={device} episodes={len(dataset)} "
             f"(train {len(train_cache)} / val {len(val_cache)}) | "
             f"pre-encoded {n_frames} train frames in {time.time() - t_enc:.1f}s | "
             f"action_dim={dataset.action_dim} proprio={d_proprio}")

    # ---- model / losses / optimizer -------------------------------------
    action_dim = dataset.action_dim + 1              # + start flag
    max_T = cfg.data.seq_len + 8                     # RoPE/KV-cache headroom
    dyn = build_dynamics(cfg.model, n_spatial=tokenizer.n_spatial,
                         d_spatial=tokenizer.d_spatial, action_dim=action_dim,
                         d_proprio=d_proprio, max_T=max_T).to(device)
    if cfg.init_from:
        init = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        dyn.load_state_dict(weights_from_checkpoint(init, prefer_ema=True))
        log.info(f"warm-started from {cfg.init_from} (step {init.get('step', '?')})")
    n_params = sum(p.numel() for p in dyn.parameters())
    log.info(f"dynamics params: {n_params / 1e6:.2f}M")

    obj = cfg.objective
    if obj.objective == "shortcut_forcing" and d_proprio is not None:
        raise SystemExit("joint proprio is supported by the clean_context "
                         "objective only")
    clean_objective = obj.objective == "clean_context"
    # The TARGET fraction decides the loss layout: a scheduled run starts at
    # fraction 0, so its bootstrap slot must already exist (and be normalized
    # against the other terms) by the time the ramp delivers the first rows.
    use_boot = clean_objective and obj.bootstrap_frac > 0.0
    weights = {"flow": 1.0,
               "proprio": obj.proprio_weight if d_proprio is not None else 0.0,
               "bootstrap": obj.boot_weight if use_boot else 0.0}
    # Normalize only when the clean-context objective has >1 active term: a
    # lone normalized flow term would just rescale the LR, and the paper
    # objective folds its bootstrap term in internally, so it always presents
    # a single term here.
    normalize_losses = d_proprio is not None or use_boot
    combiner = LossCombiner(weights, normalize=normalize_losses,
                            decay=0.99, floor_frac=0.2, device=device)
    log.info("objective: " + obj.objective + "  "
             + "  ".join(f"{k}={v}" for k, v in weights.items() if v)
             + ("  [RMS-normalized]" if combiner.normalizer else ""))
    boot_start = int(obj.bootstrap_start_frac * cfg.steps)
    boot_full = boot_start + int(obj.bootstrap_ramp_frac * cfg.steps)
    if obj.bootstrap_frac > 0.0 and boot_full > 0:
        log.info(f"bootstrap schedule: 0 until step {boot_start}, ramp to "
                 f"{obj.bootstrap_frac} by step {boot_full}, held to {cfg.steps}")
    if cfg.optim.lr_final > 0.0:
        lr_end = boot_full if boot_full > boot_start else cfg.steps
        if cfg.optim.lr_decay_steps > 0 and boot_start > 0:
            lr_end = boot_start          # decay finishes BEFORE the ramp opens
        log.info(f"lr schedule: {cfg.optim.lr:g} after warmup, decaying to "
                 f"{cfg.optim.lr_final:g} by step {lr_end}")
    opt = torch.optim.AdamW(dyn.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2),
                            weight_decay=cfg.optim.weight_decay)
    ema = EMA(dyn, cfg.optim.ema_decay) if cfg.optim.ema_decay > 0 else None

    latent_meta = {"n_spatial": tokenizer.n_spatial,
                   "d_spatial": tokenizer.d_spatial, "action_dim": action_dim,
                   "d_proprio": d_proprio, "max_T": max_T,
                   "window": cfg.data.seq_len}

    start_step = 0
    latest = ckpt_dir / "latest.pt"
    if cfg.resume and latest.exists():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        dyn.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if ema is not None:
            if ckpt.get("ema") is not None:
                ema.load_state_dict(ckpt["ema"])
            else:                       # prior run had EMA off: reseed shadow
                ema = EMA(dyn, cfg.optim.ema_decay)
        combiner.load_state_dict(ckpt.get("loss_norm"))
        start_step = int(ckpt["step"])
        log.info(f"resumed from {latest} at step {start_step}")

    def save_checkpoint(step: int) -> None:
        atomic_torch_save(latest, {
            "kind": "dreamer4-dynamics", "step": step,
            "config": config_to_dict(cfg), "latent_meta": latent_meta,
            "tokenizer_ckpt": cfg.tokenizer.ckpt,
            "model": dyn.state_dict(),
            "ema": None if ema is None else ema.state_dict(),
            "opt": opt.state_dict(), "loss_norm": combiner.state_dict()})

    # ---- fixed validation batches (deterministic across evals/resumes) ---
    val_rng = np.random.default_rng(cfg.seed + 1)
    gate_batch = sample_windows(
        val_cache, cfg.eval.episodes, cfg.eval.ctx + cfg.eval.horizon, val_rng,
        device, start_only=True, with_video=True, replace=False)
    onestep_batch = sample_windows(val_cache, min(64, cfg.eval.episodes),
                                   cfg.data.seq_len, val_rng, device)

    def run_validation(step: int) -> Dict[str, float]:
        eval_weights = ema.state_dict() if ema is not None else None
        with swap_in_weights(dyn, eval_weights):
            dyn.eval()
            metrics, (gt, dream) = gate_rollout(
                dyn, tokenizer, dataset, gate_batch, window=cfg.data.seq_len,
                ctx=cfg.eval.ctx, K=cfg.model.K, k_max=cfg.model.k_max,
                tau_ctx=obj.tau_ctx)
            metrics.update(one_step_eval(
                dyn, onestep_batch["z"], onestep_batch["actions"],
                ctx=cfg.eval.ctx, K=cfg.model.K, k_max=cfg.model.k_max,
                tau_ctx=obj.tau_ctx, proprio=onestep_batch.get("proprio")))
            dyn.train()
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", float(v), step)
        writer.add_image("val/rollout_gt_vs_dream",
                         make_recon_strip(gt, dream), step)
        log.info("  [val] step %d: %s" % (step, "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items())))
        return metrics

    # ---- loop ------------------------------------------------------------
    rng = np.random.default_rng(cfg.seed + 2)
    dyn.train()
    t0 = time.time()
    budget_s = cfg.max_minutes * 60.0
    loss_ema: Optional[float] = None
    # per-stream loss accumulators + instantaneous-rate state (see the log block)
    ctx_acc: Dict[str, list] = {}
    img_acc: Dict[str, list] = {}
    last_log_step, last_log_t = None, time.time()
    step = start_step

    for step in range(start_step + 1, cfg.steps + 1):
        if budget_s and (time.time() - t0) >= budget_s:
            log.info(f"[budget] {cfg.max_minutes} min reached at step {step - 1}")
            step -= 1
            break
        if cfg.optim.lr_decay_steps > 0 and boot_start > 0:
            # LR reaches lr_final BEFORE the first bootstrap row arrives
            lr = decayed_lr(cfg.optim, step, cfg.steps,
                            decay_from=max(cfg.optim.warmup,
                                           boot_start - cfg.optim.lr_decay_steps),
                            decay_to=boot_start)
        else:
            lr = decayed_lr(cfg.optim, step, cfg.steps,
                            decay_from=boot_start, decay_to=boot_full)
        for group in opt.param_groups:
            group["lr"] = lr
        boot_frac = bootstrap_frac_at(obj, step, cfg.steps)
        if use_boot and boot_start > 0 and step == boot_start + 1:
            # The ramp opens a NEW loss regime, and the RMS normalizer has to be
            # told. By now the flow term sits far below the peak its running
            # estimate remembers (`floor_frac` deliberately stops the divisor
            # following it down), while the bootstrap term is normalized against
            # its own first value and so arrives at O(1): left alone, the model
            # spends the rest of the run optimizing the bootstrap term almost
            # alone and the player vanishes from the rollout within ~2 000 steps
            # (measured — rollout latent_mse 3e-4 -> 3e-1). Re-seeding every slot
            # restores the 1 : 0.3 : 0.5 the weights ask for, which is what the
            # separate warm-started fine-tune used to get for free.
            combiner = LossCombiner(weights, normalize=normalize_losses,
                                    decay=0.99, floor_frac=0.2, device=device)
            log.info(f"step {step}: bootstrap ramp opens — loss normalizer re-seeded")

        clean = obj.objective == "clean_context"
        sched_p = obj.sched_sample_prob * min(
            1.0, step / max(1, int(obj.sched_warmup_frac * cfg.steps)))
        do_sched = clean and rng.random() < sched_p
        image_batch = clean and rng.random() < obj.image_batch_prob
        T_s = 1 if image_batch else cfg.data.seq_len

        opt.zero_grad(set_to_none=True)
        loss_value = 0.0
        raw_means: Dict[str, float] = {}
        for _ in range(cfg.optim.accum_steps):
            batch = sample_windows(
                train_cache, cfg.data.batch_size, T_s, rng, device,
                skip_terminal=image_batch)
            aligned = align_actions(batch["actions"], T_s)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=cfg.optim.amp):
                if not clean:                   # the paper objective, eq 4/7
                    loss_flow, sf_aux = shortcut_forcing_loss(
                        dyn, z1=batch["z"], actions=aligned,
                        k_max=cfg.model.k_max,
                        bootstrap_fraction=boot_frac)
                    terms = {"flow": loss_flow}
                    aux = {"flow_mse": float(sf_aux["flow_mse"]),
                           "boot_mse": float(sf_aux["boot_mse"])}
                else:
                    p_tgt = 0 if image_batch else int(rng.integers(1, T_s))
                    terms, aux = clean_context_loss(
                        dyn, batch["z"], aligned, p=p_tgt,
                        k_max=cfg.model.k_max, K=cfg.model.K,
                        tau_ctx=obj.tau_ctx,
                        ctx_noise_min=obj.ctx_noise_min or None,
                        ctx_noise_max=obj.ctx_noise_max or None,
                        do_sched=do_sched and not image_batch,
                        proprio=batch.get("proprio"),
                        bootstrap_frac=boot_frac)
                loss, raw = combiner(terms)
            (loss / cfg.optim.accum_steps).backward()
            loss_value += float(loss.detach()) / cfg.optim.accum_steps
            for k, v in raw.items():
                raw_means[k] = raw_means.get(k, 0.0) + v / cfg.optim.accum_steps
        raw = raw_means
        grad_norm = torch.nn.utils.clip_grad_norm_(dyn.parameters(),
                                                   cfg.optim.grad_clip)
        opt.step()
        if ema is not None:
            ema.update(dyn)

        loss_ema = loss_value if loss_ema is None else \
            0.98 * loss_ema + 0.02 * loss_value

        # Two streams, accumulated separately. `image_batch_prob` makes a
        # fraction of the steps T=1 batches with NO context — the
        # dream-from-scratch regime — whose loss is an order of magnitude above
        # a context-conditioned step's, so a single mixed curve is bimodal and
        # reads as "spiky and flat" whatever either mode is doing: on one
        # measured run 13.9% of its points sat above 50x the median while the
        # median itself was falling steadily. Each stream is averaged over the
        # window between two log points, so both curves stay dense even at an
        # 85/15 split.
        bucket = img_acc if image_batch else ctx_acc
        for name, value in list(raw.items()) + [("loss", loss_value),
                                                ("flow_mse", aux["flow_mse"]),
                                                ("boot_mse", aux.get("boot_mse", 0.0)),
                                                ("grad_norm", float(grad_norm))]:
            acc = bucket.setdefault(name, [0.0, 0])
            acc[0] += float(value)
            acc[1] += 1

        if step % cfg.log_every == 0:
            sps = (step - start_step) / max(time.time() - t0, 1e-9)
            # ...and the INSTANTANEOUS rate over the last window. The cumulative
            # form above can only drift toward the true rate, so it cannot answer
            # "is it slowing down?" — the question it looks like it answers.
            now = time.time()
            sps_inst = ((step - last_log_step) / max(now - last_log_t, 1e-9)
                        if last_log_step is not None else sps)
            last_log_step, last_log_t = step, now
            for k, v in raw.items():
                writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("train/loss_ema", loss_ema, step)
            writer.add_scalar("train/flow_mse", aux["flow_mse"], step)
            writer.add_scalar("train/boot_mse", aux.get("boot_mse", 0.0), step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("train/sched_prob", sched_p, step)
            writer.add_scalar("train/bootstrap_frac", boot_frac, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/steps_per_sec", sps, step)
            writer.add_scalar("train/steps_per_sec_inst", sps_inst, step)
            for prefix, acc in (("ctx", ctx_acc), ("img", img_acc)):
                for name, (total, count) in acc.items():
                    if count:
                        writer.add_scalar(f"{prefix}/{name}", total / count, step)
                acc.clear()
            log.info(f"step {step}/{cfg.steps} "
                     + " ".join(f"{k}={v:.4f}" for k, v in raw.items())
                     + f" loss_ema={loss_ema:.4f} gnorm={float(grad_norm):.2f}"
                     f" {sps:.1f}it/s")

        if step % cfg.val_every == 0:
            run_validation(step)
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            save_checkpoint(step)

    # ---- final gate table + artifacts ------------------------------------
    save_checkpoint(step)
    eval_weights = ema.state_dict() if ema is not None else None
    horizons = [int(h) for h in cfg.eval.final_horizons.split(",")]
    Ks = [int(k) for k in cfg.eval.final_Ks.split(",")]
    rows: List[Dict[str, float]] = []
    with swap_in_weights(dyn, eval_weights):
        dyn.eval()
        final_onestep = one_step_eval(
            dyn, onestep_batch["z"], onestep_batch["actions"],
            ctx=cfg.eval.ctx, K=cfg.model.K, k_max=cfg.model.k_max,
            tau_ctx=obj.tau_ctx, proprio=onestep_batch.get("proprio"))
        gif_pair = None
        for horizon in horizons:
            try:
                batch = sample_windows(val_cache, cfg.eval.episodes,
                                       cfg.eval.ctx + horizon, val_rng, device,
                                       start_only=True, with_video=True,
                                       replace=False)
            except ValueError:
                log.info(f"  [gate] horizon {horizon}: no long-enough episodes")
                continue
            for K in Ks:
                metrics, pair = gate_rollout(
                    dyn, tokenizer, dataset, batch, window=cfg.data.seq_len,
                    ctx=cfg.eval.ctx, K=K, k_max=cfg.model.k_max,
                    tau_ctx=obj.tau_ctx)
                rows.append({"horizon": horizon, "K": K,
                             "episodes": batch["z"].shape[0], **metrics})
                log.info(f"  [gate] H={horizon} K={K}: " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items()))
                if gif_pair is None:
                    full_gt = tokenizer.decode_latents(batch["z"][:6])
                    gif_pair = (full_gt.cpu(), pair[1][:6])
        if gif_pair is not None:
            save_demo_gif(gif_pair[0], gif_pair[1], out / "demo.gif",
                          ctx=cfg.eval.ctx)

    primary = rows[-1] if rows else {}           # largest horizon, last K
    minutes = (time.time() - t0) / 60.0
    final = {**primary, **final_onestep, "gate": rows, "steps": step,
             "params_M": n_params / 1e6, "minutes": minutes}
    (out / "final.json").write_text(json.dumps(final, indent=2))

    record = {"name": out.name, "tag": cfg.tag, "kind": "dynamics",
              "data": cfg.data.path, "tokenizer": cfg.tokenizer.ckpt,
              "objective": obj.objective, "seq_len": cfg.data.seq_len,
              "d_model": cfg.model.d_model, "depth": cfg.model.depth,
              "time_every": cfg.model.time_every, "k_max": cfg.model.k_max,
              "bootstrap_frac": obj.bootstrap_frac,
              "sched_sample_prob": obj.sched_sample_prob,
              "proprio": cfg.data.proprio, "steps": step,
              "minutes": round(minutes, 1),
              **{k: round(float(v), 5) for k, v in primary.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)},
              **({"gate_pass": primary["gate_pass"]}
                 if "gate_pass" in primary else {})}
    with open(out.parent / "experiments.jsonl", "a") as journal:
        journal.write(json.dumps(record) + "\n")

    writer.close()
    log.info("DONE " + json.dumps({k: v for k, v in final.items() if k != "gate"}))
    return final


def main(argv=None) -> None:
    """CLI entry point: resolve the config tree from ``argv`` and train."""
    cfg = parse_config(DynamicsTrainConfig, argv,
                       description=__doc__.split("\n\n")[0])
    if not cfg.data.path:
        raise SystemExit("--data.path is required (a dataset with actions)")
    train(cfg)


if __name__ == "__main__":
    main()
