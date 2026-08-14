"""
Train the Dreamer 4 dynamics model (world model) in a frozen tokenizer's
latent space — phase-1 world-model pretraining, the action-conditioned half.

Dataset-agnostic: anything :func:`dreamer4.data.open_video_dataset` can read
that carries ACTIONS — gridworld shards (discrete moves as one-hot) or
LeRobot datasets (parquet action vectors; ``observation.state`` becomes the
joint proprio stream via ``--data.proprio auto``). Episodes are encoded ONCE
by the frozen tokenizer (sliding ``tokenizer.history`` window — the temporal
receptive field stays fixed while the world model rolls any horizon) and
cached; training samples latent windows from the cache.

The objective is the production recipe (see
:mod:`dreamer4.train.dynamics_objectives`); validation is the WINDOWED
open-loop rollout (:mod:`dreamer4.dynamics_eval`) — the deployed dream
loop, scored by the dataset's own domain hooks (gridworld: sprite errors +
the ≤1-cell gate).

Usage (defaults = production recipe STEP 3):

    python -m dreamer4.train.train_dynamics \\
        --data.path data/gridworld_10k,data/gridworld_sticky3k \\
        --data.proprio player \\
        --tokenizer.ckpt runs/tok_x/checkpoints/latest.pt \\
        --out runs/dyn_x --steps 24000

    # STEP 4, the K=1-maker (bootstrap-later fine-tune):
    ... --init_from runs/dyn_x/checkpoints/latest.pt \\
        --objective.bootstrap_frac 0.5 --optim.lr 5e-5 --optim.warmup 200 \\
        --steps 36000 --objective.sched_warmup_frac 0.3 --out runs/dyn_x_boot

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
                                   config_from_dict, config_to_dict,
                                   parse_camera_list, parse_config,
                                   save_config)
from dreamer4.train.dynamics_objectives import (clean_context_loss,
                                                shortcut_forcing_loss)
from dreamer4.train.train_tokenizer import (weights_from_checkpoint,
                                            make_recon_strip)

OBJECTIVES = ("clean_context", "shortcut_forcing")


# ---------------------------------------------------------------------------
# Model building & checkpoints
# ---------------------------------------------------------------------------


def build_dynamics(model: DynamicsModelConfig, *, n_spatial: int,
                   d_spatial: int, action_dim: int,
                   d_proprio: Optional[int], max_T: int, n_agent: int = 0,
                   space_mode: str = "wm_agent") -> DynamicsModel:
    """``action_dim`` counts the ALIGNED vector (dataset dim + start flag).

    Phase-1 pretraining uses the default ``n_agent=0``: with no AGENT segment
    the ``wm_agent`` mask degenerates to full world-to-world mixing. The agent
    trainer rebuilds the same checkpointed architecture with ``n_agent > 0`` —
    world-token outputs are unchanged by construction (world rows never attend
    AGENT columns), which the no-op test asserts.
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

    Returns ``(model.eval(), checkpoint_dict)``; the checkpoint carries the
    full nested ``config``, ``latent_meta`` (n_spatial/d_spatial/action_dim/
    d_proprio/max_T/window) and ``tokenizer_ckpt`` (path of the codec it was
    trained against).
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

    Each entry: {"z": (Ti,Nz,Dz) fp16, "actions": (Ti-1,Da) float32,
    "proprio"?: (Ti,d_p) float32, "video"?: (Ti,H,W,C) uint8,
    "rewards"?/"terminals"?: (Ti-1,) when the dataset records them,
    "meta"?: collector metadata dict (see ``episode_meta``),
    "bc_weight": how much to imitate this episode (see ``bc_weight``)}. Long episodes
    are encoded in chunks with ``history-1`` frames of left context, which
    yields latents identical to a single pass (the sliding window sees at
    most ``history`` frames back).
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
                   start_only: bool = False, with_video: bool = False,
                   replace: bool = True) -> Dict[str, torch.Tensor]:
    """
    B random length-T windows from the cache (episodes chosen uniformly).

    Returns {"z": (B,T,Nz,Dz) float32 on device, "actions": (B,T-1,Da),
    "proprio"?: (B,T,d_p), "video"?: (B,T,H,W,C) uint8 cpu}.
    ``start_only`` pins windows to frame 0 (the episode-start distribution).
    """
    eligible = [e for e in episodes if e["z"].shape[0] >= T]
    if not eligible:
        raise ValueError(f"no episode has >= {T} frames")
    if not replace and len(eligible) < B:
        B = len(eligible)
    picks = (rng.choice(len(eligible), size=B, replace=replace)
             if replace else rng.permutation(len(eligible))[:B])
    zs, acts, props, vids = [], [], [], []
    for j in picks:
        e = eligible[int(j)]
        n = e["z"].shape[0]
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
    """Run dynamics training to completion; returns the final gate metrics."""
    if cfg.objective.objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logging(out)
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
    use_boot = clean_objective and obj.bootstrap_frac > 0.0
    weights = {"flow": 1.0,
               "proprio": obj.proprio_weight if d_proprio is not None else 0.0,
               "bootstrap": obj.boot_weight if use_boot else 0.0}
    # Normalize only when the clean-context objective has >1 active term (a
    # lone normalized flow term would just rescale the LR; the paper-shortcut
    # objective folds its bootstrap internally and was never normalized).
    combiner = LossCombiner(weights, normalize=(d_proprio is not None or use_boot),
                            decay=0.99, floor_frac=0.2, device=device)
    log.info("objective: " + obj.objective + "  "
             + "  ".join(f"{k}={v}" for k, v in weights.items() if v)
             + (f"  [RMS-normalized]" if combiner.normalizer else ""))
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
    step = start_step

    for step in range(start_step + 1, cfg.steps + 1):
        if budget_s and (time.time() - t0) >= budget_s:
            log.info(f"[budget] {cfg.max_minutes} min reached at step {step - 1}")
            step -= 1
            break
        lr = warmup_lr(cfg.optim.lr, step, cfg.optim.warmup)
        for group in opt.param_groups:
            group["lr"] = lr

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
                start_only=image_batch and obj.image_batch_frame0)
            aligned = align_actions(batch["actions"], T_s)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=cfg.optim.amp):
                if not clean:                   # the paper objective, eq 4/7
                    loss_flow, sf_aux = shortcut_forcing_loss(
                        dyn, z1=batch["z"], actions=aligned,
                        k_max=cfg.model.k_max,
                        bootstrap_fraction=obj.bootstrap_frac)
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
                        bootstrap_frac=obj.bootstrap_frac)
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
        if step % cfg.log_every == 0:
            sps = (step - start_step) / max(time.time() - t0, 1e-9)
            for k, v in raw.items():
                writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("train/loss_ema", loss_ema, step)
            writer.add_scalar("train/flow_mse", aux["flow_mse"], step)
            writer.add_scalar("train/boot_mse", aux.get("boot_mse", 0.0), step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("train/sched_prob", sched_p, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/steps_per_sec", sps, step)
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
    cfg = parse_config(DynamicsTrainConfig, argv,
                       description=__doc__.split("\n\n")[0])
    if not cfg.data.path:
        raise SystemExit("--data.path is required (a dataset with actions)")
    train(cfg)


if __name__ == "__main__":
    main()
