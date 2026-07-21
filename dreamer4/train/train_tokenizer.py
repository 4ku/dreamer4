"""
Train the Dreamer 4 causal video tokenizer (phase-1 world-model pretraining).

Dataset-agnostic: anything :func:`dreamer4.data.open_video_dataset` can read —
gridworld shards or LeRobot datasets (one or several cameras, tiled into one
frame per timestep). The
objective is masked-autoencoding reconstruction (pixel L1/MSE + optional
LPIPS / DINOv3 perceptual terms, optionally RMS-normalized), with optional
extras that the production gridworld recipe relies on: weight EMA, latent
noise for decoder robustness, latent-cycle consistency, decoder-only
fine-tuning, and a proprio modality (robot joint-state analog).

Usage (defaults reproduce the production gridworld tokenizer):

    python -m dreamer4.train.train_tokenizer \\
        --data.path data/gridworld_10k --out runs/tok_x --steps 16000

    # robotics: a LeRobot dataset, two cameras tiled side by side
    python -m dreamer4.train.train_tokenizer \\
        --data.path /data/lerobot/pick_place --out runs/tok_robot \\
        --data.cameras observation.images.top,observation.images.wrist \\
        --model.patch_size 8

    # every flag mirrors a config field; YAML works too:
    python -m dreamer4.train.train_tokenizer --config cfg.yaml --optim.lr 1e-4

Artifacts under ``--out``: ``config.yaml``, ``tb/`` (TensorBoard),
``checkpoints/latest.pt`` (resumable: raw + EMA weights, optimizer, loss-norm
state), ``result_strip.png``, ``final.json``; one summary line is appended to
``<out>/../experiments.jsonl``. Resume an interrupted run with ``--resume
true`` and the same ``--out``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dreamer4.data import (ClipStreamDataset, EpisodeVideoDataset,
                           open_video_dataset, sample_val_clips,
                           split_train_val, video_batch_to_frames)
from dreamer4.train.perceptual import build_perceptual
from dreamer4.models.tokenizer import Decoder, Encoder, Tokenizer
from dreamer4.train.common import (EMA, LossCombiner, atomic_torch_save,
                                   set_seed, setup_run_logging,
                                   swap_in_weights, warmup_lr)
from dreamer4.train.config import (ModelConfig, TokenizerTrainConfig,
                                   config_from_dict, config_to_dict,
                                   parse_camera_list, parse_config,
                                   save_config)
from dreamer4.models.transformer.transformer import patchify, unpatchify

PROPRIO_SENTINEL = -2.0   # "sensor missing" input value used by proprio dropout


# ---------------------------------------------------------------------------
# Model building & checkpoints
# ---------------------------------------------------------------------------


def build_tokenizer(model: ModelConfig, *, frame_shape: Tuple[int, int, int],
                    d_proprio: Optional[int], max_T: int) -> Tokenizer:
    """Instantiate encoder+decoder for frames of (H, W, C)."""
    H, W, C = frame_shape
    p = model.patch_size
    if H % p or W % p:
        raise ValueError(f"patch_size={p} must divide the composed frame "
                         f"{H}x{W} (adjust model.patch_size or the camera layout)")
    n_patches = (H // p) * (W // p)
    patch_dim = p * p * C
    common = dict(d_model=model.d_model, n_latents=model.n_latents,
                  n_patches=n_patches, n_heads=model.n_heads,
                  depth=model.depth, n_kv_heads=model.n_kv_heads,
                  mlp_ratio=model.mlp_ratio, time_every=model.time_every,
                  dropout=0.0, use_qk_norm=True, logit_cap=model.logit_cap,
                  max_T=max_T, d_proprio=d_proprio)
    encoder = Encoder(patch_dim=patch_dim, d_bottleneck=model.d_bottleneck,
                      mae_p_min=model.mae_p_min, mae_p_max=model.mae_p_max,
                      **common)
    decoder = Decoder(d_bottleneck=model.d_bottleneck, patch_dim=patch_dim,
                      space_mode=model.decoder_mode, **common)
    return Tokenizer(encoder, decoder)


def weights_from_checkpoint(ckpt: dict, prefer_ema: bool) -> dict:
    if prefer_ema and ckpt.get("ema") is not None:
        return ckpt["ema"]
    return ckpt["model"]


def load_tokenizer_checkpoint(path, *, device: str | torch.device = "cpu",
                              prefer_ema: bool = True
                              ) -> Tuple[Tokenizer, dict]:
    """
    Rebuild a tokenizer from a ``train_tokenizer`` checkpoint.

    Returns ``(tokenizer.eval(), checkpoint_dict)``; the checkpoint carries
    ``config`` (full nested train config) and ``frame_meta`` — H/W/C,
    d_proprio and max_T (used to rebuild), plus n_patches/patch_dim/seq_len
    for provenance.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = config_from_dict(ModelConfig, ckpt["config"]["model"])
    meta = ckpt["frame_meta"]
    tok = build_tokenizer(model_cfg,
                          frame_shape=(meta["H"], meta["W"], meta["C"]),
                          d_proprio=meta["d_proprio"], max_T=meta["max_T"])
    tok.load_state_dict(weights_from_checkpoint(ckpt, prefer_ema))
    return tok.to(device).eval(), ckpt


def _save_checkpoint(path: Path, *, tok: Tokenizer, ema: Optional[EMA],
                     opt: torch.optim.Optimizer, combiner: LossCombiner,
                     step: int, cfg: TokenizerTrainConfig,
                     frame_meta: dict) -> None:
    atomic_torch_save(path, {
        "kind": "dreamer4-tokenizer",
        "step": step,
        "config": config_to_dict(cfg),
        "frame_meta": frame_meta,
        "model": tok.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "opt": opt.state_dict(),
        "loss_norm": combiner.state_dict(),
    })


# ---------------------------------------------------------------------------
# Objective pieces
# ---------------------------------------------------------------------------


def latent_noise_sigma(step: int, total_steps: int, *, noise_max: float,
                       warmup_frac: float) -> float:
    """
    Ceiling of the per-frame latent-noise magnitude at ``step``.

    Held at 0 for ``warmup_frac`` of training (let the tokenizer escape
    mean-collapse on clean reconstruction), then ramped linearly to
    ``noise_max``. Teaches the decoder to render imperfect (e.g.
    dynamics-drifted) latents; encode/eval are unaffected.
    """
    if noise_max <= 0.0:
        return 0.0
    hold = int(warmup_frac * total_steps)
    if step <= hold:
        return 0.0
    return noise_max * min(1.0, (step - hold) / max(1, total_steps - hold))


def latent_consistency_loss(tok: Tokenizer, pred_patches: torch.Tensor,
                            prop_pred: Optional[torch.Tensor],
                            z: torch.Tensor, *, encoder_frozen: bool
                            ) -> torch.Tensor:
    """
    Cycle term: re-encoding the reconstruction should give back ``z``.

    Gradients flow into the DECODER only (through ``pred_patches``); encoder
    weights are excluded, and MAE masking is off for the re-encode.
    """
    enc = tok.encoder
    toggled = []
    if not encoder_frozen:
        for p in enc.parameters():
            if p.requires_grad:
                toggled.append(p)
                p.requires_grad_(False)
    mae_was_training = enc.mae.training
    enc.mae.eval()
    try:
        z_recon, _ = enc(pred_patches, proprio=prop_pred)
        return F.mse_loss(z_recon.float(), z.detach().float())
    finally:
        enc.mae.train(mae_was_training)
        for p in toggled:
            p.requires_grad_(True)


def _set_train_mode(tok: Tokenizer, freeze_encoder: bool) -> None:
    tok.train()
    if freeze_encoder:
        tok.encoder.eval()     # keeps MAE masking off for the frozen encoder


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _frames_to_images(patches: torch.Tensor, H: int, W: int, C: int,
                      patch_size: int) -> torch.Tensor:
    return unpatchify(patches.float(), H, W, C, patch_size).clamp(0.0, 1.0)


def make_recon_strip(gt: torch.Tensor, pred: torch.Tensor, n: int = 8
                     ) -> torch.Tensor:
    """
    (N, T, C, H, W) gt/pred in [0,1] -> one (C, H', W') image: per clip, the
    ground-truth row over the reconstruction row, clips stacked vertically.
    """
    n = min(n, gt.shape[0])
    C = gt.shape[2]
    rows = []
    for i in range(n):
        gt_row = torch.cat(list(gt[i]), dim=-1)     # frames left-to-right
        pr_row = torch.cat(list(pred[i]), dim=-1)
        gap = torch.ones(C, 1, gt_row.shape[-1])
        rows += [gt_row, gap, pr_row, torch.ones(C, 3, gt_row.shape[-1])]
    return torch.cat(rows, dim=-2).clamp(0.0, 1.0)


@torch.no_grad()
def validate(tok: Tokenizer, dataset: EpisodeVideoDataset,
             clips: List[Tuple[int, int]], *, cfg: TokenizerTrainConfig,
             frame_shape: Tuple[int, int, int], use_proprio: bool,
             device: torch.device
             ) -> Tuple[Dict[str, float], torch.Tensor]:
    """
    Reconstruct the fixed validation clips (no MAE masking in eval mode).

    Returns (metrics, strip image). Metrics: pixel mse / l1 / psnr, proprio
    decode MSE when enabled, plus any dataset-specific metrics (e.g. the
    gridworld sprite-position errors) and ``gate_pass`` when the dataset
    defines a gate.
    """
    H, W, C = frame_shape
    tok.eval()
    sums: Dict[str, float] = {}
    count = 0
    strip = None
    for lo in range(0, len(clips), cfg.eval_batch):
        batch = clips[lo : lo + cfg.eval_batch]
        payloads = [dataset.clip(ep, start, cfg.data.seq_len)
                    for ep, start in batch]
        video = torch.from_numpy(np.stack([p["video"] for p in payloads]))
        frames = video_batch_to_frames(video, device)
        patches = patchify(frames, cfg.model.patch_size)
        proprio = None
        if use_proprio:
            proprio = torch.from_numpy(
                np.stack([p["proprio"] for p in payloads])).to(device)
        pred, _ = tok(patches, proprio=proprio)
        prop_pred = None
        if use_proprio:
            pred, prop_pred = pred
        gt_img = _frames_to_images(patches, H, W, C, cfg.model.patch_size)
        pr_img = _frames_to_images(pred, H, W, C, cfg.model.patch_size)

        n = len(batch)
        batch_metrics = {
            "mse": float(F.mse_loss(pr_img, gt_img)),
            "l1": float((pr_img - gt_img).abs().mean()),
        }
        if prop_pred is not None:
            batch_metrics["proprio_mse"] = float(
                F.mse_loss(prop_pred.float(), proprio.float()))
        gt_np = gt_img.cpu().numpy().transpose(0, 1, 3, 4, 2)
        pr_np = pr_img.cpu().numpy().transpose(0, 1, 3, 4, 2)
        batch_metrics.update(dataset.eval_metrics(gt_np, pr_np))
        for k, v in batch_metrics.items():
            sums[k] = sums.get(k, 0.0) + v * n
        count += n
        if strip is None:
            strip = make_recon_strip(gt_img.cpu(), pr_img.cpu())

    metrics = {k: v / count for k, v in sums.items()}
    metrics["psnr"] = float(-10.0 * np.log10(max(metrics["mse"], 1e-10)))
    gate = dataset.gate(metrics)
    if gate is not None:
        metrics["gate_pass"] = bool(gate)
    return metrics, strip


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: TokenizerTrainConfig) -> Dict[str, float]:
    """Run tokenizer training to completion; returns the final val metrics."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logging(out)
    save_config(cfg, out / "config.yaml")
    writer = SummaryWriter(log_dir=str(out / "tb"))

    # ---- data ------------------------------------------------------------
    cameras = parse_camera_list(cfg.data.cameras)
    dataset = open_video_dataset(
        cfg.data.path, proprio=cfg.data.proprio,
        camera_layout=cfg.data.camera_layout, cameras=cameras,
        episode_cache=cfg.data.episode_cache)
    if cfg.data.val_path:
        val_dataset = open_video_dataset(
            cfg.data.val_path, proprio=cfg.data.proprio,
            camera_layout=cfg.data.camera_layout, cameras=cameras,
            episode_cache=cfg.data.episode_cache)
        train_eps, val_eps = None, None       # train on all, validate on all
    else:
        val_dataset = dataset
        train_eps, val_eps = split_train_val(
            len(dataset), val_frac=cfg.data.val_frac, seed=cfg.seed)

    frame_shape = dataset.frame_shape
    H, W, C = frame_shape
    d_proprio = dataset.proprio_dim
    use_proprio = d_proprio is not None
    val_clips = sample_val_clips(val_dataset, seq_len=cfg.data.seq_len,
                                 n_clips=cfg.val_clips, seed=cfg.seed + 1,
                                 episodes=val_eps)
    loader = DataLoader(
        ClipStreamDataset(dataset, seq_len=cfg.data.seq_len, seed=cfg.seed,
                          episodes=train_eps),
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False)
    cameras_info = (f" cameras={dataset.camera_names}"
                    if hasattr(dataset, "camera_names") else "")
    log.info(f"device={device} data={cfg.data.path} episodes={len(dataset)} "
             f"(val clips {len(val_clips)}"
             f"{' from ' + cfg.data.val_path if cfg.data.val_path else ''}) | "
             f"frame {H}x{W}x{C}{cameras_info} proprio={d_proprio}")

    # ---- model / optimizer ----------------------------------------------
    tok = build_tokenizer(cfg.model, frame_shape=frame_shape,
                          d_proprio=d_proprio,
                          max_T=cfg.data.seq_len + 8).to(device)
    if cfg.init_from:
        init = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        tok.load_state_dict(weights_from_checkpoint(init, prefer_ema=True))
        log.info(f"initialized from {cfg.init_from} (step {init.get('step', '?')})")
    if cfg.freeze_encoder:
        if not cfg.init_from and not cfg.resume:
            raise ValueError("--freeze_encoder only makes sense with "
                             "--init_from (or --resume)")
        for p in tok.encoder.parameters():
            p.requires_grad_(False)
        log.info("encoder FROZEN — decoder-only fine-tune, latent space unchanged")

    trainable = [p for p in tok.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in tok.parameters())
    log.info(f"tokenizer params: {n_params / 1e6:.2f}M "
             f"(trainable {sum(p.numel() for p in trainable) / 1e6:.2f}M)")
    opt = torch.optim.AdamW(trainable, lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2),
                            weight_decay=cfg.optim.weight_decay)
    ema = EMA(tok, cfg.optim.ema_decay) if cfg.optim.ema_decay > 0 else None

    perceptual = build_perceptual(
        backbone=cfg.loss.perceptual_backbone, weight=cfg.loss.perceptual_weight,
        up=cfg.loss.perceptual_up, lpips_net=cfg.loss.lpips_net,
        dino_weight=cfg.loss.dino_weight, device=device)
    weights = {"recon": cfg.loss.recon_weight}
    weights.update({name: w for name, _, w in perceptual})
    weights["consistency"] = cfg.loss.consistency_weight
    weights["proprio"] = cfg.loss.proprio_weight if use_proprio else 0.0
    combiner = LossCombiner(weights, normalize=cfg.loss.loss_norm,
                            decay=cfg.loss.loss_norm_decay,
                            floor_frac=cfg.loss.loss_norm_floor, device=device)
    log.info("loss: " + "  ".join(f"{k}={v}" for k, v in weights.items())
             + (f"  [RMS-normalized, floor {cfg.loss.loss_norm_floor}]"
                if cfg.loss.loss_norm else ""))

    frame_meta = {"H": H, "W": W, "C": C,
                  "n_patches": (H // cfg.model.patch_size) * (W // cfg.model.patch_size),
                  "patch_dim": cfg.model.patch_size ** 2 * C,
                  "d_proprio": d_proprio, "max_T": cfg.data.seq_len + 8,
                  "seq_len": cfg.data.seq_len}

    start_step = 0
    latest = ckpt_dir / "latest.pt"
    if cfg.resume and latest.exists():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        tok.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if ema is not None:
            if ckpt.get("ema") is not None:
                ema.load_state_dict(ckpt["ema"])
            else:                       # prior run had EMA off: reseed shadow
                ema = EMA(tok, cfg.optim.ema_decay)
        combiner.load_state_dict(ckpt.get("loss_norm"))
        start_step = int(ckpt["step"])
        log.info(f"resumed from {latest} at step {start_step}")

    # ---- loop ------------------------------------------------------------
    def run_validation(step: int) -> None:
        eval_weights = ema.state_dict() if ema is not None else None
        with swap_in_weights(tok, eval_weights):
            metrics, strip = validate(
                tok, val_dataset, val_clips, cfg=cfg, frame_shape=frame_shape,
                use_proprio=use_proprio, device=device)
        _set_train_mode(tok, cfg.freeze_encoder)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", float(v), step)
        writer.add_image("val/recon_gt_vs_pred", strip, step)
        log.info("  [val] step %d: %s" % (step, "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items())))

    _set_train_mode(tok, cfg.freeze_encoder)
    data_iter = iter(loader)
    t0 = time.time()
    budget_s = cfg.max_minutes * 60.0
    recon_ema: Optional[float] = None
    step = start_step

    for step in range(start_step + 1, cfg.steps + 1):
        if budget_s and (time.time() - t0) >= budget_s:
            log.info(f"[budget] {cfg.max_minutes} min reached at step {step - 1}")
            step -= 1
            break
        lr = warmup_lr(cfg.optim.lr, step, cfg.optim.warmup)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        raw_means: Dict[str, float] = {}
        mask_frac = 0.0
        for _ in range(cfg.optim.accum_steps):
            batch = next(data_iter)
            frames = video_batch_to_frames(batch["video"], device)
            patches = patchify(frames, cfg.model.patch_size)
            proprio_target = proprio_input = None
            if use_proprio:
                proprio_target = batch["proprio"].to(device, non_blocking=True)
                proprio_input = proprio_target
                if cfg.proprio_dropout > 0.0:
                    # missing-sensor robustness: sentinel input, TRUE target
                    drop = (torch.rand(proprio_target.shape[0], device=device)
                            < cfg.proprio_dropout)
                    proprio_input = torch.where(
                        drop[:, None, None],
                        torch.full_like(proprio_target, PROPRIO_SENTINEL),
                        proprio_target)

            amp_on = cfg.optim.amp
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp_on):
                z, mae_mask = tok.encoder(patches, proprio=proprio_input)
                sigma_max = latent_noise_sigma(
                    step, cfg.steps, noise_max=cfg.latent_noise_max,
                    warmup_frac=cfg.latent_noise_warmup_frac)
                z_dec = z
                if sigma_max > 0.0:
                    sigma = torch.empty(z.shape[0], z.shape[1], 1, 1,
                                        device=z.device).uniform_(0.0, sigma_max)
                    z_dec = (z + sigma * torch.randn_like(z)).clamp(-1.0, 1.0)
                dec_out = tok.decoder(z_dec)
                prop_pred = None
                if use_proprio:
                    pred, prop_pred = dec_out
                else:
                    pred = dec_out

                if cfg.loss.recon == "l1":
                    loss_recon = (pred.float() - patches.float()).abs().mean()
                else:
                    loss_recon = F.mse_loss(pred.float(), patches.float())
                loss_prop = None
                if use_proprio:
                    loss_prop = F.mse_loss(prop_pred.float(), proprio_target.float())
                loss_cons = None
                if cfg.loss.consistency_weight > 0.0:
                    loss_cons = latent_consistency_loss(
                        tok, pred, prop_pred, z, encoder_frozen=cfg.freeze_encoder)

            terms: Dict[str, Optional[torch.Tensor]] = {
                "recon": loss_recon, "consistency": loss_cons, "proprio": loss_prop}
            if perceptual:                     # fp32, outside autocast
                pr_img = _frames_to_images(pred, H, W, C, cfg.model.patch_size)
                gt_img = _frames_to_images(patches, H, W, C, cfg.model.patch_size)
                pr_flat = pr_img.reshape(-1, C, H, W)
                gt_flat = gt_img.reshape(-1, C, H, W)
                for name, module, _ in perceptual:
                    terms[name] = module(pr_flat, gt_flat)

            loss, raw = combiner(terms)
            (loss / cfg.optim.accum_steps).backward()
            for k, v in raw.items():
                raw_means[k] = raw_means.get(k, 0.0) + v / cfg.optim.accum_steps
            mask_frac += float(mae_mask.float().mean()) / cfg.optim.accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(tok.parameters(),
                                                   cfg.optim.grad_clip)
        opt.step()
        if ema is not None:
            ema.update(tok)

        recon_val = raw_means.get("recon", 0.0)
        recon_ema = recon_val if recon_ema is None else \
            0.98 * recon_ema + 0.02 * recon_val
        if step % cfg.log_every == 0:
            sps = (step - start_step) / max(time.time() - t0, 1e-9)
            for k, v in raw_means.items():
                writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("train/recon_ema", recon_ema, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("train/mask_frac", mask_frac, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/steps_per_sec", sps, step)
            log.info(f"step {step}/{cfg.steps} "
                     + " ".join(f"{k}={v:.4f}" for k, v in raw_means.items())
                     + f" recon_ema={recon_ema:.4f} gnorm={float(grad_norm):.2f}"
                     f" {sps:.1f}it/s")

        if step % cfg.val_every == 0:
            run_validation(step)
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            _save_checkpoint(latest, tok=tok, ema=ema, opt=opt,
                             combiner=combiner, step=step, cfg=cfg,
                             frame_meta=frame_meta)

    # ---- final artifacts -------------------------------------------------
    # Save first (raw + EMA weights stay separate — the run must remain
    # resumable), then evaluate/render with the EMA weights swapped in.
    # Deployment consumers get the EMA weights via load_tokenizer_checkpoint.
    _save_checkpoint(latest, tok=tok, ema=ema, opt=opt, combiner=combiner,
                     step=step, cfg=cfg, frame_meta=frame_meta)
    eval_weights = ema.state_dict() if ema is not None else None
    with swap_in_weights(tok, eval_weights):
        metrics, strip = validate(tok, val_dataset, val_clips, cfg=cfg,
                                  frame_shape=frame_shape,
                                  use_proprio=use_proprio, device=device)

    import imageio.v2 as iio
    strip_u8 = (strip.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    iio.imwrite(out / "result_strip.png", strip_u8)

    minutes = (time.time() - t0) / 60.0
    final = {**metrics, "steps": step, "params_M": n_params / 1e6,
             "minutes": minutes}
    (out / "final.json").write_text(json.dumps(final, indent=2))

    record = {"name": out.name, "tag": cfg.tag, "kind": "tokenizer",
              "data": cfg.data.path, "seq_len": cfg.data.seq_len,
              "patch_size": cfg.model.patch_size, "d_model": cfg.model.d_model,
              "depth": cfg.model.depth, "n_latents": cfg.model.n_latents,
              "d_bottleneck": cfg.model.d_bottleneck,
              "decoder_mode": cfg.model.decoder_mode,
              "perceptual": cfg.loss.perceptual_backbone,
              "perceptual_weight": cfg.loss.perceptual_weight,
              "loss_norm": cfg.loss.loss_norm, "proprio": cfg.data.proprio,
              "steps": step, "minutes": round(minutes, 1),
              **{k: round(float(v), 5) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)},
              **({"gate_pass": metrics["gate_pass"]}
                 if "gate_pass" in metrics else {})}
    with open(out.parent / "experiments.jsonl", "a") as journal:
        journal.write(json.dumps(record) + "\n")

    writer.close()
    log.info("DONE " + json.dumps(final))
    return final


def main(argv=None) -> None:
    cfg = parse_config(TokenizerTrainConfig, argv,
                       description=__doc__.split("\n\n")[0])
    if not cfg.data.path:
        raise SystemExit("--data.path is required (dataset directory; "
                         "see dreamer4.data for supported formats)")
    train(cfg)


if __name__ == "__main__":
    main()
