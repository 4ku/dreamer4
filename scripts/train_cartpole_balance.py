#!/usr/bin/env python3
"""
Dreamer 4 — Lightweight cartpole-balance training + evaluation.

Trains a tiny model on cartpole-balance for quick end-to-end validation,
with periodic eval episodes, TensorBoard video logging, and post-training
video/plot export.

Usage:
    cd /mnt/dsk1/dreamer4
    source .venv/bin/activate
    MUJOCO_GL=egl python scripts/train_cartpole_balance.py

    # Resume from checkpoint
    MUJOCO_GL=egl python scripts/train_cartpole_balance.py \
        --resume runs/cartpole_balance/checkpoints/final.pt

Monitor:
    tensorboard --logdir runs/cartpole_balance
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreamer4.config import TrainConfig, load_config, save_config
from dreamer4.driver import Driver
from dreamer4.envs import DMControlEnv, TimeLimitWrapper
from dreamer4.replay import ReplayBuffer
from dreamer4.tokenizer import (
    Decoder,
    Encoder,
    RMSLossNormalizer,
    Tokenizer,
    lpips_on_mae_recon,
    recon_loss_from_mae,
)
from dreamer4.train import Dreamer4Agent, evaluate, online_training_loop
from dreamer4.utils import patchify

logger = logging.getLogger(__name__)


def make_tokenizer(cfg: TrainConfig) -> Tokenizer:
    """Build a small tokenizer matching the light config."""
    img_size = cfg.image_size
    patch_size = cfg.patch_size
    n_patches = (img_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * cfg.channels

    tok_d_model = 128
    tok_depth = 4
    tok_heads = 4

    enc = Encoder(
        patch_dim=patch_dim,
        d_model=tok_d_model,
        n_latents=cfg.n_bottleneck,
        n_patches=n_patches,
        n_heads=tok_heads,
        depth=tok_depth,
        d_bottleneck=cfg.d_bottleneck,
        time_every=1,
        mae_p_min=0.0,
        mae_p_max=0.9,
        use_qk_norm=False,
        logit_cap=None,
    )
    dec = Decoder(
        d_bottleneck=cfg.d_bottleneck,
        d_model=tok_d_model,
        n_latents=cfg.n_bottleneck,
        n_patches=n_patches,
        patch_dim=patch_dim,
        n_heads=tok_heads,
        depth=tok_depth,
        time_every=1,
        use_qk_norm=False,
        logit_cap=None,
    )
    return Tokenizer(enc, dec)


def make_env(cfg: TrainConfig, seed: int = 0):
    """Create a wrapped environment."""
    return TimeLimitWrapper(
        DMControlEnv(
            cfg.domain, cfg.task,
            size=cfg.image_size,
            action_repeat=cfg.action_repeat,
            seed=seed,
        ),
        max_steps=cfg.time_limit,
    )


def pretrain_tokenizer(
    cfg: TrainConfig,
    tok: Tokenizer,
    device: torch.device,
    steps: int = 500,
) -> Tokenizer:
    """Pretrain tokenizer on random episodes (MSE + LPIPS)."""
    logger.info("Collecting random data for tokenizer pretraining...")

    driver = Driver([lambda: make_env(cfg)])
    episodes, _ = driver.collect_random(cfg.prefill_steps)
    driver.close()

    buf = ReplayBuffer(capacity=200_000)
    for ep in episodes:
        buf.add_episode(ep.obs, ep.actions, ep.rewards)

    logger.info(
        "Collected %d episodes, %d transitions",
        buf.total_episodes, buf.total_transitions,
    )

    tok = tok.to(device)
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
        use_lpips = True
    except ImportError:
        logger.warning("lpips not available, using MSE only")
        lpips_fn = None
        use_lpips = False

    loss_norm = RMSLossNormalizer(n_losses=2).to(device)
    optimizer = torch.optim.Adam(tok.parameters(), lr=cfg.tokenizer_lr)

    batch_size = min(cfg.batch_size, 8)
    seq_len = min(4, cfg.seq_len)

    logger.info("Pretraining tokenizer for %d steps...", steps)
    tok.train()
    for step in range(1, steps + 1):
        batch = buf.sample_sequence(batch_size, seq_len=seq_len, device=device)
        frames = batch["obs"]
        patches = patchify(frames, cfg.patch_size)

        pred, mae_mask, _ = tok(patches)
        mse = recon_loss_from_mae(pred, patches, mae_mask)

        loss_norm.update(0, mse)
        loss = loss_norm.normalize(0, mse)

        if use_lpips:
            lp = lpips_on_mae_recon(
                lpips_fn, pred, patches, mae_mask,
                H=cfg.image_size, W=cfg.image_size,
                C=cfg.channels, patch_size=cfg.patch_size,
            )
            loss_norm.update(1, lp)
            loss = loss + 0.2 * loss_norm.normalize(1, lp)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tok.parameters(), 100.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            logger.info("  tok step %4d/%d  loss=%.5f", step, steps, loss.item())

    tok.eval()
    logger.info("Tokenizer pretraining done (loss=%.5f)", loss.item())
    return tok


def save_eval_videos(
    eval_result: dict,
    output_dir: Path,
    prefix: str = "eval",
    fps: int = 15,
) -> None:
    """Save evaluation episode videos as MP4 files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_list = eval_result.get("eval_frames", [])
    for i, frames in enumerate(frames_list):
        if not frames:
            continue
        path = output_dir / f"{prefix}_ep{i:02d}.mp4"
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        logger.info("Saved video: %s (%d frames)", path, len(frames))


def main():
    parser = argparse.ArgumentParser(
        description="Dreamer 4 — Cartpole Balance (lightweight)",
    )
    parser.add_argument(
        "--config", type=str,
        default="config/cartpole_balance_light.yaml",
    )
    parser.add_argument("--logdir", type=str, default="runs/cartpole_balance")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tok-steps", type=int, default=2000)
    parser.add_argument(
        "--no-dp", action="store_true",
        help="Disable DataParallel even if multiple GPUs available",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {}
    if args.device:
        overrides["device"] = args.device

    cfg = load_config(args.config, overrides=overrides)

    base_logdir = Path(args.logdir)
    if args.resume:
        logdir = base_logdir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = base_logdir / stamp
    log_dir = str(logdir / "logs")
    ckpt_dir = str(logdir / "checkpoints")
    video_dir = logdir / "videos"

    n_gpus = torch.cuda.device_count()
    use_dp = n_gpus > 1 and not args.no_dp

    logger.info("=" * 60)
    logger.info("Dreamer 4 — Cartpole Balance (lightweight)")
    logger.info("  Domain: %s, Task: %s", cfg.domain, cfg.task)
    logger.info("  Model: d_model=%d, depth=%d, heads=%d", cfg.d_model, cfg.depth, cfg.n_heads)
    logger.info("  Total steps: %d, Prefill: %d", cfg.total_steps, cfg.prefill_steps)
    logger.info("  N envs: %d, Batch: %d, Seq: %d", cfg.n_envs, cfg.batch_size, cfg.seq_len)
    logger.info("  GPUs: %d, DataParallel: %s", n_gpus, use_dp)
    logger.info("  Device: %s", cfg.device)
    logger.info("  Log dir: %s", log_dir)
    logger.info("=" * 60)

    save_config(cfg, logdir / "config.yaml")

    device = torch.device(cfg.device)

    t_start = time.time()

    # -- Tokenizer pretraining --
    tok = make_tokenizer(cfg)
    tok = pretrain_tokenizer(cfg, tok, device, steps=args.tok_steps)

    # -- Build agent --
    agent = Dreamer4Agent(cfg)

    # -- Environment factories --
    env_fns = [
        (lambda s=seed: make_env(cfg, seed=s))
        for seed in range(cfg.n_envs)
    ]
    eval_env_fn = lambda: make_env(cfg, seed=999)

    # -- Train --
    history = online_training_loop(
        agent,
        env_fns,
        tokenizer=tok,
        eval_env_fn=eval_env_fn,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        resume_from=args.resume,
        use_data_parallel=use_dp,
    )

    elapsed = time.time() - t_start

    # -- Final evaluation + video saving --
    logger.info("Running final evaluation (10 episodes)...")
    final_eval = evaluate(
        agent, eval_env_fn,
        n_episodes=10,
        max_steps=cfg.time_limit,
        record_video=True,
        render_size=256,
    )

    save_eval_videos(final_eval, video_dir, prefix="final", fps=15)

    # -- Summary --
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Total time: %.1f min", elapsed / 60)
    logger.info(
        "  Final eval return: %.2f +/- %.2f",
        final_eval["eval_return_mean"],
        final_eval["eval_return_std"],
    )
    logger.info(
        "  Final eval range: [%.2f, %.2f]",
        final_eval["eval_return_min"],
        final_eval["eval_return_max"],
    )
    logger.info("  Videos saved to: %s", video_dir)
    logger.info("  TensorBoard: tensorboard --logdir %s", logdir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
