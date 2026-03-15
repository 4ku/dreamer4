#!/usr/bin/env python3
"""
Dreamer 4 — Full online training on DMControl.

Pretrains the tokenizer on random data, then launches the online
collect-train-imagine loop with TensorBoard logging and checkpointing.

Usage:
    cd /mnt/dsk1/dreamer4
    source .venv/bin/activate
    MUJOCO_GL=egl python scripts/train_dmcontrol.py

    # Custom config / resume
    python scripts/train_dmcontrol.py \\
        --config config/dmcontrol.yaml \\
        --logdir runs/cartpole_01 \\
        --device cuda \\
        --resume runs/cartpole_01/checkpoints/final.pt

Monitor with TensorBoard:
    tensorboard --logdir runs/
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

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
from dreamer4.train import Dreamer4Agent, online_training_loop
from dreamer4.utils import patchify

logger = logging.getLogger(__name__)


def make_tokenizer(cfg: TrainConfig) -> Tokenizer:
    """Build a tokenizer from config."""
    img_size = cfg.image_size
    patch_size = cfg.patch_size
    n_patches = (img_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * cfg.channels

    enc = Encoder(
        patch_dim=patch_dim,
        d_model=128,
        n_latents=cfg.n_bottleneck,
        n_patches=n_patches,
        n_heads=4,
        depth=4,
        d_bottleneck=cfg.d_bottleneck,
        time_every=1,
        mae_p_min=0.0,
        mae_p_max=0.9,
        use_qk_norm=False,
        logit_cap=None,
    )
    dec = Decoder(
        d_bottleneck=cfg.d_bottleneck,
        d_model=128,
        n_latents=cfg.n_bottleneck,
        n_patches=n_patches,
        patch_dim=patch_dim,
        n_heads=4,
        depth=4,
        time_every=1,
        use_qk_norm=False,
        logit_cap=None,
    )
    return Tokenizer(enc, dec)


def pretrain_tokenizer(
    cfg: TrainConfig,
    tok: Tokenizer,
    device: torch.device,
    steps: int = 2000,
    prefill_steps: int = 5000,
) -> Tokenizer:
    """Pretrain tokenizer on random episodes (MSE + 0.2*LPIPS)."""
    logger.info("Collecting random data for tokenizer pretraining...")

    def make_env():
        return TimeLimitWrapper(
            DMControlEnv(
                cfg.domain, cfg.task,
                size=cfg.image_size,
                action_repeat=cfg.action_repeat,
            ),
            max_steps=cfg.time_limit,
        )

    driver = Driver([make_env])
    episodes, _ = driver.collect_random(prefill_steps)
    driver.close()

    buf = ReplayBuffer(capacity=200_000)
    for ep in episodes:
        buf.add_episode(ep.obs, ep.actions, ep.rewards)

    logger.info(
        "Collected %d episodes, %d transitions for tokenizer pretraining",
        buf.total_episodes, buf.total_transitions,
    )

    tok = tok.to(device)
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
        use_lpips = True
    except ImportError:
        logger.warning("lpips not available, using MSE only for tokenizer")
        lpips_fn = None
        use_lpips = False

    loss_norm = RMSLossNormalizer(n_losses=2).to(device)
    optimizer = torch.optim.Adam(tok.parameters(), lr=cfg.tokenizer_lr)

    batch_size = min(cfg.batch_size, 16)
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

        if step % 200 == 0 or step == 1:
            logger.info("  tokenizer step %4d/%d  loss=%.5f", step, steps, loss.item())

    tok.eval()
    logger.info("Tokenizer pretraining complete (final loss=%.5f)", loss.item())
    return tok


def make_env_fn(cfg: TrainConfig, seed: int = 0):
    """Return a factory callable for creating wrapped environments."""
    def _make():
        return TimeLimitWrapper(
            DMControlEnv(
                cfg.domain, cfg.task,
                size=cfg.image_size,
                action_repeat=cfg.action_repeat,
                seed=seed,
            ),
            max_steps=cfg.time_limit,
        )
    return _make


def main():
    parser = argparse.ArgumentParser(description="Dreamer 4 — DMControl Training")
    parser.add_argument(
        "--config", type=str, default="config/dmcontrol.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--logdir", type=str, default=None,
        help="TensorBoard + checkpoint directory (default: runs/<domain>_<task>)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--tok-pretrain-steps", type=int, default=2000,
        help="Tokenizer pretraining steps",
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

    base_logdir = Path(args.logdir or f"runs/{cfg.domain}_{cfg.task}")
    if args.resume:
        logdir = base_logdir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = base_logdir / stamp
    log_dir = str(logdir / "logs")
    ckpt_dir = str(logdir / "checkpoints")

    logger.info("=" * 60)
    logger.info("Dreamer 4 — DMControl Training")
    logger.info("  Domain: %s, Task: %s", cfg.domain, cfg.task)
    logger.info("  Image size: %d, Action repeat: %d", cfg.image_size, cfg.action_repeat)
    logger.info("  Total steps: %d, Prefill: %d", cfg.total_steps, cfg.prefill_steps)
    logger.info("  N envs: %d, Batch size: %d", cfg.n_envs, cfg.batch_size)
    logger.info("  Device: %s", cfg.device)
    logger.info("  Log dir: %s", log_dir)
    logger.info("  Checkpoint dir: %s", ckpt_dir)
    logger.info("=" * 60)

    save_config(cfg, Path(logdir) / "config.yaml")

    device = torch.device(cfg.device)

    tok = make_tokenizer(cfg)
    tok = pretrain_tokenizer(
        cfg, tok, device,
        steps=args.tok_pretrain_steps,
        prefill_steps=cfg.prefill_steps,
    )

    agent = Dreamer4Agent(cfg)

    env_fns = [make_env_fn(cfg, seed=i) for i in range(cfg.n_envs)]

    history = online_training_loop(
        agent,
        env_fns,
        tokenizer=tok,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        resume_from=args.resume,
    )

    logger.info("Training complete! Logs at: %s", log_dir)
    logger.info("View with: tensorboard --logdir %s", logdir)


if __name__ == "__main__":
    main()
