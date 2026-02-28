#!/usr/bin/env python3
"""
Dreamer 4 — Interactive keyboard control with world-model visualization.

Load a trained checkpoint and control the cartpole in real time.
A pygame window shows:
  - Left panel:  actual environment frame
  - Right panel: world model's predicted next frame
  - Bottom bar:  predicted reward, value, mode, episode return

Controls:
  Left arrow / A : push cart left  (-1.0)
  Right arrow / D: push cart right (+1.0)
  No key         : coast (0.0)
  P              : toggle keyboard <-> learned policy
  R              : reset episode
  Q / ESC        : quit

Usage:
    cd /mnt/dsk1/dreamer4
    source .venv/bin/activate
    python scripts/play_keyboard.py --checkpoint runs/cartpole_swingup/checkpoints/final.pt

    # Without a checkpoint (random policy only, still shows WM predictions
    # if --no-checkpoint is passed):
    python scripts/play_keyboard.py --no-checkpoint
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreamer4.checkpoint import load_checkpoint
from dreamer4.config import TrainConfig, load_config
from dreamer4.dynamics import corrupt_representations, sample_one_timestep
from dreamer4.envs import DMControlEnv
from dreamer4.tokenizer import Decoder, Encoder, Tokenizer
from dreamer4.train import Dreamer4Agent
from dreamer4.utils import (
    pack_bottleneck_to_spatial,
    patchify,
    unpack_spatial_to_bottleneck,
    unpatchify,
)

logger = logging.getLogger(__name__)

DISPLAY_SCALE = 4
FPS = 30
PANEL_GAP = 4
BAR_HEIGHT = 48
FONT_SIZE = 18


def make_tokenizer(cfg: TrainConfig) -> Tokenizer:
    """Build a tokenizer from config (same architecture as training)."""
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


@torch.no_grad()
def predict_next_frame(
    agent: Dreamer4Agent,
    obs_history: list[torch.Tensor],
    act_history: list[torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """Run world model to predict next frame, reward, and value.

    Returns (predicted_rgb_uint8, predicted_reward, predicted_value).
    """
    max_ctx = cfg.policy_max_context
    obs_list = obs_history[-max_ctx:]
    act_list = act_history[-(max_ctx):]

    obs_t = torch.stack(obs_list).unsqueeze(0).to(device)  # (1, T, C, H, W)
    z_packed = agent.encode_obs(obs_t)

    T_cur = z_packed.shape[1]
    task_ids = torch.zeros(1, dtype=torch.long, device=device)
    agent_tok = agent.task_encoder(task_ids).expand(1, T_cur + 1, -1, -1)

    if act_list:
        act_t = torch.stack(act_list).unsqueeze(0).to(device)
        pad = torch.zeros(1, T_cur + 1 - act_t.shape[1], agent.action_dim, device=device)
        act_t = torch.cat([pad, act_t], dim=1)
    else:
        act_t = torch.zeros(1, T_cur + 1, agent.action_dim, device=device)

    z_next, a_out = sample_one_timestep(
        agent.dynamics,
        past_packed=z_packed,
        k_max=cfg.k_max,
        K=cfg.imagination_K,
        actions=act_t,
        tau_ctx=cfg.imagination_tau_ctx,
        agent_tokens=agent_tok,
    )

    pred_reward = 0.0
    pred_value = 0.0
    if a_out is not None:
        agent_embed = a_out.mean(dim=-2)  # (1, d_model)
        pred_reward = agent.reward_head.predict(agent_embed).item()
        pred_value = agent.value_head.predict(agent_embed).item()

    z_next_4d = z_next.unsqueeze(1)  # (1, 1, n_spatial, d_spatial)
    bottleneck = unpack_spatial_to_bottleneck(z_next_4d, k=cfg.packing_k).clamp(-1, 1)
    patch_pred = agent.tokenizer.decoder(bottleneck)
    img_pred = unpatchify(
        patch_pred, cfg.image_size, cfg.image_size, cfg.channels, cfg.patch_size,
    ).clamp(0, 1)

    img_np = (img_pred[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img_np, pred_reward, pred_value


def run_interactive(
    agent: Dreamer4Agent | None,
    cfg: TrainConfig,
    device: torch.device,
    has_checkpoint: bool,
):
    """Main pygame loop."""
    import pygame

    env = DMControlEnv(
        cfg.domain, cfg.task,
        size=cfg.image_size,
        action_repeat=cfg.action_repeat,
    )

    img_size = cfg.image_size
    display_size = img_size * DISPLAY_SCALE

    window_w = display_size * 2 + PANEL_GAP
    window_h = display_size + BAR_HEIGHT

    pygame.init()
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption(
        f"Dreamer 4 — {cfg.domain}-{cfg.task} "
        f"({'checkpoint loaded' if has_checkpoint else 'no checkpoint'})"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", FONT_SIZE)

    use_policy = False
    running = True

    obs_tensor = env.reset()
    obs_history: list[torch.Tensor] = [obs_tensor]
    act_history: list[torch.Tensor] = []
    ep_return = 0.0
    ep_steps = 0

    while running:
        keyboard_action = 0.0
        should_reset = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_p and has_checkpoint:
                    use_policy = not use_policy
                    logger.info("Mode: %s", "POLICY" if use_policy else "KEYBOARD")
                elif event.key == pygame.K_r:
                    should_reset = True

        if not running:
            break

        if should_reset:
            obs_tensor = env.reset()
            obs_history = [obs_tensor]
            act_history = []
            ep_return = 0.0
            ep_steps = 0
            continue

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            keyboard_action = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            keyboard_action = 1.0

        if use_policy and agent is not None:
            action = agent.policy_action(obs_history, act_history)
        else:
            action = torch.tensor([keyboard_action], dtype=torch.float32)

        obs_tensor, reward, done, truncated, info = env.step(action)
        obs_history.append(obs_tensor)
        act_history.append(action)
        ep_return += reward
        ep_steps += 1

        if len(obs_history) > cfg.policy_max_context + 2:
            obs_history = obs_history[-(cfg.policy_max_context + 2):]
            act_history = act_history[-(cfg.policy_max_context + 2):]

        env_rgb = env.render_rgb(size=img_size)

        pred_rgb = None
        pred_rw = 0.0
        pred_val = 0.0
        if agent is not None:
            try:
                pred_rgb, pred_rw, pred_val = predict_next_frame(
                    agent, obs_history, act_history, cfg, device,
                )
            except Exception as e:
                logger.warning("WM prediction failed: %s", e)

        screen.fill((30, 30, 30))

        env_surf = pygame.surfarray.make_surface(
            np.transpose(env_rgb, (1, 0, 2))
        )
        env_surf = pygame.transform.scale(env_surf, (display_size, display_size))
        screen.blit(env_surf, (0, 0))

        if pred_rgb is not None:
            pred_surf = pygame.surfarray.make_surface(
                np.transpose(pred_rgb, (1, 0, 2))
            )
            pred_surf = pygame.transform.scale(pred_surf, (display_size, display_size))
            screen.blit(pred_surf, (display_size + PANEL_GAP, 0))
        else:
            placeholder = font.render("No WM prediction", True, (120, 120, 120))
            screen.blit(placeholder, (display_size + PANEL_GAP + 20, display_size // 2))

        label_env = font.render("Environment", True, (200, 200, 200))
        label_wm = font.render("World Model", True, (200, 200, 200))
        screen.blit(label_env, (4, 2))
        screen.blit(label_wm, (display_size + PANEL_GAP + 4, 2))

        mode_str = "POLICY" if use_policy else "KEYBOARD"
        mode_color = (100, 200, 100) if use_policy else (200, 200, 100)
        bar_y = display_size + 4

        info_parts = [
            f"Mode: {mode_str}",
            f"Step: {ep_steps}",
            f"Return: {ep_return:.1f}",
            f"R(pred): {pred_rw:.2f}",
            f"V(pred): {pred_val:.2f}",
            f"Act: {action.tolist()}",
        ]

        text_surf = font.render("  |  ".join(info_parts), True, mode_color)
        screen.blit(text_surf, (8, bar_y))

        help_text = font.render(
            "[Left/Right] move  [P] toggle policy  [R] reset  [Q] quit",
            True, (100, 100, 100),
        )
        screen.blit(help_text, (8, bar_y + 22))

        pygame.display.flip()

        if done:
            logger.info("Episode done. Return=%.2f, Steps=%d", ep_return, ep_steps)
            obs_tensor = env.reset()
            obs_history = [obs_tensor]
            act_history = []
            ep_return = 0.0
            ep_steps = 0

        clock.tick(FPS)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Dreamer 4 — Keyboard Control")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, default="config/cartpole_balance_light.yaml",
        help="Path to YAML config (used if no config in checkpoint)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Run without a checkpoint (keyboard only, random WM)",
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
    device = torch.device(cfg.device)

    agent: Dreamer4Agent | None = None
    has_checkpoint = False

    if args.checkpoint and not args.no_checkpoint:
        logger.info("Loading checkpoint from %s", args.checkpoint)
        agent = Dreamer4Agent(cfg)
        tok = make_tokenizer(cfg)
        agent.set_tokenizer(tok)
        agent.update_prior_policy()
        agent = agent.to(device)

        meta = load_checkpoint(args.checkpoint, agent, map_location=cfg.device)
        logger.info(
            "Checkpoint loaded: train_step=%d, env_steps=%d",
            meta["train_step"], meta["env_steps"],
        )
        agent.eval()
        has_checkpoint = True

    elif not args.no_checkpoint and args.checkpoint is None:
        logger.info("No checkpoint provided. Building untrained agent for WM visualization.")
        agent = Dreamer4Agent(cfg)
        tok = make_tokenizer(cfg)
        agent.set_tokenizer(tok)
        agent = agent.to(device)
        agent.eval()

    else:
        logger.info("Running without agent (keyboard control only, no WM predictions).")

    logger.info("Starting interactive session: %s-%s", cfg.domain, cfg.task)
    logger.info("Controls: Left/Right arrows to move, P to toggle policy, R to reset, Q to quit")

    run_interactive(agent, cfg, device, has_checkpoint)


if __name__ == "__main__":
    main()
