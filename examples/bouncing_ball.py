"""
Bouncing Ball Next-Frame Prediction — tests the full image pipeline.

Generates synthetic 32x32 RGB video of a bright ball bouncing inside
a box, then trains the transformer to predict the next frame from the
current history.  This exercises the complete image path:

    raw pixels -> patchify -> project -> transformer -> project -> unpatchify

The model must learn to:
  - Spatially locate the ball within the patch grid  (space attention)
  - Infer velocity / direction from past frames      (causal time attention)
  - Predict where the ball will be next               (output projection)

Run:
    cd dreamer4_pytorch
    python -m examples.bouncing_ball
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dreamer4.modality import Modality, TokenLayout
from dreamer4.configs import make_transformer
from dreamer4.utils import patchify, unpatchify

OUTPUT_DIR = Path(__file__).parent / "outputs"

IMG_H, IMG_W, IMG_C = 32, 32, 3
PATCH_SIZE = 8
N_PATCHES = (IMG_H // PATCH_SIZE) * (IMG_W // PATCH_SIZE)  # 16
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * IMG_C                 # 192
BALL_RADIUS = 3.0
BALL_COLOR = torch.tensor([1.0, 0.8, 0.2])  # warm yellow


def _render_frames(
    B: int,
    T: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate B trajectories of a bouncing ball, each with T frames.

    Returns:
        frames: (B, T, C, H, W) float tensor in [0, 1].
        velocities: (B, T, 2) float tensor.
    """
    frames = torch.zeros(B, T, IMG_C, IMG_H, IMG_W, device=device)
    velocities = torch.zeros(B, T, 2, device=device)

    # Pixel coordinate grids
    yy = torch.arange(IMG_H, device=device, dtype=torch.float32)
    xx = torch.arange(IMG_W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")  # (H, W)

    # Random starting positions (keep away from edges)
    margin = BALL_RADIUS + 1
    px = torch.rand(B, device=device) * (IMG_W - 2 * margin) + margin
    py = torch.rand(B, device=device) * (IMG_H - 2 * margin) + margin

    # Random velocities: 1.5 to 3.0 pixels/frame in each axis, random sign
    speed = torch.rand(B, 2, device=device) * 1.5 + 1.5
    sign = (torch.randint(0, 2, (B, 2), device=device).float() * 2 - 1)
    vel = speed * sign  # (B, 2) — [vx, vy]

    for t in range(T):
        velocities[:, t] = vel

        # Distance from each pixel to the ball centre
        dx = grid_x.unsqueeze(0) - px.view(B, 1, 1)  # (B, H, W)
        dy = grid_y.unsqueeze(0) - py.view(B, 1, 1)
        dist_sq = dx * dx + dy * dy

        # Soft circle: smooth falloff gives anti-aliased edges
        alpha = torch.clamp(1.0 - (dist_sq.sqrt() - BALL_RADIUS + 1.5) / 1.5, 0, 1)

        # Paint the ball colour
        for c in range(IMG_C):
            frames[:, t, c] = alpha * BALL_COLOR[c]

        # Advance position
        px = px + vel[:, 0]
        py = py + vel[:, 1]

        # Bounce off walls
        for coord, limit in [(px, IMG_W), (py, IMG_H)]:
            below = coord < BALL_RADIUS
            above = coord > limit - BALL_RADIUS - 1
            if below.any() or above.any():
                idx = 0 if coord is px else 1
                vel[:, idx] = torch.where(below | above, -vel[:, idx], vel[:, idx])
                coord.clamp_(BALL_RADIUS, limit - BALL_RADIUS - 1)

    return frames, velocities


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Layout & model ───────────────────────────────────────────────────────
    layout = TokenLayout(
        n_latents=0,
        segments=(
            (Modality.ACTION, 1),
            (Modality.IMAGE, N_PATCHES)
        ),
    )

    # d_model must be >= patch_dim (192) to avoid an information bottleneck
    # that would make predictions blurry.  "small" gives d_model=128, so we
    # override to 256 which comfortably exceeds patch_dim.
    config_name = "small"
    model = make_transformer(
        config_name, layout, "wm_agent_isolated",
        d_model=256,
        use_qk_norm=False, logit_cap=None, dropout=0.0,
        time_every=1,
    ).to(device)
    d_model = model.d_model

    action_embed = nn.Linear(2, d_model).to(device)
    patch_embed = nn.Linear(PATCH_DIM, d_model).to(device)
    patch_decode = nn.Linear(d_model, PATCH_DIM).to(device)

    all_params = (
        list(model.parameters())
        + list(action_embed.parameters())
        + list(patch_embed.parameters())
        + list(patch_decode.parameters())
    )
    total_params = sum(p.numel() for p in all_params)
    print(f"Config:     {config_name} (d_model={d_model}, time_every=1)")
    print(f"Image:      {IMG_H}x{IMG_W}x{IMG_C}, patch {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Patches:    {N_PATCHES}  (patch_dim={PATCH_DIM})")
    print(f"d_model:    {d_model}  (>= patch_dim, no bottleneck)")
    print(f"Parameters: {total_params:,}")
    print()

    # ── Synthetic data ───────────────────────────────────────────────────────
    N_TRAIN, N_EVAL = 1024, 32
    BATCH_SIZE = 32
    T = 12
    
    print(f"Generating {N_TRAIN} train and {N_EVAL} eval trajectories...")
    train_frames, train_vels = _render_frames(N_TRAIN, T + 1, device)
    eval_frames, eval_vels = _render_frames(N_EVAL, T + 1, device)

    train_input = patchify(train_frames[:, :-1], PATCH_SIZE)
    train_input_vels = train_vels[:, :-1]
    train_target = patchify(train_frames[:, 1:], PATCH_SIZE)
    
    eval_input = patchify(eval_frames[:, :-1], PATCH_SIZE)
    eval_input_vels = eval_vels[:, :-1]
    eval_target = patchify(eval_frames[:, 1:], PATCH_SIZE)

    # ── Training loop ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(all_params, lr=3e-4)
    n_steps = 3000

    print(f"Training for {n_steps} steps  (next-frame prediction, MSE loss)")
    print("-" * 55)

    steps: list[int] = []
    train_losses: list[float] = []
    eval_losses: list[float] = []
    initial_loss = None

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        batch_input = train_input[idx]
        batch_input_vels = train_input_vels[idx]
        batch_target = train_target[idx]

        img_tokens = patch_embed(batch_input)
        vel_tokens = action_embed(batch_input_vels).unsqueeze(2)
        tokens = torch.cat([vel_tokens, img_tokens], dim=2)
        
        tokens = model(tokens)
        pred_img_tokens = tokens[:, :, 1:]
        pred_patches = patch_decode(pred_img_tokens)

        loss = (pred_patches - batch_target).pow(2).mean()

        if step == 1:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            steps.append(step)
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                curr_img_toks = patch_embed(eval_input[:, 0:1])
                preds = []
                for t_step in range(T):
                    curr_vel = action_embed(eval_input_vels[:, :t_step+1]).unsqueeze(2)
                    tokens = torch.cat([curr_vel, curr_img_toks], dim=2)
                    out = model(tokens)
                    next_img_out = out[:, t_step:t_step+1, 1:]
                    preds.append(next_img_out)
                    if t_step < T - 1:
                        next_patches = patch_decode(next_img_out)
                        next_img_toks = patch_embed(next_patches)
                        curr_img_toks = torch.cat([curr_img_toks, next_img_toks], dim=1)
                
                eval_pred_img = torch.cat(preds, dim=1)
                eval_pred = patch_decode(eval_pred_img)
                eval_loss = (eval_pred - eval_target).pow(2).mean().item()
            model.train()
            eval_losses.append(eval_loss)

        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  train_loss = {loss.item():.6f}  eval_loss = {eval_loss:.6f}")

    final_loss = eval_losses[-1]
    print("-" * 55)
    print(f"Initial train loss: {initial_loss:.6f}")
    print(f"Final eval loss:    {final_loss:.6f}")
    print(f"Reduction:          {final_loss / initial_loss:.4f}x")

    assert final_loss < initial_loss * 0.10, (
        f"Bouncing ball failed to converge: "
        f"{initial_loss:.6f} -> {final_loss:.6f}"
    )
    print("\nBouncing ball PASSED.")

    # ── Visualization ────────────────────────────────────────────────────────

    # 1) Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(steps, train_losses, linewidth=1.5, label="Train Loss", alpha=0.8)
    ax.semilogy(steps, eval_losses, linewidth=1.5, label="Eval Loss", alpha=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Bouncing Ball — Training & Eval Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = OUTPUT_DIR / "bouncing_ball_loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve     -> {loss_path}")

    # 2) Frame comparison: autoregressive prediction for multiple EVAL trajectories
    model.eval()
    with torch.no_grad():
        curr_img_toks = patch_embed(eval_input[:, 0:1])
        preds = []
        for t_step in range(T):
            curr_vel = action_embed(eval_input_vels[:, :t_step+1]).unsqueeze(2)
            tokens = torch.cat([curr_vel, curr_img_toks], dim=2)
            out = model(tokens)
            next_img_out = out[:, t_step:t_step+1, 1:]
            preds.append(next_img_out)
            if t_step < T - 1:
                next_patches = patch_decode(next_img_out)
                next_img_toks = patch_embed(next_patches)
                curr_img_toks = torch.cat([curr_img_toks, next_img_toks], dim=1)
        pred_patches_final = patch_decode(torch.cat(preds, dim=1))

    pred_images = unpatchify(
        pred_patches_final, H=IMG_H, W=IMG_W, C=IMG_C, patch_size=PATCH_SIZE
    )
    pred_images = pred_images.clamp(0, 1)

    n_traj = min(N_EVAL, 4)
    n_show = min(T, 8)
    n_rows = n_traj * 2

    fig, axes = plt.subplots(
        n_rows, n_show, figsize=(1.6 * n_show + 0.8, 1.6 * n_rows + 1.0),
    )

    for b in range(n_traj):
        for t in range(n_show):
            gt = eval_frames[b, t + 1].permute(1, 2, 0).cpu().numpy()
            pr = pred_images[b, t].permute(1, 2, 0).cpu().numpy()

            r_gt = b * 2
            r_pr = b * 2 + 1

            axes[r_gt, t].imshow(gt)
            axes[r_gt, t].set_xticks([])
            axes[r_gt, t].set_yticks([])

            axes[r_pr, t].imshow(pr)
            axes[r_pr, t].set_xticks([])
            axes[r_pr, t].set_yticks([])
            for spine in axes[r_pr, t].spines.values():
                spine.set_edgecolor("#1565C0")
                spine.set_linewidth(2)

            if b == 0:
                axes[r_gt, t].set_title(f"t={t+1}", fontsize=9)

    for b in range(n_traj):
        # Place row labels using figure-level text (survives tight_layout)
        gt_ax = axes[b * 2, 0]
        pr_ax = axes[b * 2 + 1, 0]

        gt_pos = gt_ax.get_position()
        pr_pos = pr_ax.get_position()

    fig.suptitle(
        "Bouncing Ball — Next-Frame Prediction",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0.07, 0, 1, 0.95])

    for b in range(n_traj):
        gt_pos = axes[b * 2, 0].get_position()
        pr_pos = axes[b * 2 + 1, 0].get_position()
        fig.text(
            0.01, (gt_pos.y0 + gt_pos.y1) / 2,
            f"#{b+1} TRUE", fontsize=8, fontweight="bold",
            va="center", ha="left",
        )
        fig.text(
            0.01, (pr_pos.y0 + pr_pos.y1) / 2,
            f"#{b+1} PRED", fontsize=8, fontweight="bold",
            va="center", ha="left", color="#1565C0",
        )
    frames_path = OUTPUT_DIR / "bouncing_ball_frames.png"
    fig.savefig(frames_path, dpi=150)
    plt.close(fig)
    print(f"Saved frame compare  -> {frames_path}")


if __name__ == "__main__":
    main()
