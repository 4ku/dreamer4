"""
Tokenizer Reconstruction — paper-faithful Causal Tokenizer training.

Generates synthetic 32x32 RGB bouncing-ball video, then trains the
Encoder -> bottleneck -> Decoder using the exact objective from the
Dreamer 4 paper (Section 3.1, Eq. 5):

    L = L_MSE + 0.2 * L_LPIPS        (both RMS-normalized)

with MAE masking (p ~ U(0, 0.9)) on encoder input patches.

This exercises the complete tokenizer path:

    raw pixels -> patchify -> Encoder (MAE + BCT + tanh)
              -> bottleneck z in [-1,1]
              -> Decoder (BCT + sigmoid) -> unpatchify -> reconstructed pixels

Run:
    cd dreamer4_pytorch
    python -m examples.tokenizer_recon
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import lpips as lpips_lib

from dreamer4.tokenizer import (
    Encoder, Decoder, Tokenizer,
    recon_loss_from_mae, lpips_on_mae_recon, RMSLossNormalizer,
)
from dreamer4.utils import patchify, unpatchify

OUTPUT_DIR = Path(__file__).parent / "outputs"

IMG_H, IMG_W, IMG_C = 32, 32, 3
PATCH_SIZE = 8
N_PATCHES = (IMG_H // PATCH_SIZE) * (IMG_W // PATCH_SIZE)  # 16
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * IMG_C                 # 192
BALL_RADIUS = 3.0
BALL_COLOR = torch.tensor([1.0, 0.8, 0.2])


def _render_frames(
    B: int,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate B trajectories of a bouncing ball, each with T frames.

    Returns:
        frames: (B, T, C, H, W) float tensor in [0, 1].
    """
    frames = torch.zeros(B, T, IMG_C, IMG_H, IMG_W, device=device)

    yy = torch.arange(IMG_H, device=device, dtype=torch.float32)
    xx = torch.arange(IMG_W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    margin = BALL_RADIUS + 1
    px = torch.rand(B, device=device) * (IMG_W - 2 * margin) + margin
    py = torch.rand(B, device=device) * (IMG_H - 2 * margin) + margin

    speed = torch.rand(B, 2, device=device) * 1.5 + 1.5
    sign = (torch.randint(0, 2, (B, 2), device=device).float() * 2 - 1)
    vel = speed * sign

    for t in range(T):
        dx = grid_x.unsqueeze(0) - px.view(B, 1, 1)
        dy = grid_y.unsqueeze(0) - py.view(B, 1, 1)
        dist_sq = dx * dx + dy * dy
        alpha = torch.clamp(1.0 - (dist_sq.sqrt() - BALL_RADIUS + 1.5) / 1.5, 0, 1)

        for c in range(IMG_C):
            frames[:, t, c] = alpha * BALL_COLOR[c]

        px = px + vel[:, 0]
        py = py + vel[:, 1]

        for coord, limit in [(px, IMG_W), (py, IMG_H)]:
            below = coord < BALL_RADIUS
            above = coord > limit - BALL_RADIUS - 1
            if below.any() or above.any():
                idx = 0 if coord is px else 1
                vel[:, idx] = torch.where(below | above, -vel[:, idx], vel[:, idx])
                coord.clamp_(BALL_RADIUS, limit - BALL_RADIUS - 1)

    return frames


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    D_MODEL = 256
    N_LATENTS = 16
    D_BOTTLENECK = 48

    enc = Encoder(
        patch_dim=PATCH_DIM,
        d_model=D_MODEL,
        n_latents=N_LATENTS,
        n_patches=N_PATCHES,
        n_heads=4,
        depth=4,
        d_bottleneck=D_BOTTLENECK,
        time_every=1,
        mae_p_min=0.0,
        mae_p_max=0.9,
        use_qk_norm=False,
        logit_cap=None,
    )
    dec = Decoder(
        d_bottleneck=D_BOTTLENECK,
        d_model=D_MODEL,
        n_latents=N_LATENTS,
        n_patches=N_PATCHES,
        patch_dim=PATCH_DIM,
        n_heads=4,
        depth=4,
        time_every=1,
        use_qk_norm=False,
        logit_cap=None,
    )
    tok = Tokenizer(enc, dec).to(device)

    total_params = sum(p.numel() for p in tok.parameters())
    print(f"Image:        {IMG_H}x{IMG_W}x{IMG_C}, patch {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Patches:      {N_PATCHES}  (patch_dim={PATCH_DIM})")
    print(f"d_model:      {D_MODEL}")
    print(f"Bottleneck:   {N_LATENTS} latents x {D_BOTTLENECK}d  "
          f"(compression: {N_PATCHES * PATCH_DIM} -> {N_LATENTS * D_BOTTLENECK})")
    print(f"MAE masking:  U(0.0, 0.9)")
    print(f"Parameters:   {total_params:,}")
    print()

    # ── Synthetic data ───────────────────────────────────────────────────────
    N_TRAIN, N_EVAL = 512, 32
    BATCH_SIZE = 32
    T = 8

    print(f"Generating {N_TRAIN} train and {N_EVAL} eval trajectories ({T} frames each)...")
    train_frames = _render_frames(N_TRAIN, T, device)
    eval_frames = _render_frames(N_EVAL, T, device)

    train_patches = patchify(train_frames, PATCH_SIZE)
    eval_patches = patchify(eval_frames, PATCH_SIZE)

    # ── Loss setup (paper Eq. 5) ────────────────────────────────────────────
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    loss_norm = RMSLossNormalizer(n_losses=2).to(device)

    # ── Training loop ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(tok.parameters(), lr=3e-4)
    n_steps = 3000

    print(f"Training for {n_steps} steps  (Eq. 5: MSE + 0.2*LPIPS, RMS-normalized)")
    print("-" * 60)

    steps: list[int] = []
    train_losses: list[float] = []
    train_mse_losses: list[float] = []
    train_lpips_losses: list[float] = []
    eval_losses: list[float] = []
    initial_loss = None

    for step in range(1, n_steps + 1):
        tok.train()
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        batch = train_patches[idx]

        pred, mae_mask, keep_prob = tok(batch)

        mse_loss = recon_loss_from_mae(pred, batch, mae_mask)
        perceptual_loss = lpips_on_mae_recon(
            lpips_fn, pred, batch, mae_mask,
            H=IMG_H, W=IMG_W, C=IMG_C, patch_size=PATCH_SIZE,
        )

        loss_norm.update(0, mse_loss)
        loss_norm.update(1, perceptual_loss)

        loss = loss_norm.normalize(0, mse_loss) + 0.2 * loss_norm.normalize(1, perceptual_loss)

        if step == 1:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            steps.append(step)
            train_losses.append(loss.item())
            train_mse_losses.append(mse_loss.item())
            train_lpips_losses.append(perceptual_loss.item())

            tok.eval()
            with torch.no_grad():
                eval_pred, _, _ = tok(eval_patches)
                eval_loss = (eval_pred - eval_patches).pow(2).mean().item()
            eval_losses.append(eval_loss)

        if step % 200 == 0 or step == 1:
            print(
                f"  step {step:4d}  loss={loss.item():.5f}  "
                f"mse={mse_loss.item():.5f}  lpips={perceptual_loss.item():.5f}  "
                f"eval_full_mse={eval_loss:.5f}"
            )

    final_loss = eval_losses[-1]
    initial_eval = eval_losses[0]
    print("-" * 60)
    print(f"Initial eval full MSE: {initial_eval:.6f}")
    print(f"Final eval full MSE:   {final_loss:.6f}")
    print(f"Reduction:             {final_loss / initial_eval:.4f}x")

    assert final_loss < initial_eval * 0.1, (
        f"Tokenizer reconstruction failed to converge: "
        f"{initial_eval:.6f} -> {final_loss:.6f}"
    )
    print("\nTokenizer reconstruction PASSED.")

    # ── Visualization ────────────────────────────────────────────────────────

    # 1) Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(steps, train_losses, linewidth=1.5, label="Combined (MSE + 0.2·LPIPS)", alpha=0.8)
    ax.semilogy(steps, train_mse_losses, linewidth=1.5, label="MSE (masked)", alpha=0.6, linestyle="--")
    ax.semilogy(steps, train_lpips_losses, linewidth=1.5, label="LPIPS (composited)", alpha=0.6, linestyle=":")
    ax.semilogy(steps, eval_losses, linewidth=1.5, label="Eval full MSE", alpha=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Tokenizer Reconstruction — Eq. 5 Loss (MSE + 0.2·LPIPS, RMS-norm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = OUTPUT_DIR / "tokenizer_recon_loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve       -> {loss_path}")

    # 2) Frame comparison: ground truth vs reconstruction
    tok.eval()
    with torch.no_grad():
        recon_patches = tok.decoder(tok.encode(eval_patches))
    recon_images = unpatchify(recon_patches, IMG_H, IMG_W, IMG_C, PATCH_SIZE).clamp(0, 1)

    n_traj = min(N_EVAL, 4)
    n_show = min(T, 8)
    n_rows = n_traj * 2

    fig, axes = plt.subplots(
        n_rows, n_show, figsize=(1.6 * n_show + 0.8, 1.6 * n_rows + 1.0),
    )

    for b in range(n_traj):
        for t in range(n_show):
            gt = eval_frames[b, t].permute(1, 2, 0).cpu().numpy()
            pr = recon_images[b, t].permute(1, 2, 0).cpu().numpy()

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
                axes[r_gt, t].set_title(f"t={t}", fontsize=9)

    fig.suptitle(
        "Tokenizer Reconstruction — Ground Truth vs Decoded",
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
            f"#{b+1} RECON", fontsize=8, fontweight="bold",
            va="center", ha="left", color="#1565C0",
        )
    frames_path = OUTPUT_DIR / "tokenizer_recon_frames.png"
    fig.savefig(frames_path, dpi=150)
    plt.close(fig)
    print(f"Saved frame comparison -> {frames_path}")

    # 3) MAE pipeline visualization: original -> masked input -> reconstruction
    #    Shows what the encoder sees (with patches dropped) vs what it recovers.
    tok.train()
    torch.manual_seed(123)
    sample = eval_patches[:1]  # (1, T, Np, Dp)
    with torch.no_grad():
        z, (mae_mask, _) = tok.encoder(sample)
        pred = tok.decoder(z)

        masked_patches = sample.clone()
        masked_patches[mae_mask.expand_as(sample)] = 0.0

    gt_img = eval_frames[0].cpu()           # (T, C, H, W)
    masked_img = unpatchify(masked_patches, IMG_H, IMG_W, IMG_C, PATCH_SIZE)[0].clamp(0, 1).cpu()
    recon_img2 = unpatchify(pred, IMG_H, IMG_W, IMG_C, PATCH_SIZE)[0].clamp(0, 1).cpu()

    row_labels = ["Original", "Encoder input\n(MAE masked)", "Reconstruction"]
    row_data = [gt_img, masked_img, recon_img2]

    fig, axes = plt.subplots(3, n_show, figsize=(1.6 * n_show + 0.8, 1.6 * 3 + 1.2))
    for row, (label, imgs) in enumerate(zip(row_labels, row_data)):
        for t in range(n_show):
            ax = axes[row, t]
            ax.imshow(imgs[t].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(f"t={t}", fontsize=9)

    mask_frac = mae_mask.float().mean().item()
    fig.suptitle(
        f"MAE Pipeline — {mask_frac:.0%} patches masked, then reconstructed",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    mae_path = OUTPUT_DIR / "tokenizer_recon_mae_pipeline.png"
    fig.savefig(mae_path, dpi=150)
    plt.close(fig)
    print(f"Saved MAE pipeline viz -> {mae_path}")


if __name__ == "__main__":
    main()
