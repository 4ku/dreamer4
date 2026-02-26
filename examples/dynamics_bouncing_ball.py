"""
Dynamics Bouncing Ball — end-to-end test of the shortcut forcing pipeline
with a real Tokenizer (Encoder + Decoder).

Pipeline:
    Phase 1: Pretrain causal tokenizer (MSE + 0.2·LPIPS, RMS-normalized).
    Phase 2: Freeze tokenizer, encode data to latent space, train DynamicsModel
             with shortcut forcing loss.
    Phase 3: Sample autoregressively in latent space, decode through the
             tokenizer decoder back to pixels.

    pixels -> patchify -> Encoder -> pack -> DynamicsModel (shortcut forcing)
           -> sample_sequence -> unpack -> Decoder -> pixels

Run:
    cd dreamer4_pytorch
    python -m examples.dynamics_bouncing_ball
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import lpips as lpips_lib

from dreamer4.dynamics import DynamicsModel, shortcut_forcing_loss, sample_sequence
from dreamer4.tokenizer import (
    Encoder, Decoder, Tokenizer,
    recon_loss_from_mae, lpips_on_mae_recon, RMSLossNormalizer,
)
from dreamer4.utils import (
    patchify, unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

IMG_H, IMG_W, IMG_C = 32, 32, 3
PATCH_SIZE = 8
N_PATCHES = (IMG_H // PATCH_SIZE) * (IMG_W // PATCH_SIZE)  # 16
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * IMG_C                 # 192
BALL_RADIUS = 3.0
BALL_COLOR = torch.tensor([1.0, 0.8, 0.2])

# Tokenizer config
D_MODEL_TOK = 256
N_LATENTS = 16
D_BOTTLENECK = 48

# Dynamics config — operates on packed spatial tokens
PACK_K = 1
N_SPATIAL = N_LATENTS // PACK_K       # 16
D_SPATIAL = D_BOTTLENECK * PACK_K     # 48
D_MODEL_DYN = 128
K_MAX = 4
ACTION_DIM = 2


def _render_frames(
    B: int,
    T: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate B trajectories of a bouncing ball, each with T frames.

    Returns:
        frames:     (B, T, C, H, W) float tensor in [0, 1].
        velocities: (B, T, 2) float tensor (used as actions).
    """
    frames = torch.zeros(B, T, IMG_C, IMG_H, IMG_W, device=device)
    velocities = torch.zeros(B, T, 2, device=device)

    yy = torch.arange(IMG_H, device=device, dtype=torch.float32)
    xx = torch.arange(IMG_W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    margin = BALL_RADIUS + 1
    px = torch.rand(B, device=device) * (IMG_W - 2 * margin) + margin
    py = torch.rand(B, device=device) * (IMG_H - 2 * margin) + margin

    speed = torch.rand(B, 2, device=device) * 1.5 + 1.5
    sign = torch.randint(0, 2, (B, 2), device=device).float() * 2 - 1
    vel = speed * sign

    for t in range(T):
        velocities[:, t] = vel

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

    return frames, velocities


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    N_TRAIN = 512
    N_EVAL = 32
    BATCH_SIZE = 32
    T = 8

    # ==================================================================
    # Phase 1: Pretrain tokenizer (MSE + 0.2·LPIPS, paper Eq. 5)
    # ==================================================================
    print("=" * 60)
    print("Phase 1: Tokenizer pretraining (MSE + 0.2·LPIPS)")
    print("=" * 60)

    print(f"Image: {IMG_H}x{IMG_W}x{IMG_C}, patch {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Patches: {N_PATCHES}  (patch_dim={PATCH_DIM})")
    print(f"Bottleneck: {N_LATENTS} latents x {D_BOTTLENECK}d")
    print(f"Pack factor k={PACK_K} -> dynamics sees "
          f"{N_SPATIAL} spatial x {D_SPATIAL}d")

    enc = Encoder(
        patch_dim=PATCH_DIM, d_model=D_MODEL_TOK, n_latents=N_LATENTS,
        n_patches=N_PATCHES, n_heads=4, depth=4, d_bottleneck=D_BOTTLENECK,
        time_every=1, mae_p_min=0.0, mae_p_max=0.9,
        use_qk_norm=False, logit_cap=None,
    )
    dec = Decoder(
        d_bottleneck=D_BOTTLENECK, d_model=D_MODEL_TOK, n_latents=N_LATENTS,
        n_patches=N_PATCHES, patch_dim=PATCH_DIM, n_heads=4, depth=4,
        time_every=1, use_qk_norm=False, logit_cap=None,
    )
    tok = Tokenizer(enc, dec).to(device)
    tok_params = sum(p.numel() for p in tok.parameters())
    print(f"Tokenizer params: {tok_params:,}")

    print(f"Generating {N_TRAIN} train + {N_EVAL} eval trajectories ({T} frames)...")
    train_frames, train_vels = _render_frames(N_TRAIN, T, device)
    eval_frames, eval_vels = _render_frames(N_EVAL, T, device)

    train_patches = patchify(train_frames, PATCH_SIZE)
    eval_patches = patchify(eval_frames, PATCH_SIZE)

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    loss_norm_tok = RMSLossNormalizer(n_losses=2).to(device)
    tok_optimizer = torch.optim.Adam(tok.parameters(), lr=3e-4)
    tok_steps = 3000

    tok_step_list: list[int] = []
    tok_losses: list[float] = []
    tok_eval_mse: list[float] = []

    print(f"Training tokenizer for {tok_steps} steps...")
    print("-" * 60)

    for step in range(1, tok_steps + 1):
        tok.train()
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        batch = train_patches[idx]

        pred, mae_mask, _ = tok(batch)
        mse = recon_loss_from_mae(pred, batch, mae_mask)
        lp = lpips_on_mae_recon(
            lpips_fn, pred, batch, mae_mask,
            H=IMG_H, W=IMG_W, C=IMG_C, patch_size=PATCH_SIZE,
        )
        loss_norm_tok.update(0, mse)
        loss_norm_tok.update(1, lp)
        loss = loss_norm_tok.normalize(0, mse) + 0.2 * loss_norm_tok.normalize(1, lp)

        tok_optimizer.zero_grad()
        loss.backward()
        tok_optimizer.step()

        if step % 10 == 0 or step == 1:
            tok_step_list.append(step)
            tok_losses.append(loss.item())
            tok.eval()
            with torch.no_grad():
                ev_pred, _, _ = tok(eval_patches)
                ev_mse = (ev_pred - eval_patches).pow(2).mean().item()
            tok_eval_mse.append(ev_mse)

        if step % 500 == 0 or step == 1:
            print(f"  step {step:4d}  loss={loss.item():.5f}  "
                  f"mse={mse.item():.5f}  lpips={lp.item():.5f}  "
                  f"eval_mse={tok_eval_mse[-1]:.5f}")

    print("-" * 60)
    print(f"Final eval MSE: {tok_eval_mse[-1]:.6f}")
    print("Tokenizer pretraining complete.")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(tok_step_list, tok_losses, linewidth=1.5, label="Train (combined)", alpha=0.8)
    ax.semilogy(tok_step_list, tok_eval_mse, linewidth=1.5, label="Eval full MSE", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Tokenizer Pretraining — Bouncing Ball")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dynamics_ball_tok_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_DIR / 'dynamics_ball_tok_loss.png'}")

    # ==================================================================
    # Encode all data into latent space (frozen tokenizer)
    # ==================================================================
    tok.eval()
    with torch.no_grad():
        train_z_bot = tok.encode(train_patches)  # (N_TRAIN, T, N_LATENTS, D_BOTTLENECK)
        eval_z_bot = tok.encode(eval_patches)

    train_z = pack_bottleneck_to_spatial(train_z_bot, n_spatial=N_SPATIAL, k=PACK_K)
    eval_z = pack_bottleneck_to_spatial(eval_z_bot, n_spatial=N_SPATIAL, k=PACK_K)

    print(f"Encoded latent shape: {train_z.shape}  "
          f"range [{train_z.min():.2f}, {train_z.max():.2f}]")

    all_vels = torch.cat([train_vels, eval_vels], 0)
    train_actions = all_vels[:N_TRAIN]
    eval_actions = all_vels[N_TRAIN:]

    vel_scale = train_actions.abs().max().item()
    if vel_scale > 0:
        train_actions = train_actions / vel_scale
        eval_actions = eval_actions / vel_scale

    # ==================================================================
    # Phase 2: Dynamics training with shortcut forcing
    # ==================================================================
    print()
    print("=" * 60)
    print("Phase 2: Dynamics training (shortcut forcing)")
    print("=" * 60)

    dynamics = DynamicsModel(
        d_model=D_MODEL_DYN, d_spatial=D_SPATIAL, n_spatial=N_SPATIAL,
        n_register=2, n_agent=0,
        n_heads=4, depth=6, k_max=K_MAX, action_dim=ACTION_DIM,
        time_every=2, dropout=0.0,
        use_qk_norm=False, logit_cap=None,
        space_mode="wm_agent_isolated",
    ).to(device)

    dyn_params = sum(p.numel() for p in dynamics.parameters())
    print(f"Dynamics params: {dyn_params:,}")

    dyn_optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-4)
    dyn_steps = 3000
    dyn_step_list: list[int] = []
    dyn_losses: list[float] = []
    dyn_flow: list[float] = []
    dyn_boot: list[float] = []
    initial_dyn_loss = None

    print(f"Training dynamics for {dyn_steps} steps...")
    print("-" * 60)

    for step in range(1, dyn_steps + 1):
        dynamics.train()
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        z1_batch = train_z[idx]
        act_batch = train_actions[idx]

        loss, aux = shortcut_forcing_loss(
            dynamics, z1=z1_batch, actions=act_batch,
            k_max=K_MAX, bootstrap_fraction=0.25,
        )

        if step == 1:
            initial_dyn_loss = loss.item()

        dyn_optimizer.zero_grad()
        loss.backward()
        dyn_optimizer.step()

        if step % 10 == 0 or step == 1:
            dyn_step_list.append(step)
            dyn_losses.append(loss.item())
            dyn_flow.append(aux["flow_mse"].item())
            dyn_boot.append(aux["boot_mse"].item())

        if step % 300 == 0 or step == 1:
            print(f"  step {step:4d}  loss={loss.item():.6f}  "
                  f"flow={aux['flow_mse'].item():.6f}  "
                  f"boot={aux['boot_mse'].item():.6f}")

    final_dyn_loss = dyn_losses[-1]
    print("-" * 60)
    print(f"Initial dynamics loss: {initial_dyn_loss:.6f}")
    print(f"Final dynamics loss:   {final_dyn_loss:.6f}")
    print(f"Reduction:             {final_dyn_loss / initial_dyn_loss:.4f}x")

    assert final_dyn_loss < initial_dyn_loss * 0.50, (
        f"Dynamics did not converge: {initial_dyn_loss:.6f} -> {final_dyn_loss:.6f}"
    )
    print("Dynamics training PASSED.")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(dyn_step_list, dyn_losses, linewidth=1.5, label="Combined", alpha=0.9)
    ax.semilogy(dyn_step_list, dyn_flow, linewidth=1.2, label="Flow MSE",
                alpha=0.7, linestyle="--")
    ax.semilogy(dyn_step_list, dyn_boot, linewidth=1.2, label="Boot MSE",
                alpha=0.7, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Dynamics Shortcut Forcing — Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dynamics_ball_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_DIR / 'dynamics_ball_loss.png'}")

    # ==================================================================
    # Phase 3: Sampling and visualization (decode through tokenizer)
    # ==================================================================
    print()
    print("=" * 60)
    print("Phase 3: Sampling & visualization (tokenizer decode)")
    print("=" * 60)

    dynamics.eval()
    tok.eval()
    N_CTX = 2
    HORIZON = T - N_CTX

    with torch.no_grad():
        gen_z = sample_sequence(
            dynamics,
            context=eval_z[:, :N_CTX],
            horizon=HORIZON,
            k_max=K_MAX,
            K=K_MAX,
            actions=eval_actions,
            tau_ctx=0.1,
        )  # (N_EVAL, T, N_SPATIAL, D_SPATIAL)

        gen_z_bot = unpack_spatial_to_bottleneck(gen_z, k=PACK_K)
        gen_z_bot = gen_z_bot.clamp(-1, 1)

        gen_patches = tok.decoder(gen_z_bot)  # (N_EVAL, T, N_PATCHES, PATCH_DIM)
        gen_images = unpatchify(gen_patches, IMG_H, IMG_W, IMG_C, PATCH_SIZE).clamp(0, 1)

    gt_z = eval_z[:, N_CTX:]
    gen_z_pred = gen_z[:, N_CTX:]
    latent_mse = (gen_z_pred - gt_z).pow(2).mean().item()
    print(f"Latent MSE (generated vs GT, predicted frames): {latent_mse:.6f}")

    # Also show the tokenizer-only reconstruction for reference
    with torch.no_grad():
        recon_patches = tok.decoder(eval_z_bot)
        recon_images = unpatchify(recon_patches, IMG_H, IMG_W, IMG_C, PATCH_SIZE).clamp(0, 1)

    n_traj = min(N_EVAL, 4)
    n_show = T
    n_rows = n_traj * 3  # GT, tokenizer recon, dynamics generated

    fig, axes = plt.subplots(
        n_rows, n_show,
        figsize=(1.6 * n_show + 0.8, 1.6 * n_rows + 2.0),
    )

    for b in range(n_traj):
        for t in range(n_show):
            gt = eval_frames[b, t].permute(1, 2, 0).cpu().numpy()
            rc = recon_images[b, t].permute(1, 2, 0).cpu().numpy()
            pr = gen_images[b, t].permute(1, 2, 0).cpu().numpy()

            r_gt = b * 3
            r_rc = b * 3 + 1
            r_pr = b * 3 + 2

            for r, img in [(r_gt, gt), (r_rc, rc), (r_pr, pr)]:
                axes[r, t].imshow(img)
                axes[r, t].set_xticks([])
                axes[r, t].set_yticks([])

            for spine in axes[r_rc, t].spines.values():
                spine.set_edgecolor("#9E9E9E")
                spine.set_linewidth(2)

            if t < N_CTX:
                for spine in axes[r_pr, t].spines.values():
                    spine.set_edgecolor("#4CAF50")
                    spine.set_linewidth(2)
            else:
                for spine in axes[r_pr, t].spines.values():
                    spine.set_edgecolor("#1565C0")
                    spine.set_linewidth(2)

            if b == 0:
                label = f"t={t}"
                if t < N_CTX:
                    label += " (ctx)"
                axes[r_gt, t].set_title(label, fontsize=9)

    fig.suptitle(
        "Dynamics Bouncing Ball — GT / Tokenizer Recon / Dynamics Generated",
        fontsize=12, fontweight="bold",
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="none", edgecolor="#4CAF50", linewidth=2,
              label="Context (ground-truth input)"),
        Patch(facecolor="none", edgecolor="#1565C0", linewidth=2,
              label="Predicted (dynamics rollout)"),
        Patch(facecolor="none", edgecolor="#9E9E9E", linewidth=2,
              label="Tokenizer reconstruction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, frameon=True, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.53, 0.005))

    fig.tight_layout(rect=[0.07, 0.04, 1, 0.95])

    for b in range(n_traj):
        gt_pos = axes[b * 3, 0].get_position()
        rc_pos = axes[b * 3 + 1, 0].get_position()
        pr_pos = axes[b * 3 + 2, 0].get_position()
        fig.text(
            0.01, (gt_pos.y0 + gt_pos.y1) / 2,
            f"#{b+1} TRUE", fontsize=8, fontweight="bold",
            va="center", ha="left",
        )
        fig.text(
            0.01, (rc_pos.y0 + rc_pos.y1) / 2,
            f"#{b+1} RECON", fontsize=8, fontweight="bold",
            va="center", ha="left", color="#9E9E9E",
        )
        fig.text(
            0.01, (pr_pos.y0 + pr_pos.y1) / 2,
            f"#{b+1} GEN", fontsize=8, fontweight="bold",
            va="center", ha="left", color="#1565C0",
        )

    frames_path = OUTPUT_DIR / "dynamics_ball_frames.png"
    fig.savefig(frames_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {frames_path}")

    print()
    print("All phases complete.")


if __name__ == "__main__":
    main()
