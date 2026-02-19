"""
Copy Task — simplest sanity check for the Block-Causal Transformer.

The model receives a random input tensor (B, T, S, D) and must reproduce
a fixed random target tensor of the same shape.  A small linear head on
top of the transformer output maps to the target space.  If the model
is wired correctly and gradients flow, it should overfit this in a few
hundred steps.

Run:
    cd dreamer4_pytorch
    python -m examples.copy_task
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dreamer4.modality import Modality, TokenLayout
from dreamer4.configs import make_transformer, print_config_table

OUTPUT_DIR = Path(__file__).parent / "outputs"


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Layout & model ───────────────────────────────────────────────────────
    n_latents = 4
    n_patches = 16
    layout = TokenLayout(
        n_latents=n_latents,
        segments=((Modality.IMAGE, n_patches),),
    )
    S = layout.total_tokens()  # 20

    config_name = "tiny"
    model = make_transformer(
        config_name, layout, "encoder",
        use_qk_norm=False, logit_cap=None, dropout=0.0,
    ).to(device)
    d_model = model.d_model

    head = nn.Linear(d_model, d_model).to(device)

    print(f"Config:     {config_name}")
    print(f"Layout:     {n_latents} latents + {n_patches} patches = {S} tokens")
    print(f"d_model:    {d_model}")
    print(f"Parameters: {model.count_parameters() + sum(p.numel() for p in head.parameters()):,}")
    print()

    # ── Show all available configs ───────────────────────────────────────────
    print("=== Available model configs ===")
    print_config_table(layout=layout, space_mode="encoder")
    print()

    # ── Synthetic data ───────────────────────────────────────────────────────
    B, T = 4, 8
    x = torch.randn(B, T, S, d_model, device=device)
    target = torch.randn(B, T, S, d_model, device=device)

    # ── Training loop ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=3e-4
    )
    n_steps = 500

    print(f"Training for {n_steps} steps  (copy task, MSE loss)")
    print("-" * 50)

    steps: list[int] = []
    losses: list[float] = []
    initial_loss = None

    for step in range(1, n_steps + 1):
        y = head(model(x))
        loss = (y - target).pow(2).mean()

        if step == 1:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps.append(step)
        losses.append(loss.item())

        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d}  loss = {loss.item():.6f}")

    final_loss = loss.item()
    print("-" * 50)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Reduction:    {final_loss / initial_loss:.4f}x")

    assert final_loss < initial_loss * 0.05, (
        f"Copy task failed to converge: {initial_loss:.6f} -> {final_loss:.6f}"
    )
    print("\nCopy task PASSED.")

    # ── Visualization ────────────────────────────────────────────────────────

    # 1) Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(steps, losses, linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Copy Task — Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = OUTPUT_DIR / "copy_task_loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve       -> {loss_path}")

    # 2) Target vs output heatmap (one batch element, one time step)
    model.eval()
    with torch.no_grad():
        y_final = head(model(x))

    tgt_slice = target[0, 0].cpu().numpy()  # (S, D)
    out_slice = y_final[0, 0].cpu().numpy()  # (S, D)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin = min(tgt_slice.min(), out_slice.min())
    vmax = max(tgt_slice.max(), out_slice.max())

    im0 = axes[0].imshow(tgt_slice, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[0].set_title("Target")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Spatial token")

    im1 = axes[1].imshow(out_slice, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title("Model output")
    axes[1].set_xlabel("Dimension")

    diff = out_slice - tgt_slice
    dabs = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, aspect="auto", vmin=-dabs, vmax=dabs, cmap="RdBu_r")
    axes[2].set_title("Difference (output - target)")
    axes[2].set_xlabel("Dimension")

    for im, ax in zip([im0, im1, im2], axes):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Copy Task — Target vs Model Output (batch=0, t=0)", fontsize=12)
    fig.tight_layout()
    cmp_path = OUTPUT_DIR / "copy_task_comparison.png"
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print(f"Saved output comparison -> {cmp_path}")


if __name__ == "__main__":
    main()
