"""
Pattern Sequence Continuation — tests if causal attention learns structure.

We create K distinct "pattern" vectors and build sequences that cycle
through them:  A-B-C-D-A-B-C-D-...  The model sees T steps and must
predict which pattern comes next at each position.

Unlike predicting random noise, this task has genuine temporal structure:
the model must recognise where it is in the repeating cycle and output
the correct next pattern.  This is only possible if causal time attention
works properly — the model needs to look at the recent history to decide
which pattern follows.

After training we visualise:
  1. Loss curve
  2. A similarity matrix showing, for every time step, how close the
     model's prediction is to each of the K pattern templates.
     The diagonal of the "correct-pattern" row should light up.

Run:
    cd dreamer4_pytorch
    python -m examples.next_token_prediction
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer4.modality import Modality, TokenLayout
from dreamer4.configs import make_transformer

OUTPUT_DIR = Path(__file__).parent / "outputs"

K_PATTERNS = 4  # number of distinct patterns in the cycle


def _build_pattern_data(
    B: int, T: int, S: int, d_model: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build repeating-pattern sequences.

    Returns:
        x:        (B, T, S, D)  input sequence
        target:   (B, T, S, D)  target = next pattern at each step
        patterns: (K, D)        the K pattern templates
    """
    torch.manual_seed(0)
    patterns = F.normalize(torch.randn(K_PATTERNS, d_model, device=device), dim=-1)

    # Build the cycle of pattern indices for T+1 steps
    # Each batch element uses a different starting offset for variety
    offsets = torch.arange(B, device=device) % K_PATTERNS  # (B,)
    t_idx = torch.arange(T + 1, device=device)             # (T+1,)
    pat_ids = (offsets.unsqueeze(1) + t_idx.unsqueeze(0)) % K_PATTERNS  # (B, T+1)

    # Map indices to pattern vectors and broadcast across spatial dimension
    full_seq = patterns[pat_ids]                       # (B, T+1, D)
    full_seq = full_seq.unsqueeze(2).expand(-1, -1, S, -1)  # (B, T+1, S, D)

    x = full_seq[:, :-1].clone()        # (B, T, S, D)
    target = full_seq[:, 1:].clone()    # (B, T, S, D)

    return x, target, patterns


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Layout & model ───────────────────────────────────────────────────────
    n_latents = 4
    n_patches = 12
    layout = TokenLayout(
        n_latents=n_latents,
        segments=((Modality.IMAGE, n_patches),),
    )
    S = layout.total_tokens()

    config_name = "tiny"
    model = make_transformer(
        config_name, layout, "encoder",
        use_qk_norm=False, logit_cap=None, dropout=0.0,
        time_every=1,
    ).to(device)
    d_model = model.d_model

    head = nn.Linear(d_model, d_model).to(device)

    total_params = model.count_parameters() + sum(p.numel() for p in head.parameters())
    print(f"Config:     {config_name} (time_every=1)")
    print(f"Layout:     {n_latents} latents + {n_patches} patches = {S} tokens")
    print(f"d_model:    {d_model}")
    print(f"Parameters: {total_params:,}")
    print(f"Patterns:   {K_PATTERNS} distinct vectors cycling A-B-C-D-A-B-...")
    print()

    # ── Synthetic data ───────────────────────────────────────────────────────
    B, T = 4, 16
    x, target, patterns = _build_pattern_data(B, T, S, d_model, device)

    # ── Training loop ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=3e-4
    )
    n_steps = 500

    print(f"Training for {n_steps} steps  (pattern sequence, MSE loss)")
    print("-" * 55)

    steps: list[int] = []
    losses: list[float] = []
    initial_loss = None

    for step in range(1, n_steps + 1):
        pred = head(model(x))
        loss = (pred - target).pow(2).mean()

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
    print("-" * 55)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Reduction:    {final_loss / initial_loss:.4f}x")

    assert final_loss < initial_loss * 0.05, (
        f"Pattern sequence failed to converge: "
        f"{initial_loss:.6f} -> {final_loss:.6f}"
    )
    print("\nPattern sequence PASSED.")

    # ── Visualization ────────────────────────────────────────────────────────

    # 1) Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(steps, losses, linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Pattern Sequence — Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = OUTPUT_DIR / "pattern_seq_loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve           -> {loss_path}")

    # 2) Similarity matrix: prediction vs each pattern template
    #    For batch=0, spatial=0, compute cosine similarity of the model's
    #    prediction at each time step to each of the K pattern templates.
    model.eval()
    with torch.no_grad():
        pred_final = head(model(x))

    pred_b0 = pred_final[0, :, 0, :]  # (T, D) — batch 0, spatial 0
    pred_norm = F.normalize(pred_b0, dim=-1)       # (T, D)
    pat_norm = F.normalize(patterns, dim=-1)        # (K, D)
    sim = (pred_norm @ pat_norm.T).cpu().numpy()    # (T, K)

    # Ground-truth pattern index at each target step
    offsets = torch.arange(B, device=device) % K_PATTERNS
    t_idx = torch.arange(T + 1, device=device)
    pat_ids = (offsets.unsqueeze(1) + t_idx.unsqueeze(0)) % K_PATTERNS
    gt_ids = pat_ids[0, 1:].cpu().numpy()  # (T,) — batch 0, target indices

    pat_labels = [chr(ord("A") + i) for i in range(K_PATTERNS)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(sim.T, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Pattern template")
    ax.set_yticks(range(K_PATTERNS), pat_labels)
    ax.set_xticks(range(T))
    ax.set_xticklabels(range(1, T + 1))

    for t in range(T):
        ax.add_patch(plt.Rectangle(
            (t - 0.5, gt_ids[t] - 0.5), 1, 1,
            linewidth=2, edgecolor="black", facecolor="none",
        ))

    ax.set_title("Cosine Similarity: Prediction vs Pattern Templates\n"
                 "(black boxes = correct pattern)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Right: accuracy bar — fraction of steps where argmax matches ground truth
    pred_ids = sim.argmax(axis=1)  # (T,)
    correct = (pred_ids == gt_ids)
    accuracy = correct.mean()

    ax2 = axes[1]
    colors = ["#4CAF50" if c else "#F44336" for c in correct]
    ax2.barh(range(T), [1] * T, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(T))
    ax2.set_yticklabels([f"t={t+1}" for t in range(T)])
    ax2.set_xlim(0, 1.5)
    ax2.set_xticks([])
    ax2.invert_yaxis()
    ax2.set_title(f"Correct? ({accuracy:.0%})")

    for t in range(T):
        label = f"{pat_labels[pred_ids[t]]} ({'ok' if correct[t] else pat_labels[gt_ids[t]]})"
        ax2.text(1.05, t, label, va="center", fontsize=8)

    fig.suptitle("Pattern Sequence — Does the Model Learn the Cycle?", fontsize=13)
    fig.tight_layout()
    sim_path = OUTPUT_DIR / "pattern_seq_similarity.png"
    fig.savefig(sim_path, dpi=150)
    plt.close(fig)
    print(f"Saved similarity heatmap   -> {sim_path}")


if __name__ == "__main__":
    main()
