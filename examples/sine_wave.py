"""
Sine Wave Prediction — tests temporal dynamics learning.

Each of the S spatial positions carries a sine wave with a distinct
frequency and phase.  The model observes T steps and must predict the
value at T+1 for every position.

    y(s, t) = sin(2*pi*freq_s * t + phase_s)

Because frequencies and phases are fixed across the batch, the model
can memorise the pattern.  The interesting part is that it must learn to
use causal time attention to extrapolate the next value from the history.

Run:
    cd dreamer4_pytorch
    python -m examples.sine_wave
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dreamer4.modality import Modality, TokenLayout
from dreamer4.configs import make_transformer

OUTPUT_DIR = Path(__file__).parent / "outputs"


def _generate_sine_data(
    B: int,
    T: int,
    S: int,
    d_model: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, nn.Module, torch.Tensor]:
    """
    Returns:
        x:        (B, T, S, d_model)  projected sine input
        target:   (B, T, S, d_model)  projected sine target (shifted +1)
        proj:     nn.Linear(1, d_model)  used for both input and target
        raw_vals: (T+1, S) raw scalar sine values before projection
    """
    freqs = torch.linspace(0.5, 3.0, S, device=device)     # (S,)
    phases = torch.linspace(0.0, math.pi, S, device=device) # (S,)

    t_idx = torch.arange(T + 1, dtype=torch.float32, device=device)  # (T+1,)

    # vals: (T+1, S)
    vals = torch.sin(
        2.0 * math.pi * freqs.unsqueeze(0) * t_idx.unsqueeze(1) / T
        + phases.unsqueeze(0)
    )

    raw_vals = vals.clone()

    # Expand to batch and project scalar -> d_model
    vals = vals.unsqueeze(0).expand(B, -1, -1)  # (B, T+1, S)
    vals = vals.unsqueeze(-1)                    # (B, T+1, S, 1)

    proj = nn.Linear(1, d_model, device=device)

    with torch.no_grad():
        x_full = proj(vals)  # (B, T+1, S, d_model)

    x = x_full[:, :-1].clone().detach()        # (B, T, S, d_model)
    target = x_full[:, 1:].clone().detach()     # (B, T, S, d_model)

    return x, target, proj, raw_vals


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Layout & model ───────────────────────────────────────────────────────
    n_latents = 4
    n_patches = 8
    layout = TokenLayout(
        n_latents=n_latents,
        segments=((Modality.IMAGE, n_patches),),
    )
    S = layout.total_tokens()  # 12

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
    print()

    # ── Synthetic data ───────────────────────────────────────────────────────
    B, T = 4, 16
    x, target, proj, raw_vals = _generate_sine_data(B, T, S, d_model, device)

    # ── Training loop ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=3e-4
    )
    n_steps = 500

    print(f"Training for {n_steps} steps  (sine wave prediction, MSE loss)")
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
        f"Sine wave prediction failed to converge: "
        f"{initial_loss:.6f} -> {final_loss:.6f}"
    )
    print("\nSine wave prediction PASSED.")

    # ── Visualization ────────────────────────────────────────────────────────

    # 1) Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(steps, losses, linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Sine Wave Prediction — Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = OUTPUT_DIR / "sine_wave_loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve          -> {loss_path}")

    # 2) Waveform comparison: ground truth vs predicted for 4 spatial positions
    model.eval()
    with torch.no_grad():
        pred_final = head(model(x))  # (B, T, S, D)

    # Project predictions back to scalar space via pseudo-inverse of proj
    # proj: (1,) -> (d_model,), so proj.weight is (d_model, 1)
    w = proj.weight.detach()  # (d_model, 1)
    b = proj.bias.detach()    # (d_model,)

    pred_0 = pred_final[0].cpu()   # (T, S, D) — batch element 0
    target_0 = target[0].cpu()     # (T, S, D)

    # Least-squares projection back to scalar: scalar = (h - b) @ w / (w^T w)
    w_flat = w.squeeze(-1).cpu()  # (D,)
    b_cpu = b.cpu()
    wTw = (w_flat * w_flat).sum()

    pred_scalar = ((pred_0 - b_cpu) @ w_flat) / wTw     # (T, S)
    target_scalar = ((target_0 - b_cpu) @ w_flat) / wTw  # (T, S)

    # Ground-truth raw sine values (shifted by 1 for the target)
    gt_target = raw_vals[1:, :].cpu().numpy()  # (T, S)

    pred_np = pred_scalar.numpy()
    target_np = target_scalar.numpy()

    # Pick 4 evenly spaced spatial positions to plot
    positions = np.linspace(0, S - 1, 4, dtype=int)
    t_axis = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, s_idx in zip(axes.flat, positions):
        ax.plot(t_axis, gt_target[:, s_idx], "o-", label="Ground truth", markersize=4, linewidth=1.5)
        ax.plot(t_axis, pred_np[:, s_idx], "x--", label="Predicted", markersize=5, linewidth=1.2)
        ax.set_title(f"Spatial position s={s_idx}")
        ax.set_ylabel("Sine value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("Time step")

    fig.suptitle("Sine Wave Prediction — Ground Truth vs Model Output", fontsize=13)
    fig.tight_layout()
    wave_path = OUTPUT_DIR / "sine_wave_prediction.png"
    fig.savefig(wave_path, dpi=150)
    plt.close(fig)
    print(f"Saved waveform comparison -> {wave_path}")


if __name__ == "__main__":
    main()
