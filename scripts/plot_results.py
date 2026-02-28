#!/usr/bin/env python3
"""
Dreamer 4 — Post-training plot generation from TensorBoard logs.

Reads TensorBoard event files and produces matplotlib plots:
  - Episode return over env steps
  - Eval return over training steps
  - Loss curves (dynamics, tokenizer, policy, value, BC, reward_pred)

Usage:
    cd /mnt/dsk1/dreamer4
    source .venv/bin/activate
    python scripts/plot_results.py --logdir runs/cartpole_balance/logs

    # Specify output directory
    python scripts/plot_results.py --logdir runs/cartpole_balance/logs \
        --outdir runs/cartpole_balance/plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_tb_scalars(log_dir: str) -> dict[str, tuple[list[int], list[float]]]:
    """Read all scalar events from TensorBoard logs.

    Returns:
        Dict mapping tag -> (steps, values).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()

    data: dict[str, tuple[list[int], list[float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = (steps, values)

    return data


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_scalar(
    ax: plt.Axes,
    steps: list[int],
    values: list[float],
    label: str,
    color: str | None = None,
    smooth_window: int = 10,
) -> None:
    """Plot a scalar with optional smoothing."""
    ax.plot(steps, values, alpha=0.2, color=color)
    smoothed = smooth(values, window=smooth_window)
    if len(smoothed) > 0:
        offset = len(values) - len(smoothed)
        ax.plot(steps[offset:], smoothed, label=label, color=color, linewidth=2)
    ax.set_xlabel("Training step")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def generate_plots(data: dict, outdir: Path) -> None:
    """Generate and save all plots."""
    outdir.mkdir(parents=True, exist_ok=True)

    # -- 1. Episode return over env steps --
    if "env/episode_return" in data:
        fig, ax = plt.subplots(figsize=(10, 5))
        steps, vals = data["env/episode_return"]
        plot_scalar(ax, steps, vals, "Episode Return", color="C0")
        ax.set_ylabel("Return")
        ax.set_title("Training Episode Return")
        fig.tight_layout()
        fig.savefig(outdir / "episode_return.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {outdir / 'episode_return.png'}")

    # -- 2. Eval return --
    if "eval/eval_return_mean" in data:
        fig, ax = plt.subplots(figsize=(10, 5))
        steps, means = data["eval/eval_return_mean"]
        ax.plot(steps, means, "o-", color="C1", linewidth=2, markersize=6,
                label="Eval Return (mean)")
        if "eval/eval_return_std" in data:
            _, stds = data["eval/eval_return_std"]
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.fill_between(
                steps,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.2, color="C1",
            )
        ax.set_xlabel("Training step")
        ax.set_ylabel("Return")
        ax.set_title("Evaluation Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "eval_return.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {outdir / 'eval_return.png'}")

    # -- 3. Loss curves --
    loss_tags = {
        "wm/dynamics_loss": ("Dynamics Loss", "C2"),
        "wm/tokenizer_loss": ("Tokenizer Loss", "C3"),
        "imagine/policy_loss": ("Policy Loss", "C4"),
        "imagine/value_loss": ("Value Loss", "C5"),
        "agent/bc_loss": ("BC Loss", "C6"),
        "agent/reward_pred_loss": ("Reward Pred Loss", "C7"),
    }

    available_losses = {k: v for k, v in loss_tags.items() if k in data}
    if available_losses:
        n = len(available_losses)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (tag, (label, color)) in enumerate(available_losses.items()):
            ax = axes[idx]
            steps, vals = data[tag]
            plot_scalar(ax, steps, vals, label, color=color, smooth_window=5)
            ax.set_ylabel("Loss")
            ax.set_title(label)

        for idx in range(len(available_losses), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Loss Curves", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / "losses.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outdir / 'losses.png'}")

    # -- 4. Combined overview --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "env/episode_return" in data:
        steps, vals = data["env/episode_return"]
        plot_scalar(axes[0, 0], steps, vals, "Episode Return", color="C0")
        axes[0, 0].set_title("Training Episode Return")
        axes[0, 0].set_ylabel("Return")

    if "eval/eval_return_mean" in data:
        steps, means = data["eval/eval_return_mean"]
        axes[0, 1].plot(steps, means, "o-", color="C1", linewidth=2, label="Eval Mean")
        axes[0, 1].set_title("Evaluation Return")
        axes[0, 1].set_ylabel("Return")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    if "wm/dynamics_loss" in data:
        steps, vals = data["wm/dynamics_loss"]
        plot_scalar(axes[1, 0], steps, vals, "Dynamics Loss", color="C2")
        axes[1, 0].set_title("Dynamics Loss")
        axes[1, 0].set_ylabel("Loss")

    if "imagine/policy_loss" in data:
        steps, vals = data["imagine/policy_loss"]
        plot_scalar(axes[1, 1], steps, vals, "Policy Loss", color="C4")
        axes[1, 1].set_title("Policy Loss")
        axes[1, 1].set_ylabel("Loss")

    for ax in axes.flatten():
        ax.set_xlabel("Training step")

    fig.suptitle("Dreamer 4 — Training Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "overview.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'overview.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot Dreamer 4 training results")
    parser.add_argument(
        "--logdir", type=str, required=True,
        help="TensorBoard log directory (e.g. runs/cartpole_balance/logs)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for plots (default: <logdir>/../plots)",
    )
    args = parser.parse_args()

    log_dir = Path(args.logdir)
    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        out_dir = log_dir.parent / "plots"

    print(f"Reading TensorBoard logs from: {log_dir}")
    data = read_tb_scalars(str(log_dir))
    print(f"Found {len(data)} scalar tags")

    if not data:
        print("No data found. Check the log directory.")
        return

    for tag in sorted(data.keys()):
        steps, vals = data[tag]
        print(f"  {tag}: {len(vals)} points, range [{min(vals):.4f}, {max(vals):.4f}]")

    print(f"\nGenerating plots to: {out_dir}")
    generate_plots(data, out_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
