"""
Structured logging for Dreamer 4 training.

All training metrics are written to **TensorBoard** as the primary
backend.  A periodic console summary is printed via Python ``logging``.

Usage::

    from dreamer4.logging import setup_logging, MetricsLogger

    setup_logging("runs/exp01/logs")
    ml = MetricsLogger("runs/exp01/tb")

    ml.log_scalars(step=100, metrics={"wm/dynamics_loss": 0.12, ...})
    ml.close()

Then view with::

    tensorboard --logdir runs/exp01/tb
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("dreamer4")

# Metric key -> TensorBoard prefix mapping
_PREFIX_MAP = {
    "dynamics_loss": "wm",
    "tokenizer_loss": "wm",
    "action_encoder_grad_norm": "wm",
    "action_inject_grad_norm": "wm",
    "action_embed_norm": "wm",
    "action_dep_norm": "wm",
    "reward_pred_loss": "agent",
    "replay_policy_loss": "agent",
    "replay_mean_advantage": "agent",
    "replay_advantage_std": "agent",
    "replay_advantage_pos_frac": "agent",
    "replay_policy_grad_norm": "agent",
    "policy_loss": "imagine",
    "value_loss": "imagine",
    "mean_reward": "imagine",
    "mean_value": "imagine",
    "mean_advantage": "imagine",
    "mean_lambda_return": "imagine",
    "policy_entropy": "imagine",
    "policy_grad_norm": "debug",
    "value_grad_norm": "debug",
    "advantage_std": "debug",
    "advantage_pos_frac": "debug",
    "action_mean": "debug",
    "action_std": "debug",
    "kl_to_prior": "debug",
    "log_prob_mean": "debug",
    "episode_return": "env",
    "env_steps": "env",
    "episodes_collected": "env",
    "eval_return_mean": "eval",
    "eval_return_std": "eval",
    "eval_return_min": "eval",
    "eval_return_max": "eval",
    "eval_episode_length": "eval",
}


def _prefixed_key(key: str) -> str:
    """Add a group prefix to a metric key for TensorBoard organization."""
    if "/" in key:
        return key
    prefix = _PREFIX_MAP.get(key)
    if prefix:
        return f"{prefix}/{key}"
    return f"misc/{key}"


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> None:
    """Configure the ``dreamer4`` logger for console (and optional file) output.

    Safe to call multiple times; handlers are only added once.

    Args:
        log_dir: If provided, also write log messages to a file in this dir.
        level:   Logging level (default ``INFO``).
    """
    root = logging.getLogger("dreamer4")
    if root.handlers:
        return
    root.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)


class MetricsLogger:
    """TensorBoard-backed metrics logger with periodic console output.

    All scalar metrics are written to TensorBoard on every call to
    :meth:`log_scalars`.  A one-line console summary is emitted every
    ``console_every`` calls.

    Args:
        log_dir:       TensorBoard log directory.
        console_every: Print a console line every N ``log_scalars`` calls.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        console_every: int = 1,
    ):
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._console_every = console_every
        self._call_count = 0

    @property
    def writer(self) -> SummaryWriter:
        """Direct access to the underlying ``SummaryWriter``."""
        return self._writer

    def log_scalars(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Write all scalar metrics to TensorBoard and optionally console.

        Metrics are automatically prefixed by group (``wm/``, ``agent/``,
        ``imagine/``, ``env/``) based on the key name.

        Args:
            step:    Global training step.
            metrics: Dict of metric name -> scalar value.
        """
        for key, value in metrics.items():
            self._writer.add_scalar(_prefixed_key(key), value, step)

        self._call_count += 1
        if self._call_count % self._console_every == 0:
            self._log_console(step, metrics)

    def _log_console(self, step: int, metrics: dict[str, float]) -> None:
        """Print a compact multi-line summary to the console."""
        def _f(key: str, fmt: str = ".4f") -> str:
            v = metrics.get(key)
            if v is None:
                return ""
            return f"{format(v, fmt)}"

        env_steps = int(metrics.get("env_steps", 0))
        header = f"step {step} | env {env_steps:,} steps"
        ep_ret = metrics.get("episode_return")
        if ep_ret is not None:
            header += f" | ep_return={ep_ret:.1f}"

        wm_parts = []
        for key, label in [("tokenizer_loss", "tok"), ("dynamics_loss", "dyn"),
                           ("reward_pred_loss", "rew")]:
            v = _f(key)
            if v:
                wm_parts.append(f"{label}={v}")

        rl_parts = []
        for key, label in [("policy_loss", "pol"), ("value_loss", "val"),
                           ("policy_entropy", "ent"), ("mean_reward", "R_im"),
                           ("mean_value", "V_im")]:
            v = _f(key)
            if v:
                rl_parts.append(f"{label}={v}")

        dbg_parts = []
        for key, label in [("advantage_std", "adv_std"),
                           ("advantage_pos_frac", "adv_pos"),
                           ("policy_grad_norm", "pi_grad"),
                           ("value_grad_norm", "val_grad"),
                           ("action_mean", "act_mu"),
                           ("action_std", "act_std"),
                           ("kl_to_prior", "kl")]:
            v = _f(key)
            if v:
                dbg_parts.append(f"{label}={v}")

        env_parts = []
        rpl = _f("replay_policy_loss")
        if rpl:
            env_parts.append(f"replay_pol={rpl}")

        lines = [header]
        if wm_parts:
            lines.append("  WM   " + "  ".join(wm_parts))
        if rl_parts:
            lines.append("  RL   " + "  ".join(rl_parts))
        if dbg_parts:
            lines.append("  DBG  " + "  ".join(dbg_parts))
        if env_parts:
            lines.append("  ENV  " + "  ".join(env_parts))

        logger.info("\n".join(lines))

    def log_image(
        self,
        step: int,
        tag: str,
        image: torch.Tensor,
    ) -> None:
        """Write an image to TensorBoard.

        Args:
            step:  Global step.
            tag:   TensorBoard tag (e.g. ``"gen/frames"``).
            image: (C, H, W) or (H, W, C) tensor in [0, 1].
        """
        if image.dim() == 3 and image.shape[-1] in (1, 3, 4):
            image = image.permute(2, 0, 1)
        self._writer.add_image(tag, image.clamp(0, 1), step)

    def log_video(
        self,
        step: int,
        tag: str,
        frames: Sequence[np.ndarray],
        fps: int = 10,
    ) -> None:
        """Write a video to TensorBoard.

        Attempts ``add_video`` first (requires moviepy); falls back to
        writing a GIF via imageio and logging it as an image grid.

        Args:
            step:   Global step.
            tag:    TensorBoard tag (e.g. ``"eval/episode_video"``).
            frames: List of (H, W, 3) uint8 numpy arrays.
            fps:    Frames per second for playback.
        """
        if not frames:
            return
        vid = np.stack(frames, axis=0)  # (T, H, W, 3)
        vid_t = torch.from_numpy(vid).permute(0, 3, 1, 2)  # (T, C, H, W)
        vid_t = vid_t.unsqueeze(0).float() / 255.0  # (1, T, C, H, W)
        try:
            self._writer.add_video(tag, vid_t, step, fps=fps)
        except Exception:
            n_samples = min(16, len(frames))
            indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
            sampled = [
                torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
                for i in indices
            ]
            self.log_images_grid(step, f"{tag}_grid", sampled, nrow=4)

    def log_images_grid(
        self,
        step: int,
        tag: str,
        images: Sequence[torch.Tensor],
        nrow: int = 8,
    ) -> None:
        """Write a grid of images to TensorBoard.

        Args:
            step:   Global step.
            tag:    TensorBoard tag.
            images: List of (C, H, W) tensors in [0, 1].
            nrow:   Number of images per row in the grid.
        """
        from torchvision.utils import make_grid
        grid = make_grid(
            [img.clamp(0, 1) for img in images], nrow=nrow, padding=2,
        )
        self._writer.add_image(tag, grid, step)

    def flush(self) -> None:
        """Flush the TensorBoard writer."""
        self._writer.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self._writer.flush()
        self._writer.close()
