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
from typing import Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("dreamer4")

# Metric key -> TensorBoard prefix mapping
_PREFIX_MAP = {
    "dynamics_loss": "wm",
    "tokenizer_loss": "wm",
    "bc_loss": "agent",
    "reward_pred_loss": "agent",
    "policy_loss": "imagine",
    "value_loss": "imagine",
    "mean_reward": "imagine",
    "mean_value": "imagine",
    "mean_advantage": "imagine",
    "mean_lambda_return": "imagine",
    "episode_return": "env",
    "env_steps": "env",
    "episodes_collected": "env",
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
            parts = [f"step={step}"]
            for key in ("dynamics_loss", "policy_loss", "value_loss",
                        "episode_return", "env_steps"):
                if key in metrics:
                    parts.append(f"{key}={metrics[key]:.4f}")
            logger.info(" | ".join(parts))

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

    def flush(self) -> None:
        """Flush the TensorBoard writer."""
        self._writer.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self._writer.flush()
        self._writer.close()
