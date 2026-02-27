"""
Checkpoint save / load / resume for Dreamer 4.

Stores the full training state (model weights, optimizer states, step
counters, config snapshot) so that training can be resumed exactly.

Usage::

    from dreamer4.checkpoint import save_checkpoint, load_checkpoint

    save_checkpoint("ckpt/step_1000.pt", agent, optimizers,
                    train_step=1000, env_steps=50000, config=cfg)

    meta = load_checkpoint("ckpt/step_1000.pt", agent, optimizers)
    # meta == {"train_step": 1000, "env_steps": 50000, ...}
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Union[str, Path],
    agent: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    train_step: int = 0,
    env_steps: int = 0,
    config: Any = None,
    metrics: Optional[dict[str, float]] = None,
) -> Path:
    """Save full training state to a file.

    Args:
        path:       File path for the checkpoint.
        agent:      :class:`Dreamer4Agent` (or any ``nn.Module``).
        optimizers: Dict of named optimizers.
        train_step: Current training iteration.
        env_steps:  Total environment steps collected so far.
        config:     Config object (dataclass) — serialized via ``asdict``.
        metrics:    Optional latest metric snapshot.

    Returns:
        Resolved :class:`Path` where the checkpoint was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dicts": {
            k: opt.state_dict() for k, opt in optimizers.items()
        },
        "train_step": train_step,
        "env_steps": env_steps,
    }
    if config is not None:
        try:
            payload["config"] = asdict(config)
        except TypeError:
            payload["config"] = str(config)
    if metrics is not None:
        payload["metrics"] = metrics

    torch.save(payload, path)
    logger.info("Saved checkpoint to %s (step=%d, env_steps=%d)", path, train_step, env_steps)
    return path


def load_checkpoint(
    path: Union[str, Path],
    agent: torch.nn.Module,
    optimizers: Optional[dict[str, torch.optim.Optimizer]] = None,
    *,
    map_location: Optional[str] = None,
) -> dict[str, Any]:
    """Restore training state from a checkpoint.

    Args:
        path:         Checkpoint file path.
        agent:        Agent whose ``state_dict`` will be loaded.
        optimizers:   Optional dict of optimizers to restore.  Pass ``None``
                      to load only the model weights.
        map_location: ``torch.load`` device mapping (e.g. ``"cpu"``).

    Returns:
        Metadata dict with keys ``train_step``, ``env_steps``,
        and optionally ``config`` and ``metrics``.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)

    agent.load_state_dict(payload["agent_state_dict"])

    if optimizers is not None:
        saved_opts = payload.get("optimizer_state_dicts", {})
        for name, opt in optimizers.items():
            if name in saved_opts:
                opt.load_state_dict(saved_opts[name])

    meta = {
        "train_step": payload.get("train_step", 0),
        "env_steps": payload.get("env_steps", 0),
    }
    if "config" in payload:
        meta["config"] = payload["config"]
    if "metrics" in payload:
        meta["metrics"] = payload["metrics"]

    logger.info(
        "Loaded checkpoint from %s (step=%d, env_steps=%d)",
        path, meta["train_step"], meta["env_steps"],
    )
    return meta


class AutoCheckpoint:
    """Manages periodic checkpoint saving with rolling deletion.

    Saves a checkpoint every ``every`` training steps, keeping only the
    last ``keep`` files to avoid filling the disk.

    Args:
        checkpoint_dir: Directory to store checkpoint files.
        every:          Save interval in training steps.
        keep:           Number of most-recent checkpoints to retain.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        every: int = 10_000,
        keep: int = 3,
    ):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.keep = keep
        self._saved: list[Path] = []

    def step(
        self,
        train_step: int,
        agent: torch.nn.Module,
        optimizers: dict[str, torch.optim.Optimizer],
        **kwargs: Any,
    ) -> Optional[Path]:
        """Conditionally save a checkpoint if the step matches the interval.

        Returns the checkpoint path if saved, else ``None``.
        """
        if train_step == 0 or train_step % self.every != 0:
            return None

        path = self.dir / f"step_{train_step:08d}.pt"
        save_checkpoint(path, agent, optimizers, train_step=train_step, **kwargs)
        self._saved.append(path)

        while len(self._saved) > self.keep:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()
                logger.debug("Removed old checkpoint %s", old)

        return path
