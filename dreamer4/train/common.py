"""
Small shared training utilities: seeding, LR warmup, weight EMA, weighted
loss combination with optional RMS normalization, and run logging.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dreamer4.utils import RMSLossNormalizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)


def atomic_torch_save(path: Path, payload: dict) -> None:
    """Write-then-rename so a crash mid-save never leaves a torn checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def warmup_lr(base_lr: float, step: int, warmup: int) -> float:
    """Linear warmup to ``base_lr`` over ``warmup`` steps, then constant."""
    if warmup <= 0 or step >= warmup:
        return base_lr
    return base_lr * (step + 1) / warmup


class EMA:
    """
    Exponential moving average of model weights.

    The EMA ("shadow") weights are what validation and final artifacts use —
    they smooth out reconstruction artifacts of the raw trajectory. Only
    TRAINABLE floating parameters are averaged; frozen parameters and buffers
    track the live value verbatim — so e.g. a frozen encoder stays
    bit-identical through the EMA instead of drifting by float rounding.
    Construct AFTER freezing parameters.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}
        self._averaged = {k for k, p in model.named_parameters()
                          if p.requires_grad and p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self._averaged:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        # Checkpoints are loaded on CPU; keep each shadow tensor on its
        # current (model) device or resuming a GPU run would crash on the
        # first cross-device in-place update.
        self.shadow = {
            k: v.detach().clone().to(self.shadow[k].device)
            if k in self.shadow else v.detach().clone()
            for k, v in state.items()
        }


class swap_in_weights:
    """
    Context manager: temporarily load ``weights`` into ``model`` (e.g. the
    EMA shadow for validation), restoring the originals on exit.
    """

    def __init__(self, model: nn.Module, weights: Optional[Dict[str, torch.Tensor]]):
        self.model = model
        self.weights = weights
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    def __enter__(self):
        if self.weights is not None:
            self._backup = {k: v.detach().clone()
                            for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.weights)
        return self.model

    def __exit__(self, *exc):
        if self._backup is not None:
            self.model.load_state_dict(self._backup)
        return False


class LossCombiner:
    """
    Weighted sum of named loss terms, optionally RMS-normalized.

    With normalization on (paper, end of sec. 3), every term is divided by a
    running RMS estimate before weighting, so the weights express RELATIVE
    importances independent of raw magnitudes. ``floor_frac`` keeps a term
    that has collapsed far below its peak from being amplified back to O(1)
    forever (see :class:`dreamer4.utils.RMSLossNormalizer`).

    Args:
        weights:    Ordered mapping term name -> relative weight. The order
                    fixes normalizer slots, so keep it stable across resume.
        normalize:  Enable RMS normalization.
        decay:      RMS EMA decay.
        floor_frac: Divisor floor as a fraction of each term's peak RMS.
    """

    def __init__(self, weights: Dict[str, float], *, normalize: bool,
                 decay: float = 0.99, floor_frac: float = 0.2,
                 device: Optional[torch.device] = None):
        self.weights = dict(weights)
        self.slots = {name: i for i, name in enumerate(self.weights)}
        self.normalizer = None
        if normalize:
            self.normalizer = RMSLossNormalizer(
                n_losses=len(self.weights), decay=decay, floor_frac=floor_frac)
            if device is not None:
                self.normalizer = self.normalizer.to(device)

    def __call__(self, terms: Dict[str, Optional[torch.Tensor]]
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combine raw loss tensors into the total training loss.

        Returns (total, {name: raw float value}) — raw values are logged;
        normalization only affects the total.
        """
        unknown = set(terms) - set(self.slots)
        if unknown:
            raise KeyError(f"unknown loss term(s) {sorted(unknown)}")
        total, raw = None, {}
        for name, value in terms.items():
            if value is None:
                continue
            raw[name] = float(value.detach())
            weight = self.weights[name]
            if weight == 0.0:
                continue
            if self.normalizer is not None:
                idx = self.slots[name]
                self.normalizer.update(idx, value)
                value = self.normalizer.normalize(idx, value)
            term = weight * value
            total = term if total is None else total + term
        if total is None:
            raise ValueError("no active loss terms — check loss weights")
        return total, raw

    def state_dict(self):
        return None if self.normalizer is None else self.normalizer.state_dict()

    def load_state_dict(self, state) -> None:
        if self.normalizer is not None and state is not None:
            self.normalizer.load_state_dict(state)


def setup_run_logging(out_dir: Path, name: str = "dreamer4.train") -> logging.Logger:
    """
    Console + ``<out>/train.log`` logger for a training run.

    Safe to call once per run in the same process (sweeps, fine-tunes): the
    file handler is re-pointed at the new run's ``train.log`` each time.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = (out_dir / "train.log").resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename) == log_path:
                return logger               # already pointing at this run
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger
