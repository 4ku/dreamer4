"""
Clip sampling on top of the unified episode format: the infinite training
stream (a torch ``IterableDataset``), the fixed deterministic validation
clip list, episode-level train/val splitting, and the uint8-batch → float
frames conversion used at the device boundary.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dreamer4.data.base import EpisodeVideoDataset


def eligible_episodes(dataset: EpisodeVideoDataset, seq_len: int,
                      episodes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Indices (from ``episodes`` or all) with at least ``seq_len`` frames."""
    idx = np.arange(len(dataset)) if episodes is None else np.asarray(episodes)
    keep = [i for i in idx if dataset.episode_frames(int(i)) >= seq_len]
    if not keep:
        raise ValueError(f"no episode has >= {seq_len} frames")
    return np.asarray(keep, dtype=np.int64)


def sample_val_clips(dataset: EpisodeVideoDataset, *, seq_len: int,
                     n_clips: int, seed: int,
                     episodes: Optional[Sequence[int]] = None
                     ) -> List[Tuple[int, int]]:
    """
    A FIXED list of (episode, start) pairs for deterministic validation —
    the same clips every eval, every resume, so val curves are comparable.
    """
    rng = np.random.default_rng(seed)
    idx = eligible_episodes(dataset, seq_len, episodes)
    starts = np.array([dataset.episode_frames(int(i)) - seq_len + 1 for i in idx],
                      dtype=np.float64)
    p = starts / starts.sum()
    return [(int(i), int(rng.integers(0, dataset.episode_frames(int(i)) - seq_len + 1)))
            for i in rng.choice(idx, size=n_clips, p=p)]


class ClipStreamDataset(IterableDataset):
    """
    Infinite stream of random fixed-length clips for a ``DataLoader``.

    Episodes are weighted by their number of valid windows so every window in
    the dataset is (approximately) equiprobable. Each worker derives its own
    rng from (seed, worker_id) — safe with ``num_workers > 0``, where video
    decoding parallelizes for free.

    Yields dicts in the unified clip format (see :mod:`dreamer4.data.base`);
    the default collate turns them into uint8 / float32 batch tensors.
    """

    def __init__(self, dataset: EpisodeVideoDataset, *, seq_len: int, seed: int,
                 episodes: Optional[Sequence[int]] = None):
        super().__init__()
        self.dataset = dataset
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.episodes = None if episodes is None else np.asarray(episodes)

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, worker_id])
        idx = eligible_episodes(self.dataset, self.seq_len, self.episodes)
        starts = np.array(
            [self.dataset.episode_frames(int(i)) - self.seq_len + 1 for i in idx],
            dtype=np.float64)
        p = starts / starts.sum()
        while True:
            i = int(rng.choice(idx, p=p))
            s = int(rng.integers(0, self.dataset.episode_frames(i) - self.seq_len + 1))
            yield self.dataset.clip(i, s, self.seq_len)


def split_train_val(n_episodes: int, *, val_frac: float, seed: int,
                    min_val: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Episode-level train/val split (clips never cross the split).

    ``min_val`` floors the validation pool (capped at half the dataset so
    training is never starved) — evals that need stable statistics, like the
    dynamics gate, pass 64. With a single episode both sides point at it
    (degenerate but usable for smoke tests).
    """
    if n_episodes < 2:
        only = np.arange(n_episodes)
        return only, only
    perm = np.random.default_rng(seed).permutation(n_episodes)
    n_val = max(1, min(max(min_val, round(val_frac * n_episodes)),
                       n_episodes // 2))
    return perm[n_val:], perm[:n_val]


def video_batch_to_frames(video: torch.Tensor, device: torch.device
                          ) -> torch.Tensor:
    """(B, T, H, W, C) uint8 batch -> (B, T, C, H, W) float32 in [0, 1]."""
    v = video.to(device=device, non_blocking=True).float() / 255.0
    return v.permute(0, 1, 4, 2, 3).contiguous()
