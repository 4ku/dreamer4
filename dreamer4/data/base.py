"""
The unified episode format that every dataset adapter implements.

A dataset is an indexed collection of *episodes* (contiguous recordings).
Trainers only ever ask for fixed-length windows via
:meth:`EpisodeVideoDataset.clip` and always receive the same payload,
regardless of how the data is stored:

    {"video":   (T, H, W, C)   uint8   — one frame per timestep,
     "proprio": (T, D_p)       float32 — optional; scaling is the dataset's
                                         concern (gridworld emits [-1, 1],
                                         LeRobot passes state through raw),
     "actions": (T-1, D_a)     float32 — optional, action VECTORS; a[t]
                                         drives the t -> t+1 transition
                                         (categorical actions arrive one-hot),
     "rewards": (T-1,)         float32 — optional, aligned with actions:
                                         rewards[t] is the reward of the
                                         t -> t+1 transition,
     "terminals": (T-1,)       bool    — optional, aligned with actions:
                                         True iff the t -> t+1 transition
                                         ends the episode (frame t+1 is the
                                         terminal observation)}

Datasets with actions expose :attr:`action_dim`; the dynamics trainer
requires it, the tokenizer trainer never asks. Rewards and terminals ride
along wherever the storage records them (gridworld shards do; LeRobot
demonstrations have no reward concept) — the phase-2/3 trainers require them,
the phase-1 trainers ignore them.

Storage formats are special cases in sibling modules (``gridworld``,
``lerobot``); :func:`dreamer4.data.open_video_dataset` picks one by asking
each adapter. To support a new format, subclass
:class:`EpisodeVideoDataset`, implement ``recognizes`` / ``from_path`` /
``__len__`` / ``episode_frames`` / ``_load_clip``, and add the class to
``dreamer4.data.ADAPTERS`` — nothing in the trainers changes. The optional
hooks below (``episode_meta``, ``bc_weight``, ``env_spec``,
``continues_from_reward``, ``proprio_from_info``) are what the phase-2/3
trainers ask an adapter for; each has a safe default, so an offline dataset
simply trains without the parts that need a live environment.

Domain-specific evaluation plugs in the same way: :meth:`eval_metrics` and
:meth:`gate` let a dataset judge reconstructions by what MATTERS in its
domain (e.g. gridworld's sprite positions), since pixel losses alone can
hide exactly the content the downstream world model needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class EpisodeVideoDataset:
    """Base class implementing the unified format (see module docstring)."""

    #: Proprio modes this adapter DERIVES from its own storage, beyond the
    #: generic "none"/"auto" that every adapter understands. Declaring them
    #: here keeps a domain's vocabulary out of the shared data layer and out
    #: of the training configs: nothing above this class should enumerate
    #: another domain's field names.
    PROPRIO_MODES: Tuple[str, ...] = ()

    # -- to implement ------------------------------------------------------

    @classmethod
    def recognizes(cls, root: Path) -> bool:
        """Whether ``root`` is stored in this adapter's format."""
        raise NotImplementedError

    @classmethod
    def from_path(cls, root: Path, *, proprio: str = "none",
                  actions: bool = False, **kwargs) -> "EpisodeVideoDataset":
        """Build from a directory, translating the generic open() arguments
        into this adapter's own constructor. Unknown kwargs are ignored so one
        caller can serve adapters with different options."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of episodes."""
        raise NotImplementedError

    def episode_frames(self, i: int) -> int:
        """Number of frames in episode ``i``."""
        raise NotImplementedError

    def _load_clip(self, i: int, start: int, length: int) -> dict:
        """
        Return ``{"video": (T,H,W,C) uint8, "proprio"?: (T,D_p) float32}``.
        Bounds are already validated by :meth:`clip`.
        """
        raise NotImplementedError

    # -- shared machinery --------------------------------------------------

    def clip(self, i: int, start: int, length: int) -> dict:
        """Fixed-length window of episode ``i`` in the unified format."""
        n = self.episode_frames(i)
        if not 0 <= start <= n - length:
            raise IndexError(f"clip [{start}, {start + length}) outside episode "
                             f"{i} with {n} frames")
        return self._load_clip(i, start, length)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """(H, W, C) of one frame (probes the first episode)."""
        v = self.clip(0, 0, 1)["video"]
        return tuple(v.shape[1:])

    @property
    def proprio_dim(self) -> Optional[int]:
        """Per-frame proprio vector size, or None if the dataset has none."""
        return None

    @property
    def action_dim(self) -> Optional[int]:
        """Per-step action vector size, or None if clips carry no actions."""
        return None

    def episode_meta(self, i: int) -> Dict:
        """
        Optional per-episode collector metadata (data-quality signals like
        gridworld's ``noisiness``), empty when the storage records none. The
        phase-2 trainer reports statistics from it; deciding what may be
        imitated is :meth:`bc_weight`'s job.
        """
        return {}

    def bc_weight(self, i: int) -> float:
        """
        How much episode ``i`` should be IMITATED: 1.0 = clone it, 0.0 = use
        it for the other heads but never for behavior cloning.

        Defaults to 1.0 — "every recording is a demonstration", which is true
        of teleop datasets. A collector that deliberately records mixed
        quality (gridworld dials expert->random) overrides this; the criterion
        is the DATASET's, so no trainer has to know what "noisiness" means.
        Phase 2 clones the episodes whose weight is > 0.
        """
        return 1.0

    # -- optional live-environment hooks (online agent evaluation) ----------
    #
    # A dataset that was recorded from a simulator can say how to reopen that
    # simulator, so a trained policy can be rolled in it. Offline-only data
    # (LeRobot demonstrations) leaves these unimplemented and simply has no
    # online eval.

    def env_spec(self) -> Optional[Dict]:
        """
        ``{"id": <gymnasium id>, "kwargs": {...}}`` for the environment this
        data was recorded from, or None if there is no such environment.
        """
        return None

    def continues_from_reward(self, reward_pred):
        """
        The domain's rule for spotting a terminal frame from a PREDICTED
        reward: ``(B, T) -> (B, T)`` float (1 = keep going, 0 = terminal), or
        None when this domain has no such rule.

        Phase 3 needs it because a dream has no recorded terminals — the
        reward head is the only signal available. Whether it is usable is a
        property of the ENVIRONMENT, not of the model: it holds when
        "rewarding" and "terminal" are the same event. Datasets where they
        differ (a milestone reward mid-episode, an episode that ends in
        failure with no reward) return None and need a real terminal
        predictor instead. Phase 2 measures the rule on held-out data
        (``term_f1``) before phase 3 leans on it.
        """
        return None

    def proprio_from_info(self, info: Dict) -> Optional[np.ndarray]:
        """
        One step's proprio ``(D_p,) float32`` read from a live env's ``info``
        dict — the exact state, never inferred from the render. This is the
        ONLY proprio source the online policy uses; a proprio model cannot be
        evaluated in an env that does not report it.

        It must produce the same convention the training clips carry — same
        fields in the same order, same scaling — or the policy sees a
        different input online than it was trained on.
        """
        return None

    # -- optional domain-specific evaluation -------------------------------

    def eval_metrics(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """
        Extra validation metrics from (N, T, H, W, C) float [0,1] frames.

        Pixel error can hide what matters (e.g. a 1-px sprite on a static
        background); domain adapters override this to measure it directly.
        """
        return {}

    def gate(self, metrics: Dict[str, float]) -> Optional[bool]:
        """Deployment go/no-go from averaged val metrics; None = no gate."""
        return None


class MergedEpisodeDataset(EpisodeVideoDataset):
    """
    Concatenation of several episode datasets — for training on several
    collections of the same environment at once
    (:func:`dreamer4.data.open_video_dataset` builds one from a
    comma-separated path).

    All parts must agree on frame shape, proprio_dim and action_dim; eval
    hooks are taken from the FIRST part (parts are assumed to share a domain).
    """

    def __init__(self, parts: Sequence[EpisodeVideoDataset]):
        if not parts:
            raise ValueError("MergedEpisodeDataset needs at least one part")
        self.parts = list(parts)
        shapes = {p.frame_shape for p in self.parts}
        if len(shapes) != 1:
            raise ValueError(f"parts disagree on frame shape: {sorted(shapes)}")
        dims = {p.proprio_dim for p in self.parts}
        if len(dims) != 1:
            raise ValueError(f"parts disagree on proprio_dim: {dims}")
        adims = {p.action_dim for p in self.parts}
        if len(adims) != 1:
            raise ValueError(f"parts disagree on action_dim: {adims}")
        self._offsets = np.cumsum([0] + [len(p) for p in self.parts])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, i: int) -> Tuple[EpisodeVideoDataset, int]:
        part = int(np.searchsorted(self._offsets, i, side="right") - 1)
        return self.parts[part], i - int(self._offsets[part])

    def episode_frames(self, i: int) -> int:
        ds, j = self._locate(i)
        return ds.episode_frames(j)

    def _load_clip(self, i: int, start: int, length: int) -> dict:
        ds, j = self._locate(i)
        return ds.clip(j, start, length)

    def episode_meta(self, i: int) -> Dict:
        ds, j = self._locate(i)
        return ds.episode_meta(j)

    def bc_weight(self, i: int) -> float:
        ds, j = self._locate(i)
        return ds.bc_weight(j)

    def env_spec(self) -> Optional[Dict]:
        return self.parts[0].env_spec()

    def proprio_from_info(self, info: Dict) -> Optional[np.ndarray]:
        return self.parts[0].proprio_from_info(info)

    def continues_from_reward(self, reward_pred):
        return self.parts[0].continues_from_reward(reward_pred)

    @property
    def proprio_dim(self) -> Optional[int]:
        return self.parts[0].proprio_dim

    @property
    def action_dim(self) -> Optional[int]:
        return self.parts[0].action_dim

    def eval_metrics(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        return self.parts[0].eval_metrics(gt, pred)

    def gate(self, metrics: Dict[str, float]) -> Optional[bool]:
        return self.parts[0].gate(metrics)
