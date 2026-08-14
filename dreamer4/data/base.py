"""
The unified episode format that every dataset loader implements.

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
requires it, the tokenizer trainer never asks. Rewards/terminals ride along
wherever the storage records them (gridworld shards do; LeRobot demos have
no reward concept) — the agent trainer requires them, phase-1 trainers
ignore them.

Storage formats are special cases in sibling modules (``gridworld``,
``lerobot``); :func:`dreamer4.data.open_video_dataset` picks one by looking
at the directory. To support a new format, subclass
:class:`EpisodeVideoDataset`, implement ``__len__`` / ``episode_frames`` /
``_load_clip``, and add a detection branch to ``open_video_dataset`` (in the
package ``__init__``) — nothing in the trainers changes.

Domain-specific evaluation also plugs in here: :meth:`eval_metrics` and
:meth:`gate` let a dataset judge reconstructions by what MATTERS in its
domain (e.g. gridworld's sprite positions), since pixel losses alone can
hide exactly the content the downstream world model needs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class EpisodeVideoDataset:
    """Base class implementing the unified format (see module docstring)."""

    # -- to implement ------------------------------------------------------

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
        gridworld's ``noisiness``). Empty when the storage records none;
        the agent trainer uses it to filter BC data, nothing else asks.
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
        reward: ``(B,T) -> (B,T) float`` (1 = keep going, 0 = terminal), or
        None when this domain has no such rule.

        Phase 3 needs it because a dream has no recorded terminals — the
        reward head is the only signal available. Whether it is usable is a
        property of the ENVIRONMENT, not of the model: it holds when
        "rewarding" and "terminal" are the same event. Datasets where they
        differ (a milestone reward mid-episode, an episode that ends in
        failure with no reward) return None and need a real terminal
        predictor instead. The trainer's ``term_f1`` gate measures this rule
        on held-out data before phase 3 leans on it.
        """
        return None

    def proprio_from_info(self, info: Dict) -> Optional[np.ndarray]:
        """
        One step's proprio ``(D_p,) float32`` read from a live env's ``info``
        dict — the exact state, never inferred from the render. This is the
        ONLY proprio source the online policy uses; a proprio model cannot be
        evaluated in an env that does not report it.

        It must produce the same convention the training clips carry — for
        gridworld both come from the positions the collector recorded from
        the env, scaled the same way (a test pins them together).
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
    Concatenation of several episode datasets — for training on multiple
    collections at once (e.g. plain + sticky gridworld, as the dynamics
    recipe does with its comma-separated data flag).

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
