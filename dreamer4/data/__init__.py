"""
Dataset loading for Dreamer 4: one unified episode format and one module per
storage format.

:func:`open_video_dataset` turns dataset directories into dataset objects by
asking every adapter in :data:`ADAPTERS` whether it recognises the layout
(``recognizes`` / ``from_path`` on the adapter itself). Supporting a new
format means writing its module and registering it there; this function never
learns about it.

The adapters shipped here:

- :mod:`dreamer4.data.gridworld` — the gridworld toy testbed (sprite gate,
  derived proprio; needs the companion ``gridworld`` package);
- :mod:`dreamer4.data.lerobot` — LeRobot datasets, the robotics data source
  (multi-camera video tree; parquet ``action`` vectors and
  ``observation.state`` proprio).

Trainers consume clips through :mod:`dreamer4.data.sampling` (infinite train
stream + fixed deterministic val clips) and never see storage details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from dreamer4.data.base import EpisodeVideoDataset, MergedEpisodeDataset
from dreamer4.data.gridworld import GridworldEpisodeDataset
from dreamer4.data.lerobot import LeRobotVideoDataset, compose_cameras
from dreamer4.data.sampling import (ClipStreamDataset, eligible_episodes,
                                    sample_val_clips, split_train_val,
                                    video_batch_to_frames)

#: Modes every adapter understands: no proprio at all, or whatever the
#: dataset itself stores. Anything else is an adapter's own vocabulary and is
#: declared by that adapter, never listed here.
GENERIC_PROPRIO_MODES = ("none", "auto")

#: Dataset adapters in dispatch order. Each recognises its own storage format,
#: declares the proprio modes it derives and constructs itself, so this module
#: needs no knowledge of any particular environment or robot.
ADAPTERS = (GridworldEpisodeDataset, LeRobotVideoDataset)

#: Every mode accepted by open_video_dataset -- generic plus whatever the
#: registered adapters contribute. Derived, not hand-maintained.
PROPRIO_MODES = GENERIC_PROPRIO_MODES + tuple(
    m for a in ADAPTERS for m in a.PROPRIO_MODES)

__all__ = [
    "ClipStreamDataset",
    "EpisodeVideoDataset",
    "GridworldEpisodeDataset",
    "LeRobotVideoDataset",
    "MergedEpisodeDataset",
    "PROPRIO_MODES",
    "compose_cameras",
    "eligible_episodes",
    "open_video_dataset",
    "sample_val_clips",
    "split_train_val",
    "video_batch_to_frames",
]


def open_video_dataset(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    proprio: str = "none",
    actions: bool = False,
    camera_layout: str = "hstack",
    cameras: Optional[Sequence[str]] = None,
    episode_cache: int = 64,
) -> EpisodeVideoDataset:
    """
    Open a dataset directory (or several, comma-separated or as a list),
    auto-detecting the storage format by asking each adapter in
    :data:`ADAPTERS` whether it recognises the layout. Adding a data source
    means adding an adapter, not editing this function.

    Args:
        path:          Directory, comma-separated directories, or a list.
        proprio:       "none", "auto" (the adapter's default state stream), or
                       any mode declared by the adapter that claims the
                       directory (see :data:`PROPRIO_MODES`).
        actions:       Attach action vectors to clips (dynamics training).
        camera_layout: Multi-camera tiling, LeRobot only ("hstack"|"vstack"|"grid").
        cameras:       Camera order/subset, LeRobot only.
        episode_cache: LeRobot decoded-episode cache size (gridworld manages
                       its own shard cache).

    Returns:
        One :class:`EpisodeVideoDataset`; several paths are concatenated into
        a :class:`MergedEpisodeDataset`.
    """
    if proprio not in PROPRIO_MODES:
        raise ValueError(f"proprio must be one of {PROPRIO_MODES}, got '{proprio}'")
    if isinstance(path, (str, Path)):
        paths = [p.strip() for p in str(path).split(",") if p.strip()]
    else:
        paths = [str(p) for p in path]
    if not paths:
        raise ValueError("no dataset path given")

    def _open_one(p: str) -> EpisodeVideoDataset:
        root = Path(p)
        if not root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {root}")

        for adapter in ADAPTERS:
            if adapter.recognizes(root):
                break
        else:
            raise FileNotFoundError(
                f"{root} is not a recognized dataset: none of "
                f"{[a.__name__ for a in ADAPTERS]} recognises its layout")

        # A mode one adapter derives is meaningless to another, so validate
        # against the adapter that actually claimed this directory.
        if proprio not in GENERIC_PROPRIO_MODES and \
                proprio not in adapter.PROPRIO_MODES:
            raise ValueError(
                f"proprio='{proprio}' is not derived by {adapter.__name__} "
                f"(it offers {adapter.PROPRIO_MODES or '()'}); modes every "
                f"adapter understands are {GENERIC_PROPRIO_MODES}")

        return adapter.from_path(root, proprio=proprio, actions=actions,
                                 cameras=cameras, camera_layout=camera_layout,
                                 episode_cache=episode_cache)

    parts = [_open_one(p) for p in paths]
    return parts[0] if len(parts) == 1 else MergedEpisodeDataset(parts)
