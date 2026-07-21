"""
Data loading for Dreamer 4 training.

One **unified episode format** (:mod:`dreamer4.data.base`) and the storage
formats we actually use, one module each:

- :mod:`dreamer4.data.gridworld` — the gridworld toy testbed (sprite gate,
  derived proprio; needs the companion ``gridworld`` package);
- :mod:`dreamer4.data.lerobot` — LeRobot datasets, the robotics data source
  (multi-camera video tree; parquet ``action`` vectors and
  ``observation.state`` proprio).

:func:`open_video_dataset` (below) turns dataset directories into dataset
objects, detecting the format from directory markers — to support a new
format, add its module and a detection branch there. Trainers consume clips
through :mod:`dreamer4.data.sampling` (infinite train stream + fixed
deterministic val clips) and never see storage details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

from dreamer4.data.base import EpisodeVideoDataset, MergedEpisodeDataset
from dreamer4.data.gridworld import GridworldEpisodeDataset
from dreamer4.data.lerobot import LeRobotVideoDataset, compose_cameras
from dreamer4.data.sampling import (ClipStreamDataset, eligible_episodes,
                                    sample_val_clips, split_train_val,
                                    video_batch_to_frames)

PROPRIO_MODES = ("none", "auto", "player", "player_goal")

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
    auto-detecting the storage format:

    - ``manifest.json`` with episodes  -> gridworld (needs the gridworld pkg);
    - ``meta/`` + ``videos/``          -> LeRobot dataset.

    Args:
        path:          Directory, comma-separated directories, or a list.
        proprio:       "none" | "auto" (dataset-stored proprio: LeRobot
                       ``observation.state``) | "player" | "player_goal"
                       (gridworld-derived).
        actions:       Attach action vectors to clips (dynamics training).
        camera_layout: Multi-camera tiling, LeRobot only ("hstack"|"vstack"|"grid").
        cameras:       Camera order/subset, LeRobot only.
        episode_cache: LeRobot decoded-episode cache size (gridworld manages
                       its own shard cache).
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

        manifest = root / "manifest.json"
        if manifest.exists() and "episodes" in json.loads(manifest.read_text()):
            mode = proprio if proprio != "auto" else "none"
            return GridworldEpisodeDataset(root, proprio=mode, actions=actions)

        if proprio in ("player", "player_goal"):
            raise ValueError(f"proprio='{proprio}' is gridworld-specific, but "
                             f"{root} is not a gridworld dataset")

        if (root / "meta").is_dir() and (root / "videos").is_dir():
            return LeRobotVideoDataset(root, cameras=cameras,
                                       camera_layout=camera_layout,
                                       actions=actions,
                                       state_as_proprio=(proprio == "auto"),
                                       episode_cache=episode_cache)

        raise FileNotFoundError(
            f"{root} is not a recognized dataset: expected a gridworld "
            "manifest.json or a LeRobot dataset (meta/ + videos/)")

    parts = [_open_one(p) for p in paths]
    return parts[0] if len(parts) == 1 else MergedEpisodeDataset(parts)
