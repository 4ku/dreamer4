"""
The LeRobot dataset format — the robotics data source for this project.

A LeRobot dataset (https://github.com/huggingface/lerobot) is laid out as:

    root/
      meta/info.json                              # + episodes/tasks/stats
      data/chunk-000/episode_000000.parquet       # state & actions
      videos/chunk-000/<camera>/episode_000000.mp4

The VIDEO tree is always consumed: cameras are the subdirectories under
``videos/chunk-*/`` (feature keys like ``observation.images.top``); episodes
are matched across cameras by chunk + file stem, keeping only episodes
present in every camera.

The PARQUET files are read on demand (``pyarrow``, an optional dependency —
``pip install dreamer4[lerobot]``): ``action`` vectors when the dynamics
trainer asks for actions, and ``observation.state`` as the proprio stream
when opened with ``proprio="auto"``. Row ``t`` of both belongs to frame
``t``; per the unified contract, ``action[t]`` drives the ``t -> t+1``
transition. Values are passed through UNNORMALIZED — normalize per-robot
when real data lands (LeRobot ships stats in ``meta/``).

Multiple cameras are tiled into ONE frame per timestep
(:func:`compose_cameras`): the tokenizer backbone is patch-count-agnostic
(1-D RoPE over patch index), so tiling views needs no model changes.

Videos are decoded on demand with imageio(-ffmpeg) and LRU-cached per
DataLoader worker. Each cache entry is one decoded (camera, episode) clip —
for long/high-res episodes lower ``episode_cache`` and hide decode latency
with more workers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dreamer4.data.base import EpisodeVideoDataset

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".gif")


def compose_cameras(
    cams: Dict[str, np.ndarray],
    order: Sequence[str],
    layout: str = "hstack",
) -> np.ndarray:
    """
    Tile per-camera clips into a single video clip.

    Args:
        cams:   Mapping camera name -> (T, H, W, C) uint8 clip. All cameras
                must share the same shape.
        order:  Camera names in composition order (row-major for "grid").
        layout: "hstack" (side by side), "vstack" (stacked vertically) or
                "grid" (near-square row-major grid, blank-padded).

    Returns:
        (T, H', W', C) uint8 composed clip.
    """
    clips = [cams[name] for name in order]
    if len(clips) == 1:
        return clips[0]
    shapes = {c.shape for c in clips}
    if len(shapes) != 1:
        raise ValueError(f"all cameras must share one shape, got {sorted(shapes)}")
    if layout == "hstack":
        return np.concatenate(clips, axis=2)
    if layout == "vstack":
        return np.concatenate(clips, axis=1)
    if layout == "grid":
        n = len(clips)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        blank = np.zeros_like(clips[0])
        clips = clips + [blank] * (rows * cols - n)
        row_imgs = [
            np.concatenate(clips[r * cols : (r + 1) * cols], axis=2)
            for r in range(rows)
        ]
        return np.concatenate(row_imgs, axis=1)
    raise ValueError(f"unknown camera layout '{layout}' (hstack|vstack|grid)")


class LeRobotVideoDataset(EpisodeVideoDataset):
    """
    Episodes from a LeRobot dataset's video tree (see module docstring).

    Args:
        root:          Dataset root (the directory containing ``meta/`` and
                       ``videos/``).
        cameras:       Restrict/order the camera set (default: all cameras,
                       sorted by name).
        camera_layout: How to tile several cameras ("hstack"|"vstack"|"grid").
        actions:       Attach parquet ``action`` vectors to clips.
        state_as_proprio: Attach parquet ``observation.state`` as the proprio
                       stream.
        episode_cache: Decoded (camera, episode) clips kept in memory.
    """

    def __init__(self, root: Union[str, Path], *,
                 cameras: Optional[Sequence[str]] = None,
                 camera_layout: str = "hstack", actions: bool = False,
                 state_as_proprio: bool = False, episode_cache: int = 8):
        self.root = Path(root)
        self.camera_layout = camera_layout
        self._episodes, self._parquets = self._discover(self.root)
        if not self._episodes:
            raise FileNotFoundError(
                f"no video episodes under {self.root / 'videos'} "
                "(expected videos/chunk-*/<camera>/*.mp4)")
        self._all_cameras = sorted(self._episodes[0].keys())
        self._cameras = list(cameras) if cameras else self._all_cameras
        unknown = set(self._cameras) - set(self._all_cameras)
        if unknown:
            raise ValueError(f"unknown cameras {sorted(unknown)}; "
                             f"available: {self._all_cameras}")
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._cache_order: List[Tuple[str, int]] = []
        self._cache_max = int(episode_cache)
        self._n_frames: Dict[int, int] = {}

        self._actions = bool(actions)
        self._proprio = bool(state_as_proprio)
        self._pq_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._action_dim = self._proprio_dim = None
        if self._actions or self._proprio:
            if any(p is None for p in self._parquets):
                missing = sum(p is None for p in self._parquets)
                raise FileNotFoundError(
                    f"{missing} episode(s) have no matching parquet under "
                    f"{self.root / 'data'} — required for actions/state")
            probe = self._read_parquet(0)
            if self._actions:
                self._action_dim = probe["action"].shape[-1]
            if self._proprio:
                self._proprio_dim = probe["observation.state"].shape[-1]

    @staticmethod
    def _discover(root: Path):
        """
        Match episodes by chunk+stem across camera dirs (and the data/ tree).

        Returns (video_episodes, parquets): per episode a {camera: path} dict
        and the matching ``data/chunk-*/<stem>.parquet`` path (or None).
        """
        per_cam: Dict[str, Dict[str, Path]] = {}
        for chunk in sorted((root / "videos").glob("chunk-*")):
            for cam_dir in sorted(p for p in chunk.iterdir() if p.is_dir()):
                for f in sorted(cam_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                        # key by chunk+stem so chunks never collide
                        per_cam.setdefault(cam_dir.name, {})[f"{chunk.name}/{f.stem}"] = f
        if not per_cam:
            return [], []
        pq_by_stem = {f"{chunk.name}/{f.stem}": f
                      for chunk in sorted((root / "data").glob("chunk-*"))
                      for f in sorted(chunk.glob("*.parquet"))}
        common = sorted(set.intersection(*(set(m) for m in per_cam.values())))
        episodes = [{cam: files[stem] for cam, files in per_cam.items()}
                    for stem in common]
        parquets = [pq_by_stem.get(stem) for stem in common]
        return episodes, parquets

    def _read_parquet(self, i: int) -> Dict[str, np.ndarray]:
        """Episode ``i``'s parquet columns as (T, D) float32 arrays (cached)."""
        cached = self._pq_cache.get(i)
        if cached is not None:
            return cached
        try:
            import pyarrow.parquet as pq  # lazy: optional dependency
        except ImportError as e:
            raise ImportError(
                "reading LeRobot state/actions needs the 'pyarrow' package — "
                "install the extra: pip install 'dreamer4-pytorch[lerobot]' "
                "(video-only use works without it)") from e
        columns = (["action"] if self._actions else []) + \
                  (["observation.state"] if self._proprio else [])
        table = pq.read_table(self._parquets[i], columns=columns)
        out = {name: np.stack(table.column(name).to_pylist()).astype(np.float32)
               for name in columns}
        self._pq_cache[i] = out
        # parquet rows are tiny next to decoded video — keep at least 64
        if len(self._pq_cache) > max(self._cache_max, 64):
            self._pq_cache.pop(next(iter(self._pq_cache)))
        return out

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def camera_names(self) -> List[str]:
        """Cameras in composition order."""
        return self._cameras

    @property
    def action_dim(self) -> "int | None":
        return self._action_dim

    @property
    def proprio_dim(self) -> "int | None":
        return self._proprio_dim

    def episode_frames(self, i: int) -> int:
        if i not in self._n_frames:
            import imageio.v2 as iio
            with iio.get_reader(self._episodes[i][self._cameras[0]]) as reader:
                n = reader.get_length()
                if not (isinstance(n, int) and 0 < n < 10 ** 9):
                    n = reader.count_frames()   # metadata inexact -> count
            self._n_frames[i] = int(n)
        return self._n_frames[i]

    def _decode(self, cam: str, i: int) -> np.ndarray:
        key = (cam, i)
        clip = self._cache.get(key)
        if clip is not None:
            self._cache_order.remove(key)       # refresh recency (LRU)
            self._cache_order.append(key)
            return clip
        import imageio.v2 as iio
        frames = []
        with iio.get_reader(self._episodes[i][cam]) as reader:
            for fr in reader:
                fr = np.asarray(fr)
                if fr.ndim == 2:                # grayscale -> (H, W, 1)
                    fr = fr[..., None]
                elif fr.shape[-1] == 4:         # RGBA -> RGB
                    fr = fr[..., :3]
                frames.append(fr)
        clip = np.stack(frames).astype(np.uint8)
        self._cache[key] = clip
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_max:
            self._cache.pop(self._cache_order.pop(0), None)
        return clip

    def _load_clip(self, i: int, start: int, length: int) -> dict:
        sl = slice(start, start + length)
        cams = {c: self._decode(c, i)[sl] for c in self._cameras}
        out = {"video": compose_cameras(cams, self._cameras, self.camera_layout)}
        if self._actions or self._proprio:
            rows = self._read_parquet(i)
            n_rows = next(iter(rows.values())).shape[0]
            if n_rows < start + length:
                raise ValueError(
                    f"episode {i}: parquet has {n_rows} rows but the video "
                    f"clip needs frames up to {start + length} — corrupt or "
                    "mismatched dataset")
            if self._actions:
                out["actions"] = rows["action"][start : start + length - 1]
            if self._proprio:
                out["proprio"] = rows["observation.state"][sl]
        return out
