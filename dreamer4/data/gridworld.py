"""
The gridworld toy environment — our tokenizer/dynamics testbed.

Adapts the sharded dataset written by ``gridworld-collect`` (``manifest.json``
+ ``shard_*.npz``) to the unified format. Requires the companion
``gridworld`` package — imported lazily, it is NOT a dependency of dreamer4.

All gridworld DOMAIN knowledge lives here, not in the trainer:

- derived proprio — player (row,col) in [-1,1], optionally + goal —
  extracted from the frames (exact on ground-truth renders);
- the sprite-position eval hook: mean Manhattan error (in cells) of the
  reconstructed player/goal, and the ≤1-cell deployment **gate**. Pixel
  losses alone would happily lose the 0.8%-of-pixels sprites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from dreamer4.data.base import EpisodeVideoDataset


def sprite_positions(frames: np.ndarray, which: str) -> np.ndarray:
    """
    Locate the player ("p", bluest pixel) or goal ("g", reddest) per frame.

    Args:
        frames: (..., H, W, C) float or uint8.

    Returns:
        (..., 2) float32 (row, col). Frames must be ground-truth-like renders
        for exactness; on reconstructions this measures WHERE the model put
        the sprite — which is the point.
    """
    H, W = frames.shape[-3:-1]
    r = frames[..., 0].astype(np.float32)
    b = frames[..., 2].astype(np.float32)
    score = (b - r) if which == "p" else (r - b)
    idx = score.reshape(*frames.shape[:-3], H * W).argmax(-1)
    return np.stack([idx // W, idx % W], axis=-1).astype(np.float32)


class GridworldEpisodeDataset(EpisodeVideoDataset):
    """
    Adapter over a ``gridworld-collect`` dataset (see module docstring).

    Args:
        root:    Dataset directory.
        proprio: "none" | "player" (D_p=2) | "player_goal" (D_p=4).
        actions: Attach actions to clips — the 4 discrete moves as one-hot
                 float vectors (D_a=4), per the unified contract.
    """

    N_MOVES = 4

    def __init__(self, root: Union[str, Path], *, proprio: str = "none",
                 actions: bool = False):
        try:
            from gridworld import GridWorldDataset
        except ImportError as e:
            raise ImportError(
                "this dataset requires the companion 'gridworld' package "
                "(pip install -e <workspace>/gridworld)") from e
        if proprio not in ("none", "player", "player_goal"):
            raise ValueError(f"gridworld proprio must be none|player|player_goal, "
                             f"got '{proprio}'")
        # keep every shard decompressed — gridworld datasets are small
        self._ds = GridWorldDataset(root, cache_shards=999)
        self._proprio = proprio
        self._actions = bool(actions)

    def __len__(self) -> int:
        return len(self._ds)

    @property
    def proprio_dim(self) -> Optional[int]:
        return {"none": None, "player": 2, "player_goal": 4}[self._proprio]

    @property
    def action_dim(self) -> Optional[int]:
        return self.N_MOVES if self._actions else None

    def episode_frames(self, i: int) -> int:
        return int(self._ds.episodes[i]["n_frames"])

    def _load_clip(self, i: int, start: int, length: int) -> dict:
        episode = self._ds.get_episode(i)
        video = episode["video"][start : start + length]
        out = {"video": video}
        if self._actions:
            moves = episode["actions"][start : start + length - 1]
            out["actions"] = np.eye(self.N_MOVES, dtype=np.float32)[moves]
        if self._proprio != "none":
            H, W = video.shape[1:3]
            norm = np.array([H - 1, W - 1], np.float32)
            pos = [sprite_positions(video, "p") / norm * 2.0 - 1.0]
            if self._proprio == "player_goal":
                pos.append(sprite_positions(video, "g") / norm * 2.0 - 1.0)
            out["proprio"] = np.concatenate(pos, axis=-1).astype(np.float32)
        return out

    def eval_metrics(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """
        Sprite metrics on (N, T, H, W, C) float [0,1] frames — the proven
        gate conventions (diag_openloop, 2026-07-06):

        - ``player_err``: Manhattan cell error of the player, all frames;
        - ``player_present``: fraction of frames whose reconstruction has a
          confidently-blue player pixel (dreams love to drop the sprite);
        - ``goal_err``: gated to frames where the GROUND-TRUTH goal is
          visible — episodes end with the player covering the goal, and an
          ungated metric punishes correctly rendering that.
        """
        g_player = sprite_positions(gt, "p")
        p_player = sprite_positions(pred, "p")
        blue = pred[..., 2] - pred[..., 0]
        present = blue.reshape(*pred.shape[:2], -1).max(-1) >= 0.4
        out = {
            "player_err": float(np.abs(g_player - p_player).sum(-1).mean()),
            "player_present": float(present.mean()),
        }
        gt_red = gt[..., 0] - gt[..., 2]
        goal_visible = gt_red.reshape(*gt.shape[:2], -1).max(-1) >= 0.706
        if goal_visible.any():
            g_goal = sprite_positions(gt, "g")[goal_visible]
            p_goal = sprite_positions(pred, "g")[goal_visible]
            out["goal_err"] = float(np.abs(g_goal - p_goal).sum(-1).mean())
        return out

    def gate(self, metrics: Dict[str, float]) -> Optional[bool]:
        if "player_err" not in metrics:
            return None
        return (metrics["player_err"] <= 1.0
                and metrics.get("goal_err", 0.0) <= 1.0)
