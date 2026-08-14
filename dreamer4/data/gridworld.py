"""
The gridworld toy environment — our tokenizer/dynamics testbed.

Adapts the sharded dataset written by ``gridworld-collect`` (``manifest.json``
+ ``shard_*.npz``) to the unified format. Requires the companion
``gridworld`` package — imported lazily, it is NOT a dependency of dreamer4.

All gridworld DOMAIN knowledge lives here, not in the trainer:

- proprio — player (row,col) in [-1,1], optionally + goal — read from the
  positions the collector recorded from the env's own state (dataset format
  v2; older datasets must be re-collected). The same scaling serves training
  clips and, via :meth:`proprio_from_info`, the online policy;
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
            steps = slice(start, start + length - 1)
            out["rewards"] = episode["rewards"][steps].astype(np.float32)
            out["terminals"] = episode["terminals"][steps]
        if self._proprio != "none":
            if "player_pos" not in episode:
                raise RuntimeError(
                    f"{self._ds.root} is an old (v1) dataset without recorded "
                    "positions; proprio now comes from the env's state, so "
                    "re-collect it with gridworld-collect")
            player = episode["player_pos"][start : start + length]
            pos = [self._scale(player)]
            if self._proprio == "player_goal":
                goal = np.broadcast_to(np.asarray(episode["goal"]),
                                       player.shape)
                pos.append(self._scale(goal))
            out["proprio"] = np.concatenate(pos, axis=-1).astype(np.float32)
        return out

    def _scale(self, rowcol: np.ndarray) -> np.ndarray:
        """(row, col) cells -> [-1, 1], the one proprio convention. Used by
        training clips and :meth:`proprio_from_info` alike."""
        size = int(self._ds.manifest["size"])
        norm = np.array([size - 1, size - 1], np.float32)
        return np.asarray(rowcol, np.float32) / norm * 2.0 - 1.0

    def episode_meta(self, i: int) -> Dict[str, float]:
        """Collector metadata for episode ``i`` — the agent trainer's
        data-quality signals (BC filtering by ``noisiness``/``stickiness``)."""
        rec = self._ds.episodes[i]
        return {"noisiness": float(rec["noisiness"]),
                "stickiness": float(rec.get("stickiness", 0.0)),
                "success": bool(rec["success"]),
                "return": float(rec["return"]),
                "optimal_steps": int(rec["optimal_steps"])}

    def proprio_from_info(self, info: Dict) -> Optional[np.ndarray]:
        """The env's OWN state (``info["pos"]`` / ``info["goal"]``) scaled by
        :meth:`_scale` — the same values the recorded positions carry, so the
        online policy and the training clips speak one convention."""
        if self._proprio == "none":
            return None
        pos = [self._scale(info["pos"])]
        if self._proprio == "player_goal":
            pos.append(self._scale(info["goal"]))
        return np.concatenate(pos, axis=-1).astype(np.float32)

    #: BC-eligibility thresholds on the collector's quality dials. A cloned
    #: policy is only as good as what it imitates, and this collector records
    #: the full expert->random spectrum on purpose.
    BC_MAX_NOISINESS = 0.2
    BC_MAX_STICKINESS = 0.1

    def bc_weight(self, i: int) -> float:
        """Clone only the clean episodes (see the class constants). At the
        production thresholds this keeps ~2,000 of 12,350 training episodes —
        a band with 100% success and 1.14x optimal paths."""
        rec = self._ds.episodes[i]
        return float(rec["noisiness"] <= self.BC_MAX_NOISINESS
                     and rec.get("stickiness", 0.0) <= self.BC_MAX_STICKINESS)

    #: reward above which a frame counts as terminal. Gridworld pays exactly
    #: -0.01 or +1.0, so 0.5 is the maximum-margin split between them.
    TERMINAL_REWARD = 0.5

    def continues_from_reward(self, reward_pred):
        """
        Terminal iff the ARRIVING reward says "goal" — valid here because in
        gridworld the goal is the only rewarding event AND the only way an
        episode ends: verified 0 disagreements over 47,695 transitions
        (2026-07-24). This is what phase 3 stops dreams with, and it is a
        fact about THIS env — see the base-class contract.

        Args:
            reward_pred: (B, T) decoded rewards, arrival-aligned.

        Returns:
            (B, T) float — 1.0 keep going, 0.0 terminal.
        """
        return (reward_pred <= self.TERMINAL_REWARD).float()

    def env_spec(self) -> Dict:
        """The live env this data came from, reconstructed from the manifest
        (obstacle density is not recorded — the env default matches the
        collector default)."""
        m = self._ds.manifest
        return {"id": m.get("env_id", "GridWorld-v0"),
                "kwargs": {"size": int(m["size"]),
                           "max_steps": int(m["max_steps"]),
                           "step_penalty": float(m["step_penalty"]),
                           "goal_reward": float(m["goal_reward"])}}

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
