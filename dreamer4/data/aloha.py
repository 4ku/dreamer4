"""
ALOHA pick (MuJoCo Playground ``AlohaPick``, vision) — a robot-arm dataset.

Storage: a directory with ``manifest.json`` (``"format": "aloha-npy"``) and episode-major numpy
arrays, read memory-mapped:

    state.npy            (N, T, 14)          float32   joint positions incl. grippers
    action.npy           (N, T_a, 14)        float32   in [-1, 1]; a[t] drives frame t -> t+1
    pixels_<camera>.npy  (N, T, H, W, 3)     uint8     one array per camera (view_0 top-down,
                                                       view_1 left wrist, view_2 right wrist)
    reward.npy           (N, T_a)            float32   optional: the env's dense reward of t -> t+1

``T_a`` may equal ``T`` (the recorder stores the action taken FROM every frame, the last one
without a next frame) or ``T - 1``; clips only use actions that have a next frame.

The frame the models see is the chosen cameras side by side, optionally downscaled by an integer
factor (box-filter), so ``(H / s, n_cams * W / s, 3)``. Proprio is the 14-D state standardised
with the per-dimension mean / std stored in the manifest (the std floored at ``STD_FLOOR``, so the
frozen left arm's jitter stays small), the same convention :meth:`proprio_from_info` applies to a
live env's state.

Domain facts used by the hooks:
- an episode ends at the time limit, never on an event, so no reward means "terminal"
  (:meth:`continues_from_reward` always says continue — a dream runs to its horizon and phase 3
  bootstraps the value there);
- every recording is cloned unless the manifest marks the episode as a failure or as noisy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from dreamer4.data.base import EpisodeVideoDataset

FORMAT = "aloha-npy"


def compose_frame(views: Sequence[np.ndarray], downscale: int = 1) -> np.ndarray:
    """Cameras side by side, box-filtered by ``downscale``: (..., H, W, 3) each -> (..., H/s, n*W/s, 3)."""
    frame = np.concatenate(list(views), axis=-2)
    if downscale > 1:
        *lead, H, W, C = frame.shape
        s = downscale
        frame = frame.reshape(*lead, H // s, s, W // s, s, C).mean(axis=(-4, -2))
        frame = np.round(frame).astype(np.uint8)
    return frame


class AlohaEpisodeDataset(EpisodeVideoDataset):
    """
    Adapter over an ``aloha-npy`` directory (see module docstring).

    Args:
        root:      Dataset directory.
        proprio:   "none" | "state" (the standardised 14-D joint state).
        actions:   Attach actions (and rewards, when recorded) to clips.
        cameras:   Cameras to compose, left to right; default = the manifest's.
        downscale: Integer box-filter factor applied to every camera.
    """

    PROPRIO_MODES = ("state",)

    #: Episodes whose recorded action noise is above this are not cloned.
    BC_MAX_NOISE = 0.0

    #: Floor of the standardising std: the frozen left arm only jitters (std 3e-6 - 5e-3 rad), and
    #: dividing by that would blow physics noise up to O(1); the right gripper moves with std 0.01.
    STD_FLOOR = 0.01

    @classmethod
    def recognizes(cls, root: Path) -> bool:
        manifest = root / "manifest.json"
        if not manifest.exists():
            return False
        try:
            return json.loads(manifest.read_text()).get("format") == FORMAT
        except (ValueError, OSError):
            return False

    @classmethod
    def from_path(cls, root: Path, *, proprio: str = "none", actions: bool = False,
                  cameras=None, **_):
        return cls(root, proprio=("state" if proprio == "auto" else proprio), actions=actions,
                   cameras=cameras or None)

    def __init__(self, root: Union[str, Path], *, proprio: str = "none", actions: bool = False,
                 cameras: Optional[Sequence[str]] = None, downscale: Optional[int] = None):
        if proprio not in ("none", "state"):
            raise ValueError(f"aloha proprio must be none|state, got '{proprio}'")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.cameras = list(cameras or self.manifest["cameras"])
        self.downscale = int(downscale or self.manifest.get("downscale", 1))
        self._pix = {c: np.load(self.root / f"pixels_{c}.npy", mmap_mode="r") for c in self.cameras}
        self._state = np.load(self.root / "state.npy", mmap_mode="r")
        self._action = np.load(self.root / "action.npy", mmap_mode="r")
        rew = self.root / "reward.npy"
        self._reward = np.load(rew, mmap_mode="r") if rew.exists() else None
        self._proprio = proprio
        self._actions = bool(actions)
        self._mean = np.asarray(self.manifest["state_mean"], np.float32)
        std = np.asarray(self.manifest["state_std"], np.float32)
        self._std = np.maximum(std, self.STD_FLOOR).astype(np.float32)
        self._n_frames = self._state.shape[1]
        self._n_trans = min(self._action.shape[1], self._n_frames - 1)
        self._episodes = self.manifest.get("episodes") or [{} for _ in range(len(self._state))]

    def __len__(self) -> int:
        return len(self._state)

    @property
    def proprio_dim(self) -> Optional[int]:
        return None if self._proprio == "none" else self._state.shape[-1]

    @property
    def action_dim(self) -> Optional[int]:
        return self._action.shape[-1] if self._actions else None

    def episode_frames(self, i: int) -> int:
        return self._n_trans + 1

    def _load_clip(self, i: int, start: int, length: int) -> dict:
        frames = slice(start, start + length)
        out = {"video": compose_frame([np.asarray(self._pix[c][i, frames]) for c in self.cameras],
                                      self.downscale)}
        if self._proprio != "none":
            out["proprio"] = self.standardise(np.asarray(self._state[i, frames]))
        if self._actions:
            steps = slice(start, start + length - 1)
            out["actions"] = np.asarray(self._action[i, steps], np.float32)
            if self._reward is not None:
                out["rewards"] = np.asarray(self._reward[i, steps], np.float32)
                out["terminals"] = np.zeros(length - 1, bool)
        return out

    def standardise(self, state: np.ndarray) -> np.ndarray:
        """Raw joint state -> the proprio convention (training clips and the live env alike)."""
        return ((np.asarray(state, np.float32) - self._mean) / self._std).astype(np.float32)

    def episode_meta(self, i: int) -> Dict:
        return dict(self._episodes[i])

    def bc_weight(self, i: int) -> float:
        """Clone successful, clean recordings; noisy or failed ones still teach the world model."""
        rec = self._episodes[i]
        if not rec.get("success", True):
            return 0.0
        return float(rec.get("noise", 0.0) <= self.BC_MAX_NOISE)

    def continues_from_reward(self, reward_pred):
        """Episodes end at the time limit only: nothing a reward says is terminal."""
        return reward_pred * 0.0 + 1.0

    def proprio_from_info(self, info: Dict) -> Optional[np.ndarray]:
        if self._proprio == "none":
            return None
        return self.standardise(info["state"])

    def env_spec(self) -> Optional[Dict]:
        """The live env -- N worlds in lockstep (:class:`dreamer4.envs.aloha_pick.AlohaPickBatch`)
        with the same cameras and scale as the clips."""
        env = self.manifest.get("env")
        if not env:
            return None
        return {"id": "dreamer4.envs.aloha_pick:AlohaPickBatch", "batched": True,
                "kwargs": {**env, "cameras": self.cameras, "downscale": self.downscale}}


def write_manifest(root: Union[str, Path], *, cameras: Sequence[str], downscale: int = 1,
                   env: Optional[Dict] = None, episodes: Optional[list] = None) -> Dict:
    """Write ``manifest.json`` for a directory of ``aloha-npy`` arrays (standardisation stats
    computed from ``state.npy``), so :class:`AlohaEpisodeDataset` recognises it."""
    root = Path(root)
    state = np.load(root / "state.npy", mmap_mode="r")
    flat = np.asarray(state, np.float64).reshape(-1, state.shape[-1])
    manifest = {"format": FORMAT, "cameras": list(cameras), "downscale": int(downscale),
                "n_episodes": int(state.shape[0]),
                "state_mean": flat.mean(0).round(6).tolist(),
                "state_std": flat.std(0).round(6).tolist()}
    if env:
        manifest["env"] = env
    if episodes:
        manifest["episodes"] = episodes
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write manifest.json for a directory of aloha-npy arrays.")
    ap.add_argument("root")
    ap.add_argument("--cameras", default="view_0,view_2")
    ap.add_argument("--downscale", type=int, default=1)
    a = ap.parse_args()
    m = write_manifest(a.root, cameras=a.cameras.split(","), downscale=a.downscale)
    print(json.dumps({k: v for k, v in m.items() if k != "episodes"}))
