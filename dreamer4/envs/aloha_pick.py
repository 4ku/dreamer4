"""AlohaPick for real-environment evaluation: N worlds in one MuJoCo Playground worker process.

The simulator needs JAX + MJX + a batch renderer, which live in their own Python environment
(Ivan's MuJoCo Playground fork); this class starts :mod:`dreamer4.envs.aloha_pick_worker` there as
a subprocess and steps all N worlds in lockstep -- every AlohaPick episode lasts the same 120 steps,
so lockstep loses nothing. Frames are composed exactly as the dataset adapter composes them
(:func:`dreamer4.data.aloha.compose_frame`), and ``info["state"]`` carries the raw joint state for
``proprio_from_info``.

The worker's interpreter comes from ``ALOHA_PYTHON`` (the fork's ``.venv/bin/python`` on the
machine at hand); it runs on the caller's GPU unless ``ALOHA_GPU`` says otherwise.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from dreamer4.data.aloha import compose_frame
from dreamer4.envs.aloha_pick_worker import read_msg, write_msg


class AlohaPickBatch:
    """``n`` AlohaPick worlds stepped together (see module docstring)."""

    batched = True

    def __init__(self, n: int, *, cameras: Sequence[str] = ("view_0", "view_2"), downscale: int = 1,
                 cam_res: Sequence[int] = (128, 128), max_steps: int = 120, lift_target: float = 0.10,
                 reward: str = "v1", task: str = "pick"):
        """``lift_target``: the box height (m) where the lift reward stops growing (the env's
        ``box_z_lift_target``). ``reward``: "v1" = the env's own reward (success checked at the last step),
        "v2" = :mod:`dreamer4.envs.aloha_reward_v2` (success ends the episode). ``task``: "pick" or "handover"
        (:mod:`dreamer4.envs.aloha_handover_v2`, always reward v2). A dataset records all of them in its manifest's
        ``env``, so evaluation plays the task and pays the reward the data was labelled with."""
        python = os.environ.get("ALOHA_PYTHON")
        if not python:
            raise RuntimeError("set ALOHA_PYTHON to the MuJoCo Playground environment's python "
                               "(the fork's .venv/bin/python) to evaluate on AlohaPick")
        self.n, self.cameras, self.downscale = int(n), list(cameras), int(downscale)
        self.max_steps = int(max_steps)
        env = dict(os.environ)
        if os.environ.get("ALOHA_GPU"):
            env["CUDA_VISIBLE_DEVICES"] = os.environ["ALOHA_GPU"]
        worker = Path(__file__).with_name("aloha_pick_worker.py")
        self._proc = subprocess.Popen([python, str(worker), str(self.n), str(int(cam_res[0])),
                                       ",".join(self.cameras), str(float(lift_target)), str(reward), str(task)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def _call(self, *msg):
        write_msg(self._proc.stdin, msg)
        out = read_msg(self._proc.stdout)
        if out is None:
            raise RuntimeError(f"AlohaPick worker exited (code {self._proc.poll()})")
        return out

    def _frames(self, obs):
        return compose_frame([obs[c] for c in self.cameras], self.downscale)

    def reset(self, seeds):
        """Reset every world (one seed each) -> ``(frames (n,H,W,3) u8, infos)``."""
        obs = self._call("reset", np.asarray(seeds, np.int64))
        return self._frames(obs), [{"state": s} for s in obs["state"]]

    def step(self, actions):
        """One lockstep step -> ``(frames, rewards (n,), terminated (n,), infos)``."""
        obs = self._call("step", np.asarray(actions, np.float32))
        infos = [{"state": s, "is_success": bool(k)} for s, k in zip(obs["state"], obs["success"])]
        return self._frames(obs), obs["reward"], obs["done"], infos

    def close(self):
        if self._proc.poll() is None:
            try:
                write_msg(self._proc.stdin, ("close",))
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    # smoke test: python -m dreamer4.envs.aloha_pick  (with ALOHA_PYTHON set)
    env = AlohaPickBatch(4)
    f, info = env.reset([0, 1, 2, 3])
    total = np.zeros(4)
    for _ in range(env.max_steps):
        f, r, d, info = env.step(np.zeros((4, 14), np.float32))
        total += r
    print("frames", f.shape, f.dtype, "state", info[0]["state"].shape, "return (zero actions)", total.round(2),
          "success", [i["is_success"] for i in info])
    env.close()
    sys.exit(0)
