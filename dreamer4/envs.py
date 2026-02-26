"""
Environment wrappers for Dreamer 4.

Provides a thin Gymnasium-style interface around dm_control so that
observations are returned as PyTorch tensors suitable for the tokenizer.

    env = DMControlEnv("cartpole", "swingup", size=64, action_repeat=2)
    obs = env.reset()           # (C, H, W) float32 in [0, 1]
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from dm_control import suite


class DMControlEnv:
    """Gymnasium-style wrapper around ``dm_control.suite``.

    Args:
        domain:        dm_control domain name (e.g. ``"cartpole"``).
        task:          dm_control task name (e.g. ``"swingup"``).
        size:          Pixel observation size (square).
        action_repeat: Number of physics steps per agent step.
        seed:          Random seed for the environment.
        camera_id:     Camera to render from.
    """

    def __init__(
        self,
        domain: str,
        task: str,
        size: int = 64,
        action_repeat: int = 2,
        seed: int = 0,
        camera_id: int = 0,
    ):
        self._env = suite.load(
            domain, task, task_kwargs={"random": seed},
        )
        self._size = size
        self._action_repeat = action_repeat
        self._camera_id = camera_id
        self._done = False

        action_spec = self._env.action_spec()
        self._action_dim = int(np.prod(action_spec.shape))

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Returns (C, H, W)."""
        return (3, self._size, self._size)

    def _render(self) -> torch.Tensor:
        """Render current state as (C, H, W) float32 in [0, 1]."""
        pixels = self._env.physics.render(
            height=self._size, width=self._size, camera_id=self._camera_id,
        )
        img = torch.from_numpy(pixels.copy()).float().div(255.0)
        return img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    def reset(self) -> torch.Tensor:
        """Reset environment. Returns observation (C, H, W)."""
        self._env.reset()
        self._done = False
        return self._render()

    def step(
        self, action: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """Execute action with action repeat.

        Args:
            action: (action_dim,) in [-1, 1].

        Returns:
            obs:       (C, H, W) float32 in [0, 1].
            reward:    Scalar float (summed over repeats).
            done:      True if episode ended.
            truncated: Always False (dm_control doesn't truncate).
            info:      Empty dict.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = action.flatten()

        total_reward = 0.0
        for _ in range(self._action_repeat):
            ts = self._env.step(action)
            total_reward += ts.reward or 0.0
            if ts.last():
                self._done = True
                break

        obs = self._render()
        return obs, total_reward, self._done, False, {}

    def sample_random_action(self) -> torch.Tensor:
        """Sample a uniformly random action in [-1, 1]."""
        return torch.rand(self._action_dim) * 2.0 - 1.0
