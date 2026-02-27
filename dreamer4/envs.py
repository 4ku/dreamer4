"""
Environment wrappers for Dreamer 4.

Provides a thin Gymnasium-style interface around dm_control so that
observations are returned as PyTorch tensors suitable for the tokenizer.

Base environment::

    env = DMControlEnv("cartpole", "swingup", size=64, action_repeat=2)
    obs = env.reset()           # (C, H, W) float32 in [0, 1]
    obs, reward, done, truncated, info = env.step(action)

Composable wrappers::

    env = DMControlEnv("cartpole", "swingup", size=64, action_repeat=1)
    env = ActionRepeatWrapper(env, repeat=2)
    env = TimeLimitWrapper(env, max_steps=500)

Multi-camera (robotics)::

    env = MultiCameraEnv("cartpole", "swingup", camera_ids=[0, 1], size=64)
    obs = env.reset()           # (N_cam, C, H, W)
"""

from __future__ import annotations

import abc
import os
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from dm_control import suite


# ---------------------------------------------------------------------------
# Protocol: all envs and wrappers implement this interface
# ---------------------------------------------------------------------------

class EnvProtocol(Protocol):
    @property
    def action_dim(self) -> int: ...
    @property
    def observation_shape(self) -> tuple[int, ...]: ...
    def reset(self) -> torch.Tensor: ...
    def step(self, action: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]: ...
    def sample_random_action(self) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Base: DMControlEnv
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Composable wrappers
# ---------------------------------------------------------------------------

class EnvWrapper:
    """Base class for composable environment wrappers."""

    def __init__(self, env: Any):
        self._env = env

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._env.observation_shape

    @property
    def unwrapped(self) -> Any:
        if hasattr(self._env, "unwrapped"):
            return self._env.unwrapped
        return self._env

    def reset(self) -> torch.Tensor:
        return self._env.reset()

    def step(
        self, action: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        return self._env.step(action)

    def sample_random_action(self) -> torch.Tensor:
        return self._env.sample_random_action()


class ActionRepeatWrapper(EnvWrapper):
    """Repeats each action for *repeat* inner environment steps.

    Rewards are summed over repeats.  If the inner env already has
    action_repeat > 1 you typically set it to 1 and use this wrapper instead
    for composability.

    Args:
        env:    Inner environment.
        repeat: Number of times to repeat the action.
    """

    def __init__(self, env: Any, repeat: int = 2):
        super().__init__(env)
        assert repeat >= 1
        self._repeat = repeat

    def step(
        self, action: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, done, truncated, info = self._env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info


class NormalizeActionWrapper(EnvWrapper):
    """Rescales actions from ``[-1, 1]`` to the environment's native range.

    Useful when the inner env expects actions outside ``[-1, 1]`` (e.g.
    Gymnasium continuous envs).  DMControl already uses ``[-1, 1]`` so this
    is mainly for non-DMC backends.

    Args:
        env:  Inner environment.
        low:  Lower bound of the native action range.
        high: Upper bound of the native action range.
    """

    def __init__(
        self,
        env: Any,
        low: float = -1.0,
        high: float = 1.0,
    ):
        super().__init__(env)
        self._low = low
        self._high = high

    def step(
        self, action: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        if isinstance(action, torch.Tensor):
            scaled = (action.clamp(-1.0, 1.0) + 1.0) / 2.0
            scaled = scaled * (self._high - self._low) + self._low
        else:
            action_clipped = np.clip(action, -1.0, 1.0)
            scaled = (action_clipped + 1.0) / 2.0
            scaled = scaled * (self._high - self._low) + self._low
        return self._env.step(scaled)


class TimeLimitWrapper(EnvWrapper):
    """Truncates episodes after *max_steps* agent steps.

    Sets the ``truncated`` flag to ``True`` when the limit is reached,
    without marking the episode as ``done`` (allows value bootstrapping).

    Args:
        env:       Inner environment.
        max_steps: Maximum steps before truncation.
    """

    def __init__(self, env: Any, max_steps: int = 1000):
        super().__init__(env)
        self._max_steps = max_steps
        self._step_count = 0

    def reset(self) -> torch.Tensor:
        self._step_count = 0
        return self._env.reset()

    def step(
        self, action: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self._env.step(action)
        self._step_count += 1
        if self._step_count >= self._max_steps and not done:
            truncated = True
            done = True
        return obs, reward, done, truncated, info


# ---------------------------------------------------------------------------
# Multi-Camera Environment
# ---------------------------------------------------------------------------

class MultiCameraEnv:
    """DMControl environment that renders from multiple cameras.

    Each ``reset`` / ``step`` returns a stacked observation tensor of shape
    ``(N_cam, C, H, W)`` containing one image per camera.

    For the Dreamer 4 robotics setup (paper Section 4) each camera stream
    is tokenized independently and the resulting spatial tokens are
    concatenated before being fed to the dynamics model.

    Args:
        domain:        dm_control domain name.
        task:          dm_control task name.
        camera_ids:    Sequence of camera IDs to render from.
        size:          Pixel observation size (square).
        action_repeat: Number of physics steps per agent step.
        seed:          Random seed.
    """

    def __init__(
        self,
        domain: str,
        task: str,
        camera_ids: Sequence[int] = (0,),
        size: int = 64,
        action_repeat: int = 2,
        seed: int = 0,
    ):
        self._env = suite.load(
            domain, task, task_kwargs={"random": seed},
        )
        self._camera_ids = list(camera_ids)
        self._size = size
        self._action_repeat = action_repeat
        self._done = False

        action_spec = self._env.action_spec()
        self._action_dim = int(np.prod(action_spec.shape))

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_cameras(self) -> int:
        return len(self._camera_ids)

    @property
    def observation_shape(self) -> tuple[int, int, int, int]:
        """Returns (N_cam, C, H, W)."""
        return (len(self._camera_ids), 3, self._size, self._size)

    def _render(self) -> torch.Tensor:
        """Render all cameras. Returns (N_cam, C, H, W) float32 in [0, 1]."""
        frames = []
        for cam_id in self._camera_ids:
            pixels = self._env.physics.render(
                height=self._size, width=self._size, camera_id=cam_id,
            )
            img = torch.from_numpy(pixels.copy()).float().div(255.0)
            frames.append(img.permute(2, 0, 1))
        return torch.stack(frames, dim=0)

    def reset(self) -> torch.Tensor:
        """Reset environment. Returns (N_cam, C, H, W)."""
        self._env.reset()
        self._done = False
        return self._render()

    def step(
        self, action: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """Execute action with action repeat.

        Returns:
            obs:       (N_cam, C, H, W) float32 in [0, 1].
            reward:    Scalar float (summed over repeats).
            done:      True if episode ended.
            truncated: Always False.
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
