"""
Parallel environment driver for Dreamer 4.

Runs N environment instances (optionally via threads) and collects
experience using a provided policy function.  Completed episodes are
returned as ``Episode`` dataclasses ready for insertion into a
``ReplayBuffer``.

Usage::

    from dreamer4.driver import Driver, Episode

    driver = Driver(
        env_fns=[lambda: DMControlEnv("cartpole", "swingup")] * 4,
    )
    episodes = driver.collect(
        n_steps=1000,
        policy_fn=lambda obs_hist, act_hist: env.sample_random_action(),
    )
    for ep in episodes:
        replay_buffer.add_episode(ep.obs, ep.actions, ep.rewards)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch


@dataclass
class Episode:
    """A complete episode of experience.

    All tensors have a leading time dimension ``T`` (the episode length).
    """

    obs: torch.Tensor         # (T, C, H, W) or (T, N_cam, C, H, W)
    actions: torch.Tensor     # (T, action_dim)
    rewards: torch.Tensor     # (T,)
    dones: torch.Tensor       # (T,) bool — True only on the last step
    total_return: float = 0.0
    length: int = 0


PolicyFn = Callable[
    [list[torch.Tensor], list[torch.Tensor]],  # obs_history, action_history
    torch.Tensor,                                # action
]


class _EnvState:
    """Tracks per-environment rolling state during collection."""

    __slots__ = ("env", "obs_history", "act_history", "obs_buf",
                 "act_buf", "rew_buf", "done_buf", "ep_return")

    def __init__(self, env: Any):
        self.env = env
        self.obs_history: list[torch.Tensor] = []
        self.act_history: list[torch.Tensor] = []
        self.obs_buf: list[torch.Tensor] = []
        self.act_buf: list[torch.Tensor] = []
        self.rew_buf: list[float] = []
        self.done_buf: list[bool] = []
        self.ep_return: float = 0.0

    def start_episode(self) -> torch.Tensor:
        obs = self.env.reset()
        self.obs_history = [obs]
        self.act_history = []
        self.obs_buf = [obs]
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.ep_return = 0.0
        return obs

    def finish_episode(self) -> Episode:
        T = len(self.obs_buf)
        if len(self.act_buf) < T:
            action_dim = self.act_buf[0].shape[-1] if self.act_buf else self.env.action_dim
            self.act_buf.insert(0, torch.zeros(action_dim))
        return Episode(
            obs=torch.stack(self.obs_buf),
            actions=torch.stack(self.act_buf[:T]),
            rewards=torch.tensor(self.rew_buf + [0.0] * (T - len(self.rew_buf)),
                                 dtype=torch.float32),
            dones=torch.tensor(self.done_buf + [False] * (T - len(self.done_buf)),
                               dtype=torch.bool),
            total_return=self.ep_return,
            length=T,
        )


class Driver:
    """Parallel environment driver for experience collection.

    Instantiates environments from factory functions and steps them
    in lock-step, calling ``policy_fn`` to choose actions.  When an
    episode terminates it is auto-reset and the completed episode
    is recorded.

    Env states are **persistent** across ``collect`` calls so that
    partial episodes continue from where they left off.

    Args:
        env_fns:     List of callables, each returning an env instance.
        n_threads:   Worker threads for stepping envs.  0 = sequential.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        n_threads: int = 0,
    ):
        self._env_fns = list(env_fns)
        self._n_envs = len(env_fns)
        self._n_threads = n_threads
        self._envs: list[Any] | None = None
        self._states: list[_EnvState] | None = None

    def _ensure_envs(self) -> list[Any]:
        if self._envs is None:
            self._envs = [fn() for fn in self._env_fns]
        return self._envs

    def _ensure_states(self) -> list[_EnvState]:
        """Get (or create) persistent per-env states."""
        if self._states is None:
            envs = self._ensure_envs()
            self._states = [_EnvState(env) for env in envs]
            for s in self._states:
                s.start_episode()
        return self._states

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def collect(
        self,
        n_steps: int,
        policy_fn: PolicyFn,
    ) -> tuple[list[Episode], int]:
        """Collect *n_steps* total transitions across all environments.

        Env states persist across calls so partial episodes are continued,
        not discarded. Completed episodes are returned.

        Args:
            n_steps:   Minimum total transitions to collect.
            policy_fn: ``(obs_history, act_history) -> action`` callable.

        Returns:
            Tuple of (completed episodes, actual transitions taken).
        """
        states = self._ensure_states()
        completed: list[Episode] = []
        total_transitions = 0

        while total_transitions < n_steps:
            if self._n_threads > 0 and self._n_envs > 1:
                actions = self._collect_actions_threaded(states, policy_fn)
            else:
                actions = [
                    policy_fn(s.obs_history, s.act_history)
                    for s in states
                ]

            for i, (s, action) in enumerate(zip(states, actions)):
                obs, reward, done, truncated, info = s.env.step(action)

                s.act_buf.append(action.detach().cpu() if isinstance(action, torch.Tensor) else torch.tensor(action))
                s.act_history.append(action)
                s.obs_buf.append(obs)
                s.obs_history.append(obs)
                s.rew_buf.append(reward)
                s.done_buf.append(done)
                s.ep_return += reward
                total_transitions += 1

                if done:
                    ep = s.finish_episode()
                    completed.append(ep)
                    s.start_episode()

        return completed, total_transitions

    def _collect_actions_threaded(
        self,
        states: list[_EnvState],
        policy_fn: PolicyFn,
    ) -> list[torch.Tensor]:
        actions: list[Optional[torch.Tensor]] = [None] * len(states)
        with ThreadPoolExecutor(max_workers=self._n_threads) as pool:
            futures = {
                pool.submit(policy_fn, s.obs_history, s.act_history): i
                for i, s in enumerate(states)
            }
            for future in as_completed(futures):
                idx = futures[future]
                actions[idx] = future.result()
        return actions  # type: ignore[return-value]

    def collect_random(self, n_steps: int) -> tuple[list[Episode], int]:
        """Collect transitions using random actions (no policy needed).

        Convenience method for initial replay buffer prefilling.

        Returns:
            Tuple of (completed episodes, actual transitions taken).
        """
        states = self._ensure_states()

        def random_policy(
            obs_history: list[torch.Tensor],
            act_history: list[torch.Tensor],
        ) -> torch.Tensor:
            return states[0].env.sample_random_action()

        return self.collect(n_steps, random_policy)

    def close(self) -> None:
        """Close all environments."""
        self._states = None
        if self._envs is not None:
            self._envs = None
