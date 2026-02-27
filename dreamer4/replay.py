"""
Episode-based replay buffer for Dreamer 4.

Stores experience as complete episodes with O(1) eviction via
:class:`collections.deque` and supports contiguous subsequence
sampling that never crosses episode boundaries.

Usage::

    buf = ReplayBuffer(capacity=100_000)
    buf.add(obs, action, reward, done)          # transition-level
    buf.add_episode(obs, actions, rewards)       # episode-level
    batch = buf.sample_sequence(batch_size=16, seq_len=16)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class _StoredEpisode:
    """Internal storage for a single episode."""
    obs: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[float]
    dones: list[bool]

    @property
    def length(self) -> int:
        return len(self.obs)


class ReplayBuffer:
    """Episode-based replay buffer with O(1) eviction.

    Episodes are stored in a :class:`collections.deque` and evicted
    oldest-first when the total transition count exceeds ``capacity``.

    Supports both transition-level insertion (``add``) and bulk episode
    insertion (``add_episode``).  ``sample_sequence`` draws random
    contiguous subsequences that stay within a single episode.

    Args:
        capacity:     Maximum number of transitions stored.
        min_episodes: Minimum episodes before ``sample_sequence`` is allowed
                      (useful as a prefill gate for online training).
    """

    def __init__(self, capacity: int = 100_000, min_episodes: int = 0):
        self.capacity = capacity
        self.min_episodes = min_episodes

        self._episodes: deque[_StoredEpisode] = deque()
        self._current: _StoredEpisode | None = None
        self._total_transitions: int = 0
        self._total_episodes: int = 0

    # -- Public properties ---------------------------------------------------

    def __len__(self) -> int:
        n = self._total_transitions
        if self._current is not None:
            n += self._current.length
        return n

    @property
    def total_transitions(self) -> int:
        """Number of transitions in completed episodes."""
        return self._total_transitions

    @property
    def total_episodes(self) -> int:
        """Number of completed episodes currently stored."""
        return self._total_episodes

    @property
    def is_ready(self) -> bool:
        """True when there are enough completed episodes for sampling."""
        return self._total_episodes >= max(1, self.min_episodes)

    # -- Transition-level insertion ------------------------------------------

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        """Store one transition.

        When ``done=True``, the current episode is closed and the next
        ``add`` starts a new episode.

        Args:
            obs:    (C, H, W) or (D,) observation tensor.
            action: (action_dim,) action tensor.
            reward: Scalar reward.
            done:   Whether the episode ended.
        """
        if self._current is None:
            self._current = _StoredEpisode([], [], [], [])

        self._current.obs.append(obs.detach().cpu())
        self._current.actions.append(action.detach().cpu())
        self._current.rewards.append(reward)
        self._current.dones.append(done)

        if done:
            self._commit_current()

    def _commit_current(self) -> None:
        """Move the in-progress episode into permanent storage."""
        if self._current is None or self._current.length == 0:
            return
        self._episodes.append(self._current)
        self._total_transitions += self._current.length
        self._total_episodes += 1
        self._current = None
        self._evict()

    def _evict(self) -> None:
        """Remove oldest episodes until we are within capacity."""
        while self._total_transitions > self.capacity and self._episodes:
            old = self._episodes.popleft()
            self._total_transitions -= old.length
            self._total_episodes -= 1

    # -- Episode-level insertion ---------------------------------------------

    def add_episode(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        """Add a complete episode at once.

        Args:
            obs:     (T, C, H, W) or (T, D) observations.
            actions: (T, action_dim) actions.
            rewards: (T,) scalar rewards.
        """
        T = obs.shape[0]
        ep = _StoredEpisode(
            obs=[obs[t].detach().cpu() for t in range(T)],
            actions=[actions[t].detach().cpu() for t in range(T)],
            rewards=[rewards[t].item() for t in range(T)],
            dones=[t == T - 1 for t in range(T)],
        )
        self._episodes.append(ep)
        self._total_transitions += T
        self._total_episodes += 1
        self._evict()

    # -- Sampling ------------------------------------------------------------

    def sample_sequence(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """Sample random contiguous subsequences.

        Each subsequence stays within one episode.  Sampling is uniform
        over valid starting positions across all episodes.

        Args:
            batch_size: Number of sequences to sample.
            seq_len:    Length of each sequence.
            device:     Target device for tensors.

        Returns:
            Dict with keys:
                ``obs``:     (B, T, ...) observations
                ``actions``: (B, T, action_dim)
                ``rewards``: (B, T)
                ``dones``:   (B, T) bool
        """
        valid: list[tuple[int, int]] = []  # (episode_idx, start_within_ep)
        for ep_idx, ep in enumerate(self._episodes):
            ep_len = ep.length
            if ep_len >= seq_len:
                for s in range(ep_len - seq_len + 1):
                    valid.append((ep_idx, s))

        if not valid:
            raise ValueError(
                f"No valid subsequences of length {seq_len} found in "
                f"{self._total_episodes} episodes "
                f"({self._total_transitions} transitions total)."
            )

        indices = torch.randint(0, len(valid), (batch_size,))

        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []

        for idx in indices:
            ep_idx, start = valid[idx.item()]
            ep = self._episodes[ep_idx]
            end = start + seq_len

            obs_batch.append(torch.stack(ep.obs[start:end]))
            act_batch.append(torch.stack(ep.actions[start:end]))
            rew_batch.append(torch.tensor(ep.rewards[start:end],
                                          dtype=torch.float32))
            done_batch.append(torch.tensor(ep.dones[start:end],
                                           dtype=torch.bool))

        result = {
            "obs": torch.stack(obs_batch),
            "actions": torch.stack(act_batch),
            "rewards": torch.stack(rew_batch),
            "dones": torch.stack(done_batch),
        }
        if device is not None:
            result = {k: v.to(device) for k, v in result.items()}
        return result

    # -- Compatibility shims -------------------------------------------------

    @property
    def _obs(self) -> list[torch.Tensor]:
        """Flat list of observations (for backward compat with old tests)."""
        out: list[torch.Tensor] = []
        for ep in self._episodes:
            out.extend(ep.obs)
        if self._current is not None:
            out.extend(self._current.obs)
        return out

    @property
    def _dones(self) -> list[bool]:
        """Flat list of done flags (for backward compat with old tests)."""
        out: list[bool] = []
        for ep in self._episodes:
            out.extend(ep.dones)
        if self._current is not None:
            out.extend(self._current.dones)
        return out
