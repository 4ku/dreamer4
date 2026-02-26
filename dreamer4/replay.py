"""
Simple episode-based replay buffer for Dreamer 4.

Stores transitions as complete episodes and samples contiguous
subsequences for world model and agent training.

    buf = ReplayBuffer(capacity=100_000)
    buf.add(obs, action, reward, done)
    batch = buf.sample_sequence(batch_size=16, seq_len=16)
"""

from __future__ import annotations

from typing import Optional

import torch


class ReplayBuffer:
    """Episode-based replay buffer.

    Stores observations, actions, rewards, and done flags.  Episodes are
    delimited by ``done=True`` transitions.  ``sample_sequence`` draws
    random contiguous subsequences that do not cross episode boundaries.

    Args:
        capacity: Maximum number of transitions stored.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._obs: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._episode_starts: list[int] = []
        self._current_ep_start: int = 0

    def __len__(self) -> int:
        return len(self._obs)

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
        idx = len(self._obs)

        if idx == 0 or (len(self._dones) > 0 and self._dones[-1]):
            self._current_ep_start = idx
            self._episode_starts.append(idx)

        self._obs.append(obs.detach().cpu())
        self._actions.append(action.detach().cpu())
        self._rewards.append(reward)
        self._dones.append(done)

        while len(self._obs) > self.capacity:
            self._obs.pop(0)
            self._actions.pop(0)
            self._rewards.pop(0)
            self._dones.pop(0)
            self._episode_starts = [
                s - 1 for s in self._episode_starts if s > 0
            ]
            if self._episode_starts and self._episode_starts[0] < 0:
                self._episode_starts.pop(0)

    def _rebuild_episode_starts(self) -> list[int]:
        """Recompute episode start indices from done flags."""
        starts = [0]
        for i, d in enumerate(self._dones):
            if d and i + 1 < len(self._dones):
                starts.append(i + 1)
        return starts

    def sample_sequence(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """Sample random contiguous subsequences.

        Each subsequence stays within one episode.

        Args:
            batch_size: Number of sequences to sample.
            seq_len:    Length of each sequence.
            device:     Target device for tensors.

        Returns:
            Dict with keys:
                ``obs``:     (B, T, C, H, W) or (B, T, D)
                ``actions``: (B, T, action_dim)
                ``rewards``: (B, T)
                ``dones``:   (B, T) bool
        """
        ep_starts = self._rebuild_episode_starts()
        N = len(self._obs)

        valid_starts: list[int] = []
        for i, es in enumerate(ep_starts):
            if i + 1 < len(ep_starts):
                ep_end = ep_starts[i + 1]
            else:
                ep_end = N
            ep_len = ep_end - es
            if ep_len >= seq_len:
                for s in range(es, ep_end - seq_len + 1):
                    valid_starts.append(s)

        if not valid_starts:
            raise ValueError(
                f"No valid subsequences of length {seq_len} found in "
                f"{len(ep_starts)} episodes ({N} transitions total)."
            )

        idxs = torch.randint(0, len(valid_starts), (batch_size,))
        chosen = [valid_starts[i.item()] for i in idxs]

        obs_batch = []
        act_batch = []
        rew_batch = []
        done_batch = []

        for start in chosen:
            obs_seq = torch.stack(self._obs[start : start + seq_len])
            act_seq = torch.stack(self._actions[start : start + seq_len])
            rew_seq = torch.tensor(
                self._rewards[start : start + seq_len], dtype=torch.float32,
            )
            done_seq = torch.tensor(
                self._dones[start : start + seq_len], dtype=torch.bool,
            )
            obs_batch.append(obs_seq)
            act_batch.append(act_seq)
            rew_batch.append(rew_seq)
            done_batch.append(done_seq)

        result = {
            "obs": torch.stack(obs_batch),
            "actions": torch.stack(act_batch),
            "rewards": torch.stack(rew_batch),
            "dones": torch.stack(done_batch),
        }
        if device is not None:
            result = {k: v.to(device) for k, v in result.items()}
        return result

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
        for t in range(T):
            done = t == T - 1
            self.add(obs[t], actions[t], rewards[t].item(), done)
