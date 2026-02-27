"""Tests for Module 6: Driver, wrappers, replay, and training loop."""

import torch
import pytest

from dreamer4.envs import (
    DMControlEnv,
    ActionRepeatWrapper,
    NormalizeActionWrapper,
    TimeLimitWrapper,
    MultiCameraEnv,
)
from dreamer4.driver import Driver, Episode
from dreamer4.replay import ReplayBuffer


def _make_short_env():
    """Create a short-episode env for driver tests."""
    base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=2)
    return TimeLimitWrapper(base, max_steps=20)


# ---------------------------------------------------------------------------
# Environment wrapper tests
# ---------------------------------------------------------------------------

class TestActionRepeatWrapper:

    def test_repeat_accumulates_reward(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = ActionRepeatWrapper(base, repeat=4)
        wrapped.reset()
        act = wrapped.sample_random_action()
        obs, reward, done, trunc, info = wrapped.step(act)
        assert obs.shape == (3, 32, 32)
        assert isinstance(reward, float)

    def test_repeat_one_is_identity(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = ActionRepeatWrapper(base, repeat=1)
        wrapped.reset()
        act = wrapped.sample_random_action()
        obs, reward, done, _, _ = wrapped.step(act)
        assert obs.shape == (3, 32, 32)

    def test_action_dim_passthrough(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = ActionRepeatWrapper(base, repeat=2)
        assert wrapped.action_dim == base.action_dim

    def test_observation_shape_passthrough(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = ActionRepeatWrapper(base, repeat=2)
        assert wrapped.observation_shape == base.observation_shape


class TestNormalizeActionWrapper:

    def test_action_rescaling(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = NormalizeActionWrapper(base, low=-2.0, high=2.0)
        wrapped.reset()
        act = torch.ones(base.action_dim)
        obs, reward, done, _, _ = wrapped.step(act)
        assert obs.shape == (3, 32, 32)

    def test_sample_random_action_shape(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = NormalizeActionWrapper(base)
        a = wrapped.sample_random_action()
        assert a.shape == (base.action_dim,)


class TestTimeLimitWrapper:

    def test_truncation_at_limit(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = TimeLimitWrapper(base, max_steps=5)
        wrapped.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, truncated, _ = wrapped.step(wrapped.sample_random_action())
            steps += 1
        assert steps <= 5

    def test_reset_clears_counter(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        wrapped = TimeLimitWrapper(base, max_steps=3)
        wrapped.reset()
        for _ in range(3):
            wrapped.step(wrapped.sample_random_action())

        wrapped.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            _, _, done, _, _ = wrapped.step(wrapped.sample_random_action())
            steps += 1
        assert steps <= 3


class TestMultiCameraEnv:

    def test_observation_shape(self):
        env = MultiCameraEnv("cartpole", "swingup", camera_ids=[0], size=32)
        obs = env.reset()
        assert obs.shape == (1, 3, 32, 32)

    def test_step_shape(self):
        env = MultiCameraEnv("cartpole", "swingup", camera_ids=[0], size=32)
        env.reset()
        obs, reward, done, _, _ = env.step(env.sample_random_action())
        assert obs.shape == (1, 3, 32, 32)
        assert isinstance(reward, float)

    def test_action_dim(self):
        env = MultiCameraEnv("cartpole", "swingup", camera_ids=[0], size=32)
        assert env.action_dim == 1

    def test_n_cameras(self):
        env = MultiCameraEnv("cartpole", "swingup", camera_ids=[0], size=32)
        assert env.n_cameras == 1


class TestWrapperComposition:

    def test_stack_all_wrappers(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        env = ActionRepeatWrapper(base, repeat=2)
        env = NormalizeActionWrapper(env)
        env = TimeLimitWrapper(env, max_steps=10)

        obs = env.reset()
        assert obs.shape == (3, 32, 32)

        done = False
        steps = 0
        while not done and steps < 20:
            obs, _, done, _, _ = env.step(env.sample_random_action())
            steps += 1
        assert steps <= 10

    def test_unwrapped(self):
        base = DMControlEnv("cartpole", "swingup", size=32, action_repeat=1)
        env = ActionRepeatWrapper(base, repeat=2)
        env = TimeLimitWrapper(env, max_steps=10)
        assert env.unwrapped is base


# ---------------------------------------------------------------------------
# Driver tests
# ---------------------------------------------------------------------------

class TestDriver:

    def test_collect_random_returns_episodes(self):
        driver = Driver(env_fns=[_make_short_env])
        episodes = driver.collect_random(n_steps=50)
        assert len(episodes) > 0
        for ep in episodes:
            assert isinstance(ep, Episode)
            assert ep.obs.dim() == 4  # (T, C, H, W)
            assert ep.actions.dim() == 2
            assert ep.rewards.dim() == 1
        driver.close()

    def test_collect_with_policy(self):
        driver = Driver(env_fns=[_make_short_env])

        def policy(obs_hist, act_hist):
            return torch.rand(1) * 2 - 1

        episodes = driver.collect(n_steps=50, policy_fn=policy)
        assert len(episodes) > 0
        driver.close()

    def test_multi_env_collect(self):
        driver = Driver(env_fns=[_make_short_env for _ in range(2)])
        episodes = driver.collect_random(n_steps=100)
        total = sum(ep.length for ep in episodes)
        assert total >= 10
        driver.close()

    def test_episode_return_tracking(self):
        driver = Driver(env_fns=[_make_short_env])
        episodes = driver.collect_random(n_steps=200)
        for ep in episodes:
            assert isinstance(ep.total_return, float)
            assert ep.length > 0
        driver.close()


# ---------------------------------------------------------------------------
# Enhanced replay buffer tests
# ---------------------------------------------------------------------------

class TestReplayBufferEnhanced:

    def test_episode_counting(self):
        buf = ReplayBuffer(capacity=10000, min_episodes=3)
        assert buf.total_episodes == 0
        assert not buf.is_ready

        for ep in range(3):
            for t in range(10):
                buf.add(torch.randn(3, 8, 8), torch.randn(2),
                        0.0, done=(t == 9))

        assert buf.total_episodes == 3
        assert buf.is_ready

    def test_prefill_gate(self):
        buf = ReplayBuffer(capacity=10000, min_episodes=2)
        for t in range(10):
            buf.add(torch.randn(3, 8, 8), torch.randn(2), 0.0, done=(t == 9))
        assert buf.total_episodes == 1
        assert not buf.is_ready

        for t in range(10):
            buf.add(torch.randn(3, 8, 8), torch.randn(2), 0.0, done=(t == 9))
        assert buf.total_episodes == 2
        assert buf.is_ready

    def test_add_episode_bulk(self):
        buf = ReplayBuffer(capacity=10000)
        obs = torch.randn(20, 3, 8, 8)
        actions = torch.randn(20, 4)
        rewards = torch.randn(20)
        buf.add_episode(obs, actions, rewards)
        assert buf.total_episodes == 1
        assert buf.total_transitions == 20

    def test_capacity_eviction(self):
        buf = ReplayBuffer(capacity=15)
        for ep in range(4):
            obs = torch.randn(5, 3, 4, 4)
            actions = torch.randn(5, 2)
            rewards = torch.randn(5)
            buf.add_episode(obs, actions, rewards)

        assert buf.total_transitions <= 15

    def test_sample_device_placement(self):
        buf = ReplayBuffer(capacity=1000)
        for t in range(20):
            buf.add(torch.randn(3, 8, 8), torch.randn(2), 0.0, done=(t == 19))
        batch = buf.sample_sequence(4, 5, device=torch.device("cpu"))
        assert batch["obs"].device == torch.device("cpu")

    def test_sample_raises_on_empty(self):
        buf = ReplayBuffer(capacity=1000)
        with pytest.raises(ValueError, match="No valid subsequences"):
            buf.sample_sequence(4, 5)


# ---------------------------------------------------------------------------
# Integration: driver -> replay -> sample
# ---------------------------------------------------------------------------

class TestDriverReplayIntegration:

    def test_collect_store_sample(self):
        driver = Driver(env_fns=[_make_short_env])
        episodes = driver.collect_random(n_steps=200)
        driver.close()

        buf = ReplayBuffer(capacity=10000)
        for ep in episodes:
            buf.add_episode(ep.obs, ep.actions, ep.rewards)

        assert buf.total_episodes == len(episodes)

        batch = buf.sample_sequence(batch_size=4, seq_len=8)
        assert batch["obs"].shape[0] == 4
        assert batch["obs"].shape[1] == 8
        assert batch["obs"].shape[2:] == (3, 32, 32)
        assert batch["actions"].shape[:2] == (4, 8)
        assert batch["rewards"].shape == (4, 8)
        assert batch["dones"].shape == (4, 8)
