#!/usr/bin/env python3
"""
Minimal SAC baseline for dm_control cartpole:balance (state-based).

Verifies the environment is solvable and establishes a target return
for the Dreamer4 agent. Uses the same action_repeat=2 and time_limit=250
so returns are directly comparable.

Usage:
    cd /mnt/dsk1/dreamer4
    source .venv/bin/activate
    MUJOCO_GL=egl python scripts/baseline_sac_cartpole.py
"""
from __future__ import annotations

import copy
import logging
import os
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from dm_control import suite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment wrapper (state-based, matching Dreamer4 settings)
# ---------------------------------------------------------------------------

class CartpoleBalanceEnv:
    """Thin wrapper around dm_control cartpole:balance returning flat state."""

    def __init__(self, action_repeat: int = 2, time_limit: int = 250, seed: int = 0):
        self._env = suite.load("cartpole", "balance", task_kwargs={"random": seed})
        self._action_repeat = action_repeat
        self._time_limit = time_limit
        self._step_count = 0
        self.state_dim = 5   # position(3) + velocity(2)
        self.action_dim = 1

    def _get_state(self) -> np.ndarray:
        obs = self._env.physics.get_state()
        pos = self._env.physics.named.data.qpos[:]
        vel = self._env.physics.named.data.qvel[:]
        return np.concatenate([
            self._env._task.get_observation(self._env.physics)["position"],
            self._env._task.get_observation(self._env.physics)["velocity"],
        ]).astype(np.float32)

    def reset(self) -> np.ndarray:
        self._env.reset()
        self._step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        total_reward = 0.0
        done = False
        for _ in range(self._action_repeat):
            ts = self._env.step(action)
            total_reward += ts.reward or 0.0
            if ts.last():
                done = True
                break
        self._step_count += 1
        if self._step_count >= self._time_limit:
            done = True
        return self._get_state(), total_reward, done

    def render_rgb(self, size: int = 256) -> np.ndarray:
        """Render current state as (H, W, 3) uint8 array."""
        return self._env.physics.render(height=size, width=size, camera_id=0).copy()


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf: deque = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(np.array(a), dtype=torch.float32),
            torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class PolicyNetwork(nn.Module):
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)

    def forward(self, s):
        h = self.trunk(s)
        mean = self.mean_head(h)
        log_std = self.logstd_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, s):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, s):
        mean, _ = self.forward(s)
        return torch.tanh(mean)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(np.log(alpha_init), device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            a = self.policy.deterministic(s)
        else:
            a, _ = self.policy.sample(s)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        s, a, r, s2, d = [x.to(self.device) for x in batch]

        with torch.no_grad():
            a_next, lp_next = self.policy.sample(s2)
            q1_next = self.q1_target(s2, a_next)
            q2_next = self.q2_target(s2, a_next)
            q_next = torch.min(q1_next, q2_next) - self.alpha * lp_next
            target = r + (1.0 - d) * self.gamma * q_next

        q1_loss = F.mse_loss(self.q1(s, a), target)
        q2_loss = F.mse_loss(self.q2(s, a), target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        a_new, lp_new = self.policy.sample(s)
        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha.detach() * lp_new - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (lp_new.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.lerp_(p.data, self.tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.lerp_(p.data, self.tau)

        return {
            "q_loss": (q1_loss.item() + q2_loss.item()) / 2,
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    agent: SAC,
    n_episodes: int = 10,
    action_repeat: int = 2,
    time_limit: int = 250,
    record_video: bool = False,
) -> tuple[float, list[list[np.ndarray]]]:
    returns = []
    all_frames: list[list[np.ndarray]] = []
    for i in range(n_episodes):
        env = CartpoleBalanceEnv(action_repeat=action_repeat, time_limit=time_limit, seed=1000 + i)
        s = env.reset()
        ep_return = 0.0
        done = False
        frames: list[np.ndarray] = []
        if record_video:
            frames.append(env.render_rgb())
        while not done:
            a = agent.select_action(s, deterministic=True)
            s, r, done = env.step(a)
            ep_return += r
            if record_video:
                frames.append(env.render_rgb())
        returns.append(ep_return)
        all_frames.append(frames)
    return float(np.mean(returns)), all_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ACTION_REPEAT = 2
    TIME_LIMIT = 250
    TOTAL_STEPS = 50_000
    WARMUP = 1000
    BATCH_SIZE = 256
    EVAL_EVERY = 2000
    EVAL_EPISODES = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("SAC baseline for cartpole:balance (device=%s)", device)
    logger.info("action_repeat=%d, time_limit=%d, max_return=%.0f",
                ACTION_REPEAT, TIME_LIMIT, TIME_LIMIT * ACTION_REPEAT * 1.0)

    env = CartpoleBalanceEnv(action_repeat=ACTION_REPEAT, time_limit=TIME_LIMIT, seed=42)
    agent = SAC(state_dim=env.state_dim, action_dim=env.action_dim, device=device)
    replay = ReplayBuffer(capacity=100_000)

    eval_steps: list[int] = []
    eval_returns: list[float] = []

    s = env.reset()
    ep_return = 0.0
    ep_count = 0

    for step in range(1, TOTAL_STEPS + 1):
        if step < WARMUP:
            a = np.random.uniform(-1, 1, size=(env.action_dim,)).astype(np.float32)
        else:
            a = agent.select_action(s)

        s_next, r, done = env.step(a)
        replay.add(s, a, r, s_next, float(done))
        s = s_next
        ep_return += r

        if done:
            ep_count += 1
            s = env.reset()
            ep_return = 0.0

        if step >= WARMUP and len(replay) >= BATCH_SIZE:
            batch = replay.sample(BATCH_SIZE)
            agent.update(batch)

        if step % EVAL_EVERY == 0:
            mean_ret, _ = evaluate(agent, EVAL_EPISODES, ACTION_REPEAT, TIME_LIMIT)
            eval_steps.append(step)
            eval_returns.append(mean_ret)
            logger.info("step %6d | eval_return = %.1f | alpha = %.3f",
                        step, mean_ret, agent.alpha.item())

    out_dir = Path("runs/baseline_sac")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    final_ret, final_frames = evaluate(
        agent, EVAL_EPISODES, ACTION_REPEAT, TIME_LIMIT, record_video=True,
    )
    eval_steps.append(TOTAL_STEPS)
    eval_returns.append(final_ret)
    logger.info("FINAL eval_return = %.1f", final_ret)

    import imageio
    for i, frames in enumerate(final_frames):
        if not frames:
            continue
        path = video_dir / f"final_ep{i:02d}.mp4"
        writer = imageio.get_writer(str(path), fps=30, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        logger.info("Saved video: %s (%d frames)", path, len(frames))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_steps, eval_returns, "o-", linewidth=2, markersize=5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return (mean over 10 eps)")
    ax.set_title("SAC Baseline — cartpole:balance (action_repeat=2, time_limit=250)")
    ax.axhline(y=TIME_LIMIT * ACTION_REPEAT, color="green", linestyle="--",
               label=f"Max possible ({TIME_LIMIT * ACTION_REPEAT})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "eval_return.png"), dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", out_dir / "eval_return.png")


if __name__ == "__main__":
    main()
