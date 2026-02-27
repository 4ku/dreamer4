"""
Online training loop for Dreamer 4.

Implements the DreamerV3-style collect-train-imagine loop, wrapping all
model components into a single :class:`Dreamer4Agent` orchestrator.

Usage::

    from dreamer4.config import TrainConfig, load_config
    from dreamer4.train import Dreamer4Agent, online_training_loop

    cfg = load_config("config/dmcontrol.yaml")
    agent = Dreamer4Agent(cfg)
    online_training_loop(agent, env_fns)

The loop alternates between:
    1. **Collect** — run policy in environments, store episodes.
    2. **Train WM** — tokenizer reconstruction + shortcut forcing on replay data.
    3. **Imagine** — roll out the world model, train policy + value via PMPO/TD(lambda).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn

from dreamer4.agent import (
    PolicyHead,
    RewardHead,
    TaskEncoder,
    ValueHead,
    behavior_cloning_loss,
    reward_prediction_loss,
)
from dreamer4.checkpoint import AutoCheckpoint, load_checkpoint, save_checkpoint
from dreamer4.config import TrainConfig
from dreamer4.driver import Driver, Episode
from dreamer4.dynamics import (
    DynamicsModel,
    corrupt_representations,
    sample_flow_schedule,
    shortcut_forcing_loss,
)
from dreamer4.imagination import (
    ImaginedTrajectory,
    imagination_training_step,
    imagine_rollout,
    make_prior_policy,
)
from dreamer4.logging import MetricsLogger, setup_logging
from dreamer4.replay import ReplayBuffer
from dreamer4.tokenizer import RMSLossNormalizer, Tokenizer, recon_loss_from_mae
from dreamer4.utils import pack_bottleneck_to_spatial, patchify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dreamer4Agent: bundles all model components
# ---------------------------------------------------------------------------

class Dreamer4Agent(nn.Module):
    """Bundles tokenizer, dynamics, and agent heads into a single agent.

    Provides high-level methods for the three training phases:
        - ``train_world_model_step`` (Phase 1+2)
        - ``imagine_and_train`` (Phase 3)
        - ``policy_action`` (for env interaction)

    Args:
        config: :class:`TrainConfig` with all hyperparameters.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        action_dim = self._infer_action_dim()

        self.tokenizer: Optional[Tokenizer] = None

        self.dynamics = DynamicsModel(
            d_model=config.d_model,
            d_spatial=config.d_spatial,
            n_spatial=config.n_spatial,
            n_register=config.n_register,
            n_agent=config.n_agent,
            n_heads=config.n_heads,
            depth=config.depth,
            k_max=config.k_max,
            action_dim=action_dim,
        )

        self.task_encoder = TaskEncoder(
            d_model=config.d_model,
            n_agent=config.n_agent,
            num_tasks=config.num_tasks,
        )
        self.policy = PolicyHead(
            d_model=config.d_model,
            action_dim=action_dim,
            action_type="continuous",
            mtp_length=config.mtp_length,
            mlp_depth=config.policy_mlp_depth,
            mlp_ratio=config.policy_mlp_ratio,
        )
        self.reward_head = RewardHead(
            d_model=config.d_model,
            mtp_length=config.mtp_length,
            num_bins=config.reward_num_bins,
        )
        self.value_head = ValueHead(
            d_model=config.d_model,
            num_bins=config.value_num_bins,
        )

        self.prior_policy: Optional[PolicyHead] = None
        self._action_dim = action_dim

        # RMS loss normalization (paper: "normalize all loss terms by running
        # estimates of their root-mean-square")
        # Indices: 0=tokenizer, 1=dynamics, 2=bc, 3=reward_pred
        self.loss_normalizer = RMSLossNormalizer(n_losses=4)

    def _infer_action_dim(self) -> int:
        """Infer action dim from a temporary env instance."""
        from dreamer4.envs import DMControlEnv
        env = DMControlEnv(
            self.config.domain, self.config.task,
            size=self.config.image_size, action_repeat=self.config.action_repeat,
        )
        return env.action_dim

    def set_tokenizer(self, tokenizer: Tokenizer) -> None:
        """Attach a pretrained tokenizer."""
        self.tokenizer = tokenizer

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # -- Optimizers ----------------------------------------------------------

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Build optimizer dict for each component group."""
        cfg = self.config
        optimizers: dict[str, torch.optim.Optimizer] = {}

        if self.tokenizer is not None:
            optimizers["tokenizer"] = torch.optim.Adam(
                self.tokenizer.parameters(), lr=cfg.tokenizer_lr,
            )

        optimizers["dynamics"] = torch.optim.Adam(
            self.dynamics.parameters(), lr=cfg.dynamics_lr,
        )

        agent_params = (
            list(self.task_encoder.parameters())
            + list(self.policy.parameters())
            + list(self.reward_head.parameters())
        )
        optimizers["agent"] = torch.optim.Adam(
            agent_params, lr=cfg.agent_lr,
        )

        optimizers["value"] = torch.optim.Adam(
            self.value_head.parameters(), lr=cfg.agent_lr,
        )
        optimizers["policy"] = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.agent_lr,
        )

        return optimizers

    # -- Observation encoding ------------------------------------------------

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode raw pixel observations to packed spatial tokens.

        Args:
            obs: (B, T, C, H, W) or (B, T, N_cam, C, H, W) in [0, 1].

        Returns:
            (B, T, n_spatial, d_spatial) packed spatial representations.
        """
        cfg = self.config
        assert self.tokenizer is not None, "Tokenizer not set"

        if obs.dim() == 6:
            B, T, N_cam, C, H, W = obs.shape
            obs_flat = obs.reshape(B * T * N_cam, 1, C, H, W)
        else:
            B, T, C, H, W = obs.shape
            N_cam = 1
            obs_flat = obs.reshape(B * T, 1, C, H, W)

        patches = patchify(obs_flat, cfg.patch_size)
        with torch.no_grad():
            z = self.tokenizer.encode(patches)

        if N_cam > 1:
            z = z.reshape(B, T, N_cam, -1, z.shape[-1])
            z = z.reshape(B, T, -1, z.shape[-1])
        else:
            z = z.reshape(B, T, -1, z.shape[-1])

        packed = pack_bottleneck_to_spatial(
            z, n_spatial=cfg.n_spatial, k=cfg.packing_k,
        )
        return packed

    # -- World model training step -------------------------------------------

    def train_world_model_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        """One gradient step of combined tokenizer + dynamics + agent heads.

        Args:
            batch:      Dict from ``ReplayBuffer.sample_sequence``.
            optimizers: Dict of optimizers from ``build_optimizers``.

        Returns:
            Dict of scalar loss metrics.
        """
        cfg = self.config
        device = self.device
        metrics: dict[str, float] = {}

        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        B, T = obs.shape[:2]

        # --- 1. Tokenizer reconstruction ---
        if self.tokenizer is not None and "tokenizer" in optimizers:
            patches = patchify(obs, cfg.patch_size)
            pred, mae_mask, keep_prob = self.tokenizer(patches)
            tok_loss_raw = recon_loss_from_mae(pred, patches, mae_mask)

            self.loss_normalizer.update(0, tok_loss_raw)
            tok_loss = self.loss_normalizer.normalize(0, tok_loss_raw)

            optimizers["tokenizer"].zero_grad()
            tok_loss.backward()
            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.tokenizer.parameters(), cfg.max_grad_norm,
                )
            optimizers["tokenizer"].step()
            metrics["tokenizer_loss"] = tok_loss_raw.item()

        # --- 2. Encode to latents ---
        z_packed = self.encode_obs(obs)

        # --- 3. Dynamics (shortcut forcing) ---
        task_ids = torch.zeros(B, dtype=torch.long, device=device)
        agent_tok = self.task_encoder(task_ids)
        agent_tok = agent_tok.expand(B, T, -1, -1)

        dyn_loss_raw, dyn_aux = shortcut_forcing_loss(
            self.dynamics,
            z1=z_packed,
            actions=actions,
            k_max=cfg.k_max,
            bootstrap_fraction=cfg.bootstrap_fraction,
            agent_tokens=agent_tok,
        )

        self.loss_normalizer.update(1, dyn_loss_raw)
        dyn_loss = self.loss_normalizer.normalize(1, dyn_loss_raw)

        optimizers["dynamics"].zero_grad()
        dyn_loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.dynamics.parameters(), cfg.max_grad_norm,
            )
        optimizers["dynamics"].step()
        metrics["dynamics_loss"] = dyn_loss_raw.item()

        # --- 4. Agent heads (BC + reward prediction) ---
        with torch.no_grad():
            d, step_idx, tau, signal_idx = sample_flow_schedule(
                B, T, cfg.k_max, device,
            )
            z_tilde, _ = corrupt_representations(z_packed, tau)
            _, agent_embed = self.dynamics(
                actions, step_idx, signal_idx, z_tilde,
                agent_tokens=agent_tok,
            )

        if agent_embed is not None:
            if agent_embed.dim() == 4:
                agent_embed = agent_embed.mean(dim=-2)

            bc_loss_raw = behavior_cloning_loss(
                self.policy, agent_embed, actions,
                mtp_length=cfg.mtp_length,
            )
            rew_loss_raw = reward_prediction_loss(
                self.reward_head, agent_embed, rewards,
                mtp_length=cfg.mtp_length,
            )

            self.loss_normalizer.update(2, bc_loss_raw)
            self.loss_normalizer.update(3, rew_loss_raw)
            bc_loss = self.loss_normalizer.normalize(2, bc_loss_raw)
            rew_loss = self.loss_normalizer.normalize(3, rew_loss_raw)

            agent_loss = bc_loss + rew_loss
            optimizers["agent"].zero_grad()
            agent_loss.backward()
            if cfg.max_grad_norm > 0:
                params = (
                    list(self.task_encoder.parameters())
                    + list(self.policy.parameters())
                    + list(self.reward_head.parameters())
                )
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizers["agent"].step()

            metrics["bc_loss"] = bc_loss_raw.item()
            metrics["reward_pred_loss"] = rew_loss_raw.item()

        return metrics

    # -- Imagination training step -------------------------------------------

    def imagine_and_train(
        self,
        batch: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        """Imagination rollout + policy/value gradient step.

        Args:
            batch:      Dict from ``ReplayBuffer.sample_sequence`` (for context).
            optimizers: Dict of optimizers.

        Returns:
            Dict of scalar loss metrics.
        """
        cfg = self.config
        device = self.device

        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)

        z_packed = self.encode_obs(obs)

        ctx_len = min(cfg.imagination_ctx_len, z_packed.shape[1])
        context = z_packed[:, :ctx_len]
        ctx_actions = actions[:, :ctx_len]

        trajectory = imagine_rollout(
            self.dynamics,
            self.policy,
            self.reward_head,
            self.value_head,
            self.task_encoder,
            context=context,
            context_actions=ctx_actions,
            horizon=cfg.imagination_horizon,
            k_max=cfg.k_max,
            K=cfg.imagination_K,
            tau_ctx=cfg.imagination_tau_ctx,
        )

        if self.prior_policy is None:
            self.prior_policy = make_prior_policy(self.policy)

        metrics = imagination_training_step(
            trajectory,
            self.policy,
            self.value_head,
            optimizers["policy"],
            optimizers["value"],
            prior_policy=self.prior_policy,
            gamma=cfg.gamma,
            lam=cfg.lam,
            alpha=cfg.pmpo_alpha,
            beta=cfg.pmpo_beta,
            max_grad_norm=cfg.max_grad_norm,
        )

        return metrics

    # -- Policy action (for env interaction) ---------------------------------

    @torch.no_grad()
    def policy_action(
        self,
        obs_history: list[torch.Tensor],
        act_history: list[torch.Tensor],
    ) -> torch.Tensor:
        """Select an action from the current observation history.

        Args:
            obs_history: List of (C, H, W) observation tensors.
            act_history: List of (A,) action tensors.

        Returns:
            (A,) action tensor.
        """
        cfg = self.config
        device = self.device

        max_context = cfg.policy_max_context
        obs_list = obs_history[-max_context:]
        act_list = act_history[-(max_context - 1):] if act_history else []

        obs_t = torch.stack(obs_list).unsqueeze(0).to(device)
        z_packed = self.encode_obs(obs_t)

        T = z_packed.shape[1]
        task_ids = torch.zeros(1, dtype=torch.long, device=device)
        agent_tok = self.task_encoder(task_ids).expand(1, T, -1, -1)

        if act_list:
            act_t = torch.stack(act_list).unsqueeze(0).to(device)
            pad = torch.zeros(1, T - act_t.shape[1], self._action_dim, device=device)
            act_t = torch.cat([pad, act_t], dim=1)
        else:
            act_t = torch.zeros(1, T, self._action_dim, device=device)

        step_idx = torch.zeros(1, T, dtype=torch.long, device=device)
        signal_idx = torch.full((1, T), cfg.k_max, dtype=torch.long, device=device)

        z_hat, agent_out = self.dynamics(
            act_t, step_idx, signal_idx, z_packed,
            agent_tokens=agent_tok,
        )

        if agent_out is not None:
            agent_embed = agent_out[:, -1].mean(dim=-2)
        else:
            agent_embed = z_hat[:, -1].mean(dim=-2)

        embed = agent_embed.squeeze(0)  # (d_model,)
        params = self.policy.forward(embed.unsqueeze(0), head_idx=0)
        action, _ = self.policy.sample(params)
        return action.squeeze(0).cpu()

    # -- Prior policy update -------------------------------------------------

    def update_prior_policy(self) -> None:
        """Refresh the frozen behavioral prior from the current policy."""
        self.prior_policy = make_prior_policy(self.policy)


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------

def online_training_loop(
    agent: Dreamer4Agent,
    env_fns: Sequence[Callable[[], Any]],
    config: Optional[TrainConfig] = None,
    *,
    tokenizer: Optional[Tokenizer] = None,
    log_fn: Optional[Callable[[int, dict[str, float]], None]] = None,
    log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> dict[str, list[float]]:
    """Main Dreamer 4 online training loop.

    Follows the DreamerV3-style loop::

        1. Prefill replay buffer with random actions
        2. Loop:
           a. Collect env transitions with learned policy
           b. Train world model (tokenizer + dynamics + agent heads)
           c. Imagine rollouts, train policy + value

    Args:
        agent:          :class:`Dreamer4Agent` instance.
        env_fns:        Factory callables, each returning an environment.
        config:         Override config (defaults to ``agent.config``).
        tokenizer:      Optional pretrained tokenizer to attach.
        log_fn:         Extra ``(step, metrics) -> None`` callback.
        log_dir:        TensorBoard log directory.
        checkpoint_dir: Directory for periodic checkpoints.
        resume_from:    Path to a checkpoint to resume from.

    Returns:
        Dict mapping metric names to lists of values over training.
    """
    cfg = config or agent.config
    device = agent.device
    agent = agent.to(device)

    if tokenizer is not None:
        agent.set_tokenizer(tokenizer)
        agent.tokenizer = agent.tokenizer.to(device)

    optimizers = agent.build_optimizers()

    # -- Resume from checkpoint if requested --
    train_step = 0
    total_env_steps = 0

    if resume_from is not None:
        meta = load_checkpoint(resume_from, agent, optimizers, map_location=cfg.device)
        train_step = meta["train_step"]
        total_env_steps = meta["env_steps"]
        logger.info("Resumed from step %d (env_steps=%d)", train_step, total_env_steps)

    # -- Set up logging --
    if log_dir is not None:
        setup_logging(log_dir)
    metrics_logger: Optional[MetricsLogger] = None
    if log_dir is not None:
        metrics_logger = MetricsLogger(log_dir)

    # -- Set up checkpointing --
    auto_ckpt: Optional[AutoCheckpoint] = None
    if checkpoint_dir is not None:
        auto_ckpt = AutoCheckpoint(
            checkpoint_dir,
            every=cfg.checkpoint_every,
            keep=cfg.keep_last_n,
        )

    replay = ReplayBuffer(
        capacity=cfg.replay_capacity,
        min_episodes=max(1, cfg.prefill_steps // cfg.time_limit),
    )
    driver = Driver(env_fns)

    history: dict[str, list[float]] = {}

    def _log(step: int, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            history.setdefault(k, []).append(v)
        if metrics_logger is not None:
            metrics_logger.log_scalars(step, metrics)
        if log_fn is not None:
            log_fn(step, metrics)

    # -- 1. Prefill with random actions --
    logger.info("Prefilling replay with %d random steps...", cfg.prefill_steps)
    prefill_episodes = driver.collect_random(cfg.prefill_steps)
    for ep in prefill_episodes:
        replay.add_episode(ep.obs, ep.actions, ep.rewards)
    logger.info(
        "Prefill complete: %d episodes, %d transitions",
        replay.total_episodes, replay.total_transitions,
    )

    # -- 2. Main training loop --
    wm_metrics: dict[str, float] = {}
    im_metrics: dict[str, float] = {}

    logger.info("Starting online training for %d env steps...", cfg.total_steps)

    while total_env_steps < cfg.total_steps:
        # -- 2a. Collect experience --
        def policy_fn(
            obs_hist: list[torch.Tensor],
            act_hist: list[torch.Tensor],
        ) -> torch.Tensor:
            return agent.policy_action(obs_hist, act_hist)

        episodes = driver.collect(cfg.train_every, policy_fn)
        for ep in episodes:
            replay.add_episode(ep.obs, ep.actions, ep.rewards)
            total_env_steps += ep.length

        if not replay.is_ready:
            continue

        # -- 2b. Train world model --
        for _ in range(cfg.wm_train_steps):
            batch = replay.sample_sequence(
                cfg.batch_size, cfg.seq_len, device=device,
            )
            wm_metrics = agent.train_world_model_step(batch, optimizers)

        # -- 2c. Imagine and train policy/value --
        for _ in range(cfg.imagine_train_steps):
            batch = replay.sample_sequence(
                cfg.batch_size, cfg.seq_len, device=device,
            )
            im_metrics = agent.imagine_and_train(batch, optimizers)

        train_step += 1

        # -- Update prior policy periodically --
        if train_step % cfg.prior_update_interval == 0:
            agent.update_prior_policy()

        # -- Logging --
        if train_step % cfg.log_every == 0:
            all_metrics = {**wm_metrics, **im_metrics}
            all_metrics["env_steps"] = float(total_env_steps)
            all_metrics["episodes_collected"] = float(len(episodes))
            if episodes:
                all_metrics["episode_return"] = float(
                    sum(ep.total_return for ep in episodes) / len(episodes)
                )
            _log(train_step, all_metrics)

        # -- Checkpointing --
        if auto_ckpt is not None:
            auto_ckpt.step(
                train_step, agent, optimizers,
                env_steps=total_env_steps, config=cfg,
                metrics={**wm_metrics, **im_metrics},
            )

    driver.close()

    # -- Save final checkpoint --
    if checkpoint_dir is not None:
        save_checkpoint(
            f"{checkpoint_dir}/final.pt", agent, optimizers,
            train_step=train_step, env_steps=total_env_steps,
            config=cfg, metrics={**wm_metrics, **im_metrics},
        )

    if metrics_logger is not None:
        metrics_logger.close()

    logger.info("Training complete. Total env steps: %d", total_env_steps)
    return history
