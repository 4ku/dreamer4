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

import numpy as np
import torch
import torch.nn as nn

from dreamer4.agent import (
    PolicyHead,
    RewardHead,
    TaskEncoder,
    ValueHead,
    compute_lambda_returns,
    pmpo_policy_loss,
    reward_prediction_loss,
)
from dreamer4.checkpoint import AutoCheckpoint, load_checkpoint, save_checkpoint
from dreamer4.config import TrainConfig
from dreamer4.driver import Driver, Episode
from dreamer4.dynamics import (
    DynamicsModel,
    sample_one_timestep,
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
from dreamer4.utils import (
    pack_bottleneck_to_spatial,
    patchify,
    unpack_spatial_to_bottleneck,
    unpatchify,
)

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
            bin_low=config.bin_low,
            bin_high=config.bin_high,
        )
        self.value_head = ValueHead(
            d_model=config.d_model,
            num_bins=config.value_num_bins,
            bin_low=config.bin_low,
            bin_high=config.bin_high,
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
        tok = _unwrap(self.tokenizer)
        with torch.no_grad():
            z = tok.encode(patches)

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

        # --- 4. Agent heads (reward prediction) ---
        # Forward pass with clean signal (tau=1) to get agent embeddings.
        # Gradients flow into task_encoder and reward_head; dynamics weights
        # are detached so this loss doesn't interfere with the dynamics loss.
        dynamics_raw = _unwrap(self.dynamics)
        step_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        signal_idx = torch.full((B, T), cfg.k_max, dtype=torch.long, device=device)

        z_packed_detached = z_packed.detach()
        agent_tok_fresh = self.task_encoder(
            torch.zeros(B, dtype=torch.long, device=device),
        ).expand(B, T, -1, -1)

        _, agent_embed = dynamics_raw(
            actions.detach(), step_idx, signal_idx, z_packed_detached,
            agent_tokens=agent_tok_fresh,
        )

        if agent_embed is not None:
            if agent_embed.dim() == 4:
                agent_embed = agent_embed.mean(dim=-2)

            rew_loss_raw = reward_prediction_loss(
                self.reward_head, agent_embed, rewards,
                mtp_length=cfg.mtp_length,
            )

            self.loss_normalizer.update(3, rew_loss_raw)
            rew_loss = self.loss_normalizer.normalize(3, rew_loss_raw)

            optimizers["agent"].zero_grad()
            rew_loss.backward()
            if cfg.max_grad_norm > 0:
                params = (
                    list(self.task_encoder.parameters())
                    + list(self.reward_head.parameters())
                )
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizers["agent"].step()

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

        dynamics_raw = _unwrap(self.dynamics)
        trajectory = imagine_rollout(
            dynamics_raw,
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
            max_context_window=cfg.policy_max_context,
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
            entropy_scale=cfg.entropy_scale,
            max_grad_norm=cfg.max_grad_norm,
        )

        return metrics

    # -- Real-env policy learning ---------------------------------------------

    def train_policy_on_replay(
        self,
        batch: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        """Train policy using PMPO on real environment transitions.

        Runs the dynamics model on real replay data to obtain agent
        embeddings, then computes TD(lambda) returns from real rewards
        and trains the policy with PMPO using ground-truth advantages.
        """
        cfg = self.config
        device = self.device

        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        B, T = obs.shape[:2]
        if T < 3:
            return {}

        z_packed = self.encode_obs(obs)

        dynamics_raw = _unwrap(self.dynamics)
        step_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        signal_idx = torch.full((B, T), cfg.k_max, dtype=torch.long, device=device)

        task_ids = torch.zeros(B, dtype=torch.long, device=device)
        agent_tok = self.task_encoder(task_ids).expand(B, T, -1, -1)

        with torch.no_grad():
            _, agent_embed = dynamics_raw(
                actions, step_idx, signal_idx, z_packed.detach(),
                agent_tokens=agent_tok,
            )

        if agent_embed is None:
            return {}

        if agent_embed.dim() == 4:
            agent_embed = agent_embed.mean(dim=-2)

        with torch.no_grad():
            values = self.value_head.predict(agent_embed)
            continues = torch.ones(B, T, device=device)
            lambda_returns = compute_lambda_returns(
                rewards, values, gamma=cfg.gamma, lam=cfg.lam, continues=continues,
            )
            advantages = (lambda_returns - values)[:, :-1]

        agent_embed_policy = agent_embed[:, :-1].detach()
        actions_policy = actions[:, :-1]
        params = self.policy(agent_embed_policy)
        log_probs = self.policy.log_prob(params, actions_policy)

        prior_log_probs = None
        if self.prior_policy is not None:
            with torch.no_grad():
                prior_params = self.prior_policy(agent_embed_policy)
                prior_log_probs = self.prior_policy.log_prob(prior_params, actions_policy)

        loss, pmpo_info = pmpo_policy_loss(
            log_probs, advantages,
            prior_log_probs=prior_log_probs,
            alpha=cfg.pmpo_alpha,
            beta=cfg.pmpo_beta,
        )

        optimizers["policy"].zero_grad()
        loss.backward()
        grad_norm = _grad_norm(self.policy)
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        optimizers["policy"].step()

        return {
            "replay_policy_loss": loss.item(),
            "replay_mean_advantage": advantages.mean().item(),
            "replay_advantage_std": pmpo_info["advantage_std"],
            "replay_advantage_pos_frac": pmpo_info["advantage_pos_frac"],
            "replay_policy_grad_norm": grad_norm,
        }

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

        dynamics_raw = _unwrap(self.dynamics)
        z_hat, agent_out = dynamics_raw(
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

    # -- WM comparison video --------------------------------------------------

    @torch.no_grad()
    def generate_wm_comparison_video(
        self,
        replay: "ReplayBuffer",
        device: torch.device,
        n_context: int = 4,
        n_predict: int = 12,
    ) -> list[np.ndarray]:
        """Generate side-by-side real vs WM-predicted frames for logging.

        Returns a list of (H, W*2, 3) uint8 frames: left=real, right=predicted.
        """
        cfg = self.config
        seq_len = n_context + n_predict
        batch = replay.sample_sequence(1, seq_len, device=device)
        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)

        z_packed = self.encode_obs(obs)
        context = z_packed[:, :n_context]

        tok = _unwrap(self.tokenizer)
        dynamics_raw = _unwrap(self.dynamics)

        task_ids = torch.zeros(1, dtype=torch.long, device=device)
        predicted_frames: list[torch.Tensor] = []
        frames_list = list(context.unbind(dim=1))
        all_actions = list(actions[:, :n_context].unbind(dim=1))

        for h in range(n_predict):
            t_act = n_context + h
            if t_act < actions.shape[1]:
                all_actions.append(actions[:, t_act])
            else:
                all_actions.append(torch.zeros(1, self._action_dim, device=device))

            win = min(len(frames_list), cfg.policy_max_context)
            past = torch.stack(frames_list[-win:], dim=1)
            act_past = torch.stack(all_actions[-win:], dim=1)
            T_cur = past.shape[1]
            agent_tok = self.task_encoder(task_ids).expand(1, T_cur + 1, -1, -1)
            act_pad = torch.zeros(1, 1, self._action_dim, device=device)
            act_full = torch.cat([act_past, act_pad], dim=1)

            z_next, _ = sample_one_timestep(
                dynamics_raw, past_packed=past, k_max=cfg.k_max,
                K=cfg.imagination_K, actions=act_full,
                tau_ctx=cfg.imagination_tau_ctx, agent_tokens=agent_tok,
            )
            frames_list.append(z_next)
            predicted_frames.append(z_next)

        H = W = cfg.image_size
        C = cfg.channels
        ps = cfg.patch_size
        result_frames: list[np.ndarray] = []

        for t in range(n_predict):
            real_obs = obs[0, n_context + t]
            real_np = (real_obs.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            z_pred = predicted_frames[t].unsqueeze(1)
            bn = unpack_spatial_to_bottleneck(z_pred, k=cfg.packing_k).clamp(-1, 1)
            patches = tok.decoder(bn)
            img = unpatchify(patches, H, W, C, ps).clamp(0, 1)
            pred_np = (img[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            gap = np.ones((H, 2, 3), dtype=np.uint8) * 128
            combined = np.concatenate([real_np, gap, pred_np], axis=1)
            result_frames.append(combined)

        return result_frames

    # -- Tokenizer reconstruction grid ----------------------------------------

    @torch.no_grad()
    def generate_tokenizer_recon_grid(
        self,
        replay: "ReplayBuffer",
        device: torch.device,
        n_images: int = 8,
    ) -> list[torch.Tensor]:
        """Generate [input, reconstruction] image pairs for TensorBoard grid.

        Returns list of (C, H, W) tensors alternating input/recon.
        """
        cfg = self.config
        batch = replay.sample_sequence(n_images, 1, device=device)
        obs = batch["obs"].to(device)

        patches = patchify(obs, cfg.patch_size)
        tok = _unwrap(self.tokenizer)
        pred, _, _ = tok(patches)

        H = W = cfg.image_size
        C = cfg.channels
        ps = cfg.patch_size

        orig_imgs = unpatchify(patches, H, W, C, ps).clamp(0, 1)
        recon_imgs = unpatchify(pred, H, W, C, ps).clamp(0, 1)

        grid_images: list[torch.Tensor] = []
        for i in range(min(n_images, orig_imgs.shape[0])):
            grid_images.append(orig_imgs[i, 0].cpu())
            grid_images.append(recon_imgs[i, 0].cpu())

        return grid_images

    # -- Prior policy update -------------------------------------------------

    def update_prior_policy(self) -> None:
        """Refresh the frozen behavioral prior from the current policy."""
        self.prior_policy = make_prior_policy(self.policy)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    agent: Dreamer4Agent,
    env_fn: Callable[[], Any],
    n_episodes: int = 5,
    max_steps: int = 1000,
    record_video: bool = True,
    render_size: int = 128,
) -> dict[str, Any]:
    """Run the learned policy for evaluation (no gradient, no exploration).

    Args:
        agent:        Trained :class:`Dreamer4Agent` (will be set to eval mode).
        env_fn:       Factory callable returning an environment instance.
        n_episodes:   Number of evaluation episodes.
        max_steps:    Max steps per episode (safety cap).
        record_video: Whether to record RGB frames for video logging.
        render_size:  Resolution for recorded video frames.

    Returns:
        Dict with keys:
            - ``eval_return_mean``, ``eval_return_std``
            - ``eval_return_min``, ``eval_return_max``
            - ``eval_episode_length``
            - ``eval_frames``: list of lists of (H, W, 3) uint8 arrays
              (one list per episode), only if *record_video* is True.
    """
    agent.eval()
    env = env_fn()
    returns: list[float] = []
    lengths: list[int] = []
    all_frames: list[list[np.ndarray]] = []

    for _ in range(n_episodes):
        obs = env.reset()
        obs_history: list[torch.Tensor] = [obs]
        act_history: list[torch.Tensor] = []
        ep_return = 0.0
        ep_frames: list[np.ndarray] = []

        if record_video and hasattr(env, "render_rgb"):
            ep_frames.append(env.render_rgb(size=render_size))
        elif record_video:
            frame = (obs.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ep_frames.append(frame)

        for t in range(max_steps):
            action = agent.policy_action(obs_history, act_history)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward
            obs_history.append(obs)
            act_history.append(action)

            if record_video:
                if hasattr(env, "render_rgb"):
                    ep_frames.append(env.render_rgb(size=render_size))
                else:
                    frame = (obs.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    ep_frames.append(frame)

            if done:
                break

        returns.append(ep_return)
        lengths.append(t + 1)
        if record_video:
            all_frames.append(ep_frames)

    agent.train()

    result: dict[str, Any] = {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_return_min": float(np.min(returns)),
        "eval_return_max": float(np.max(returns)),
        "eval_episode_length": float(np.mean(lengths)),
    }
    if record_video:
        result["eval_frames"] = all_frames
    return result


def _unwrap(module: nn.Module) -> nn.Module:
    """Unwrap DataParallel if present."""
    if isinstance(module, nn.DataParallel):
        return module.module
    return module


def _grad_norm(module: nn.Module) -> float:
    """Total L2 norm of gradients across all parameters."""
    params = [p for p in module.parameters() if p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, float("inf")).item()


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------

def online_training_loop(
    agent: Dreamer4Agent,
    env_fns: Sequence[Callable[[], Any]],
    config: Optional[TrainConfig] = None,
    *,
    tokenizer: Optional[Tokenizer] = None,
    eval_env_fn: Optional[Callable[[], Any]] = None,
    log_fn: Optional[Callable[[int, dict[str, float]], None]] = None,
    log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    use_data_parallel: bool = False,
) -> dict[str, list[float]]:
    """Main Dreamer 4 online training loop.

    Follows the DreamerV3-style loop::

        1. Prefill replay buffer with random actions
        2. Loop:
           a. Collect env transitions with learned policy
           b. Train world model (tokenizer + dynamics + agent heads)
           c. Imagine rollouts, train policy + value

    Args:
        agent:              :class:`Dreamer4Agent` instance.
        env_fns:            Factory callables, each returning an environment.
        config:             Override config (defaults to ``agent.config``).
        tokenizer:          Optional pretrained tokenizer to attach.
        eval_env_fn:        Factory for evaluation environments (separate from train).
        log_fn:             Extra ``(step, metrics) -> None`` callback.
        log_dir:            TensorBoard log directory.
        checkpoint_dir:     Directory for periodic checkpoints.
        resume_from:        Path to a checkpoint to resume from.
        use_data_parallel:  Wrap dynamics & tokenizer with ``nn.DataParallel``.

    Returns:
        Dict mapping metric names to lists of values over training.
    """
    cfg = config or agent.config
    device = agent.device
    agent = agent.to(device)

    if tokenizer is not None:
        agent.set_tokenizer(tokenizer)
        agent.tokenizer = agent.tokenizer.to(device)

    # -- Multi-GPU DataParallel --
    n_gpus = torch.cuda.device_count()
    if use_data_parallel and n_gpus > 1:
        logger.info("Wrapping dynamics and tokenizer with DataParallel (%d GPUs)", n_gpus)
        agent.dynamics = nn.DataParallel(agent.dynamics)
        if agent.tokenizer is not None:
            agent.tokenizer = nn.DataParallel(agent.tokenizer)

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
    prefill_episodes, prefill_transitions = driver.collect_random(cfg.prefill_steps)
    for ep in prefill_episodes:
        replay.add_episode(ep.obs, ep.actions, ep.rewards)
    total_env_steps += prefill_transitions
    logger.info(
        "Prefill complete: %d episodes, %d transitions (env_steps=%d)",
        replay.total_episodes, replay.total_transitions, total_env_steps,
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

        episodes, n_transitions = driver.collect(cfg.train_every, policy_fn)
        total_env_steps += n_transitions
        for ep in episodes:
            replay.add_episode(ep.obs, ep.actions, ep.rewards)

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

        # -- 2d. Train policy on real environment data --
        replay_policy_metrics: dict[str, float] = {}
        if train_step % 2 == 0:
            batch = replay.sample_sequence(
                cfg.batch_size, cfg.seq_len, device=device,
            )
            replay_policy_metrics = agent.train_policy_on_replay(batch, optimizers)

        train_step += 1

        # -- Update prior policy periodically --
        if train_step % cfg.prior_update_interval == 0:
            agent.update_prior_policy()

        # -- Logging --
        if train_step % cfg.log_every == 0:
            all_metrics = {**wm_metrics, **im_metrics, **replay_policy_metrics}
            all_metrics["env_steps"] = float(total_env_steps)
            all_metrics["episodes_collected"] = float(len(episodes))
            if episodes:
                all_metrics["episode_return"] = float(
                    sum(ep.total_return for ep in episodes) / len(episodes)
                )
            _log(train_step, all_metrics)

        # -- Evaluation --
        if (
            eval_env_fn is not None
            and cfg.eval_every > 0
            and train_step % cfg.eval_every == 0
        ):
            logger.info("Running evaluation at step %d...", train_step)
            eval_result = evaluate(
                agent, eval_env_fn,
                n_episodes=5,
                max_steps=cfg.time_limit,
                record_video=True,
            )
            eval_scalars = {
                k: v for k, v in eval_result.items() if isinstance(v, float)
            }
            _log(train_step, eval_scalars)
            logger.info(
                "Eval: return=%.2f +/- %.2f (min=%.2f, max=%.2f, len=%.0f)",
                eval_scalars["eval_return_mean"],
                eval_scalars["eval_return_std"],
                eval_scalars["eval_return_min"],
                eval_scalars["eval_return_max"],
                eval_scalars["eval_episode_length"],
            )
            if metrics_logger is not None and "eval_frames" in eval_result:
                best_idx = int(np.argmax([
                    sum(1 for _ in frames) for frames in eval_result["eval_frames"]
                ]))
                best_frames = eval_result["eval_frames"][best_idx]
                if best_frames:
                    metrics_logger.log_video(
                        train_step, "eval/episode_video", best_frames, fps=15,
                    )

        # -- WM prediction video --
        if (
            metrics_logger is not None
            and cfg.wm_video_every > 0
            and total_env_steps % cfg.wm_video_every < cfg.train_every
            and total_env_steps > cfg.prefill_steps + cfg.wm_video_every
        ):
            logger.info("Generating WM comparison video at step %d...", train_step)
            try:
                wm_frames = agent.generate_wm_comparison_video(replay, device)
                if wm_frames:
                    metrics_logger.log_video(
                        train_step, "wm/prediction_vs_real", wm_frames, fps=5,
                    )
            except Exception as e:
                logger.warning("WM video generation failed: %s", e)

        # -- Tokenizer reconstruction grid --
        if (
            metrics_logger is not None
            and cfg.eval_every > 0
            and train_step % cfg.eval_every == 0
            and agent.tokenizer is not None
        ):
            try:
                grid_imgs = agent.generate_tokenizer_recon_grid(replay, device)
                if grid_imgs:
                    metrics_logger.log_images_grid(
                        train_step, "wm/tokenizer_recon", grid_imgs, nrow=4,
                    )
            except Exception as e:
                logger.warning("Tokenizer recon grid failed: %s", e)

        # -- Checkpointing --
        if auto_ckpt is not None:
            auto_ckpt.step(
                train_step, agent, optimizers,
                env_steps=total_env_steps, config=cfg,
                metrics={**wm_metrics, **im_metrics},
            )

    driver.close()

    # -- Unwrap DataParallel before saving --
    if isinstance(agent.dynamics, nn.DataParallel):
        agent.dynamics = agent.dynamics.module
    if agent.tokenizer is not None and isinstance(agent.tokenizer, nn.DataParallel):
        agent.tokenizer = agent.tokenizer.module

    # -- Final evaluation --
    if eval_env_fn is not None:
        logger.info("Running final evaluation (10 episodes)...")
        final_eval = evaluate(
            agent, eval_env_fn,
            n_episodes=10,
            max_steps=cfg.time_limit,
            record_video=True,
        )
        final_scalars = {k: v for k, v in final_eval.items() if isinstance(v, float)}
        _log(train_step, {f"final_{k}": v for k, v in final_scalars.items()})
        logger.info(
            "Final eval: return=%.2f +/- %.2f",
            final_scalars["eval_return_mean"],
            final_scalars["eval_return_std"],
        )
        if metrics_logger is not None and "eval_frames" in final_eval:
            returns = []
            for i, frames in enumerate(final_eval["eval_frames"]):
                returns.append(len(frames))
            best_idx = int(np.argmax(returns))
            metrics_logger.log_video(
                train_step, "eval/final_video",
                final_eval["eval_frames"][best_idx], fps=15,
            )

        history["final_eval"] = [final_scalars]

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
