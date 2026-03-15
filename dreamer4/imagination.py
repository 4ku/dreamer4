"""
Imagination training for Dreamer 4 (Paper Section 3.3, Algorithm 1 Phase 3).

Generates rollouts inside the world model and trains policy + value heads
via PMPO and TD(lambda) respectively, with the dynamics transformer frozen.

    from dreamer4.imagination import imagine_rollout, imagination_training_step

Flow:
    1. ``imagine_rollout`` starts from dataset context latents, unrolls the
       dynamics model autoregressively while sampling actions from the policy.
    2. ``imagination_training_step`` computes lambda-returns from the rollout
       and produces PMPO policy loss + value TD loss for a single gradient step.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from dreamer4.dynamics import DynamicsModel, sample_one_timestep
from dreamer4.agent import (
    PolicyHead,
    RewardHead,
    TaskEncoder,
    ValueHead,
    compute_lambda_returns,
    pmpo_policy_loss,
)


def _grad_norm(module: nn.Module) -> float:
    """Total L2 norm of gradients across all parameters."""
    params = [p for p in module.parameters() if p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, float("inf")).item()


@dataclass
class ImaginedTrajectory:
    """Data produced by a single imagination rollout.

    All tensors have batch dimension B and time dimension H (horizon).
    """

    latents: torch.Tensor        # (B, H, n_spatial, d_spatial)
    actions: torch.Tensor        # (B, H, action_dim)  — in (-1, 1) via TanhNormal
    log_probs: torch.Tensor      # (B, H)
    rewards: torch.Tensor        # (B, H)
    values: torch.Tensor         # (B, H)
    continues: torch.Tensor      # (B, H)  — 1.0 for non-terminal
    agent_embeds: torch.Tensor   # (B, H, d_model)


@torch.no_grad()
def imagine_rollout(
    dynamics: DynamicsModel,
    policy: PolicyHead,
    reward_head: RewardHead,
    value_head: ValueHead,
    task_encoder: TaskEncoder,
    *,
    context: torch.Tensor,
    context_actions: Optional[torch.Tensor] = None,
    task_ids: Optional[torch.Tensor] = None,
    horizon: int = 15,
    k_max: int = 4,
    K: int = 4,
    tau_ctx: float = 0.1,
    max_context_window: int = 16,
) -> ImaginedTrajectory:
    """Generate imagined trajectories inside the world model.

    Paper Section 3.3: "Imagined rollouts start from contexts of the dataset.
    We start only one rollout from each context, prioritizing data diversity."

    Uses a sliding context window to avoid quadratic memory growth.

    Args:
        dynamics:            Dynamics model (frozen, eval mode).
        policy:              Policy head for action sampling.
        reward_head:         Reward head for trajectory annotation.
        value_head:          Value head for trajectory annotation.
        task_encoder:        Task encoder for agent token inputs.
        context:             (B, t_ctx, n_spatial, d_spatial) tokenized context.
        context_actions:     (B, t_ctx, action_dim) or None.
        task_ids:            (B,) long task indices, or None (defaults to 0).
        horizon:             Number of steps to imagine.
        k_max:               Maximum shortcut sampling steps.
        K:                   Denoising steps per generated frame.
        tau_ctx:             Context corruption signal level.
        max_context_window:  Maximum frames kept as context (sliding window).

    Returns:
        ImaginedTrajectory with H=horizon time steps.
    """
    device = context.device
    B, t_ctx = context.shape[:2]
    action_dim = dynamics.action_encoder.action_dim

    if task_ids is None:
        task_ids = torch.zeros(B, dtype=torch.long, device=device)
    agent_tok_base = task_encoder(task_ids)  # (B, 1, n_agent, d_model)

    frames: list[torch.Tensor] = [context[:, t] for t in range(t_ctx)]

    all_actions_list: list[torch.Tensor] = []
    if context_actions is not None:
        for t in range(t_ctx):
            all_actions_list.append(context_actions[:, t])
    else:
        for _ in range(t_ctx):
            all_actions_list.append(torch.zeros(B, action_dim, device=device))

    latents_out: list[torch.Tensor] = []
    actions_out: list[torch.Tensor] = []
    log_probs_out: list[torch.Tensor] = []
    rewards_out: list[torch.Tensor] = []
    values_out: list[torch.Tensor] = []
    agent_embeds_out: list[torch.Tensor] = []

    for h in range(horizon):
        # Sliding window: keep at most max_context_window frames
        win_start = max(0, len(frames) - max_context_window)
        past_frames = frames[win_start:]
        past_actions = all_actions_list[win_start:]

        past = torch.stack(past_frames, dim=1)  # (B, W, Nz, Dz)
        T_cur = past.shape[1]

        all_act = torch.stack(past_actions, dim=1)  # (B, W, A)
        placeholder_act = torch.zeros(B, 1, action_dim, device=device)
        act_in = torch.cat([all_act, placeholder_act], dim=1)  # (B, W+1, A)

        agent_tok = agent_tok_base.expand(B, T_cur + 1, -1, -1)

        z_next, a_out = sample_one_timestep(
            dynamics,
            past_packed=past,
            k_max=k_max,
            K=K,
            actions=act_in,
            tau_ctx=tau_ctx,
            agent_tokens=agent_tok,
        )

        agent_embed = a_out.mean(dim=-2)  # (B, d_model)

        reward_t = reward_head.predict(agent_embed)  # (B,)
        value_t = value_head.predict(agent_embed)    # (B,)

        policy_params = policy.forward(agent_embed, head_idx=0)
        action_t, lp_t = policy.sample(policy_params)

        frames.append(z_next)
        all_actions_list.append(action_t)

        latents_out.append(z_next)
        actions_out.append(action_t)
        log_probs_out.append(lp_t)
        rewards_out.append(reward_t)
        values_out.append(value_t)
        agent_embeds_out.append(agent_embed)

    return ImaginedTrajectory(
        latents=torch.stack(latents_out, dim=1),
        actions=torch.stack(actions_out, dim=1),
        log_probs=torch.stack(log_probs_out, dim=1),
        rewards=torch.stack(rewards_out, dim=1),
        values=torch.stack(values_out, dim=1),
        continues=torch.ones(B, horizon, device=device),
        agent_embeds=torch.stack(agent_embeds_out, dim=1),
    )


def imagination_training_step(
    trajectory: ImaginedTrajectory,
    policy: PolicyHead,
    value_head: ValueHead,
    policy_optim: torch.optim.Optimizer,
    value_optim: torch.optim.Optimizer,
    *,
    prior_policy: Optional[PolicyHead] = None,
    gamma: float = 0.997,
    lam: float = 0.95,
    beta: float = 0.3,
    entropy_scale: float = 3e-2,
    max_grad_norm: float = 100.0,
    policy_max_grad_norm: float = 10.0,
) -> dict[str, float]:
    """One gradient step of imagination training (Phase 3).

    Only updates ``policy`` and ``value_head``; the dynamics model and
    reward head are assumed frozen (their outputs are baked into the
    trajectory).

    Args:
        trajectory:     Output of ``imagine_rollout``.
        policy:         Policy head (trainable).
        value_head:     Value head (trainable).
        policy_optim:   Optimizer for the policy head.
        value_optim:    Optimizer for the value head.
        prior_policy:   Frozen behavioral prior (for KL in PMPO).
        gamma:          Discount factor for lambda-returns.
        lam:            Lambda for TD(lambda).
        beta:           KL regularization weight.
        entropy_scale:  Weight on entropy bonus to prevent policy collapse.
        max_grad_norm:  Gradient clipping norm for value head.
        policy_max_grad_norm: Gradient clipping norm for policy head.

    Returns:
        Dict with scalar loss values for logging.
    """
    agent_embeds = trajectory.agent_embeds.detach()  # (B, H, D)
    actions = trajectory.actions.detach()            # (B, H, A)
    rewards = trajectory.rewards.detach()            # (B, H)
    continues = trajectory.continues.detach()        # (B, H)

    # -- Value target computation (no grad through targets) --
    with torch.no_grad():
        values_sg = value_head.predict(agent_embeds)
        lambda_returns = compute_lambda_returns(
            rewards, values_sg, continues, gamma=gamma, lam=lam,
        )

    # -- Value loss (Eq. 10): cross-entropy against lambda-return targets --
    value_logits = value_head(agent_embeds)
    val_loss = value_head.twohot.loss(
        value_logits.reshape(-1, value_logits.shape[-1]),
        lambda_returns.reshape(-1),
    )

    value_optim.zero_grad()
    val_loss.backward()
    value_grad_norm = _grad_norm(value_head)
    if max_grad_norm > 0:
        nn.utils.clip_grad_norm_(value_head.parameters(), max_grad_norm)
    value_optim.step()

    # -- Policy loss with entropy regularization --
    advantages = (lambda_returns - values_sg).detach()

    policy_params = policy.forward(agent_embeds, head_idx=0)
    log_probs = policy.log_prob(policy_params, actions)

    prior_lp: Optional[torch.Tensor] = None
    if prior_policy is not None:
        with torch.no_grad():
            prior_params = prior_policy.forward(agent_embeds, head_idx=0)
            prior_lp = prior_policy.log_prob(prior_params, actions)

    pol_loss, pmpo_info = pmpo_policy_loss(
        log_probs, advantages,
        prior_log_probs=prior_lp,
        beta=beta,
    )

    entropy = policy.entropy(policy_params).mean()
    pol_loss = pol_loss - entropy_scale * entropy

    policy_optim.zero_grad()
    pol_loss.backward()

    policy_grad_norm = _grad_norm(policy)
    if policy_max_grad_norm > 0:
        nn.utils.clip_grad_norm_(policy.parameters(), policy_max_grad_norm)
    policy_optim.step()

    return {
        "policy_loss": pol_loss.item(),
        "value_loss": val_loss.item(),
        "policy_entropy": entropy.item(),
        "mean_reward": rewards.mean().item(),
        "mean_value": values_sg.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "mean_lambda_return": lambda_returns.mean().item(),
        "policy_grad_norm": policy_grad_norm,
        "value_grad_norm": value_grad_norm,
        "action_mean": actions.mean().item(),
        "action_std": actions.std().item(),
        **pmpo_info,
    }


def make_prior_policy(policy: PolicyHead) -> PolicyHead:
    """Create a frozen copy of the policy head to serve as the behavioral prior.

    Paper Section 3.3 (Eq. 11): "a frozen copy of the policy head that
    serves as a behavioral prior."
    """
    prior = copy.deepcopy(policy)
    for p in prior.parameters():
        p.requires_grad_(False)
    prior.eval()
    return prior
