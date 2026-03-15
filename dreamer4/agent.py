"""
Agent heads for Dreamer 4: policy, reward, and value prediction.

This module implements the three heads that consume agent token embeddings
from the dynamics transformer, plus the objective functions for training them.

Phase 2 (Agent Finetuning — paper Section 3.3, Eq. 9):
    - TaskEncoder produces agent token inputs conditioned on task identity.
    - PolicyHead predicts actions via multi-token prediction (MTP, L=8).
    - RewardHead predicts rewards via MTP with SymExpTwoHot output.

Phase 3 (Imagination Training — paper Section 3.3, Eq. 10-11):
    - ValueHead predicts discounted returns with SymExpTwoHot output.
    - TD(lambda) returns computed from imagined rollouts.
    - PMPO policy objective with reverse KL to behavioral prior.

Architecture note: agent tokens attend to all other modalities in the
dynamics transformer (wm_agent mode), but no other modality can attend
back to agent tokens, preventing causal confusion of the world model.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from dreamer4.distributions import SymExpTwoHot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mlp(d_in: int, d_hidden: int, d_out: int, depth: int) -> nn.Sequential:
    """MLP with LayerNorm + SiLU activations, `depth` hidden layers."""
    layers: list[nn.Module] = []
    prev = d_in
    for _ in range(depth):
        layers += [nn.Linear(prev, d_hidden), nn.LayerNorm(d_hidden), nn.SiLU()]
        prev = d_hidden
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Task Encoder
# ---------------------------------------------------------------------------

class TaskEncoder(nn.Module):
    """
    Produces agent token inputs from task identifiers.

    Each task is mapped to a d_model embedding via a lookup table and summed
    with a learned positional embedding per agent slot. The result is passed
    as ``agent_tokens`` to ``DynamicsModel.forward()``.

    Args:
        num_tasks: Number of distinct tasks.
        d_model:   Transformer hidden dimension.
        n_agent:   Number of agent tokens per time step.
    """

    def __init__(self, *, num_tasks: int, d_model: int, n_agent: int = 1):
        super().__init__()
        self.n_agent = n_agent
        self.task_embed = nn.Embedding(num_tasks, d_model)
        self.agent_pos = nn.Parameter(torch.empty(n_agent, d_model))
        nn.init.normal_(self.agent_pos, std=0.02)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_ids: (B,) or (B, T) long — task indices.

        Returns:
            (B, T, n_agent, d_model) agent token inputs (broadcastable
            over T if task_ids is 1-D).
        """
        emb = self.task_embed(task_ids)  # (B, D) or (B, T, D)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)  # (B, 1, D) — broadcast over T
        # (B, T, 1, D) + (n_agent, D) -> (B, T, n_agent, D)
        return emb.unsqueeze(-2) + self.agent_pos


# ---------------------------------------------------------------------------
# Policy Head
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """
    Maps agent embeddings to action distributions.

    Supports continuous (TanhNormal) and discrete (Categorical) action spaces.
    For continuous actions, uses a Normal distribution squashed through tanh
    to naturally produce actions in (-1, 1) with correct log_prob Jacobian.
    Implements multi-token prediction (MTP) with ``L`` output heads that
    share an MLP trunk but have separate final linear layers, one per
    prediction distance n=0..L-1 (paper Eq. 9).

    Args:
        d_model:    Input dimension (agent embedding size).
        action_dim: Number of action dimensions.
        action_type: "continuous" or "discrete".
        num_categories: For discrete actions, number of categories per dim.
        mtp_length: Multi-token prediction horizon L (default 8).
        mlp_depth:  Number of hidden layers in the MLP trunk.
        mlp_ratio:  Hidden dimension multiplier.
        min_std:    Minimum pre-tanh standard deviation for continuous actions.
        max_std:    Maximum pre-tanh standard deviation for continuous actions.
    """

    def __init__(
        self,
        *,
        d_model: int,
        action_dim: int,
        action_type: str = "continuous",
        num_categories: int = 0,
        mtp_length: int = 8,
        mlp_depth: int = 3,
        mlp_ratio: float = 4.0,
        min_std: float = 0.01,
        max_std: float = 2.0,
        mean_scale: float = 1.5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        self.num_categories = num_categories
        self.mtp_length = mtp_length
        self.min_std = min_std
        self.max_std = max_std
        self.mean_scale = mean_scale

        d_hidden = int(d_model * mlp_ratio)

        trunk_layers: list[nn.Module] = []
        prev = d_model
        for _ in range(mlp_depth):
            trunk_layers += [nn.Linear(prev, d_hidden), nn.LayerNorm(d_hidden), nn.SiLU()]
            prev = d_hidden
        self.trunk = nn.Sequential(*trunk_layers)

        if action_type == "continuous":
            out_dim = action_dim * 2  # mean + log_std
        elif action_type == "discrete":
            assert num_categories > 0
            out_dim = action_dim * num_categories
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        self.heads = nn.ModuleList([
            nn.Linear(d_hidden, out_dim) for _ in range(mtp_length)
        ])

    def forward(
        self, agent_embed: torch.Tensor, head_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute raw distribution parameters for one MTP head.

        Args:
            agent_embed: (*, d_model) agent token embeddings.
            head_idx:    Which MTP head to use (0 = current step).

        Returns:
            (*,  action_dim * 2) for continuous (mean, log_std).
            (*, action_dim * num_categories) for discrete (logits).
        """
        h = self.trunk(agent_embed)
        return self.heads[head_idx](h)

    def forward_all(
        self, agent_embed: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Compute distribution parameters for all MTP heads.

        Args:
            agent_embed: (*, d_model).

        Returns:
            List of L tensors, each (*, out_dim).
        """
        h = self.trunk(agent_embed)
        return [head(h) for head in self.heads]

    def _base_normal(self, params: torch.Tensor) -> Normal:
        """Build the pre-tanh Normal from raw network output."""
        mean_raw, std_raw = params.split(self.action_dim, dim=-1)
        mean = self.mean_scale * torch.tanh(mean_raw / self.mean_scale)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std_raw)
        return Normal(mean, std)

    def dist(
        self, params: torch.Tensor
    ) -> TransformedDistribution | Categorical:
        """
        Build a distribution from raw parameters.

        For continuous actions returns a TanhNormal distribution
        (Normal squashed through tanh), which naturally produces
        actions in (-1, 1) with correct log_prob Jacobian correction.

        Args:
            params: Output of ``forward()``.

        Returns:
            TransformedDistribution (continuous) or Categorical (discrete).
        """
        if self.action_type == "continuous":
            base = self._base_normal(params)
            return TransformedDistribution(base, [TanhTransform(cache_size=1)])
        else:
            logits = params.reshape(*params.shape[:-1],
                                    self.action_dim, self.num_categories)
            return Categorical(logits=logits)

    def log_prob(
        self, params: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Log probability of actions under the distribution.

        For continuous actions, clamps inputs to (-1+eps, 1-eps) to
        avoid infinite log_prob at the tanh boundaries.

        Args:
            params:  Output of ``forward()``.
            actions: (*, action_dim) for continuous (in [-1,1]);
                     (*, action_dim) long for discrete.

        Returns:
            (*) total log probability (summed over action dimensions).
        """
        if self.action_type == "continuous":
            actions = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        d = self.dist(params)
        return d.log_prob(actions).sum(dim=-1)

    def sample(
        self, params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions and compute their log probabilities.

        For continuous actions, uses TanhNormal so actions are
        naturally in (-1, 1) with correct Jacobian-corrected log_prob.

        Args:
            params: Output of ``forward()``.

        Returns:
            actions:   (*, action_dim) in (-1, 1) for continuous.
            log_probs: (*) total log probability.
        """
        d = self.dist(params)
        if self.action_type == "continuous":
            actions = d.rsample()
            lp = d.log_prob(actions).sum(dim=-1)
            return actions, lp
        else:
            actions = d.sample()
            lp = d.log_prob(actions).sum(dim=-1)
            return actions, lp

    def entropy(self, params: torch.Tensor) -> torch.Tensor:
        """
        Entropy of the distribution (summed over action dims).

        For continuous (TanhNormal) actions, uses a sample-based estimate:
        H(TanhNormal) = H(Normal) - E[sum(log(1 - tanh(x)^2))]
        where x ~ Normal(mean, std). This correctly accounts for the
        tanh squashing, which saturates the entropy as std grows large.

        Args:
            params: Output of ``forward()``.

        Returns:
            (*) entropy estimate.
        """
        if self.action_type == "continuous":
            base = self._base_normal(params)
            base_entropy = base.entropy().sum(dim=-1)
            x = base.rsample()
            log_det = torch.log1p(-x.tanh().pow(2) + 1e-6).sum(dim=-1)
            return base_entropy + log_det
        d = self.dist(params)
        return d.entropy().sum(dim=-1)


# ---------------------------------------------------------------------------
# Reward Head
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """
    Predicts scalar rewards via MLP + SymExpTwoHot.

    Uses multi-token prediction (MTP): one output layer per prediction
    distance, sharing the MLP trunk. Each output layer produces logits
    over SymExpTwoHot bins.

    Args:
        d_model:    Input dimension.
        mtp_length: Multi-token prediction horizon L.
        mlp_depth:  Hidden layers in MLP trunk.
        mlp_ratio:  Hidden dimension multiplier.
        num_bins:   Bins for SymExpTwoHot.
        bin_low:    Lower bound in symlog space.
        bin_high:   Upper bound in symlog space.
    """

    def __init__(
        self,
        *,
        d_model: int,
        mtp_length: int = 8,
        mlp_depth: int = 3,
        mlp_ratio: float = 4.0,
        num_bins: int = 255,
        bin_low: float = -20.0,
        bin_high: float = 20.0,
    ):
        super().__init__()
        self.mtp_length = mtp_length
        self.twohot = SymExpTwoHot(num_bins=num_bins, low=bin_low, high=bin_high)

        d_hidden = int(d_model * mlp_ratio)
        trunk_layers: list[nn.Module] = []
        prev = d_model
        for _ in range(mlp_depth):
            trunk_layers += [nn.Linear(prev, d_hidden), nn.LayerNorm(d_hidden), nn.SiLU()]
            prev = d_hidden
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList([
            nn.Linear(d_hidden, num_bins) for _ in range(mtp_length)
        ])

    def forward(
        self, agent_embed: torch.Tensor, head_idx: int = 0
    ) -> torch.Tensor:
        """
        Predict reward logits for one MTP distance.

        Args:
            agent_embed: (*, d_model).
            head_idx:    MTP prediction distance.

        Returns:
            (*, num_bins) raw logits.
        """
        h = self.trunk(agent_embed)
        return self.heads[head_idx](h)

    def forward_all(
        self, agent_embed: torch.Tensor
    ) -> list[torch.Tensor]:
        """Predict reward logits for all MTP distances."""
        h = self.trunk(agent_embed)
        return [head(h) for head in self.heads]

    def predict(
        self, agent_embed: torch.Tensor, head_idx: int = 0
    ) -> torch.Tensor:
        """
        Predict scalar reward (decode from logits).

        Args:
            agent_embed: (*, d_model).
            head_idx:    MTP prediction distance.

        Returns:
            (*) scalar rewards.
        """
        logits = self.forward(agent_embed, head_idx)
        return self.twohot.decode(logits)


# ---------------------------------------------------------------------------
# Value Head
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """
    Predicts scalar state values via MLP + SymExpTwoHot.

    Unlike RewardHead, this has no MTP — it predicts a single value per
    time step, trained with TD(lambda) targets (paper Eq. 10).

    Args:
        d_model:   Input dimension.
        mlp_depth: Hidden layers in MLP.
        mlp_ratio: Hidden dimension multiplier.
        num_bins:  Bins for SymExpTwoHot.
        bin_low:   Lower bound in symlog space.
        bin_high:  Upper bound in symlog space.
    """

    def __init__(
        self,
        *,
        d_model: int,
        mlp_depth: int = 3,
        mlp_ratio: float = 4.0,
        num_bins: int = 255,
        bin_low: float = -20.0,
        bin_high: float = 20.0,
    ):
        super().__init__()
        self.twohot = SymExpTwoHot(num_bins=num_bins, low=bin_low, high=bin_high)

        d_hidden = int(d_model * mlp_ratio)
        self.mlp = _build_mlp(d_model, d_hidden, num_bins, mlp_depth)

    def forward(self, agent_embed: torch.Tensor) -> torch.Tensor:
        """
        Predict value logits.

        Args:
            agent_embed: (*, d_model).

        Returns:
            (*, num_bins) raw logits.
        """
        return self.mlp(agent_embed)

    def predict(self, agent_embed: torch.Tensor) -> torch.Tensor:
        """
        Predict scalar values (decode from logits).

        Args:
            agent_embed: (*, d_model).

        Returns:
            (*) scalar values.
        """
        return self.twohot.decode(self.forward(agent_embed))


# ---------------------------------------------------------------------------
# Objective Functions
# ---------------------------------------------------------------------------

def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    *,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Compute TD(lambda) returns (paper Eq. 10).

        R_t^lambda = r_t + gamma * c_t * ((1 - lam) * v_{t+1} + lam * R_{t+1}^lambda)
        R_T^lambda = v_T

    The recursion is computed backwards from t=T to t=1.

    Args:
        rewards:   (B, T) scalar rewards.
        values:    (B, T) predicted values (from the value head).
        continues: (B, T) float — 1.0 for non-terminal, 0.0 for terminal.
        gamma:     Discount factor.
        lam:       Lambda for TD(lambda) mixing.

    Returns:
        (B, T) lambda-return targets.
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)

    # Bootstrap from the last value
    returns[:, -1] = values[:, -1]

    for t in reversed(range(T - 1)):
        returns[:, t] = (
            rewards[:, t]
            + gamma * continues[:, t] * (
                (1.0 - lam) * values[:, t + 1]
                + lam * returns[:, t + 1]
            )
        )

    return returns


def pmpo_policy_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    prior_log_probs: Optional[torch.Tensor] = None,
    beta: float = 0.3,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """
    REINFORCE policy loss with percentile + unit-variance advantage normalization.

    First applies percentile normalization (5th-95th range) to remove outliers,
    then normalizes to unit variance so the policy gradient has consistent
    magnitude regardless of the absolute advantage scale.

    Args:
        log_probs:       (B, T) log pi_theta(a|s) under the current policy.
        advantages:      (B, T) A_t = R_t^lambda - v_t.
        prior_log_probs: (B, T) log pi_prior(a|s) from behavioral prior.
        beta:            KL regularization strength.

    Returns:
        (loss, info) where info contains diagnostic scalars.
    """
    adv = advantages.detach()

    pct_5 = torch.quantile(adv, 0.05)
    pct_95 = torch.quantile(adv, 0.95)
    scale = (pct_95 - pct_5).clamp_min(1e-8)
    adv = adv / scale

    adv = adv / adv.std().clamp_min(1e-5)

    loss = -(log_probs * adv).mean()

    kl_val = 0.0
    if prior_log_probs is not None and beta > 0:
        kl = (log_probs - prior_log_probs).clamp(-5.0, 5.0).mean()
        kl_val = kl.item()
        loss = loss + beta * kl

    pos_frac = (adv >= 0).float().sum() / max(adv.numel(), 1)

    info = {
        "advantage_std": adv.std().item(),
        "advantage_pos_frac": pos_frac.item(),
        "kl_to_prior": kl_val,
        "log_prob_mean": log_probs.mean().item(),
    }

    return loss, info


def behavior_cloning_loss(
    policy_head: PolicyHead,
    agent_embed: torch.Tensor,
    target_actions: torch.Tensor,
    mtp_length: int = 8,
) -> torch.Tensor:
    """
    Behavior cloning loss with multi-token prediction (paper Eq. 9).

    Computes the negative log-likelihood of target actions under each MTP
    head, averaged over all heads and valid time steps.

        L = -(1/L) * sum_{n=0}^{L-1} log p(a_{t+n} | h_t)

    Args:
        policy_head:    PolicyHead instance.
        agent_embed:    (B, T, d_model) agent token embeddings.
        target_actions: (B, T, action_dim) ground-truth actions.
        mtp_length:     MTP horizon (should match policy_head.mtp_length).

    Returns:
        Scalar BC loss.
    """
    L = min(mtp_length, policy_head.mtp_length)
    B, T = agent_embed.shape[:2]

    all_params = policy_head.forward_all(agent_embed)
    total_loss = torch.tensor(0.0, device=agent_embed.device)
    count = 0

    for n in range(L):
        valid_T = T - n
        if valid_T <= 0:
            break
        params_n = all_params[n][:, :valid_T]        # (B, valid_T, ...)
        actions_n = target_actions[:, n:n + valid_T]  # (B, valid_T, action_dim)
        lp = policy_head.log_prob(params_n, actions_n)
        total_loss = total_loss - lp.mean()
        count += 1

    return total_loss / max(count, 1)


def reward_prediction_loss(
    reward_head: RewardHead,
    agent_embed: torch.Tensor,
    target_rewards: torch.Tensor,
    mtp_length: int = 8,
) -> torch.Tensor:
    """
    Reward prediction loss with multi-token prediction (paper Eq. 9).

    Computes cross-entropy of SymExpTwoHot logits against target rewards
    for each MTP distance, averaged over all heads and valid time steps.

    Args:
        reward_head:    RewardHead instance.
        agent_embed:    (B, T, d_model) agent token embeddings.
        target_rewards: (B, T) scalar ground-truth rewards.
        mtp_length:     MTP horizon.

    Returns:
        Scalar reward prediction loss.
    """
    L = min(mtp_length, reward_head.mtp_length)
    B, T = agent_embed.shape[:2]

    all_logits = reward_head.forward_all(agent_embed)
    total_loss = torch.tensor(0.0, device=agent_embed.device)
    count = 0

    for n in range(L):
        valid_T = T - n
        if valid_T <= 0:
            break
        logits_n = all_logits[n][:, :valid_T]          # (B, valid_T, num_bins)
        rewards_n = target_rewards[:, n:n + valid_T]    # (B, valid_T)
        loss_n = reward_head.twohot.loss(logits_n, rewards_n)
        total_loss = total_loss + loss_n
        count += 1

    return total_loss / max(count, 1)
