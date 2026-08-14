"""
Training objectives for the agent heads (phases 2-3) — ALL of them.

Phase 2 (agent finetuning, paper Eq. 9): masked multi-token-prediction
losses for behavior cloning and reward prediction. The masks exist because
gridworld windows are short (T=4) and may end at the episode boundary: each
target index is real or padded per row, and BC rows are additionally
weighted by data quality (the gridworld analog of the paper's "BC loss only
on the task-relevant fraction" — see the trainer's noisiness filter).

Phase 3 (imagination training, paper Eq. 10-11): TD(lambda) returns and the
paper's PMPO policy objective — a pure sign-of-advantage split pooled across
batch x time (alpha = 0.5), plus a reverse KL to a frozen behavioral prior
(beta = 0.3). NO advantage normalization and no entropy bonus: the paper is
explicit that PMPO makes both unnecessary (Appendix F) — an earlier
percentile + unit-variance REINFORCE variant lived here until 2026-08-11 and
was deleted unused; git history has it if an A/B is ever wanted.

Alignment convention (matches the data contract in ``dreamer4.data.base``):
``a[t]`` / ``r[t]`` / ``terminals[t]`` describe the ``t -> t+1`` transition,
so the head at embed position ``t`` with MTP distance ``n`` predicts target
index ``t + n``. ``continues[t]`` describes FRAME t (1 = not terminal),
built from the data by :func:`continues_from_terminals`. In dreams there
are no recorded terminals, so the flag must be PREDICTED — how, is a
property of the environment, so it lives on the dataset
(``EpisodeVideoDataset.continues_from_reward``), not here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from dreamer4.models.agent import PolicyHead, RewardHead


# ---------------------------------------------------------------------------
# Phase 2: masked MTP losses (paper Eq. 9)
# ---------------------------------------------------------------------------


def behavior_cloning_loss(
    policy_head: PolicyHead,
    agent_embed: torch.Tensor,
    target_actions: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    row_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-token behavior cloning: head n at position t predicts a[t+n].

    Args:
        agent_embed:    (B, Te, d_model) policy-slot embeddings.
        target_actions: (B, Ta) long (discrete) or (B, Ta, Da) float —
                        target index t aligns with embed position t at n=0.
        target_mask:    (B, Ta) float — 1 where the target index is real
                        (windows at the episode end have a padded tail).
        row_weight:     (B,) float — per-row weight (the BC data-quality
                        filter); None = all rows.

    Returns:
        (scalar loss, info) — info carries ``bc_acc`` (n=0 argmax accuracy,
        discrete only) and ``bc_targets`` (weighted target count).
    """
    B, Te = agent_embed.shape[:2]
    Ta = target_actions.shape[1]
    L = policy_head.mtp_length
    w_row = (torch.ones(B, device=agent_embed.device)
             if row_weight is None else row_weight.float())

    all_params = policy_head.forward_all(agent_embed)
    total = agent_embed.new_zeros(())
    heads_used = 0
    info: Dict[str, float] = {}
    for n in range(L):
        Tv = min(Te, Ta - n)
        if Tv <= 0:
            break
        params_n = all_params[n][:, :Tv]
        actions_n = target_actions[:, n:n + Tv]
        weight = target_mask[:, n:n + Tv].float() * w_row[:, None]  # (B, Tv)
        wsum = weight.sum()
        if float(wsum) <= 0:
            continue
        lp = policy_head.log_prob(params_n, actions_n)              # (B, Tv)
        total = total - (weight * lp).sum() / wsum
        heads_used += 1
        if n == 0 and policy_head.action_type == "discrete":
            logits = params_n.reshape(*params_n.shape[:-1],
                                      policy_head.action_dim,
                                      policy_head.num_categories)
            pred = logits.argmax(-1)                    # (B, Tv, action_dim)
            hits = (pred == actions_n).all(-1).float() * weight
            info["bc_acc"] = float(hits.sum() / wsum)
            info["bc_targets"] = float(wsum)
    return total / max(heads_used, 1), info


def reward_prediction_loss(
    reward_head: RewardHead,
    agent_embed: torch.Tensor,
    target_rewards: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-token reward prediction: head n at position t predicts r[t+n].

    Args:
        agent_embed:    (B, Te, d_model) reward-slot embeddings.
        target_rewards: (B, Ta) scalar rewards (index t <-> embed t at n=0).
        target_mask:    (B, Ta) float validity mask.

    Returns:
        (scalar loss, info) — info carries ``reward_mae`` (n=0, masked).
    """
    B, Te = agent_embed.shape[:2]
    Ta = target_rewards.shape[1]
    L = reward_head.mtp_length

    all_logits = reward_head.forward_all(agent_embed)
    total = agent_embed.new_zeros(())
    heads_used = 0
    info: Dict[str, float] = {}
    for n in range(L):
        Tv = min(Te, Ta - n)
        if Tv <= 0:
            break
        logits_n = all_logits[n][:, :Tv]
        rewards_n = target_rewards[:, n:n + Tv]
        weight = target_mask[:, n:n + Tv].float()
        wsum = weight.sum()
        if float(wsum) <= 0:
            continue
        lp = reward_head.twohot.log_prob(logits_n, rewards_n)       # (B, Tv)
        total = total - (weight * lp).sum() / wsum
        heads_used += 1
        if n == 0:
            pred = reward_head.twohot.decode(logits_n.detach())
            info["reward_mae"] = float(
                ((pred - rewards_n).abs() * weight).sum() / wsum)
    return total / max(heads_used, 1), info


def continues_from_terminals(terminals: torch.Tensor, T: int) -> torch.Tensor:
    """
    (B, >=T-1) transition terminals -> (B, T) per-FRAME continue flags:
    frame t is terminal iff the t-1 -> t transition ended the episode.
    Frame 0 of a window is never terminal (windows lie within episodes and
    the terminal observation is always an episode's last frame).
    """
    B = terminals.shape[0]
    c = torch.ones(B, T, device=terminals.device, dtype=torch.float32)
    n = min(T - 1, terminals.shape[1])
    c[:, 1:1 + n] = 1.0 - terminals[:, :n].float()
    return c


# ---------------------------------------------------------------------------
# Phase 3: TD(lambda) returns + PMPO (paper Eq. 10-11)
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
    kl_to_prior: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    The paper's PMPO objective (Eq. 11), exactly:

        L = (1-alpha)/|D-| sum_{D-} ln pi(a|s)
          - alpha/|D+|     sum_{D+} ln pi(a|s)
          + beta * mean KL[pi_theta || pi_prior]

    where D+/D- pool ALL states across batch and time by the SIGN of the
    advantage (A >= 0 vs A < 0). Only the sign is used — no percentile
    thresholds, no advantage normalization, no entropy bonus (Appendix F:
    PMPO makes normalization unnecessary; the KL to the frozen behavioral
    prior replaces the entropy regularizer, and it is the REVERSE direction
    to constrain the policy to the space of reasonable behaviors).

    Empty-set guard: with sparse rewards and a fresh value head an early
    batch can land entirely in D+ or D-; the missing term is skipped (its
    mean over an empty set is undefined) and reported via ``pos_frac``.

    Args:
        log_probs:   (B, T) ln pi_theta(a_i | s_i) with grad.
        advantages:  (B, T) A_i = R_i^lambda - v_i (detached inside).
        kl_to_prior: (B, T) exact KL[pi_theta(.|s_i) || pi_prior(.|s_i)]
                     per state (computed by the caller — closed form for
                     categorical policies), or None to skip the prior term.
        alpha:       Positive-set weight (paper: 0.5).
        beta:        Prior-KL weight (paper: 0.3).

    Returns:
        (loss, info) — info: pos_frac, kl_to_prior, log_prob_(pos|neg).
    """
    adv = advantages.detach().reshape(-1)
    lp = log_probs.reshape(-1)
    pos = adv >= 0
    neg = ~pos

    loss = lp.new_zeros(())
    info: Dict[str, float] = {
        "pos_frac": float(pos.float().mean()),
        "kl_to_prior": 0.0,
    }
    if pos.any():
        lp_pos = lp[pos].mean()
        loss = loss - alpha * lp_pos
        info["log_prob_pos"] = float(lp_pos.detach())
    if neg.any():
        lp_neg = lp[neg].mean()
        loss = loss + (1.0 - alpha) * lp_neg
        info["log_prob_neg"] = float(lp_neg.detach())
    if kl_to_prior is not None and beta > 0:
        kl = kl_to_prior.mean()
        loss = loss + beta * kl
        info["kl_to_prior"] = float(kl.detach())
    return loss, info
