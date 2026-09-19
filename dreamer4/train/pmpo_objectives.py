"""
Phase-3 objectives: TD(lambda) returns and the PMPO policy term (paper
Eq. 10-11).

:func:`compute_lambda_returns` builds the critic's target;
:func:`pmpo_coeffs` turns advantages into per-transition coefficients on
``ln pi(a | s)``. Both are pure tensor functions over one imagined rollout,
collected and fed to them by :mod:`dreamer4.train.train_pmpo`, which adds
the reverse KL to the frozen behavioral prior.

PMPO is a pure sign-of-advantage split pooled across batch x time
(``alpha = 0.5``): every transition whose advantage is positive is pushed up
with weight ``alpha / |D+|``, every negative one down with
``(1 - alpha) / |D-|``, with no advantage normalization -- the paper is
explicit that PMPO makes it unnecessary (Appendix F).

Alignment convention (matches the data contract in ``dreamer4.data.base``):
``r[t]`` and ``continues[t]`` describe the ``t -> t+1`` transition.
In imagination there are no recorded terminals, so ``continues`` must be
PREDICTED -- how, is a property of the environment, so it lives on the
dataset (``EpisodeVideoDataset.continues_from_reward``), not here.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    *,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    TD(lambda) returns -- the critic's regression target (paper Eq. 10).

        R_t^lambda = r_t + gamma * c_t * ((1 - lam) * v_{t+1} + lam * R_{t+1}^lambda)
        R_T^lambda = v_T

    The recursion runs backwards from the last step, which bootstraps from
    ``values[:, -1]``. Nothing is detached here: the caller decides what the
    gradient may flow through, and the trainer passes detached values so that
    the target is a constant the critic regresses onto.

    Args:
        rewards:   (B, T) scalar rewards.
        values:    (B, T) predicted values (from the value head).
        continues: (B, T) float -- 1.0 for non-terminal, 0.0 for terminal.
        gamma:     Discount factor.
        lam:       TD(lambda) mixing; 0 = one-step TD, 1 = Monte-Carlo return.

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


def pmpo_coeffs(
    advantages: torch.Tensor,
    *,
    alpha: float = 0.5,
    drop_frac: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Per-transition coefficients ``c_i`` such that

        sum_i c_i * ln pi(a_i | s_i)

    is exactly the PMPO policy term of paper Eq. 11 (the reverse KL to the
    behavioral prior is added by the trainer). Only the SIGN of the advantage
    is used: every transition with ``A >= 0`` is pushed up with weight
    ``alpha / |D+|``, every transition with ``A < 0`` is pushed down with
    ``(1 - alpha) / |D-|``. Advantages enter detached; the only gradient path
    is ``ln pi``.

    ``drop_frac > 0`` is NOT in the paper: it removes, from each pool
    separately, that fraction of transitions with the smallest ``|A|`` before
    the weights are set. A sign split votes a coin-flip advantage exactly as
    hard as a decisive one, and once the policy is mostly right nearly all of
    D- is optimal actions whose advantage is zero plus critic noise (measured
    on the empty maze: 86 % of D-). Dropping the least decisive ones removes
    that vote. A quantile per pool, not an absolute cut: ``|A|`` shrinks as
    the policy converges, and the two pools have different scales.

    Args:
        advantages: advantages of the transitions to score, any shape;
                    flattened and detached. Typically
                    ``lambda_return - value`` on the states acted in.
        alpha:      share of the total weight given to D+; D- gets
                    ``1 - alpha``.
        drop_frac:  fraction of each pool dropped, smallest ``|A|`` first;
                    0 = the paper's objective.

    Returns:
        (coeffs (N,) float32, info) -- ``info["pos_frac"]`` is the share of
        transitions with a non-negative advantage, ``info["kept_frac"]`` the
        share that kept a non-zero coefficient.
    """
    adv = advantages.detach().reshape(-1)
    pos, neg = adv >= 0, adv < 0
    pos_frac = float(pos.float().mean()) if adv.numel() else 0.0
    if drop_frac > 0 and adv.numel():
        mag = adv.abs().float()
        keep = torch.ones_like(pos)
        for pool in (pos, neg):
            if pool.any():
                keep &= ~pool | (mag >= torch.quantile(mag[pool], drop_frac))
        pos, neg = pos & keep, neg & keep
    coeff = torch.zeros(adv.shape, dtype=torch.float32, device=adv.device)
    if pos.any():
        coeff[pos] = -alpha / float(pos.sum())
    if neg.any():
        coeff[neg] = (1.0 - alpha) / float(neg.sum())
    info: Dict[str, float] = {
        "pos_frac": pos_frac,
        "kept_frac": float((pos | neg).float().mean()) if adv.numel() else 0.0,
    }
    return coeff, info
