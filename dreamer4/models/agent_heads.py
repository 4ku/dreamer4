"""The paper's agent side: policy, reward and value read off AGENT tokens inside the world model.

Paper Section 3.3 -- "we insert agent tokens as an additional modality into the world model
transformer and interleave it with the image representations, actions, and register tokens. The
agent tokens receive task embeddings as input and we use them to predict the policy and reward
model using MLP heads. While the agent tokens attend to themselves and all other modalities, no
other modalities can attend back to the agent tokens."

So there is no second network and no readout over frozen latents: the policy IS the world model.
This module holds the two halves the transformer itself does not:

    AgentTokenEncoder   the INPUT side -- what an agent token carries into the block at time t
                        (a task embedding, plus a learned per-slot embedding);
    PolicyHead          ) the OUTPUT side -- small MLPs on the agent token's output embedding
    RewardHead          ) h_t, with ONE OUTPUT LAYER PER MTP DISTANCE (paper Eq. 9, L = 8);
    ValueHead           the phase-3 critic, a symexp two-hot distribution trained by
                        cross-entropy on the lambda-return (paper Eq. 10) -- not a scalar MSE.

The mask that makes this legal (``wm_agent`` in
:mod:`dreamer4.models.transformer.modality`) is already the world model's only space mode, and
``DynamicsModel`` already carries ``n_agent`` / ``agent_slice`` / ``agent_out``; phase-1
checkpoints simply have ``n_agent = 0``, which degenerates the mask to plain world-to-world
attention. Rebuilding the same checkpoint with ``n_agent > 0`` therefore leaves every world-token
output bit-identical -- that is the property :func:`agent_tokens_are_write_only` asserts.

Shapes. ``h_t`` is the transformer output at the agent tokens of time step ``t``,
``(B, T, n_agent, d_model)``; the heads consume it flattened to ``(B, T, n_agent * d_model)``
(:func:`flatten_agent_out`), so ``n_agent`` is a width knob and not a change of interface.

    heads = build_agent_heads(cfg)          # cfg is what the checkpoint carries
    h = flatten_agent_out(agent_out)        # (B, T, d_in)
    logits = heads.policy(h)                # (B, T, L+1, n_actions) -- n = 0..L
    a = heads.policy.sample(heads.policy.at(h, 0))
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from dreamer4.models.distributions import SymExpTwoHot


# ---------------------------------------------------------------------------
# Input side
# ---------------------------------------------------------------------------


class AgentTokenEncoder(nn.Module):
    """Builds the ``(B, T, n_agent, d_model)`` agent tokens the dynamics model takes as input.

    Paper Section 3.3: "The agent tokens receive task embeddings as input." ``n_tasks`` one-hot
    task indicators become an embedding lookup (the paper's own choice -- "We use one-hot task
    indicators but text embeddings could easily be used"); a learned per-slot embedding is added so
    the ``n_agent`` tokens of one block are distinguishable.
    """

    def __init__(self, d_model: int, *, n_agent: int = 1, n_tasks: int = 1):
        super().__init__()
        if n_agent < 1:
            raise ValueError(f"n_agent must be >= 1, got {n_agent}")
        self.d_model = d_model
        self.n_agent = n_agent
        self.n_tasks = n_tasks
        self.task = nn.Embedding(n_tasks, d_model)
        nn.init.normal_(self.task.weight, std=0.02)
        self.slot = nn.Parameter(torch.empty(n_agent, d_model))
        nn.init.normal_(self.slot, std=0.02)

    def forward(self, batch_time_shape, *, task_ids: Optional[torch.Tensor] = None,
                device=None, dtype=None) -> torch.Tensor:
        """``(B, T)`` -> ``(B, T, n_agent, d_model)``.

        Args:
            batch_time_shape: ``(B, T)``.
            task_ids: ``(B, T)`` or ``(B,)`` long task indicators; None = task 0 everywhere
                      (the single-task case).
        """
        B, T = batch_time_shape
        dev = device if device is not None else self.slot.device
        if task_ids is None:
            task_ids = torch.zeros(B, T, dtype=torch.long, device=dev)
        elif task_ids.dim() == 1:
            task_ids = task_ids.to(dev).long().reshape(B, 1).expand(B, T)
        tok = self.task(task_ids.to(dev)).unsqueeze(2)            # (B, T, 1, D)
        tok = tok + self.slot.view(1, 1, self.n_agent, self.d_model)
        if dtype is None and torch.is_autocast_enabled(dev.type):
            # An embedding lookup is not autocast-eligible, so this token would come out float32
            # while every other token in the block is bfloat16, and the concatenation inside the
            # dynamics model would raise on the dtype mismatch.
            dtype = torch.get_autocast_dtype(dev.type)
        return tok if dtype is None else tok.to(dtype)


# ---------------------------------------------------------------------------
# Output side
# ---------------------------------------------------------------------------


class MTPHead(nn.Module):
    """A small MLP with one output layer per multi-token-prediction distance.

    Paper Section 3.3: "We parameterize the policy and reward heads using small MLPs with one
    output layer per MTP distance", predicting ``a_{t+n}`` and ``r_{t+n}`` for ``n = 0..L`` from
    the single embedding ``h_t`` (Eq. 9, ``L = 8``). The body is shared across distances and only
    the last layer is per-distance, so the cost of MTP is ``L+1`` linear maps, not ``L+1`` MLPs.

    ``mtp = 0`` is the degenerate one-layer case, which is what the value head wants (Eq. 10 sums
    over time, not over distances).
    """

    def __init__(self, d_in: int, d_out: int, *, mtp: int = 8, d_hidden: int = 256):
        super().__init__()
        if mtp < 0:
            raise ValueError(f"mtp must be >= 0, got {mtp}")
        self.mtp = mtp
        self.d_out = d_out
        self.body = nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU())
        self.outs = nn.ModuleList([nn.Linear(d_hidden, d_out) for _ in range(mtp + 1)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``(..., d_in)`` -> ``(..., mtp+1, d_out)``."""
        x = self.body(h)
        return torch.stack([layer(x) for layer in self.outs], dim=-2)

    def at(self, h: torch.Tensor, n: int = 0) -> torch.Tensor:
        """``(..., d_in)`` -> ``(..., d_out)`` for ONE distance -- the acting path.

        Imagination and evaluation only ever ask for ``n = 0``; running the whole stack there
        would be ``L+1`` times the output-layer work for one column of it.
        """
        return self.outs[n](self.body(h))


class PolicyHead(nn.Module):
    """The policy: a categorical distribution over ``n_actions`` on the agent embedding.

    Paper Section 3.3 parameterizes it "as categorical or vectorized binary distribution, depending
    on the action space of the dataset" -- the binary form is what its 23 Minecraft keyboard keys
    need. Only the categorical form is built here: the dataset contract of
    :mod:`dreamer4.data.base` carries one-hot action VECTORS, so no adapter can describe a
    multi-key action space, and the imagination update's PMPO term is written against a single
    categorical. A binary head is a small addition once the data layer can say it needs one.
    """

    def __init__(self, d_in: int, *, n_actions: int, mtp: int = 8, d_hidden: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.mtp = mtp
        self.net = MTPHead(d_in, n_actions, mtp=mtp, d_hidden=d_hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``(..., d_in)`` -> ``(..., mtp+1, n_actions)`` logits."""
        return self.net(h)

    def at(self, h: torch.Tensor, n: int = 0) -> torch.Tensor:
        """``(..., d_in)`` -> ``(..., n_actions)`` logits at MTP distance ``n``."""
        return self.net.at(h, n)

    def dist(self, logits: torch.Tensor) -> torch.distributions.Categorical:
        """Logits ``(..., n_actions)`` -> the distribution of ONE action."""
        return torch.distributions.Categorical(logits=logits)

    def sample(self, logits: torch.Tensor, *, greedy: bool = False,
               temperature: float = 1.0) -> torch.Tensor:
        """Act. ``greedy`` takes the mode; ``temperature`` flattens the logits first."""
        if greedy:
            return logits.argmax(-1)
        return self.dist(logits / temperature).sample()

    def log_prob(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``ln pi(a | h)`` -- ``(..., n_actions)`` logits against ``(...)`` action ids."""
        return self.dist(logits).log_prob(actions.long())

    def clone_loss(self, logits: torch.Tensor, actions: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The policy half of Eq. 9: ``- ln p(a | h)``, averaged over the valid entries of ``mask``.

        ``logits`` ``(..., n_actions)``, ``actions`` ``(...)`` action ids, ``mask`` ``(...)``
        float/bool -- 0 where the target does not exist (past the end of the clip or of the
        episode), which is what keeps MTP from training on padding.
        """
        nll = -self.log_prob(logits, actions)
        return _masked_mean(nll, mask)


class TwoHotHead(nn.Module):
    """A scalar prediction as a symexp two-hot distribution, trained by cross-entropy.

    Paper Section 3.3 for the reward -- "Following Dreamer 3, the reward head is parameterized as a
    symexp twohot output to robustly learn stochastic rewards across varying orders of magnitude"
    -- and again for the value: "It uses a symexp twohot output to robustly learn across different
    scales of values", with Eq. 10 spelling out the loss as ``- ln p(R^lambda | s)``.

    The bin count and range are NOT in the paper (Dreamer 3's 255 bins over [-20, 20] are implied
    by the citation). They are a constructor argument here, and the trainers' defaults are narrower
    for the reason recorded in ``train_heads``: over wide symexp bins a thousandth of the
    probability mass on a far bin moves the decoded mean by hundreds.
    """

    def __init__(self, d_in: int, *, mtp: int = 0, d_hidden: int = 256,
                 num_bins: int = 255, bin_low: float = -1.0, bin_high: float = 1.0):
        super().__init__()
        self.net = MTPHead(d_in, num_bins, mtp=mtp, d_hidden=d_hidden)
        self.twohot = SymExpTwoHot(num_bins, bin_low, bin_high)
        self.mtp = mtp

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``(..., d_in)`` -> ``(..., mtp+1, num_bins)`` logits."""
        return self.net(h)

    def at(self, h: torch.Tensor, n: int = 0) -> torch.Tensor:
        """``(..., d_in)`` -> ``(..., num_bins)`` logits at MTP distance ``n``."""
        return self.net.at(h, n)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits to a scalar -- the exact inverse of the two-hot encoding."""
        return self.twohot.decode(logits)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``- ln p(target | h)``, averaged over the valid entries of ``mask``."""
        return _masked_mean(-self.twohot.log_prob(logits, targets), mask)


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of ``x`` over the entries ``mask`` keeps; a plain mean when ``mask`` is None.

    Written as sum/sum rather than ``x[mask].mean()`` so the shape is static -- the trainers run
    this under ``torch.compile`` and boolean indexing would recompile on every new count.
    """
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# The bundle a checkpoint carries
# ---------------------------------------------------------------------------


class AgentHeads(nn.Module):
    """Policy + reward + value on ``h_t``, i.e. everything phase 2 adds to the world model.

    Phase 2 trains the policy and reward heads (Eq. 9) together with the video-prediction loss on
    the whole transformer; phase 3 freezes the transformer and trains the policy (Eq. 11) and the
    value head (Eq. 10) alone. The value head is therefore dead weight until phase 3 -- it is built
    here anyway so one checkpoint format serves both phases and ``--init_from`` never has to guess.

    ``cfg`` is the single source of truth for the architecture: phase 2 writes it, phase 3 and the
    evaluator rebuild from it verbatim, and nothing downstream re-derives a shape.
    """

    def __init__(self, *, d_model: int, n_agent: int = 1, n_tasks: int = 1,
                 n_actions: int, mtp: int = 8, d_hidden: int = 256,
                 reward_bins: int = 255, reward_bin_low: float = -1.0,
                 reward_bin_high: float = 1.0, value_bins: int = 255,
                 value_bin_low: float = -1.0, value_bin_high: float = 1.0):
        super().__init__()
        d_in = n_agent * d_model
        self.d_model = d_model
        self.n_agent = n_agent
        self.d_in = d_in
        self.mtp = mtp
        self.tokens = AgentTokenEncoder(d_model, n_agent=n_agent, n_tasks=n_tasks)
        self.policy = PolicyHead(d_in, n_actions=n_actions, mtp=mtp, d_hidden=d_hidden)
        self.reward = TwoHotHead(d_in, mtp=mtp, d_hidden=d_hidden, num_bins=reward_bins,
                                 bin_low=reward_bin_low, bin_high=reward_bin_high)
        self.value = TwoHotHead(d_in, mtp=0, d_hidden=d_hidden, num_bins=value_bins,
                                bin_low=value_bin_low, bin_high=value_bin_high)


def build_agent_heads(cfg: Dict) -> AgentHeads:
    """Rebuild the heads from the ``agent_cfg`` dict a checkpoint carries."""
    return AgentHeads(**cfg)


def terminal_bins(reward_head: TwoHotHead, continues_from_reward, device) -> torch.Tensor:
    """The dataset's terminal rule evaluated on the reward head's own bin centres.

    The paper has no continue head, so an imagined episode ends when the reward head says it did.
    The rule is applied to the head's DISTRIBUTION rather than to its decoded mean: a frame either
    pays nothing or pays the terminal reward, so the distribution is bimodal and its mean can sit
    below the rule's threshold while much of the mass is on "terminal". Taking the expectation of
    the indicator avoids that:

        P(terminal) = sum_b softmax(logits)_b * (1 - continues_from_reward(bin_value_b))

    Args:
        reward_head:           the reward :class:`TwoHotHead`, for its bin centres.
        continues_from_reward: the dataset's rule, reward tensor -> continue flag in [0, 1].
        device:                device of the returned tensor.

    Returns:
        (num_bins,) float -- the terminal mass of each bin, to dot with the softmax.
    """
    if continues_from_reward is None:
        raise ValueError("this dataset has no continues_from_reward rule, so a predicted reward "
                         "cannot say when an episode ends (see "
                         "EpisodeVideoDataset.continues_from_reward)")
    bins = reward_head.twohot.bin_values.detach()
    cont = continues_from_reward(bins.reshape(1, -1).cpu()).reshape(-1)
    return (1.0 - cont).to(device)


def flatten_agent_out(agent_out: torch.Tensor) -> torch.Tensor:
    """``(B, T, n_agent, d_model)`` -> ``(B, T, n_agent * d_model)``: the heads' input ``h_t``.

    Concatenation rather than pooling, so a second agent token is extra width the heads can use
    and not an averaged-away duplicate. With the default ``n_agent = 1`` this is a reshape.
    """
    return agent_out.reshape(*agent_out.shape[:-2], -1)


# ---------------------------------------------------------------------------
# The attention property the whole design rests on
# ---------------------------------------------------------------------------


@torch.no_grad()
def agent_tokens_are_write_only(dynamics, *, atol: float = 0.0, device=None) -> Dict[str, float]:
    """Check the property Section 3.3 calls crucial: perturbing an agent token must not move ANY
    world output.

    "While the agent tokens attend to themselves and all other modalities, no other modalities can
    attend back to the agent tokens. This is crucial for avoiding causal confusion of the world
    model -- its future predictions can only be directly influenced by actions, not by the current
    task."

    Runs the model twice on identical inputs, the second time with the agent tokens replaced by a
    large random perturbation, and reports the largest absolute change in the flow-head output (and
    the proprio head, when the model has one) against the change in the agent output itself. A
    faithful mask gives ``world_delta = 0`` exactly -- the world rows never attend the agent
    columns, so the arithmetic is bit-identical -- and ``agent_delta > 0``, which is what proves
    the test would have noticed a leak.

    Returns ``{"world_delta", "proprio_delta", "agent_delta", "ok"}``.
    """
    if dynamics.n_agent == 0:
        raise ValueError("this model has no agent tokens (n_agent = 0): nothing to check")
    dev = device if device is not None else next(dynamics.parameters()).device
    B, T = 2, 3
    was_training = dynamics.training
    dynamics.eval()
    z = torch.randn(B, T, dynamics.n_spatial, dynamics.d_spatial, device=dev)
    act = torch.randn(B, T, dynamics.action_encoder.components[0].dim, device=dev)
    step = torch.zeros(B, T, dtype=torch.long, device=dev)
    sig = torch.zeros(B, T, dtype=torch.long, device=dev)
    prop = (torch.randn(B, T, dynamics.d_proprio, device=dev)
            if dynamics.d_proprio is not None else None)
    a1 = torch.randn(B, T, dynamics.n_agent, dynamics.d_model, device=dev)
    a2 = a1 + 10.0 * torch.randn_like(a1)
    out1 = dynamics(act, step, sig, z, agent_tokens=a1, proprio_noisy=prop)
    out2 = dynamics(act, step, sig, z, agent_tokens=a2, proprio_noisy=prop)
    world = float((out1[0] - out2[0]).abs().max())
    prop_delta = float((out1[1] - out2[1]).abs().max()) if prop is not None else 0.0
    agent = float((out1[-1] - out2[-1]).abs().max())
    dynamics.train(was_training)
    return {"world_delta": world, "proprio_delta": prop_delta, "agent_delta": agent,
            "ok": float(world <= atol and prop_delta <= atol and agent > 0.0)}


def mtp_targets(values: torch.Tensor, valid: torch.Tensor, mtp: int, *,
                pad: float = 0.0):
    """Gather the multi-token-prediction targets ``x_{t+n}`` for ``n = 0..mtp`` and their mask.

    Paper Eq. 9 sums ``ln p(a_{t+n} | h_t)`` over ``n = 0..L``, which needs, for every time step
    ``t`` of the clip, the next ``L+1`` targets and a mask saying which of them exist. Targets past
    the end of the clip -- or past the end of the episode, which ``valid`` marks -- are padded and
    masked out, so MTP never trains on invented labels.

    Args:
        values: ``(B, T, ...)`` per-step targets (action ids, action bit vectors, rewards).
        valid:  ``(B, T)`` bool/float -- 1 where ``values[:, t]`` is a real target.
        mtp:    ``L``.

    Returns:
        ``(targets (B, T, L+1, ...), mask (B, T, L+1) float)``.
    """
    B, T = values.shape[:2]
    tail = values.shape[2:]
    pad_v = values.new_full((B, mtp, *tail), pad)
    pad_m = valid.new_zeros((B, mtp))
    ext_v = torch.cat([values, pad_v], dim=1)                       # (B, T+L, ...)
    ext_m = torch.cat([valid, pad_m], dim=1).to(torch.float32)
    tgt = torch.stack([ext_v[:, n:n + T] for n in range(mtp + 1)], dim=2)
    msk = torch.stack([ext_m[:, n:n + T] for n in range(mtp + 1)], dim=2)
    return tgt, msk


def pretty_agent_summary(heads: AgentHeads) -> str:
    """One line for the trainers' log: what was built, and how big it is."""
    n = sum(p.numel() for p in heads.parameters())
    return (f"agent heads {n / 1e6:.2f}M | h_t {heads.d_in} "
            f"({heads.n_agent} x {heads.d_model}) | MTP L={heads.mtp} | "
            f"policy categorical({heads.policy.n_actions}) | "
            f"reward {heads.reward.twohot.num_bins} bins | "
            f"value {heads.value.twohot.num_bins} bins")
