"""
Token modalities and the space-attention masks they induce.

Every timestep carries S tokens of several kinds (image patches, latents, actions, proprio,
registers, agent tokens). This module defines:

1. ``Modality`` — the token-type labels.
2. ``TokenLayout`` — how the S axis is split into contiguous per-modality segments.
3. ``build_space_attn_mask(layout, mode)`` — the (S, S) boolean mask saying which token may
   attend to which *within one timestep*. Time attention is not masked here; it is plain
   causal masking over T.

MODES AND THEIR ATTENTION PATTERNS:

  "encoder" (tokenizer encoder):
    LATENT tokens attend to everything; every other token attends only within its own
    modality. Latents are the summary that patches write into, and patches never read that
    summary back.

  "decoder" (tokenizer decoder):
    LATENT tokens attend only to latents; other tokens attend to their own modality plus the
    latents, so patches reconstruct from the latent summary.

  "decoder_cross" (tokenizer decoder, Perceiver-IO style):
    Every token attends to latents only — no patch-to-patch mixing.

  "wm_agent" (dynamics model):
    AGENT tokens attend to everything; no other token can see an AGENT token. The agent reads
    world state to predict actions, rewards and values, while the world model's predictions
    stay independent of agent tokens — otherwise it learns to predict the future from what the
    agent intends rather than from the action actually taken ("causal confusion").
    Pretraining uses this same mode with ``n_agent=0``: with no AGENT segment the mask
    degenerates to "every world token sees every world token", which is the pretraining
    pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple

import torch


class Modality(IntEnum):
    """
    Token type identifiers.

    Each token in the spatial dimension of the transformer belongs to exactly
    one modality. The modality determines the attention pattern.
    """
    LATENT = -1           # Bottleneck tokens in the tokenizer
    IMAGE = 0             # Image patch tokens
    ACTION = 1            # Action embedding tokens
    PROPRIO = 2           # Proprioceptive state tokens (for robotics)
    REGISTER = 3          # Learnable per-frame scratch tokens; no head reads their outputs
    SPATIAL = 4           # Packed spatial tokens (tokenizer output in dynamics)
    SHORTCUT_SIGNAL = 5   # Shortcut signal token carrying both tau and d (concat along channels)
    AGENT = 7             # Agent tokens (for policy/reward/value heads)


@dataclass(frozen=True)
class TokenLayout:
    """
    Describes the composition of the spatial dimension S.

    The spatial dimension is composed of an optional leading block of
    LATENT tokens, followed by contiguous segments of other modalities.

    Args:
        n_latents: Number of LATENT tokens at the start (can be 0).
        segments: Tuple of (Modality, count) pairs describing the rest.

    Example — tokenizer encoder layout:
        TokenLayout(
            n_latents=16,
            segments=((Modality.IMAGE, 64),)
        )
        Total S = 16 + 64 = 80

    Example — world model layout:
        TokenLayout(
            n_latents=0,
            segments=(
                (Modality.ACTION, 1),
                (Modality.SHORTCUT_SIGNAL, 1),
                (Modality.SPATIAL, 128),
                (Modality.REGISTER, 4),
                (Modality.AGENT, 1),
            )
        )
        Total S = 0 + 1 + 1 + 128 + 4 + 1 = 135
    """
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def total_tokens(self) -> int:
        """Total number of spatial tokens S."""
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        """
        Returns a 1D tensor of length S where each entry is the Modality
        integer for that token position.

        Example:
            >>> layout = TokenLayout(n_latents=2, segments=((Modality.IMAGE, 3),))
            >>> layout.modality_ids()
            tensor([-1, -1,  0,  0,  0], dtype=torch.int32)
        """
        parts: list[torch.Tensor] = []
        if self.n_latents > 0:
            parts.append(torch.full(
                (self.n_latents,), int(Modality.LATENT), dtype=torch.int32
            ))
        for modality, count in self.segments:
            if count > 0:
                parts.append(torch.full(
                    (count,), int(modality), dtype=torch.int32
                ))
        if parts:
            return torch.cat(parts, dim=0)
        return torch.zeros(0, dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        """
        Returns a dict mapping each Modality to its slice within the S dim.

        A modality that appears in several segments is reported once, at its first
        segment; empty segments (count 0) are omitted.

        Example:
            >>> layout = TokenLayout(n_latents=2, segments=((Modality.IMAGE, 3),))
            >>> layout.slices()
            {<Modality.LATENT: -1>: slice(0, 2), <Modality.IMAGE: 0>: slice(2, 5)}
        """
        result: Dict[Modality, slice] = {}
        idx = 0
        if self.n_latents > 0:
            result[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for modality, count in self.segments:
            if count > 0 and modality not in result:
                result[modality] = slice(idx, idx + count)
            idx += count
        return result


def build_space_attn_mask(layout: TokenLayout, mode: str) -> torch.Tensor:
    """
    Build the (S, S) boolean attention mask for space (within-timestep) attention.

    True means "query at row i is allowed to attend to key at column j".

    Args:
        layout: The TokenLayout describing the spatial dimension.
        mode: One of "encoder", "decoder", "decoder_cross", "wm_agent".

    Returns:
        (S, S) boolean tensor.
    """
    S = layout.total_tokens()
    mod_ids = layout.modality_ids()  # (S,)
    n_lat = layout.n_latents

    # Indices for building the mask
    q_idx = torch.arange(S).unsqueeze(1)  # (S, 1)
    k_idx = torch.arange(S).unsqueeze(0)  # (1, S)

    # Boolean masks for latent positions
    is_q_lat = q_idx < n_lat  # (S, 1) broadcast
    is_k_lat = k_idx < n_lat  # (1, S) broadcast

    # Modality of each query/key position
    q_mod = mod_ids[q_idx]  # (S, 1)
    k_mod = mod_ids[k_idx]  # (1, S)
    same_mod = q_mod == k_mod  # (S, S)

    if mode == "encoder":
        # Latent queries attend to everything
        # Non-latent queries attend only within their modality
        mask = torch.where(is_q_lat, torch.ones(S, S, dtype=torch.bool), same_mod)

    elif mode == "decoder":
        # Latent queries attend only to other latents
        # Non-latent queries attend to same modality AND latents
        lat_to_lat = is_k_lat
        nonlat_row = same_mod | is_k_lat
        mask = torch.where(is_q_lat, lat_to_lat, nonlat_row)

    elif mode == "decoder_cross":
        # Perceiver-IO style decode: every query attends to latents only, patches and
        # latents alike. With no patch<->patch mixing the decoder loses the constant-output
        # escape hatch that collapses the plain "decoder" mask: every spatial output is
        # forced to be a function of the latents.
        mask = is_k_lat.expand(S, S).clone()

    elif mode == "wm_agent":
        # Agent tokens can see everything
        # Non-agent tokens cannot see agent tokens
        is_q_agent = q_mod == int(Modality.AGENT)
        is_k_agent = k_mod == int(Modality.AGENT)

        mask = torch.ones(S, S, dtype=torch.bool)
        # Non-agent queries: block agent keys
        mask = torch.where(~is_q_agent & is_k_agent, False, mask)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: "
            "'encoder', 'decoder', 'decoder_cross', 'wm_agent'"
        )

    return mask
