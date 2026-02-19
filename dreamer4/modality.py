"""
Token modality system and attention masks for Dreamer 4.

The Dreamer 4 transformer processes several types of tokens simultaneously
within each time step. This module defines:

1. **Modality enum**: Labels for each token type (IMAGE, LATENT, ACTION, etc.)
2. **TokenLayout**: A description of how the spatial dimension S is divided
   into contiguous segments of different modalities.
3. **build_space_attn_mask**: Builds the (S, S) boolean attention mask that
   controls which tokens can attend to which other tokens, based on the
   modality layout and the current mode.

MODES AND THEIR ATTENTION PATTERNS:

  "encoder" mode (used in the tokenizer encoder):
    - LATENT tokens can attend to ALL tokens (they aggregate information)
    - Non-LATENT tokens can only attend within their own modality
    Intuition: latents are the "summary" tokens that read from patches,
    while patches only see other patches (not latents).

  "decoder" mode (used in the tokenizer decoder):
    - LATENT tokens can only attend to other LATENT tokens
    - Non-LATENT tokens can attend to themselves AND to LATENT tokens
    Intuition: patches read from the latent summary to reconstruct the image.

  "wm_agent_isolated" mode (used during world model pretraining):
    - All non-AGENT tokens can attend to each other (full mixing)
    - AGENT tokens can only attend to other AGENT tokens
    - Non-AGENT tokens CANNOT see AGENT tokens
    Intuition: agent tokens are "invisible" during pretraining, so the
    world model doesn't learn to depend on them. This prevents
    "causal confusion" — the world model should predict the future based
    on actions, not based on what the agent intends to do.

  "wm_agent" mode (used during agent finetuning):
    - AGENT tokens can attend to ALL tokens (including actions, images, etc.)
    - Non-AGENT tokens still CANNOT see AGENT tokens
    Intuition: the agent reads the world state to predict actions and rewards,
    but the world model's predictions remain independent of agent tokens.
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
    REGISTER = 3          # Register tokens (learnable, for temporal consistency)
    SPATIAL = 4           # Packed spatial tokens (tokenizer output in dynamics)
    SHORTCUT_SIGNAL = 5   # Shortcut signal level token (tau)
    SHORTCUT_STEP = 6     # Shortcut step size token (d)
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
                (Modality.SHORTCUT_STEP, 1),
                (Modality.SPATIAL, 128),
                (Modality.REGISTER, 4),
                (Modality.AGENT, 1),
            )
        )
        Total S = 0 + 1 + 1 + 1 + 128 + 4 + 1 = 136
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
        mode: One of "encoder", "decoder", "wm_agent_isolated", "wm_agent".

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

    elif mode == "wm_agent_isolated":
        # Agent tokens only see agent tokens
        # Non-agent tokens see everything EXCEPT agent tokens
        is_q_agent = q_mod == int(Modality.AGENT)  # (S, 1)
        is_k_agent = k_mod == int(Modality.AGENT)  # (1, S)

        # Start with full connectivity
        mask = torch.ones(S, S, dtype=torch.bool)
        # Non-agent queries: block agent keys
        mask = torch.where(~is_q_agent & is_k_agent, False, mask)
        # Agent queries: only see agent keys
        mask = torch.where(is_q_agent & ~is_k_agent, False, mask)

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
            "'encoder', 'decoder', 'wm_agent_isolated', 'wm_agent'"
        )

    return mask
