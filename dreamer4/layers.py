"""
Transformer layers for Dreamer 4: SpaceAttention, TimeAttention, BlockCausalLayer.

This module combines the building blocks (attention, MLP, norms, RoPE, modality
masks) into the two types of attention layers and the full transformer layer.

THE BLOCK-CAUSAL PATTERN:

  Given input (B, T, S, D):
    - B = batch, T = time steps, S = spatial tokens, D = model dim

  SpaceAttention: processes each time step independently.
    Reshape (B, T, S, D) -> (B*T, S, D), apply attention with modality mask,
    reshape back. This lets all tokens within a time step interact.

  TimeAttention: processes each spatial position independently across time.
    Reshape (B, T, S, D) -> (B*S, T, D), apply CAUSAL attention, reshape back.
    This lets each spatial position attend to its history (past time steps).

  The combination gives "block-causal": tokens can see all tokens at the
  same time step, and all tokens at past time steps, but NOT future time steps.

  TimeAttention is only applied every Nth layer (default: every 4th layer,
  as in the paper Section 3.4). This saves compute and provides an inductive
  bias that focuses most computation on spatial processing.

ONE TRANSFORMER LAYER:
  x = x + SpaceAttention(RMSNorm(x))      # always
  x = x + TimeAttention(RMSNorm(x))       # only every N layers
  x = x + MLP(RMSNorm(x))                 # always

This is the "pre-norm" residual pattern: normalize before each sublayer,
add the result to the residual. This is more stable than post-norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dreamer4.attention import MultiheadAttention
from dreamer4.modality import TokenLayout, build_space_attn_mask
from dreamer4.mlp import SwiGLU
from dreamer4.norms import RMSNorm
from dreamer4.rope import build_rope_cache


class SpaceAttention(nn.Module):
    """
    Attention within each time step (across spatial tokens).

    Applies modality-aware attention: which tokens can see which is controlled
    by the layout and mode (encoder/decoder/wm_agent_isolated/wm_agent).

    Applies 1D RoPE over spatial positions.

    Args:
        d_model:     Model dimension.
        n_heads:     Number of query heads.
        n_kv_heads:  Number of KV heads for GQA.
        layout:      TokenLayout describing the spatial dimension.
        mode:        Attention mode ("encoder", "decoder", etc.).
        dropout:     Attention dropout.
        use_qk_norm: Use QKNorm.
        logit_cap:   Logit soft capping value.

    Shape:
        Input:  (B, T, S, D)
        Output: (B, T, S, D)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None,
        layout: TokenLayout,
        mode: str,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
    ):
        super().__init__()
        self.S = layout.total_tokens()

        # Build and register the modality attention mask
        mask = build_space_attn_mask(layout, mode)  # (S, S)
        # Add head and batch dims for attention: (1, 1, S, S)
        self.register_buffer("attn_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        self.attn = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            logit_cap=logit_cap,
        )

        # Precompute RoPE cache for spatial positions
        head_dim = d_model // n_heads
        rope_cos, rope_sin = build_rope_cache(self.S, head_dim)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D)
        Returns:
            (B, T, S, D)
        """
        B, T, S, D = x.shape
        assert S == self.S, f"Expected S={self.S}, got {S}"

        # Flatten batch and time: (B*T, S, D)
        x_flat = x.reshape(B * T, S, D)

        # Apply attention with modality mask and spatial RoPE
        y_flat = self.attn(
            x_flat,
            attn_mask=self.attn_mask.expand(B * T, -1, -1, -1),
            rope_cos=self.rope_cos,
            rope_sin=self.rope_sin,
        )

        return y_flat.reshape(B, T, S, D)


class TimeAttention(nn.Module):
    """
    Causal attention across time steps (for each spatial position).

    Each spatial token at time t attends to the same spatial position at
    times 0, 1, ..., t (causal: cannot see the future).

    Applies 1D RoPE over temporal positions.

    Optionally, only the first `n_latents` spatial positions receive temporal
    attention (latents_only=True). This saves compute in the tokenizer where
    only the latent tokens need temporal context.

    Args:
        d_model:     Model dimension.
        n_heads:     Number of query heads.
        n_kv_heads:  Number of KV heads for GQA.
        dropout:     Attention dropout.
        use_qk_norm: Use QKNorm.
        logit_cap:   Logit soft capping value.
        latents_only: If True, only apply time attention to the first
                      n_latents positions (for tokenizer efficiency).
        n_latents:   Number of latent positions (used if latents_only=True).
        max_T:       Maximum number of time steps to precompute RoPE for.

    Shape:
        Input:  (B, T, S, D)
        Output: (B, T, S, D)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        latents_only: bool = False,
        n_latents: int = 0,
        max_T: int = 1024,
    ):
        super().__init__()
        self.latents_only = latents_only
        self.n_latents = n_latents

        self.attn = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            logit_cap=logit_cap,
        )

        # Precompute RoPE cache for temporal positions
        head_dim = d_model // n_heads
        rope_cos, rope_sin = build_rope_cache(max_T, head_dim)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D)
        Returns:
            (B, T, S, D)
        """
        B, T, S, D = x.shape

        if self.latents_only and self.n_latents > 0:
            # Only the first n_latents spatial positions get time attention
            L = self.n_latents
            lat = x[:, :, :L, :]  # (B, T, L, D)

            # Reshape: (B, T, L, D) -> (B, L, T, D) -> (B*L, T, D)
            lat = lat.permute(0, 2, 1, 3).contiguous().reshape(B * L, T, D)

            # Causal attention with temporal RoPE
            lat_out = self.attn(
                lat,
                is_causal=True,
                rope_cos=self.rope_cos[:T],
                rope_sin=self.rope_sin[:T],
            )

            # Reshape back and write into output
            lat_out = lat_out.reshape(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            out = x.clone()
            out[:, :, :L, :] = lat_out
            return out
        else:
            # All spatial positions get time attention
            # Reshape: (B, T, S, D) -> (B, S, T, D) -> (B*S, T, D)
            x_t = x.permute(0, 2, 1, 3).contiguous().reshape(B * S, T, D)

            y_t = self.attn(
                x_t,
                is_causal=True,
                rope_cos=self.rope_cos[:T],
                rope_sin=self.rope_sin[:T],
            )

            return y_t.reshape(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    """
    One transformer layer in the block-causal architecture.

    Structure (pre-norm residual):
        x = x + SpaceAttention(RMSNorm(x))
        if has_time_attention:
            x = x + TimeAttention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Time attention is included only if (layer_index + 1) % time_every == 0.
    For example, with time_every=4 and 0-indexed layers:
        Layer 0: space + mlp
        Layer 1: space + mlp
        Layer 2: space + mlp
        Layer 3: space + TIME + mlp   (layer 3+1=4, 4%4==0)
        Layer 4: space + mlp
        ...

    Args:
        d_model:          Model dimension.
        n_heads:          Query attention heads.
        n_kv_heads:       KV heads for GQA.
        layout:           TokenLayout for space attention.
        space_mode:       Modality mask mode ("encoder", "decoder", etc.).
        dropout:          Dropout rate.
        mlp_ratio:        MLP hidden size = d_model * mlp_ratio.
        layer_index:      0-based index of this layer in the stack.
        time_every:       Include time attention every N layers.
        use_qk_norm:      Use QKNorm in attention.
        logit_cap:        Logit soft capping value.
        latents_only_time: Only apply time attention to latent tokens.
        n_latents:        Number of latent tokens (for latents_only_time).
        max_T:            Max time steps for RoPE.

    Shape:
        Input:  (B, T, S, D)
        Output: (B, T, S, D)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None,
        layout: TokenLayout,
        space_mode: str,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        layer_index: int = 0,
        time_every: int = 4,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        latents_only_time: bool = False,
        n_latents: int = 0,
        max_T: int = 1024,
    ):
        super().__init__()
        self.has_time = (layer_index + 1) % time_every == 0

        # Space attention sublayer
        self.norm_space = RMSNorm(d_model)
        self.space_attn = SpaceAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            layout=layout,
            mode=space_mode,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            logit_cap=logit_cap,
        )

        # Time attention sublayer (optional)
        if self.has_time:
            self.norm_time = RMSNorm(d_model)
            self.time_attn = TimeAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                use_qk_norm=use_qk_norm,
                logit_cap=logit_cap,
                latents_only=latents_only_time,
                n_latents=n_latents,
                max_T=max_T,
            )

        # MLP sublayer
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D)
        Returns:
            (B, T, S, D)
        """
        # Space attention with residual
        x = x + self.space_attn(self.norm_space(x))

        # Time attention with residual (if this layer has it)
        if self.has_time:
            x = x + self.time_attn(self.norm_time(x))

        # MLP with residual
        x = x + self.mlp(self.norm_mlp(x))

        return x
