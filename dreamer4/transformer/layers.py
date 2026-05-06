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

from dreamer4.transformer.attention import MultiheadAttention
from dreamer4.transformer.modality import TokenLayout, build_space_attn_mask
from dreamer4.transformer.mlp import SwiGLU
from dreamer4.transformer.norms import RMSNorm
from dreamer4.transformer.rope import build_rope_cache


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
            attn_mask=self.attn_mask.expand(B * T, -1, -1, -1), # (B*T, 1, S, S)
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

    Args:
        d_model:     Model dimension.
        n_heads:     Number of query heads.
        n_kv_heads:  Number of KV heads for GQA.
        dropout:     Attention dropout.
        use_qk_norm: Use QKNorm.
        logit_cap:   Logit soft capping value.
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
        max_T: int = 256,
    ):
        super().__init__()

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

    def forward_incremental(
        self,
        new_x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Incremental time attention that projects Q, K, V only for new tokens
        and pulls past K, V directly from the cache.

        Args:
            new_x:   (B, T_new, S, D) — already ``norm_time``-normalised.
            past_kv: Optional (K_past, V_past), each
                     (B*S, n_kv_heads, T_past, head_dim), post-QKNorm and
                     post-RoPE at positions ``[0, T_past)``.

        Returns:
            output: (B, T_new, S, D) time-attention output for new timesteps.
            k_new:  (B*S, n_kv_heads, T_new, head_dim) — K for the new tokens,
                    post-QKNorm/post-RoPE at positions ``[T_past, T_total)``.
                    Ready to be appended to a ``KVCache``.
            v_new:  (B*S, n_kv_heads, T_new, head_dim).
        """
        B, T_new, S, D = new_x.shape
        x_t = new_x.permute(0, 2, 1, 3).contiguous().reshape(B * S, T_new, D)

        y_t, k_new, v_new = self.attn.forward_kv_cached(
            x_t,
            past_kv=past_kv,
            rope_cos=self.rope_cos,
            rope_sin=self.rope_sin,
        )  # y_t: (B*S, T_new, D)

        out = y_t.reshape(B, S, T_new, D).permute(0, 2, 1, 3).contiguous()
        return out, k_new, v_new


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
        mlp_ratio: float = 8/3,
        layer_index: int = 0,
        time_every: int = 4,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        max_T: int = 256,
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
                max_T=max_T,
            )

        # MLP sublayer
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard (training / non-cached inference) forward.

        Args:
            x: (B, T, S, D).

        Returns:
            (B, T, S, D).
        """
        x = x + self.space_attn(self.norm_space(x))
        if self.has_time:
            x = x + self.time_attn(self.norm_time(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x

    def forward_incremental(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Cache-aware forward used during autoregressive rollout.

        Space attention and MLP operate only on ``x`` (the new tokens). If
        this layer has time attention, past K/V are pulled from ``past_kv``
        (or freshly projected when ``past_kv is None``) and the newly
        projected K/V for ``x`` are returned for caching.

        Args:
            x: ``(B, T_new, S, D)`` — new tokens only.
            past_kv: Optional ``(K, V)`` from the cache, each
                     ``(B*S, n_kv_heads, T_past, head_dim)``.

        Returns:
            output: ``(B, T_new, S, D)`` — layer output for new tokens.
            kv_new: ``(K_new, V_new)`` at positions ``[T_past, T_total)`` —
                    ready to be appended to a ``KVCache`` — or ``None`` when
                    this layer has no time attention.
        """
        post_space = x + self.space_attn(self.norm_space(x))

        if self.has_time:
            normed = self.norm_time(post_space)
            attn_out, k_new, v_new = self.time_attn.forward_incremental(
                normed, past_kv=past_kv,
            )
            x_out = post_space + attn_out
            kv_new: tuple[torch.Tensor, torch.Tensor] | None = (k_new, v_new)
        else:
            x_out = post_space
            kv_new = None

        x_out = x_out + self.mlp(self.norm_mlp(x_out))
        return x_out, kv_new
