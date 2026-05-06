"""
Multi-Head Attention with GQA, logit capping, QKNorm, and RoPE for Dreamer 4.

This module implements the core attention mechanism with all enhancements
described in the Dreamer 4 paper (Section 3.4):

1. **Grouped Query Attention (GQA)**: Multiple query heads share the same
   key/value head, reducing KV cache size for faster inference on long video
   sequences. E.g., 8 query heads with 2 KV heads = 4x smaller KV cache.

2. **QKNorm**: Normalizes Q and K before the dot product to stabilize
   training of deep transformers.

3. **Attention logit soft capping**: Applies `cap * tanh(logits / cap)` to
   prevent any single attention score from dominating. Used in Gemma 2.

4. **RoPE integration**: Applies rotary position embeddings to Q and K
   before the dot product to encode relative positions.

The typical data flow through this module:

    Input (N, L, D)
    -> Linear projection to Q, K, V
    -> Reshape to heads: Q (N, n_heads, L, head_dim)
                          K (N, n_kv_heads, L, head_dim)
                          V (N, n_kv_heads, L, head_dim)
    -> QKNorm (optional)
    -> RoPE (optional)
    -> Expand K,V to match Q heads (GQA repeat)
    -> Scaled dot-product attention (with optional mask + logit capping)
    -> Concatenate heads and project out
    -> Output (N, L, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer4.transformer.norms import QKNorm
from dreamer4.transformer.rope import apply_rope


class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention with GQA, QKNorm, logit capping, and RoPE.

    Args:
        d_model:       Total model dimension (e.g., 256 or 512).
        n_heads:       Number of query attention heads.
        n_kv_heads:    Number of key/value heads. Must divide n_heads evenly.
                       Default = n_heads (standard multi-head attention).
                       Set lower for GQA (e.g., n_kv_heads=2 with n_heads=8).
        dropout:       Dropout on attention weights (default 0.0).
        use_qk_norm:   Whether to apply QKNorm (default True).
        logit_cap:     Soft capping value for attention logits. Set to 0 or
                       None to disable (default 50.0).

    Shape:
        Input:  (N, L, D) where N = batch, L = sequence length, D = d_model.
        Output: (N, L, D) same shape as input.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        assert n_heads % n_kv_heads == 0, (
            f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # how many Q heads share each KV head
        self.dropout_p = float(dropout)
        self.logit_cap = logit_cap if logit_cap and logit_cap > 0 else None

        # Q has n_heads * head_dim = d_model parameters
        # K, V each have n_kv_heads * head_dim parameters (less if GQA)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Optional QKNorm
        self.qk_norm: QKNorm | None = None
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (N, L, D) input tensor where
               N = batch size (number of sequences processed in parallel),
               L = sequence length (number of tokens),
               D = model dimension (d_model, total embedding size per token).
            attn_mask: (N, 1, L, L) or (1, 1, L, L) boolean mask where True
                       means "allowed to attend" (PyTorch SDPA convention).
                       Mutually exclusive with is_causal.
            is_causal: If True, apply causal (autoregressive) masking.
            rope_cos: (L, head_dim) cosine cache for RoPE. Optional.
            rope_sin: (L, head_dim) sine cache for RoPE. Optional.

        Returns:
            (N, L, D) output tensor, same shape as input.
        """
        N, L, D = x.shape

        # Project to Q, K, V and reshape to (N, n_heads/n_kv_heads, L, head_dim)
        q = self.q_proj(x).view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(N, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(N, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (N, n_heads, L, head_dim)
        # k: (N, n_kv_heads, L, head_dim)
        # v: (N, n_kv_heads, L, head_dim)

        # QKNorm: normalize Q, K before dot product
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # RoPE: rotate Q, K using position-dependent angles
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # GQA: repeat K, V to match the number of query heads
        # If n_kv_heads == n_heads, this is a no-op
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (N, n_heads, L, head_dim)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Compute attention
        if self.logit_cap is not None:
            # Manual attention with logit capping:
            # logits = (Q @ K^T) / sqrt(head_dim)
            # logits = cap * tanh(logits / cap)
            # weights = softmax(logits + mask)
            # output = weights @ V
            scale = self.head_dim ** -0.5
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (N, H, L, L)

            # Soft capping: squashes logits to [-cap, +cap]
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)

            # Apply mask
            if is_causal:
                causal = torch.tril(
                    torch.ones(L, L, device=x.device, dtype=torch.bool)
                )
                logits = logits.masked_fill(~causal, float("-inf"))
            elif attn_mask is not None:
                logits = logits.masked_fill(~attn_mask, float("-inf"))

            weights = F.softmax(logits, dim=-1) # (N, n_heads, L, L)
            drop = self.dropout_p if self.training else 0.0
            if drop > 0:
                weights = F.dropout(weights, p=drop)
            y = torch.matmul(weights, v) # (N, n_heads, L, head_dim)
        else:
            # Use PyTorch's fused SDPA (faster, no logit capping)
            drop = self.dropout_p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=drop,
                is_causal=is_causal,
            ) # (N, n_heads, L, head_dim)

        # Concat heads and project out: (N, n_heads, L, head_dim) -> (N, L, D)
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        return self.out_proj(y)

    def forward_kv_cached(
        self,
        x_new: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Incremental attention that projects Q, K, V only for new tokens and
        optionally concatenates K, V with a cache of past K, V.

        Past K, V are expected **post-QKNorm and post-RoPE** at positions
        ``[0, T_past)``. Newly projected K_new is QKNorm'd and RoPE'd at
        positions ``[T_past, T_total)`` before being concatenated. Q_new gets
        the same positional rotation as K_new.

        Args:
            x_new:   (N, T_new, D) — new tokens only.
            past_kv: Optional (K_past, V_past); each (N, n_kv_heads, T_past,
                     head_dim). Absent → fresh attention over ``x_new`` alone.
            rope_cos, rope_sin: RoPE caches of length ≥ ``T_past + T_new``.

        Returns:
            output:   (N, T_new, D) — attention output for new tokens only.
            k_new:    (N, n_kv_heads, T_new, head_dim) — post-QKNorm/post-RoPE
                     K for ``x_new`` at positions ``[T_past, T_total)``.
                     Ready to be ``append``ed to a ``KVCache``.
            v_new:    (N, n_kv_heads, T_new, head_dim).
        """
        N, T_new, D = x_new.shape
        T_past = 0 if past_kv is None else past_kv[0].shape[-2]
        T_total = T_past + T_new

        q = self.q_proj(x_new).view(N, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_new).view(N, T_new, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_new).view(N, T_new, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k_new = self.qk_norm(q, k_new)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos[T_past:T_total], rope_sin[T_past:T_total])
            k_new = apply_rope(k_new, rope_cos[T_past:T_total], rope_sin[T_past:T_total])

        if past_kv is not None:
            k_past, v_past = past_kv
            k_full = torch.cat([k_past, k_new], dim=-2)
            v_full = torch.cat([v_past, v_new], dim=-2)
        else:
            k_full, v_full = k_new, v_new

        if self.n_rep > 1:
            k_attn = k_full.repeat_interleave(self.n_rep, dim=1)
            v_attn = v_full.repeat_interleave(self.n_rep, dim=1)
        else:
            k_attn, v_attn = k_full, v_full

        # Causal mask: Q at offset i (absolute T_past + i) attends to K at j ≤ T_past + i.
        device = x_new.device
        i = torch.arange(T_new, device=device).unsqueeze(1) # (T_new, 1)
        j = torch.arange(T_total, device=device).unsqueeze(0) # (1, T_total)
        causal = j <= (i + T_past)
        mask = causal.view(1, 1, T_new, T_total) # (1, 1, T_new, T_total)

        if self.logit_cap is not None:
            scale = self.head_dim ** -0.5
            logits = torch.matmul(q, k_attn.transpose(-2, -1)) * scale
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)
            logits = logits.masked_fill(~mask, float("-inf"))
            weights = F.softmax(logits, dim=-1)
            drop = self.dropout_p if self.training else 0.0
            if drop > 0:
                weights = F.dropout(weights, p=drop)
            y = torch.matmul(weights, v_attn)
        else:
            drop = self.dropout_p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=mask, dropout_p=drop, is_causal=False,
            )

        y = y.transpose(1, 2).contiguous().view(N, T_new, D)
        return self.out_proj(y), k_new, v_new
