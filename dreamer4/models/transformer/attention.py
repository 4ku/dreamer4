"""
Multi-head attention with GQA, QKNorm, RoPE and attention-logit soft capping.

Single attention primitive shared by both attention types in ``layers.py``: space attention
(over the S tokens of one timestep) and time attention (over T timesteps at one spatial
position). Input and output are ``(N, L, D)``; what ``N`` and ``L`` mean is decided by the
caller, which flattens ``(B, T, S, D)`` onto those two axes before calling.

Features (Dreamer 4 paper, Section 3.4):
  - GQA: several query heads share one key/value head, shrinking the KV cache for long
    rollouts (8 query heads over 2 KV heads = 4x smaller cache).
  - QKNorm: Q and K are normalized before the dot product, which keeps the logits of deep
    stacks bounded.
  - Logit soft capping: ``cap * tanh(logits / cap)`` keeps a single score from dominating
    (as in Gemma 2).
  - RoPE: relative positions are encoded by rotating Q and K.

Per call: project x to Q, K, V -> split into heads -> QKNorm -> RoPE -> repeat K/V up to the
query head count -> scaled dot-product attention (masked, optionally capped) -> merge heads ->
output projection.

``forward`` attends over a whole sequence. ``forward_kv_cached`` attends incrementally: it
projects only the new tokens and reads the past from cached K/V, for autoregressive rollout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer4.models.transformer.norms import QKNorm
from dreamer4.models.transformer.rope import apply_rope


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
        Attend over a full sequence.

        Args:
            x: (N, L, D) — N sequences of L tokens, model dimension D.
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
            # Fused SDPA cannot cap logits, so the capped path attends by hand and
            # materializes the full (N, n_heads, L, L) logit matrix:
            #   logits = (Q @ K^T) / sqrt(head_dim) -> cap -> mask -> softmax -> @ V
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
        Attend incrementally: project Q, K, V for new tokens only and prepend cached past K, V.

        Past K, V must already be **post-QKNorm and post-RoPE** at positions ``[0, T_past)``;
        the cache stores them in that form so this path only projects and rotates the new
        tokens, at positions ``[T_past, T_total)``. Q_new gets the same rotation as K_new.
        Attention is causal by construction, so no mask argument is accepted.

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
