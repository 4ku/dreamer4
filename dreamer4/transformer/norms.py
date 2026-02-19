"""
Normalization layers for Dreamer 4.

This module provides two normalization layers used in the transformer:

1. **RMSNorm** — Root Mean Square Normalization.
   A simpler alternative to LayerNorm that skips the mean-centering step.
   Used as "pre-layer normalization" before every attention and MLP sublayer.

   Why not LayerNorm?  RMSNorm is ~10-20% faster because it doesn't compute
   the mean, and empirically performs just as well for transformers.
   Reference: Zhang & Sennrich, 2019 — "Root Mean Square Layer Normalization"

2. **QKNorm** — Query-Key Normalization.
   Normalizes Q and K vectors in attention before the dot product, preventing
   the attention logits from growing unboundedly as the model gets deeper.
   This is crucial for training stability in deep (20+ layer) transformers.
   Reference: Dehghani et al., 2023 — "Scaling Vision Transformers to 22B"

   How it works: L2-normalize each head's Q/K vector, then multiply by a
   learnable per-head scale parameter. This keeps the dot products bounded
   while still allowing the model to learn the right scale.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Formula:
        RMSNorm(x) = x * scale / sqrt(mean(x^2) + eps)

    Args:
        dim: The last dimension of the input tensor to normalize over.
        eps: Small constant for numerical stability (default: 1e-6).

    Shape:
        Input:  (..., dim) — any number of leading dimensions.
        Output: (..., dim) — same shape as input.

    Example:
        >>> norm = RMSNorm(256)
        >>> x = torch.randn(2, 16, 256)
        >>> y = norm(x)           # shape: (2, 16, 256)
        >>> y.shape
        torch.Size([2, 16, 256])
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable per-feature scale parameter, initialized to 1
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square along last dimension
        # x.pow(2).mean(-1) computes mean(x_i^2) for each position
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and apply learnable scale
        return x * (self.scale / rms)


class QKNorm(nn.Module):
    """
    Query-Key Normalization for multi-head attention.

    Normalizes Q and K independently using L2 normalization along the
    head dimension, then multiplies by a learnable per-head scale.

    This bounds the maximum attention logit to `scale^2 * head_dim`,
    preventing instabilities in deep transformers.

    Args:
        head_dim: Dimension of each attention head.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads (for GQA). Defaults to n_heads.

    Shape:
        Input Q:  (batch, n_heads, seq_len, head_dim)
        Input K:  (batch, n_kv_heads, seq_len, head_dim)
        Output Q: (batch, n_heads, seq_len, head_dim)
        Output K: (batch, n_kv_heads, seq_len, head_dim)

    Example:
        >>> qknorm = QKNorm(head_dim=64, n_heads=8, n_kv_heads=2)
        >>> q = torch.randn(2, 8, 32, 64)
        >>> k = torch.randn(2, 2, 32, 64)
        >>> q_normed, k_normed = qknorm(q, k)
    """

    def __init__(self, head_dim: int, n_heads: int, n_kv_heads: int | None = None):
        super().__init__()
        self.head_dim = head_dim
        if n_kv_heads is None:
            n_kv_heads = n_heads
        # Learnable scale per head, initialized to sqrt(head_dim) so that
        # after normalization the initial dot products have similar magnitude
        # to standard scaled dot-product attention.
        # Q scale has shape (n_heads, 1, 1) and K scale has (n_kv_heads, 1, 1)
        # to support Grouped Query Attention where K has fewer heads.
        self.q_scale = nn.Parameter(torch.full((n_heads, 1, 1), head_dim ** 0.5))
        self.k_scale = nn.Parameter(torch.full((n_kv_heads, 1, 1), head_dim ** 0.5))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize Q and K independently.

        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_kv_heads, seq_len, head_dim)

        Returns:
            Tuple of (normalized_q, normalized_k), same shapes as inputs.
        """
        # L2-normalize along head dimension (last dim), eps for stability
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6) * self.q_scale
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6) * self.k_scale
        return q, k
