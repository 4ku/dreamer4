"""
The two normalization layers of the transformer: RMSNorm and QKNorm. Both preserve shape.

RMSNorm — ``(..., dim) -> (..., dim)``. Rescales by the root mean square without subtracting
the mean, which is cheaper than LayerNorm and performs comparably in transformers. Used as
the pre-norm in front of every attention and MLP sublayer.
Reference: Zhang & Sennrich, 2019 — "Root Mean Square Layer Normalization"

QKNorm — ``(N, H, L, head_dim) -> (N, H, L, head_dim)`` for Q and K. L2-normalizes each
head's vector and rescales it by a learnable per-head scale, so attention logits stay bounded
instead of growing with depth. This matters for training stability in deep stacks.
Reference: Dehghani et al., 2023 — "Scaling Vision Transformers to 22B"
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
        # Root mean square over the last dimension, per position
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and apply learnable scale
        return x * (self.scale / rms)


class QKNorm(nn.Module):
    """
    Query-Key Normalization for multi-head attention.

    Normalizes Q and K independently using L2 normalization along the
    head dimension, then multiplies by a learnable per-head scale.

    Because the rescaling fixes the length of every Q and K vector, the raw dot product
    cannot exceed ``q_scale * k_scale`` (= head_dim at initialization) however deep the
    stack is, which is what prevents logit blow-up.

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
        # Learnable per-head scale, initialized to sqrt(head_dim): with unit-norm Q and K
        # that puts the initial logits at the same magnitude as standard scaled
        # dot-product attention. Shapes (n_heads, 1, 1) / (n_kv_heads, 1, 1) broadcast over
        # (seq_len, head_dim) and let K keep fewer heads under GQA.
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
