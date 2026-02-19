"""
Rotary Position Embeddings (RoPE) for Dreamer 4.

RoPE encodes token positions by *rotating* the query and key vectors in
the complex plane. Two tokens with the same relative distance will always
produce the same dot-product contribution, regardless of their absolute
positions — this is the key advantage over additive sinusoidal embeddings.

HOW IT WORKS (intuition):
  Think of each pair of consecutive dimensions (d0, d1) as a 2D point.
  RoPE rotates this point by an angle proportional to the position:
    angle = position * frequency

  Low-frequency dimensions capture coarse position (far vs close),
  high-frequency dimensions capture fine position (exact offset).

  When computing Q · K, the rotation angles subtract, so the attention
  score only depends on the *relative* position (pos_q - pos_k).

FOR THE 2D CASE (space + time):
  Dreamer 4 has tokens arranged on a 2D grid: (time_step, spatial_token).
  We use "axial RoPE": split the head dimension in half, apply 1D RoPE
  with spatial positions on the first half and temporal positions on the
  second half. This is the standard approach for video transformers.

Reference: Su et al., 2021 — "RoFormer: Enhanced Transformer with Rotary
Position Embedding"
"""

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for 1D Rotary Position Embedding.

    The frequencies are computed as:
        freq_i = 1 / (base ^ (2i / head_dim))   for i = 0, 1, ..., head_dim/2 - 1

    Then for each position p:
        angle_i = p * freq_i

    Args:
        seq_len: Maximum sequence length to precompute for.
        head_dim: Dimension of each attention head (must be even).
        base: Base for the geometric frequency progression (default 10000).
        device: Device for the output tensors.

    Returns:
        cos_cache: (seq_len, head_dim) — cosines, repeated for each dim pair.
        sin_cache: (seq_len, head_dim) — sines, repeated for each dim pair.

    Example:
        >>> cos, sin = build_rope_cache(128, head_dim=64)
        >>> cos.shape, sin.shape
        (torch.Size([128, 64]), torch.Size([128, 64]))
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    half = head_dim // 2

    # Frequencies: freq_i = 1 / base^(2i/d) for i in [0, half)
    # We compute in log-space for numerical stability
    i = torch.arange(half, device=device, dtype=torch.float32)
    freq = torch.exp(-i * (2.0 / head_dim) * torch.log(torch.tensor(base)))
    # freq shape: (half,)

    # Positions: 0, 1, 2, ..., seq_len - 1
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Angles: outer product of positions and frequencies
    # angles[p, i] = pos[p] * freq[i]
    angles = torch.outer(pos, freq)  # (seq_len, half)

    # Duplicate each angle for the pair of dimensions it applies to:
    # [angle_0, angle_0, angle_1, angle_1, ...]
    # This makes the shape (seq_len, head_dim), matching the head dimension
    cos_cache = torch.cos(angles).repeat(1, 2)  # (seq_len, head_dim)
    sin_cache = torch.sin(angles).repeat(1, 2)  # (seq_len, head_dim)

    return cos_cache, sin_cache


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to a tensor.

    For each consecutive pair of dimensions (x0, x1), applies:
        x0' = x0 * cos - x1 * sin
        x1' = x0 * sin + x1 * cos

    This is equivalent to rotating the 2D vector (x0, x1) by the angle.

    Args:
        x:   (..., seq_len, head_dim) — typically Q or K.
        cos: (seq_len, head_dim) — cosine cache from build_rope_cache.
        sin: (seq_len, head_dim) — sine cache from build_rope_cache.

    Returns:
        Tensor of same shape as x, with rotary embedding applied.
    """
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]

    # Trim cache to actual sequence length
    cos = cos[:seq_len, :head_dim]
    sin = sin[:seq_len, :head_dim]

    # Broadcast cos/sin to match x's leading dimensions
    # cos, sin: (seq_len, head_dim) -> (1, ..., 1, seq_len, head_dim)
    n_leading = x.dim() - 2
    for _ in range(n_leading):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # Build the "rotated" version: swap pairs and negate first of each pair
    # For dimensions [d0, d1, d2, d3, ...] -> [-d1, d0, -d3, d2, ...]
    # This is the standard RoPE rotation formula
    half = head_dim // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    return x * cos + x_rot * sin


def build_rope_2d(
    T: int,
    S: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for 2D axial RoPE (space + time).

    Splits head_dim in half:
      - First half:  spatial position encoding (0..S-1)
      - Second half: temporal position encoding (0..T-1)

    The result is a (T, S, head_dim) tensor that can be indexed for any
    (time_step, spatial_position) pair.

    Args:
        T: Number of time steps.
        S: Number of spatial tokens per time step.
        head_dim: Dimension of each attention head (must be divisible by 4,
                  since we split in half for space/time, and each half must
                  be even for the rotation pairs).
        base: Frequency base (default 10000).
        device: Device for output tensors.

    Returns:
        cos_cache: (T, S, head_dim)
        sin_cache: (T, S, head_dim)

    Example:
        >>> cos, sin = build_rope_2d(T=16, S=64, head_dim=64)
        >>> cos.shape
        torch.Size([16, 64, 64])
    """
    assert head_dim % 4 == 0, (
        f"head_dim must be divisible by 4 for 2D axial RoPE, got {head_dim}"
    )
    half_dim = head_dim // 2  # dim for each axis

    # Build 1D caches for each axis
    cos_s, sin_s = build_rope_cache(S, half_dim, base=base, device=device)  # (S, half_dim)
    cos_t, sin_t = build_rope_cache(T, half_dim, base=base, device=device)  # (T, half_dim)

    # Expand to 2D grid:
    # Spatial: (S, half_dim) -> (1, S, half_dim) -> (T, S, half_dim)
    cos_s = cos_s.unsqueeze(0).expand(T, -1, -1)
    sin_s = sin_s.unsqueeze(0).expand(T, -1, -1)

    # Temporal: (T, half_dim) -> (T, 1, half_dim) -> (T, S, half_dim)
    cos_t = cos_t.unsqueeze(1).expand(-1, S, -1)
    sin_t = sin_t.unsqueeze(1).expand(-1, S, -1)

    # Concatenate along the head dimension: [spatial_part | temporal_part]
    cos_2d = torch.cat([cos_s, cos_t], dim=-1)  # (T, S, head_dim)
    sin_2d = torch.cat([sin_s, sin_t], dim=-1)  # (T, S, head_dim)

    return cos_2d, sin_2d
