"""
Rotary position embeddings (RoPE): cache construction, application, and re-rotation.

RoPE encodes a position by rotating the Q and K vectors by an angle proportional to it. The
angles cancel in the dot product, so an attention score depends only on the distance between
two tokens and not on where the pair sits in the sequence. Low-frequency dimensions carry
coarse position, high-frequency ones fine position.

Everything here is 1D. Tokens live on a 2D grid (timestep, spatial index), and the two axes
are handled by rotating over the *full* head dimension separately in each attention type:
space attention rotates by spatial index inside a frame, time attention by timestep. The head
dimension is split in half and ``(x_i, x_i+half)`` is the pair rotated together, so the
angle table is tiled twice rather than interleaved.

    build_rope_cache(seq_len, head_dim) -> (cos, sin), each (seq_len, head_dim)
    apply_rope(x, cos, sin)             -> x rotated to positions [0, seq_len)
    shift_rope(x, cos, sin, shift)      -> x re-rotated backward by a constant offset, used
                                           when the KV cache evicts its oldest frames

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
        cos_cache: (seq_len, head_dim) — cosines; the half-length angle table is tiled
                   twice, so dimensions i and i + head_dim/2 share an angle.
        sin_cache: (seq_len, head_dim) — sines, same layout.

    Example:
        >>> cos, sin = build_rope_cache(128, head_dim=64)
        >>> cos.shape, sin.shape
        (torch.Size([128, 64]), torch.Size([128, 64]))
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    half = head_dim // 2

    # Frequencies: freq_i = 1 / base^(2i/d) for i in [0, half),
    # evaluated as exp(-2i/d * log(base)) for numerical stability
    i = torch.arange(half, device=device, dtype=torch.float32)
    freq = torch.exp(-i * (2.0 / head_dim) * torch.log(torch.tensor(base)))
    # freq shape: (half,)

    # Positions: 0, 1, 2, ..., seq_len - 1
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Angles: outer product of positions and frequencies
    # angles[p, i] = pos[p] * freq[i]
    angles = torch.outer(pos, freq)  # (seq_len, half)

    # Tile the angle table so dimensions i and i+half share an angle:
    # [a_0, ..., a_half-1, a_0, ..., a_half-1] — the half-split layout that
    # apply_rope rotates in. Shape becomes (seq_len, head_dim).
    cos_cache = torch.cos(angles).repeat(1, 2)  # (seq_len, head_dim)
    sin_cache = torch.sin(angles).repeat(1, 2)  # (seq_len, head_dim)

    return cos_cache, sin_cache


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to Q or K.

    Dimensions are paired half against half — ``(x_i, x_i+half)`` is one 2D vector, rotated
    by that position's angle:
        x_i'      = x_i * cos - x_i+half * sin
        x_i+half' = x_i * sin + x_i+half * cos
    This is the layout build_rope_cache emits (i and i+half share an angle), not the
    interleaved-pair variant.

    Args:
        x:   (..., seq_len, head_dim) — typically Q or K.
        cos: (seq_len, head_dim) — cosine cache from build_rope_cache. A single-row
             (1, head_dim) cache is allowed and applies one rotation to every position.
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

    # Partner vector of the rotation: the upper half moves to the front, negated.
    # [d_0 ... d_half-1, d_half ... d_2half-1] -> [-d_half ... -d_2half-1, d_0 ... d_half-1]
    half = head_dim // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    return x * cos + x_rot * sin


def shift_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """
    Rotate ``x`` **backward** by a constant ``shift`` positions in RoPE space.

    If ``x`` carries RoPE for absolute positions ``[p, p+L)``, then after
    ``shift_rope(x, cos, sin, shift=s)`` it carries RoPE for ``[p - s, p - s + L)``. It is the
    inverse rotation by ``s`` positions, applied as RoPE with angle ``-s * freq`` via
    ``cos(-a) = cos(a)``, ``sin(-a) = -sin(a)``.

    Used by the KV cache sliding window: evicting the oldest ``s`` frames moves the remaining
    cached K from effective positions ``[s, T_cached)`` back to ``[0, T_cached - s)``, which
    is one uniform rotation by ``-s`` rather than a re-encode.

    Args:
        x: ``(..., seq_len, head_dim)`` — e.g. cached K tensors of shape
           ``(N, n_kv_heads, T_cached, head_dim)``.
        cos, sin: RoPE caches of shape ``(max_T, head_dim)`` built via
           :func:`build_rope_cache`.
        shift: non-negative shift amount (in positions). ``shift=0`` is a
           no-op; must satisfy ``0 <= shift < max_T``.

    Returns:
        Tensor of the same shape as ``x``, with RoPE rotated back by ``shift``.
    """
    if shift == 0:
        return x
    # Pick the cos/sin at position `shift` (shape (1, head_dim)) — broadcasting
    # across seq_len applies the same rotation to every time position.
    cos_s = cos[shift:shift + 1]
    sin_s = sin[shift:shift + 1]
    return apply_rope(x, cos_s, -sin_s)
