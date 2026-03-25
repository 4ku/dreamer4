"""Tests for dreamer4.rope — Rotary Position Embeddings."""

import torch
import pytest

from dreamer4.transformer.rope import build_rope_cache, apply_rope


# ===== Shape tests =====

def test_build_rope_cache_shape():
    """Cos/sin caches have the correct shape."""
    seq_len, head_dim = 128, 64
    cos, sin = build_rope_cache(seq_len, head_dim)
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)


def test_apply_rope_preserves_shape():
    """apply_rope should not change the tensor shape."""
    B, H, L, D = 2, 4, 32, 64
    cos, sin = build_rope_cache(L, D)
    x = torch.randn(B, H, L, D)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape

# ===== Identity at position 0 =====

def test_position_zero_is_identity():
    """
    At position 0, all rotation angles are 0, so cos=1, sin=0.
    Applying RoPE at position 0 should return the input unchanged.
    """
    head_dim = 64
    cos, sin = build_rope_cache(4, head_dim)
    x = torch.randn(1, 1, 1, head_dim)  # single token at position 0
    y = apply_rope(x, cos[:1], sin[:1])
    torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)


# ===== Relative position invariance =====

def test_relative_position_invariance():
    """
    The dot product of two RoPE-encoded vectors should depend only on
    their relative position, not absolute position.

    Specifically: dot(RoPE(q, pos_a), RoPE(k, pos_b))
                  should equal
                  dot(RoPE(q, pos_a + shift), RoPE(k, pos_b + shift))
    """
    head_dim = 32
    max_len = 100
    cos, sin = build_rope_cache(max_len, head_dim)

    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    # Positions (3, 7) -> relative distance = 4
    q1 = apply_rope(q.unsqueeze(0), cos[3:4], sin[3:4]).squeeze(0)
    k1 = apply_rope(k.unsqueeze(0), cos[7:8], sin[7:8]).squeeze(0)
    dot1 = torch.dot(q1, k1)

    # Positions (20, 24) -> same relative distance = 4
    q2 = apply_rope(q.unsqueeze(0), cos[20:21], sin[20:21]).squeeze(0)
    k2 = apply_rope(k.unsqueeze(0), cos[24:25], sin[24:25]).squeeze(0)
    dot2 = torch.dot(q2, k2)

    torch.testing.assert_close(dot1, dot2, atol=1e-4, rtol=1e-4)


def test_different_relative_positions_give_different_dots():
    """
    Vectors with different relative positions should generally have
    different dot products (the encoding is not trivial).
    """
    head_dim = 32
    cos, sin = build_rope_cache(50, head_dim)

    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    # Relative distance = 1
    q1 = apply_rope(q.unsqueeze(0), cos[0:1], sin[0:1]).squeeze(0)
    k1 = apply_rope(k.unsqueeze(0), cos[1:2], sin[1:2]).squeeze(0)
    dot_close = torch.dot(q1, k1)

    # Relative distance = 10
    q2 = apply_rope(q.unsqueeze(0), cos[0:1], sin[0:1]).squeeze(0)
    k2 = apply_rope(k.unsqueeze(0), cos[10:11], sin[10:11]).squeeze(0)
    dot_far = torch.dot(q2, k2)

    # They should differ
    assert not torch.allclose(dot_close, dot_far, atol=1e-3)


# ===== RoPE preserves norm =====

def test_rope_preserves_norm():
    """
    Rotation should preserve the vector norm (it's a unitary operation).
    """
    head_dim = 64
    cos, sin = build_rope_cache(32, head_dim)
    x = torch.randn(2, 4, 32, head_dim)
    y = apply_rope(x, cos, sin)

    x_norms = x.norm(dim=-1)
    y_norms = y.norm(dim=-1)
    torch.testing.assert_close(x_norms, y_norms, atol=1e-5, rtol=1e-5)

