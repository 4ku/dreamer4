"""Tests for dreamer4.norms — RMSNorm and QKNorm."""

import torch
import pytest

from dreamer4.norms import RMSNorm, QKNorm


# ===== RMSNorm tests =====

def test_rmsnorm_output_shape():
    """Output shape should match input shape."""
    norm = RMSNorm(128)
    x = torch.randn(2, 16, 128)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_unit_rms():
    """After RMSNorm with scale=1, the RMS of each position should be ~1.0."""
    dim = 256
    norm = RMSNorm(dim)
    # scale is already ones by default
    x = torch.randn(4, 32, dim)
    y = norm(x)
    # RMS per position: sqrt(mean(y_i^2))
    rms = y.pow(2).mean(dim=-1).sqrt()
    # Should be close to 1.0 (within tolerance of learnable scale=1)
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)


def test_rmsnorm_learnable_scale():
    """Changing the scale parameter should proportionally scale the output."""
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 8, dim)

    y1 = norm(x)

    # Double the scale
    with torch.no_grad():
        norm.scale.mul_(2.0)
    y2 = norm(x)

    # Output should be exactly 2x
    torch.testing.assert_close(y2, y1 * 2.0, atol=1e-5, rtol=1e-5)


def test_rmsnorm_gradient_flows():
    """Gradients should flow through RMSNorm to both input and scale."""
    dim = 32
    norm = RMSNorm(dim)
    x = torch.randn(2, 4, dim, requires_grad=True)
    y = norm(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.scale.grad is not None
    assert x.grad.shape == x.shape


def test_rmsnorm_invariant_to_input_scale():
    """
    RMSNorm should produce the same direction regardless of input magnitude.
    If we multiply input by a constant, the output direction stays the same.
    """
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(1, 1, dim)
    y1 = norm(x)
    y2 = norm(x * 100.0)
    # Same direction means y1 and y2 should be equal
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)


# ===== QKNorm tests =====

def test_qknorm_output_shape():
    """Output shapes should match input shapes."""
    qknorm = QKNorm(head_dim=64, n_heads=8)
    q = torch.randn(2, 8, 32, 64)
    k = torch.randn(2, 8, 32, 64)
    q_out, k_out = qknorm(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_qknorm_bounds_attention_logits():
    """
    After QKNorm, attention logits (Q @ K^T) should be bounded.
    Without normalization, large inputs can produce unbounded logits.
    """
    head_dim = 64
    n_heads = 4
    qknorm = QKNorm(head_dim=head_dim, n_heads=n_heads)

    # Very large inputs that would cause huge logits without normalization
    q = torch.randn(1, n_heads, 16, head_dim) * 1000.0
    k = torch.randn(1, n_heads, 16, head_dim) * 1000.0

    q_normed, k_normed = qknorm(q, k)

    # Compute attention logits
    logits = torch.matmul(q_normed, k_normed.transpose(-2, -1))

    # The max logit should be bounded (roughly by scale^2 * head_dim)
    # With default init, scale = sqrt(head_dim) = 8, so max ~= 64 * head_dim
    # The point is it shouldn't be millions like unnormalized
    assert logits.abs().max() < 1e4, f"Logits too large: {logits.abs().max()}"


def test_qknorm_gradient_flows():
    """Gradients should flow through QKNorm."""
    qknorm = QKNorm(head_dim=32, n_heads=4)
    q = torch.randn(2, 4, 8, 32, requires_grad=True)
    k = torch.randn(2, 4, 8, 32, requires_grad=True)
    q_out, k_out = qknorm(q, k)
    loss = q_out.sum() + k_out.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert qknorm.q_scale.grad is not None
    assert qknorm.k_scale.grad is not None


def test_qknorm_different_kv_heads():
    """QKNorm should work when K has fewer heads (for GQA)."""
    head_dim = 64
    n_heads = 8
    n_kv_heads = 2
    qknorm = QKNorm(head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)

    q = torch.randn(2, n_heads, 16, head_dim)
    k = torch.randn(2, n_kv_heads, 16, head_dim)

    q_out, k_out = qknorm(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_qknorm_initial_scale_preserves_magnitude():
    """
    With default initialization (scale=sqrt(head_dim)), the dot product
    magnitude after QKNorm should be similar to standard scaled dot-product.
    """
    head_dim = 64
    n_heads = 4
    seq_len = 16
    qknorm = QKNorm(head_dim=head_dim, n_heads=n_heads)

    q = torch.randn(8, n_heads, seq_len, head_dim)
    k = torch.randn(8, n_heads, seq_len, head_dim)

    q_normed, k_normed = qknorm(q, k)
    logits = torch.matmul(q_normed, k_normed.transpose(-2, -1))

    # Standard attention has logits with std ~1 after /sqrt(d)
    # QKNorm with scale=sqrt(d) should produce similar statistics
    logit_std = logits.std().item()
    # Should be in a reasonable range (not 0 and not huge)
    assert 0.1 < logit_std < 100, f"Logit std out of range: {logit_std}"
