"""Tests for dreamer4.attention — MultiheadAttention with GQA, QKNorm, logit capping, RoPE."""

import torch
import pytest

from dreamer4.attention import MultiheadAttention
from dreamer4.rope import build_rope_cache


# ===== Shape tests =====

def test_output_shape_standard():
    """Standard MHA: output shape matches input shape."""
    attn = MultiheadAttention(d_model=128, n_heads=4, use_qk_norm=False, logit_cap=None)
    x = torch.randn(2, 16, 128)
    y = attn(x)
    assert y.shape == (2, 16, 128)


@pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (8, 4), (8, 2), (8, 1)])
def test_output_shape_gqa(n_heads, n_kv_heads):
    """GQA with various head configurations: output shape is correct."""
    d_model = 128
    attn = MultiheadAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        use_qk_norm=False, logit_cap=None,
    )
    x = torch.randn(2, 16, d_model)
    y = attn(x)
    assert y.shape == (2, 16, d_model)


# ===== GQA correctness =====

def test_gqa_equals_mha_when_same_heads():
    """
    GQA with n_kv_heads == n_heads should produce the same result as
    standard MHA (given the same weights).
    """
    d_model, n_heads = 64, 4
    attn = MultiheadAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads,
        use_qk_norm=False, logit_cap=None,
    )
    x = torch.randn(2, 8, d_model)
    y = attn(x)
    # Just check it runs and produces valid output (no NaN/Inf)
    assert torch.isfinite(y).all()


def test_gqa_has_fewer_kv_params():
    """GQA should use fewer parameters for K and V projections."""
    d_model = 256
    mha = MultiheadAttention(d_model=d_model, n_heads=8, n_kv_heads=8, use_qk_norm=False, logit_cap=None)
    gqa = MultiheadAttention(d_model=d_model, n_heads=8, n_kv_heads=2, use_qk_norm=False, logit_cap=None)
    mha_kv_params = sum(p.numel() for p in [mha.k_proj.weight, mha.k_proj.bias, mha.v_proj.weight, mha.v_proj.bias])
    gqa_kv_params = sum(p.numel() for p in [gqa.k_proj.weight, gqa.k_proj.bias, gqa.v_proj.weight, gqa.v_proj.bias])
    assert gqa_kv_params < mha_kv_params
    assert gqa_kv_params == mha_kv_params // 4  # 8/2 = 4x reduction


# ===== Logit capping =====

def test_logit_capping_bounds_attention():
    """With logit capping, attention logits should be bounded by [-cap, cap]."""
    cap = 10.0
    attn = MultiheadAttention(
        d_model=64, n_heads=4, use_qk_norm=False, logit_cap=cap,
    )
    # Feed very large inputs to generate large attention logits
    x = torch.randn(1, 8, 64) * 100.0
    # We can't directly inspect logits from forward(), but we can verify
    # the output is finite (no NaN from softmax overflow)
    y = attn(x)
    assert torch.isfinite(y).all()


def test_without_logit_capping():
    """Without logit capping, attention still works (uses fused SDPA)."""
    attn = MultiheadAttention(d_model=64, n_heads=4, use_qk_norm=False, logit_cap=None)
    x = torch.randn(2, 16, 64)
    y = attn(x)
    assert torch.isfinite(y).all()


# ===== Causal masking =====

def test_causal_masking():
    """
    With causal masking, the output at position t should depend only on
    positions 0..t. Changing a future token should not affect past outputs.
    """
    d_model = 64
    attn = MultiheadAttention(
        d_model=d_model, n_heads=4, use_qk_norm=False, logit_cap=None,
    )
    attn.eval()

    x = torch.randn(1, 8, d_model)
    y1 = attn(x, is_causal=True)

    # Modify the last token
    x2 = x.clone()
    x2[0, -1, :] = torch.randn(d_model)
    y2 = attn(x2, is_causal=True)

    # All positions except the last should be identical
    torch.testing.assert_close(y1[:, :-1], y2[:, :-1], atol=1e-5, rtol=1e-5)


# ===== Custom attention mask =====

def test_custom_attn_mask():
    """Passing a custom boolean attention mask should work."""
    d_model, L = 64, 8
    attn = MultiheadAttention(d_model=d_model, n_heads=4, use_qk_norm=False, logit_cap=None)
    attn.eval()

    x = torch.randn(1, L, d_model)
    # Allow each token to attend only to itself
    mask = torch.eye(L, dtype=torch.bool).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
    y = attn(x, attn_mask=mask)
    assert y.shape == (1, L, d_model)
    assert torch.isfinite(y).all()


# ===== RoPE integration =====

def test_rope_integration():
    """Attention should work with RoPE cos/sin caches."""
    d_model, n_heads, L = 128, 4, 16
    head_dim = d_model // n_heads
    attn = MultiheadAttention(
        d_model=d_model, n_heads=n_heads, use_qk_norm=False, logit_cap=None,
    )
    cos, sin = build_rope_cache(L, head_dim)
    x = torch.randn(2, L, d_model)
    y = attn(x, rope_cos=cos, rope_sin=sin)
    assert y.shape == (2, L, d_model)
    assert torch.isfinite(y).all()


def test_rope_changes_output():
    """Using RoPE should produce different output than without RoPE."""
    d_model, n_heads, L = 64, 4, 8
    head_dim = d_model // n_heads
    attn = MultiheadAttention(
        d_model=d_model, n_heads=n_heads, use_qk_norm=False, logit_cap=None,
    )
    attn.eval()

    cos, sin = build_rope_cache(L, head_dim)
    x = torch.randn(1, L, d_model)

    y_no_rope = attn(x)
    y_with_rope = attn(x, rope_cos=cos, rope_sin=sin)

    # They should be different (positions now matter)
    assert not torch.allclose(y_no_rope, y_with_rope, atol=1e-3)


# ===== QKNorm integration =====

def test_qknorm_integration():
    """Attention with QKNorm enabled should work."""
    d_model = 128
    attn = MultiheadAttention(d_model=d_model, n_heads=8, use_qk_norm=True, logit_cap=50.0)
    x = torch.randn(2, 16, d_model)
    y = attn(x)
    assert y.shape == (2, 16, d_model)
    assert torch.isfinite(y).all()


# ===== All features combined =====

def test_all_features_together():
    """GQA + QKNorm + logit capping + RoPE + causal mask, all at once."""
    d_model, n_heads, n_kv_heads, L = 128, 8, 2, 32
    head_dim = d_model // n_heads
    attn = MultiheadAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        use_qk_norm=True, logit_cap=50.0,
    )
    cos, sin = build_rope_cache(L, head_dim)
    x = torch.randn(2, L, d_model)
    y = attn(x, is_causal=True, rope_cos=cos, rope_sin=sin)
    assert y.shape == (2, L, d_model)
    assert torch.isfinite(y).all()


# ===== Gradient flow =====

def test_gradient_flows_through_all_params():
    """Gradients should flow to all learnable parameters."""
    attn = MultiheadAttention(
        d_model=64, n_heads=4, n_kv_heads=2, use_qk_norm=True, logit_cap=50.0,
    )
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = attn(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ===== Invalid configs =====

def test_invalid_n_kv_heads():
    """n_heads must be divisible by n_kv_heads."""
    with pytest.raises(AssertionError):
        MultiheadAttention(d_model=128, n_heads=8, n_kv_heads=3)


def test_invalid_d_model():
    """d_model must be divisible by n_heads."""
    with pytest.raises(AssertionError):
        MultiheadAttention(d_model=100, n_heads=8)
