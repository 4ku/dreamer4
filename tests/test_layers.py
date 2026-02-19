"""Tests for dreamer4.layers — SpaceAttention, TimeAttention, BlockCausalLayer."""

import torch
import pytest

from dreamer4.modality import Modality, TokenLayout
from dreamer4.layers import SpaceAttention, TimeAttention, BlockCausalLayer


def _encoder_layout(n_latents: int = 4, n_patches: int = 16) -> TokenLayout:
    return TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))


def _wm_layout(n_spatial: int = 8, n_register: int = 2, n_agent: int = 1) -> TokenLayout:
    return TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SHORTCUT_SIGNAL, 1),
        (Modality.SHORTCUT_STEP, 1),
        (Modality.SPATIAL, n_spatial),
        (Modality.REGISTER, n_register),
        (Modality.AGENT, n_agent),
    ))


# ===== SpaceAttention =====

def test_space_attn_output_shape():
    """SpaceAttention output shape matches input."""
    layout = _encoder_layout()
    S = layout.total_tokens()
    attn = SpaceAttention(d_model=64, n_heads=4, n_kv_heads=None, layout=layout, mode="encoder")
    x = torch.randn(2, 8, S, 64)
    y = attn(x)
    assert y.shape == (2, 8, S, 64)


def test_space_attn_time_independence():
    """
    SpaceAttention processes each time step independently.
    Changing one time step should not affect others.
    """
    layout = _encoder_layout(n_latents=2, n_patches=4)
    S = layout.total_tokens()
    attn = SpaceAttention(d_model=32, n_heads=2, n_kv_heads=None, layout=layout, mode="encoder")
    attn.eval()

    x = torch.randn(1, 4, S, 32)
    y1 = attn(x)

    # Modify time step 3 only
    x2 = x.clone()
    x2[0, 3] = torch.randn(S, 32)
    y2 = attn(x2)

    # Time steps 0, 1, 2 should be unchanged
    torch.testing.assert_close(y1[:, :3], y2[:, :3], atol=1e-5, rtol=1e-5)
    # Time step 3 should differ
    assert not torch.allclose(y1[:, 3], y2[:, 3], atol=1e-3)


def test_space_attn_gqa():
    """SpaceAttention works with GQA."""
    layout = _wm_layout()
    S = layout.total_tokens()
    attn = SpaceAttention(
        d_model=64, n_heads=8, n_kv_heads=2,
        layout=layout, mode="wm_agent_isolated",
    )
    x = torch.randn(2, 4, S, 64)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# ===== TimeAttention =====

def test_time_attn_output_shape():
    """TimeAttention output shape matches input."""
    attn = TimeAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 8, 6, 64)
    y = attn(x)
    assert y.shape == (2, 8, 6, 64)


def test_time_attn_causal():
    """
    TimeAttention is causal: changing a future time step should NOT affect
    the output at past time steps.
    """
    attn = TimeAttention(d_model=32, n_heads=2, logit_cap=None, use_qk_norm=False)
    attn.eval()

    x = torch.randn(1, 6, 4, 32)
    y1 = attn(x)

    # Modify time step 5 (the last one)
    x2 = x.clone()
    x2[0, 5] = torch.randn(4, 32)
    y2 = attn(x2)

    # Time steps 0-4 should be unchanged
    torch.testing.assert_close(y1[:, :5], y2[:, :5], atol=1e-5, rtol=1e-5)


def test_time_attn_latents_only():
    """
    With latents_only=True, only the first n_latents spatial positions
    should change; the rest should pass through unchanged.
    """
    n_latents = 3
    S = 8
    attn = TimeAttention(
        d_model=32, n_heads=2,
        latents_only=True, n_latents=n_latents,
        logit_cap=None, use_qk_norm=False,
    )
    attn.eval()

    x = torch.randn(1, 4, S, 32)
    y = attn(x)

    # Non-latent positions (3:8) should be unchanged
    torch.testing.assert_close(y[:, :, n_latents:], x[:, :, n_latents:], atol=1e-6, rtol=1e-6)

    # Latent positions (0:3) should generally differ (attention changes them)
    # (Not guaranteed, but extremely unlikely with random weights)
    assert not torch.allclose(y[:, :, :n_latents], x[:, :, :n_latents], atol=1e-3)


def test_time_attn_gqa():
    """TimeAttention works with GQA."""
    attn = TimeAttention(d_model=64, n_heads=8, n_kv_heads=2)
    x = torch.randn(2, 8, 4, 64)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# ===== BlockCausalLayer =====

def test_block_causal_layer_output_shape():
    """BlockCausalLayer output shape matches input."""
    layout = _encoder_layout()
    S = layout.total_tokens()
    layer = BlockCausalLayer(
        d_model=64, n_heads=4, n_kv_heads=None,
        layout=layout, space_mode="encoder",
        layer_index=3, time_every=4,  # layer 3 HAS time attention
        n_latents=4, latents_only_time=True,
    )
    x = torch.randn(2, 8, S, 64)
    y = layer(x)
    assert y.shape == x.shape


def test_block_causal_layer_has_time():
    """Layer at index 3 with time_every=4 should have time attention."""
    layout = _encoder_layout()
    layer = BlockCausalLayer(
        d_model=32, n_heads=2, n_kv_heads=None,
        layout=layout, space_mode="encoder",
        layer_index=3, time_every=4,
    )
    assert layer.has_time
    assert hasattr(layer, "time_attn")


def test_block_causal_layer_no_time():
    """Layer at index 0 with time_every=4 should NOT have time attention."""
    layout = _encoder_layout()
    layer = BlockCausalLayer(
        d_model=32, n_heads=2, n_kv_heads=None,
        layout=layout, space_mode="encoder",
        layer_index=0, time_every=4,
    )
    assert not layer.has_time
    assert not hasattr(layer, "time_attn")


@pytest.mark.parametrize("layer_index", range(8))
def test_time_every_pattern(layer_index):
    """Check which layers get time attention with time_every=4."""
    layout = _encoder_layout()
    layer = BlockCausalLayer(
        d_model=32, n_heads=2, n_kv_heads=None,
        layout=layout, space_mode="encoder",
        layer_index=layer_index, time_every=4,
    )
    expected = (layer_index + 1) % 4 == 0
    assert layer.has_time == expected, f"Layer {layer_index}: expected has_time={expected}"


def test_block_causal_layer_gradient():
    """Gradients flow through the block-causal layer."""
    layout = _encoder_layout(n_latents=2, n_patches=4)
    S = layout.total_tokens()
    layer = BlockCausalLayer(
        d_model=32, n_heads=2, n_kv_heads=None,
        layout=layout, space_mode="encoder",
        layer_index=3, time_every=4,
        n_latents=2, latents_only_time=True,
    )
    x = torch.randn(2, 4, S, 32, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_block_causal_layer_residual():
    """
    The residual connection means output should be close to input
    when all weights are very small (near-identity transformation).
    """
    layout = _encoder_layout(n_latents=2, n_patches=4)
    S = layout.total_tokens()
    layer = BlockCausalLayer(
        d_model=32, n_heads=2, n_kv_heads=None,
        layout=layout, space_mode="encoder",
        layer_index=0, time_every=4,
    )

    # Make all parameters very small
    with torch.no_grad():
        for p in layer.parameters():
            p.mul_(0.001)

    x = torch.randn(1, 2, S, 32)
    y = layer(x)

    # Output should be close to input (residual dominates)
    diff = (y - x).abs().mean().item()
    assert diff < 1.0, f"Residual should dominate, but diff={diff}"


def test_block_causal_layer_wm_mode():
    """BlockCausalLayer works with world model layout and mode."""
    layout = _wm_layout()
    S = layout.total_tokens()
    layer = BlockCausalLayer(
        d_model=64, n_heads=4, n_kv_heads=2,
        layout=layout, space_mode="wm_agent_isolated",
        layer_index=3, time_every=4,
    )
    x = torch.randn(2, 4, S, 64)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
