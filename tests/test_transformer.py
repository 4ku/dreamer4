"""Tests for dreamer4.transformer — Full BlockCausalTransformer."""

import torch
import pytest

from dreamer4.modality import Modality, TokenLayout
from dreamer4.transformer import BlockCausalTransformer


def _encoder_layout(n_latents=4, n_patches=16):
    return TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))


def _wm_layout(n_spatial=8, n_register=2, n_agent=1):
    return TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SHORTCUT_SIGNAL, 1),
        (Modality.SHORTCUT_STEP, 1),
        (Modality.SPATIAL, n_spatial),
        (Modality.REGISTER, n_register),
        (Modality.AGENT, n_agent),
    ))


# ===== Shape tests =====

def test_output_shape_encoder():
    """Encoder-mode transformer preserves shape."""
    layout = _encoder_layout()
    S = layout.total_tokens()
    model = BlockCausalTransformer(
        d_model=64, n_heads=4, depth=4,
        layout=layout, space_mode="encoder",
        time_every=4,
    )
    x = torch.randn(2, 8, S, 64)
    y = model(x)
    assert y.shape == (2, 8, S, 64)


def test_output_shape_wm():
    """World model-mode transformer preserves shape."""
    layout = _wm_layout()
    S = layout.total_tokens()
    model = BlockCausalTransformer(
        d_model=64, n_heads=4, depth=8,
        layout=layout, space_mode="wm_agent_isolated",
        time_every=4,
    )
    x = torch.randn(2, 16, S, 64)
    y = model(x)
    assert y.shape == (2, 16, S, 64)


def test_output_shape_realistic():
    """Realistic sizes: B=2, T=16, S=64, D=256, depth=4."""
    layout = TokenLayout(n_latents=8, segments=((Modality.IMAGE, 56),))
    S = layout.total_tokens()
    model = BlockCausalTransformer(
        d_model=256, n_heads=4, depth=4,
        layout=layout, space_mode="encoder",
        time_every=4, latents_only_time=True,
    )
    x = torch.randn(2, 16, S, 256)
    y = model(x)
    assert y.shape == x.shape


# ===== Time layer counting =====

def test_time_layer_count():
    """With depth=8 and time_every=4, there should be 2 time layers."""
    layout = _encoder_layout()
    model = BlockCausalTransformer(
        d_model=32, n_heads=2, depth=8,
        layout=layout, space_mode="encoder",
        time_every=4,
    )
    assert model.count_time_layers() == 2  # layers 3, 7


def test_time_layer_count_every_layer():
    """With time_every=1, every layer has time attention."""
    layout = _encoder_layout()
    model = BlockCausalTransformer(
        d_model=32, n_heads=2, depth=4,
        layout=layout, space_mode="encoder",
        time_every=1,
    )
    assert model.count_time_layers() == 4


def test_time_layer_count_never():
    """With time_every > depth, no layer has time attention."""
    layout = _encoder_layout()
    model = BlockCausalTransformer(
        d_model=32, n_heads=2, depth=4,
        layout=layout, space_mode="encoder",
        time_every=100,
    )
    assert model.count_time_layers() == 0


# ===== GQA config =====

def test_gqa_config_propagates():
    """GQA configuration should propagate to all layers."""
    layout = _wm_layout()
    model = BlockCausalTransformer(
        d_model=64, n_heads=8, n_kv_heads=2, depth=4,
        layout=layout, space_mode="wm_agent_isolated",
    )
    for layer in model.layers:
        assert layer.space_attn.attn.n_kv_heads == 2
        assert layer.space_attn.attn.n_rep == 4  # 8 / 2


# ===== Gradient flow =====

def test_gradient_flows_to_all_params():
    """Gradients should flow to every parameter in the transformer."""
    layout = _encoder_layout(n_latents=2, n_patches=4)
    S = layout.total_tokens()
    model = BlockCausalTransformer(
        d_model=32, n_heads=2, depth=4,
        layout=layout, space_mode="encoder",
        time_every=4, latents_only_time=True,
    )
    x = torch.randn(1, 4, S, 32, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ===== Output is finite =====

def test_output_finite():
    """All outputs should be finite (no NaN or Inf)."""
    layout = _wm_layout()
    S = layout.total_tokens()
    model = BlockCausalTransformer(
        d_model=64, n_heads=4, n_kv_heads=2, depth=4,
        layout=layout, space_mode="wm_agent_isolated",
        use_qk_norm=True, logit_cap=50.0,
    )
    x = torch.randn(2, 8, S, 64)
    y = model(x)
    assert torch.isfinite(y).all()


# ===== Parameter counting =====

def test_parameter_count():
    """Parameter count should be positive and consistent."""
    layout = _encoder_layout()
    model = BlockCausalTransformer(
        d_model=64, n_heads=4, depth=4,
        layout=layout, space_mode="encoder",
        time_every=4,
    )
    count = model.count_parameters()
    assert count > 0
    # Count should match manual sum
    manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == manual_count


# ===== Causal property (end-to-end) =====

def test_causal_property_end_to_end():
    """
    Changing a future time step should not affect outputs at past time steps
    (end-to-end through the full transformer).
    """
    layout = _encoder_layout(n_latents=2, n_patches=4)
    S = layout.total_tokens()
    model = BlockCausalTransformer(
        d_model=32, n_heads=2, depth=4,
        layout=layout, space_mode="encoder",
        time_every=4, latents_only_time=True,
        logit_cap=None, use_qk_norm=False,
    )
    model.eval()

    x = torch.randn(1, 6, S, 32)
    y1 = model(x)

    x2 = x.clone()
    x2[0, 5] = torch.randn(S, 32)  # change last time step
    y2 = model(x2)

    # First 5 time steps should be unchanged
    torch.testing.assert_close(y1[:, :5], y2[:, :5], atol=1e-5, rtol=1e-5)


# ===== Smoke overfit test =====

def test_smoke_overfit():
    """
    A small transformer should be able to overfit a tiny random target.
    This verifies the model is trainable end-to-end.
    """
    torch.manual_seed(42)

    layout = TokenLayout(n_latents=2, segments=((Modality.IMAGE, 4),))
    S = layout.total_tokens()  # 6
    d_model = 32

    model = BlockCausalTransformer(
        d_model=d_model, n_heads=2, depth=2,
        layout=layout, space_mode="encoder",
        time_every=2,
        latents_only_time=True,
        use_qk_norm=False,
        logit_cap=None,
        dropout=0.0,
    )

    # Fixed random input and target
    x = torch.randn(1, 4, S, d_model)
    target = torch.randn(1, 4, S, d_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for a few steps
    initial_loss = None
    for step in range(200):
        y = model(x)
        loss = (y - target).pow(2).mean()
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # The loss should decrease significantly
    assert final_loss < initial_loss * 0.1, (
        f"Model failed to overfit: initial_loss={initial_loss:.4f}, "
        f"final_loss={final_loss:.4f}"
    )


# ===== Different modes =====

@pytest.mark.parametrize("mode", ["encoder", "decoder", "wm_agent_isolated", "wm_agent"])
def test_all_modes(mode):
    """Transformer should work in all attention modes."""
    if mode in ("encoder", "decoder"):
        layout = _encoder_layout()
    else:
        layout = _wm_layout()
    S = layout.total_tokens()

    model = BlockCausalTransformer(
        d_model=32, n_heads=2, depth=2,
        layout=layout, space_mode=mode,
        time_every=2,
    )
    x = torch.randn(1, 4, S, 32)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
