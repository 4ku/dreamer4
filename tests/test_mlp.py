"""Tests for dreamer4.mlp — SwiGLU feed-forward network."""

import torch
import pytest

from dreamer4.mlp import SwiGLU


def test_output_shape():
    """Output shape should match input shape."""
    mlp = SwiGLU(d_model=128)
    x = torch.randn(2, 16, 128)
    y = mlp(x)
    assert y.shape == (2, 16, 128)


def test_output_shape_arbitrary_leading_dims():
    """SwiGLU should work with any number of leading dimensions."""
    mlp = SwiGLU(d_model=64)
    x = torch.randn(2, 3, 4, 64)
    y = mlp(x)
    assert y.shape == (2, 3, 4, 64)


def test_gradient_flows():
    """Gradients should flow to both the gate and up paths."""
    mlp = SwiGLU(d_model=32)
    x = torch.randn(2, 4, 32, requires_grad=True)
    y = mlp(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_custom_mlp_ratio():
    """Different mlp_ratio should change the hidden dimension."""
    mlp_small = SwiGLU(d_model=64, mlp_ratio=2.0)
    mlp_large = SwiGLU(d_model=64, mlp_ratio=8.0)

    # fc_in output size is 2 * hidden, and hidden = d_model * mlp_ratio
    # So fc_in.weight has shape (2 * hidden, d_model)
    assert mlp_small.fc_in.weight.shape[0] == 2 * 64 * 2   # 256
    assert mlp_large.fc_in.weight.shape[0] == 2 * 64 * 8   # 1024


def test_output_is_finite():
    """Output should be finite for reasonable inputs."""
    mlp = SwiGLU(d_model=128, mlp_ratio=4.0)
    x = torch.randn(4, 32, 128)
    y = mlp(x)
    assert torch.isfinite(y).all()


def test_dropout_changes_training_behavior():
    """With dropout > 0, outputs should differ between train and eval mode."""
    mlp = SwiGLU(d_model=64, dropout=0.5)
    x = torch.randn(2, 8, 64)

    mlp.train()
    y_train = mlp(x)

    mlp.eval()
    y_eval = mlp(x)

    # In eval mode, dropout is disabled, so outputs should differ
    # (with high probability, since dropout is 0.5)
    assert not torch.allclose(y_train, y_eval, atol=1e-6)


def test_zero_input_gives_zero_output():
    """
    With zero input, SiLU(0) = 0 * sigmoid(0) = 0, so gate path is 0,
    and the output should be near-zero (bias terms might add a constant).
    """
    mlp = SwiGLU(d_model=32, dropout=0.0)
    # Zero out all biases for a clean test
    with torch.no_grad():
        for p in mlp.parameters():
            if p.dim() == 1:  # bias
                p.zero_()

    x = torch.zeros(1, 1, 32)
    y = mlp(x)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-7, rtol=1e-7)
