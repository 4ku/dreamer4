"""
SwiGLU Feed-Forward Network for Dreamer 4.

This module implements the MLP (feed-forward) block used after every
attention sublayer in the transformer.

WHAT IS SwiGLU?

Standard transformer MLPs use:
    MLP(x) = W2(ReLU(W1(x)))

SwiGLU replaces ReLU with a *gated* activation:
    MLP(x) = W_out(SiLU(W_gate(x)) * W_up(x))

Here:
  - W_gate and W_up each project from d_model to hidden_dim
  - SiLU(z) = z * sigmoid(z)  (also called "Swish")
  - The element-wise product SiLU(gate) * up acts as a learnable gate

WHY SwiGLU?

The gating mechanism lets the network learn to selectively pass or
block information through each hidden dimension, producing smoother
gradients and better training dynamics than ReLU. It's used in most
modern transformers (LLaMA, PaLM, Gemma, etc.).

IMPLEMENTATION NOTE:

We fuse W_gate and W_up into a single linear layer that outputs
2 * hidden_dim, then split (chunk) the output. This is more memory-
efficient than two separate linear layers and fuses better on GPU.

Reference: Shazeer, 2020 — "GLU Variants Improve Transformer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Args:
        d_model:   Input and output dimension.
        mlp_ratio: Hidden dimension = d_model * mlp_ratio (default 4.0).
        dropout:   Dropout rate applied after each linear layer (default 0.0).

    Shape:
        Input:  (..., d_model) — any number of leading dimensions.
        Output: (..., d_model) — same shape as input.

    Example:
        >>> mlp = SwiGLU(d_model=256, mlp_ratio=4.0)
        >>> x = torch.randn(2, 16, 256)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([2, 16, 256])
    """

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)

        # Single linear that produces both gate and up projections
        # Output is 2 * hidden, which we split into two halves
        self.fc_in = nn.Linear(d_model, 2 * hidden)

        # Project back down to d_model
        self.fc_out = nn.Linear(hidden, d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to 2 * hidden_dim, then split into gate and up
        gate, up = self.fc_in(x).chunk(2, dim=-1)

        # SiLU activation on the gate path, element-wise multiply with up
        hidden = F.silu(gate) * up

        hidden = self.drop(hidden)

        # Project back to d_model
        out = self.fc_out(hidden)
        out = self.drop(out)

        return out
