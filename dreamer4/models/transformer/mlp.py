"""
SwiGLU feed-forward block, applied after every attention sublayer of the transformer.

    MLP(x) = W_out(SiLU(W_gate(x)) * W_up(x)),   SiLU(z) = z * sigmoid(z)

The element-wise product gates each hidden unit, which trains better than a plain ReLU MLP
and is the usual choice in modern transformers. W_gate and W_up are fused into one linear
layer of width ``2 * hidden`` and chunked afterwards: one GEMM instead of two.

Input and output are ``(..., d_model)`` — the block acts per token, so the leading axes
(B, T, S) pass through untouched.

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
        mlp_ratio: Hidden dimension = d_model * mlp_ratio (default 8/3, which keeps the
                   parameter count of a gated MLP close to a 4x ungated one).
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

    def __init__(self, d_model: int, mlp_ratio: float = 8/3, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)

        # Single linear producing both the gate and the up projection;
        # its 2 * hidden outputs are chunked in forward()
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
