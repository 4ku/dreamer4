"""
Utility functions for Dreamer 4.

This module provides low-level helpers used across the codebase:

1. **Bottleneck packing** — reshape tokenizer bottleneck representations
   for consumption by the dynamics model. The tokenizer produces
   (N_b, D_b) latents which are packed to (N_z, D_b * k) spatial tokens
   where N_z = N_b / k  (Paper Appendix A).

2. **RMS loss normalization** — normalize losses by their running RMS estimate.
   This helps control the relative importance of different loss terms.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Bottleneck packing (tokenizer <-> dynamics interface)
# ---------------------------------------------------------------------------


def pack_bottleneck_to_spatial(
    z: torch.Tensor, *, n_spatial: int, k: int
) -> torch.Tensor:
    """
    Pack tokenizer bottleneck into dynamics spatial tokens.

    The tokenizer bottleneck has shape (B, T, N_b, D_b) where N_b = n_spatial * k.
    This reshapes to (B, T, n_spatial, D_b * k) for the dynamics model.

    Example (Minecraft, Appendix A):
        N_b=512, D_b=16, k=2 -> n_spatial=256, D_spatial=32

    Args:
        z: (B, T, N_b, D_b) bottleneck tensor.
        n_spatial: Number of spatial tokens for dynamics.
        k: Packing factor (N_b / n_spatial).

    Returns:
        (B, T, n_spatial, D_b * k) packed tensor.
    """
    B, T, N_b, D_b = z.shape
    assert N_b == n_spatial * k, f"N_b={N_b} must equal n_spatial*k={n_spatial * k}"
    return z.reshape(B, T, n_spatial, k * D_b)


def unpack_spatial_to_bottleneck(z: torch.Tensor, *, k: int) -> torch.Tensor:
    """
    Unpack dynamics spatial tokens back to tokenizer bottleneck shape.

    Inverse of pack_bottleneck_to_spatial.

    Args:
        z: (B, T, n_spatial, D_b * k) packed tensor.
        k: Packing factor.

    Returns:
        (B, T, n_spatial * k, D_b) bottleneck tensor.
    """
    B, T, S, DK = z.shape
    assert DK % k == 0, f"Last dim {DK} must be divisible by k={k}"
    D_b = DK // k
    return z.reshape(B, T, S * k, D_b)


class RMSLossNormalizer(nn.Module):
    """
    Running RMS loss normalization.

    Maintains an exponential moving average of each loss term's RMS.
    Divides each loss by its running RMS estimate, so fixed coefficients
    control relative importance independent of raw loss magnitudes.

    Args:
        n_losses:  Number of loss terms to track.
        decay:     EMA decay factor (default 0.99).
        eps:       Floor for the RMS estimate (default 1e-8).
    """

    def __init__(self, n_losses: int = 2, decay: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("rms_ema", torch.ones(n_losses))
        self.register_buffer("initialized", torch.zeros(n_losses, dtype=torch.bool))

    @torch.no_grad()
    def update(self, idx: int, loss_val: torch.Tensor) -> None:
        """Update the running RMS estimate for loss at index `idx`."""
        val = loss_val.detach().float().abs().clamp_min(self.eps)
        if not self.initialized[idx]:
            self.rms_ema[idx] = val
            self.initialized[idx] = True
        else:
            self.rms_ema[idx] = self.decay * self.rms_ema[idx] + (1 - self.decay) * val

    def normalize(self, idx: int, loss: torch.Tensor) -> torch.Tensor:
        """Divide loss by its running RMS estimate."""
        return loss / self.rms_ema[idx].clamp_min(self.eps)
