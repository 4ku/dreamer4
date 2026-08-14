"""
Distributions for Dreamer 4 agent heads.

Provides:
    symlog / symexp — Symmetric log/exp transforms (Dreamer 3, Appendix B).
    SymExpTwoHot   — Discretized distribution used for reward and value
                     prediction. Bin centers are linearly spaced in symlog
                     space then mapped to value space via symexp.

The two-hot encoding places probability mass on exactly two adjacent bins,
interpolating linearly based on proximity to the true value. This gives a
smooth, differentiable target for cross-entropy training while remaining
expressive enough to represent multi-modal reward distributions.

Paper reference: Section 3.3 (reward/value heads use symexp twohot output).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: sign(x) * ln(|x| + 1)."""
    return x.sign() * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential (inverse of symlog): sign(x) * (exp(|x|) - 1)."""
    return x.sign() * (torch.exp(x.abs()) - 1.0)


class SymExpTwoHot(nn.Module):
    """
    Discretized scalar distribution with symexp-spaced bins.

    Bin centers are uniformly spaced in symlog space over [low, high],
    then mapped to value space via symexp. This concentrates resolution
    near zero (where most rewards live) while covering a wide range.

    Args:
        num_bins: Number of discrete bins (odd recommended for a bin at 0).
        low:      Lower bound in symlog space.
        high:     Upper bound in symlog space.
    """

    def __init__(
        self,
        num_bins: int = 255,
        low: float = -2.0,
        high: float = 2.0,
    ):
        super().__init__()
        self.num_bins = num_bins

        symlog_bins = torch.linspace(low, high, num_bins)
        bin_values = symexp(symlog_bins)
        self.register_buffer("bin_values", bin_values)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Produce a soft two-hot encoding for scalar values.

        For each value, finds the two adjacent bins and distributes
        weight proportional to proximity (linear interpolation).

        Args:
            values: (*) arbitrary-shape scalar tensor.

        Returns:
            (*, num_bins) two-hot encoding with exactly two non-zero
            entries per row that sum to 1.
        """
        shape = values.shape
        flat = values.reshape(-1)

        lo = self.bin_values[0]
        hi = self.bin_values[-1]
        flat = flat.clamp(lo, hi)

        idx = torch.searchsorted(self.bin_values, flat)
        left = (idx - 1).clamp(min=0)
        right = (left + 1).clamp(max=self.num_bins - 1)

        left_val = self.bin_values[left]
        right_val = self.bin_values[right]

        span = (right_val - left_val).clamp_min(1e-8)
        w_right = (flat - left_val) / span
        w_left = 1.0 - w_right

        encoded = torch.zeros(flat.shape[0], self.num_bins,
                              device=flat.device, dtype=flat.dtype)
        encoded.scatter_(-1, left.unsqueeze(-1), w_left.unsqueeze(-1))
        encoded.scatter_(-1, right.unsqueeze(-1), w_right.unsqueeze(-1))

        return encoded.reshape(*shape, self.num_bins)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decode bin logits to scalar values.

        Applies softmax then computes expectation over bin values.

        Args:
            logits: (*, num_bins) raw logits from network.

        Returns:
            (*) scalar predicted values.
        """
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_values).sum(dim=-1)

    def log_prob(
        self, logits: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative cross-entropy of logits against two-hot targets.

        Args:
            logits: (*, num_bins) raw logits.
            values: (*) scalar targets.

        Returns:
            (*) log probability (negative cross-entropy per element).
        """
        target = self.encode(values)
        log_probs = F.log_softmax(logits, dim=-1)
        return (target * log_probs).sum(dim=-1)

    def loss(
        self, logits: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean cross-entropy loss against two-hot targets.

        Args:
            logits: (*, num_bins) raw logits.
            values: (*) scalar targets.

        Returns:
            Scalar loss (mean over all elements).
        """
        return -self.log_prob(logits, values).mean()
