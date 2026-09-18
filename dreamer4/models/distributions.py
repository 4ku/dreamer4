"""Symlog and two-hot primitives for the scalar heads (reward and value).

The reward and value heads do not regress a scalar directly; they predict a categorical
distribution over fixed bins and take its expectation, as in DreamerV3. Bin centres are spaced
uniformly in symlog space and mapped back with symexp, which concentrates resolution near zero
(where most rewards live) while still covering a wide range.

    symlog / symexp -- symmetric log and exp transforms, inverses of each other.
    SymExpTwoHot    -- the binned distribution: ``encode`` builds the soft target, ``decode``
                       turns logits back into a scalar, ``loss`` is the cross-entropy.

The two-hot target puts mass on exactly the two bins adjacent to the true value, interpolating
linearly between them, which keeps the cross-entropy target smooth while letting the prediction
stay multi-modal.
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
    """Discretized scalar distribution with symexp-spaced bins.

    Args:
        num_bins: Number of discrete bins (odd recommended, so that one bin sits at 0).
        low:      Lower bound in SYMLOG space; the lowest bin centre is ``symexp(low)``.
        high:     Upper bound in SYMLOG space. The range must cover the environment's rewards
                  and little more -- far-out bins carrying even a thousandth of the probability
                  mass move the decoded scalar by orders of magnitude.
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
        """Produce a soft two-hot encoding for scalar values.

        For each value, finds the two adjacent bins and splits the weight between them in
        proportion to proximity (linear interpolation). Values outside the bin range are clamped.

        Args:
            values: (*) arbitrary-shape scalar tensor.

        Returns:
            (*, num_bins) two-hot encoding with at most two non-zero entries per row, summing to 1.
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
        """Decode bin logits to scalar values: softmax, then expectation over the bin centres.

        Args:
            logits: (*, num_bins) raw logits from the head.

        Returns:
            (*) scalar predicted values.
        """
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_values).sum(dim=-1)

    def log_prob(
        self, logits: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Negative cross-entropy of logits against the two-hot targets of ``values``.

        Args:
            logits: (*, num_bins) raw logits.
            values: (*) scalar targets.

        Returns:
            (*) log probability, one per element.
        """
        target = self.encode(values)
        log_probs = F.log_softmax(logits, dim=-1)
        return (target * log_probs).sum(dim=-1)

    def loss(
        self, logits: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Mean cross-entropy loss against the two-hot targets of ``values``.

        Args:
            logits: (*, num_bins) raw logits.
            values: (*) scalar targets.

        Returns:
            Scalar loss, averaged over all elements.
        """
        return -self.log_prob(logits, values).mean()
