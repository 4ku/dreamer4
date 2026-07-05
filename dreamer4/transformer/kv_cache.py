"""
KV-cache for incremental decoding of the block-causal transformer.

When the dynamics model rolls out autoregressively (paper Section 3.2
inference), each new frame is produced by K=4 denoising steps that all
condition on the same fixed past. Without a cache, every denoising step
re-runs the whole transformer on the growing past.

Time attention is the *only* operation in this architecture that mixes
across timesteps (space attention is within-frame; the MLP is per-token).
So during an incremental forward for a new token, past contributes only
through time attention's K and V. We cache exactly those tensors —
post-QKNorm, post-RoPE — at each time-attention layer, and nothing else.

Non-time-attention layers never touch the cache: the new token passes
through their space attention and MLP without looking up any past state.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch


class KVCache:
    """
    K/V cache indexed by time-attention layer.

    For each time-attention layer ``L`` the cache stores two tensors of shape
    ``(N, n_kv_heads, T_cached, head_dim)`` (with ``N = B * S`` — the
    flattened (batch, spatial) axis that time attention operates over).
    Tensors are stored **post-QKNorm and post-RoPE** so that the incremental
    forward only needs to project K, V for the new tokens and concatenate.

    Non-time-attention layer indices are simply absent from the cache — they
    have no cross-time state to remember.

    Args:
        time_layer_indices: Layer indices (into the transformer's ``layers``
            list) that perform time attention and therefore need a slot.
        max_T: Maximum number of past frames the cache is allowed to hold.
            The transformer is responsible for enforcing this bound — when a
            commit would exceed ``max_T``, it evicts the oldest cached frames
            (re-rotating the remaining K via :func:`dreamer4.transformer.rope.shift_rope`)
            so that ``t_cached <= max_T`` always holds post-commit.

    Attributes:
        k: dict[int, Tensor | None]
        v: dict[int, Tensor | None]
        max_T: int
    """

    def __init__(self, time_layer_indices: Iterable[int], *, max_T: int):
        self._time_layers: Tuple[int, ...] = tuple(sorted(set(time_layer_indices)))
        self.k: Dict[int, Optional[torch.Tensor]] = {L: None for L in self._time_layers}
        self.v: Dict[int, Optional[torch.Tensor]] = {L: None for L in self._time_layers}
        self.max_T: int = int(max_T)
        # Total timesteps EVER committed (monotone; unlike ``t_cached`` it does not
        # shrink when the sliding window evicts old frames). Callers that track
        # "which of my frames are already in the cache" must use this, not
        # ``t_cached`` — after the first eviction the two diverge.
        self.committed: int = 0

    @property
    def time_layers(self) -> Tuple[int, ...]:
        return self._time_layers

    @property
    def t_cached(self) -> int:
        """Number of past timesteps currently cached (0 if empty)."""
        for L in self._time_layers:
            k = self.k[L]
            if k is not None:
                return k.shape[-2]
        return 0

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.k

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return ``(K, V)`` for this layer or None if empty / non-time layer."""
        if layer_idx not in self.k:
            return None
        k = self.k[layer_idx]
        v = self.v[layer_idx]
        if k is None or v is None:
            return None
        return k, v

    def clear_kv(self) -> None:
        """Drop stored K/V only — ``committed`` keeps counting. Used internally
        when a single oversized commit replaces the whole window."""
        for L in self._time_layers:
            self.k[L] = None
            self.v[L] = None

    def reset(self) -> None:
        """Clear all cached K/V and the committed-timestep counter."""
        self.clear_kv()
        self.committed = 0

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """
        Append newly committed K, V for this layer along the time axis.

        Args:
            layer_idx: time-attention layer index.
            k_new, v_new: ``(N, n_kv_heads, T_new, head_dim)`` — already
                post-QKNorm and post-RoPE at absolute positions
                ``[t_cached, t_cached + T_new)``. Detached on insert so the
                cache never holds a live grad tape.
        """
        if layer_idx not in self.k:
            raise KeyError(
                f"layer {layer_idx} is not a time-attention layer; "
                f"cache only stores K/V at {self._time_layers}"
            )
        k_det = k_new.detach()
        v_det = v_new.detach()
        existing_k = self.k[layer_idx]
        if existing_k is None:
            self.k[layer_idx] = k_det
            self.v[layer_idx] = v_det
        else:
            self.k[layer_idx] = torch.cat([existing_k, k_det], dim=-2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_det], dim=-2)
