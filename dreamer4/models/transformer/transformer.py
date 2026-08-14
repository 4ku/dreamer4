"""
Block-Causal Transformer for Dreamer 4.

This is the top-level module that stacks BlockCausalLayers into a full
transformer. It is the shared backbone used by both the tokenizer
(encoder + decoder) and the dynamics model.

ARCHITECTURE SUMMARY:

    Input: (B, T, S, D)
      |
      v
    [BlockCausalLayer 0]  space_attn + mlp
    [BlockCausalLayer 1]  space_attn + mlp
    [BlockCausalLayer 2]  space_attn + mlp
    [BlockCausalLayer 3]  space_attn + TIME_attn + mlp   (time_every=4)
    [BlockCausalLayer 4]  space_attn + mlp
    ...
    [BlockCausalLayer N-1]
      |
      v
    RMSNorm (final)
      |
      v
    Output: (B, T, S, D)

Each layer has:
  - Space attention with modality mask (within each time step)
  - Time attention with causal mask (every Nth layer, across time steps)
  - SwiGLU MLP

Key features:
  - RoPE for positional encoding (separate spatial and temporal)
  - GQA for efficient KV cache
  - QKNorm + logit soft capping for training stability
  - Pre-norm residual connections

USAGE:

    For the tokenizer encoder:
        layout = TokenLayout(n_latents=16, segments=((Modality.IMAGE, 196),))
        transformer = BlockCausalTransformer(
            d_model=256, n_heads=4, depth=8,
            layout=layout, space_mode="encoder",
            ...
        )

    For the dynamics model:
        layout = TokenLayout(n_latents=0, segments=(
            (Modality.ACTION, 1), (Modality.SPATIAL, 128),
            (Modality.REGISTER, 4), (Modality.AGENT, 1), ...
        ))
        transformer = BlockCausalTransformer(
            d_model=512, n_heads=8, n_kv_heads=2, depth=16,
            layout=layout, space_mode="wm_agent", ...
        )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer4.models.transformer.modality import TokenLayout
from dreamer4.models.transformer.norms import RMSNorm
from dreamer4.models.transformer.layers import BlockCausalLayer
from dreamer4.models.transformer.kv_cache import KVCache
from dreamer4.models.transformer.rope import shift_rope


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Split video frames into non-overlapping patches.

    Takes a batch of video frames and cuts each frame into a grid of
    square patches, then flattens each patch into a vector.

    Args:
        images: (B, T, C, H, W) float tensor — batch of video frames.
                B = batch size, T = number of frames,
                C = channels (e.g., 3 for RGB),
                H, W = height and width (must be divisible by patch_size).
        patch_size: Side length of each square patch (e.g., 4, 8, or 16).

    Returns:
        (B, T, N_patches, patch_dim) float tensor where:
            N_patches = (H / patch_size) * (W / patch_size)
            patch_dim = patch_size * patch_size * C

    Example:
        >>> images = torch.randn(2, 4, 3, 64, 64)   # 2 videos, 4 frames, 64x64 RGB
        >>> patches = patchify(images, patch_size=8)
        >>> patches.shape
        torch.Size([2, 4, 64, 192])   # 64 patches of dim 8*8*3=192
    """
    assert images.dim() == 5, f"Expected 5D tensor (B,T,C,H,W), got {images.dim()}D"
    B, T, C, H, W = images.shape
    assert H % patch_size == 0, f"Height {H} not divisible by patch_size {patch_size}"
    assert W % patch_size == 0, f"Width {W} not divisible by patch_size {patch_size}"

    # Merge batch and time for F.unfold, which expects 4D input
    x = images.reshape(B * T, C, H, W)

    # F.unfold extracts sliding local blocks (patches) from the image.
    # With stride == kernel_size, we get non-overlapping patches.
    # Output shape: (B*T, C * patch_size * patch_size, N_patches)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)

    # Transpose so patches dimension comes before channels:
    # (B*T, N_patches, patch_dim)
    patches = patches.transpose(1, 2).contiguous()

    N_patches = patches.shape[1]
    patch_dim = patches.shape[2]

    # Restore batch and time dimensions
    return patches.reshape(B, T, N_patches, patch_dim)


def unpatchify(
    patches: torch.Tensor,
    H: int,
    W: int,
    C: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Reassemble patches back into images (inverse of patchify).

    Args:
        patches: (B, T, N_patches, patch_dim) float tensor.
        H: Original image height.
        W: Original image width.
        C: Number of channels (e.g., 3 for RGB).
        patch_size: Side length of each square patch.

    Returns:
        (B, T, C, H, W) float tensor — reconstructed video frames.

    Example:
        >>> patches = torch.randn(2, 4, 64, 192)     # 64 patches of dim 192
        >>> images = unpatchify(patches, H=64, W=64, C=3, patch_size=8)
        >>> images.shape
        torch.Size([2, 4, 3, 64, 64])
    """
    assert patches.dim() == 4, f"Expected 4D tensor (B,T,N,D), got {patches.dim()}D"
    B, T, N_patches, patch_dim = patches.shape
    assert (
        patch_dim == C * patch_size * patch_size
    ), f"patch_dim {patch_dim} != C*p*p = {C * patch_size * patch_size}"

    # Merge batch and time, transpose for F.fold:
    # (B*T, N_patches, patch_dim) -> (B*T, patch_dim, N_patches)
    x = patches.reshape(B * T, N_patches, patch_dim).transpose(1, 2).contiguous()

    # F.fold is the inverse of F.unfold — it reassembles patches into an image.
    # Output shape: (B*T, C, H, W)
    images = F.fold(x, output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    return images.reshape(B, T, C, H, W)


class BlockCausalTransformer(nn.Module):
    """
    Stack of BlockCausalLayers with a final RMSNorm.

    Args:
        d_model:          Model (hidden) dimension.
        n_heads:          Number of query attention heads.
        depth:            Number of transformer layers.
        layout:           TokenLayout describing the spatial dimension.
        space_mode:       Modality mask mode ("encoder", "decoder",
                          "decoder_cross", "wm_agent").
        n_kv_heads:       Number of KV heads for GQA. Default = n_heads.
        mlp_ratio:        MLP hidden = d_model * mlp_ratio (default 8/3).
        time_every:       Apply time attention every N layers (default 4).
        dropout:          Dropout rate (default 0.0).
        use_qk_norm:      Use QKNorm in attention (default True).
        logit_cap:        Attention logit soft capping (default 50.0).
        max_T:            Max time steps for RoPE cache (default 1024).

    Shape:
        Input:  (B, T, S, D) where
                B = batch size,
                T = number of time steps,
                S = number of spatial tokens per time step,
                D = model dimension (d_model).
        Output: (B, T, S, D) same shape as input.

    Example:
        >>> from dreamer4.models.transformer.modality import Modality, TokenLayout
        >>> layout = TokenLayout(n_latents=4, segments=((Modality.IMAGE, 16),))
        >>> model = BlockCausalTransformer(
        ...     d_model=64, n_heads=4, depth=4,
        ...     layout=layout, space_mode="encoder",
        ... )
        >>> x = torch.randn(2, 8, 20, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 8, 20, 64])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        layout: TokenLayout,
        space_mode: str,
        n_kv_heads: int | None = None,
        mlp_ratio: float = 8 / 3,
        time_every: int = 4,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        max_T: int = 256,
    ):
        super().__init__()

        self.d_model = d_model
        self.depth = depth
        self.layout = layout
        self.max_T = max_T

        self.layers = nn.ModuleList(
            [
                BlockCausalLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    layout=layout,
                    space_mode=space_mode,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    layer_index=i,
                    time_every=time_every,
                    use_qk_norm=use_qk_norm,
                    logit_cap=logit_cap,
                    max_T=max_T,
                )
                for i in range(depth)
            ]
        )

        # Final normalization after all layers
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: KVCache | None = None,
        commit: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, S, D)`` input tensor.
            cache: optional ``KVCache`` for incremental decoding.
                   - ``cache is None``: plain full forward (training / non-cached).
                   - ``cache`` non-None: ``x`` contains the timesteps to
                     process (the full past on the first call, or only the new
                     tokens on subsequent calls). Each time-attention layer
                     reads past K/V from the cache; non-time-attention layers
                     never touch the cache. With ``commit=True`` the newly
                     projected K/V for the timesteps in ``x`` are appended to
                     the cache along the time axis — oldest frames are
                     evicted (and cached K is re-rotated by :func:`shift_rope`)
                     to keep ``t_cached`` bounded by ``cache.max_T``.
            commit: see ``cache``. Ignored when ``cache is None``.

        Returns:
            ``(B, T_processed, S, D)`` output for the timesteps in ``x``.
        """
        if cache is None:
            for layer in self.layers:
                x = layer(x)
            return self.final_norm(x)

        if commit:
            T_new = x.shape[1]
            cache.committed += T_new   # caller-frame counter; monotone across eviction
            if T_new > cache.max_T:
                # Single commit bigger than the whole window — keep the tail,
                # drop anything currently cached (it'd be overwritten anyway).
                x = x[:, -cache.max_T:]
                T_new = cache.max_T
                cache.clear_kv()

            excess = cache.t_cached + T_new - cache.max_T
            if excess > 0:
                self._slide_window(cache, excess)

        for L, layer in enumerate(self.layers):
            if layer.has_time:
                past_kv = cache.get(L)
                x, kv_new = layer.forward_incremental(x, past_kv=past_kv)
                if commit and kv_new is not None:
                    cache.append(L, *kv_new)
            else:
                # Non-time layer: no cross-time state, no cache interaction.
                x, _ = layer.forward_incremental(x, past_kv=None)
        return self.final_norm(x)

    def _slide_window(self, cache: KVCache, excess: int) -> None:
        """
        Evict ``excess`` oldest cached entries from every time-attn layer and
        rotate the remaining K tensors backward by that many RoPE positions so
        their effective positions stay contiguous at ``[0, T_cached - excess)``.
        """
        for L in cache.time_layers:
            k = cache.k[L]
            v = cache.v[L]
            if k is None:
                continue
            k_kept = k[..., excess:, :]
            v_kept = v[..., excess:, :]
            rope_cos = self.layers[L].time_attn.rope_cos
            rope_sin = self.layers[L].time_attn.rope_sin
            cache.k[L] = shift_rope(k_kept, rope_cos, rope_sin, excess)
            cache.v[L] = v_kept

    def make_kv_cache(self) -> KVCache:
        """
        Build an empty ``KVCache`` sized to this transformer's time-attn
        layers. Cache capacity (``max_T``) is one less than the transformer's
        RoPE cache size — the extra slot is reserved for the new-query token's
        RoPE position during denoising, so the cache can safely be read with
        one additional token without overflowing.
        """
        return KVCache(
            time_layer_indices=[L for L, layer in enumerate(self.layers) if layer.has_time],
            max_T=self.max_T - 1,
        )

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_time_layers(self) -> int:
        """Number of layers that have time attention."""
        return sum(1 for layer in self.layers if layer.has_time)
