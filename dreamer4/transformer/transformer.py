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
            latents_only_time=True, ...
        )

    For the dynamics model:
        layout = TokenLayout(n_latents=0, segments=(
            (Modality.ACTION, 1), (Modality.SPATIAL, 128),
            (Modality.REGISTER, 4), (Modality.AGENT, 1), ...
        ))
        transformer = BlockCausalTransformer(
            d_model=512, n_heads=8, n_kv_heads=2, depth=16,
            layout=layout, space_mode="wm_agent_isolated", ...
        )
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dreamer4.modality import TokenLayout
from dreamer4.transformer.norms import RMSNorm
from dreamer4.transformer.layers import BlockCausalLayer


class BlockCausalTransformer(nn.Module):
    """
    Stack of BlockCausalLayers with a final RMSNorm.

    Args:
        d_model:          Model (hidden) dimension.
        n_heads:          Number of query attention heads.
        depth:            Number of transformer layers.
        layout:           TokenLayout describing the spatial dimension.
        space_mode:       Modality mask mode ("encoder", "decoder",
                          "wm_agent_isolated", "wm_agent").
        n_kv_heads:       Number of KV heads for GQA. Default = n_heads.
        mlp_ratio:        MLP hidden = d_model * mlp_ratio (default 4.0).
        time_every:       Apply time attention every N layers (default 4).
        dropout:          Dropout rate (default 0.0).
        use_qk_norm:      Use QKNorm in attention (default True).
        logit_cap:        Attention logit soft capping (default 50.0).
        latents_only_time: Apply time attention only to latent positions
                           (saves compute in tokenizer). Default False.
        max_T:            Max time steps for RoPE cache (default 1024).

    Shape:
        Input:  (B, T, S, D)
        Output: (B, T, S, D)

    Example:
        >>> from dreamer4.modality import Modality, TokenLayout
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
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        latents_only_time: bool = False,
        max_T: int = 1024,
    ):
        super().__init__()

        self.d_model = d_model
        self.depth = depth
        self.layout = layout
        n_latents = layout.n_latents

        self.layers = nn.ModuleList([
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
                latents_only_time=latents_only_time,
                n_latents=n_latents,
                max_T=max_T,
            )
            for i in range(depth)
        ])

        # Final normalization after all layers
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, S, D) input tensor.

        Returns:
            (B, T, S, D) output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_time_layers(self) -> int:
        """Number of layers that have time attention."""
        return sum(1 for layer in self.layers if layer.has_time)
