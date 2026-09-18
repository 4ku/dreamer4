"""
Block-causal transformer backbone, shared by the Dreamer 4 tokenizer and dynamics model.

Re-exports the pieces a model is assembled from: the full stack (``BlockCausalTransformer``),
one layer with its two attention types, the norms, the attention primitive, the SwiGLU MLP and
the RoPE helpers. Token layouts and space-attention masks live in the ``modality`` submodule,
the autoregressive rollout cache in ``kv_cache``; both are imported from there directly.
"""

from dreamer4.models.transformer.transformer import BlockCausalTransformer
from dreamer4.models.transformer.layers import SpaceAttention, TimeAttention, BlockCausalLayer
from dreamer4.models.transformer.norms import RMSNorm, QKNorm
from dreamer4.models.transformer.attention import MultiheadAttention
from dreamer4.models.transformer.mlp import SwiGLU
from dreamer4.models.transformer.rope import build_rope_cache, apply_rope

__all__ = [
    "BlockCausalTransformer",
    "BlockCausalLayer",
    "SpaceAttention",
    "TimeAttention",
    "MultiheadAttention",
    "SwiGLU",
    "RMSNorm",
    "QKNorm",
    "build_rope_cache",
    "apply_rope",
]
