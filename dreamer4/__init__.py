# Dreamer 4 — PyTorch Implementation
# Efficient Block-Causal Transformer and World Model Agent

from dreamer4.transformer import BlockCausalTransformer
from dreamer4.modality import Modality, TokenLayout
from dreamer4.norms import RMSNorm, QKNorm
from dreamer4.attention import MultiheadAttention
from dreamer4.mlp import SwiGLU
from dreamer4.rope import build_rope_cache, apply_rope, build_rope_2d
from dreamer4.utils import patchify, unpatchify
from dreamer4.layers import SpaceAttention, TimeAttention, BlockCausalLayer
from dreamer4.configs import TransformerConfig, CONFIGS, make_transformer, list_configs

__all__ = [
    "BlockCausalTransformer",
    "BlockCausalLayer",
    "SpaceAttention",
    "TimeAttention",
    "MultiheadAttention",
    "SwiGLU",
    "RMSNorm",
    "QKNorm",
    "Modality",
    "TokenLayout",
    "build_rope_cache",
    "apply_rope",
    "build_rope_2d",
    "patchify",
    "unpatchify",
    "TransformerConfig",
    "CONFIGS",
    "make_transformer",
    "list_configs",
]
