"""
Model architecture for Dreamer 4 — and nothing else.

- :mod:`dreamer4.models.transformer` — the shared block-causal backbone
  (space/time attention, modality masks, RoPE, KV cache);
- :mod:`dreamer4.models.tokenizer` — causal video tokenizer
  (+ :class:`FrozenTokenizer`, its deployment form);
- :mod:`dreamer4.models.dynamics` — the world model and its samplers;
- :mod:`dreamer4.models.agent` — policy/reward/value heads (phase 2/3);
- :mod:`dreamer4.models.distributions` — symlog / two-hot primitives.

Training objectives, data loading and evaluation live OUTSIDE this
subpackage (``dreamer4.train``, ``dreamer4.data``, ``dreamer4.dynamics_eval``).
"""

from dreamer4.models.agent import (POLICY_SLOT, REWARD_SLOT, VALUE_SLOT,
                                   PolicyHead, RewardHead, TaskEncoder,
                                   ValueHead)
from dreamer4.models.dynamics import DynamicsModel
from dreamer4.models.tokenizer import Decoder, Encoder, FrozenTokenizer, Tokenizer

__all__ = ["POLICY_SLOT", "REWARD_SLOT", "VALUE_SLOT", "Decoder",
           "DynamicsModel", "Encoder", "FrozenTokenizer", "PolicyHead",
           "RewardHead", "TaskEncoder", "Tokenizer", "ValueHead"]
