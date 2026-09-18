"""Model architecture for Dreamer 4 -- and nothing else.

- :mod:`dreamer4.models.transformer` -- the shared block-causal backbone
  (space/time attention, modality masks, RoPE, KV cache);
- :mod:`dreamer4.models.tokenizer` -- causal video tokenizer
  (+ :class:`FrozenTokenizer`, its deployment form);
- :mod:`dreamer4.models.dynamics` -- the world model and its samplers;
- :mod:`dreamer4.models.agent_heads` -- the paper's policy, reward and value
  heads on the agent tokens inside the dynamics transformer;
- :mod:`dreamer4.models.world_model` -- a trained dynamics checkpoint as one
  usable object, and the agent policy that acts through it;
- :mod:`dreamer4.models.distributions` -- symlog / two-hot primitives.

Training objectives, data loading and evaluation live OUTSIDE this
subpackage (``dreamer4.train``, ``dreamer4.data``, ``dreamer4.dynamics_eval``).
"""

# Only the world-model architecture is re-exported here: the agent heads are
# built from a checkpoint by their own module, and
# importing them eagerly would make every tokenizer/dynamics import pay for
# them -- and world_model reaches into dreamer4.train, which would cycle.
from dreamer4.models.dynamics import DynamicsModel
from dreamer4.models.tokenizer import Decoder, Encoder, FrozenTokenizer, Tokenizer

__all__ = ["Decoder", "DynamicsModel", "Encoder", "FrozenTokenizer",
           "Tokenizer"]
