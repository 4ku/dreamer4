"""
Training entrypoints for Dreamer 4.

Each phase gets its own runnable module; the first one is the tokenizer:

    python -m dreamer4.train.train_tokenizer --data.path <dir> --out runs/tok

Shared plumbing lives in :mod:`dreamer4.train.config` (dataclass configs,
YAML + CLI) and :mod:`dreamer4.train.common` (EMA, schedules, loss
normalization, logging).
"""
