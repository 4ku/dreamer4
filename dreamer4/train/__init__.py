"""
Training entrypoints for Dreamer 4 — one runnable module per phase:

    python -m dreamer4.train.train_tokenizer  # 1a  causal video tokenizer
    python -m dreamer4.train.train_dynamics   # 1b  shortcut-forcing world model; its late
                                              #     bootstrap ramp makes K=1 sampling legal
    python -m dreamer4.train.train_heads      # 2   agent finetuning (agent tokens + heads)
    python -m dreamer4.train.train_pmpo       # 3   PMPO RL in imagination

Every phase starts from the checkpoint of the previous one: the tokenizer is
frozen after 1a; phase 2 fine-tunes the world model together with the agent
heads; phase 3 freezes it and updates only the policy and value heads.

Phases 1a-1b are configured by the dataclass tree in
:mod:`dreamer4.train.config` (defaults < ``--config`` YAML < CLI flags);
phases 2-3 take plain flags, since they configure nothing but themselves —
their architecture arrives inside the checkpoint they start from.

Shared plumbing: :mod:`dreamer4.train.common` (EMA, schedules, loss
normalization, logging), :mod:`dreamer4.train.dynamics_objectives`
(phase-1 losses) and :mod:`dreamer4.train.pmpo_objectives` (phase-3
TD(lambda) + PMPO coefficients).
"""
