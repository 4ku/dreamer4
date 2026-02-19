"""
Model size configurations for the Block-Causal Transformer.

Provides named presets so you can quickly instantiate transformers of
different sizes without remembering all the hyperparameters:

    from dreamer4.configs import make_transformer, print_config_table
    from dreamer4.modality import Modality, TokenLayout

    layout = TokenLayout(n_latents=4, segments=((Modality.IMAGE, 16),))
    model = make_transformer("small", layout, space_mode="encoder")

Three families of configs:

  Generic    (tiny / small / base / large)    — for experimentation
  Tokenizer  (tok-small / tok-base / tok-large) — encoder/decoder presets
  Dynamics   (dyn-small / dyn-base / dyn-large) — world model presets
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any

from dreamer4.modality import TokenLayout
from dreamer4.transformer import BlockCausalTransformer


@dataclass(frozen=True)
class TransformerConfig:
    """All hyperparameters needed to build a BlockCausalTransformer (minus layout/mode)."""

    d_model: int
    n_heads: int
    depth: int
    n_kv_heads: int | None = None
    mlp_ratio: float = 4.0
    time_every: int = 4
    dropout: float = 0.0
    use_qk_norm: bool = True
    logit_cap: float | None = 50.0
    latents_only_time: bool = False
    max_T: int = 1024


# ── Generic sizes ────────────────────────────────────────────────────────────

CONFIGS: dict[str, TransformerConfig] = {
    "tiny": TransformerConfig(
        d_model=64, n_heads=2, depth=2,
        time_every=2,
    ),
    "small": TransformerConfig(
        d_model=128, n_heads=4, depth=4,
        time_every=4,
    ),
    "base": TransformerConfig(
        d_model=256, n_heads=4, depth=8,
        time_every=4,
    ),
    "large": TransformerConfig(
        d_model=512, n_heads=8, depth=16,
        n_kv_heads=2, time_every=4,
    ),

    # ── Tokenizer sizes ──────────────────────────────────────────────────────

    "tok-small": TransformerConfig(
        d_model=128, n_heads=4, depth=4,
        time_every=1, latents_only_time=True,
    ),
    "tok-base": TransformerConfig(
        d_model=256, n_heads=4, depth=8,
        time_every=1, latents_only_time=True,
    ),
    "tok-large": TransformerConfig(
        d_model=512, n_heads=8, depth=12,
        time_every=1, latents_only_time=True,
    ),

    # ── Dynamics sizes ────────────────────────────────────────────────────────

    "dyn-small": TransformerConfig(
        d_model=256, n_heads=4, depth=4,
        time_every=4,
    ),
    "dyn-base": TransformerConfig(
        d_model=512, n_heads=4, depth=8,
        time_every=4,
    ),
    "dyn-large": TransformerConfig(
        d_model=1024, n_heads=8, depth=16,
        n_kv_heads=2, time_every=4,
    ),
}


def list_configs() -> dict[str, TransformerConfig]:
    """Return a copy of the full config registry."""
    return dict(CONFIGS)


def make_transformer(
    config_name: str,
    layout: TokenLayout,
    space_mode: str,
    **overrides: Any,
) -> BlockCausalTransformer:
    """
    Instantiate a BlockCausalTransformer from a named config.

    Args:
        config_name: Key in CONFIGS (e.g. "small", "tok-base", "dyn-large").
        layout:      TokenLayout for the spatial dimension.
        space_mode:  Attention mask mode ("encoder", "decoder", etc.).
        **overrides: Any TransformerConfig field to override, e.g.
                     ``make_transformer("small", layout, "encoder", dropout=0.1)``

    Returns:
        A freshly constructed BlockCausalTransformer.
    """
    if config_name not in CONFIGS:
        raise KeyError(
            f"Unknown config '{config_name}'. "
            f"Available: {sorted(CONFIGS.keys())}"
        )
    cfg = CONFIGS[config_name]
    if overrides:
        cfg = replace(cfg, **overrides)

    return BlockCausalTransformer(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        depth=cfg.depth,
        layout=layout,
        space_mode=space_mode,
        n_kv_heads=cfg.n_kv_heads,
        mlp_ratio=cfg.mlp_ratio,
        time_every=cfg.time_every,
        dropout=cfg.dropout,
        use_qk_norm=cfg.use_qk_norm,
        logit_cap=cfg.logit_cap,
        latents_only_time=cfg.latents_only_time,
        max_T=cfg.max_T,
    )


def print_config_table(
    layout: TokenLayout | None = None,
    space_mode: str = "encoder",
) -> None:
    """
    Print a formatted table of all configs with their hyperparameters
    and (optionally) parameter counts.

    If *layout* is provided, each config is instantiated so the exact
    parameter count can be shown.  Otherwise the "Params" column is
    omitted (instantiation requires a concrete layout).
    """
    header_fields = [
        "Name", "d_model", "n_heads", "n_kv", "depth",
        "time_every", "mlp_ratio", "qk_norm", "logit_cap",
    ]
    show_params = layout is not None
    if show_params:
        header_fields.append("Params")

    rows: list[list[str]] = []
    for name, cfg in CONFIGS.items():
        row = [
            name,
            str(cfg.d_model),
            str(cfg.n_heads),
            str(cfg.n_kv_heads or cfg.n_heads),
            str(cfg.depth),
            str(cfg.time_every),
            f"{cfg.mlp_ratio:.1f}",
            "Y" if cfg.use_qk_norm else "N",
            str(cfg.logit_cap) if cfg.logit_cap else "-",
        ]
        if show_params:
            model = make_transformer(name, layout, space_mode)
            count = model.count_parameters()
            if count >= 1_000_000:
                row.append(f"{count / 1_000_000:.1f}M")
            elif count >= 1_000:
                row.append(f"{count / 1_000:.1f}K")
            else:
                row.append(str(count))
        rows.append(row)

    col_widths = [
        max(len(h), *(len(r[i]) for r in rows))
        for i, h in enumerate(header_fields)
    ]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(header_fields))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))
