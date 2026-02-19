"""Tests for dreamer4.configs — model size presets and factory functions."""

import pytest
import torch

from dreamer4.modality import Modality, TokenLayout
from dreamer4.configs import (
    TransformerConfig,
    CONFIGS,
    list_configs,
    make_transformer,
    print_config_table,
)
from dreamer4.transformer import BlockCausalTransformer


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encoder_layout(n_latents: int = 4, n_patches: int = 16) -> TokenLayout:
    return TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))


def _wm_layout() -> TokenLayout:
    return TokenLayout(
        n_latents=0,
        segments=(
            (Modality.ACTION, 1),
            (Modality.SPATIAL, 8),
            (Modality.REGISTER, 2),
            (Modality.AGENT, 1),
        ),
    )


# ── Registry completeness ───────────────────────────────────────────────────

EXPECTED_NAMES = [
    "tiny", "small", "base", "large",
    "tok-small", "tok-base", "tok-large",
    "dyn-small", "dyn-base", "dyn-large",
]


def test_all_expected_configs_present():
    for name in EXPECTED_NAMES:
        assert name in CONFIGS, f"Missing config: {name}"


def test_list_configs_matches_registry():
    listed = list_configs()
    assert set(listed.keys()) == set(CONFIGS.keys())
    for name, cfg in listed.items():
        assert cfg == CONFIGS[name]


# ── Instantiation ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_instantiate_encoder_mode(name):
    """Every config should produce a valid transformer in encoder mode."""
    layout = _encoder_layout()
    model = make_transformer(name, layout, "encoder")
    assert isinstance(model, BlockCausalTransformer)
    S = layout.total_tokens()
    x = torch.randn(1, 2, S, CONFIGS[name].d_model)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ["tiny", "small", "dyn-small"])
def test_instantiate_wm_mode(name):
    """Configs without latents_only_time should work in wm_agent_isolated mode."""
    layout = _wm_layout()
    model = make_transformer(name, layout, "wm_agent_isolated")
    S = layout.total_tokens()
    x = torch.randn(1, 2, S, CONFIGS[name].d_model)
    y = model(x)
    assert y.shape == x.shape


# ── Overrides ────────────────────────────────────────────────────────────────

def test_override_dropout():
    layout = _encoder_layout()
    model = make_transformer("tiny", layout, "encoder", dropout=0.5)
    first_layer = model.layers[0]
    assert first_layer.mlp.drop.p == 0.5


def test_override_depth():
    layout = _encoder_layout()
    model = make_transformer("tiny", layout, "encoder", depth=6)
    assert len(model.layers) == 6


def test_invalid_config_name():
    layout = _encoder_layout()
    with pytest.raises(KeyError, match="Unknown config"):
        make_transformer("nonexistent", layout, "encoder")


# ── Parameter count ordering ─────────────────────────────────────────────────

def _param_count(name: str, layout: TokenLayout, mode: str = "encoder") -> int:
    model = make_transformer(name, layout, mode)
    return model.count_parameters()


def test_generic_sizes_ordered():
    """tiny < small < base < large in parameter count."""
    layout = _encoder_layout()
    counts = [_param_count(n, layout) for n in ["tiny", "small", "base", "large"]]
    for i in range(len(counts) - 1):
        assert counts[i] < counts[i + 1], (
            f"Expected ascending params: {list(zip(['tiny','small','base','large'], counts))}"
        )


def test_tokenizer_sizes_ordered():
    layout = _encoder_layout()
    counts = [_param_count(n, layout) for n in ["tok-small", "tok-base", "tok-large"]]
    for i in range(len(counts) - 1):
        assert counts[i] < counts[i + 1]


def test_dynamics_sizes_ordered():
    layout = _wm_layout()
    counts = [
        _param_count(n, layout, "wm_agent_isolated")
        for n in ["dyn-small", "dyn-base", "dyn-large"]
    ]
    for i in range(len(counts) - 1):
        assert counts[i] < counts[i + 1]


# ── Config table printing ───────────────────────────────────────────────────

def test_print_config_table_no_layout(capsys):
    """Table should print without errors when no layout is given."""
    print_config_table()
    captured = capsys.readouterr()
    assert "tiny" in captured.out
    assert "large" in captured.out
    assert "Params" not in captured.out


def test_print_config_table_with_layout(capsys):
    """Table should include a Params column when layout is given."""
    layout = _encoder_layout()
    print_config_table(layout=layout, space_mode="encoder")
    captured = capsys.readouterr()
    assert "Params" in captured.out
    assert "tiny" in captured.out


# ── TransformerConfig dataclass ──────────────────────────────────────────────

def test_config_is_frozen():
    cfg = TransformerConfig(d_model=64, n_heads=2, depth=2)
    with pytest.raises(AttributeError):
        cfg.d_model = 128  # type: ignore[misc]


def test_config_defaults():
    cfg = TransformerConfig(d_model=64, n_heads=2, depth=2)
    assert cfg.n_kv_heads is None
    assert cfg.mlp_ratio == 4.0
    assert cfg.time_every == 4
    assert cfg.dropout == 0.0
    assert cfg.use_qk_norm is True
    assert cfg.logit_cap == 50.0
    assert cfg.latents_only_time is False
    assert cfg.max_T == 1024
