"""Tests for dreamer4.modality — TokenLayout, Modality, and attention masks."""

import torch
import pytest

from dreamer4.modality import Modality, TokenLayout, build_space_attn_mask


# ===== TokenLayout basics =====

def test_total_tokens():
    layout = TokenLayout(n_latents=4, segments=((Modality.IMAGE, 16),))
    assert layout.total_tokens() == 20


def test_total_tokens_multiple_segments():
    layout = TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SPATIAL, 8),
        (Modality.REGISTER, 2),
    ))
    assert layout.total_tokens() == 11


def test_modality_ids():
    layout = TokenLayout(n_latents=2, segments=((Modality.IMAGE, 3),))
    ids = layout.modality_ids()
    expected = torch.tensor([-1, -1, 0, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(ids, expected)


def test_modality_ids_wm_layout():
    layout = TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SHORTCUT_SIGNAL, 1),
        (Modality.SPATIAL, 3),
        (Modality.AGENT, 1),
    ))
    ids = layout.modality_ids()
    expected = torch.tensor([1, 5, 4, 4, 4, 7], dtype=torch.int32)
    torch.testing.assert_close(ids, expected)


def test_slices():
    layout = TokenLayout(n_latents=2, segments=(
        (Modality.IMAGE, 4),
        (Modality.REGISTER, 2),
    ))
    sl = layout.slices()
    assert sl[Modality.LATENT] == slice(0, 2)
    assert sl[Modality.IMAGE] == slice(2, 6)
    assert sl[Modality.REGISTER] == slice(6, 8)


# ===== Encoder mode mask =====

def test_encoder_latent_row_all_true():
    """In encoder mode, latent queries can attend to ALL tokens."""
    layout = TokenLayout(n_latents=4, segments=((Modality.IMAGE, 8),))
    mask = build_space_attn_mask(layout, mode="encoder")
    # First 4 rows (latent queries) should be all True
    assert mask[:4, :].all()


def test_encoder_image_row_only_same_modality():
    """In encoder mode, image queries can only attend to other image tokens."""
    layout = TokenLayout(n_latents=4, segments=((Modality.IMAGE, 8),))
    mask = build_space_attn_mask(layout, mode="encoder")
    # Image queries (rows 4-11) should NOT attend to latent keys (cols 0-3)
    assert not mask[4:, :4].any()
    # Image queries SHOULD attend to other image tokens
    assert mask[4:, 4:].all()


def test_encoder_multi_modality():
    """In encoder with multiple modalities, each only sees its own kind."""
    layout = TokenLayout(n_latents=2, segments=(
        (Modality.IMAGE, 3),
        (Modality.PROPRIO, 2),
    ))
    mask = build_space_attn_mask(layout, mode="encoder")
    S = layout.total_tokens()  # 7

    # Image tokens (pos 2,3,4) should not see proprio (pos 5,6) or latent (0,1)
    assert not mask[2, 5]
    assert not mask[2, 0]
    # Image sees image
    assert mask[2, 3]
    assert mask[3, 4]
    # Proprio sees proprio
    assert mask[5, 6]
    assert not mask[5, 2]  # proprio doesn't see image


# ===== Decoder mode mask =====

def test_decoder_latent_only_sees_latent():
    """In decoder mode, latent queries attend only to other latents."""
    layout = TokenLayout(n_latents=4, segments=((Modality.IMAGE, 8),))
    mask = build_space_attn_mask(layout, mode="decoder")
    # Latent rows (0-3): True for cols 0-3, False for cols 4-11
    assert mask[:4, :4].all()
    assert not mask[:4, 4:].any()


def test_decoder_image_sees_self_and_latent():
    """In decoder mode, image queries attend to images AND latents."""
    layout = TokenLayout(n_latents=4, segments=((Modality.IMAGE, 8),))
    mask = build_space_attn_mask(layout, mode="decoder")
    # Image rows (4-11): True for latent cols (0-3) and image cols (4-11)
    assert mask[4:, :4].all()   # can see latents
    assert mask[4:, 4:].all()   # can see images


# ===== WM agent isolated mode =====

def test_wm_isolated_agent_only_sees_agent():
    """In wm_agent_isolated mode, agent queries only see agent keys."""
    layout = TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SPATIAL, 4),
        (Modality.AGENT, 2),
    ))
    mask = build_space_attn_mask(layout, mode="wm_agent_isolated")
    agent_slice = layout.slices()[Modality.AGENT]  # slice(5, 7)

    # Agent rows: only True for agent columns
    for i in range(agent_slice.start, agent_slice.stop):
        for j in range(layout.total_tokens()):
            if agent_slice.start <= j < agent_slice.stop:
                assert mask[i, j], f"Agent({i}) should see Agent({j})"
            else:
                assert not mask[i, j], f"Agent({i}) should NOT see non-Agent({j})"


def test_wm_isolated_nonagent_cannot_see_agent():
    """In wm_agent_isolated mode, non-agent queries cannot see agent keys."""
    layout = TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SPATIAL, 4),
        (Modality.AGENT, 2),
    ))
    mask = build_space_attn_mask(layout, mode="wm_agent_isolated")
    agent_slice = layout.slices()[Modality.AGENT]

    # Non-agent rows: False for agent columns, True otherwise
    for i in range(agent_slice.start):
        for j in range(agent_slice.start, agent_slice.stop):
            assert not mask[i, j], f"Non-agent({i}) should NOT see Agent({j})"
        # Non-agent sees all non-agent
        for j in range(agent_slice.start):
            assert mask[i, j], f"Non-agent({i}) should see non-Agent({j})"


# ===== WM agent (full) mode =====

def test_wm_agent_sees_everything():
    """In wm_agent mode, agent queries can attend to ALL tokens."""
    layout = TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SPATIAL, 4),
        (Modality.AGENT, 2),
    ))
    mask = build_space_attn_mask(layout, mode="wm_agent")
    agent_slice = layout.slices()[Modality.AGENT]

    # Agent rows: all True
    assert mask[agent_slice.start:agent_slice.stop, :].all()


def test_wm_agent_nonagent_still_cannot_see_agent():
    """In wm_agent mode, non-agent queries still cannot see agent tokens."""
    layout = TokenLayout(n_latents=0, segments=(
        (Modality.ACTION, 1),
        (Modality.SPATIAL, 4),
        (Modality.AGENT, 2),
    ))
    mask = build_space_attn_mask(layout, mode="wm_agent")
    agent_slice = layout.slices()[Modality.AGENT]

    for i in range(agent_slice.start):
        for j in range(agent_slice.start, agent_slice.stop):
            assert not mask[i, j]


# ===== Edge cases =====

def test_no_latents():
    """Layout with no latents should work."""
    layout = TokenLayout(n_latents=0, segments=((Modality.SPATIAL, 4),))
    mask = build_space_attn_mask(layout, mode="encoder")
    assert mask.shape == (4, 4)
    assert mask.all()  # all same modality, all can see each other


def test_invalid_mode():
    """Invalid mode should raise ValueError."""
    layout = TokenLayout(n_latents=2, segments=((Modality.IMAGE, 4),))
    with pytest.raises(ValueError, match="Unknown mode"):
        build_space_attn_mask(layout, mode="nonexistent")


def test_mask_shape():
    """Mask should be (S, S)."""
    layout = TokenLayout(n_latents=3, segments=(
        (Modality.IMAGE, 5),
        (Modality.REGISTER, 2),
    ))
    mask = build_space_attn_mask(layout, mode="encoder")
    S = layout.total_tokens()
    assert mask.shape == (S, S)
    assert mask.dtype == torch.bool
