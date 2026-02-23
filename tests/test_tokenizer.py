"""Tests for dreamer4.tokenizer — Causal Tokenizer (Encoder, Decoder, MAE, losses)."""

import torch
import pytest

from dreamer4.modality import Modality, TokenLayout
from dreamer4.tokenizer import (
    MAEReplacer,
    Encoder,
    Decoder,
    Tokenizer,
    recon_loss_from_mae,
    RMSLossNormalizer,
)
from dreamer4.utils import (
    patchify,
    unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

B, T, C, H, W = 2, 4, 3, 32, 32
PATCH_SIZE = 8
N_PATCHES = (H // PATCH_SIZE) * (W // PATCH_SIZE)  # 16
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * C             # 192

D_MODEL = 64
N_HEADS = 2
DEPTH = 2
N_LATENTS = 4
D_BOTTLENECK = 16


def _make_encoder(**overrides):
    kw = dict(
        patch_dim=PATCH_DIM,
        d_model=D_MODEL,
        n_latents=N_LATENTS,
        n_patches=N_PATCHES,
        n_heads=N_HEADS,
        depth=DEPTH,
        d_bottleneck=D_BOTTLENECK,
        time_every=1,
        mae_p_min=0.0,
        mae_p_max=0.9,
        use_qk_norm=False,
        logit_cap=None,
    )
    kw.update(overrides)
    return Encoder(**kw)


def _make_decoder(**overrides):
    kw = dict(
        d_bottleneck=D_BOTTLENECK,
        d_model=D_MODEL,
        n_latents=N_LATENTS,
        n_patches=N_PATCHES,
        patch_dim=PATCH_DIM,
        n_heads=N_HEADS,
        depth=DEPTH,
        time_every=1,
        use_qk_norm=False,
        logit_cap=None,
    )
    kw.update(overrides)
    return Decoder(**kw)


def _make_tokenizer(**enc_overrides):
    enc = _make_encoder(**enc_overrides)
    dec = _make_decoder()
    return Tokenizer(enc, dec)


def _random_patches():
    return torch.rand(B, T, N_PATCHES, PATCH_DIM)


# =====================================================================
# MAEReplacer tests
# =====================================================================

class TestMAEReplacer:

    def test_output_shapes(self):
        mae = MAEReplacer(D_MODEL, p_min=0.0, p_max=0.9)
        mae.train()
        x = torch.randn(B, T, N_PATCHES, D_MODEL)
        replaced, mask, keep_prob = mae(x)

        assert replaced.shape == (B, T, N_PATCHES, D_MODEL)
        assert mask.shape == (B, T, N_PATCHES, 1)
        assert mask.dtype == torch.bool
        assert keep_prob.shape == (B, T, 1)

    def test_no_masking_at_eval(self):
        mae = MAEReplacer(D_MODEL, p_min=0.0, p_max=0.9)
        mae.eval()
        x = torch.randn(B, T, N_PATCHES, D_MODEL)
        replaced, mask, keep_prob = mae(x)

        assert mask.sum() == 0
        torch.testing.assert_close(replaced, x)
        assert (keep_prob == 1.0).all()

    def test_no_masking_when_p_zero(self):
        mae = MAEReplacer(D_MODEL, p_min=0.0, p_max=0.0)
        mae.train()
        x = torch.randn(B, T, N_PATCHES, D_MODEL)
        replaced, mask, keep_prob = mae(x)

        assert mask.sum() == 0
        torch.testing.assert_close(replaced, x)

    def test_masking_stats(self):
        """Mean masking fraction should be ~0.45 (midpoint of U(0, 0.9))."""
        torch.manual_seed(0)
        mae = MAEReplacer(D_MODEL, p_min=0.0, p_max=0.9)
        mae.train()

        mask_fracs = []
        for _ in range(200):
            x = torch.randn(8, 4, N_PATCHES, D_MODEL)
            _, mask, _ = mae(x)
            mask_fracs.append(mask.float().mean().item())

        mean_frac = sum(mask_fracs) / len(mask_fracs)
        assert 0.35 < mean_frac < 0.55, f"Expected ~0.45, got {mean_frac:.3f}"

    def test_masked_positions_use_mask_token(self):
        mae = MAEReplacer(D_MODEL, p_min=0.9, p_max=0.9)
        mae.train()
        x = torch.randn(1, 1, N_PATCHES, D_MODEL)
        replaced, mask, _ = mae(x)

        masked_vecs = replaced[mask.squeeze(-1)]
        if masked_vecs.numel() > 0:
            expected = mae.mask_token.detach().expand_as(masked_vecs)
            torch.testing.assert_close(masked_vecs, expected)


# =====================================================================
# Encoder tests
# =====================================================================

class TestEncoder:

    def test_output_shape(self):
        enc = _make_encoder()
        patches = _random_patches()
        z, (mae_mask, keep_prob) = enc(patches)

        assert z.shape == (B, T, N_LATENTS, D_BOTTLENECK)
        assert mae_mask.shape == (B, T, N_PATCHES, 1)
        assert keep_prob.shape == (B, T, 1)

    def test_bottleneck_bounds(self):
        """Encoder output must be in [-1, 1] due to tanh."""
        enc = _make_encoder()
        patches = _random_patches()
        z, _ = enc(patches)

        assert z.min() >= -1.0
        assert z.max() <= 1.0

    def test_output_finite(self):
        enc = _make_encoder()
        patches = _random_patches()
        z, _ = enc(patches)
        assert torch.isfinite(z).all()

    def test_eval_no_masking(self):
        enc = _make_encoder()
        enc.eval()
        patches = _random_patches()
        z, (mae_mask, keep_prob) = enc(patches)

        assert mae_mask.sum() == 0
        assert (keep_prob == 1.0).all()

    def test_gradient_flows(self):
        enc = _make_encoder()
        patches = _random_patches()
        patches.requires_grad_(True)
        z, _ = enc(patches)
        z.sum().backward()

        assert patches.grad is not None
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# =====================================================================
# Decoder tests
# =====================================================================

class TestDecoder:

    def test_output_shape(self):
        dec = _make_decoder()
        z = torch.randn(B, T, N_LATENTS, D_BOTTLENECK)
        out = dec(z)
        assert out.shape == (B, T, N_PATCHES, PATCH_DIM)

    def test_output_bounds(self):
        """Decoder output must be in [0, 1] due to sigmoid."""
        dec = _make_decoder()
        z = torch.randn(B, T, N_LATENTS, D_BOTTLENECK)
        out = dec(z)

        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_output_finite(self):
        dec = _make_decoder()
        z = torch.randn(B, T, N_LATENTS, D_BOTTLENECK)
        out = dec(z)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        dec = _make_decoder()
        z = torch.randn(B, T, N_LATENTS, D_BOTTLENECK, requires_grad=True)
        out = dec(z)
        out.sum().backward()

        assert z.grad is not None
        for name, p in dec.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# =====================================================================
# Tokenizer (Encoder + Decoder) tests
# =====================================================================

class TestTokenizer:

    def test_output_shape(self):
        tok = _make_tokenizer()
        patches = _random_patches()
        pred, mae_mask, keep_prob = tok(patches)

        assert pred.shape == patches.shape
        assert mae_mask.shape == (B, T, N_PATCHES, 1)
        assert keep_prob.shape == (B, T, 1)

    def test_pred_bounds(self):
        tok = _make_tokenizer()
        patches = _random_patches()
        pred, _, _ = tok(patches)

        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_encode_method(self):
        tok = _make_tokenizer()
        tok.eval()
        patches = _random_patches()
        z = tok.encode(patches)

        assert z.shape == (B, T, N_LATENTS, D_BOTTLENECK)
        assert z.min() >= -1.0
        assert z.max() <= 1.0

    def test_roundtrip_image_dimensions(self):
        """patchify -> tokenizer -> unpatchify gives correct image shape."""
        tok = _make_tokenizer()
        images = torch.rand(B, T, C, H, W)
        patches = patchify(images, PATCH_SIZE)
        pred, _, _ = tok(patches)
        recon = unpatchify(pred, H, W, C, PATCH_SIZE)

        assert recon.shape == (B, T, C, H, W)

    def test_gradient_flows_end_to_end(self):
        tok = _make_tokenizer()
        patches = _random_patches()
        patches.requires_grad_(True)
        pred, _, _ = tok(patches)
        pred.sum().backward()

        assert patches.grad is not None

    def test_smoke_overfit(self):
        """Tiny tokenizer should overfit a single batch."""
        torch.manual_seed(42)

        enc = Encoder(
            patch_dim=PATCH_DIM,
            d_model=64,
            n_latents=N_LATENTS,
            n_patches=N_PATCHES,
            n_heads=2,
            depth=2,
            d_bottleneck=D_BOTTLENECK,
            time_every=1,
            mae_p_min=0.0,
            mae_p_max=0.0,  # no masking for overfit test
            use_qk_norm=False,
            logit_cap=None,
        )
        dec = _make_decoder()
        tok = Tokenizer(enc, dec)
        tok.train()

        target = torch.rand(1, 2, N_PATCHES, PATCH_DIM)
        optimizer = torch.optim.Adam(tok.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(300):
            pred, _, _ = tok(target)
            loss = (pred - target).pow(2).mean()
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f"Overfit failed: {initial_loss:.4f} -> {final_loss:.4f}"
        )


# =====================================================================
# Loss function tests
# =====================================================================

class TestReconLoss:

    def test_zero_when_no_mask(self):
        """If nothing is masked, MSE loss should be 0."""
        pred = torch.rand(B, T, N_PATCHES, PATCH_DIM)
        target = torch.rand(B, T, N_PATCHES, PATCH_DIM)
        mask = torch.zeros(B, T, N_PATCHES, 1, dtype=torch.bool)

        loss = recon_loss_from_mae(pred, target, mask)
        assert loss.item() == 0.0

    def test_correct_when_all_masked(self):
        """If everything is masked, loss should equal full MSE."""
        pred = torch.rand(B, T, N_PATCHES, PATCH_DIM)
        target = torch.rand(B, T, N_PATCHES, PATCH_DIM)
        mask = torch.ones(B, T, N_PATCHES, 1, dtype=torch.bool)

        loss = recon_loss_from_mae(pred, target, mask)
        expected = (pred.float() - target.float()).pow(2).mean()
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        pred = torch.rand(B, T, N_PATCHES, PATCH_DIM, requires_grad=True)
        target = torch.rand(B, T, N_PATCHES, PATCH_DIM)
        mask = torch.ones(B, T, N_PATCHES, 1, dtype=torch.bool)

        loss = recon_loss_from_mae(pred, target, mask)
        loss.backward()
        assert pred.grad is not None

    def test_only_masked_contributes(self):
        """Only masked patches should affect the gradient."""
        pred = torch.rand(1, 1, 4, PATCH_DIM, requires_grad=True)
        target = torch.rand(1, 1, 4, PATCH_DIM)
        mask = torch.zeros(1, 1, 4, 1, dtype=torch.bool)
        mask[0, 0, 0, 0] = True  # only patch 0 masked

        loss = recon_loss_from_mae(pred, target, mask)
        loss.backward()

        # Gradient should be nonzero only for patch 0
        assert pred.grad[0, 0, 0].abs().sum() > 0
        assert pred.grad[0, 0, 1].abs().sum() == 0
        assert pred.grad[0, 0, 2].abs().sum() == 0
        assert pred.grad[0, 0, 3].abs().sum() == 0

    def test_finite_output(self):
        pred = torch.rand(B, T, N_PATCHES, PATCH_DIM)
        target = torch.rand(B, T, N_PATCHES, PATCH_DIM)
        mask = torch.rand(B, T, N_PATCHES, 1) > 0.5

        loss = recon_loss_from_mae(pred, target, mask)
        assert torch.isfinite(loss)


# =====================================================================
# RMS Loss Normalizer tests
# =====================================================================

class TestRMSLossNormalizer:

    def test_normalize_reduces_scale(self):
        norm = RMSLossNormalizer(n_losses=2, decay=0.99)
        big_loss = torch.tensor(100.0)
        norm.update(0, big_loss)
        normed = norm.normalize(0, big_loss)
        assert normed.item() < big_loss.item()

    def test_two_losses_independent(self):
        norm = RMSLossNormalizer(n_losses=2, decay=0.99)
        norm.update(0, torch.tensor(10.0))
        norm.update(1, torch.tensor(0.01))

        n0 = norm.normalize(0, torch.tensor(10.0))
        n1 = norm.normalize(1, torch.tensor(0.01))

        # Both should be approximately 1.0 (loss / rms ≈ 1 on first update)
        assert 0.5 < n0.item() < 2.0
        assert 0.5 < n1.item() < 2.0

    def test_ema_tracks_changes(self):
        norm = RMSLossNormalizer(n_losses=1, decay=0.9)
        for _ in range(50):
            norm.update(0, torch.tensor(1.0))
        rms_before = norm.rms_ema[0].item()

        for _ in range(50):
            norm.update(0, torch.tensor(10.0))
        rms_after = norm.rms_ema[0].item()

        assert rms_after > rms_before


# =====================================================================
# Bottleneck packing tests
# =====================================================================

class TestBottleneckPacking:

    def test_pack_shape(self):
        z = torch.randn(B, T, 8, 16)  # N_b=8, D_b=16
        packed = pack_bottleneck_to_spatial(z, n_spatial=4, k=2)
        assert packed.shape == (B, T, 4, 32)  # 4 tokens, dim 32

    def test_unpack_shape(self):
        z = torch.randn(B, T, 4, 32)
        unpacked = unpack_spatial_to_bottleneck(z, k=2)
        assert unpacked.shape == (B, T, 8, 16)

    def test_roundtrip(self):
        """pack -> unpack should be exact inverse."""
        z = torch.randn(B, T, 8, 16)
        packed = pack_bottleneck_to_spatial(z, n_spatial=4, k=2)
        recovered = unpack_spatial_to_bottleneck(packed, k=2)
        torch.testing.assert_close(z, recovered)

    def test_roundtrip_reverse(self):
        """unpack -> pack should be exact inverse."""
        z = torch.randn(B, T, 4, 32)
        unpacked = unpack_spatial_to_bottleneck(z, k=2)
        recovered = pack_bottleneck_to_spatial(unpacked, n_spatial=4, k=2)
        torch.testing.assert_close(z, recovered)

    def test_minecraft_dimensions(self):
        """Paper Appendix A: (512, 16) -> (256, 32)."""
        z = torch.randn(1, 1, 512, 16)
        packed = pack_bottleneck_to_spatial(z, n_spatial=256, k=2)
        assert packed.shape == (1, 1, 256, 32)

    def test_bad_n_spatial_raises(self):
        z = torch.randn(1, 1, 8, 16)
        with pytest.raises(AssertionError):
            pack_bottleneck_to_spatial(z, n_spatial=3, k=2)  # 8 != 3*2

    def test_bad_k_raises(self):
        z = torch.randn(1, 1, 4, 31)  # 31 not divisible by 3
        with pytest.raises(AssertionError):
            unpack_spatial_to_bottleneck(z, k=3)


# =====================================================================
# Encoder attention pattern test
# =====================================================================

class TestEncoderAttentionPattern:

    def test_encoder_mask_latents_see_all(self):
        """In encoder mode, latent tokens should attend to all tokens."""
        from dreamer4.modality import build_space_attn_mask

        layout = TokenLayout(
            n_latents=N_LATENTS,
            segments=((Modality.IMAGE, N_PATCHES),),
        )
        mask = build_space_attn_mask(layout, "encoder")
        S = layout.total_tokens()

        # Latent rows (0..N_LATENTS-1) should be all True
        assert mask[:N_LATENTS, :].all()

        # Patch rows should only see other patches (not latents)
        for i in range(N_LATENTS, S):
            # Can see other patches
            assert mask[i, N_LATENTS:].all()
            # Cannot see latents
            assert not mask[i, :N_LATENTS].any()


class TestDecoderAttentionPattern:

    def test_decoder_mask_patches_see_latents(self):
        """In decoder mode, patch queries should see latents + other patches."""
        from dreamer4.modality import build_space_attn_mask

        layout = TokenLayout(
            n_latents=N_LATENTS,
            segments=((Modality.IMAGE, N_PATCHES),),
        )
        mask = build_space_attn_mask(layout, "decoder")
        S = layout.total_tokens()

        # Latent rows should only see other latents
        for i in range(N_LATENTS):
            assert mask[i, :N_LATENTS].all()
            assert not mask[i, N_LATENTS:].any()

        # Patch rows should see latents AND other patches
        for i in range(N_LATENTS, S):
            assert mask[i, :N_LATENTS].all()   # see latents
            assert mask[i, N_LATENTS:].all()   # see patches


# =====================================================================
# Causal property through tokenizer
# =====================================================================

class TestTokenizerCausalProperty:

    def test_future_does_not_affect_past(self):
        """Changing a future frame should not affect past frame outputs."""
        enc = Encoder(
            patch_dim=PATCH_DIM,
            d_model=64,
            n_latents=N_LATENTS,
            n_patches=N_PATCHES,
            n_heads=2,
            depth=2,
            d_bottleneck=D_BOTTLENECK,
            time_every=1,
            mae_p_min=0.0,
            mae_p_max=0.0,  # no masking
            use_qk_norm=False,
            logit_cap=None,
        )
        dec = _make_decoder()
        tok = Tokenizer(enc, dec)
        tok.eval()

        patches = torch.randn(1, 6, N_PATCHES, PATCH_DIM)
        pred1, _, _ = tok(patches)

        patches2 = patches.clone()
        patches2[0, 5] = torch.randn(N_PATCHES, PATCH_DIM)
        pred2, _, _ = tok(patches2)

        torch.testing.assert_close(
            pred1[:, :5], pred2[:, :5], atol=1e-5, rtol=1e-5
        )
