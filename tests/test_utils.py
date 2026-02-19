"""Tests for dreamer4.utils — patchify / unpatchify."""

import torch
import pytest

from dreamer4.utils import patchify, unpatchify


# ---------------------------------------------------------------------------
# Roundtrip: patchify -> unpatchify should recover the original exactly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("patch_size", [4, 8, 16])
@pytest.mark.parametrize("H,W", [(64, 64), (128, 128), (224, 224)])
def test_roundtrip(patch_size, H, W):
    """Patchify followed by unpatchify recovers the original image."""
    B, T, C = 2, 3, 3
    images = torch.randn(B, T, C, H, W)
    patches = patchify(images, patch_size)
    recovered = unpatchify(patches, H=H, W=W, C=C, patch_size=patch_size)
    torch.testing.assert_close(recovered, images)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_patchify_shape():
    """Check output dimensions of patchify."""
    B, T, C, H, W = 2, 4, 3, 64, 64
    patch_size = 8
    patches = patchify(torch.randn(B, T, C, H, W), patch_size)
    n_patches = (H // patch_size) * (W // patch_size)  # 64
    patch_dim = patch_size * patch_size * C              # 192
    assert patches.shape == (B, T, n_patches, patch_dim)


def test_unpatchify_shape():
    """Check output dimensions of unpatchify."""
    B, T, C, H, W = 2, 4, 3, 64, 64
    patch_size = 8
    n_patches = (H // patch_size) * (W // patch_size)
    patch_dim = patch_size * patch_size * C
    patches = torch.randn(B, T, n_patches, patch_dim)
    images = unpatchify(patches, H=H, W=W, C=C, patch_size=patch_size)
    assert images.shape == (B, T, C, H, W)


# ---------------------------------------------------------------------------
# Edge cases and input validation
# ---------------------------------------------------------------------------

def test_single_patch_per_image():
    """When patch_size == image size, we get exactly 1 patch per image."""
    B, T, C, H, W = 1, 1, 3, 16, 16
    images = torch.randn(B, T, C, H, W)
    patches = patchify(images, patch_size=16)
    assert patches.shape == (1, 1, 1, 16 * 16 * 3)
    recovered = unpatchify(patches, H=H, W=W, C=C, patch_size=16)
    torch.testing.assert_close(recovered, images)


def test_non_square_image():
    """Patchify works on non-square images too."""
    B, T, C, H, W = 2, 2, 3, 64, 128
    images = torch.randn(B, T, C, H, W)
    patches = patchify(images, patch_size=16)
    n_patches = (64 // 16) * (128 // 16)  # 4 * 8 = 32
    assert patches.shape[2] == n_patches
    recovered = unpatchify(patches, H=H, W=W, C=C, patch_size=16)
    torch.testing.assert_close(recovered, images)


def test_patchify_rejects_non_divisible():
    """Patchify should fail if H or W is not divisible by patch_size."""
    with pytest.raises(AssertionError):
        patchify(torch.randn(1, 1, 3, 65, 64), patch_size=8)


def test_patchify_rejects_wrong_dims():
    """Patchify should fail on non-5D input."""
    with pytest.raises(AssertionError):
        patchify(torch.randn(1, 3, 64, 64), patch_size=8)


# ---------------------------------------------------------------------------
# Content correctness
# ---------------------------------------------------------------------------

def test_patch_content_matches_original():
    """
    Verify that the first patch contains exactly the top-left corner
    of the image (first patch_size x patch_size pixels).
    """
    B, T, C, H, W = 1, 1, 1, 8, 8
    patch_size = 4
    images = torch.arange(64, dtype=torch.float32).reshape(B, T, C, H, W)
    patches = patchify(images, patch_size)

    # First patch should be the top-left 4x4 block
    top_left = images[0, 0, 0, :4, :4].reshape(-1)
    torch.testing.assert_close(patches[0, 0, 0], top_left)
