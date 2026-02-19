"""
Utility functions for Dreamer 4.

This module provides low-level helpers used across the codebase,
starting with image patchification (splitting images into non-overlapping
patches and reassembling them).

Patchification is the first step in the Dreamer 4 tokenizer:
  raw image (C, H, W) -> grid of patches (N_patches, patch_dim)

Each patch is a small square region of the image (e.g., 16x16 pixels)
flattened into a vector. For a 224x224 image with 16x16 patches, we get
(224/16) * (224/16) = 196 patches, each of dimension 16*16*3 = 768.
"""

import torch
import torch.nn.functional as F


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Split video frames into non-overlapping patches.

    Takes a batch of video frames and cuts each frame into a grid of
    square patches, then flattens each patch into a vector.

    Args:
        images: (B, T, C, H, W) float tensor — batch of video frames.
                B = batch size, T = number of frames,
                C = channels (e.g., 3 for RGB),
                H, W = height and width (must be divisible by patch_size).
        patch_size: Side length of each square patch (e.g., 4, 8, or 16).

    Returns:
        (B, T, N_patches, patch_dim) float tensor where:
            N_patches = (H / patch_size) * (W / patch_size)
            patch_dim = patch_size * patch_size * C

    Example:
        >>> images = torch.randn(2, 4, 3, 64, 64)   # 2 videos, 4 frames, 64x64 RGB
        >>> patches = patchify(images, patch_size=8)
        >>> patches.shape
        torch.Size([2, 4, 64, 192])   # 64 patches of dim 8*8*3=192
    """
    assert images.dim() == 5, f"Expected 5D tensor (B,T,C,H,W), got {images.dim()}D"
    B, T, C, H, W = images.shape
    assert H % patch_size == 0, f"Height {H} not divisible by patch_size {patch_size}"
    assert W % patch_size == 0, f"Width {W} not divisible by patch_size {patch_size}"

    # Merge batch and time for F.unfold, which expects 4D input
    x = images.reshape(B * T, C, H, W)

    # F.unfold extracts sliding local blocks (patches) from the image.
    # With stride == kernel_size, we get non-overlapping patches.
    # Output shape: (B*T, C * patch_size * patch_size, N_patches)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)

    # Transpose so patches dimension comes before channels:
    # (B*T, N_patches, patch_dim)
    patches = patches.transpose(1, 2).contiguous()

    N_patches = patches.shape[1]
    patch_dim = patches.shape[2]

    # Restore batch and time dimensions
    return patches.reshape(B, T, N_patches, patch_dim)


def unpatchify(
    patches: torch.Tensor,
    H: int,
    W: int,
    C: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Reassemble patches back into images (inverse of patchify).

    Args:
        patches: (B, T, N_patches, patch_dim) float tensor.
        H: Original image height.
        W: Original image width.
        C: Number of channels (e.g., 3 for RGB).
        patch_size: Side length of each square patch.

    Returns:
        (B, T, C, H, W) float tensor — reconstructed video frames.

    Example:
        >>> patches = torch.randn(2, 4, 64, 192)     # 64 patches of dim 192
        >>> images = unpatchify(patches, H=64, W=64, C=3, patch_size=8)
        >>> images.shape
        torch.Size([2, 4, 3, 64, 64])
    """
    assert patches.dim() == 4, f"Expected 4D tensor (B,T,N,D), got {patches.dim()}D"
    B, T, N_patches, patch_dim = patches.shape
    assert patch_dim == C * patch_size * patch_size, (
        f"patch_dim {patch_dim} != C*p*p = {C * patch_size * patch_size}"
    )

    # Merge batch and time, transpose for F.fold:
    # (B*T, N_patches, patch_dim) -> (B*T, patch_dim, N_patches)
    x = patches.reshape(B * T, N_patches, patch_dim).transpose(1, 2).contiguous()

    # F.fold is the inverse of F.unfold — it reassembles patches into an image.
    # Output shape: (B*T, C, H, W)
    images = F.fold(x, output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    return images.reshape(B, T, C, H, W)
