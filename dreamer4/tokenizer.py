"""
Causal Tokenizer for Dreamer 4.

Compresses raw video frames into continuous latent representations (Encoder)
and reconstructs frames from latents (Decoder). Both components use the
block-causal transformer backbone and are causal in time, enabling
frame-by-frame decoding for interactive inference.

Paper reference: Section 3.1, Figure 2(a), Eq. 5.

Architecture (Encoder):
    patches (B,T,Np,Dp)
      -> linear project to d_model
      -> MAE masking (p ~ U(0, 0.9))
      -> prepend N learned latent tokens
      -> BlockCausalTransformer (encoder mode)
      -> extract latent tokens
      -> linear project to d_bottleneck
      -> tanh
    output: (B,T,N_latents,d_bottleneck) in [-1, 1]

Architecture (Decoder):
    z (B,T,N_latents,d_bottleneck)
      -> linear project up to d_model
      -> append N_patches learned query tokens
      -> BlockCausalTransformer (decoder mode)
      -> extract patch tokens
      -> linear project to patch_dim
      -> sigmoid
    output: (B,T,Np,Dp) in [0, 1]

Attention patterns:
    Encoder: latents attend to all tokens; each modality only sees itself.
    Decoder: latents attend to latents only; patches see themselves + latents.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import lpips

from dreamer4.transformer.modality import Modality, TokenLayout
from dreamer4.transformer import BlockCausalTransformer
from dreamer4.transformer.transformer import unpatchify


# ---------------------------------------------------------------------------
# MAE Masking
# ---------------------------------------------------------------------------

class MAEReplacer(nn.Module):
    """
    Masked Autoencoding patch dropout.

    During training, randomly masks out image patches by replacing them with
    a learned embedding. The masking probability is sampled per image from
    U(p_min, p_max), so the model sometimes sees fully unmasked inputs
    (the inference regime).

    Args:
        d_model: Dimension of projected patch tokens.
        p_min:   Minimum masking probability (default 0.0).
        p_max:   Maximum masking probability (default 0.9).
    """

    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, patches: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: (B, T, Np, D) projected patch tokens.

        Returns:
            replaced:  (B, T, Np, D) patches with masked positions replaced.
            mae_mask:  (B, T, Np, 1) bool — True where masked (must reconstruct).
        """
        B, T, Np, D = patches.shape
        device = patches.device

        if not self.training or (self.p_min == 0.0 and self.p_max == 0.0):
            keep_prob = torch.ones(B, T, 1, device=device, dtype=patches.dtype)
            mae_mask = torch.zeros(B, T, Np, 1, device=device, dtype=torch.bool)
            return patches, mae_mask, keep_prob

        # Sample per-image masking probability p ~ U(p_min, p_max)
        p = torch.empty(B, T, device=device).uniform_(self.p_min, self.p_max)
        keep_prob = (1.0 - p).unsqueeze(-1)  # (B, T, 1)

        # Bernoulli mask: True = keep, False = mask
        keep = torch.rand(B, T, Np, device=device) < keep_prob  # (B, T, Np)
        mae_mask = (~keep).unsqueeze(-1)  # (B, T, Np, 1) True = masked

        mask_tok = self.mask_token.to(dtype=patches.dtype)
        replaced = torch.where(keep.unsqueeze(-1), patches, mask_tok)

        return replaced, mae_mask


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Tokenizer encoder: compress video patches into a continuous bottleneck.

    Args:
        patch_dim:     Raw patch vector dimension (patch_size^2 * C).
        d_model:       Transformer hidden dimension.
        n_latents:     Number of learned latent tokens.
        n_patches:     Number of image patches per frame.
        n_heads:       Number of attention heads.
        depth:         Number of transformer layers.
        d_bottleneck:  Bottleneck output dimension per latent token.
        n_kv_heads:    KV heads for GQA (default = n_heads).
        mlp_ratio:     MLP expansion ratio.
        time_every:    Apply time attention every N layers.
        dropout:       Dropout rate.
        use_qk_norm:   Use QKNorm in attention.
        logit_cap:     Logit soft capping value.
        mae_p_min:     Minimum MAE masking probability.
        mae_p_max:     Maximum MAE masking probability.
        max_T:         Maximum time steps for RoPE cache.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        d_model: int,
        n_latents: int,
        n_patches: int,
        n_heads: int,
        depth: int,
        d_bottleneck: int,
        n_kv_heads: int | None = None,
        mlp_ratio: float = 8/3,
        time_every: int = 1,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        mae_p_min: float = 0.0,
        mae_p_max: float = 0.9,
        max_T: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

        self.mae = MAEReplacer(d_model, p_min=mae_p_min, p_max=mae_p_max)

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

        layout = TokenLayout(
            n_latents=n_latents,
            segments=((Modality.IMAGE, n_patches),),
        )

        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            layout=layout,
            space_mode="encoder",
            n_kv_heads=n_kv_heads,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            logit_cap=logit_cap,
            max_T=max_T,
        )

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, T, n_patches, patch_dim) raw patch vectors.

        Returns:
            z: (B, T, N_latents, d_bottleneck) in [-1, 1].
            aux: mae_mask for loss computation.
        """
        B, T, n_patches, _ = patch_tokens.shape
        assert n_patches == self.n_patches

        proj = self.patch_proj(patch_tokens)  # (B, T, n_patches, d_model)
        proj_masked, mae_mask = self.mae(proj)

        lat = self.latents.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        tokens = torch.cat([lat, proj_masked], dim=2)  # (B, T, n_latents + n_patches, d_model)

        enc = self.transformer(tokens) # (B, T, n_latents + n_patches, d_model)

        z = torch.tanh(self.bottleneck_proj(enc[:, :, : self.n_latents, :]))
        return z, mae_mask


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Tokenizer decoder: reconstruct video patches from bottleneck latents.

    Args:
        d_bottleneck: Bottleneck dimension (input).
        d_model:      Transformer hidden dimension.
        n_latents:    Number of latent tokens.
        n_patches:    Number of image patches per frame.
        patch_dim:    Output patch vector dimension (patch_size^2 * C).
        n_heads:      Number of attention heads.
        depth:        Number of transformer layers.
        n_kv_heads:   KV heads for GQA.
        mlp_ratio:    MLP expansion ratio.
        time_every:   Apply time attention every N layers.
        dropout:      Dropout rate.
        use_qk_norm:  Use QKNorm in attention.
        logit_cap:    Logit soft capping value.
        max_T:        Maximum time steps for RoPE cache.
    """

    def __init__(
        self,
        *,
        d_bottleneck: int,
        d_model: int,
        n_latents: int,
        n_patches: int,
        patch_dim: int,
        n_heads: int,
        depth: int,
        n_kv_heads: int | None = None,
        mlp_ratio: float = 8/3,
        time_every: int = 1,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        max_T: int = 1024,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_patches = n_patches

        self.up_proj = nn.Linear(d_bottleneck, d_model)

        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_head = nn.Linear(d_model, patch_dim)

        layout = TokenLayout(
            n_latents=n_latents,
            segments=((Modality.IMAGE, n_patches),),
        )

        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            layout=layout,
            space_mode="decoder",
            n_kv_heads=n_kv_heads,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            logit_cap=logit_cap,
            max_T=max_T,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, N_latents, d_bottleneck) bottleneck representations.

        Returns:
            (B, T, Np, patch_dim) reconstructed patches in [0, 1].
        """
        B, T, L, _ = z.shape
        assert L == self.n_latents

        lat = self.up_proj(z)  # (B, T, n_latents, d_model)
        qry = self.patch_queries.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1) # (B, T, n_patches, d_model)
        tokens = torch.cat([lat, qry], dim=2)  # (B, T, n_patches + n_latents, d_model)

        x = self.transformer(tokens) # (B, T, n_patches + n_latents, d_model)
        patches_out = x[:, :, self.n_latents :, :] # (B, T, n_patches, d_model)
        return torch.sigmoid(self.patch_head(patches_out))


# ---------------------------------------------------------------------------
# Tokenizer (Encoder + Decoder wrapper)
# ---------------------------------------------------------------------------

class Tokenizer(nn.Module):
    """
    Full causal tokenizer: Encoder -> bottleneck -> Decoder.

    Args:
        encoder: Encoder instance.
        decoder: Decoder instance.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, T, Np, patch_dim) raw patch vectors.

        Returns:
            pred:      (B, T, Np, patch_dim) reconstructed patches in [0, 1].
            mae_mask:  (B, T, Np, 1) bool mask (True = masked).
        """
        z, mae_mask = self.encoder(patch_tokens)
        pred = self.decoder(z)
        return pred, mae_mask

    def encode(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Encode patches to bottleneck latents (no MAE masking at eval)."""
        z, _ = self.encoder(patch_tokens)
        return z


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------


class MAECompositedLPIPS(nn.Module):
    """
    Official LPIPS (``lpips`` package) on MAE-composited full frames.

    LPIPS is defined on full images. We composite: visible patches use the
    decoder prediction; masked patches (``mae_mask`` True) use ground truth,
    then measure LPIPS against the full target image. Patch tensors are in
    ``[0, 1]``; the underlying metric uses ``normalize=True`` (``[0, 1]`` RGB).

    Args:
        net: Trunk for ``lpips.LPIPS`` (e.g. ``"alex"``, ``"vgg"``, ``"vgg16"``).
        H, W, C: Image shape fed to ``unpatchify``.
        patch_size: Patch side length.
        verbose: Passed through to ``lpips.LPIPS``.
    """

    def __init__(
        self,
        *,
        net: str,
        H: int,
        W: int,
        C: int,
        patch_size: int,
        verbose: bool = False,
    ):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.C = int(C)
        self.patch_size = int(patch_size)
        self.lpips_metric = lpips.LPIPS(net=net, verbose=verbose)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mae_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:       (B, T, n_patches, patch_dim) decoder predictions.
            target:     (B, T, n_patches, patch_dim) ground-truth patches.
            mae_mask:   (B, T, n_patches, 1) bool — True where encoder mask replaced the patch.

        Returns:
            Scalar mean LPIPS over batch*time.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        H, W, C, patch_size = self.H, self.W, self.C, self.patch_size

        recon_patches = torch.where(mae_mask, target, pred)
        recon_img = unpatchify(recon_patches.float(), H, W, C, patch_size) # (B, T, C, H, W)
        tgt_img = unpatchify(target.float(), H, W, C, patch_size) # (B, T, C, H, W)

        # Images are already in [0, 1], so no need to clamp.
        # recon_img = recon_img.clamp(0.0, 1.0)
        # tgt_img = tgt_img.clamp(0.0, 1.0)

        if C == 1:
            recon_img = recon_img.repeat(1, 1, 3, 1, 1) 
            tgt_img = tgt_img.repeat(1, 1, 3, 1, 1)
            c_lpips = 3
        elif C == 3:
            c_lpips = 3
        else:
            raise ValueError(f"LPIPS expects C in {{1, 3}}, got {C}")

        B, T = recon_img.shape[:2]
        recon_flat = recon_img.reshape(B * T, c_lpips, H, W)
        tgt_flat = tgt_img.reshape(B * T, c_lpips, H, W)

        with torch.autocast(device_type=recon_flat.device.type, enabled=False):
            lp = self.lpips_metric(
                recon_flat.float(), tgt_flat.float(), normalize=True
            )
        return lp.mean()
