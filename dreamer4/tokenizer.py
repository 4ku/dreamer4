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

from dreamer4.modality import Modality, TokenLayout
from dreamer4.transformer import BlockCausalTransformer
from dreamer4.utils import unpatchify


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
            keep_prob: (B, T, 1) float — fraction of patches kept per image.
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

        return replaced, mae_mask, keep_prob


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
        mlp_ratio: float = 4.0,
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
            latents_only_time=True,
            max_T=max_T,
        )

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            patch_tokens: (B, T, Np, patch_dim) raw patch vectors.

        Returns:
            z: (B, T, N_latents, d_bottleneck) in [-1, 1].
            aux: (mae_mask, keep_prob) for loss computation.
        """
        B, T, Np, _ = patch_tokens.shape
        assert Np == self.n_patches

        proj = self.patch_proj(patch_tokens)  # (B, T, Np, D)
        proj_masked, mae_mask, keep_prob = self.mae(proj)

        lat = self.latents.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        tokens = torch.cat([lat, proj_masked], dim=2)  # (B, T, S, D)

        enc = self.transformer(tokens)

        z = torch.tanh(self.bottleneck_proj(enc[:, :, : self.n_latents, :]))
        return z, (mae_mask, keep_prob)


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
        mlp_ratio: float = 4.0,
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
            latents_only_time=True,
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

        lat = self.up_proj(z)  # (B, T, L, D)
        qry = self.patch_queries.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        tokens = torch.cat([lat, qry], dim=2)  # (B, T, S, D)

        x = self.transformer(tokens)
        patches_out = x[:, :, self.n_latents :, :]
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
            keep_prob: (B, T, 1) per-image keep probability.
        """
        z, (mae_mask, keep_prob) = self.encoder(patch_tokens)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob

    def encode(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Encode patches to bottleneck latents (no MAE masking at eval)."""
        z, _ = self.encoder(patch_tokens)
        return z


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def recon_loss_from_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mae_mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE reconstruction loss on patches the encoder could see.

    During MAE training, some input patches are replaced with a learned
    mask token before entering the encoder (mae_mask=True for those).
    This loss is computed only on the *visible* patches — those the
    encoder received unmodified — by inverting the mask.  When the
    masking probability p=0 (no patches hidden), every patch contributes
    to the loss, which matches the inference regime.

    Args:
        pred:     (B, T, Np, Dp) predicted patches from the decoder.
        target:   (B, T, Np, Dp) ground-truth patches.
        mae_mask: (B, T, Np, 1) bool — True where the patch was hidden
                  from the encoder (replaced with the learned mask token).

    Returns:
        Scalar MSE averaged over visible (encoder-seen) patch elements.
    """
    mask_f = (~mae_mask).to(dtype=torch.float32)  # 1.0 for visible, 0.0 for masked
    diff = pred.float() - target.float()
    sq = diff.pow(2) * mask_f
    denom = mask_f.sum().clamp_min(1.0) * diff.shape[-1]
    return sq.sum() / denom


def lpips_on_mae_recon(
    lpips_fn: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    mae_mask: torch.Tensor,
    *,
    H: int,
    W: int,
    C: int,
    patch_size: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    """
    LPIPS perceptual loss on patches the encoder could see.

    LPIPS operates on full images, so we composite before measuring:
    visible patches (encoder-seen) use the decoder prediction, while
    hidden patches (mae_mask=True) are filled with ground truth.  This
    isolates the perceptual error to the patches the encoder actually
    processed.  When p=0 (no masking), every patch comes from the
    decoder prediction, matching the inference regime.

    Args:
        lpips_fn:        LPIPS network (e.g., lpips.LPIPS(net='alex')).
        pred:            (B, T, Np, Dp) predicted patches from the decoder.
        target:          (B, T, Np, Dp) ground-truth patches.
        mae_mask:        (B, T, Np, 1) bool — True where the patch was
                         hidden from the encoder (replaced with the
                         learned mask token).
        H, W, C:         Image dimensions.
        patch_size:      Patch side length.
        subsample_frac:  Fraction of time steps to use (saves memory).

    Returns:
        Scalar mean LPIPS loss.
    """
    recon_patches = torch.where(mae_mask, target, pred)
    recon_img = unpatchify(recon_patches.float(), H, W, C, patch_size)
    tgt_img = unpatchify(target.float(), H, W, C, patch_size)

    if subsample_frac < 1.0:
        step = max(1, int(1.0 / subsample_frac))
        recon_img = recon_img[:, ::step]
        tgt_img = tgt_img[:, ::step]

    # LPIPS expects input in [-1, 1]
    recon_img = recon_img.clamp(0, 1) * 2.0 - 1.0
    tgt_img = tgt_img.clamp(0, 1) * 2.0 - 1.0

    B, T = recon_img.shape[:2]
    recon_flat = recon_img.reshape(B * T, C, H, W)
    tgt_flat = tgt_img.reshape(B * T, C, H, W)

    with torch.autocast(device_type="cuda", enabled=False):
        lp = lpips_fn(recon_flat.float(), tgt_flat.float())
    return lp.mean()


class RMSLossNormalizer(nn.Module):
    """
    Running RMS loss normalization.

    Maintains an exponential moving average of each loss term's RMS.
    Divides each loss by its running RMS estimate, so fixed coefficients
    control relative importance independent of raw loss magnitudes.

    Args:
        n_losses:  Number of loss terms to track.
        decay:     EMA decay factor (default 0.99).
        eps:       Floor for the RMS estimate (default 1e-8).
    """

    def __init__(self, n_losses: int = 2, decay: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("rms_ema", torch.ones(n_losses))
        self.register_buffer("initialized", torch.zeros(n_losses, dtype=torch.bool))

    @torch.no_grad()
    def update(self, idx: int, loss_val: torch.Tensor) -> None:
        """Update the running RMS estimate for loss at index `idx`."""
        val = loss_val.detach().float().abs().clamp_min(self.eps)
        if not self.initialized[idx]:
            self.rms_ema[idx] = val
            self.initialized[idx] = True
        else:
            self.rms_ema[idx] = self.decay * self.rms_ema[idx] + (1 - self.decay) * val

    def normalize(self, idx: int, loss: torch.Tensor) -> torch.Tensor:
        """Divide loss by its running RMS estimate."""
        return loss / self.rms_ema[idx].clamp_min(self.eps)
