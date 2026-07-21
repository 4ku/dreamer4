"""
Causal Tokenizer for Dreamer 4.

Compresses raw video frames into continuous latent representations (Encoder)
and reconstructs frames from latents (Decoder). Both components use the
block-causal transformer backbone and are causal in time, enabling
frame-by-frame decoding for interactive inference.

Also here: :class:`FrozenTokenizer` — a trained checkpoint wrapped as the
frozen pixels<->latents codec that dynamics training and inference consume
(sliding temporal history + bottleneck packing).

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

import numpy as np
import torch
import torch.nn as nn

from dreamer4.models.transformer.modality import Modality, TokenLayout
from dreamer4.models.transformer import BlockCausalTransformer
from dreamer4.models.transformer.transformer import patchify, unpatchify
from dreamer4.utils import pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck


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
            mae_mask = torch.zeros(B, T, Np, 1, device=device, dtype=torch.bool)
            return patches, mae_mask

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
        d_proprio:     Optional proprioceptive input dimension (e.g. robot joint
                       state, or player position on the gridworld). When set, one
                       PROPRIO token per timestep is appended after the patches.
                       Per the paper's multi-modality rule, latent tokens attend to
                       all modalities while each modality attends only within
                       itself — the existing "encoder" mask already implements
                       this, so no masking changes are needed.
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
        max_T: int = 256,
        d_proprio: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.d_proprio = d_proprio

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

        self.mae = MAEReplacer(d_model, p_min=mae_p_min, p_max=mae_p_max)

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

        segments = [(Modality.IMAGE, n_patches)]
        if d_proprio is not None:
            self.proprio_proj = nn.Linear(d_proprio, d_model)
            segments.append((Modality.PROPRIO, 1))

        layout = TokenLayout(
            n_latents=n_latents,
            segments=tuple(segments),
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
        self, patch_tokens: torch.Tensor, proprio: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, T, n_patches, patch_dim) raw patch vectors.
            proprio:      (B, T, d_proprio) proprioceptive state; required iff the
                          encoder was built with ``d_proprio``.

        Returns:
            z: (B, T, N_latents, d_bottleneck) in [-1, 1].
            aux: mae_mask for loss computation.
        """
        B, T, n_patches, _ = patch_tokens.shape
        assert n_patches == self.n_patches
        assert (proprio is not None) == (self.d_proprio is not None), \
            "pass `proprio` iff the encoder was built with d_proprio"

        proj = self.patch_proj(patch_tokens)  # (B, T, n_patches, d_model)
        proj_masked, mae_mask = self.mae(proj)

        lat = self.latents.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        parts = [lat, proj_masked]
        if proprio is not None:
            parts.append(self.proprio_proj(proprio).unsqueeze(2))  # (B, T, 1, d_model)
        tokens = torch.cat(parts, dim=2)  # (B, T, n_latents + n_patches (+1), d_model)

        enc = self.transformer(tokens)

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
        space_mode: str = "decoder",
        max_T: int = 256,
        d_proprio: int | None = None,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.d_proprio = d_proprio

        self.up_proj = nn.Linear(d_bottleneck, d_model)

        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_head = nn.Linear(d_model, patch_dim)

        segments = [(Modality.IMAGE, n_patches)]
        if d_proprio is not None:
            # one learned PROPRIO query decoded back to the proprio vector; per the
            # paper each decoder modality attends within itself and to the latents
            # ("decoder" mode) or, under "decoder_cross", to the latents only —
            # both existing masks handle the extra segment unchanged.
            self.proprio_query = nn.Parameter(torch.empty(1, d_model))
            nn.init.normal_(self.proprio_query, std=0.02)
            self.proprio_head = nn.Linear(d_model, d_proprio)
            segments.append((Modality.PROPRIO, 1))

        layout = TokenLayout(
            n_latents=n_latents,
            segments=tuple(segments),
        )

        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            layout=layout,
            space_mode=space_mode,
            n_kv_heads=n_kv_heads,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            logit_cap=logit_cap,
            max_T=max_T,
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, T, N_latents, d_bottleneck) bottleneck representations.

        Returns:
            (B, T, Np, patch_dim) reconstructed patches in [0, 1]; when the decoder
            was built with ``d_proprio``, a tuple ``(patches, proprio_pred)`` with
            proprio_pred (B, T, d_proprio) (linear, no activation).
        """
        B, T, L, _ = z.shape
        assert L == self.n_latents

        lat = self.up_proj(z)  # (B, T, n_latents, d_model)
        qry = self.patch_queries.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1) # (B, T, n_patches, d_model)
        parts = [lat, qry]
        if self.d_proprio is not None:
            parts.append(self.proprio_query.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1))
        tokens = torch.cat(parts, dim=2)

        x = self.transformer(tokens)
        patches_out = x[:, :, self.n_latents : self.n_latents + self.n_patches, :]
        patches = torch.sigmoid(self.patch_head(patches_out))
        if self.d_proprio is None:
            return patches
        proprio_pred = self.proprio_head(x[:, :, -1, :])       # (B, T, d_proprio)
        return patches, proprio_pred


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
        self, patch_tokens: torch.Tensor, proprio: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, T, Np, patch_dim) raw patch vectors.
            proprio:      (B, T, d_proprio), iff built with proprio support.

        Returns:
            pred:      (B, T, Np, patch_dim) reconstructed patches in [0, 1]
                       (or ``(patches, proprio_pred)`` when the decoder decodes proprio).
            mae_mask:  (B, T, Np, 1) bool mask (True = masked).
        """
        z, mae_mask = self.encoder(patch_tokens, proprio=proprio)
        pred = self.decoder(z)
        return pred, mae_mask

    def encode(self, patch_tokens: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        """Encode patches to bottleneck latents (no MAE masking at eval)."""
        z, _ = self.encoder(patch_tokens, proprio=proprio)
        return z


# ---------------------------------------------------------------------------
# FrozenTokenizer (deployment form: pixels <-> packed dynamics latents)
# ---------------------------------------------------------------------------


class FrozenTokenizer(nn.Module):
    """
    A TRAINED tokenizer, frozen, as a pixels<->latents codec — the form in
    which dynamics training and inference consume it:

        encode_frames(video (B,T,H,W,C) uint8) -> z (B,T,n_spatial,d_spatial)
        decode_latents(z) -> images (B,T,C,H,W) in [0, 1]

    **Sliding history.** ``latent[t]`` is the time-attention encoding of the
    ``history`` frames ``[t-history+1 .. t]`` (taking the last position),
    replicate-padding before the episode start. This DECOUPLES the
    tokenizer's temporal receptive field (fixed ``history``) from the
    dynamics rollout horizon: every latent depends on at most ``history``
    frames, so it is a deterministic per-frame function — episodes can be
    encoded once and cached, and the world model can roll arbitrarily far.
    ``history=1`` is plain per-frame encoding (the same code path — a window
    of one). Decoding uses the same sliding window over latents, matching
    how the production stack renders dreams.

    The bottleneck ``(n_latents, d_bottleneck)`` is packed to
    ``(n_spatial, d_bottleneck * pack_k)`` — the tokenizer <-> dynamics
    interface from the paper's Appendix A.

    Args:
        ckpt_path:  ``train_tokenizer`` checkpoint (EMA weights preferred).
        history:    Sliding temporal window w >= 1.
        pack_k:     Bottleneck packing factor (must divide n_latents).
        decoder_ckpt: Optional second checkpoint whose DECODER replaces this
                    one's (e.g. a noise-robust decoder fine-tuned with a
                    frozen encoder — the latent space must be identical,
                    which is verified).
        device:     Where to run.
    """

    def __init__(self, ckpt_path, *, history: int = 1, pack_k: int = 1,
                 decoder_ckpt=None, device: str | torch.device = "cuda"):
        super().__init__()
        # lazy: the checkpoint loader lives in the trainer module, which
        # imports the model classes above — a top-level import would cycle
        from dreamer4.train.train_tokenizer import load_tokenizer_checkpoint
        tok, ckpt = load_tokenizer_checkpoint(ckpt_path, device=device)
        if ckpt["frame_meta"]["d_proprio"] is not None:
            raise ValueError("proprio-decoding tokenizers are not supported as "
                             "a dynamics codec — train the tokenizer with "
                             "data.proprio=none for this use")
        if decoder_ckpt:
            robust, _ = load_tokenizer_checkpoint(decoder_ckpt, device=device)
            enc_a, enc_b = tok.encoder.state_dict(), robust.encoder.state_dict()
            if any(not torch.equal(enc_a[k], enc_b[k]) for k in enc_a):
                raise ValueError(f"{decoder_ckpt} was not fine-tuned from "
                                 f"{ckpt_path}: encoders differ, so its decoder "
                                 "speaks a different latent space")
            tok.decoder = robust.decoder
        meta = ckpt["frame_meta"]
        self.H, self.W, self.C = meta["H"], meta["W"], meta["C"]
        self.patch_size = ckpt["config"]["model"]["patch_size"]
        self.n_latents = ckpt["config"]["model"]["n_latents"]
        self.d_bottleneck = ckpt["config"]["model"]["d_bottleneck"]
        self.history = int(history)
        if not 1 <= self.history <= meta["max_T"]:
            raise ValueError(f"history={history} must be in [1, max_T="
                             f"{meta['max_T']}] of the tokenizer")
        if self.n_latents % pack_k:
            raise ValueError(f"pack_k={pack_k} must divide n_latents={self.n_latents}")
        self.pack_k = int(pack_k)
        self.n_spatial = self.n_latents // self.pack_k
        self.d_spatial = self.d_bottleneck * self.pack_k
        for p in tok.parameters():
            p.requires_grad_(False)
        self.tok = tok.eval()
        self.device = torch.device(device)

    def _slide(self, x: torch.Tensor) -> torch.Tensor:
        """(B,T,*) -> (B*T, w, *): causal length-w windows per position,
        replicate-padding the first w-1 slots (a static clip — exactly what
        the tokenizer saw at every episode start)."""
        B, T = x.shape[:2]
        w = self.history
        if w == 1:
            return x.reshape(B * T, 1, *x.shape[2:])
        pad = x[:, :1].expand(B, w - 1, *x.shape[2:])
        padded = torch.cat([pad, x], dim=1)                    # (B, T+w-1, *)
        windows = padded.unfold(1, w, 1).movedim(-1, 2)        # (B, T, w, *)
        return windows.reshape(B * T, w, *x.shape[2:])

    @torch.no_grad()
    def encode_frames(self, video) -> torch.Tensor:
        """(B,T,H,W,C) uint8 (numpy or tensor) -> (B,T,n_spatial,d_spatial)."""
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        v = video.to(self.device).float() / 255.0
        v = v.permute(0, 1, 4, 2, 3).contiguous()              # (B,T,C,H,W)
        B, T = v.shape[:2]
        windows = self._slide(v)                               # (B*T,w,C,H,W)
        patches = patchify(windows, self.patch_size)
        z = self.tok.encode(patches)[:, -1]                    # last frame per window
        z = z.reshape(B, T, self.n_latents, self.d_bottleneck)
        return pack_bottleneck_to_spatial(z, n_spatial=self.n_spatial, k=self.pack_k)

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """(B,T,n_spatial,d_spatial) -> images (B,T,C,H,W) in [0,1]."""
        B, T = z.shape[:2]
        latents = unpack_spatial_to_bottleneck(z.to(self.device), k=self.pack_k)
        windows = self._slide(latents)                         # (B*T,w,Nl,Db)
        patches = self.tok.decoder(windows)[:, -1]             # last frame per window
        patches = patches.reshape(B, T, *patches.shape[-2:])
        img = unpatchify(patches, self.H, self.W, C=self.C,
                         patch_size=self.patch_size)
        return img.clamp(0.0, 1.0)
