"""
Perceptual reconstruction losses for tokenizer training.

Two frozen backbones, usable alone or summed ("hybrid"):

- **LPIPS** (``lpips`` package, AlexNet trunk by default). Early dense conv
  features localize small/sparse content well; frames are upscaled
  (nearest) to ``up`` pixels first — LPIPS is calibrated around 64+ px.
- **DINOv3** (HF ``transformers``, lazy import). Early ViT patch features
  add richer global structure. Deep layers are deliberately NOT used: on
  sparse content (a 1-px sprite) deep/abstract features are blind and the
  ViT averages the signal away; early, pixel-near layers carry it. Both
  points were verified by perceptual inversion on the gridworld.

Inputs are always full frames ``(N, C, H, W)`` in ``[0, 1]``; grayscale is
repeated to 3 channels. Both modules keep their parameters frozen and should
be run in fp32 (outside autocast).

DINOv3 model access: set ``HF_TOKEN`` (or point ``HF_TOKEN_FILE`` at a file
containing it) if the checkpoint is gated; ``HF_HOME`` controls the cache.
``DINO_MODEL`` / ``DINO_SIZE`` / ``DINO_LAYERS`` / ``DINO_UPSAMPLE`` /
``DINO_NORM`` override the defaults without touching configs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) with C in {1, 3} -> (N, 3, H, W)."""
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"perceptual losses expect 1 or 3 channels, got {x.shape[1]}")


class LpipsPerceptual(nn.Module):
    """
    LPIPS distance on (optionally upscaled) frames.

    Args:
        net: LPIPS trunk — "alex" (default), "vgg" or "squeeze".
        up:  Upscale frames (nearest) to this size before LPIPS; 0 = never.
             Nearest keeps low-res pixel content crisp instead of smearing it.
    """

    def __init__(self, *, net: str = "alex", up: int = 128):
        super().__init__()
        import lpips
        self.metric = lpips.LPIPS(net=net, verbose=False).eval()
        self.up = int(up)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _to_rgb(pred), _to_rgb(target)
        if self.up and pred.shape[-1] != self.up:
            pred = F.interpolate(pred, size=self.up, mode="nearest")
            target = F.interpolate(target, size=self.up, mode="nearest")
        return self.metric(pred, target, normalize=True).mean()


class DinoV3Perceptual(nn.Module):
    """
    Distance in a frozen DINOv3/v2 ViT's early patch-feature space.

    Frames are upscaled to ``size``, ImageNet-normalized, and compared by
    patch-feature MSE across ``layers`` (multi-scale, like LPIPS over convs).
    Target features are detached.

    Args:
        model_id:  HF model id (default DINOv3 ViT-S/16; env ``DINO_MODEL``).
        size:      Input resolution fed to the ViT (env ``DINO_SIZE``).
        layers:    Hidden-state indices to compare (env ``DINO_LAYERS``,
                   comma-separated). Default (1, 2, 3) — early on purpose.
        normalize: L2-normalize patch features first (env ``DINO_NORM``);
                   off by default — normalizing discards magnitude, which is
                   exactly the signal sparse content lives in.
        upsample:  Interpolation mode for the upscale (env ``DINO_UPSAMPLE``).
    """

    def __init__(self, *, model_id: str | None = None, size: int | None = None,
                 layers: Tuple[int, ...] | None = None,
                 normalize: bool | None = None, upsample: str | None = None):
        super().__init__()
        try:
            from transformers import AutoModel  # lazy: optional dependency
        except ImportError as e:
            raise ImportError(
                "the DINOv3 perceptual backbone needs the 'transformers' "
                "package — install the extra: pip install 'dreamer4-pytorch[dino]'"
            ) from e
        model_id = model_id or os.environ.get("DINO_MODEL", _DEFAULT_DINO_MODEL)
        self.model = AutoModel.from_pretrained(model_id, token=_hf_token()).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.size = int(size or os.environ.get("DINO_SIZE", 224))
        self.upsample = upsample or os.environ.get("DINO_UPSAMPLE", "bilinear")
        if layers is None:
            env = os.environ.get("DINO_LAYERS")
            layers = tuple(int(x) for x in env.split(",")) if env else (1, 2, 3)
        n_layers = int(self.model.config.num_hidden_layers)
        self.layers = tuple(l for l in layers if l <= n_layers) or (n_layers,)
        if normalize is None:
            normalize = os.environ.get("DINO_NORM", "0") == "1"
        self.normalize = bool(normalize)
        # tokens preceding the patch tokens: 1 CLS + any register tokens
        self.n_skip = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x: torch.Tensor) -> List[torch.Tensor]:
        align = False if self.upsample in ("bilinear", "bicubic") else None
        x = F.interpolate(x, size=self.size, mode=self.upsample, align_corners=align)
        x = (x - self.mean.to(x)) / self.std.to(x)
        hidden = self.model(x, output_hidden_states=True).hidden_states
        feats = [hidden[l][:, self.n_skip:, :] for l in self.layers]
        return [F.normalize(f, dim=-1) for f in feats] if self.normalize else feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = self._features(_to_rgb(pred))
        with torch.no_grad():
            ft = self._features(_to_rgb(target))
        return sum(F.mse_loss(a, b) for a, b in zip(fp, ft)) / len(fp)


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = os.environ.get("HF_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        return Path(token_file).read_text().strip()
    return None


def build_perceptual(*, backbone: str, weight: float, up: int, lpips_net: str,
                     dino_weight: float, device: torch.device
                     ) -> List[Tuple[str, nn.Module, float]]:
    """
    Instantiate the configured perceptual terms.

    Args:
        backbone:    "none" | "lpips" | "dinov3" | "hybrid" (LPIPS + DINOv3).
        weight:      Weight of the primary term (LPIPS in lpips/hybrid,
                     DINOv3 in dinov3 mode).
        up:          LPIPS upscale resolution.
        lpips_net:   LPIPS trunk name.
        dino_weight: DINOv3 term weight in hybrid mode.
        device:      Where to place the frozen backbones.

    Returns:
        List of (name, module, weight) — empty when disabled.
    """
    if backbone == "none" or weight == 0.0:
        return []
    if backbone not in ("lpips", "dinov3", "hybrid"):
        raise ValueError(f"unknown perceptual backbone '{backbone}' "
                         "(none|lpips|dinov3|hybrid)")
    terms: List[Tuple[str, nn.Module, float]] = []
    if backbone in ("lpips", "hybrid"):
        terms.append(("lpips", LpipsPerceptual(net=lpips_net, up=up).to(device), weight))
    if backbone in ("dinov3", "hybrid"):
        w = weight if backbone == "dinov3" else dino_weight
        terms.append(("dino", DinoV3Perceptual().to(device), w))
    return terms
