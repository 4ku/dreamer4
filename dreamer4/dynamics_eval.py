"""
Evaluation of a phase-1b dynamics model — three tools, three questions.

All three take a trained dynamics model and a batch of REAL latents and
actions; :func:`gate_rollout` additionally needs the frozen tokenizer (to
decode the dream back to pixels) and the dataset (for its domain hooks).
:mod:`dreamer4.train.train_dynamics` calls them during validation and again
for the final gate table.

- :func:`one_step_eval` — **"is it even a conditional model?"** No dreaming:
  at every step the model gets the TRUE history (teacher-forced) and
  predicts one frame; the error is compared against the copy-last-frame
  baseline. Errors cannot compound here, so this isolates whether
  context+action conditioning is used at all (healthy: ``onestep_mse`` far
  below ``copy_mse``). A model that fails this is broken at step one.
- :func:`windowed_rollout` — **"produce the dream."** The mechanism, no
  scoring: after ``ctx`` real frames the model generates frame after frame
  conditioned on its OWN previous generations, seeing only the last
  ``window-1`` frames — exactly the loop imagination runs in (sliding
  window, per-window start-flag action alignment, ``tau_ctx`` context
  corruption).
- :func:`gate_rollout` — **"judge the dream."** Runs the rollout, decodes it,
  and scores it against the REAL frames: latent/pixel error, the dataset's
  domain hooks (``eval_metrics`` — e.g. gridworld sprite errors — and the
  pass/fail ``gate``), plus ms/frame. This is the number that decides
  whether a world model ships: teacher-forced losses and rollouts that keep
  the whole history both flatter the model, so only the windowed gate is
  trustworthy.

The two scored evals bracket the failure modes: gate bad + one-step good =>
drift (errors compound once the model feeds on itself); gate bad + one-step
bad => conditioning itself is broken. Also here: :func:`save_demo_gif`, the
human-eyeball artifact.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dreamer4.data import EpisodeVideoDataset
from dreamer4.models.dynamics import DynamicsModel, align_actions, sample_one_timestep
from dreamer4.models.tokenizer import FrozenTokenizer


@torch.no_grad()
def windowed_rollout(
    dynamics: DynamicsModel,
    *,
    z_full: torch.Tensor,
    actions: torch.Tensor,
    proprio_full: Optional[torch.Tensor] = None,
    window: int,
    ctx: int,
    K: int,
    k_max: int,
    tau_ctx: float = 0.1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """
    PRODUCE a dream (no scoring).

    Frames ``ctx..T-1`` are generated open-loop under the real action
    sequence, each conditioned on the model's own previous generations
    through the sliding window.

    Args:
        z_full:       (B, T, Nz, Dz) real latents; only the first ``ctx`` are
                      used as context, the rest exist for the caller to score
                      against.
        actions:      (B, T-1, D_a) dataset action vectors.
        proprio_full: (B, T, d_p) iff the model has a joint proprio stream.
        window:       Sliding context window; use the trained ``seq_len``, as
                      the model never saw a longer one.
        ctx:          Real frames handed over before dreaming starts.
        K:            Denoising steps per generated frame.
        k_max:        Finest shortcut grid the model was trained on.
        tau_ctx:      Context corruption level, matching training.

    Returns:
        ``(z_dream (B, T-ctx, Nz, Dz), proprio_dream | None, ms_per_frame)``.
    """
    if window < 2:
        raise ValueError("windowed_rollout needs window >= 2 (one context "
                         "slot plus the generated frame)")
    B, T = z_full.shape[:2]
    device = z_full.device
    aligned = align_actions(actions, T)                    # (B, T, Da+1)
    start_vec = torch.zeros_like(aligned[:, 0])
    start_vec[:, -1] = 1.0

    zs = [z_full[:, t] for t in range(ctx)]                # real context
    ps = [proprio_full[:, t] for t in range(ctx)] if proprio_full is not None else None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(ctx, T):
        p = min(len(zs), window - 1)
        z_ctx = torch.stack(zs[-p:], 1)                    # (B, p, Nz, Dz)
        act_win = aligned[:, t - p : t + 1].clone()        # (B, p+1, Da+1)
        act_win[:, 0] = start_vec                          # window start: no arrival
        if ps is not None:
            z_new, prop_new, _ = sample_one_timestep(
                dynamics, past_packed=z_ctx, k_max=k_max, K=K, actions=act_win,
                tau_ctx=tau_ctx, past_proprio=torch.stack(ps[-p:], 1), cache=None)
            ps.append(prop_new.clamp(-1, 1))
        else:
            z_new, _ = sample_one_timestep(
                dynamics, past_packed=z_ctx, k_max=k_max, K=K, actions=act_win,
                tau_ctx=tau_ctx, cache=None)
        zs.append(z_new.clamp(-1, 1))
    if device.type == "cuda":
        torch.cuda.synchronize()
    # ms per generated frame PER EPISODE: the loop's wall time divided by both
    # the dreamed horizon and the batch, so the number stays comparable across
    # batch sizes. Every speed figure quoted for this model uses it.
    ms = (time.perf_counter() - t0) * 1000.0 / (max(T - ctx, 1) * B)

    z_dream = torch.stack(zs[ctx:], 1)
    prop_dream = torch.stack(ps[ctx:], 1) if ps is not None else None
    return z_dream, prop_dream, ms


@torch.no_grad()
def gate_rollout(
    dynamics: DynamicsModel,
    tokenizer: FrozenTokenizer,
    dataset: EpisodeVideoDataset,
    batch: Dict[str, torch.Tensor],
    *,
    window: int,
    ctx: int,
    K: int,
    k_max: int,
    tau_ctx: float = 0.1,
) -> Tuple[Dict[str, float], Tuple[torch.Tensor, torch.Tensor]]:
    """
    JUDGE the dream — the shipping verdict for a dynamics model.

    Runs :func:`windowed_rollout`, decodes the result with the frozen
    tokenizer and scores it against the REAL frames.

    Args:
        tokenizer: The frozen tokenizer that defines this latent space; it
                   decodes the dream, and must be the SAME one the dynamics
                   model was trained against or the pixels are meaningless.
        dataset:   Supplies the domain hooks ``eval_metrics`` and ``gate``.
        batch:     {"z": (B, T, Nz, Dz), "actions": (B, T-1, Da), "video":
                   (B, T, H, W, C) uint8, "proprio"?: (B, T, d_p)}, with
                   ``T = ctx + horizon``.
        window, ctx, K, k_max, tau_ctx: See :func:`windowed_rollout`.

    Returns:
        ``(metrics, (gt_frames, dream_frames))``. ``metrics`` holds
        latent/pixel MSE over the dreamed span, per-frame ms, the dataset's
        own hook metrics (e.g. sprite errors), ``gate_pass`` when the dataset
        defines a gate, and proprio rollout error when the model has that
        stream. The frame pair is (B, T-ctx, C, H, W) in [0,1], for strips.
    """
    z_full, video = batch["z"], batch["video"]
    proprio = batch.get("proprio")
    z_dream, prop_dream, ms = windowed_rollout(
        dynamics, z_full=z_full, actions=batch["actions"], proprio_full=proprio,
        window=window, ctx=ctx, K=K, k_max=k_max, tau_ctx=tau_ctx)

    dream_img = tokenizer.decode_latents(z_dream)              # (B,h,C,H,W)
    true_img = video[:, ctx:].to(dream_img.device).float() / 255.0
    true_img = true_img.permute(0, 1, 4, 2, 3).contiguous()

    metrics = {
        "latent_mse": float(F.mse_loss(z_dream, z_full[:, ctx:])),
        "pixel_mse": float(F.mse_loss(dream_img, true_img)),
        "ms_per_frame": ms,
    }
    if prop_dream is not None:
        metrics["proprio_rollout_mae"] = float(
            (prop_dream - proprio[:, ctx:]).abs().mean())
    true_np = true_img.cpu().numpy().transpose(0, 1, 3, 4, 2)
    dream_np = dream_img.cpu().numpy().transpose(0, 1, 3, 4, 2)
    metrics.update(dataset.eval_metrics(true_np, dream_np))
    gate = dataset.gate(metrics)
    if gate is not None:
        metrics["gate_pass"] = bool(gate)
    return metrics, (true_img.cpu(), dream_img.cpu())


@torch.no_grad()
def one_step_eval(
    dynamics: DynamicsModel,
    z1: torch.Tensor,
    actions: torch.Tensor,
    *,
    ctx: int,
    K: int,
    k_max: int,
    tau_ctx: float = 0.1,
    proprio: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    PROBE the basics (no dreaming).

    Teacher-forced 1-step prediction from TRUE history, against the copy
    baseline (``z[t-1]`` used as the prediction of ``z[t]``). Errors cannot
    compound, so this isolates whether context+action conditioning is used at
    all: healthy => ``onestep_mse << copy_mse``.

    Args:
        z1:      (B, T, Nz, Dz) real latents.
        actions: (B, T-1, D_a) dataset action vectors.
        ctx:     First frame to predict; frames before it are history only.
        K:       Denoising steps per predicted frame.
        k_max:   Finest shortcut grid the model was trained on.
        tau_ctx: Context corruption level, matching training.
        proprio: (B, T, d_p) iff the model has a joint proprio stream.

    Returns:
        {"onestep_mse": mean over predicted frames, "copy_mse": the same
        span scored by the copy-last-frame baseline}.
    """
    T = z1.shape[1]
    aligned = align_actions(actions, T)
    step_mses = []
    for t in range(ctx, T):
        prop_kw = ({"past_proprio": proprio[:, :t]}
                   if proprio is not None else {})
        out = sample_one_timestep(
            dynamics, past_packed=z1[:, :t], k_max=k_max, K=K,
            actions=aligned[:, : t + 1], tau_ctx=tau_ctx, cache=None, **prop_kw)
        z_next = out[0]
        step_mses.append(float(((z_next - z1[:, t]) ** 2).mean()))
    copy_mse = float(((z1[:, ctx - 1 : T - 1] - z1[:, ctx:]) ** 2).mean())
    return {"onestep_mse": float(np.mean(step_mses)), "copy_mse": copy_mse}


@torch.no_grad()
def save_demo_gif(gt: torch.Tensor, dream: torch.Tensor, out_path, *,
                  ctx: int, scale: int = 10, fps: int = 4) -> None:
    """
    Write a side-by-side GT (top) vs dream (bottom) animated GIF.

    Args:
        gt, dream: (B, T, C, H, W) in [0,1]. ``dream`` covers frames
                   ``ctx..T-1`` only; pass the FULL ground truth so the
                   context phase is visible in both rows.
        out_path:  Destination file.
        ctx:       Context length, used to label frames and to index into
                   ``dream``.
        scale:     Nearest-neighbour upscale factor per pixel.
        fps:       Playback rate of the written GIF.
    """
    from PIL import Image, ImageDraw
    gt_np = (gt.numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
    dr_np = (dream.numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
    if gt_np.shape[-1] == 1:
        gt_np, dr_np = [np.repeat(x, 3, axis=-1) for x in (gt_np, dr_np)]
    B, T, H, W = gt_np.shape[:4]

    def up(frame):
        return frame.repeat(scale, 0).repeat(scale, 1)

    cell_h, cell_w = H * scale, W * scale
    sep, label_h = 2, 14
    canvas_w = sep + B * (cell_w + sep)
    canvas_h = label_h + 2 * cell_h + 3 * sep
    frames = []
    for t in range(T):
        canvas = np.full((canvas_h, canvas_w, 3), 20, np.uint8)
        for b in range(B):
            x0 = sep + b * (cell_w + sep)
            canvas[label_h : label_h + cell_h, x0 : x0 + cell_w] = up(gt_np[b, t])
            y1 = label_h + cell_h + sep
            src = gt_np[b, t] if t < ctx else dr_np[b, t - ctx]
            canvas[y1 : y1 + cell_h, x0 : x0 + cell_w] = up(src)
        img = Image.fromarray(canvas)
        phase = "context" if t < ctx else "DREAMED"
        ImageDraw.Draw(img).text(
            (3, 1), f"t={t} ({phase})  top=real  bottom=world model",
            fill=(235, 235, 235))
        frames.append(img)
    frames += [frames[-1]] * 3
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
