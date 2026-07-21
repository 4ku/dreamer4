"""
Training objectives for the latent-space dynamics model — ALL of them.

Two objectives, selected by ``--objective.objective``:

1. :func:`shortcut_forcing_loss` — the paper's objective (Section 3.2,
   Eq. 4/7). Every frame of the window is independently noised at its own
   random signal level and reconstructed at once (flow term at the finest
   step + a bootstrap term distilling two half-steps). Kept as the faithful
   baseline; on sparse content it collapses — the model learns to denoise
   each frame from its own input instead of reading context and actions.
2. :func:`clean_context_loss` — the production fix. Context frames stay
   nearly clean (the inference regime); ONE target frame is noised across
   the full signal grid and is the only graded frame — the model can only
   score by reading context + action. Adds the exposure-bias machinery the
   recipe relies on: a context-noise band, scheduled sampling, an optional
   single-slot bootstrap term, and a joint proprio stream.

Both are built from the same blocks, defined here once: the (tau, d)
schedules, signal-level corruption, the ramp weight (Eq. 8), and the
velocity-space steps of the bootstrap teacher (Eq. 7).

**Action convention.** Datasets provide action VECTORS ``a[t]`` (one-hot for
discrete spaces) driving the ``t -> t+1`` transition. The model is fed
per-frame ALIGNED actions with a start flag: ``A[t] = [a[t-1], 0]`` — "the
action that led into frame t" — and ``A[0] = [zeros, 1]`` (nothing led into
the first frame of a window). For gridworld's one-hot moves this reproduces
the proven 5-category NULL encoding exactly; for continuous robot actions it
is the natural generalization. The helper is the model's input convention and
lives with the model: :func:`dreamer4.models.dynamics.align_actions`.

**Two noise parameterizations to keep straight** (both inherited from the
paper): ``tau`` is a SIGNAL level (``tau=1`` is clean — corruption mixes
``tau * clean + (1-tau) * noise``), while ``tau_ctx`` / the context band are
NOISE fractions (``0.1`` means 10% noise — mixing ``(1-tc) * clean +
tc * noise``). The context helpers below say which one they take.

Deliberately NOT ported from the root-level experimental script: naive deep
roll-in (measured harmful — contradictory supervision), relabeled roll-in
(requires a queryable env transition rule — impossible on real video),
alternating batch lengths and the shortcut-mix schedule (both superseded by
the bootstrap-later fine-tune in the production recipe).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from dreamer4.models.dynamics import DynamicsModel, _log2_int, sample_one_timestep


# ---------------------------------------------------------------------------
# Shared building blocks: schedules (Eq. 4), corruption, ramp weight (Eq. 8)
# ---------------------------------------------------------------------------


def sample_flow_schedule(
    B: int, T: int, k_max: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Schedule for the flow-matching (empirical) portion of a batch: always the
    finest step size d_min = 1/k_max, tau uniform on its grid.

    Returns (d, step_idx, tau, signal_idx), each (B, T).
    """
    emax = _log2_int(k_max)
    step_idx = torch.full((B, T), emax, device=device, dtype=torch.long)
    d = torch.full((B, T), 1.0 / k_max, device=device, dtype=torch.float32)
    j = torch.randint(0, k_max, (B, T), device=device, dtype=torch.long)
    return d, step_idx, j.float() / float(k_max), j


def sample_bootstrap_schedule(
    B: int, T: int, k_max: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Schedule for the bootstrap (self-distillation) portion, paper Eq. 4:
    ``d ~ 1/U({1, 2, ..., k_max/2})`` (excluding d_min), ``tau`` uniform on
    the grid reachable with that d.

    Returns (d, step_idx, tau, signal_idx), each (B, T).
    """
    emax = _log2_int(k_max)
    # step_idx in [0, emax): 0 -> d=1, 1 -> d=1/2, ..., emax-1 -> d=2/k_max
    step_idx = torch.randint(0, max(1, emax), (B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).float()
    K = (1 << step_idx).long()                   # grid points for this d
    j = torch.floor(torch.rand(B, T, device=device) * K.float()).long()
    j = j.clamp(max=K - 1)
    return d, step_idx, j.float() / K.float(), j * (k_max // K)


def _expand(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Append trailing singleton dims so ``t`` broadcasts against ``like``."""
    return t.reshape(t.shape + (1,) * (like.ndim - t.ndim))


def corrupt_representations(clean: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Corrupt to SIGNAL level ``tau``: ``(1 - tau) * z0 + tau * clean`` with
    ``z0 ~ N(0, I)``. ``tau``'s shape must be a leading prefix of ``clean``'s.
    """
    z0 = torch.randn_like(clean)
    tau = _expand(tau, clean)
    return (1.0 - tau) * z0 + tau * clean


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Loss weight w(tau) = 0.9 tau + 0.1 (paper Eq. 8): emphasize high
    signal levels, where the learning signal is strongest."""
    return 0.9 * tau + 0.1


# --- velocity-space steps of the bootstrap teacher (Eq. 7) -----------------


def velocity(x_pred: torch.Tensor, z_tilde: torch.Tensor,
             tau: torch.Tensor) -> torch.Tensor:
    """x-prediction -> velocity: ``v = (x_hat - z_tilde) / (1 - tau)``."""
    return (x_pred - z_tilde) / _expand((1.0 - tau).clamp_min(1e-6), x_pred)


def advance(z_tilde: torch.Tensor, v: torch.Tensor,
            dt: torch.Tensor) -> torch.Tensor:
    """Euler step along a velocity: ``z_tilde + v * dt``."""
    return z_tilde + v * _expand(dt, v)


def signal_after(signal_idx: torch.Tensor, dt: torch.Tensor,
                 k_max: int) -> torch.Tensor:
    """Discrete signal index after advancing time by ``dt``."""
    return (signal_idx + (k_max * dt).long()).clamp(max=k_max)


# ---------------------------------------------------------------------------
# Objective 1: shortcut forcing — the paper's Eq. 7 (baseline)
# ---------------------------------------------------------------------------


def shortcut_forcing_loss(
    dynamics: DynamicsModel,
    *,
    z1: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    k_max: int,
    bootstrap_fraction: float = 0.25,
    agent_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Flow term (every frame, d = d_min) + bootstrap term (a batch fraction at
    coarser d; teacher = two half-steps in velocity space, no-grad), both
    ramp-weighted. See the module docstring for when NOT to use this.

    Args:
        z1:      (B, T, Nz, Dz) clean packed representations.
        actions: (B, T, D_a+1) aligned actions (or None for unconditional).

    Returns:
        (scalar loss, aux dict of detached diagnostics).
    """
    device = z1.device
    B, T = z1.shape[:2]
    B_boot = max(0, min(B - 1, int(round(bootstrap_fraction * B))))
    B_flow = B - B_boot

    # -- corrupt: flow rows on the finest grid, bootstrap rows on coarser d --
    d_f, step_f, tau_f, sig_f = sample_flow_schedule(B_flow, T, k_max, device)
    z_tilde_f = corrupt_representations(z1[:B_flow], tau_f)
    if B_boot > 0:
        d_b, step_b, tau_b, sig_b = sample_bootstrap_schedule(B_boot, T, k_max, device)
        z_tilde_b = corrupt_representations(z1[B_flow:], tau_b)
    else:
        d_b = torch.zeros(0, T, device=device)
        step_b = torch.zeros(0, T, device=device, dtype=torch.long)
        tau_b = torch.zeros(0, T, device=device)
        sig_b = torch.zeros(0, T, device=device, dtype=torch.long)
        z_tilde_b = torch.zeros(0, T, *z1.shape[2:], device=device, dtype=z1.dtype)

    # -- one forward over the whole batch --
    step_full = torch.cat([step_f, step_b], dim=0)
    tau_full = torch.cat([tau_f, tau_b], dim=0)
    sig_full = torch.cat([sig_f, sig_b], dim=0)
    z_tilde_full = torch.cat([z_tilde_f, z_tilde_b], dim=0)
    z1_hat_full, _ = dynamics(actions, step_full, sig_full, z_tilde_full,
                              agent_tokens=agent_tokens)

    # -- flow loss: x-prediction against the clean target --
    w_f = ramp_weight(tau_f)
    flow_per = (z1_hat_full[:B_flow].float() - z1[:B_flow].float()) \
        .pow(2).mean(dim=(2, 3))
    loss_flow = (flow_per * w_f).mean()

    # -- bootstrap loss: distill two half-steps into one (teacher no-grad) --
    loss_boot = torch.tensor(0.0, device=device)
    boot_mse = torch.tensor(0.0, device=device)
    if B_boot > 0:
        z_tilde_boot = z_tilde_full[B_flow:]
        actions_boot = actions[B_flow:] if actions is not None else None
        agent_boot = agent_tokens[B_flow:] if agent_tokens is not None else None
        d_half = d_b / 2.0
        step_half = step_b + 1                   # one finer level
        with torch.no_grad():
            x1, _ = dynamics(actions_boot, step_half, sig_b, z_tilde_boot,
                             agent_tokens=agent_boot)
            b_prime = velocity(x1.float(), z_tilde_boot.float(), tau_b)
            z_prime = advance(z_tilde_boot.float(), b_prime, d_half)
            x2, _ = dynamics(actions_boot, step_half,
                             signal_after(sig_b, d_half, k_max),
                             z_prime.to(z_tilde_boot.dtype),
                             agent_tokens=agent_boot)
            b_double = velocity(x2.float(), z_prime, tau_b + d_half)
            v_target = (b_prime + b_double) / 2.0
        v_hat = velocity(z1_hat_full[B_flow:].float(), z_tilde_boot.float(), tau_b)
        boot_per = (1.0 - tau_b).pow(2) * (v_hat - v_target).pow(2).mean(dim=(2, 3))
        loss_boot = (boot_per * ramp_weight(tau_b)).mean()
        boot_mse = boot_per.mean().detach()

    loss = (loss_flow * B_flow + loss_boot * B_boot) / B
    aux = {"flow_mse": flow_per.mean().detach(), "boot_mse": boot_mse,
           "loss_flow": loss_flow.detach(), "loss_boot": loss_boot.detach(),
           "tau_mean": tau_full.mean().detach()}
    return loss, aux


# ---------------------------------------------------------------------------
# Objective 2: clean context — the production fix
# ---------------------------------------------------------------------------


def clean_context_loss(
    dynamics: DynamicsModel,
    z1: torch.Tensor,
    actions: torch.Tensor,
    *,
    p: int,
    k_max: int,
    K: int = 4,
    tau_ctx: float = 0.1,
    ctx_noise_min: Optional[float] = None,
    ctx_noise_max: Optional[float] = None,
    do_sched: bool = False,
    proprio: Optional[torch.Tensor] = None,
    bootstrap_frac: float = 0.0,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, float]]:
    """
    Predict frame ``p`` of ``z1`` from its (nearly) clean context.

    Args:
        z1:       (B, T, Nz, Dz) clean packed latents, T > p.
        actions:  (B, T, D_a+1) ALIGNED actions (:func:`align_actions`).
        p:        Target frame index (0 = no context: image batch).
        proprio:  (B, T, d_p) joint proprio stream, iff the model has one —
                  context corrupted like the latents, target noised with the
                  SAME tau and denoised jointly.
        (others:  see the module docstring.)

    Returns:
        terms: {"flow": t, "proprio": t | None, "bootstrap": t | None} for the
               trainer's LossCombiner.
        aux:   float diagnostics (flow_mse, boot_mse).
    """
    B = z1.shape[0]
    device = z1.device
    emax = _log2_int(k_max)
    has_prop = proprio is not None
    B_boot = min(B - 1, int(round(bootstrap_frac * B))) if bootstrap_frac > 0 else 0
    B_flow = B - B_boot

    # ---- context: (optionally) one own-generation, then light noise -------
    ctx = z1[:, :p].clone()
    prop_ctx = proprio[:, :p].clone() if has_prop else None
    if do_sched and p >= 2:
        # scheduled sampling: the LAST context frame becomes the model's own
        # 1-step generation, so training sees real rollout error
        if has_prop:
            z_gen, prop_gen, _ = sample_one_timestep(
                dynamics, past_packed=z1[:, :p - 1], k_max=k_max, K=K,
                actions=actions[:, :p], tau_ctx=tau_ctx,
                past_proprio=proprio[:, :p - 1], cache=None)
            prop_ctx[:, p - 1] = prop_gen.clamp(-1, 1)
        else:
            z_gen, _ = sample_one_timestep(
                dynamics, past_packed=z1[:, :p - 1], k_max=k_max, K=K,
                actions=actions[:, :p], tau_ctx=tau_ctx, cache=None)
        ctx[:, p - 1] = z_gen.clamp(-1, 1)

    # tc is a NOISE fraction (not a signal level): fixed tau_ctx, or drawn
    # per frame per sample from the [min, max] band for rollout robustness
    lo = tau_ctx if ctx_noise_min is None else ctx_noise_min
    hi = ctx_noise_max if (ctx_noise_max is not None and ctx_noise_max > lo) else None
    if hi is not None and p > 0:
        tc = torch.empty(B, p, 1, 1, device=device).uniform_(lo, hi)
    else:
        tc = lo
    ctx_noised = (1 - tc) * ctx + tc * torch.randn_like(ctx)

    # ---- target: noised across the full signal grid -----------------------
    # flow rows at the finest d; optional bootstrap rows at coarser d
    tgt = z1[:, p]
    j_f = torch.randint(0, k_max, (B_flow,), device=device)
    tau_f = j_f.float() / k_max
    step_tgt = torch.full((B,), emax, device=device, dtype=torch.long)
    sig_tgt = torch.empty(B, device=device, dtype=torch.long)
    sig_tgt[:B_flow] = j_f
    if B_boot > 0:
        d_b, step_b, tau_b, sig_b = sample_bootstrap_schedule(B_boot, 1, k_max, device)
        d_b, step_b, tau_b, sig_b = d_b[:, 0], step_b[:, 0], tau_b[:, 0], sig_b[:, 0]
        step_tgt[B_flow:] = step_b
        sig_tgt[B_flow:] = sig_b
        tau = torch.cat([tau_f, tau_b])
    else:
        tau = tau_f
    ztilde = corrupt_representations(tgt, tau)

    # ---- assemble the window and run the model ----------------------------
    z_in = torch.cat([ctx_noised, ztilde[:, None]], dim=1)      # (B, p+1, ...)
    step_idx = torch.full((B, p + 1), emax, device=device, dtype=torch.long)
    step_idx[:, p] = step_tgt
    ctx_sig = min(int(round((1 - tau_ctx) * k_max)), k_max)
    signal_idx = torch.full((B, p + 1), ctx_sig, device=device, dtype=torch.long)
    signal_idx[:, p] = sig_tgt
    act = actions[:, :p + 1]

    prop_in = prop_tilde = None
    if has_prop:
        # proprio rides along: same context noise, same target tau
        tc_prop = tc[..., 0] if torch.is_tensor(tc) else tc
        prop_ctx_noised = (1 - tc_prop) * prop_ctx \
            + tc_prop * torch.randn_like(prop_ctx)
        prop_tilde = corrupt_representations(proprio[:, p], tau)
        prop_in = torch.cat([prop_ctx_noised, prop_tilde[:, None]], dim=1)
        z1_hat, prop_hat, _ = dynamics(act, step_idx, signal_idx, z_in,
                                       proprio_noisy=prop_in)
    else:
        z1_hat, _ = dynamics(act, step_idx, signal_idx, z_in)

    # ---- flow loss (x-prediction) on the d_min rows -----------------------
    flow_per = (z1_hat[:B_flow, p].float() - tgt[:B_flow].float()) ** 2
    terms: Dict[str, Optional[torch.Tensor]] = {
        "flow": (_expand(ramp_weight(tau_f), flow_per) * flow_per).mean(),
        "proprio": None, "bootstrap": None}
    aux = {"flow_mse": float(flow_per.mean().detach()), "boot_mse": 0.0}

    # ---- bootstrap term on the target slot (teacher = two half-steps) -----
    if B_boot > 0:
        zt_b = ztilde[B_flow:]
        with torch.no_grad():
            st1 = step_idx[B_flow:].clone()
            st1[:, p] = (step_b + 1).clamp(max=emax)            # one finer level
            si1 = signal_idx[B_flow:]
            act_b = act[B_flow:]
            pin1 = prop_in[B_flow:] if has_prop else None
            out1 = dynamics(act_b, st1, si1, z_in[B_flow:],
                            **({"proprio_noisy": pin1} if has_prop else {}))
            b1 = velocity(out1[0][:, p].float(), zt_b.float(), tau_b)
            z_half = advance(zt_b.float(), b1, d_b / 2.0)
            si2 = si1.clone()
            si2[:, p] = signal_after(sig_b, d_b / 2.0, k_max)
            zin2 = z_in[B_flow:].clone()
            zin2[:, p] = z_half.to(z_in.dtype)
            pin2 = None
            if has_prop:
                bp1 = velocity(out1[1][:, p].float(),
                               prop_tilde[B_flow:].float(), tau_b)
                prop_half = advance(prop_tilde[B_flow:].float(), bp1, d_b / 2.0)
                pin2 = prop_in[B_flow:].clone()
                pin2[:, p] = prop_half.to(prop_in.dtype)
            out2 = dynamics(act_b, st1, si2, zin2,
                            **({"proprio_noisy": pin2} if has_prop else {}))
            b2 = velocity(out2[0][:, p].float(), z_half, tau_b + d_b / 2.0)
            v_target = (b1 + b2) / 2.0
        v_hat = velocity(z1_hat[B_flow:, p].float(), zt_b.float(), tau_b)
        boot_per = _expand((1.0 - tau_b).pow(2), v_hat) * (v_hat - v_target) ** 2
        terms["bootstrap"] = (_expand(ramp_weight(tau_b), boot_per) * boot_per).mean()
        aux["boot_mse"] = float(boot_per.mean().detach())

    # ---- proprio loss (x-prediction, all rows) ----------------------------
    if has_prop:
        prop_per = (prop_hat[:, p].float() - proprio[:, p].float()) ** 2
        terms["proprio"] = (_expand(ramp_weight(tau), prop_per) * prop_per).mean()

    return terms, aux
