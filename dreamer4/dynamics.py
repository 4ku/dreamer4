"""
Dynamics model with Shortcut Forcing for Dreamer 4.

Predicts future latent representations given actions using the shortcut
forcing objective — a combination of flow matching, shortcut models, and
diffusion forcing (Paper Section 3.2, Eq. 6-8).

Key design choices from the paper:
  - X-prediction (not V-prediction) to prevent error accumulation
  - Ramp loss weight: w(tau) = 0.9*tau + 0.1 (Eq. 8)
  - Bootstrap loss distills two half-steps into one full step
  - At inference: K=4 sampling steps, context corruption tau_ctx=0.1

Architecture (Paper Figure 2b):
  The dynamics model operates on the interleaved sequence:
    [action_token, shortcut_signal_token, spatial_tokens..., register_tokens..., (agent_tokens...)]
  per time step, using the block-causal transformer backbone.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer4.modality import Modality, TokenLayout
from dreamer4.transformer import BlockCausalTransformer


# ---------------------------------------------------------------------------
# Action Encoder
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    """
    Encodes continuous actions into a single token per time step.

    When actions are None (unlabeled video pretraining), emits a learned
    base embedding. Otherwise projects actions through an MLP and adds
    the base embedding.

    Paper Section 3.2: "Continuous action components are linearly projected
    and categorical or binary components use an embedding lookup."

    Args:
        d_model:     Model dimension for the output token.
        action_dim:  Dimensionality of the continuous action vector.
        hidden_mult: Hidden layer multiplier for the MLP.

    Shape:
        Input:  (B, T, action_dim) or None
        Output: (B, T, 1, d_model)
    """

    def __init__(self, d_model: int, action_dim: int, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        hidden = int(d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.base, std=0.02)

        self.fc1 = nn.Linear(action_dim, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor] = None,
        *,
        batch_time_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            actions: (B, T, action_dim) or None for unlabeled data.
            batch_time_shape: Required when actions is None to infer (B, T).

        Returns:
            (B, T, 1, d_model) action token.
        """
        if actions is None:
            assert batch_time_shape is not None, "batch_time_shape required when actions is None"
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            x = actions.clamp(-1, 1)
            out = self.fc2(F.silu(self.fc1(x))) + self.base.view(1, 1, -1)

        return out.unsqueeze(2)  # (B, T, 1, D)


# ---------------------------------------------------------------------------
# Shortcut Signal Encoder
# ---------------------------------------------------------------------------

class ShortcutSignalEncoder(nn.Module):
    """
    Encodes discrete shortcut signal level (tau) and step size (d) into
    a single token per time step.

    Paper Section 3.2: "Since the signal level and step size are discrete,
    we encode each with a discrete embedding lookup and concatenate their
    channels."

    Args:
        d_model: Output dimension for the combined token.
        k_max:   Maximum number of sampling steps (must be power of 2).
                 Defines the finest step size d_min = 1/k_max.

    Shape:
        Input:  step_idx (B, T), signal_idx (B, T) — both long tensors
        Output: (B, T, 1, d_model)
    """

    def __init__(self, d_model: int, k_max: int):
        super().__init__()
        self.d_model = d_model
        self.k_max = k_max

        n_step_bins = int(math.log2(k_max)) + 1
        half_d = d_model // 2

        self.step_embed = nn.Embedding(n_step_bins, half_d)
        self.signal_embed = nn.Embedding(k_max + 1, d_model - half_d)

        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        step_idx: torch.Tensor,
        signal_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            step_idx:   (B, T) long — log2-index of the step size d.
                        0 -> d=1, 1 -> d=1/2, ..., log2(k_max) -> d=d_min.
            signal_idx: (B, T) long — discrete signal level index.
                        Maps to tau = signal_idx / k_max.

        Returns:
            (B, T, 1, d_model) shortcut signal token.
        """
        s_emb = self.step_embed(step_idx)      # (B, T, half_d)
        t_emb = self.signal_embed(signal_idx)  # (B, T, d_model - half_d)
        combined = torch.cat([s_emb, t_emb], dim=-1)  # (B, T, d_model)
        return self.proj(combined).unsqueeze(2)        # (B, T, 1, d_model)


# ---------------------------------------------------------------------------
# Dynamics Model
# ---------------------------------------------------------------------------

class DynamicsModel(nn.Module):
    """
    Dynamics model for Dreamer 4 world model.

    Operates on the interleaved per-timestep sequence:
        [action(1), shortcut_signal(1), spatial(n_spatial), register(n_register), (agent(n_agent))]

    Uses the block-causal transformer backbone with space_mode controlling
    the attention pattern (wm_agent_isolated for pretraining, wm_agent for
    agent finetuning).

    The model predicts clean representations z1 (x-prediction) via a
    zero-initialized flow head, which prevents error accumulation in long
    autoregressive rollouts.

    Args:
        d_model:     Transformer hidden dimension.
        d_spatial:   Dimension of packed spatial tokens (d_bottleneck * k).
        n_spatial:   Number of spatial tokens per time step.
        n_register:  Number of learnable register tokens.
        n_agent:     Number of agent tokens (0 during pretraining).
        n_heads:     Number of query attention heads.
        depth:       Number of transformer layers.
        k_max:       Maximum number of sampling steps (power of 2).
        action_dim:  Dimensionality of the action space.
        n_kv_heads:  KV heads for GQA (default = n_heads).
        mlp_ratio:   MLP hidden size multiplier.
        time_every:  Apply time attention every N layers.
        dropout:     Dropout rate.
        use_qk_norm: Use QKNorm in attention.
        logit_cap:   Attention logit soft capping.
        space_mode:  Attention mode ("wm_agent_isolated" or "wm_agent").
        max_T:       Maximum time steps for RoPE cache.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int = 4,
        n_agent: int = 0,
        n_heads: int = 4,
        depth: int = 8,
        k_max: int = 8,
        action_dim: int = 16,
        n_kv_heads: int | None = None,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        space_mode: str = "wm_agent",
        max_T: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_spatial = d_spatial
        self.n_spatial = n_spatial
        self.n_register = n_register
        self.n_agent = n_agent
        self.k_max = k_max

        self.action_encoder = ActionEncoder(
            d_model=d_model, action_dim=action_dim,
        )
        self.shortcut_encoder = ShortcutSignalEncoder(
            d_model=d_model, k_max=k_max,
        )

        self.spatial_proj = nn.Linear(d_spatial, d_model)

        self.register_tokens = nn.Parameter(torch.empty(n_register, d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        segments: list[tuple[Modality, int]] = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SPATIAL, n_spatial),
            (Modality.REGISTER, n_register),
        ]
        if n_agent > 0:
            segments.append((Modality.AGENT, n_agent))

        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))

        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            layout=self.layout,
            space_mode=space_mode,
            n_kv_heads=n_kv_heads,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            logit_cap=logit_cap,
            latents_only_time=False,
            max_T=max_T,
        )

        self.flow_head = nn.Linear(d_model, d_spatial)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],
        step_idx: torch.Tensor,
        signal_idx: torch.Tensor,
        z_noisy: torch.Tensor,
        *,
        agent_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            actions:      (B, T, action_dim) or None for unlabeled data.
            step_idx:     (B, T) long — step size index.
            signal_idx:   (B, T) long — signal level index.
            z_noisy:      (B, T, n_spatial, d_spatial) corrupted representations.
            agent_tokens: (B, T, n_agent, d_model) or None.

        Returns:
            z1_hat:        (B, T, n_spatial, d_spatial) predicted clean representations.
            agent_out:     (B, T, n_agent, d_model) or None if n_agent == 0.
        """
        B, T = z_noisy.shape[:2]

        act_tok = self.action_encoder(
            actions, batch_time_shape=(B, T),
        )  # (B, T, 1, D)

        sc_tok = self.shortcut_encoder(
            step_idx, signal_idx,
        )  # (B, T, 1, D)

        spatial_tok = self.spatial_proj(z_noisy)  # (B, T, n_spatial, D)

        reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        parts = [act_tok, sc_tok, spatial_tok, reg]

        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros(
                    B, T, self.n_agent, self.d_model,
                    device=z_noisy.device, dtype=z_noisy.dtype,
                )
            parts.append(agent_tokens)

        tokens = torch.cat(parts, dim=2)  # (B, T, S, D)
        x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]
        z1_hat = self.flow_head(spatial_out)  # (B, T, n_spatial, d_spatial)

        agent_out = None
        if self.n_agent > 0:
            agent_out = x[:, :, self.agent_slice, :]

        return z1_hat, agent_out


# ---------------------------------------------------------------------------
# Schedule Sampling (Eq. 4)
# ---------------------------------------------------------------------------

def _log2_int(k_max: int) -> int:
    """Compute log2(k_max), asserting k_max is a power of 2."""
    e = int(round(math.log2(k_max)))
    assert (1 << e) == k_max, f"k_max={k_max} must be a power of 2"
    return e


def sample_flow_schedule(
    B: int,
    T: int,
    k_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample schedule for the flow matching (empirical) portion of the batch.

    Uses the finest step size d_min = 1/k_max and uniform tau.

    Returns:
        d:          (B, T) float — step size (always d_min).
        step_idx:   (B, T) long  — step index (always emax).
        tau:        (B, T) float — signal level in [0, 1).
        signal_idx: (B, T) long  — discrete signal index.
    """
    emax = _log2_int(k_max)

    step_idx = torch.full((B, T), emax, device=device, dtype=torch.long)
    d = torch.full((B, T), 1.0 / k_max, device=device, dtype=torch.float32)

    j = torch.randint(0, k_max, (B, T), device=device, dtype=torch.long)
    tau = j.float() / float(k_max)
    signal_idx = j

    return d, step_idx, tau, signal_idx


def sample_bootstrap_schedule(
    B: int,
    T: int,
    k_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample schedule for the bootstrap (self-consistency) portion.

    Step size d is sampled uniformly as a power of two from
    {1, 1/2, 1/4, ..., 1/(k_max/2)}, excluding d_min.
    Tau is sampled on the grid reachable by the current d.

    Paper Eq. 4:
        d ~ 1/U({1,2,...,K_max/2})
        tau ~ U({0, d, 2d, ..., 1-d})

    Returns:
        d:          (B, T) float
        step_idx:   (B, T) long
        tau:        (B, T) float
        signal_idx: (B, T) long
    """
    emax = _log2_int(k_max)

    # step_idx in [0, emax): 0 -> d=1, 1 -> d=1/2, ..., emax-1 -> d=2/k_max
    step_idx = torch.randint(0, max(1, emax), (B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).float()  # (B, T)

    K = (1 << step_idx).long()  # number of grid points for this d
    j = torch.floor(torch.rand(B, T, device=device) * K.float()).long()
    j = j.clamp(max=K - 1)
    tau = j.float() / K.float()

    scale = k_max // K
    signal_idx = j * scale

    return d, step_idx, tau, signal_idx


# ---------------------------------------------------------------------------
# Corruption & Weight
# ---------------------------------------------------------------------------

def corrupt_representations(
    z1: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Corrupt clean representations via linear interpolation with noise.

    z_tilde = (1 - tau) * z0 + tau * z1,  where z0 ~ N(0, I)

    Args:
        z1:  (B, T, Nz, Dz) clean representations.
        tau: (B, T) signal levels in [0, 1].

    Returns:
        z_tilde: (B, T, Nz, Dz) corrupted representations.
        z0:      (B, T, Nz, Dz) the noise sample used.
    """
    z0 = torch.randn_like(z1)
    tau_4d = tau[..., None, None]  # (B, T, 1, 1)
    z_tilde = (1.0 - tau_4d) * z0 + tau_4d * z1
    return z_tilde, z0


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """
    Ramp loss weight from paper Eq. 8: w(tau) = 0.9 * tau + 0.1

    Focuses model capacity on higher signal levels where learning signal
    is strongest. At tau=0 (pure noise), weight is 0.1; at tau=1, weight is 1.0.

    Args:
        tau: (...) signal levels in [0, 1].

    Returns:
        (...) weights in [0.1, 1.0].
    """
    return 0.9 * tau + 0.1


# ---------------------------------------------------------------------------
# Shortcut Forcing Loss (Eq. 7)
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
    Combined shortcut forcing loss: flow term + bootstrap term.

    Paper Section 3.2, Eq. 7:
      - Flow term (d=d_min):
            ||f(z_tilde, tau, d, a) - z1||^2
      - Bootstrap term (d > d_min):
            (1-tau)^2 * ||v_hat - sg(b' + b'')/2||^2
        where b', b'' are velocity predictions from two half-steps.
      - Both terms are weighted by ramp_weight(tau).

    Args:
        dynamics:            The DynamicsModel.
        z1:                  (B, T, Nz, Dz) clean packed representations.
        actions:             (B, T, A) or None.
        k_max:               Maximum sampling steps.
        bootstrap_fraction:  Fraction of the batch for bootstrap targets.
        agent_tokens:        (B, T, n_agent, D) or None.

    Returns:
        loss: Scalar combined loss.
        aux:  Dict of diagnostic tensors (all detached).
    """
    device = z1.device
    B, T = z1.shape[:2]
    emax = _log2_int(k_max)

    B_boot = max(0, min(B - 1, int(round(bootstrap_fraction * B))))
    B_flow = B - B_boot

    # --- Flow (empirical) portion ---
    d_f, step_f, tau_f, sig_f = sample_flow_schedule(B_flow, T, k_max, device)
    z_tilde_f, _ = corrupt_representations(z1[:B_flow], tau_f)

    # --- Bootstrap portion ---
    if B_boot > 0:
        d_b, step_b, tau_b, sig_b = sample_bootstrap_schedule(B_boot, T, k_max, device)
        z_tilde_b, _ = corrupt_representations(z1[B_flow:], tau_b)
    else:
        d_b = torch.zeros(0, T, device=device)
        step_b = torch.zeros(0, T, device=device, dtype=torch.long)
        tau_b = torch.zeros(0, T, device=device)
        sig_b = torch.zeros(0, T, device=device, dtype=torch.long)
        z_tilde_b = torch.zeros(0, T, *z1.shape[2:], device=device, dtype=z1.dtype)

    # Concatenate full batch for one forward pass
    d_full = torch.cat([d_f, d_b], dim=0)
    step_full = torch.cat([step_f, step_b], dim=0)
    tau_full = torch.cat([tau_f, tau_b], dim=0)
    sig_full = torch.cat([sig_f, sig_b], dim=0)
    z_tilde_full = torch.cat([z_tilde_f, z_tilde_b], dim=0)

    z1_hat_full, _ = dynamics(
        actions, step_full, sig_full, z_tilde_full,
        agent_tokens=agent_tokens,
    )

    # --- Flow loss: ||z1_hat - z1||^2 weighted by ramp ---
    z1_hat_flow = z1_hat_full[:B_flow]
    w_f = ramp_weight(tau_f)  # (B_flow, T)
    flow_per = (z1_hat_flow.float() - z1[:B_flow].float()).pow(2).mean(dim=(2, 3))
    loss_flow = (flow_per * w_f).mean()

    # --- Bootstrap loss (Eq. 7) ---
    loss_boot = torch.tensor(0.0, device=device)
    boot_mse = torch.tensor(0.0, device=device)

    if B_boot > 0:
        z1_hat_boot = z1_hat_full[B_flow:]
        z_tilde_boot = z_tilde_full[B_flow:]
        actions_boot = actions[B_flow:] if actions is not None else None
        agent_boot = agent_tokens[B_flow:] if agent_tokens is not None else None

        d_half = d_b / 2.0
        step_half = step_b + 1  # one finer level

        tau_plus = tau_b + d_half
        sig_plus = sig_b + (torch.tensor(k_max, device=device).float() * d_half).long()
        sig_plus = sig_plus.clamp(max=k_max)

        # First half-step: predict z1 from z_tilde at (tau, d/2)
        with torch.no_grad():
            z1_h1, _ = dynamics(
                actions_boot, step_half, sig_b, z_tilde_boot,
                agent_tokens=agent_boot,
            )
            b_prime = (z1_h1.float() - z_tilde_boot.float()) / (1.0 - tau_b).clamp_min(1e-6)[..., None, None]
            z_prime = z_tilde_boot.float() + b_prime * d_half[..., None, None]

            # Second half-step: predict z1 from z_prime at (tau + d/2, d/2)
            z1_h2, _ = dynamics(
                actions_boot, step_half, sig_plus, z_prime.to(z_tilde_boot.dtype),
                agent_tokens=agent_boot,
            )
            b_double = (z1_h2.float() - z_prime) / (1.0 - tau_plus).clamp_min(1e-6)[..., None, None]

            v_target = ((b_prime + b_double) / 2.0)

        # Convert model prediction to v-space
        v_hat = (z1_hat_boot.float() - z_tilde_boot.float()) / (1.0 - tau_b).clamp_min(1e-6)[..., None, None]

        w_b = ramp_weight(tau_b)
        boot_per = (1.0 - tau_b).pow(2) * (v_hat - v_target).pow(2).mean(dim=(2, 3))
        loss_boot = (boot_per * w_b).mean()
        boot_mse = boot_per.mean().detach()

    loss = (loss_flow * B_flow + loss_boot * B_boot) / B

    aux = {
        "flow_mse": flow_per.mean().detach(),
        "boot_mse": boot_mse,
        "loss_flow": loss_flow.detach(),
        "loss_boot": loss_boot.detach(),
        "tau_mean": tau_full.mean().detach(),
    }
    return loss, aux


# ---------------------------------------------------------------------------
# Sampling / Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one_timestep(
    dynamics: DynamicsModel,
    *,
    past_packed: torch.Tensor,
    k_max: int,
    K: int = 4,
    actions: Optional[torch.Tensor] = None,
    tau_ctx: float = 0.1,
    agent_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate one new frame by K-step shortcut denoising.

    Paper Section 3.2: "We sample autoregressively in time and generate the
    representations of each frame using the shortcut model with K=4 sample
    steps. We slightly corrupt the past inputs to signal level tau_ctx=0.1."

    Args:
        dynamics:     The DynamicsModel (should be in eval mode).
        past_packed:  (B, t, n_spatial, d_spatial) context frames.
        k_max:        Maximum sampling steps.
        K:            Number of denoising steps for the new frame.
        actions:      (B, t+1, action_dim) or None. Must include the action
                      for the frame being generated.
        tau_ctx:      Signal level for corrupting context (default 0.1).
        agent_tokens: (B, t+1, n_agent, d_model) or None.  When provided,
                      passed to the dynamics forward call and the agent_out
                      for the *last* timestep of the final denoising step
                      is returned.

    Returns:
        z_new:     (B, n_spatial, d_spatial) generated frame.
        agent_out: (B, n_agent, d_model) or None.  Agent embeddings for the
                   newly generated timestep (from the last denoising step).
    """
    device = past_packed.device
    dtype = past_packed.dtype
    B, t_ctx = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]
    emax = _log2_int(k_max)

    signal_level = 1.0 - tau_ctx
    if tau_ctx > 0 and t_ctx > 0:
        noise = torch.randn_like(past_packed)
        past_corrupted = tau_ctx * noise + signal_level * past_packed
    else:
        past_corrupted = past_packed

    d = 1.0 / K
    step_e = int(round(math.log2(K)))

    z = torch.randn(B, 1, n_spatial, d_spatial, device=device, dtype=dtype)

    ctx_signal_val = int(round(signal_level * k_max))
    ctx_signal_val = min(ctx_signal_val, k_max)

    last_agent_out: Optional[torch.Tensor] = None

    for i in range(K):
        tau_i = i * d
        sig_i = int(round(tau_i * k_max))
        sig_i = min(sig_i, k_max)

        packed_seq = torch.cat([past_corrupted, z], dim=1)  # (B, t+1, Nz, Dz)
        T_total = packed_seq.shape[1]

        step_idxs = torch.full((B, T_total), emax, device=device, dtype=torch.long)
        step_idxs[:, -1] = step_e

        signal_idxs = torch.full((B, T_total), ctx_signal_val, device=device, dtype=torch.long)
        signal_idxs[:, -1] = sig_i

        actions_in = actions[:, :T_total] if actions is not None else None
        agent_in = agent_tokens[:, :T_total] if agent_tokens is not None else None

        z1_hat, a_out = dynamics(
            actions_in, step_idxs, signal_idxs, packed_seq,
            agent_tokens=agent_in,
        )
        z1_hat_new = z1_hat[:, -1:, :, :]  # (B, 1, Nz, Dz)

        if a_out is not None:
            last_agent_out = a_out[:, -1]  # (B, n_agent, d_model)

        denom = max(1e-4, 1.0 - tau_i)
        velocity = (z1_hat_new.float() - z.float()) / denom
        z = (z.float() + velocity * d).to(dtype)

    return z[:, 0], last_agent_out


@torch.no_grad()
def sample_sequence(
    dynamics: DynamicsModel,
    *,
    context: torch.Tensor,
    horizon: int,
    k_max: int,
    K: int = 4,
    actions: Optional[torch.Tensor] = None,
    tau_ctx: float = 0.1,
    agent_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
    """
    Autoregressively generate a sequence of frames.

    Args:
        dynamics:     The DynamicsModel (should be in eval mode).
        context:      (B, t_ctx, n_spatial, d_spatial) context frames.
        horizon:      Number of frames to generate.
        k_max:        Maximum sampling steps.
        K:            Denoising steps per frame.
        actions:      (B, t_ctx + horizon, action_dim) or None.
        tau_ctx:      Context corruption level.
        agent_tokens: (B, t_ctx + horizon, n_agent, d_model) or None.
                      When provided, agent embeddings from each generated
                      timestep are collected and returned.

    Returns:
        frames:     (B, t_ctx + horizon, n_spatial, d_spatial) full sequence
                    (context + generated).
        agent_outs: List of ``horizon`` tensors each (B, n_agent, d_model),
                    one per generated timestep.  None when agent_tokens is
                    not provided.
    """
    B = context.shape[0]
    t_ctx = context.shape[1]

    frames = [context[:, t] for t in range(t_ctx)]
    agent_outs: list[torch.Tensor] = []

    for h in range(horizon):
        past = torch.stack(frames, dim=1)  # (B, t, Nz, Dz)

        z_next, a_out = sample_one_timestep(
            dynamics,
            past_packed=past,
            k_max=k_max,
            K=K,
            actions=actions,
            tau_ctx=tau_ctx,
            agent_tokens=agent_tokens,
        )
        frames.append(z_next)
        if a_out is not None:
            agent_outs.append(a_out)

    seq = torch.stack(frames, dim=1)
    return seq, agent_outs if agent_outs else None
