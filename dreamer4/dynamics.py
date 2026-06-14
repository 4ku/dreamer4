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
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from dreamer4.transformer.modality import Modality, TokenLayout
from dreamer4.transformer import BlockCausalTransformer
from dreamer4.transformer.kv_cache import KVCache


ActionInput = Union[torch.Tensor, Dict[str, torch.Tensor], None]


# ---------------------------------------------------------------------------
# Action Encoder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionComponent:
    """
    One component of a composite action (paper Section 3.2).

    Fields:
        name: Identifier (key in the dict passed to ActionEncoder.forward).
        kind: "continuous" | "categorical" | "binary".
        dim:  For "continuous" and "binary", the vector length;
              for "categorical", the number of classes.
    """
    name: str
    kind: str
    dim: int


class ActionEncoder(nn.Module):
    """
    Encodes actions into a single action token per time step.

    Two input modes are supported:

    1. **Flat continuous vector** — pass ``action_dim`` at construction and
       feed ``(B, T, action_dim)`` float tensors. Intended for toy envs and
       simple robotic setups (e.g. 14-D joint + gripper commands).

    2. **Dict of components** — pass ``action_components`` and feed a
       ``{name: tensor}`` dict. Each component is encoded separately per
       its ``kind`` (paper Section 3.2):
         - ``continuous``:  linear projection ``(B, T, dim) -> (B, T, d_model)``.
         - ``categorical``: embedding lookup on ``(B, T)`` long class ids.
         - ``binary``:      linear projection (no bias) on a ``(B, T, dim)``
                            0/1 vector. Algebraically equivalent to summing
                            learned per-bit embeddings for the active bits.

    Component tokens are summed and a learned base embedding is added. When
    ``actions is None`` (unlabeled video), only the base embedding is emitted
    — this is the fallback path that enables mixed labeled/unlabeled training.

    Args:
        d_model:           Model dimension for the output token.
        action_dim:        Dim of a single flat continuous vector. Mutually
                           exclusive with ``action_components``.
        action_components: Sequence of ``ActionComponent`` specs for the
                           multi-component path. Mutually exclusive with
                           ``action_dim``.

    Shape:
        Input:
            None                          — emit base embedding; (B, T) from kwarg.
            Tensor (B, T, action_dim)     — flat continuous path.
            dict {name: Tensor}           — multi-component path.
        Output:
            (B, T, 1, d_model) action token.
    """

    _FLAT_NAME = "__flat__"

    def __init__(
        self,
        d_model: int,
        *,
        action_dim: Optional[int] = None,
        action_components: Optional[Sequence[ActionComponent]] = None,
    ):
        super().__init__()
        if (action_dim is None) == (action_components is None):
            raise ValueError(
                "Specify exactly one of action_dim or action_components"
            )

        self.d_model = d_model
        self.base = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.base, std=0.001)

        if action_dim is not None:
            self.components: Tuple[ActionComponent, ...] = (
                ActionComponent(name=self._FLAT_NAME, kind="continuous", dim=action_dim),
            )
        else:
            self.components = tuple(action_components)

        self.encoders = nn.ModuleDict()
        for comp in self.components:
            self.encoders[comp.name] = self._make_encoder(comp)

    def _make_encoder(self, comp: ActionComponent) -> nn.Module:
        if comp.kind == "continuous":
            # Linear: (B, T, comp.dim) -> (B, T, d_model)
            enc = nn.Linear(comp.dim, self.d_model, bias=True)
            nn.init.normal_(enc.weight, std=0.02)
            nn.init.zeros_(enc.bias)
            return enc
        if comp.kind == "categorical":
            # Embedding: (B, T) long -> (B, T, d_model)
            enc = nn.Embedding(comp.dim, self.d_model)
            nn.init.normal_(enc.weight, std=0.02)
            return enc
        if comp.kind == "binary":
            # Linear no-bias: (B, T, comp.dim) of 0/1 -> (B, T, d_model)
            enc = nn.Linear(comp.dim, self.d_model, bias=False)
            nn.init.normal_(enc.weight, std=0.02)
            return enc
        raise ValueError(
            f"Unknown component kind '{comp.kind}' for '{comp.name}'. "
            "Expected 'continuous', 'categorical', or 'binary'."
        )

    def _encode(self, comp: ActionComponent, value: torch.Tensor) -> torch.Tensor:
        # value: (B, T) long  for categorical
        #        (B, T, comp.dim) float/0-1  for continuous / binary
        # returns: (B, T, d_model)
        enc = self.encoders[comp.name]
        if comp.kind == "categorical":
            return enc(value.long())
        return enc(value.to(torch.float32) if value.dtype not in (torch.float32, torch.float16, torch.bfloat16) else value)

    def forward(
        self,
        actions: ActionInput = None,
        *,
        batch_time_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        # actions: None | Tensor (B, T, action_dim) | dict {name: Tensor}
        # returns: (B, T, 1, d_model)
        if actions is None:
            if batch_time_shape is None:
                raise ValueError("batch_time_shape required when actions is None")
            B, T = batch_time_shape
            # self.base:                  (d_model,)
            # self.base.view(1, 1, -1):   (1, 1, d_model)
            # expand(B, T, -1):           (B, T, d_model)
            out = self.base.view(1, 1, -1).expand(B, T, -1)
            return out.unsqueeze(2)  # (B, T, 1, d_model)

        if isinstance(actions, torch.Tensor):
            # Flat continuous path:
            # actions:   (B, T, action_dim)
            if len(self.components) != 1 or self.components[0].name != self._FLAT_NAME:
                raise TypeError(
                    "ActionEncoder was built with action_components; expected "
                    "a dict input keyed by component names."
                )
            # comp_tok:  (B, T, d_model)
            comp_tok = self._encode(self.components[0], actions)
        else:
            # Multi-component path:
            # actions:   {name: Tensor}, each (B, T) or (B, T, dim)
            if not isinstance(actions, dict):
                raise TypeError(
                    f"actions must be None, Tensor, or dict; got {type(actions).__name__}"
                )
            if len(self.components) == 1 and self.components[0].name == self._FLAT_NAME:
                raise TypeError(
                    "ActionEncoder was built with action_dim; expected a Tensor input."
                )
            comp_tok = None  # accumulator: (B, T, d_model) once any component is added
            for comp in self.components:
                if comp.name not in actions:
                    raise KeyError(f"missing action component '{comp.name}'")
                # t: (B, T, d_model)
                t = self._encode(comp, actions[comp.name])
                comp_tok = t if comp_tok is None else comp_tok + t

        # comp_tok:                 (B, T, d_model)
        # self.base.view(1, 1, -1): (1, 1, d_model)  — broadcasts
        # out:                      (B, T, d_model)
        out = comp_tok + self.base.view(1, 1, -1)
        return out.unsqueeze(2)  # (B, T, 1, d_model)


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
        d_model:            Transformer hidden dimension.
        d_spatial:          Dimension of packed spatial tokens (d_bottleneck * k).
        n_spatial:          Number of spatial tokens per time step.
        n_register:         Number of learnable register tokens.
        n_agent:            Number of agent tokens (0 during pretraining).
        n_heads:            Number of query attention heads.
        depth:              Number of transformer layers.
        k_max:              Maximum number of sampling steps (power of 2).
        action_dim:         Flat continuous action vector dim (toy / simple
                            robotics). Mutually exclusive with ``action_components``.
        action_components:  Multi-component action spec (e.g. Minecraft keyboard +
                            mouse). Mutually exclusive with ``action_dim``.
        n_kv_heads:         KV heads for GQA (default = n_heads).
        mlp_ratio:          MLP hidden size multiplier.
        time_every:         Apply time attention every N layers.
        dropout:            Dropout rate.
        use_qk_norm:        Use QKNorm in attention.
        logit_cap:          Attention logit soft capping.
        space_mode:         Attention mode ("wm_agent_isolated" or "wm_agent").
        max_T:              Maximum time steps for RoPE cache.
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
        action_dim: Optional[int] = None,
        action_components: Optional[Sequence[ActionComponent]] = None,
        n_kv_heads: int | None = None,
        mlp_ratio: float = 8/3,
        time_every: int = 4,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        logit_cap: float | None = 50.0,
        space_mode: str = "wm_agent",
        max_T: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_spatial = d_spatial
        self.n_spatial = n_spatial
        self.n_register = n_register
        self.n_agent = n_agent
        self.k_max = k_max

        self.action_encoder = ActionEncoder(
            d_model=d_model,
            action_dim=action_dim,
            action_components=action_components,
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
            max_T=max_T,
        )

        self.flow_head = nn.Linear(d_model, d_spatial)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def _build_tokens(
        self,
        actions: ActionInput,
        step_idx: torch.Tensor,
        signal_idx: torch.Tensor,
        z_noisy: torch.Tensor,
        agent_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T = z_noisy.shape[:2]

        act_tok = self.action_encoder(actions, batch_time_shape=(B, T))  # (B,T,1,D)
        sc_tok = self.shortcut_encoder(step_idx, signal_idx)              # (B,T,1,D)
        spatial_tok = self.spatial_proj(z_noisy)                          # (B,T,Nz,D)
        reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        parts = [act_tok, sc_tok, spatial_tok, reg]
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros(
                    B, T, self.n_agent, self.d_model,
                    device=z_noisy.device, dtype=z_noisy.dtype,
                )
            parts.append(agent_tokens)
        return torch.cat(parts, dim=2)  # (B, T, S, D)

    def _read_heads(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        spatial_out = x[:, :, self.spatial_slice, :]
        z1_hat = self.flow_head(spatial_out)
        agent_out = x[:, :, self.agent_slice, :] if self.n_agent > 0 else None
        return z1_hat, agent_out

    def forward(
        self,
        actions: ActionInput,
        step_idx: torch.Tensor,
        signal_idx: torch.Tensor,
        z_noisy: torch.Tensor,
        *,
        agent_tokens: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        commit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            actions:      None, Tensor ``(B, T, action_dim)``, or dict of components.
            step_idx:     ``(B, T)`` long — step size index.
            signal_idx:   ``(B, T)`` long — signal level index.
            z_noisy:      ``(B, T, n_spatial, d_spatial)`` corrupted representations.
            agent_tokens: ``(B, T, n_agent, d_model)`` or None.
            cache:        Optional ``KVCache`` for incremental decoding. When given:
                          - if empty, runs standard forward; with ``commit=True``
                            fills the cache from ``z_noisy``'s timesteps.
                          - if non-empty, ``z_noisy`` and siblings must contain
                            only the *new* timesteps beyond ``cache.t_cached``;
                            ``commit`` controls whether those timesteps are
                            appended to the cache.
            commit:       See ``cache``. Ignored when ``cache is None``.

        Returns:
            z1_hat:        predictions for the processed timesteps.
            agent_out:     agent-token outputs for the processed timesteps,
                           or None if n_agent == 0.
        """
        tokens = self._build_tokens(actions, step_idx, signal_idx, z_noisy, agent_tokens)
        x = self.transformer(tokens, cache=cache, commit=commit)
        return self._read_heads(x)


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
    """
    z0 = torch.randn_like(z1)
    tau_4d = tau[..., None, None]  # (B, T, 1, 1)
    z_tilde = (1.0 - tau_4d) * z0 + tau_4d * z1
    return z_tilde


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

    B_boot = max(0, min(B - 1, int(round(bootstrap_fraction * B))))
    B_flow = B - B_boot

    # --- Flow (empirical) portion ---
    d_f, step_f, tau_f, sig_f = sample_flow_schedule(B_flow, T, k_max, device)
    z_tilde_f = corrupt_representations(z1[:B_flow], tau_f)

    # --- Bootstrap portion ---
    if B_boot > 0:
        d_b, step_b, tau_b, sig_b = sample_bootstrap_schedule(B_boot, T, k_max, device)
        z_tilde_b = corrupt_representations(z1[B_flow:], tau_b)
    else:
        d_b = torch.zeros(0, T, device=device)
        step_b = torch.zeros(0, T, device=device, dtype=torch.long)
        tau_b = torch.zeros(0, T, device=device)
        sig_b = torch.zeros(0, T, device=device, dtype=torch.long)
        z_tilde_b = torch.zeros(0, T, *z1.shape[2:], device=device, dtype=z1.dtype)

    # Concatenate full batch for one forward pass
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

def _slice_actions(
    actions: ActionInput, start: int, stop: int,
) -> ActionInput:
    """Slice flat-tensor or dict-of-tensors actions along the time axis."""
    if actions is None:
        return None
    if isinstance(actions, torch.Tensor):
        return actions[:, start:stop]
    return {name: t[:, start:stop] for name, t in actions.items()}


def _corrupt_past(
    past: torch.Tensor, tau_ctx: float,
) -> torch.Tensor:
    """Mix clean past with standard-normal noise at magnitude ``tau_ctx``."""
    if tau_ctx <= 0 or past.shape[1] == 0:
        return past
    noise = torch.randn_like(past)
    return tau_ctx * noise + (1.0 - tau_ctx) * past


@torch.no_grad()
def sample_one_timestep(
    dynamics: DynamicsModel,
    *,
    past_packed: torch.Tensor,
    k_max: int,
    K: int = 4,
    actions: ActionInput = None,
    tau_ctx: float = 0.1,
    agent_tokens: Optional[torch.Tensor] = None,
    cache: Optional[KVCache] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate one new frame by K-step shortcut denoising.

    Paper Section 3.2: "We sample autoregressively in time and generate the
    representations of each frame using the shortcut model with K=4 sample
    steps. We slightly corrupt the past inputs to signal level tau_ctx=0.1."

    Args:
        dynamics:     The DynamicsModel (should be in eval mode).
        past_packed:  (B, t_ctx, n_spatial, d_spatial) context frames.
        k_max:        Maximum sampling steps.
        K:            Number of denoising steps for the new frame.
        actions:      None, Tensor ``(B, t_ctx+1, action_dim)``, or a dict of
                      components each with leading shape ``(B, t_ctx+1, ...)``.
                      Must cover the frame being generated.
        tau_ctx:      Noise magnitude for past inputs (default 0.1).
        agent_tokens: (B, t_ctx+1, n_agent, d_model) or None.
        cache:        Optional ``KVCache``. When provided, any uncached past
                      frames are committed first (with the same ``tau_ctx``
                      corruption), then the K denoising iterations run
                      incrementally against the cache. The newly generated
                      frame is **not** committed here — the caller
                      (``sample_sequence``) commits it after selecting it as
                      the output.

    Returns:
        z_new:     (B, n_spatial, d_spatial) generated frame.
        agent_out: (B, n_agent, d_model) or None.
    """
    device = past_packed.device
    dtype = past_packed.dtype
    B, t_ctx = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]
    emax = _log2_int(k_max)

    signal_level = 1.0 - tau_ctx
    ctx_signal_val = min(int(round(signal_level * k_max)), k_max)

    d_new = 1.0 / K
    step_e = int(round(math.log2(K)))

    # --- Cache-based path --------------------------------------------------
    if cache is not None:
        t_missing = t_ctx - cache.t_cached
        if t_missing > 0:
            past_tail = past_packed[:, -t_missing:]
            past_tail_corrupt = _corrupt_past(past_tail, tau_ctx)

            step_ctx = torch.full((B, t_missing), emax, device=device, dtype=torch.long)
            sig_ctx = torch.full((B, t_missing), ctx_signal_val, device=device, dtype=torch.long)
            act_ctx = _slice_actions(actions, cache.t_cached, t_ctx)
            agent_ctx = (
                agent_tokens[:, cache.t_cached:t_ctx] if agent_tokens is not None else None
            )

            _ = dynamics(
                act_ctx, step_ctx, sig_ctx, past_tail_corrupt,
                agent_tokens=agent_ctx, cache=cache, commit=True,
            )

        z = torch.randn(B, 1, n_spatial, d_spatial, device=device, dtype=dtype)
        last_agent_out: Optional[torch.Tensor] = None

        act_new = _slice_actions(actions, t_ctx, t_ctx + 1)
        agent_new = agent_tokens[:, t_ctx:t_ctx + 1] if agent_tokens is not None else None

        for i in range(K):
            tau_i = i * d_new
            sig_i = min(int(round(tau_i * k_max)), k_max)

            step_new = torch.full((B, 1), step_e, device=device, dtype=torch.long)
            signal_new = torch.full((B, 1), sig_i, device=device, dtype=torch.long)

            z1_hat, a_out = dynamics(
                act_new, step_new, signal_new, z,
                agent_tokens=agent_new, cache=cache, commit=False,
            )  # (B, 1, Nz, Dz)

            if a_out is not None:
                last_agent_out = a_out[:, -1]

            denom = max(1e-4, 1.0 - tau_i)
            velocity = (z1_hat.float() - z.float()) / denom
            z = (z.float() + velocity * d_new).to(dtype)

        return z[:, 0], last_agent_out

    # --- Non-cache path ---------------------------------------------------
    past_corrupted = _corrupt_past(past_packed, tau_ctx) 
    z = torch.randn(B, 1, n_spatial, d_spatial, device=device, dtype=dtype)
    last_agent_out = None

    for i in range(K):
        tau_i = i * d_new
        sig_i = min(int(round(tau_i * k_max)), k_max)

        packed_seq = torch.cat([past_corrupted, z], dim=1)
        T_total = packed_seq.shape[1]

        step_idxs = torch.full((B, T_total), emax, device=device, dtype=torch.long)
        step_idxs[:, -1] = step_e

        signal_idxs = torch.full((B, T_total), ctx_signal_val, device=device, dtype=torch.long)
        signal_idxs[:, -1] = sig_i

        actions_in = _slice_actions(actions, 0, T_total)
        agent_in = agent_tokens[:, :T_total] if agent_tokens is not None else None

        z1_hat, a_out = dynamics(
            actions_in, step_idxs, signal_idxs, packed_seq,
            agent_tokens=agent_in,
        )
        z1_hat_new = z1_hat[:, -1:, :, :]

        if a_out is not None:
            last_agent_out = a_out[:, -1]

        denom = max(1e-4, 1.0 - tau_i)
        velocity = (z1_hat_new.float() - z.float()) / denom
        z = (z.float() + velocity * d_new).to(dtype)

    return z[:, 0], last_agent_out


@torch.no_grad()
def sample_sequence(
    dynamics: DynamicsModel,
    *,
    context: torch.Tensor,
    horizon: int,
    k_max: int,
    K: int = 4,
    actions: ActionInput = None,
    tau_ctx: float = 0.1,
    agent_tokens: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
    """
    Autoregressively generate a sequence of frames.

    With ``use_cache=True`` (default) a single ``KVCache`` is kept across
    frames so each denoising step only runs the transformer over the new
    timestep, pulling K/V for the growing past from the cache.

    Args:
        dynamics:     The DynamicsModel (should be in eval mode).
        context:      (B, t_ctx, n_spatial, d_spatial) context frames.
        horizon:      Number of frames to generate.
        k_max:        Maximum sampling steps.
        K:            Denoising steps per frame.
        actions:      None, Tensor ``(B, t_ctx + horizon, action_dim)``, or a
                      dict of components keyed by name.
        tau_ctx:      Context noise magnitude.
        agent_tokens: (B, t_ctx + horizon, n_agent, d_model) or None.
        use_cache:    Toggle KV-cache (default True). Set False to recover
                      the plain rollout — useful for equivalence tests.

    Returns:
        frames:     (B, t_ctx + horizon, n_spatial, d_spatial).
        agent_outs: list of ``horizon`` tensors each (B, n_agent, d_model),
                    or None when agent_tokens is not provided.
    """
    B = context.shape[0]
    t_ctx = context.shape[1]
    device = context.device

    frames = [context[:, t] for t in range(t_ctx)]
    agent_outs: list[torch.Tensor] = []

    cache: Optional[KVCache] = None
    if use_cache:
        cache = dynamics.transformer.make_kv_cache()

    signal_level = 1.0 - tau_ctx
    ctx_signal_val = min(int(round(signal_level * k_max)), k_max)
    emax = _log2_int(k_max)

    for h in range(horizon):
        past = torch.stack(frames, dim=1)

        z_next, a_out = sample_one_timestep(
            dynamics,
            past_packed=past,
            k_max=k_max,
            K=K,
            actions=actions,
            tau_ctx=tau_ctx,
            agent_tokens=agent_tokens,
            cache=cache,
        )
        frames.append(z_next)
        if a_out is not None:
            agent_outs.append(a_out)

        if cache is not None:
            # Commit the newly generated frame so it becomes cached past for
            # subsequent frames' incremental forwards.
            new_z = z_next.unsqueeze(1)
            new_z_corrupt = _corrupt_past(new_z, tau_ctx)

            step_idx = torch.full((B, 1), emax, device=device, dtype=torch.long)
            signal_idx = torch.full((B, 1), ctx_signal_val, device=device, dtype=torch.long)

            act_in = _slice_actions(actions, t_ctx + h, t_ctx + h + 1)
            agent_in = (
                agent_tokens[:, t_ctx + h:t_ctx + h + 1]
                if agent_tokens is not None
                else None
            )

            _ = dynamics(
                act_in, step_idx, signal_idx, new_z_corrupt,
                agent_tokens=agent_in, cache=cache, commit=True,
            )

    seq = torch.stack(frames, dim=1)
    return seq, agent_outs if agent_outs else None
