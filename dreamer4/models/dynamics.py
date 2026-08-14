"""
Dynamics model for Dreamer 4: the MODEL and its inference-time sampling.

Predicts future latent representations given actions, trained with shortcut
forcing — a combination of flow matching, shortcut models, and diffusion
forcing (Paper Section 3.2, Eq. 6-8). The TRAINING OBJECTIVES themselves
(``shortcut_forcing_loss`` and the production ``clean_context_loss``) live in
:mod:`dreamer4.train.dynamics_objectives` together with their
schedule/corruption/ramp-weight building blocks; this module contains the
architecture and the autoregressive samplers.

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

from dreamer4.models.transformer.modality import Modality, TokenLayout
from dreamer4.models.transformer import BlockCausalTransformer
from dreamer4.models.transformer.kv_cache import KVCache


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
    the attention pattern (``wm_agent`` throughout; pretraining just runs it
    with ``n_agent=0``).

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
        space_mode:         Attention mode (the world model uses "wm_agent").
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
        d_proprio: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_spatial = d_spatial
        self.n_spatial = n_spatial
        self.n_register = n_register
        self.n_agent = n_agent
        self.k_max = k_max
        self.d_proprio = d_proprio

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
        ]
        if d_proprio is not None:
            # proprioceptive stream (robot joint state): ONE token per timestep,
            # noised and denoised JOINTLY with the spatial latents (paper's design;
            # the wm_* masks treat PROPRIO as a world modality — no mask changes).
            self.proprio_proj = nn.Linear(d_proprio, d_model)
            self.proprio_head = nn.Linear(d_model, d_proprio)
            nn.init.zeros_(self.proprio_head.weight)
            nn.init.zeros_(self.proprio_head.bias)
            segments.append((Modality.PROPRIO, 1))
        segments.append((Modality.REGISTER, n_register))
        if n_agent > 0:
            segments.append((Modality.AGENT, n_agent))

        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.proprio_slice = sl.get(Modality.PROPRIO, slice(0, 0))
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
        proprio_noisy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = z_noisy.shape[:2]
        assert (proprio_noisy is not None) == (self.d_proprio is not None), \
            "pass `proprio_noisy` iff the model was built with d_proprio"

        act_tok = self.action_encoder(actions, batch_time_shape=(B, T))  # (B,T,1,D)
        sc_tok = self.shortcut_encoder(step_idx, signal_idx)              # (B,T,1,D)
        spatial_tok = self.spatial_proj(z_noisy)                          # (B,T,Nz,D)
        reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        parts = [act_tok, sc_tok, spatial_tok]
        if proprio_noisy is not None:
            parts.append(self.proprio_proj(proprio_noisy).unsqueeze(2))   # (B,T,1,D)
        parts.append(reg)
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros(
                    B, T, self.n_agent, self.d_model,
                    device=z_noisy.device, dtype=z_noisy.dtype,
                )
            parts.append(agent_tokens)
        return torch.cat(parts, dim=2)  # (B, T, S, D)

    def _read_heads(self, x: torch.Tensor):
        spatial_out = x[:, :, self.spatial_slice, :]
        z1_hat = self.flow_head(spatial_out)
        agent_out = x[:, :, self.agent_slice, :] if self.n_agent > 0 else None
        if self.d_proprio is None:
            return z1_hat, agent_out
        proprio_hat = self.proprio_head(x[:, :, self.proprio_slice, :][:, :, 0])
        return z1_hat, proprio_hat, agent_out

    def forward(
        self,
        actions: ActionInput,
        step_idx: torch.Tensor,
        signal_idx: torch.Tensor,
        z_noisy: torch.Tensor,
        *,
        agent_tokens: Optional[torch.Tensor] = None,
        proprio_noisy: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        commit: bool = False,
    ):
        """
        Args:
            actions:      None, Tensor ``(B, T, action_dim)``, or dict of components.
            step_idx:     ``(B, T)`` long — step size index.
            signal_idx:   ``(B, T)`` long — signal level index.
            z_noisy:      ``(B, T, n_spatial, d_spatial)`` corrupted representations.
            proprio_noisy: ``(B, T, d_proprio)`` corrupted proprio stream (iff built
                          with ``d_proprio``; denoised jointly, x-prediction).
            agent_tokens: ``(B, T, n_agent, d_model)`` or None.
            cache:        Optional ``KVCache`` for incremental decoding. When given:
                          - if empty, runs standard forward; with ``commit=True``
                            fills the cache from ``z_noisy``'s timesteps.
                          - if non-empty, ``z_noisy`` and siblings must contain
                            only the *new* timesteps beyond ``cache.committed``;
                            ``commit`` controls whether those timesteps are
                            appended to the cache.
            commit:       See ``cache``. Ignored when ``cache is None``.

        Returns:
            z1_hat:        predictions for the processed timesteps.
            (proprio_hat:  ``(B, T, d_proprio)`` — only when built with d_proprio.)
            agent_out:     agent-token outputs for the processed timesteps,
                           or None if n_agent == 0.
        """
        tokens = self._build_tokens(actions, step_idx, signal_idx, z_noisy, agent_tokens,
                                    proprio_noisy=proprio_noisy)
        x = self.transformer(tokens, cache=cache, commit=commit)
        return self._read_heads(x)


# ---------------------------------------------------------------------------
# Shortcut step sizes are powers of two, d = 1/2^e, and the model consumes
# the EXPONENT e as a discrete step index (embedding lookup): step_idx == e,
# and the finest step d_min = 1/k_max has index emax = log2(k_max). This
# helper converts k_max to that exponent. Used by the inference samplers
# below and imported by dreamer4.train.dynamics_objectives.
# ---------------------------------------------------------------------------

def _log2_int(k_max: int) -> int:
    """log2(k_max) as an int, asserting k_max is a power of 2."""
    e = int(round(math.log2(k_max)))
    assert (1 << e) == k_max, f"k_max={k_max} must be a power of 2"
    return e


# ---------------------------------------------------------------------------
# Action alignment — the model's input convention
# ---------------------------------------------------------------------------


def align_actions(actions: torch.Tensor, T: int) -> torch.Tensor:
    """
    (B, T-1, D_a) dataset actions -> (B, T, D_a+1) per-frame aligned actions
    with the start flag in the last channel (see module docstring).
    """
    B, _, D = actions.shape
    aligned = torch.zeros(B, T, D + 1, device=actions.device, dtype=torch.float32)
    aligned[:, 0, D] = 1.0                       # start flag: nothing led in
    aligned[:, 1:, :D] = actions.float()
    return aligned


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
    past_proprio: Optional[torch.Tensor] = None,
    cache: Optional[KVCache] = None,
):
    """
    Generate one new frame by K-step shortcut denoising.

    When the model was built with ``d_proprio``, pass ``past_proprio``
    ``(B, t_ctx, d_proprio)``; the proprio stream is denoised jointly and the
    return becomes ``(z_new, proprio_new, agent_out)``. Works with and without
    a ``cache`` — the PROPRIO token is one more S position, so its K/V ride
    the same cache; uncached past frames are committed with their (corrupted)
    proprio, and the new frame's noisy proprio is threaded through each
    incremental denoising call.

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
        has_proprio = dynamics.d_proprio is not None
        assert not (has_proprio and past_proprio is None), \
            "model has d_proprio: pass past_proprio (B, t_ctx, d_proprio)"
        # ``cache.committed`` counts caller frames monotonically — ``t_cached``
        # shrinks when the sliding window evicts, so it would over-count
        # t_missing (and re-commit tail frames) on horizons beyond max_T.
        t_missing = t_ctx - cache.committed
        if t_missing > 0:
            past_tail = past_packed[:, -t_missing:]
            past_tail_corrupt = _corrupt_past(past_tail, tau_ctx)
            prop_tail_corrupt = (
                _corrupt_past(past_proprio[:, -t_missing:], tau_ctx)
                if has_proprio else None
            )

            step_ctx = torch.full((B, t_missing), emax, device=device, dtype=torch.long)
            sig_ctx = torch.full((B, t_missing), ctx_signal_val, device=device, dtype=torch.long)
            act_ctx = _slice_actions(actions, t_ctx - t_missing, t_ctx)
            agent_ctx = (
                agent_tokens[:, t_ctx - t_missing:t_ctx] if agent_tokens is not None else None
            )

            _ = dynamics(
                act_ctx, step_ctx, sig_ctx, past_tail_corrupt,
                agent_tokens=agent_ctx, proprio_noisy=prop_tail_corrupt,
                cache=cache, commit=True,
            )

        z = torch.randn(B, 1, n_spatial, d_spatial, device=device, dtype=dtype)
        prop = (torch.randn(B, 1, dynamics.d_proprio, device=device, dtype=dtype)
                if has_proprio else None)
        last_agent_out: Optional[torch.Tensor] = None

        act_new = _slice_actions(actions, t_ctx, t_ctx + 1)
        agent_new = agent_tokens[:, t_ctx:t_ctx + 1] if agent_tokens is not None else None

        for i in range(K):
            tau_i = i * d_new
            sig_i = min(int(round(tau_i * k_max)), k_max)

            step_new = torch.full((B, 1), step_e, device=device, dtype=torch.long)
            signal_new = torch.full((B, 1), sig_i, device=device, dtype=torch.long)

            denom = max(1e-4, 1.0 - tau_i)
            if has_proprio:
                z1_hat, prop_hat, a_out = dynamics(
                    act_new, step_new, signal_new, z,
                    agent_tokens=agent_new, proprio_noisy=prop,
                    cache=cache, commit=False,
                )  # (B, 1, Nz, Dz), (B, 1, d_p)
                prop_vel = (prop_hat.float() - prop.float()) / denom
                prop = (prop.float() + prop_vel * d_new).to(dtype)
            else:
                z1_hat, a_out = dynamics(
                    act_new, step_new, signal_new, z,
                    agent_tokens=agent_new, cache=cache, commit=False,
                )  # (B, 1, Nz, Dz)

            if a_out is not None:
                last_agent_out = a_out[:, -1]

            velocity = (z1_hat.float() - z.float()) / denom
            z = (z.float() + velocity * d_new).to(dtype)

        if has_proprio:
            return z[:, 0], prop[:, 0], last_agent_out
        return z[:, 0], last_agent_out

    # --- Non-cache path ---------------------------------------------------
    has_proprio = dynamics.d_proprio is not None
    assert not (has_proprio and past_proprio is None), \
        "model has d_proprio: pass past_proprio (B, t_ctx, d_proprio)"

    past_corrupted = _corrupt_past(past_packed, tau_ctx)
    z = torch.randn(B, 1, n_spatial, d_spatial, device=device, dtype=dtype)
    prop_past_corrupted = prop = None
    if has_proprio:
        prop_past_corrupted = _corrupt_past(past_proprio, tau_ctx)
        prop = torch.randn(B, 1, dynamics.d_proprio, device=device, dtype=dtype)
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

        denom = max(1e-4, 1.0 - tau_i)
        if has_proprio:
            prop_seq = torch.cat([prop_past_corrupted, prop], dim=1)
            z1_hat, prop_hat, a_out = dynamics(
                actions_in, step_idxs, signal_idxs, packed_seq,
                agent_tokens=agent_in, proprio_noisy=prop_seq,
            )
            prop_vel = (prop_hat[:, -1:].float() - prop.float()) / denom
            prop = (prop.float() + prop_vel * d_new).to(dtype)
        else:
            z1_hat, a_out = dynamics(
                actions_in, step_idxs, signal_idxs, packed_seq,
                agent_tokens=agent_in,
            )
        z1_hat_new = z1_hat[:, -1:, :, :]

        if a_out is not None:
            last_agent_out = a_out[:, -1]

        velocity = (z1_hat_new.float() - z.float()) / denom
        z = (z.float() + velocity * d_new).to(dtype)

    if has_proprio:
        return z[:, 0], prop[:, 0], last_agent_out
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
    proprio_context: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, ...]:
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
        proprio_context: (B, t_ctx, d_proprio) — required iff the model was
                      built with ``d_proprio``. The proprio stream is denoised
                      jointly with the frames and rolled forward alongside them
                      (cached and uncached alike).
        use_cache:    Toggle KV-cache (default True). Set False to recover
                      the plain rollout — useful for equivalence tests.

    Returns:
        frames:     (B, t_ctx + horizon, n_spatial, d_spatial).
        (proprio:   (B, t_ctx + horizon, d_proprio) — only for proprio models.)
        agent_outs: list of ``horizon`` tensors each (B, n_agent, d_model),
                    or None when agent_tokens is not provided.
    """
    B = context.shape[0]
    t_ctx = context.shape[1]
    device = context.device
    has_proprio = dynamics.d_proprio is not None
    assert (proprio_context is not None) == has_proprio, \
        "pass `proprio_context` iff the model was built with d_proprio"

    frames = [context[:, t] for t in range(t_ctx)]
    props = [proprio_context[:, t] for t in range(t_ctx)] if has_proprio else None
    agent_outs: list[torch.Tensor] = []

    cache: Optional[KVCache] = None
    if use_cache:
        cache = dynamics.transformer.make_kv_cache()

    signal_level = 1.0 - tau_ctx
    ctx_signal_val = min(int(round(signal_level * k_max)), k_max)
    emax = _log2_int(k_max)

    for h in range(horizon):
        past = torch.stack(frames, dim=1)
        past_prop = torch.stack(props, dim=1) if has_proprio else None

        out = sample_one_timestep(
            dynamics,
            past_packed=past,
            k_max=k_max,
            K=K,
            actions=actions,
            tau_ctx=tau_ctx,
            agent_tokens=agent_tokens,
            past_proprio=past_prop,
            cache=cache,
        )
        if has_proprio:
            z_next, prop_next, a_out = out
            props.append(prop_next)
        else:
            z_next, a_out = out
        frames.append(z_next)
        if a_out is not None:
            agent_outs.append(a_out)

        if cache is not None:
            # Commit the newly generated frame so it becomes cached past for
            # subsequent frames' incremental forwards.
            new_z = z_next.unsqueeze(1)
            new_z_corrupt = _corrupt_past(new_z, tau_ctx)
            new_prop_corrupt = (
                _corrupt_past(prop_next.unsqueeze(1), tau_ctx) if has_proprio else None
            )

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
                agent_tokens=agent_in, proprio_noisy=new_prop_corrupt,
                cache=cache, commit=True,
            )

    seq = torch.stack(frames, dim=1)
    if has_proprio:
        return seq, torch.stack(props, dim=1), agent_outs if agent_outs else None
    return seq, agent_outs if agent_outs else None
