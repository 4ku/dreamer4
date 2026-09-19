"""The world model as phases 2 and 3 consume it, plus the paper's policy-inside-the-world-model.

Two things live here.

:class:`WorldModel` gathers what the dream loop and the real-environment evaluator read out of
a dynamics checkpoint: the dynamics model, the tokenizer that turns frames into its latents, the
dataset it was trained on (the source of the demonstrations dreams start from, of the terminal rule
and of the env spec) and the three numbers the rollout needs (``window``, ``tau_ctx``,
``action_vec_dim``). The dataset comes from the CHECKPOINT rather than from a flag on purpose: it
makes it impossible to dream inside one dataset's world model while seeding the dreams from
another's recordings.

:class:`AgentPolicy` is the paper's agent (Section 3.3). There the policy is not a network beside
the world model but a set of MLP heads on AGENT tokens interleaved into the dynamics transformer
itself, so "run the policy for one step" means "run the world model over the episode's context and
read the agent token's output embedding h_t". One class does that for both consumers -- imagination
in phase 3 and the live environment in the evaluator -- so the policy cannot have one context in
training and a different one at play.

CONTEXT WINDOW CONVENTION, shared by phase 2, imagination and evaluation. The policy reads a window
of ``T`` frames ending at the current one. Before ``T`` frames exist the window is left-padded by
repeating the episode's first frame, which is the same "static clip" the tokenizer pads its own
history with. Slot 0 of any window, and every padded slot, carries the ACTION START FLAG rather
than an action -- matching :func:`dreamer4.models.dynamics.align_actions`, which is how every
window the dynamics model has ever been trained on was built. The action in slot ``j > 0`` is the
one that LED INTO that slot's frame, never the one taken from it, so the agent token at a slot
cannot read the action it is being asked to predict.
"""
import numpy as np
import torch
import torch.nn as nn

from dreamer4.data import open_video_dataset
from dreamer4.models.agent_heads import flatten_agent_out
from dreamer4.models.dynamics import _log2_int, sample_one_timestep
from dreamer4.models.tokenizer import FrozenTokenizer
from dreamer4.train.config import DynamicsModelConfig, config_from_dict
from dreamer4.train.train_dynamics import build_dynamics, load_dynamics_checkpoint
from dreamer4.train.train_tokenizer import weights_from_checkpoint


class WorldModel:
    """Everything phases 2 and 3 need out of a ``train_dynamics`` checkpoint.

    Args:
        path:   path to a phase-1b dynamics checkpoint.
        device: device the dynamics model and tokenizer are loaded onto.
        n_agent: number of AGENT tokens to interleave into each time step. 0 rebuilds the
            checkpoint exactly as pretraining left it. Any positive value appends an AGENT segment
            to the layout; the dynamics model gains NO parameters from it (agent tokens are built
            outside, by :class:`dreamer4.models.agent_heads.AgentTokenEncoder`) and no world-token
            output changes, because the ``wm_agent`` mask never lets a world row attend an agent
            column. The state dict therefore still loads strictly.
        train: leave the dynamics parameters trainable. Phase 2 finetunes the transformer, so it
            asks for this; phases 3 and the evaluators do not.
    """

    def __init__(self, path, device="cuda", *, n_agent=0, train=False):
        dyn, ckpt = load_dynamics_checkpoint(path, device=device)
        config, meta = ckpt["config"], ckpt["latent_meta"]
        tok = config["tokenizer"]
        if n_agent > 0:
            dyn = self._rebuild_with_agent_tokens(ckpt, n_agent, device)
        self.dyn = dyn
        self.n_agent = int(n_agent)
        if not train:
            for p in self.dyn.parameters():
                p.requires_grad_(False)
        self.tok = FrozenTokenizer(tok["ckpt"], history=tok["history"], pack_k=tok["pack_k"],
                                   decoder_ckpt=(tok["decoder_ckpt"] or None), device=device)
        self.data_path = config["data"]["path"]
        self.dataset = open_video_dataset(self.data_path, proprio=config["data"]["proprio"],
                                          actions=True)
        self.env_spec = self.dataset.env_spec()                  # None for offline-only datasets
        self.continues_from_reward = self.dataset.continues_from_reward   # the terminal rule
        self.proprio_from_info = self.dataset.proprio_from_info  # live env state -> proprio vector
        self.window = int(meta["window"])                        # context frames the model reads
        self.max_T = int(meta["max_T"])                          # longest window its RoPE covers
        self.k_max = int(config["model"]["k_max"])
        self.tau_ctx = float(config["objective"]["tau_ctx"])     # signal level it reads them at
        # The pretraining objective's own settings (context-noise band, scheduled sampling, ...):
        # phase 2 keeps the video-prediction loss running and has to run it the way phase 1b did.
        self.objective = dict(config["objective"])
        # meta["action_dim"] counts the dataset's action vector PLUS the start flag the dynamics
        # model prepends; the rollout builds the action vector itself, so it needs the dataset's.
        self.action_vec_dim = int(meta["action_dim"]) - 1
        self.latent_meta = dict(meta)
        self.model_cfg = dict(config["model"])

    @staticmethod
    def _rebuild_with_agent_tokens(ckpt, n_agent, device):
        """The same architecture the checkpoint describes, plus an AGENT segment."""
        meta = ckpt["latent_meta"]
        model_cfg = config_from_dict(DynamicsModelConfig, ckpt["config"]["model"])
        dyn = build_dynamics(model_cfg, n_spatial=meta["n_spatial"], d_spatial=meta["d_spatial"],
                             action_dim=meta["action_dim"], d_proprio=meta["d_proprio"],
                             max_T=meta["max_T"], n_agent=n_agent)
        dyn.load_state_dict(weights_from_checkpoint(ckpt, prefer_ema=True))
        return dyn.to(device)

    def load_dynamics_weights(self, state_dict):
        """Replace the dynamics weights, e.g. with the copy phase 2 finetuned."""
        self.dyn.load_state_dict(state_dict)
        return self.dyn


# ---------------------------------------------------------------------------
# The context window
# ---------------------------------------------------------------------------


def window_frames(end, T):
    """Frame indices of a ``T``-slot context window ending at frame ``end`` (see module docstring).

    Left-padding is a clamp at 0, so slots before the episode start repeat frame 0.

    Args:
        end: ``(B,)`` int array/tensor -- the current frame index of each row.
        T:   window length.

    Returns:
        ``(B, T)`` of the same type as ``end``.
    """
    offs = np.arange(-(T - 1), 1)
    return np.clip(np.asarray(end).reshape(-1, 1) + offs.reshape(1, -1), 0, None)


class AgentPolicy(nn.Module):
    """Act by running the world model and reading the agent token's output embedding.

    Paper Section 3.3: the policy and reward heads read ``h_t``, the transformer output at the
    agent tokens of step ``t``. Because attention is block-causal in time, ``h_t`` is a function of
    the WHOLE context window up to and including frame ``t`` -- the policy is a sequence model, not
    a per-frame readout, and this class is what gives it that context at play time.

    State is one rolling window per row (see the module docstring), so rows may reset independently
    -- which is what a batch of live environments does. One transformer forward per step serves the
    whole batch.

    Args:
        dyn:    the dynamics model, built with ``n_agent > 0``.
        heads:  :class:`dreamer4.models.agent_heads.AgentHeads`.
        window: context slots ``T``.
        tau_ctx: the noise fraction context frames are read at -- the same one imagination uses,
            and the reason it is applied to REAL frames too: the policy must meet its inputs in the
            regime it was trained in.
        k_max:  the model's signal-level grid, for the context signal index.
    """

    def __init__(self, dyn, heads, *, window, tau_ctx, k_max, task_id=0):
        super().__init__()
        self.dyn = dyn
        self.heads = heads
        self.T = int(window)
        self.tau_ctx = float(tau_ctx)
        self.k_max = int(k_max)
        self.task_id = int(task_id)
        self.ctx_signal = min(int(round((1.0 - self.tau_ctx) * self.k_max)), self.k_max)
        self.emax = _log2_int(self.k_max)
        self.Z = self.P = self.A = None
        self.filled = None

    # -- rolling state ----------------------------------------------------

    def _allocate(self, z, proprio):
        B, N, D = z.shape
        dev = z.device
        self.Z = torch.zeros(B, self.T, N, D, device=dev, dtype=z.dtype)
        self.P = torch.zeros(B, self.T, proprio.shape[-1], device=dev, dtype=torch.float32)
        self.A = torch.zeros(B, self.T, self.dyn.action_encoder.components[0].dim,
                             device=dev, dtype=torch.float32)
        self.filled = torch.zeros(B, dtype=torch.long, device=dev)

    def reset_rows(self, rows=None):
        """Forget the context of these rows; the next :meth:`observe` restarts their window."""
        if self.filled is None:
            return
        if rows is None:
            self.filled.zero_()
        else:
            self.filled[torch.as_tensor(rows, device=self.filled.device, dtype=torch.long)] = 0

    def observe(self, z, proprio, prev_action=None):
        """Append frame ``z`` (and its proprio) to every row's window.

        ``prev_action`` is the action that produced this frame: ``(B,)`` ids for a categorical
        policy, ``(B, D_a)`` vectors otherwise. Rows whose window was just reset ignore it and
        start a fresh (left-padded) window instead.
        """
        if self.Z is None or self.Z.shape[0] != z.shape[0]:
            self._allocate(z, proprio)
        D = self.A.shape[-1] - 1
        fresh = self.filled == 0
        if (~fresh).any():
            self.Z[~fresh] = torch.roll(self.Z[~fresh], -1, dims=1)
            self.P[~fresh] = torch.roll(self.P[~fresh], -1, dims=1)
            self.A[~fresh] = torch.roll(self.A[~fresh], -1, dims=1)
            self.Z[~fresh, -1] = z[~fresh].to(self.Z.dtype)
            self.P[~fresh, -1] = proprio[~fresh].float()
            self.A[~fresh, -1] = 0.0
            if prev_action is not None:
                act = prev_action[~fresh]
                if act.dim() == 1:
                    self.A[~fresh, -1, :D] = torch.nn.functional.one_hot(
                        act.long().clamp(max=D - 1), D).float()
                else:
                    self.A[~fresh, -1, :D] = act.float()
            else:
                self.A[~fresh, -1, D] = 1.0
        if fresh.any():
            # A fresh row: the whole window is this one frame, and nothing led into any slot.
            self.Z[fresh] = z[fresh].to(self.Z.dtype).unsqueeze(1)
            self.P[fresh] = proprio[fresh].float().unsqueeze(1)
            self.A[fresh] = 0.0
            self.A[fresh, :, D] = 1.0
        # Slot 0 of a window always carries the start flag: it is the oldest frame the model can
        # see, so whatever led into it is outside the window.
        self.A[:, 0] = 0.0
        self.A[:, 0, self.A.shape[-1] - 1] = 1.0
        self.filled = (self.filled + 1).clamp(max=self.T)

    # -- the forward ------------------------------------------------------

    def agent_tokens(self, B, T, device):
        """The agent-token inputs for a whole window: the task embedding."""
        ids = torch.full((B, T), self.task_id, dtype=torch.long, device=device)
        return self.heads.tokens((B, T), task_ids=ids, device=device)

    def h(self):
        """``h_t`` for the current frame of every row: ``(B, n_agent * d_model)``."""
        B, T = self.Z.shape[:2]
        dev = self.Z.device
        noise = self.tau_ctx
        z_in = (1 - noise) * self.Z.float() + noise * torch.randn_like(self.Z.float())
        p_in = (1 - noise) * self.P + noise * torch.randn_like(self.P)
        step = torch.full((B, T), self.emax, device=dev, dtype=torch.long)
        sig = torch.full((B, T), self.ctx_signal, device=dev, dtype=torch.long)
        out = self.dyn(self.A, step, sig, z_in, agent_tokens=self.agent_tokens(B, T, dev),
                       proprio_noisy=p_in)
        return flatten_agent_out(out[-1][:, -1])

    def act(self, z, proprio, *, greedy=False, temperature=1.0, prev_action=None):
        """Observe a frame and return the action to take from it."""
        self.observe(z, proprio, prev_action)
        logits = self.heads.policy.at(self.h(), 0)
        return self.heads.policy.sample(logits, greedy=greedy, temperature=temperature)


def dream_context(world, heads, *, z_hist, prop_hist, act_hist, tau_ctx, task_ids,
                  cache=None, commit=True):
    """Commit one imagined frame to the KV cache and read its ``h_t`` on the way through.

    The frame the policy acts on is read at the CONTEXT signal level ``1 - tau_ctx``, not at one of
    the K partially-denoised shortcut passes: that is the level the frame will have for every later
    step, and the level a real frame has at evaluation, so the policy meets its input in one regime
    everywhere. The pass is not an extra cost -- the same forward is what puts the frame into the
    cache for the next step's denoising.

    Args:
        z_hist / prop_hist / act_hist: the NEW timesteps only when ``cache`` already holds the
            past, or the whole history on the first call.

    Returns:
        ``h`` ``(B, n_agent * d_model)`` for the last timestep passed in.
    """
    B, T = z_hist.shape[:2]
    dev = z_hist.device
    k_max = world.k_max
    sig = min(int(round((1.0 - tau_ctx) * k_max)), k_max)
    step_idx = torch.full((B, T), _log2_int(k_max), device=dev, dtype=torch.long)
    sig_idx = torch.full((B, T), sig, device=dev, dtype=torch.long)
    z_in = (1 - tau_ctx) * z_hist + tau_ctx * torch.randn_like(z_hist)
    p_in = (1 - tau_ctx) * prop_hist + tau_ctx * torch.randn_like(prop_hist)
    ag = heads.tokens((B, T), task_ids=task_ids.reshape(B, 1).expand(B, T), device=dev)
    out = world.dyn(act_hist, step_idx, sig_idx, z_in, agent_tokens=ag, proprio_noisy=p_in,
                    cache=cache, commit=commit)
    return flatten_agent_out(out[-1][:, -1])


def dream_step(world, *, z_hist, prop_hist, act_hist, K, cache=None):
    """Generate the next imagined frame. Agent tokens are deliberately absent.

    Nothing in the world model attends to an agent column, so the agent tokens cannot change a
    generated frame by even one bit -- which is exactly why they need not be built for the K
    denoising passes.
    """
    return sample_one_timestep(world.dyn, past_packed=z_hist, k_max=world.k_max, K=K,
                               actions=act_hist, tau_ctx=world.tau_ctx,
                               past_proprio=prop_hist, cache=cache)[:2]
