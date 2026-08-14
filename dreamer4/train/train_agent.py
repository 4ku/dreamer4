"""
Phase-2 agent finetuning (paper Section 3.3, Eq. 9): behavior-cloning and
reward heads on agent tokens, finetuning the WHOLE dynamics transformer
while the phase-1 world-model loss keeps running.

Episode termination is NOT a separate head: phase 3 derives it from the
reward head via the DATASET's terminal rule
(``EpisodeVideoDataset.continues_from_reward``, since whether that rule is
valid is a property of the environment) — exact in gridworld: reward +1 iff
goal reached, 0 disagreements in 47,695 transitions. The trainer GATES that
derived signal directly: ``term_f1`` / ``term_recall`` in the validation
metrics measure, on held-out data, the very signal the imagination loop will
stop its dreams with.

Requires ``--init_from`` (a ``train_dynamics`` checkpoint): the exact
checkpointed architecture is rebuilt with ``n_agent`` agent tokens in
``wm_agent`` mode — world-token outputs are unchanged by construction (the
mask lets nothing attend to AGENT columns), verified by the no-op test.

Two forwards per step, by design:

1. **WM forward** — the untouched phase-1 ``clean_context_loss`` recipe
   (scheduled sampling, ctx-noise band, image batches) on a batch from ALL
   episodes. This is the paper's retained video-prediction loss; keeping it
   verbatim protects the windowed gate (the forgetting alarm).
2. **Agent forward** — a separate batch pushed through the transformer with
   EVERY frame corrupted to the ctx-noise band (signal = the tau_ctx level):
   the exact regime context frames occupy at deployment. The heads read
   agent-token slots (0 policy, 1 reward; slot 2 is reserved for the phase-3
   value head) at every position, so the embedding the policy trains on IS
   the embedding the online loop reads — no train/inference readout mismatch
   exists to A/B. BC targets are departure-aligned (head n at position t
   predicts a[t+n]) with masked tails at episode ends, and the BC loss is
   weighted to LOW-NOISE, NON-STICKY episodes only (the gridworld analog of
   the paper's "BC loss on the task-relevant fraction"); reward targets are
   ARRIVAL-aligned and train on every row.

All terms — flow, proprio, bootstrap, bc, reward — go through one
RMS-normalizing LossCombiner (paper: no hand-tuned scales). Pretrained
transformer parameters step at ``--optim.lr`` (5e-5, the proven ft rate);
fresh head parameters at ``--head_lr``.

Usage (defaults = the phase-2 recipe):

    python -m dreamer4.train.train_agent \\
        --data.path data/gridworld_10k,data/gridworld_sticky3k \\
        --init_from runs/dyn_e2e_base/checkpoints/latest.pt \\
        --out runs/agent_x --steps 20000

Artifacts under ``--out``: ``config.yaml``, ``tb/``, resumable
``checkpoints/latest.pt``, ``final.json`` (gate table + offline agent
metrics + ONLINE env success rates incl. the random floor); one line
appended to ``../experiments.jsonl``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dreamer4.agent_eval import (AgentPolicy, evaluate_policy,
                                 evaluate_random, make_env)
from dreamer4.data import open_video_dataset, split_train_val
from dreamer4.dynamics_eval import gate_rollout, one_step_eval
from dreamer4.models.agent import (POLICY_SLOT, REWARD_SLOT, PolicyHead,
                                   RewardHead, TaskEncoder)
from dreamer4.models.dynamics import align_actions, _log2_int
from dreamer4.models.tokenizer import FrozenTokenizer
from dreamer4.train.agent_objectives import (behavior_cloning_loss,
                                             continues_from_terminals,
                                             reward_prediction_loss)
from dreamer4.train.common import (EMA, LossCombiner, atomic_torch_save,
                                   set_seed, setup_run_logging,
                                   swap_in_weights, warmup_lr)
from dreamer4.train.config import (AgentTrainConfig, DynamicsModelConfig,
                                   TokenizerRefConfig, config_from_dict,
                                   config_to_dict, parse_config, save_config)
from dreamer4.train.dynamics_objectives import clean_context_loss
from dreamer4.train.train_dynamics import (build_dynamics, encode_episodes,
                                           sample_windows)
from dreamer4.train.train_tokenizer import (make_recon_strip,
                                            weights_from_checkpoint)

# ---------------------------------------------------------------------------
# Model assembly & checkpoints
# ---------------------------------------------------------------------------


def build_agent_modules(cfg: AgentTrainConfig, dyn_model_cfg: DynamicsModelConfig,
                        *, latent_meta: Dict, n_moves: int) -> nn.ModuleDict:
    """The full phase-2 model: finetuned dynamics + fresh agent heads."""
    if cfg.agent.n_agent < 2:
        raise ValueError("n_agent >= 2 required (policy + reward slots; "
                         "slot 2 is the phase-3 value head)")
    dyn = build_dynamics(
        dyn_model_cfg, n_spatial=latent_meta["n_spatial"],
        d_spatial=latent_meta["d_spatial"], action_dim=latent_meta["action_dim"],
        d_proprio=latent_meta["d_proprio"], max_T=latent_meta["max_T"],
        n_agent=cfg.agent.n_agent, space_mode="wm_agent")
    d_model = dyn_model_cfg.d_model
    a = cfg.agent
    return nn.ModuleDict({
        "dynamics": dyn,
        "task": TaskEncoder(num_tasks=a.num_tasks, d_model=d_model,
                            n_agent=a.n_agent),
        "policy": PolicyHead(d_model=d_model, action_dim=1,
                             action_type="discrete", num_categories=n_moves,
                             mtp_length=a.mtp_length,
                             mlp_depth=a.head_mlp_depth,
                             mlp_ratio=a.head_mlp_ratio),
        "reward": RewardHead(d_model=d_model, mtp_length=a.mtp_length,
                             mlp_depth=a.head_mlp_depth,
                             mlp_ratio=a.head_mlp_ratio,
                             num_bins=a.num_bins, bin_low=a.bin_low,
                             bin_high=a.bin_high),
    })


def load_agent_checkpoint(path, *, device: str | torch.device = "cpu",
                          prefer_ema: bool = True
                          ) -> Tuple[nn.ModuleDict, dict]:
    """Rebuild the phase-2 model bundle from a ``train_agent`` checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt.get("kind") == "dreamer4-agent", f"not an agent ckpt: {path}"
    cfg = config_from_dict(AgentTrainConfig, ckpt["config"])
    dyn_model_cfg = config_from_dict(DynamicsModelConfig,
                                     ckpt["dyn_model_config"])
    modules = build_agent_modules(cfg, dyn_model_cfg,
                                  latent_meta=ckpt["latent_meta"],
                                  n_moves=ckpt["latent_meta"]["n_moves"])
    weights = {k: v for k, v in weights_from_checkpoint(ckpt, prefer_ema).items()
               if not k.startswith("cont.")}      # pre-2026-07-24 ckpts
    modules.load_state_dict(weights)
    return modules.to(device).eval(), ckpt


def load_agent_policy(path, *, device: str | torch.device = "cuda",
                      greedy: bool = True) -> Tuple[AgentPolicy, Optional[Dict]]:
    """
    Checkpoint -> ``(policy, env_spec)`` for online evaluation. Open the env
    with ``agent_eval.make_env(env_spec)``.

    The env-specific readers come from the training dataset, which the
    checkpoint records by path: ``proprio_from_info`` (required by a proprio
    model) reads the env state in the same convention its clips used, and
    ``continues_from_reward`` is the rule that turns a predicted reward into
    "the episode ended here". Both are facts about the environment, so
    neither belongs in the deployment loop. A dataset that is not on this
    machine costs only those readouts — unless the model needs proprio, in
    which case it cannot be deployed at all and the error says so.
    """
    modules, ckpt = load_agent_checkpoint(path, device=device)
    meta = ckpt["latent_meta"]
    tok = ckpt["tokenizer_ref"]
    tokenizer = FrozenTokenizer(tok["ckpt"], history=tok["history"],
                                pack_k=tok["pack_k"],
                                decoder_ckpt=(tok["decoder_ckpt"] or None),
                                device=device)
    needs_proprio = modules["dynamics"].d_proprio is not None
    cfg_data = ckpt["config"]["data"]
    try:
        dataset = open_video_dataset(cfg_data["path"],
                                     proprio=cfg_data["proprio"], actions=True)
    except Exception:
        if needs_proprio:
            raise
        dataset = None
    policy = AgentPolicy(
        dynamics=modules["dynamics"], task_encoder=modules["task"],
        policy_head=modules["policy"], tokenizer=tokenizer,
        window=meta["window"], tau_ctx=meta["tau_ctx"], greedy=greedy,
        proprio_from_info=(dataset.proprio_from_info if needs_proprio else None),
        continues_from_reward=(None if dataset is None
                               else dataset.continues_from_reward),
        reward_head=modules["reward"], device=device)
    spec = ckpt.get("env_spec")
    if spec is None and ckpt.get("env_kwargs"):     # pre-2026-08-11 ckpts
        spec = {"id": "GridWorld-v0", "kwargs": ckpt["env_kwargs"]}
    return policy, spec


# ---------------------------------------------------------------------------
# Agent batch sampling (the second forward's data)
# ---------------------------------------------------------------------------


def sample_agent_windows(episodes: List[dict], B: int, T: int,
                         rng: np.random.Generator, device: torch.device, *,
                         bc_frac: float, end_frac: float
                         ) -> Dict[str, torch.Tensor]:
    """
    B windows for the agent forward. Which episodes are worth imitating is
    the dataset's call (``bc_weight``, cached per episode); ``bc_frac`` of
    rows are drawn from those, the rest from ALL episodes — the other heads
    need states an expert never visits, or they would be blind exactly where
    a phase-3 dream wanders. Each draw is end-pinned with probability
    ``end_frac`` so rare terminal frames reach the reward head. Targets carry one extra transition past the window when the
    episode continues (position T-1's n=0 action), masked otherwise.

    Reward targets use ARRIVAL alignment, like the continue flags: position
    t's n=0 target is ``r[t-1]`` — the reward of the transition INTO the
    visible frame. The departure reward ``r[t]`` depends on an action the
    embed cannot see (actions enter as a[t-1]), so predicting it from h_t is
    information-theoretically capped at decision states (measured: goal-F1
    plateaus ~0.85); the arrival reward is fully observable (the goal frame
    shows the player ON the goal). The phase-3 dream annotator reads the
    same head at the newly generated frame.

    Returns {"z": (B,T,Nz,Dz), "actions": (B,T-1,Da) window actions,
    "proprio"?: (B,T,d_p), "tgt_actions": (B,T,1) long, "tgt_mask": (B,T),
    "rewards": (B,T) arrival-aligned, "reward_mask": (B,T) — position 0 has
    no arrival, "continues": (B,T), "bc_row": (B,)}.
    """
    pool = [e for e in episodes if e["z"].shape[0] >= T]
    if not pool:
        raise ValueError(f"no episode has >= {T} frames")
    bc_pool = [e for e in pool if e.get("bc_weight", 1.0) > 0]
    zs, acts, props, tgt_a, tgt_m, rews, conts, bc_rows = \
        [], [], [], [], [], [], [], []
    has_prop = "proprio" in pool[0]
    for b in range(B):
        use_bc = bool(bc_pool) and rng.random() < bc_frac
        e = (bc_pool if use_bc else pool)[
            int(rng.integers(len(bc_pool if use_bc else pool)))]
        n = e["z"].shape[0]
        s = n - T if (rng.random() < end_frac or n == T) \
            else int(rng.integers(0, n - T))
        zs.append(e["z"][s : s + T])
        acts.append(e["actions"][s : s + T - 1])
        if has_prop:
            props.append(e["proprio"][s : s + T])

        # BC targets: action indices s..s+T-1, real while < n-1
        k = min(T, (n - 1) - s)
        moves = e["actions"][s : s + k].argmax(-1)           # one-hot -> ids
        tgt_a.append(np.pad(moves, (0, T - k)))
        tgt_m.append(np.pad(np.ones(k, np.float32), (0, T - k)))
        # reward targets: ARRIVAL alignment — position t <- r[s+t-1]
        rews.append(np.concatenate(
            [[0.0], e["rewards"][s : s + T - 1]]).astype(np.float32))
        term = torch.from_numpy(
            np.asarray(e["terminals"][s : s + T - 1], np.float32))[None]
        conts.append(continues_from_terminals(term, T)[0])
        bc_rows.append(float(e.get("bc_weight", 1.0)))

    reward_mask = torch.ones(B, T, device=device)
    reward_mask[:, 0] = 0.0                      # position 0 has no arrival
    out = {"z": torch.stack(zs).float().to(device),
           "actions": torch.from_numpy(np.stack(acts)).float().to(device),
           "tgt_actions": torch.from_numpy(
               np.stack(tgt_a)).long().unsqueeze(-1).to(device),
           "tgt_mask": torch.from_numpy(np.stack(tgt_m)).to(device),
           "rewards": torch.from_numpy(np.stack(rews)).to(device),
           "reward_mask": reward_mask,
           "continues": torch.stack(conts).to(device),
           "bc_row": torch.tensor(bc_rows, device=device)}
    if has_prop:
        out["proprio"] = torch.from_numpy(np.stack(props)).to(device)
    return out


# ---------------------------------------------------------------------------
# The agent forward (deployment-regime embeddings) + losses
# ---------------------------------------------------------------------------


def agent_head_losses(modules: nn.ModuleDict, batch: Dict[str, torch.Tensor],
                      *, k_max: int, tau_ctx: float, ctx_noise_min: float,
                      ctx_noise_max: float, continues_fn=None
                      ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Push an agent batch through the transformer in the context regime (every
    frame band-corrupted at the tau_ctx signal level — the deployment
    distribution) and score the heads on their slots.
    """
    dyn = modules["dynamics"]
    z, T = batch["z"], batch["z"].shape[1]
    B = z.shape[0]
    device = z.device

    lo = ctx_noise_min if ctx_noise_min > 0 else tau_ctx
    hi = ctx_noise_max if ctx_noise_max > lo else lo
    tc = torch.empty(B, T, 1, 1, device=device).uniform_(lo, hi) \
        if hi > lo else lo
    z_c = (1 - tc) * z + tc * torch.randn_like(z)

    emax = _log2_int(k_max)
    step_idx = torch.full((B, T), emax, device=device, dtype=torch.long)
    ctx_sig = min(int(round((1 - tau_ctx) * k_max)), k_max)
    signal_idx = torch.full((B, T), ctx_sig, device=device, dtype=torch.long)
    aligned = align_actions(batch["actions"], T)
    task_ids = torch.zeros(B, dtype=torch.long, device=device)
    agent_in = modules["task"](task_ids).expand(B, T, -1, -1)

    kwargs = {}
    if dyn.d_proprio is not None:
        prop = batch["proprio"]
        tc_p = tc[..., 0] if torch.is_tensor(tc) else tc
        kwargs["proprio_noisy"] = (1 - tc_p) * prop \
            + tc_p * torch.randn_like(prop)
    out = dyn(aligned, step_idx, signal_idx, z_c, agent_tokens=agent_in,
              **kwargs)
    agent_out = out[-1]                                  # (B, T, n_agent, D)

    bc, bc_info = behavior_cloning_loss(
        modules["policy"], agent_out[:, :, POLICY_SLOT],
        batch["tgt_actions"], batch["tgt_mask"], row_weight=batch["bc_row"])
    rew, rew_info = reward_prediction_loss(
        modules["reward"], agent_out[:, :, REWARD_SLOT],
        batch["rewards"], batch["reward_mask"])
    info = {**bc_info, **rew_info}
    with torch.no_grad():
        # THE gate on the derived termination signal: with no continue head,
        # phase 3 stops its dreams with the DATASET's terminal rule
        # (EpisodeVideoDataset.continues_from_reward), so its quality must be
        # measured here on held-out data. Domains without such a rule get no
        # term_f1 and will need a real terminal predictor.
        r_pred = modules["reward"].twohot.decode(
            modules["reward"].forward_all(agent_out[:, :, REWARD_SLOT])[0])
        valid = batch["reward_mask"] > 0
        derived = None if continues_fn is None else continues_fn(r_pred)
        for name, truth in (("goal", (batch["rewards"] > 0.5) & valid),
                            ("term", (batch["continues"] < 0.5) & valid)):
            if not truth.any() or derived is None:
                continue
            pred = (derived < 0.5) & valid
            tp = float((truth & pred).sum())
            prec = tp / max(float(pred.sum()), 1.0)
            rec = tp / max(float(truth.sum()), 1.0)
            info[f"{name}_f1"] = 2 * prec * rec / max(prec + rec, 1e-8)
            info[f"{name}_recall"] = rec
    return {"bc": bc, "reward": rew}, info


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: AgentTrainConfig) -> Dict[str, float]:
    """Run phase-2 agent finetuning to completion; returns final metrics."""
    if not cfg.init_from:
        raise SystemExit("--init_from is required (a train_dynamics "
                         "checkpoint to finetune)")
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logging(out)
    save_config(cfg, out / "config.yaml")
    writer = SummaryWriter(log_dir=str(out / "tb"))

    # ---- the warm-start dynamics checkpoint ------------------------------
    init = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    assert init.get("kind") == "dreamer4-dynamics", \
        f"--init_from must be a train_dynamics checkpoint, got {cfg.init_from}"
    latent_meta = dict(init["latent_meta"])
    dyn_model_cfg = config_from_dict(DynamicsModelConfig,
                                     init["config"]["model"])
    tok_ref = (cfg.tokenizer if cfg.tokenizer.ckpt else
               config_from_dict(TokenizerRefConfig, init["config"]["tokenizer"]))

    tokenizer = FrozenTokenizer(
        tok_ref.ckpt, history=tok_ref.history, pack_k=tok_ref.pack_k,
        decoder_ckpt=(tok_ref.decoder_ckpt or None), device=device)
    log.info(f"warm start: {cfg.init_from} (step {init.get('step', '?')}) | "
             f"tokenizer {tok_ref.ckpt}")

    # ---- data (must carry rewards) ---------------------------------------
    dataset = open_video_dataset(cfg.data.path, proprio=cfg.data.proprio,
                                 actions=True,
                                 episode_cache=cfg.data.episode_cache)
    if dataset.action_dim is None:
        raise SystemExit(f"{cfg.data.path} carries no actions")
    probe = dataset.clip(0, 0, 2)
    if "rewards" not in probe:
        raise SystemExit(f"{cfg.data.path} carries no rewards — phase 2 "
                         "needs a reward-annotated dataset")
    n_moves = dataset.action_dim
    if latent_meta["action_dim"] != n_moves + 1:
        raise SystemExit(
            f"dataset action_dim {n_moves}+1 != checkpoint "
            f"action_dim {latent_meta['action_dim']}")
    env_spec = dataset.env_spec()
    if env_spec is None:
        log.info("dataset has no env_spec — online evaluation disabled")

    train_eps, val_eps = split_train_val(len(dataset),
                                         val_frac=cfg.data.val_frac,
                                         seed=cfg.seed, min_val=64)
    t_enc = time.time()
    train_cache = encode_episodes(tokenizer, dataset, train_eps)
    val_cache = encode_episodes(tokenizer, dataset, val_eps, keep_video=True)
    n_frames = sum(e["z"].shape[0] for e in train_cache)
    n_bc = sum(e.get("bc_weight", 1.0) > 0 for e in train_cache)
    log.info(f"device={device} episodes={len(dataset)} "
             f"(train {len(train_cache)} / val {len(val_cache)}) | "
             f"BC-eligible {n_bc}/{len(train_cache)} (the dataset's call) | "
             f"pre-encoded {n_frames} frames in {time.time() - t_enc:.1f}s")

    # ---- model -----------------------------------------------------------
    modules = build_agent_modules(cfg, dyn_model_cfg, latent_meta=latent_meta,
                                  n_moves=n_moves).to(device)
    # warm-start the transformer from the EMA weights (the gated artifact);
    # the agent segment (masks are parameter-free) adds nothing to load
    missing, unexpected = modules["dynamics"].load_state_dict(
        weights_from_checkpoint(init, prefer_ema=True), strict=False)
    assert not unexpected, f"unexpected keys in warm start: {unexpected}"
    assert not missing, f"missing keys in warm start: {missing}"
    n_params = sum(p.numel() for p in modules.parameters())
    n_head = n_params - sum(p.numel() for p in modules["dynamics"].parameters())
    log.info(f"params: {n_params / 1e6:.2f}M ({n_head / 1e6:.2f}M fresh heads) "
             f"| n_agent={cfg.agent.n_agent} mtp={cfg.agent.mtp_length}")

    obj = cfg.objective
    use_boot = obj.bootstrap_frac > 0.0
    d_proprio = latent_meta["d_proprio"]
    weights = {"flow": 1.0,
               "proprio": obj.proprio_weight if d_proprio is not None else 0.0,
               "bootstrap": obj.boot_weight if use_boot else 0.0,
               "bc": cfg.loss.bc_weight,
               "reward": cfg.loss.reward_weight}
    combiner = LossCombiner(weights, normalize=True, decay=0.99,
                            floor_frac=0.2, device=device)
    log.info("losses: " + "  ".join(f"{k}={v}" for k, v in weights.items() if v)
             + "  [RMS-normalized]")

    head_params = [p for name, m in modules.items() if name != "dynamics"
                   for p in m.parameters()]
    opt = torch.optim.AdamW(
        [{"params": list(modules["dynamics"].parameters()),
          "lr": cfg.optim.lr},
         {"params": head_params, "lr": cfg.head_lr}],
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        weight_decay=cfg.optim.weight_decay)
    base_lrs = [cfg.optim.lr, cfg.head_lr]
    ema = EMA(modules, cfg.optim.ema_decay) if cfg.optim.ema_decay > 0 else None

    latent_meta.update({"n_agent": cfg.agent.n_agent, "n_moves": n_moves,
                        "num_tasks": cfg.agent.num_tasks,
                        "mtp_length": cfg.agent.mtp_length,
                        "tau_ctx": obj.tau_ctx,
                        "window": cfg.data.seq_len})

    start_step = 0
    latest = ckpt_dir / "latest.pt"
    if cfg.resume and latest.exists():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        modules.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        combiner.load_state_dict(ckpt.get("loss_norm"))
        start_step = int(ckpt["step"])
        log.info(f"resumed from {latest} at step {start_step}")

    def save_checkpoint(step: int) -> None:
        atomic_torch_save(latest, {
            "kind": "dreamer4-agent", "step": step,
            "config": config_to_dict(cfg),
            "dyn_model_config": config_to_dict(dyn_model_cfg),
            "latent_meta": latent_meta,
            "tokenizer_ref": {"ckpt": tok_ref.ckpt,
                              "decoder_ckpt": tok_ref.decoder_ckpt,
                              "history": tok_ref.history,
                              "pack_k": tok_ref.pack_k},
            "env_spec": env_spec, "init_from": cfg.init_from,
            "model": modules.state_dict(),
            "ema": None if ema is None else ema.state_dict(),
            "opt": opt.state_dict(), "loss_norm": combiner.state_dict()})

    # ---- fixed validation batches ----------------------------------------
    val_rng = np.random.default_rng(cfg.seed + 1)
    gate_batch = sample_windows(
        val_cache, cfg.eval.episodes, cfg.eval.ctx + cfg.eval.horizon, val_rng,
        device, start_only=True, with_video=True, replace=False)
    onestep_batch = sample_windows(val_cache, min(64, cfg.eval.episodes),
                                   cfg.data.seq_len, val_rng, device)
    agent_kw = dict(bc_frac=cfg.data.bc_frac, end_frac=cfg.data.end_frac)
    val_agent_batch = sample_agent_windows(
        val_cache, 256, cfg.data.seq_len, val_rng, device, **agent_kw)
    val_end_batch = sample_agent_windows(
        val_cache, 256, cfg.data.seq_len, val_rng, device,
        **{**agent_kw, "end_frac": 1.0})

    dyn = modules["dynamics"]
    k_max = dyn_model_cfg.k_max

    def run_validation(step: int) -> Dict[str, float]:
        eval_weights = ema.state_dict() if ema is not None else None
        with swap_in_weights(modules, eval_weights):
            modules.eval()
            with torch.no_grad():
                metrics, (gt, dream) = gate_rollout(
                    dyn, tokenizer, dataset, gate_batch,
                    window=cfg.data.seq_len, ctx=cfg.eval.ctx,
                    K=dyn_model_cfg.K, k_max=k_max, tau_ctx=obj.tau_ctx)
                metrics.update(one_step_eval(
                    dyn, onestep_batch["z"], onestep_batch["actions"],
                    ctx=cfg.eval.ctx, K=dyn_model_cfg.K, k_max=k_max,
                    tau_ctx=obj.tau_ctx, proprio=onestep_batch.get("proprio")))
                for name, batch in (("", val_agent_batch),
                                    ("end_", val_end_batch)):
                    terms, info = agent_head_losses(
                        modules, batch, k_max=k_max, tau_ctx=obj.tau_ctx,
                        ctx_noise_min=obj.ctx_noise_min,
                        ctx_noise_max=obj.ctx_noise_max,
                        continues_fn=dataset.continues_from_reward)
                    metrics[f"{name}bc_nll"] = float(terms["bc"].detach())
                    for k, v in info.items():
                        metrics[f"{name}{k}"] = v
            modules.train()
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", float(v), step)
        writer.add_image("val/rollout_gt_vs_dream",
                         make_recon_strip(gt, dream), step)
        log.info("  [val] step %d: %s" % (step, "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items())))
        return metrics

    online_env = make_env(env_spec) if env_spec else None

    def run_online(episodes: int, greedy: bool, seed: int) -> Dict[str, float]:
        eval_weights = ema.state_dict() if ema is not None else None
        with swap_in_weights(modules, eval_weights):
            modules.eval()
            policy = AgentPolicy(
                dynamics=dyn, task_encoder=modules["task"],
                policy_head=modules["policy"], tokenizer=tokenizer,
                window=cfg.data.seq_len, tau_ctx=obj.tau_ctx, greedy=greedy,
                proprio_from_info=dataset.proprio_from_info, device=device)
            if not greedy:
                torch.manual_seed(seed)
            metrics, _ = evaluate_policy(policy, online_env, episodes=episodes,
                                         seed=seed)
            modules.train()
        return metrics

    # ---- loop ------------------------------------------------------------
    rng = np.random.default_rng(cfg.seed + 2)
    modules.train()
    t0 = time.time()
    budget_s = cfg.max_minutes * 60.0
    loss_ema: Optional[float] = None
    step = start_step

    for step in range(start_step + 1, cfg.steps + 1):
        if budget_s and (time.time() - t0) >= budget_s:
            log.info(f"[budget] {cfg.max_minutes} min reached at step {step - 1}")
            step -= 1
            break
        warm = warmup_lr(1.0, step, cfg.optim.warmup)
        for group, base in zip(opt.param_groups, base_lrs):
            group["lr"] = base * warm

        sched_p = obj.sched_sample_prob * min(
            1.0, step / max(1, int(obj.sched_warmup_frac * cfg.steps)))
        do_sched = rng.random() < sched_p
        image_batch = rng.random() < obj.image_batch_prob
        T_s = 1 if image_batch else cfg.data.seq_len

        opt.zero_grad(set_to_none=True)
        loss_value = 0.0
        raw_means: Dict[str, float] = {}
        info: Dict[str, float] = {}
        for _ in range(cfg.optim.accum_steps):
            # 1) the retained phase-1 WM objective, verbatim
            wm_batch = sample_windows(
                train_cache, cfg.data.batch_size, T_s, rng, device,
                start_only=image_batch and obj.image_batch_frame0)
            aligned = align_actions(wm_batch["actions"], T_s)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=cfg.optim.amp):
                p_tgt = 0 if image_batch else int(rng.integers(1, T_s))
                terms, aux = clean_context_loss(
                    dyn, wm_batch["z"], aligned, p=p_tgt, k_max=k_max,
                    K=dyn_model_cfg.K, tau_ctx=obj.tau_ctx,
                    ctx_noise_min=obj.ctx_noise_min or None,
                    ctx_noise_max=obj.ctx_noise_max or None,
                    do_sched=do_sched and not image_batch,
                    proprio=wm_batch.get("proprio"),
                    bootstrap_frac=obj.bootstrap_frac)

                # 2) the agent forward (deployment-regime embeddings)
                agent_batch = sample_agent_windows(
                    train_cache, cfg.data.agent_batch_size, cfg.data.seq_len,
                    rng, device, **agent_kw)
                agent_terms, info = agent_head_losses(
                    modules, agent_batch, k_max=k_max, tau_ctx=obj.tau_ctx,
                    ctx_noise_min=obj.ctx_noise_min,
                    ctx_noise_max=obj.ctx_noise_max,
                    continues_fn=dataset.continues_from_reward)
                terms.update(agent_terms)
                loss, raw = combiner(terms)
            (loss / cfg.optim.accum_steps).backward()
            loss_value += float(loss.detach()) / cfg.optim.accum_steps
            for k, v in raw.items():
                raw_means[k] = raw_means.get(k, 0.0) + v / cfg.optim.accum_steps
        raw = raw_means
        grad_norm = torch.nn.utils.clip_grad_norm_(modules.parameters(),
                                                   cfg.optim.grad_clip)
        opt.step()
        if ema is not None:
            ema.update(modules)

        loss_ema = loss_value if loss_ema is None else \
            0.98 * loss_ema + 0.02 * loss_value
        if step % cfg.log_every == 0:
            sps = (step - start_step) / max(time.time() - t0, 1e-9)
            for k, v in raw.items():
                writer.add_scalar(f"train/{k}", v, step)
            for k in ("bc_acc", "reward_mae", "goal_f1", "term_recall"):
                if k in info:
                    writer.add_scalar(f"train/{k}", info[k], step)
            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("train/steps_per_sec", sps, step)
            log.info(f"step {step}/{cfg.steps} "
                     + " ".join(f"{k}={v:.4f}" for k, v in raw.items())
                     + (f" bc_acc={info['bc_acc']:.3f}" if "bc_acc" in info else "")
                     + f" loss_ema={loss_ema:.4f} gnorm={float(grad_norm):.2f}"
                     f" {sps:.1f}it/s")

        if step % cfg.val_every == 0:
            run_validation(step)
        if online_env is not None and cfg.online.every \
                and step % cfg.online.every == 0:
            online = run_online(cfg.online.episodes, True, cfg.online.seed)
            for k, v in online.items():
                writer.add_scalar(f"online/{k}", float(v), step)
            log.info("  [online greedy] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in online.items()))
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            save_checkpoint(step)

    # ---- final: gate table + offline agent metrics + ONLINE eval ---------
    save_checkpoint(step)
    val_metrics = run_validation(step)
    eval_weights = ema.state_dict() if ema is not None else None
    horizons = [int(h) for h in cfg.eval.final_horizons.split(",")]
    Ks = [int(k) for k in cfg.eval.final_Ks.split(",")]
    rows: List[Dict[str, float]] = []
    with swap_in_weights(modules, eval_weights):
        modules.eval()
        with torch.no_grad():
            for horizon in horizons:
                try:
                    batch = sample_windows(val_cache, cfg.eval.episodes,
                                           cfg.eval.ctx + horizon, val_rng,
                                           device, start_only=True,
                                           with_video=True, replace=False)
                except ValueError:
                    log.info(f"  [gate] horizon {horizon}: no long-enough episodes")
                    continue
                for K in Ks:
                    metrics, _ = gate_rollout(
                        dyn, tokenizer, dataset, batch, window=cfg.data.seq_len,
                        ctx=cfg.eval.ctx, K=K, k_max=k_max, tau_ctx=obj.tau_ctx)
                    rows.append({"horizon": horizon, "K": K,
                                 "episodes": batch["z"].shape[0], **metrics})
                    log.info(f"  [gate] H={horizon} K={K}: " + "  ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items()))
        modules.train()

    online: Dict[str, Dict[str, float]] = {}
    if online_env is not None:
        for mode, greedy in (("greedy", True), ("sample", False)):
            online[mode] = run_online(cfg.online.episodes, greedy,
                                      cfg.online.seed)
            log.info(f"  [online {mode}] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in online[mode].items()))
        online["random"] = evaluate_random(online_env,
                                           episodes=cfg.online.episodes,
                                           seed=cfg.online.seed)
        log.info("  [online random] " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in online["random"].items()))

    primary = rows[-1] if rows else {}
    minutes = (time.time() - t0) / 60.0
    final = {**primary,
             **{k: v for k, v in val_metrics.items() if k not in primary},
             "online": online, "gate": rows, "steps": step,
             "params_M": n_params / 1e6, "minutes": minutes}
    (out / "final.json").write_text(json.dumps(final, indent=2))

    record = {"name": out.name, "tag": cfg.tag, "kind": "agent",
              "data": cfg.data.path, "init_from": cfg.init_from,
              "n_agent": cfg.agent.n_agent, "mtp": cfg.agent.mtp_length,
              "bc_frac": cfg.data.bc_frac,
              "steps": step, "minutes": round(minutes, 1),
              "bc_acc": round(val_metrics.get("bc_acc", -1.0), 4),
              "goal_f1": round(val_metrics.get("end_goal_f1", -1.0), 4),
              "term_f1": round(val_metrics.get("end_term_f1", -1.0), 4),
              **({"success_greedy": round(online["greedy"]["success_rate"], 4),
                  "success_sample": round(online["sample"]["success_rate"], 4),
                  "steps_over_optimal": round(
                      online["greedy"].get("steps_over_optimal", -1.0), 4)}
                 if online else {}),
              **{k: round(float(v), 5) for k, v in primary.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)},
              **({"gate_pass": primary["gate_pass"]}
                 if "gate_pass" in primary else {})}
    with open(out.parent / "experiments.jsonl", "a") as journal:
        journal.write(json.dumps(record) + "\n")

    writer.close()
    log.info("DONE " + json.dumps(
        {k: v for k, v in final.items() if k not in ("gate",)}, default=str))
    return final


def main(argv=None) -> None:
    cfg = parse_config(AgentTrainConfig, argv,
                       description=__doc__.split("\n\n")[0])
    if not cfg.data.path:
        raise SystemExit("--data.path is required (a dataset with actions "
                         "and rewards)")
    train(cfg)


if __name__ == "__main__":
    main()
