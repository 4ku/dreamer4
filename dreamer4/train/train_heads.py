"""Phase 2 -- turn the pretrained world model into an agent.

    Phase 2: Agent Finetuning -- Finetune world model with task inputs for policy and reward heads
    using (7) and (9).                                               (paper Algorithm 1)

This is the paper's Section 3.3. AGENT tokens are interleaved into the
dynamics transformer, carry a task embedding as input, attend to every other modality while nothing
attends back, and the policy and reward heads are small MLPs on their output embeddings ``h_t``
with one output layer per multi-token-prediction distance (``--mtp``, the paper's L = 8). The whole
transformer is finetuned: the video-prediction loss (Eq. 7) keeps running on noisy representations
so the world model does not forget what it is, and the agent loss (Eq. 9) is added to it, with
every term RMS-normalised as the paper's Section 3 prescribes. Because attention is block-causal in
time, ``h_t`` is a function of the whole context window -- so phase 2 samples CLIPS, not frames.

The data mixture follows Section 4.1: half of each agent batch is drawn from the episodes the
DATASET calls demonstrations and half from all episodes, the behavioral-cloning term is applied
only to the demonstration half, and the video-prediction term only to windows drawn from the whole
collection, so imagination is never trained to be optimistic about states it reached by accident.

Model selection is by REAL-ENV success: every ``--eval_every`` steps the policy plays ``--eval_n``
held-out layouts greedy and sampled, and the best greedy success rate (ties: shorter paths) is kept
as ``best.pt``. Clone accuracy is logged and decides nothing. A dataset with no live environment
(``env_spec() is None``) trains fine and simply skips that evaluation.

Writes to ``--out``: ``config.json``, ``checkpoints/{latest,best}.pt``, ``final.json`` and
TensorBoard scalars under ``tb/`` (``bc/*``, ``reward/*``, ``dyn/*``, ``eval/*``). The checkpoint
carries the finetuned dynamics weights as well as the heads, because after phase 2 the policy and
the world model are the same network.

    python -m dreamer4.train.train_heads --dyn <phase-1 ckpt> --out runs/heads
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dreamer4.agent_eval import evaluate, make_env
from dreamer4.models.agent_heads import (agent_tokens_are_write_only, build_agent_heads,
                                         flatten_agent_out, mtp_targets, pretty_agent_summary,
                                         terminal_bins)
from dreamer4.models.dynamics import _log2_int, align_actions
from dreamer4.train.common import LossCombiner
from dreamer4.train.dynamics_objectives import clean_context_loss
from dreamer4.train.train_dynamics import encode_episodes, sample_windows
from dreamer4.models.world_model import AgentPolicy, WorldModel, window_frames


def param_count(module):
    return sum(p.numel() for p in module.parameters())


def check_memory_budget(dataset, tokenizer, device, log=print):
    """Log how much GPU memory the encoded dataset will hold, and refuse to start if it will not fit.

    The whole dataset is kept resident -- every frame's latents and proprio -- which is what makes
    tens of thousands of steps take minutes instead of hours. It is also the one assumption in this
    trainer that does not scale: past a few tens of gigabytes the right answer is a sampler over CPU
    tensors, and the failure should say so rather than arrive as a CUDA OOM in the middle of
    encoding.
    """
    frames = sum(dataset.episode_frames(i) for i in range(len(dataset)))
    per_frame = tokenizer.n_spatial * tokenizer.d_spatial * 2          # latents, fp16
    per_frame += (dataset.proprio_dim or 0) * 4                        # proprio, fp32
    need = frames * per_frame
    log(f"encoded dataset: {frames} frames, {need / 2**30:.2f} GiB on {device}", flush=True)
    if device.type != "cuda":
        return
    free = torch.cuda.mem_get_info(device)[0]
    if need > 0.8 * free:
        raise RuntimeError(
            f"the encoded dataset needs {need / 2**30:.2f} GiB but only {free / 2**30:.2f} GiB is "
            f"free on {device}: train on a subset of the episodes, or stream clips from CPU")


def demonstration_ids(episodes, n_train, bc_frac, bc_seed, log=print):
    """Which episodes may be imitated -- the DATASET's criterion, optionally thinned by ``bc_frac``.

    ``bc_weight(i) > 0`` is the dataset's own judgement of what a demonstration is; the trainer
    never inspects episode quality itself. The seeded subset exists because cloning every
    demonstration can already solve an easy task outright and leave phase 3 nothing to improve.
    """
    demos = [i for i in range(n_train) if episodes[i]["bc_weight"] > 0]
    if bc_frac < 1.0:
        rng = np.random.default_rng(bc_seed)
        pick = rng.choice(len(demos), int(round(bc_frac * len(demos))), replace=False)
        demos = sorted(demos[j] for j in pick)
    if not demos:
        raise ValueError("no training episode passes the dataset's bc_weight criterion, so there "
                         "is nothing to clone (see EpisodeVideoDataset.bc_weight)")
    ratios = [len(episodes[i]["rewards"]) / episodes[i]["meta"]["optimal_steps"]
              for i in demos
              if episodes[i].get("meta", {}).get("optimal_steps", 0) > 0]
    quality = f" | their path / shortest: mean {np.mean(ratios):.3f}" if ratios else ""
    log(f"demonstrations {len(demos)} of {n_train} training episodes (bc_frac {bc_frac})"
        f"{quality}", flush=True)
    return demos


def add_returns(episodes, gamma):
    """Attach ``ret``: the discounted return-to-go the RECORDED behaviour collected from each frame.

    ``ret[t] = r_t + gamma * ret[t+1]`` with ``r_t`` the reward of the ``t -> t+1`` transition and
    zero after the last frame. It is the value of the behaviour that produced the episode, not of
    the policy being trained -- which is why it is only a pretraining target for the critic
    (``--value_weight``): phase 3 starts from a value head, and from agent features, that already
    know how far the reward is, instead of from uniform logits.
    """
    for ep in episodes:
        r = np.asarray(ep["rewards"], np.float32)
        g = np.zeros(len(r) + 1, np.float32)
        for t in range(len(r) - 1, -1, -1):
            g[t] = r[t] + gamma * g[t + 1]
        ep["ret"] = g[:ep["z"].shape[0]] if len(g) >= ep["z"].shape[0] else np.pad(g, (0, ep["z"].shape[0] - len(g)))


def sample_clips(episodes, ids, B, T, rng, device):
    """``B`` context windows of ``T`` slots, each ending at a uniformly chosen frame.

    The window convention is :mod:`dreamer4.models.world_model`'s, and it is the same one imagination and
    the live evaluator use: slot ``j`` holds frame ``end - (T-1) + j``, clamped at the episode
    start so an early window is left-padded by repeating frame 0, and the aligned action of a slot
    is the action that LED INTO it (the start flag at slot 0 and at every padded slot).

    Ending the window at a uniformly drawn frame -- rather than starting it at one -- is what makes
    the training distribution of contexts identical to the evaluation one, short windows at the
    beginning of an episode included, and at the same frequency.

    Returns a dict of device tensors:
        ``z`` (B,T,Nz,Dz), ``proprio`` (B,T,Dp), ``actions`` (B,T,Da+1) aligned,
        ``a_id`` (B,T) long -- the action taken FROM the slot's frame (the policy target),
        ``valid_a`` (B,T), ``reward`` (B,T) -- the reward that ARRIVED WITH the slot's frame,
        ``valid_r`` (B,T), ``task`` (B,) long -- the episode's task indicator,
        ``ret`` (B,T) -- the recorded discounted return-to-go from the slot's frame (zeros unless
        :func:`add_returns` ran), ``valid_v`` (B,T) -- 0 on a padded slot.
    """
    pick = rng.choice(np.asarray(ids), size=B, replace=True)
    n_frames = np.array([episodes[int(i)]["z"].shape[0] for i in pick])
    ends = (rng.random(B) * n_frames).astype(np.int64).clip(max=n_frames - 1)
    slots = window_frames(ends, T)                                        # (B, T)
    fresh = np.ones((B, T), bool)
    fresh[:, 1:] = slots[:, 1:] != slots[:, :-1]                          # False on a padded slot

    n_lat, d_lat = episodes[int(pick[0])]["z"].shape[1:]
    d_prop = episodes[int(pick[0])]["proprio"].shape[-1]
    d_act = episodes[int(pick[0])]["actions"].shape[-1]
    z = torch.empty(B, T, n_lat, d_lat, dtype=torch.float32, device=device)
    proprio = np.zeros((B, T, d_prop), np.float32)
    actions = np.zeros((B, T, d_act + 1), np.float32)
    a_id = np.zeros((B, T), np.int64)
    valid_a = np.zeros((B, T), np.float32)
    reward = np.zeros((B, T), np.float32)
    valid_r = np.zeros((B, T), np.float32)
    ret = np.zeros((B, T), np.float32)
    # The paper's one-hot task indicator. A collector that records one puts it in the episode's
    # metadata (``EpisodeVideoDataset.episode_meta``); a single-task dataset records none and every
    # clip is task 0.
    task = np.array([int(episodes[int(i)].get("meta", {}).get("task", 0)) for i in pick], np.int64)
    for b in range(B):
        ep = episodes[int(pick[b])]
        s = slots[b]
        z[b] = ep["z"][torch.as_tensor(s, device=ep["z"].device)].float()
        proprio[b] = ep["proprio"][s]
        n_act = len(ep["actions"])
        actions[b, 0, d_act] = 1.0
        for j in range(1, T):
            if fresh[b, j]:
                actions[b, j, :d_act] = ep["actions"][s[j] - 1]
            else:
                actions[b, j, d_act] = 1.0
        act_here = s < n_act
        a_id[b] = np.where(act_here, ep["actions"][np.minimum(s, n_act - 1)].argmax(-1), 0)
        valid_a[b] = (act_here & fresh[b]).astype(np.float32)
        rew_here = (s >= 1) & (s - 1 < n_act)
        reward[b] = np.where(rew_here, ep["rewards"][np.maximum(s - 1, 0)], 0.0)
        valid_r[b] = (rew_here & fresh[b]).astype(np.float32)
        if "ret" in ep:
            ret[b] = ep["ret"][s]
    as_t = (lambda x: torch.as_tensor(x, device=device))
    return {"z": z, "proprio": as_t(proprio), "actions": as_t(actions), "a_id": as_t(a_id),
            "valid_a": as_t(valid_a), "reward": as_t(reward), "valid_r": as_t(valid_r),
            "task": as_t(task), "ret": as_t(ret), "valid_v": as_t(fresh.astype(np.float32))}


def agent_forward(world, heads, clip, *, tau_ctx, ctx_signal, emax):
    """One transformer pass over a clip batch, returning ``h_t`` for every slot.

    "To preserve existing capabilities, we reuse the pretraining setting with this additional loss
    function, so the representations are noisy and we continue to apply the video prediction loss."
    (paper Section 3.3) -- here the representations sit at the pretraining CONTEXT noise level,
    which is the level every frame the policy will ever act on has: an imagined frame once it is
    committed, and a real frame at evaluation.
    """
    z, prop = clip["z"], clip["proprio"]
    B, T = z.shape[:2]
    dev = z.device
    z_in = (1 - tau_ctx) * z + tau_ctx * torch.randn_like(z)
    p_in = (1 - tau_ctx) * prop + tau_ctx * torch.randn_like(prop)
    step = torch.full((B, T), emax, device=dev, dtype=torch.long)
    sig = torch.full((B, T), ctx_signal, device=dev, dtype=torch.long)
    ag = heads.tokens((B, T), task_ids=clip["task"], device=dev)
    out = world.dyn(clip["actions"], step, sig, z_in, agent_tokens=ag, proprio_noisy=p_in)
    return flatten_agent_out(out[-1])


def agent_losses(heads, h, clip, *, mtp, demo_mask):
    """Paper Eq. 9 -- ``- sum_n ln p(a_{t+n} | h_t) - sum_n ln p(r_{t+n} | h_t)``.

    The two sums are returned separately so the RMS normaliser can hold one running scale per term.
    ``demo_mask`` (B,) restricts the behavioral-cloning sum to the relevant half of the mixture
    (Section 4.1); the reward sum runs over the whole batch, because a state that pays nothing is
    exactly what the failures teach.
    """
    a_tgt, a_msk = mtp_targets(clip["a_id"], clip["valid_a"], mtp)
    r_tgt, r_msk = mtp_targets(clip["reward"], clip["valid_r"], mtp)
    a_msk = a_msk * demo_mask.reshape(-1, 1, 1)
    return (heads.policy.clone_loss(heads.policy(h), a_tgt, a_msk),
            heads.reward.loss(heads.reward(h), r_tgt, r_msk))


@torch.no_grad()
def paper_metrics(world, heads, episodes, ids, *, T, batch, n_batches, rng, device,
                  term_bin, term_prob, tau_ctx, ctx_signal, emax,
                  continues_from_reward):
    """Held-out clone accuracy, reward MAE and the terminal precision / recall / F1.

    The F1 is the gate on phase 3: a dream ends when the reward head's own distribution says the
    episode ended, so a rule that misses terminal states -- or invents them -- is phase 3
    optimising the wrong task.
    """
    was_heads, was_dyn = heads.training, world.dyn.training
    heads.eval()
    world.dyn.eval()
    hits = seen = 0
    mae, wr, tp, fp, fn = 0.0, 0, 0, 0, 0
    bins = heads.reward.twohot.bin_values
    for _ in range(n_batches):
        clip = sample_clips(episodes, ids, batch, T, rng, device)
        h = agent_forward(world, heads, clip, tau_ctx=tau_ctx,
                          ctx_signal=ctx_signal, emax=emax)
        va = clip["valid_a"] > 0
        pred = heads.policy.at(h, 0).argmax(-1)
        hits += int(((pred == clip["a_id"]) & va).sum())
        seen += int(va.sum())
        vr = clip["valid_r"] > 0
        pr = heads.reward.at(h, 0).float().softmax(-1)
        mae += float((((pr * bins).sum(-1) - clip["reward"]).abs() * vr).sum())
        wr += int(vr.sum())
        is_term = (pr * term_bin).sum(-1) > term_prob
        true = (continues_from_reward(clip["reward"].reshape(1, -1).cpu()).reshape(
            clip["reward"].shape).to(device) < 0.5)
        tp += int((is_term & true & vr).sum())
        fp += int((is_term & ~true & vr).sum())
        fn += int((~is_term & true & vr).sum())
    heads.train(was_heads)
    world.dyn.train(was_dyn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"acc": hits / max(1, seen), "mae": mae / max(1, wr), "term_precision": prec,
            "term_recall": rec, "term_f1": 2 * prec * rec / max(1e-9, prec + rec)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Dreamer 4 phase 2: agent finetuning")
    ap.add_argument("--dyn", required=True,
                    help="phase-1 dynamics checkpoint: tokenizer, dataset and env spec come from it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32, help="clips per agent step")
    ap.add_argument("--lr", type=float, default=5e-4, help="learning rate of the agent heads")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--reward_weight", type=float, default=1.0)
    ap.add_argument("--value_weight", type=float, default=1.0,
                    help="NOT in the paper (there the value head is born in phase 3): weight of a "
                         "critic-pretraining term, the value head regressed on the recorded "
                         "return-to-go of ALL training episodes. It makes the agent features carry "
                         "the distance to the reward and hands phase 3 a critic that is not blank; "
                         "without it a fast phase 3 can collapse. 0 = the paper's objective")
    ap.add_argument("--gamma", type=float, default=0.997,
                    help="discount of the recorded returns for --value_weight; use phase 3's")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="last fraction of the episodes, held out for the agent metrics")
    ap.add_argument("--bc_frac", type=float, default=1.0,
                    help="fraction of the dataset's demonstrations to clone")
    ap.add_argument("--bc_seed", type=int, default=0)
    ap.add_argument("--reward_bins", type=int, default=255)
    ap.add_argument("--reward_bin_low", type=float, default=-1.0,
                    help="two-hot range in symlog space; must cover the environment's rewards")
    ap.add_argument("--reward_bin_high", type=float, default=1.0)
    ap.add_argument("--term_prob", type=float, default=0.5,
                    help="terminal-rule threshold used by the held-out terminal metrics")
    ap.add_argument("--n_agent", type=int, default=8,
                    help="agent tokens per time step, concatenated into h_t (the paper gives no "
                         "number). With ONE the critic of phase 3 resolves the distance to the "
                         "reward only to +-1 cell on the gridworld and PMPO, which reads the sign of "
                         "the advantage, stalls at path/shortest 1.15; 4-8 remove that")
    ap.add_argument("--n_tasks", type=int, default=1,
                    help="minimum size of the task-embedding table; the trainer widens it to cover "
                         "every task id the dataset's episode metadata records")
    ap.add_argument("--mtp", type=int, default=8,
                    help="multi-token-prediction length L (paper Eq. 9); 1 makes phase 2 cheap")
    ap.add_argument("--clip_T", type=int, default=0,
                    help="context slots the agent token sees; 0 = the window the dynamics model "
                         "was trained on (its data.seq_len), the longest context phase 3 can "
                         "dream with")
    ap.add_argument("--d_hidden", type=int, default=256, help="width of the head MLPs")
    ap.add_argument("--dyn_lr", type=float, default=5e-5,
                    help="learning rate of the transformer being finetuned; 0 freezes it")
    ap.add_argument("--dyn_warmup", type=int, default=0,
                    help="steps over which the TRANSFORMER's learning rate ramps up from 0 (the "
                         "heads train at their full rate from step 1)")
    ap.add_argument("--dyn_batch", type=int, default=32,
                    help="windows per video-prediction step (paper Eq. 7)")
    ap.add_argument("--dyn_weight", type=float, default=1.0,
                    help="weight of the video-prediction term; 0 drops it (and the paper's "
                         "'preserve existing capabilities' with it)")
    ap.add_argument("--boot_frac", type=float, default=0.5,
                    help="bootstrap fraction of each video-prediction batch, as at the end of "
                         "phase 1b: it keeps single-step (K=1) sampling legal for imagination")
    ap.add_argument("--image_batch_prob", type=float, default=0.15,
                    help="share of video-prediction steps with no context at all")
    ap.add_argument("--loss_norm", type=int, default=1,
                    help="RMS-normalise every loss term (paper Section 3)")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_n", type=int, default=500)
    ap.add_argument("--final_eval_n", type=int, default=1000)
    ap.add_argument("--eval_seed", type=int, default=31337,
                    help="episode seeds for the real-env evaluation: held-out layouts")
    ap.add_argument("--eval_envs", type=int, default=64, help="envs played in parallel")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--amp", type=int, default=1, help="bf16 autocast (no grad scaler needed)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


class Run:
    """Run bookkeeping: TensorBoard, best-checkpoint selection, ``final.json``."""

    def __init__(self, cfg, world, out, writer):
        self.cfg, self.world, self.out, self.writer = cfg, world, out, writer
        self.envs = ([make_env(world.env_spec) for _ in range(cfg.eval_envs)]
                     if world.env_spec else [])
        if not self.envs:
            print("no env_spec in this dataset: real-env evaluation is skipped and no best.pt is "
                  "written (offline metrics only)", flush=True)
        self.best = (-1.0, float("nan"))
        self.last_eval, self.best_eval = {}, {}
        self.t0 = time.time()

    def play(self, policy, n, device):
        """Greedy and sampled real-environment success over ``n`` held-out layouts."""
        kw = dict(tok=self.world.tok, proprio_from_info=self.world.proprio_from_info,
                  seed=self.cfg.eval_seed, device=device)
        g = evaluate(policy, self.envs, n=n, greedy=True, **kw)
        s = evaluate(policy, self.envs, n=n, greedy=False, **kw)
        return {"greedy": g[0], "greedy_opt": g[2], "sampled": s[0], "sampled_opt": s[2],
                "eval_n": n}

    def record(self, step, metrics, save):
        """Log one evaluation, write ``latest.pt``, and keep ``best.pt`` by greedy success."""
        self.last_eval = {"step": step, **metrics}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for k, x in value.items():
                    self.writer.add_scalar(f"reward/{k}_heldout", x, step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(f"eval/{key}", value, step)
        self.writer.flush()
        save(self.out / "checkpoints" / "latest.pt", self.last_eval)
        if self.envs and (metrics["greedy"], -metrics["greedy_opt"]) > (self.best[0], -self.best[1]):
            self.best = (metrics["greedy"], metrics["greedy_opt"])
            self.best_eval = self.last_eval
            save(self.out / "checkpoints" / "best.pt", self.last_eval)

    def finish(self, extra):
        wall = time.time() - self.t0
        json.dump({"steps": self.cfg.steps, "wall_s": round(wall, 1), **extra,
                   "best_greedy_sr": self.best[0], "best_greedy_opt": self.best[1],
                   "best": self.best_eval, "final": self.last_eval},
                  open(self.out / "final.json", "w"), indent=1)
        print(f"HEADS-DONE best greedy SR {self.best[0]:.4f} opt {self.best[1]:.4f} | "
              f"{wall / 60:.1f} min", flush=True)


def train(cfg, world, out, writer, episodes, demos, n_train, dev, rng):
    """Phase 2 as the paper writes it: agent tokens, MTP heads, the transformer finetuned."""
    ds = world.dataset
    tasks = {int(ep.get("meta", {}).get("task", 0)) for ep in episodes}
    n_tasks = max(cfg.n_tasks, max(tasks) + 1)
    agent_cfg = {"d_model": world.model_cfg["d_model"], "n_agent": cfg.n_agent,
                 "n_tasks": n_tasks, "n_actions": ds.action_dim,
                 "mtp": cfg.mtp, "d_hidden": cfg.d_hidden,
                 "reward_bins": cfg.reward_bins, "reward_bin_low": cfg.reward_bin_low,
                 "reward_bin_high": cfg.reward_bin_high, "value_bins": cfg.reward_bins,
                 "value_bin_low": cfg.reward_bin_low, "value_bin_high": cfg.reward_bin_high}
    heads = build_agent_heads(agent_cfg).to(dev)
    T = cfg.clip_T
    print(pretty_agent_summary(heads) + f" | clip {T} slots | tasks {sorted(tasks)}", flush=True)
    print(f"dynamics {param_count(world.dyn) / 1e6:.2f}M finetuned at lr {cfg.dyn_lr:g}",
          flush=True)
    # The property Section 3.3 calls crucial, checked on the model actually about to be trained
    # rather than assumed: perturb the agent tokens and no world output may move. If it ever did,
    # the world model would be predicting the future from what the agent INTENDS, and every number
    # phase 2 and phase 3 produce afterwards would be measuring a different algorithm.
    leak = agent_tokens_are_write_only(world.dyn, device=dev)
    if not leak["ok"]:
        raise RuntimeError(
            f"agent tokens are not write-only: perturbing them moved the flow head by "
            f"{leak['world_delta']:.3e} and the proprio head by {leak['proprio_delta']:.3e} "
            f"(agent output moved {leak['agent_delta']:.3e}). The wm_agent mask is wrong.")
    print(f"agent tokens write-only: world delta {leak['world_delta']:.1e}, "
          f"agent delta {leak['agent_delta']:.3f}", flush=True)

    json.dump({**vars(cfg), "data": world.data_path, "agent": agent_cfg,
               "latent_meta": world.latent_meta, "model": world.model_cfg},
              open(out / "config.json", "w"), indent=1)

    groups = [{"params": list(heads.parameters()), "lr": cfg.lr}]
    if cfg.dyn_lr > 0:
        groups.append({"params": list(world.dyn.parameters()), "lr": cfg.dyn_lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    def cosine(s):
        return 0.5 * (1.0 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps))

    # Cosine decay for both groups; the transformer's rate additionally ramps in over
    # --dyn_warmup steps. At step 1 the heads are random, so the gradient they send into the
    # shared weights is noise with respect to everything the world model knows.
    lambdas = [cosine]
    if cfg.dyn_lr > 0:
        lambdas.append(lambda s: cosine(s) * min(1.0, (s + 1) / max(1, cfg.dyn_warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    # Paper Section 3: "we normalize all loss terms by running estimates of their root-mean-square
    # (RMS)". Order fixes the normaliser slots, so it must stay stable across a resume.
    combiner = LossCombiner({"policy": 1.0, "reward": cfg.reward_weight,
                             "flow": cfg.dyn_weight, "proprio": cfg.dyn_weight * 0.3,
                             "bootstrap": cfg.dyn_weight * 0.5, "value": cfg.value_weight},
                            normalize=bool(cfg.loss_norm), device=dev)

    ctx_signal = min(int(round((1.0 - world.tau_ctx) * world.k_max)), world.k_max)
    emax = _log2_int(world.k_max)
    fwd = dict(tau_ctx=world.tau_ctx, ctx_signal=ctx_signal, emax=emax)
    term_bin = terminal_bins(heads.reward, world.continues_from_reward, dev)
    train_ids = list(range(n_train))
    # Held-out DEMONSTRATIONS: clone accuracy against mixed-quality episodes measures how well the
    # policy imitates noise, which is not the number anyone wants.
    val_ids = [i for i in range(n_train, len(episodes)) if episodes[i]["bc_weight"] > 0] or demos
    policy = AgentPolicy(world.dyn, heads, window=T, tau_ctx=world.tau_ctx, k_max=world.k_max)

    def save(path, extra):
        torch.save({"agent": heads.state_dict(), "agent_cfg": agent_cfg,
                    "dyn": world.dyn.state_dict(), "n_agent": cfg.n_agent,
                    "clip_T": T, "dyn_source": cfg.dyn,
                    "train_cfg": vars(cfg), **extra}, path)

    run = Run(cfg, world, out, writer)
    half = max(1, cfg.batch // 2)
    t_last = time.time()
    for s in range(1, cfg.steps + 1):
        # 50 % relevant (demonstrations), 50 % uniform -- paper Section 4.1.
        rel = sample_clips(episodes, demos, half, T, rng, dev)
        uni = sample_clips(episodes, train_ids, cfg.batch - half, T, rng, dev)
        clip = {k: torch.cat([rel[k], uni[k]], 0) for k in rel}
        demo_mask = torch.cat([torch.ones(half, device=dev),
                               torch.zeros(cfg.batch - half, device=dev)])
        dyn_aux = {}
        with torch.autocast("cuda", torch.bfloat16, enabled=bool(cfg.amp)):
            h = agent_forward(world, heads, clip, **fwd)
            pol, rew = agent_losses(heads, h.float(), clip, mtp=cfg.mtp, demo_mask=demo_mask)
            terms = {"policy": pol, "reward": rew}
            if cfg.value_weight > 0:
                terms["value"] = heads.value.loss(heads.value(h.float()), clip["ret"][..., None],
                                                  clip["valid_v"][..., None])
            if cfg.dyn_weight > 0:
                # The video-prediction loss, unchanged from pretraining and applied to windows
                # drawn from the WHOLE collection: "the dynamics loss is applied only on the
                # uniform sequences to avoid optimistic generations" (Section 4.1).
                # "Reuse the pretraining setting" means the WHOLE objective phase 1b ended with:
                # the context-noise band and scheduled sampling are what make the model survive
                # its own rollouts, and a finetune that drops them trains that robustness away.
                obj = world.objective
                image_batch = rng.random() < cfg.image_batch_prob
                T_dyn = 1 if image_batch else world.window
                batch = sample_windows(episodes[:n_train], cfg.dyn_batch, T_dyn, rng, dev,
                                       skip_terminal=image_batch)
                do_sched = (not image_batch
                            and rng.random() < float(obj.get("sched_sample_prob", 0.0)))
                dyn_terms, dyn_aux = clean_context_loss(
                    world.dyn, batch["z"], align_actions(batch["actions"], T_dyn),
                    p=0 if image_batch else int(rng.integers(1, T_dyn)),
                    k_max=world.k_max, K=world.model_cfg["K"], tau_ctx=world.tau_ctx,
                    ctx_noise_min=obj.get("ctx_noise_min") or None,
                    ctx_noise_max=obj.get("ctx_noise_max") or None, do_sched=do_sched,
                    proprio=batch.get("proprio"), bootstrap_frac=cfg.boot_frac)
                terms.update({k: v for k, v in dyn_terms.items() if v is not None})
            loss, raw = combiner(terms)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for g in opt.param_groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if s % cfg.log_every == 0:
            now = time.time()
            for k, x in raw.items():
                writer.add_scalar(("bc/cross_entropy" if k == "policy" else
                                   "reward/loss" if k == "reward" else
                                   "critic/pretrain_loss" if k == "value" else f"dyn/{k}"), x, s)
            for k, x in dyn_aux.items():
                writer.add_scalar(f"dyn/{k}", x, s)
            writer.add_scalar("train/loss_total", float(loss.detach()), s)
            writer.add_scalar("train/grad_norm", float(gn), s)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], s)
            if cfg.dyn_lr > 0:
                writer.add_scalar("train/dyn_lr", sched.get_last_lr()[1], s)
            writer.add_scalar("train/steps_per_sec_inst", cfg.log_every / max(now - t_last, 1e-9), s)
            t_last = now

        if s % cfg.eval_every == 0 or s == cfg.steps:
            m = paper_metrics(world, heads, episodes, val_ids, T=T, batch=cfg.batch,
                              n_batches=8, rng=rng, device=dev, term_bin=term_bin,
                              term_prob=cfg.term_prob,
                              continues_from_reward=world.continues_from_reward, **fwd)
            n_ev = cfg.final_eval_n if s == cfg.steps else cfg.eval_n
            metrics = {"acc_heldout": m["acc"],
                       "reward_heldout": {k: v for k, v in m.items() if k != "acc"}}
            if run.envs:
                metrics.update(run.play(policy, n_ev, dev))
            run.record(s, metrics, save)
            line = f"[{time.strftime('%H:%M:%S')}] HEADS-EVAL step {s}"
            if run.envs:
                line += (f" greedy SR {metrics['greedy']:.4f} opt {metrics['greedy_opt']:.4f}"
                         f" | sampled SR {metrics['sampled']:.4f} "
                         f"opt {metrics['sampled_opt']:.4f} | n={n_ev}")
            print(f"{line} | acc_heldout {m['acc']:.4f} | reward mae {m['mae']:.4f} "
                  f"term_f1 {m['term_f1']:.4f} | {(time.time() - run.t0) / 60:.1f} min", flush=True)
    run.finish({"demonstrations": len(demos)})


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    dev = torch.device(cfg.device)
    out = Path(cfg.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out / "tb"), purge_step=0)
    world = WorldModel(cfg.dyn, device=cfg.device, n_agent=cfg.n_agent,
                             train=cfg.dyn_lr > 0)
    ds = world.dataset
    if ds.proprio_dim is None:
        raise ValueError("this world model was trained without a proprio stream; the agent reads "
                         "one (open the dataset with a proprio mode and retrain phase 1)")
    if cfg.clip_T <= 0:
        # The window the dynamics model was trained on. The policy's context is also the context
        # imagination dreams with, and the flow head only follows the actions inside the window
        # lengths it has seen: on the gridworld recipe (seq_len 4) a dream with 7 or more past
        # frames puts the player in the wrong cell on >90 % of the steps. A longer policy context
        # therefore needs a phase 1b trained with a longer --data.seq_len, not a larger --clip_T.
        cfg.clip_T = world.window
    if cfg.clip_T > world.window:
        print(f"WARNING: --clip_T {cfg.clip_T} exceeds the {world.window}-frame windows the "
              f"dynamics model was trained on; phase 3 will refuse this checkpoint", flush=True)
    if not 1 <= cfg.clip_T <= world.max_T:
        raise ValueError(f"--clip_T {cfg.clip_T} is outside [1, max_T={world.max_T}] of this "
                         "world model: its positional encoding does not reach that far")

    t0 = time.time()
    check_memory_budget(ds, world.tok, dev)
    episodes = encode_episodes(world.tok, ds, range(len(ds)))
    if "rewards" not in episodes[0] or "terminals" not in episodes[0]:
        raise ValueError("this dataset records no rewards/terminals, so the reward head cannot be "
                         "trained on it")
    n_train = int(round(len(ds) * (1.0 - cfg.val_frac)))
    if cfg.value_weight > 0:
        add_returns(episodes, cfg.gamma)
    demos = demonstration_ids(episodes, n_train, cfg.bc_frac, cfg.bc_seed)
    print(f"data {world.data_path} encoded in {time.time() - t0:.0f}s | batch {cfg.batch}",
          flush=True)

    rng = np.random.default_rng(cfg.seed)
    train(cfg, world, out, writer, episodes, demos, n_train, dev, rng)
    writer.close()


if __name__ == "__main__":
    main()
