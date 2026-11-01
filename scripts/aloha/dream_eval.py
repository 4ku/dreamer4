"""Whole episodes of a policy inside its own world model, next to its real ones.

Imagination RL can only climb what the dreams reward. Each policy plays full episodes in the world model it was
trained in (its checkpoint's transformer and reward head, closed loop, sampled actions as in phase 3), starting from
the FIRST frames of recorded episodes, and the dreamed return and box-held rate are printed next to the policy's real
held-out numbers. Policies that share a transformer (a phase-2 clone and the phase-3 runs started from it, in
imagination or in the real env) are scored in the same world: a world model that ranks them the way the environment
does can be climbed by imagination RL, one that dreams them alike (or prefers the worse) cannot.

    PYTHONPATH=. python scripts/aloha/dream_eval.py --dyn <phase-1 world model> --data <aloha-npy dir> \
        --out out.json [--heldout <dir of held-out JSONs>] name=<checkpoint> ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dreamer4.train.train_dynamics import encode_episodes
from dreamer4.train.train_pmpo import DemoStarts, load_agent

ap = argparse.ArgumentParser()
ap.add_argument("--dyn", required=True)
ap.add_argument("--data", required=True, help="recorded episodes: their first frames start the dreams")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--horizon", type=int, default=120, help="dreamed steps (AlohaPick episodes are 120)")
ap.add_argument("--lift_r", type=float, default=0.7,
                help="box held at the end: the mean dreamed reward of the last 10 steps above this")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--K", type=int, default=1, help="denoising steps per dreamed frame (phase 3 dreams with 1)")
ap.add_argument("--repeat", type=int, default=1,
                help="dream every start this many times: with --greedy the only randomness left is the world "
                     "model's own, and the spread of the outcomes per start shows how stochastic its dreams are")
ap.add_argument("--greedy", action="store_true", help="greedy actions instead of sampled ones")
ap.add_argument("--reward_decode", choices=("mode", "mean"), default="mode",
                help="the dreamed reward: phase 3's default 'mode', or the reward head's expectation")
ap.add_argument("--heldout", default="", help="dir of <name>.json held-out results printed alongside")
ap.add_argument("--grasp_split", action="store_true",
                help="HandOver: sort the dreamed failures by the dreamed gripper joints as plot_ho_failures.py "
                     "sorts real ones -- missed first grasp (left), missed hand-over grasp (right), both held")
ap.add_argument("policies", nargs="+", help="name=checkpoint")
cfg = ap.parse_args()

dev = torch.device("cuda")
results = {}


def holds(g, lo=0.040, hi=0.052, n=8):
    """Rows (episodes) whose gripper stopped closing at the box's width for n steps: it held the box."""
    ok = (g > lo) & (g < hi)
    run = np.zeros(len(g), int)
    best = np.zeros(len(g), int)
    for t in range(g.shape[1]):
        run = np.where(ok[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best >= n
for spec in cfg.policies:
    name, ck = spec.split("=", 1)
    torch.manual_seed(cfg.seed)
    world, agent = load_agent(argparse.Namespace(init_from=ck, dyn=cfg.dyn, device="cuda", data=cfg.data,
                                                 reward_decode=cfg.reward_decode, compact=0), dev)
    starts = DemoStarts(encode_episodes(world.tok, world.dataset, range(len(world.dataset))),
                        window=agent.clip_T)
    if cfg.greedy:
        sample = agent.heads.policy.sample
        agent.heads.policy.sample = lambda logits, **kw: sample(logits, greedy=True)
    firsts = np.flatnonzero(starts.index == starts.base)          # each episode's frame 0
    pick = np.random.default_rng(cfg.seed).choice(firsts, cfg.n, replace=len(firsts) < cfg.n)
    pick = np.repeat(pick, cfg.repeat)
    rets, held, ends, lens, grips = [], [], [], [], []
    if cfg.grasp_split:
        # the dreamed proprio is standardised: undo the (affine, per-joint) standardisation of the data
        leaf = (getattr(world.dataset, "parts", None) or [world.dataset])[0]
        dim = leaf.proprio_dim
        z0, z1 = leaf.standardise(np.zeros(dim)), leaf.standardise(np.ones(dim))
        p_std = 1.0 / (z1 - z0)
        p_mean = -z0 * p_std
    for i in range(0, len(pick), cfg.batch):
        rows = pick[i:i + cfg.batch]
        with torch.autocast("cuda", torch.bfloat16):
            roll = agent.rollout(starts.sample(len(rows), pick=rows), horizon=cfg.horizon, K=cfg.K, term_prob=0.5,
                                 keep_proprio=cfg.grasp_split)
        r = roll["rewards"].float()
        rets.append(r.sum(1).cpu().numpy())
        held.append((r[:, -10:].mean(1) > cfg.lift_r).cpu().numpy())
        ends.append(roll["terminated"].cpu().numpy())          # reward v2: the dream ended on the success bonus
        lens.append((roll["alive"].sum(1) - 1).cpu().numpy())
        if cfg.grasp_split:
            st = roll["proprio"].cpu().numpy() * p_std + p_mean
            alive = roll["alive"].cpu().numpy() > 0
            # frames after a dream's end repeat nothing useful: park them outside the hold band
            left, right = np.where(alive, st[..., 6], 0.0), np.where(alive, st[..., 13], 0.0)
            grips.append(np.stack([holds(left), holds(right)], 1))
    rets, held, ends, lens = map(np.concatenate, (rets, held, ends, lens))
    res = {"ckpt": ck, "dream_return": float(rets.mean()), "dream_held": float(held.mean()), "n": cfg.n,
           "dream_success": float(ends.mean()),
           "dream_steps_success": float(lens[ends].mean()) if ends.any() else float("nan")}
    if cfg.grasp_split:
        g = np.concatenate(grips)
        fail = ~ends
        res.update(dream_missed_grasp=float((fail & ~g[:, 0]).mean()),
                   dream_missed_handover=float((fail & g[:, 0] & ~g[:, 1]).mean()),
                   dream_both_held_failed=float((fail & g[:, 0] & g[:, 1]).mean()),
                   dream_left_held=float(g[:, 0].mean()))
    if cfg.repeat > 1:
        per_start = held.reshape(cfg.n, cfg.repeat).mean(1)
        res.update(starts_always_held=float((per_start == 1).mean()), starts_never_held=float((per_start == 0).mean()),
                   starts_mixed=float(((per_start > 0) & (per_start < 1)).mean()),
                   return_sd_within_start=float(rets.reshape(cfg.n, cfg.repeat).std(1).mean()))
    hf = Path(cfg.heldout) / f"{name}.json" if cfg.heldout else None
    if hf is not None and hf.exists():
        h = json.load(open(hf))
        res.update(real_return=h["sampled_return"], real_success=h["sampled_sr"])
    results[name] = res
    line = (f"{name:28s} dream return {res['dream_return']:7.2f}  held {res['dream_held']:.2f}  "
            f"success {res['dream_success']:.3f} (steps {res['dream_steps_success']:.1f})")
    if "real_return" in res:
        line += f" | real (sampled) return {res['real_return']:6.1f}  success {res['real_success']:.2f}"
    if cfg.repeat > 1:
        line += (f" | per start: always held {res['starts_always_held']:.2f}, never {res['starts_never_held']:.2f}, "
                 f"mixed {res['starts_mixed']:.2f}, return sd {res['return_sd_within_start']:.1f}")
    if cfg.grasp_split:
        line += (f" | failures: missed grasp {res['dream_missed_grasp']:.3f}, missed hand-over "
                 f"{res['dream_missed_handover']:.3f}, both held {res['dream_both_held_failed']:.3f}")
    print(line, flush=True)
    del world, agent, starts
    torch.cuda.empty_cache()
json.dump({"data": cfg.data, "horizon": cfg.horizon, "K": cfg.K, "repeat": cfg.repeat, "greedy": cfg.greedy,
           "results": results}, open(cfg.out, "w"), indent=1)
