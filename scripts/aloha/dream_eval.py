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
ap.add_argument("policies", nargs="+", help="name=checkpoint")
cfg = ap.parse_args()

dev = torch.device("cuda")
results = {}
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
    rets, held = [], []
    for i in range(0, len(pick), cfg.batch):
        rows = pick[i:i + cfg.batch]
        with torch.autocast("cuda", torch.bfloat16):
            roll = agent.rollout(starts.sample(len(rows), pick=rows), horizon=cfg.horizon, K=cfg.K, term_prob=0.5)
        r = roll["rewards"].float()
        rets.append(r.sum(1).cpu().numpy())
        held.append((r[:, -10:].mean(1) > cfg.lift_r).cpu().numpy())
    rets, held = np.concatenate(rets), np.concatenate(held)
    res = {"ckpt": ck, "dream_return": float(rets.mean()), "dream_held": float(held.mean()), "n": cfg.n}
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
    line = f"{name:28s} dream return {res['dream_return']:6.1f}  held {res['dream_held']:.2f}"
    if "real_return" in res:
        line += f" | real (sampled) return {res['real_return']:6.1f}  success {res['real_success']:.2f}"
    if cfg.repeat > 1:
        line += (f" | per start: always held {res['starts_always_held']:.2f}, never {res['starts_never_held']:.2f}, "
                 f"mixed {res['starts_mixed']:.2f}, return sd {res['return_sd_within_start']:.1f}")
    print(line, flush=True)
    del world, agent, starts
    torch.cuda.empty_cache()
json.dump({"data": cfg.data, "horizon": cfg.horizon, "K": cfg.K, "repeat": cfg.repeat, "greedy": cfg.greedy,
           "results": results}, open(cfg.out, "w"), indent=1)
