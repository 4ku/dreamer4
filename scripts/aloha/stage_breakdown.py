"""Where does an AlohaHandOver policy fail? Plays greedy episodes in the real env and sorts every episode by the furthest
stage it reached, from the env worker's per-step success-test signals (``viz``):

    no lift            the left hand never held the box off the table
    lifted only        it held it, but the box never came within 10 cm of the green target
    near the target    the box came within 10 cm, but the success test (box < 5 cm from the target in the closed
                       right hand, left hand away, 5 steps in a row) never passed
    success

    ALOHA_PYTHON=... PYTHONPATH=. python scripts/aloha/stage_breakdown.py --dyn <world model> --data <base dataset> \
        --out stages.json [--n 128] [--envs 32] [--seed 900000] name=<checkpoint> ...
"""
import argparse
import json

import numpy as np
import torch

from dreamer4.agent_eval import make_eval_envs
from dreamer4.train.train_pmpo import load_agent

ap = argparse.ArgumentParser()
ap.add_argument("--dyn", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--envs", type=int, default=32)
ap.add_argument("--seed", type=int, default=900000)
ap.add_argument("--near", type=float, default=0.10, help="'near the target' distance (m)")
ap.add_argument("policies", nargs="+", help="name=checkpoint")
cfg = ap.parse_args()
dev = torch.device("cuda")
results = {}
for spec in cfg.policies:
    name, ck = spec.split("=", 1)
    world, agent = load_agent(argparse.Namespace(init_from=ck, dyn=cfg.dyn, device="cuda", reward_decode="mode",
                                                 compact=0, data=cfg.data), dev)
    env = make_eval_envs(world.env_spec, cfg.envs)
    policy = agent.eval_policy()
    tok, hist, T = world.tok, world.tok.history, env.max_steps
    recs = []
    for b in range(0, cfg.n, cfg.envs):
        m = min(cfg.envs, cfg.n - b)
        frames, infos = env.reset([cfg.seed + b + i for i in range(cfg.envs)])
        window = [[f] * hist for f in frames]
        policy.reset_rows(None)
        prev = None
        lift_t = np.full(cfg.envs, -1)
        near_t = np.full(cfg.envs, -1)
        min_d = np.full(cfg.envs, np.inf)
        ok = np.zeros(cfg.envs, bool)
        ret = np.zeros(cfg.envs)
        alive = np.ones(cfg.envs, bool)
        with torch.no_grad():
            for t in range(T):
                z = tok.encode_last(np.stack([np.stack(w[-hist:]) for w in window])).float().to(dev)
                pr = torch.as_tensor(np.stack([world.proprio_from_info(i) for i in infos]), dtype=torch.float32,
                                     device=dev)
                a = policy.act(z, pr, greedy=True, prev_action=prev)
                prev = a
                frames, r, done, infos = env.step(policy.env_action(a))
                ret += np.asarray(r) * alive
                for w, f in zip(window, frames):
                    w.append(f)
                for i, inf in enumerate(infos):
                    if not alive[i]:
                        continue
                    v = inf["viz"]
                    if lift_t[i] < 0 and bool(v["left_held"]):
                        lift_t[i] = t + 1
                    d = float(v["d_target"])
                    min_d[i] = min(min_d[i], d)
                    if near_t[i] < 0 and d < cfg.near:
                        near_t[i] = t + 1
                    ok[i] |= bool(inf["is_success"])
                alive &= ~np.asarray(done, bool)
                if not alive.any():
                    break
        for i in range(m):
            stage = ("success" if ok[i] else "near the target" if near_t[i] >= 0 else
                     "lifted only" if lift_t[i] >= 0 else "no lift")
            recs.append({"stage": stage, "lift_t": int(lift_t[i]), "near_t": int(near_t[i]),
                         "min_d_target": round(float(min_d[i]), 4), "return": round(float(ret[i]), 3)})
    env.close()
    stages = ["no lift", "lifted only", "near the target", "success"]
    frac = {s: float(np.mean([r["stage"] == s for r in recs])) for s in stages}
    lifts = [r["lift_t"] for r in recs if r["lift_t"] >= 0]
    results[name] = {"ckpt": ck, "n": len(recs), "stages": frac, "return": float(np.mean([r["return"] for r in recs])),
                     "lift_step_median": float(np.median(lifts)) if lifts else None, "episodes": recs}
    print(f"{name:24s} " + "  ".join(f"{s} {frac[s]:.3f}" for s in stages)
          + f"  | return {results[name]['return']:+.2f}  lift step {results[name]['lift_step_median']}", flush=True)
    del world, agent, policy
    torch.cuda.empty_cache()
json.dump({"seed": cfg.seed, "near": cfg.near, "results": results}, open(cfg.out, "w"), indent=1)
