"""How close are the 'failures' to the success line? Held-out play of AlohaPick policies with the reward of the LAST
step recorded per episode.

Success needs the box above 10 cm and grasped at the last step, and the lift reward stops growing at exactly 10 cm,
so a return-maximising policy may end its episodes holding the box just below the line. With the box grasped, the
last-step reward is ~0.1 approach + 0.2 grasp + 0.7 lift_frac + 0.1 no-collision: about 1.09 at >= 10 cm, about 0.99
at 9 cm. Reports the official success next to "last-step reward >= 0.99" (grasped at roughly 9 cm or more).

    ALOHA_PYTHON=... PYTHONPATH=. python scripts/aloha/eval_margin.py --dyn <world model> --out out.json \
        [--n 512] [--seed 900000] name=<checkpoint> ...
"""
import argparse
import json

import numpy as np
import torch

from dreamer4.agent_eval import make_eval_envs
from dreamer4.train.train_pmpo import load_agent

ap = argparse.ArgumentParser()
ap.add_argument("--dyn", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--envs", type=int, default=64)
ap.add_argument("--seed", type=int, default=900000)
ap.add_argument("--sampled", action="store_true")
ap.add_argument("policies", nargs="+", help="name=checkpoint")
cfg = ap.parse_args()

dev = torch.device("cuda")
results, env = {}, None
for spec in cfg.policies:
    name, ck = spec.split("=", 1)
    world, agent = load_agent(argparse.Namespace(init_from=ck, dyn=cfg.dyn, device="cuda", reward_decode="mode",
                                                 compact=0), dev)
    if env is None:
        env = make_eval_envs(world.env_spec, cfg.envs)
    policy = agent.eval_policy()
    tok, hist, T, B = world.tok, world.tok.history, env.max_steps, env.n
    succ, rets, last = [], [], []
    torch.manual_seed(cfg.seed)
    with torch.no_grad():
        for start in range(0, cfg.n, B):
            m = min(B, cfg.n - start)
            frames, infos = env.reset([cfg.seed + start + i for i in range(B)])
            window = [[f] * hist for f in frames]
            policy.reset_rows(None)
            prev, ret = None, np.zeros(B)
            alive, ok, r_last = np.ones(B, bool), np.zeros(B, bool), np.zeros(B)
            for t in range(T):
                z = tok.encode_last(np.stack([np.stack(w[-hist:]) for w in window])).float().to(dev)
                pr = torch.as_tensor(np.stack([world.proprio_from_info(i) for i in infos]), dtype=torch.float32,
                                     device=dev)
                a = policy.act(z, pr, greedy=not cfg.sampled, prev_action=prev)
                prev = a
                frames, r, term, infos = env.step(policy.env_action(a))
                r = np.asarray(r, np.float64)
                ret += np.where(alive, r, 0.0)
                ended = alive & (np.asarray(term, bool) | (t == T - 1))
                ok |= ended & np.array([i["is_success"] for i in infos])
                r_last = np.where(ended, r, r_last)
                alive &= ~ended
                for w, f in zip(window, frames):
                    w.append(f)
                if not alive.any():
                    break
            succ += ok[:m].tolist()
            rets += ret[:m].tolist()
            last += r_last[:m].tolist()
    succ, rets, last = np.array(succ), np.array(rets), np.array(last)
    res = {"ckpt": ck, "n": int(len(succ)), "success": float(succ.mean()), "return": float(rets.mean()),
           "held_9cm": float((last >= 0.99).mean()), "held_9_5cm": float((last >= 1.04).mean()),
           "fail_but_held_9cm": float(((~succ) & (last >= 0.99)).mean()),
           "fail_return": float(rets[~succ].mean()) if (~succ).any() else float("nan")}
    results[name] = res
    print(f"{name:24s} success {res['success']:.3f} | grasped >= ~9 cm at the end {res['held_9cm']:.3f} "
          f"(>= ~9.5 cm {res['held_9_5cm']:.3f}) | return {res['return']:.1f} | failures that still hold >= 9 cm "
          f"{res['fail_but_held_9cm']:.3f}", flush=True)
    del world, agent, policy
    torch.cuda.empty_cache()
env.close()
json.dump({"n": cfg.n, "seed": cfg.seed, "greedy": not cfg.sampled, "results": results}, open(cfg.out, "w"), indent=1)
