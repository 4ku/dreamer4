"""Collect AlohaPick episodes with a trained Dreamer4 policy, in the aloha-npy layout of a base dataset.

    ALOHA_PYTHON=... python collect_policy.py --ckpt <phase-3 ckpt> --dyn <world model> --base <base dataset dir> \
        --out <new dir> --n 1000 [--envs 64] [--seed 500000] [--greedy | --temperature T]

The manifest copies the base dataset's cameras, env spec and proprio standardisation (so a merged dataset reads
one convention); each episode records source = "policy", the checkpoint, success and return. "noise" is set to
1.0 so the adapter does NOT clone these episodes unless asked (they teach the world model where the policy goes).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dreamer4.agent_eval import make_eval_envs
from dreamer4.models.agent_heads import build_agent_heads
from dreamer4.models.world_model import AgentPolicy, WorldModel

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dyn", required=True)
ap.add_argument("--base", required=True, help="dataset whose manifest conventions are copied")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--envs", type=int, default=64)
ap.add_argument("--seed", type=int, default=500000)
ap.add_argument("--greedy", action="store_true", help="argmax actions (default: sample the policy)")
ap.add_argument("--temperature", type=float, default=1.0,
                help="sampling temperature (> 1: more varied episodes for the world model and the self-imitation)")
cfg = ap.parse_args()

dev = torch.device("cuda")
base = json.loads((Path(cfg.base) / "manifest.json").read_text())
ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
world = WorldModel(cfg.dyn, device=dev, n_agent=int(ckpt["n_agent"]))
world.load_dynamics_weights(ckpt["dyn"])
heads = build_agent_heads(ckpt["agent_cfg"]).to(dev)
heads.load_state_dict(ckpt["agent"])
policy = AgentPolicy(world.dyn.eval(), heads.eval(), window=int(ckpt["clip_T"]),
                     tau_ctx=world.tau_ctx, k_max=world.k_max).eval()
env = make_eval_envs(world.env_spec, cfg.envs)
tok, hist, T = world.tok, world.tok.history, env.max_steps
cams = base["cameras"]
W = base["env"]["cam_res"][1]

out = Path(cfg.out)
out.mkdir(parents=True, exist_ok=True)
N = cfg.n
mm = np.lib.format.open_memmap
state = mm(out / "state.npy", "w+", np.float32, (N, T + 1, 14))
action = mm(out / "action.npy", "w+", np.float32, (N, T, 14))
reward = mm(out / "reward.npy", "w+", np.float32, (N, T))
pix = {c: mm(out / f"pixels_{c}.npy", "w+", np.uint8, (N, T + 1, 128, W, 3)) for c in cams}
episodes = []
t0 = time.time()
done = 0
torch.manual_seed(cfg.seed)
with torch.no_grad():
    while done < N:
        B = env.n
        m = min(B, N - done)
        frames, infos = env.reset([cfg.seed + done + i for i in range(B)])
        window = [[f] * hist for f in frames]
        policy.reset_rows(None)
        prev = None
        ret = np.zeros(B)

        def record(t, frames, infos):
            for k, c in enumerate(cams):
                pix[c][done:done + m, t] = np.asarray(frames)[:m, :, k * W:(k + 1) * W]
            state[done:done + m, t] = np.stack([i["state"] for i in infos])[:m]

        record(0, frames, infos)
        for t in range(T):
            vid = np.stack([np.stack(w[-hist:]) for w in window])
            z = tok.encode_last(vid).float().to(dev)
            pr = torch.as_tensor(np.stack([world.proprio_from_info(i) for i in infos]), dtype=torch.float32,
                                 device=dev)
            a = policy.act(z, pr, greedy=cfg.greedy, temperature=cfg.temperature, prev_action=prev)
            prev = a
            env_a = np.asarray(policy.env_action(a), np.float32)
            frames, r, _, infos = env.step(env_a)
            action[done:done + m, t] = env_a[:m]
            reward[done:done + m, t] = np.asarray(r)[:m]
            ret += np.asarray(r)
            for w, f in zip(window, frames):
                w.append(f)
            record(t + 1, frames, infos)
        ok = np.array([i["is_success"] for i in infos])
        for i in range(m):
            episodes.append({"source": "policy", "ckpt": cfg.ckpt, "greedy": bool(cfg.greedy),
                             "temperature": cfg.temperature, "seed": cfg.seed + done + i, "speed": 1.0, "noise": 1.0,
                             "success": bool(ok[i]), "return": round(float(ret[i]), 3)})
        done += m
        print(f"{done}/{N}: round success {ok[:m].mean():.2f} return {ret[:m].mean():.1f} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
env.close()
for a in (state, action, reward, *pix.values()):
    a.flush()
manifest = {k: base[k] for k in ("format", "cameras", "downscale", "env", "state_mean", "state_std")}
manifest.update({"n_episodes": N, "episodes": episodes,
                 "collector": {"policy": cfg.ckpt, "greedy": bool(cfg.greedy), "temperature": cfg.temperature,
                               "seed": cfg.seed, "proprio_stats_from": str(cfg.base)}})
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
succ = np.mean([e["success"] for e in episodes])
rets = np.array([e["return"] for e in episodes])
print(json.dumps({"out": str(out), "n": N, "success": round(float(succ), 3), "return_mean": round(float(rets.mean()), 2),
                  "return_max": round(float(rets.max()), 2), "minutes": round((time.time() - t0) / 60, 1)}))
