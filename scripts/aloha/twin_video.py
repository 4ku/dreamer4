"""The same start, twice: a policy in the REAL environment and in the dream of its world model.

Each episode starts from a real reset frame. The policy (greedy by default) plays it in the environment and, from
that same first frame, inside the world model it was trained in (phase 3's dream loop with its checkpoint's
transformer and reward head). With a greedy policy the two differ only where the world model is wrong. Prints how
often a real failure is dreamed as the box held (what imagination RL can exploit) and saves a video: real episodes
on top, their dreams below, real failures that the dream calls a success first.

    ALOHA_PYTHON=... PYTHONPATH=. python scripts/aloha/twin_video.py --ckpt <phase-3 ckpt> --dyn <world model> \
        --out twin.mp4 [--n 64] [--show 6] [--seed 900000] [--sampled] [--label text]
"""
import argparse
import json

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import dreamer4.train.train_pmpo as tp
from dreamer4.agent_eval import make_eval_envs

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dyn", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--show", type=int, default=6, help="episodes in the video")
ap.add_argument("--seed", type=int, default=900000)
ap.add_argument("--sampled", action="store_true", help="sample actions (default: greedy in both worlds)")
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--lift_r", type=float, default=0.7,
                help="box held at the end: mean reward of the last 10 steps above this (real and dreamed alike)")
ap.add_argument("--label", default="")
ap.add_argument("--json", default="")
cfg = ap.parse_args()

dev = torch.device("cuda")
torch.manual_seed(cfg.seed)
world, agent = tp.load_agent(argparse.Namespace(init_from=cfg.ckpt, dyn=cfg.dyn, device="cuda",
                                                reward_decode="mode", compact=0), dev)
heads, n = agent.heads, cfg.n
greedy = not cfg.sampled
env = make_eval_envs(world.env_spec, n)
tok, hist, T = world.tok, world.tok.history, env.max_steps


def observe(window, infos):
    z = tok.encode_last(np.stack([np.stack(w[-hist:]) for w in window])).float().to(dev)
    pr = torch.as_tensor(np.stack([world.proprio_from_info(i) for i in infos]), dtype=torch.float32, device=dev)
    return z, pr


# 1. the real episodes
policy = agent.eval_policy()
frames, infos = env.reset([cfg.seed + i for i in range(n)])
window = [[f] * hist for f in frames]
policy.reset_rows(None)
z0, p0 = observe(window, infos)
real_frames, real_r, prev = [np.stack(frames)], np.zeros((n, T), np.float32), None
with torch.no_grad():
    for t in range(T):
        z, pr = (z0, p0) if t == 0 else observe(window, infos)
        a = policy.act(z, pr, greedy=greedy, prev_action=prev)
        prev = a
        frames, r, _, infos = env.step(policy.env_action(a))
        real_r[:, t] = r
        for w, f in zip(window, frames):
            w.append(f)
        real_frames.append(np.stack(frames))
env.close()
real_frames = np.stack(real_frames, 1)                                   # (n, T+1, H, W, 3) uint8

# 2. the dreams from the same first frames: phase 3's own rollout, its dreamed latents recorded on the way
C, D = agent.clip_T, world.action_vec_dim
start = {"z": z0[:, None].expand(-1, C, *z0.shape[1:]).contiguous(),
         "proprio": p0[:, None].expand(-1, C, -1).contiguous(),
         "actions": np.zeros((n, C, D + 1), np.float32), "task": np.zeros(n, np.int64)}
start["actions"][..., -1] = 1.0                    # a window padded with the reset frame: start flags only
if greedy:
    sample = heads.policy.sample
    heads.policy.sample = lambda logits, **kw: sample(logits, greedy=True)
dreamed, step = [], tp.dream_step


def recording_step(*args, **kwargs):
    out = step(*args, **kwargs)
    dreamed.append(out[0].detach())
    return out


tp.dream_step = recording_step
with torch.autocast("cuda", torch.bfloat16):
    roll = agent.rollout(start, horizon=T, K=cfg.K, term_prob=0.5)
tp.dream_step = step
dream_r = roll["rewards"].float().cpu().numpy()
zd = torch.stack(dreamed, 1)                                             # (n, T, Nz, Dz): frames 1..T

real_held = real_r[:, -10:].mean(1) > cfg.lift_r
dream_held = dream_r[:, -10:].mean(1) > cfg.lift_r
fooled = ~real_held & dream_held
stats = {"ckpt": cfg.ckpt, "n": n, "greedy": greedy, "real_return": float(real_r.sum(1).mean()),
         "dream_return": float(dream_r.sum(1).mean()), "real_held": float(real_held.mean()),
         "dream_held": float(dream_held.mean()), "real_fail_dreamed_held": int(fooled.sum()),
         "real_fail": int((~real_held).sum()), "real_held_dreamed_fail": int((real_held & ~dream_held).sum()),
         "corr_return": float(np.corrcoef(real_r.sum(1), dream_r.sum(1))[0, 1])}
print(json.dumps(stats), flush=True)
if cfg.json:
    json.dump(stats, open(cfg.json, "w"), indent=1)

# 3. the video: real failures the dream calls a success first, then episodes both call a success
both = np.flatnonzero(real_held & dream_held)
show = list(np.flatnonzero(fooled)[:max(cfg.show - 2, 0)])
show += list(both[:cfg.show - len(show)])


def tile(img, text, color):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 12], fill=(0, 0, 0))
    d.text((3, 0), text, fill=color)
    return np.asarray(im)


decoded = {}
with torch.no_grad():
    for j in show:
        dec = tok.decode_latents(zd[j:j + 1].float()).permute(0, 1, 3, 4, 2)[0].float().cpu().numpy()
        dec = (np.clip(dec, 0, 1) * 255).astype(np.uint8)
        H, W = real_frames.shape[2:4]
        if dec.shape[1:3] != (H, W):
            dec = np.stack([np.asarray(Image.fromarray(f).resize((W, H), Image.NEAREST)) for f in dec])
        decoded[j] = dec
video = []
for t in range(1, T + 1):
    cols = []
    for j in show:
        ok_r, ok_d = real_held[j], dream_held[j]
        top = tile(real_frames[j, t], f"REAL  t={t:3d}  R={real_r[j, :t].sum():5.1f} {'held' if ok_r else 'FAIL'}",
                   (160, 200, 255) if ok_r else (255, 110, 110))
        bot = tile(decoded[j][t - 1], f"DREAM t={t:3d}  R={dream_r[j, :t].sum():5.1f} {'held' if ok_d else 'FAIL'}",
                   (255, 200, 60))
        cols.append(np.concatenate([top, bot], 0))
    grid = np.concatenate(cols, 1)
    bar = Image.new("RGB", (grid.shape[1], 16), (25, 25, 25))
    ImageDraw.Draw(bar).text((4, 2), f"{cfg.label}  top: real env  -  bottom: its world model's dream from the SAME "
                                     f"first frame ({'greedy' if greedy else 'sampled'} policy in both)",
                             fill=(255, 255, 255))
    video.append(np.concatenate([np.asarray(bar), grid], 0))
imageio.mimsave(cfg.out, video, fps=10, macro_block_size=1)
print("saved", cfg.out, "episodes", [int(j) for j in show])
