"""Video: an AlohaHandOver policy in the real env, with what the success test sees. Plays N episodes greedy, keeps
successes and failures, and renders each with its three cameras (overview, left wrist, right wrist) and a status
strip: has the left hand lifted the box, the box's distance to the green target, the right hand's hold count (5 in a
row = success: +1 and the episode ends), the return so far.

    ALOHA_PYTHON=... PYTHONPATH=. python scripts/aloha/handover_video.py --ckpt <phase-2/3 ckpt> --dyn <world model> \
        --data <base dataset> --out video.mp4 --png last.png [--n 24] [--show 6] [--seed 900000] [--label text]
"""
import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from dreamer4.agent_eval import make_eval_envs
from dreamer4.train.train_pmpo import load_agent

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dyn", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--png", required=True)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--show", type=int, default=6)
ap.add_argument("--seed", type=int, default=900000)
ap.add_argument("--label", default="")
ap.add_argument("--first", action="store_true",
                help="show the first --show episodes as they come (the same seeds for every policy: a fair before/after)")
ap.add_argument("--upscale", type=int, default=2, help="pixel upscaling of the video (1 keeps the file small)")
ap.add_argument("--lifted_fails", type=int, default=0,
                help="show this many failures that HAD lifted the box (the hand-over failed), the rest successes")
cfg = ap.parse_args()
dev = torch.device("cuda")
world, agent = load_agent(argparse.Namespace(init_from=cfg.ckpt, dyn=cfg.dyn, device="cuda", reward_decode="mode",
                                             compact=0, data=cfg.data), dev)
env = make_eval_envs(world.env_spec, cfg.n)
policy = agent.eval_policy()
tok, hist, T = world.tok, world.tok.history, env.max_steps
frames, infos = env.reset([cfg.seed + i for i in range(cfg.n)])
window = [[f] * hist for f in frames]
policy.reset_rows(None)
video, viz, rew, prev = [np.stack(frames)], [], np.zeros((cfg.n, T)), None
ended = np.full(cfg.n, T)        # the step an episode ended at (success / out of bounds); T = the time limit
ok = np.zeros(cfg.n, bool)
lifted = np.zeros(cfg.n, bool)
with torch.no_grad():
    for t in range(T):
        z = tok.encode_last(np.stack([np.stack(w[-hist:]) for w in window])).float().to(dev)
        pr = torch.as_tensor(np.stack([world.proprio_from_info(i) for i in infos]), dtype=torch.float32, device=dev)
        a = policy.act(z, pr, greedy=True, prev_action=prev)
        prev = a
        frames, r, done, infos = env.step(policy.env_action(a))
        rew[:, t] = r
        for w, f in zip(window, frames):
            w.append(f)
        video.append(np.stack(frames))
        viz.append([i["viz"] for i in infos])
        newly = np.asarray(done, bool) & (ended == T)
        ended[newly] = t + 1
        ok |= np.array([i["is_success"] for i in infos])
        lifted |= (ended >= t + 1) & np.array([bool(i["viz"]["left_held"]) for i in infos])
        if (ended < T).all():
            break
env.close()
print(f"{cfg.n} episodes: success {ok.mean():.3f}, success at step {ended[ok].mean() if ok.any() else float('nan'):.0f}")

succ, fail = np.flatnonzero(ok), np.flatnonzero(~ok)
show = list(succ[:cfg.show // 2]) + list(fail[:cfg.show - min(len(succ), cfg.show // 2)])
show = show[:cfg.show]
if cfg.first:
    show = list(range(min(cfg.show, cfg.n)))
if cfg.lifted_fails:
    lf = np.flatnonzero(lifted & ~ok)
    show = list(lf[:cfg.lifted_fails]) + list(succ[:cfg.show - min(len(lf), cfg.lifted_fails)])
    print(f"lifted, then failed: {len(lf)} of {cfg.n}")
font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
font_b = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 14)
font_t = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 17)
H, W = video[0].shape[1:3]
STRIP = 50


def tile(i, t):
    tt = min(t, ended[i])
    img = Image.new("RGB", (W, H + STRIP), (20, 20, 20))
    img.paste(Image.fromarray(video[tt][i]), (0, 0))
    d = ImageDraw.Draw(img)
    ret = rew[i, :tt].sum()
    d.text((4, H + 2), f"step {tt:3d}   return {ret:+.2f}", fill=(220, 220, 220), font=font)
    if tt >= 1:
        v = viz[tt - 1][i]
        x = 4
        lifted = bool(v["left_held"])
        d.text((x, H + 18), "left lift ✓" if lifted else "left lift …", font=font,
               fill=(120, 230, 120) if lifted else (180, 180, 180))
        x += 92
        dist = float(v["d_target"]) * 100
        d.text((x, H + 18), f"box→target {dist:4.1f} cm", font=font,
               fill=(120, 230, 120) if dist < 5 else (230, 200, 120))
        x += 150
        held = int(v["held"])
        d.text((x, H + 18), f"right hold {min(held, 5)}/5", font=font,
               fill=(120, 230, 120) if held > 0 else (180, 180, 180))
    if ok[i] and t >= ended[i]:
        d.text((4, H + 33), f"SUCCESS at step {ended[i]}: +1, the episode ends", fill=(80, 255, 80), font=font_b)
    elif not ok[i] and t >= T - 1:
        d.text((4, H + 33), "FAIL: lifted, but no hand-over by step 250" if lifted[i] else
               "FAIL: never lifted the box (grasp missed)", fill=(255, 90, 90), font=font_b)
    return np.asarray(img)


def frame(t):
    tiles = [tile(i, t) for i in show]
    cols = 2
    rows = [np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range((len(tiles) + cols - 1) // cols)]
    rows = [np.pad(r, ((0, 0), (0, cols * W - r.shape[1]), (0, 0))) for r in rows]
    grid = np.concatenate(rows, 0)
    head = Image.new("RGB", (grid.shape[1], 46), (255, 255, 255))
    dh = ImageDraw.Draw(head)
    dh.text((6, 2), f"AlohaHandOver · {cfg.label} · real env, greedy", fill=(0, 0, 0), font=font_t)
    dh.text((6, 24), "success = left hand lifted the box, then box < 5 cm from the GREEN target in the closed right "
                     "hand, left hand away — 5 steps in a row", fill=(60, 60, 60), font=font)
    out = np.concatenate([np.asarray(head), grid], 0)
    s = max(1, cfg.upscale)
    return np.asarray(Image.fromarray(out).resize((out.shape[1] * s, out.shape[0] * s), Image.NEAREST))


last = int(min(T, ended[show].max() + 1))
frames_out = [frame(t) for t in range(0, last + 1)]
frames_out += [frames_out[-1]] * 30
imageio.mimsave(cfg.out, frames_out, fps=15, macro_block_size=1)
imageio.imwrite(cfg.png, frames_out[-1])
print("saved", cfg.out, cfg.png, "shown", [(int(i), bool(ok[i]), int(ended[i])) for i in show])
