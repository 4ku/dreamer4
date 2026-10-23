"""Does the world model predict FAILURE? Recorded AlohaPick episodes -- successes AND failures -- replayed
through phase 3's dream loop with their RECORDED actions (open loop after a real context), and the reward
head's per-step reward in the dream compared with the real one. The same head read on the REAL frames
(teacher forcing) separates a reward-head error from a dynamics error.

    python wm_replay.py --dyn <world model> --heads <phase-2/3 ckpt> --data <aloha-npy dir> --out out.json \
        [--n 48] [--start S --horizon H] [--png out.png] [--video out.mp4]

The context is the clip_T real frames ending at frame S (default: the episode's first frames); the dream
runs H frames (default: to the episode end). Use episodes the world model never trained on (a later
round's collection). An episode counts as "box held" when its reward over the window's last 10 steps
averages above --lift_r (the env pays ~1.09 per step while the box is held), real and dreamed alike.
"""
import argparse
import json

import numpy as np
import torch

from dreamer4.data import open_video_dataset
from dreamer4.dynamics_eval import windowed_rollout
from dreamer4.models.world_model import WorldModel, dream_context, dream_step
from dreamer4.train.train_dynamics import encode_episodes
from dreamer4.train.train_pmpo import load_agent

ap = argparse.ArgumentParser()
ap.add_argument("--dyn", required=True)
ap.add_argument("--heads", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--png", default="")
ap.add_argument("--video", default="")
ap.add_argument("--video_n", type=int, default=4, help="failures in the video (plus 2 successes)")
ap.add_argument("--n", type=int, default=48, help="episodes per outcome")
ap.add_argument("--start", type=int, default=-1, help="last real context frame (default: clip_T - 1)")
ap.add_argument("--horizon", type=int, default=0, help="dreamed frames (default: to the episode end)")
ap.add_argument("--gen_dyn", default="",
                help="also dream with THIS phase-1 world model and read the reward with --heads on its "
                     "frames: separates what phase 1 generates from what the phase-2 finetune generates")
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lift_r", type=float, default=0.7)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--label", default="")
cfg = ap.parse_args()

dev = torch.device("cuda")
torch.manual_seed(cfg.seed)
world, agent = load_agent(argparse.Namespace(init_from=cfg.heads, dyn=cfg.dyn, device="cuda",
                                             reward_decode="mode", compact=0), dev)
heads, W, D = agent.heads, agent.clip_T, world.action_vec_dim
gen = WorldModel(cfg.gen_dyn, device=dev) if cfg.gen_dyn else None

ds = open_video_dataset(cfg.data, proprio="state", actions=True)
outcome = np.array([bool(ds.episode_meta(i).get("success", False)) for i in range(len(ds))])
rng = np.random.default_rng(cfg.seed)
fail = sorted(rng.choice(np.flatnonzero(~outcome), min(cfg.n, int((~outcome).sum())), replace=False).tolist())
succ = sorted(rng.choice(np.flatnonzero(outcome), min(cfg.n, int(outcome.sum())), replace=False).tolist())
picks = fail + succ
T_ep = min(ds.episode_frames(i) for i in picks)
S = cfg.start if cfg.start >= 0 else W - 1             # last context frame
C0 = S - W + 1                                         # first context frame
END = T_ep if cfg.horizon <= 0 else min(T_ep, S + 1 + cfg.horizon)
assert C0 >= 0, f"--start must be >= {W - 1}"
print(f"{len(fail)} failures + {len(succ)} successes from {cfg.data}; context frames {C0}-{S}, "
      f"dream {S + 1}-{END - 1}", flush=True)


@torch.no_grad()
def replay(eps, mode):
    """Reward head along each episode for frames S+1 .. END-1, after the real context: "dream" -- the
    --heads transformer dreams them from the recorded actions, as in phase 3; "gen" -- --gen_dyn dreams
    them and the --heads transformer reads them; "real" -- read on the real frames. Returns the rewards
    (B, END-S-1) and the latents."""
    B = len(eps)
    z_all = torch.stack([e["z"][:END] for e in eps]).float().to(dev)
    p_all = torch.as_tensor(np.stack([e["proprio"][:END] for e in eps]), dtype=torch.float32, device=dev)
    into = torch.zeros(B, END, D + 1, device=dev)         # the action INTO each frame
    into[:, 1:, :D] = torch.as_tensor(np.stack([e["actions"][:END - 1] for e in eps]), dtype=torch.float32,
                                      device=dev)
    if mode == "gen":
        zg, pg, _ = windowed_rollout(gen.dyn, z_full=z_all[:, C0:END], actions=into[:, C0 + 1:END, :D],
                                     proprio_full=p_all[:, C0:END], window=gen.window, ctx=W, K=cfg.K,
                                     k_max=gen.k_max, tau_ctx=gen.tau_ctx)
        z_all = torch.cat([z_all[:, :S + 1], zg.float()], 1)
        p_all = torch.cat([p_all[:, :S + 1], pg.float()], 1)
    open_loop = mode == "dream"
    act = into[:, C0:S + 1].clone()
    act[:, 0] = 0.0
    act[:, 0, D] = 1.0                                    # slot 0 carries the start flag (DemoStarts)
    task = torch.zeros(B, dtype=torch.long, device=dev)
    cache = world.dyn.transformer.make_kv_cache()
    cache.max_T = max(1, W - 1)                           # as in Imagination.rollout
    z, prop = z_all[:, C0:S + 1], p_all[:, C0:S + 1]
    dream_context(world, heads, z_hist=z, prop_hist=prop, act_hist=act, tau_ctx=world.tau_ctx,
                  task_ids=task, cache=cache, commit=True)
    rewards = torch.zeros(B, END - S - 1, device=dev)
    for t in range(S + 1, END):
        act = torch.cat([act, into[:, t:t + 1]], 1)
        if open_loop:
            z_new, p_new = dream_step(world, z_hist=z, prop_hist=prop, act_hist=act[:, -(z.shape[1] + 1):],
                                      K=cfg.K, cache=cache)
        else:
            z_new, p_new = z_all[:, t], p_all[:, t]
        z = torch.cat([z, z_new[:, None]], 1)
        prop = torch.cat([prop, p_new[:, None]], 1)
        h = dream_context(world, heads, z_hist=z[:, -1:], prop_hist=prop[:, -1:], act_hist=act[:, -1:],
                          tau_ctx=world.tau_ctx, task_ids=task, cache=cache, commit=True)
        rewards[:, t - S - 1] = agent.decode_reward(heads.reward.at(h, 0).float().softmax(-1))
    return rewards.cpu().numpy(), z


real_r, dream_r, teach_r, gen_r = [], [], [], []
show = set(fail[:cfg.video_n] + succ[:2]) if cfg.video else set()
shown = {}
for s in range(0, len(picks), cfg.batch):
    idx = picks[s:s + cfg.batch]
    eps = encode_episodes(world.tok, ds, idx, keep_video=bool(show & set(idx)))
    real_r.append(np.stack([e["rewards"][S:END - 1] for e in eps]))   # reward INTO frames S+1 .. END-1
    dr, zd = replay(eps, "dream")
    dream_r.append(dr)
    teach_r.append(replay(eps, "real")[0])
    if gen is not None:
        gen_r.append(replay(eps, "gen")[0])
    for j, i in enumerate(idx):
        if i in show:                                     # frames C0 .. END-1
            dec = world.tok.decode_latents(zd[j:j + 1]).permute(0, 1, 3, 4, 2)[0].float().cpu().numpy()
            shown[i] = (eps[j]["video"][:END], dec)
    print(f"{min(s + cfg.batch, len(picks))}/{len(picks)} replayed", flush=True)
real_r, dream_r, teach_r = (np.concatenate(x) for x in (real_r, dream_r, teach_r))
srcs = {"real": real_r, "dream": dream_r, "teacher": teach_r}
if gen_r:
    srcs["gen"] = np.concatenate(gen_r)
is_succ = np.array([False] * len(fail) + [True] * len(succ))
held = {k: r[:, -10:].mean(1) > cfg.lift_r for k, r in srcs.items()}


def summary(mask):
    out = {"n": int(mask.sum())}
    for k, r in srcs.items():
        out[f"{k}_return"] = round(float(r[mask].sum(1).mean()), 2)
        out[f"{k}_held"] = round(float(held[k][mask].mean()), 3)
        if k != "real":
            out[f"{k}_abs_err"] = round(float(np.abs(r[mask].sum(1) - real_r[mask].sum(1)).mean()), 2)
            out[f"{k}_agrees"] = round(float((held[k] == held["real"])[mask].mean()), 3)
    return out


res = {"dyn": cfg.dyn, "heads": cfg.heads, "gen_dyn": cfg.gen_dyn, "data": cfg.data, "K": cfg.K,
       "context": [C0, S], "dream": [S + 1, END - 1], "lift_r": cfg.lift_r,
       "failures": summary(~is_succ), "successes": summary(is_succ),
       **{f"corr_{k}_real": round(float(np.corrcoef(r.sum(1), real_r.sum(1))[0, 1]), 3)
          for k, r in srcs.items() if k != "real"},
       "episodes": [{"i": int(i), "success": bool(is_succ[j]),
                     **{k: round(float(r[j].sum()), 2) for k, r in srcs.items()}}
                    for j, i in enumerate(picks)]}
json.dump(res, open(cfg.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "episodes"}), flush=True)

if cfg.png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for m, c, name in ((~is_succ, "#c44e52", "real FAILURE"), (is_succ, "#4c72b0", "real success")):
        ax[0].scatter(real_r[m].sum(1), dream_r[m].sum(1), s=16, c=c, alpha=0.75, label=name)
    hi = max(real_r.sum(1).max(), dream_r.sum(1).max()) * 1.05
    ax[0].plot([0, hi], [0, hi], "k:", lw=1)
    ax[0].set_xlabel(f"real return (frames {S + 1}-{END - 1})")
    ax[0].set_ylabel("world-model return (dream, recorded actions)")
    ax[0].set_title("Does the dream reproduce the outcome?", fontsize=10)
    ax[0].legend(fontsize=8)
    t = np.arange(S + 1, END)
    for m, c, name in ((~is_succ, "#c44e52", "failures"), (is_succ, "#4c72b0", "successes")):
        ax[1].plot(t, real_r[m].mean(0), c=c, lw=2, label=f"{name}: real")
        ax[1].plot(t, dream_r[m].mean(0), c=c, lw=2, ls="--", label=f"{name}: dream")
        ax[1].plot(t, teach_r[m].mean(0), c=c, lw=1, ls=":", label=f"{name}: reward head on real frames")
        if "gen" in srcs:
            ax[1].plot(t, srcs["gen"][m].mean(0), c=c, lw=1.5, ls="-.", label=f"{name}: phase-1 model's dream")
    ax[1].axvline(S + 0.5, color="gray", lw=1)
    ax[1].set_xlabel("env step")
    ax[1].set_ylabel("reward per step (mean)")
    ax[1].set_title(f"Per-step reward: real vs world model (context ends at {S})", fontsize=10)
    ax[1].legend(fontsize=7)
    x = np.arange(2)
    names = {"real": "real rewards", "dream": "dream (phase-2 transformer)", "teacher": "reward head on real frames",
             "gen": "dream (phase-1 world model)"}
    colors = {"real": "#55a868", "dream": "#dd8452", "teacher": "#8172b2", "gen": "#937860"}
    wbar = 0.8 / len(srcs)
    for k, g in enumerate(srcs):
        vals = [held[g][~is_succ].mean(), held[g][is_succ].mean()]
        xs = x + (k - (len(srcs) - 1) / 2) * wbar
        ax[2].bar(xs, vals, wbar, color=colors[g], label=names[g])
        for xi, v in zip(xs, vals):
            ax[2].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"real FAILURES (n={int((~is_succ).sum())})", f"real successes (n={int(is_succ.sum())})"])
    ax[2].set_ylim(0, 1.12)
    ax[2].set_ylabel(f"fraction 'box held' at frame {END - 1}")
    ax[2].set_title("Box held in the window's last 10 steps?", fontsize=10)
    ax[2].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"AlohaPick world model vs reality on unseen episodes {cfg.label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(cfg.png, dpi=110)
    print("saved", cfg.png)

if cfg.video and shown:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    order = [i for i in picks if i in shown]
    pos = {i: picks.index(i) for i in order}

    def tile(img, text, color):
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 12], fill=(0, 0, 0))
        d.text((3, 0), text, fill=color)
        return np.asarray(im)

    frames = []
    for t in range(S + 1, END):
        k = t - S - 1
        cols = []
        for i in order:
            p = pos[i]
            vid, dec = shown[i]
            tag = "FAIL" if not is_succ[p] else "OK"
            top = tile(vid[t] / 255.0, f"REAL {tag} t={t} r={real_r[p, k]:.2f} R={real_r[p, :k + 1].sum():.0f}",
                       (255, 120, 120) if not is_succ[p] else (160, 200, 255))
            bot = tile(dec[t - C0], f"DREAM t={t} r={dream_r[p, k]:.2f} R={dream_r[p, :k + 1].sum():.0f}",
                       (255, 200, 60))
            cols.append(np.concatenate([top, bot], 0))
        grid = np.concatenate(cols, 1)
        bar = Image.new("RGB", (grid.shape[1], 16), (25, 25, 25))
        ImageDraw.Draw(bar).text((4, 2), f"{cfg.label}  top: real episode  -  bottom: world model replaying the SAME "
                                         f"recorded actions open loop from real frames {C0}-{S} (K={cfg.K})",
                                 fill=(255, 255, 255))
        frames.append(np.concatenate([np.asarray(bar), grid], 0))
    imageio.mimsave(cfg.video, frames, fps=10, macro_block_size=1)
    print("saved", cfg.video)
