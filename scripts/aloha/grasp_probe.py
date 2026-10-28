"""Can one frame before a grasp tell whether it will hold? Probes from the frames around the moment the left gripper
starts closing to the episode's grasp outcome, on recorded HandOver episodes with mixed outcomes (``ho_off_fixed_1024``:
the expert sees the box 2-12 mm off, ~40 % of its grasps miss).

A world model can only dream which grasps hold if its latents carry that BEFORE the gripper closes; RL in dreams has
nothing to learn a precise grasp from otherwise. Inputs, scored on held-out episodes (ROC AUC, balanced accuracy) at
every frame offset around the closing step:

    privileged   the simulator's left gripper - box position and box orientation (what decides it: the ceiling)
    proprio      the 14-D joint state alone
    latents      one frame's tokenizer latents + proprio

each by a ridge-logistic read-out and a small MLP. The latent rows say how much of the ceiling the tokenizer keeps.

    PYTHONPATH=. python scripts/aloha/grasp_probe.py --tok <tokenizer ckpt> [--history 4 --pack_k 1] \
        --data <aloha-npy dir with privileged.npy> --out probe.json
    (or --dyn <world model>: its tokenizer)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dreamer4.data import open_video_dataset
from dreamer4.models.tokenizer import FrozenTokenizer
from dreamer4.train.train_dynamics import encode_episodes

ap = argparse.ArgumentParser()
ap.add_argument("--tok", default="")
ap.add_argument("--history", type=int, default=4)
ap.add_argument("--pack_k", type=int, default=1)
ap.add_argument("--dyn", default="", help="take the tokenizer (and its history / packing) from this world model")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--offsets", default="-8,-4,-2,-1,0,2,4,8,15")
ap.add_argument("--test_frac", type=float, default=0.25)
ap.add_argument("--seeds", type=int, default=3, help="MLP restarts (averaged)")
cfg = ap.parse_args()
dev = torch.device("cuda")

if cfg.dyn:
    ref = torch.load(cfg.dyn, map_location="cpu", weights_only=False)["config"]["tokenizer"]
    cfg.tok, cfg.history, cfg.pack_k = ref["ckpt"], ref["history"], ref["pack_k"]
tok = FrozenTokenizer(cfg.tok, history=cfg.history, pack_k=cfg.pack_k, device=dev)
ds = open_video_dataset(cfg.data, proprio="state", actions=True)
eps = encode_episodes(tok, ds, range(len(ds)))
priv = np.load(Path(cfg.data) / "privileged.npy", mmap_mode="r")
state = np.load(Path(cfg.data) / "state.npy", mmap_mode="r")
offsets = [int(k) for k in cfg.offsets.split(",")]

# the closing step: the left gripper (state 6, ~0.062 open, ~0.048 on the box, ~0.02 shut on nothing) first sits 5 mm
# below its widest opening so far -- the same test for grasps that hold and grasps that miss, so the offsets line up
# before contact either way. The outcome: the left hand held the box above the table at any step (probe_latents.py's
# test: within 5 cm, box higher than 5 cm).
rows, label = [], []
for k, e in enumerate(eps):
    T = e["z"].shape[0]
    g = np.asarray(state[k, :T, 6], np.float32)
    shut = np.nonzero((np.arange(T) >= 3) & (g < np.maximum.accumulate(g) - 0.005))[0]
    if len(shut) == 0:
        continue
    tc = int(shut[0])
    if tc + min(offsets) < 1 or tc + max(offsets) >= T:
        continue
    pv = np.asarray(priv[k, :T], np.float32)
    box, lg = pv[:, 16:19], pv[:, 55:58]
    held = (np.linalg.norm(lg - box, axis=1) < 0.05) & (box[:, 2] > 0.05)
    rows.append((k, tc))
    label.append(float(held.any()))
label = np.asarray(label, np.float32)
print(f"{len(rows)} of {len(eps)} episodes with a closing step; held {label.mean():.3f}; "
      f"closing step median {np.median([tc for _, tc in rows]):.0f}", flush=True)
rng = np.random.default_rng(0)
test = np.zeros(len(rows), bool)
test[rng.choice(len(rows), int(cfg.test_frac * len(rows)), replace=False)] = True


def features(kind, off):
    out = []
    for k, tc in rows:
        t = tc + off
        if kind == "privileged":
            pv = np.asarray(priv[k, t], np.float32)
            out.append(np.concatenate([pv[55:58] - pv[16:19], pv[19:23]]))     # left gripper - box, box quaternion
        else:
            pr = eps[k]["proprio"]
            pr = (pr.float().cpu().numpy() if torch.is_tensor(pr) else np.asarray(pr, np.float32))[t]
            if kind == "proprio":
                out.append(pr)
            else:
                out.append(np.concatenate([eps[k]["z"][t].float().flatten().cpu().numpy(), pr]))
    X = torch.as_tensor(np.stack(out), dtype=torch.float32, device=dev)
    mu, sd = X[~torch.as_tensor(test)].mean(0), X[~torch.as_tensor(test)].std(0).clamp_min(1e-6)
    return (X - mu) / sd


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def fit(X, mlp, seed):
    torch.manual_seed(seed)
    y = torch.as_tensor(label, device=dev)
    tr = torch.as_tensor(~test, device=dev)
    d = X.shape[1]
    net = (torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 256),
                               torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 1))
           if mlp else torch.nn.Linear(d, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2 if mlp else 0.0)
    wd = 0.0 if mlp else 1e-3 * d / 100
    # the training episodes' last fifth chooses the step count (early stopping)
    idx = torch.nonzero(tr).flatten()
    val = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))[: len(idx) // 5].to(dev)]
    fit_m = tr.clone()
    fit_m[val] = False
    best, best_state = -1.0, None
    for step in range(1, 1501):
        net.train()
        logit = net(X[fit_m]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[fit_m])
        if not mlp:
            loss = loss + wd * (net.weight ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            net.eval()
            with torch.no_grad():
                a = auc(net(X[val]).squeeze(-1).cpu().numpy(), label[val.cpu().numpy()])
            if a > best:
                best, best_state = a, {n: p.detach().clone() for n, p in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        s = net(X[torch.as_tensor(test, device=dev)]).squeeze(-1).cpu().numpy()
    return s


res = {"episodes": len(rows), "held_rate": float(label.mean()), "test_episodes": int(test.sum()),
       "closing_step_median": float(np.median([tc for _, tc in rows])), "tok": cfg.tok, "offsets": offsets,
       "probes": {}}
yt = label[test]
for kind in ("privileged", "proprio", "latents"):
    for mlp in (False, True):
        name = f"{kind} {'mlp' if mlp else 'linear'}"
        res["probes"][name] = {}
        for off in offsets:
            X = features(kind, off)
            s = np.mean([fit(X, mlp, sd) for sd in range(cfg.seeds if mlp else 1)], axis=0)
            pred = s > 0
            bal = 0.5 * ((pred & (yt > 0.5)).sum() / max(1, (yt > 0.5).sum())
                         + (~pred & (yt < 0.5)).sum() / max(1, (yt < 0.5).sum()))
            res["probes"][name][str(off)] = {"auc": round(auc(s, yt), 4), "bal_acc": round(float(bal), 4)}
        print(f"{name:20s} AUC by offset from closing: " + "  ".join(
            f"{o:+d}: {res['probes'][name][str(o)]['auc']:.3f}" for o in offsets), flush=True)
json.dump(res, open(cfg.out, "w"), indent=1)
print("saved", cfg.out)
