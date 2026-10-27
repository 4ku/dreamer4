"""Does the frozen tokenizer SEE the grasp? Linear probes from one frame's latents (+ proprio) to the simulator's state.

A world model can only dream a missed grasp if its latents carry where the box is relative to the gripper. Ridge
regressions on frames of recorded HandOver episodes (``privileged.npy``: the env's 83-D observation, box pose from
qpos 16:19, left / right gripper sites at 55:58 / 64:67) predict the box position, the box relative to each gripper
and "the left hand holds the box off the table"; held-out episodes score them, next to a proprio-only baseline.

    PYTHONPATH=. python scripts/aloha/probe_latents.py --dyn <world model> --ckpt <phase-2/3 checkpoint> \
        --data <aloha-npy dir with privileged.npy> --out probe.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dreamer4.train.train_dynamics import encode_episodes
from dreamer4.train.train_pmpo import load_agent

ap = argparse.ArgumentParser()
ap.add_argument("--dyn", required=True)
ap.add_argument("--ckpt", required=True, help="any checkpoint on this world model (only its tokenizer is used)")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=3, help="use every N-th frame")
ap.add_argument("--test_frac", type=float, default=0.2)
ap.add_argument("--lam", type=float, default=1.0, help="ridge strength on standardised inputs")
cfg = ap.parse_args()

dev = torch.device("cuda")
world, _ = load_agent(argparse.Namespace(init_from=cfg.ckpt, dyn=cfg.dyn, device="cuda", data=cfg.data,
                                         reward_decode="mode", compact=0), dev)
eps = encode_episodes(world.tok, world.dataset, range(len(world.dataset)))
priv = np.load(Path(cfg.data) / "privileged.npy", mmap_mode="r")
assert len(priv) == len(eps), (len(priv), len(eps))

zs, ps, ys, ep_id = [], [], [], []
for k, e in enumerate(eps):
    T = e["z"].shape[0]
    t = np.arange(1, T, cfg.every)                          # frame 0: the reset obs has no site positions yet
    pv = np.asarray(priv[k, t], np.float32)
    box, lg, rg = pv[:, 16:19], pv[:, 55:58], pv[:, 64:67]
    held = ((np.linalg.norm(lg - box, axis=1) < 0.05) & (box[:, 2] > 0.05)).astype(np.float32)
    ys.append(np.concatenate([box, lg - box, rg - box, held[:, None]], 1))
    zs.append(e["z"][torch.as_tensor(t)].float().flatten(1).cpu().numpy())
    pr = e["proprio"]
    ps.append((pr.float().cpu().numpy() if torch.is_tensor(pr) else np.asarray(pr, np.float32))[t])
    ep_id.append(np.full(len(t), k))
Z, P, Y, E = map(np.concatenate, (zs, ps, ys, ep_id))
rng = np.random.default_rng(0)
test_eps = rng.choice(len(eps), int(cfg.test_frac * len(eps)), replace=False)
te = np.isin(E, test_eps)
print(f"{len(Y)} frames of {len(eps)} episodes ({te.sum()} test frames); latent dim {Z.shape[1]}", flush=True)


def ridge(X, Y, te, lam):
    X = torch.as_tensor(X, device=dev, dtype=torch.float64)
    Yt = torch.as_tensor(Y, device=dev, dtype=torch.float64)
    tr = torch.as_tensor(~te, device=dev)
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    X = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=dev, dtype=X.dtype)], 1)
    A = X[tr].T @ X[tr] + lam * torch.eye(X.shape[1], device=dev, dtype=X.dtype)
    W = torch.linalg.solve(A, X[tr].T @ Yt[tr])
    return (X[~tr] @ W).cpu().numpy()


names = ["box_x", "box_y", "box_z", "left-box_x", "left-box_y", "left-box_z", "right-box_x", "right-box_y",
         "right-box_z"]
near = te & (np.linalg.norm(Y[:, 3:6], axis=1) < 0.08)        # test frames with the left gripper at the box
res = {"frames": int(len(Y)), "test_frames": int(te.sum()), "near_test_frames": int(near.sum())}
for tag, X in (("proprio", P), ("latent+proprio", np.concatenate([Z, P], 1))):
    pred = ridge(X, Y, te, cfg.lam)
    true = Y[te]
    err = pred[:, :9] - true[:, :9]
    rmse = np.sqrt((err ** 2).mean(0)) * 1000
    r2 = 1 - (err ** 2).sum(0) / ((true[:, :9] - true[:, :9].mean(0)) ** 2).sum(0)
    near_te = near[te]
    near_rmse = np.sqrt((err[near_te] ** 2).mean(0)) * 1000
    h_pred, h_true = pred[:, 9] > 0.5, true[:, 9] > 0.5
    bal = 0.5 * ((h_pred & h_true).sum() / max(1, h_true.sum()) + (~h_pred & ~h_true).sum() / max(1, (~h_true).sum()))
    res[tag] = {"rmse_mm": dict(zip(names, rmse.round(1).tolist())), "r2": dict(zip(names, r2.round(3).tolist())),
                "near_rmse_mm": dict(zip(names, near_rmse.round(1).tolist())),
                "held_balanced_acc": float(bal), "held_rate": float(h_true.mean())}
    print(f"{tag:15s} RMSE mm: " + "  ".join(f"{n} {v:.1f}" for n, v in zip(names, rmse))
          + f"\n{'':15s} near the box:  " + "  ".join(f"{n} {v:.1f}" for n, v in zip(names[3:6], near_rmse[3:6]))
          + f"\n{'':15s} R2: " + "  ".join(f"{n} {v:.2f}" for n, v in zip(names, r2))
          + f"\n{'':15s} left hand holds the box: balanced accuracy {bal:.3f} (rate {h_true.mean():.2f})",
          flush=True)
json.dump(res, open(cfg.out, "w"), indent=1)
