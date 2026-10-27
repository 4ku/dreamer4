"""Re-price the reward of an AlohaHandOver dataset (reward v2) without the simulator: a new dataset directory whose
frames, states and actions are symlinks to the old one, with a new reward.npy and manifest.

Reward v2 per step = (phi' - phi) - TIME - TABLE * contact + BONUS * success, and an early end without success
(out of bounds) is charged TIME for every skipped step. With phi.npy / contact.npy (collect_handover.py records them)
the new reward is rebuilt exactly; without them (e.g. episodes a policy collected) only TIME can change, exactly:
the old time cost is added back and the new one charged.

    python reprice_handover.py <old dir> <new dir> --time 0 [--table 0.1] [--old_time 0.01] [--old_table 0.1]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("old")
ap.add_argument("new")
ap.add_argument("--time", type=float, required=True)
ap.add_argument("--table", type=float, default=None, help="new table cost (needs contact.npy)")
ap.add_argument("--old_time", type=float, default=0.01)
ap.add_argument("--old_table", type=float, default=0.1)
ap.add_argument("--bonus", type=float, default=1.0)
cfg = ap.parse_args()
old, new = Path(cfg.old), Path(cfg.new)
man = json.loads((old / "manifest.json").read_text())
eps = man["episodes"]
r_old = np.load(old / "reward.npy")
N, T = r_old.shape
exact = (old / "phi.npy").exists() and (old / "contact.npy").exists()
table = cfg.old_table if cfg.table is None else cfg.table
if cfg.table is not None and cfg.table != cfg.old_table and not exact:
    raise SystemExit(f"{old} has no phi.npy / contact.npy: only the time cost can be re-priced")
new.mkdir(parents=True, exist_ok=True)
for f in old.iterdir():
    if f.name not in ("reward.npy", "manifest.json") and not (new / f.name).exists():
        os.symlink(f.resolve(), new / f.name)
r = np.zeros_like(r_old)
for i, e in enumerate(eps):
    L = int(e.get("length", T))
    if exact:
        phi = np.load(old / "phi.npy", mmap_mode="r")[i]
        c = np.load(old / "contact.npy", mmap_mode="r")[i, :L].astype(np.float32)
        x = np.diff(np.asarray(phi[:L + 1], np.float64)) - cfg.time - table * c
        if e.get("end") == "success":
            x[L - 1] += cfg.bonus
    else:
        x = np.asarray(r_old[i, :L], np.float64) + cfg.old_time - cfg.time
    if e.get("end") in ("out", "drop") and not e.get("success"):
        # an early end without success pays the skipped time (the old charge is already in r_old when not exact)
        skipped = T - L
        x[L - 1] += (0.0 if exact else cfg.old_time * skipped) - cfg.time * skipped
    r[i, :L] = x
    e["return"] = round(float(x.sum()), 3)
np.save(new / "reward.npy", r.astype(np.float32))
man["reward_v2"] = {**man.get("reward_v2", {}), "time": cfg.time, "table": table, "repriced_from": str(old)}
(new / "manifest.json").write_text(json.dumps(man, indent=1))
ret = np.array([e["return"] for e in eps])
print(f"{new}: {N} episodes re-priced ({'exact' if exact else 'time only'}), time {cfg.time}, table {table}: "
      f"return {ret.mean():.3f} (success {np.mean([e['success'] for e in eps]):.3f})")
