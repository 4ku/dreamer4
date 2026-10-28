"""llm3: free the disk of collections whose latents are cached. For each dataset dir: find its latent cache under the
world model's tokenizer (the key encode_episodes uses), check that it holds every episode at its full length, then
replace each pixels_<camera>.npy by a sparse all-zero file of the same shape (written beside it, then renamed).
Nothing after the cache reads real pixels; a missing or incomplete cache leaves the dir untouched.

    python k3_sparsify.py <world-model ckpt> <data dir> ...
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dreamer4.data import open_video_dataset
from dreamer4.train.train_dynamics import latent_cache_file

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
tok = ck["config"]["tokenizer"]
t = SimpleNamespace(ckpt_path=tok["ckpt"], history=tok["history"], pack_k=tok["pack_k"])
for d in map(Path, sys.argv[2:]):
    ds = open_video_dataset(str(d), proprio=ck["config"]["data"]["proprio"], actions=True)
    f = latent_cache_file(t, ds)
    if f is None or not f.exists():
        print(d, "no cache: kept", flush=True)
        continue
    cached = torch.load(f, map_location="cpu")
    bad = [i for i in range(len(ds)) if i not in cached or cached[i].shape[0] != ds.episode_frames(i)]
    if bad:
        print(d, f"cache incomplete ({len(bad)} episodes): kept", flush=True)
        continue
    freed = 0
    for p in sorted(d.glob("pixels_*.npy")):
        a = np.load(p, mmap_mode="r")
        shape, dtype = a.shape, a.dtype
        del a
        before = p.stat().st_blocks * 512
        tmp = p.with_name(p.name + ".sparse")
        np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape).flush()
        tmp.replace(p)
        freed += before - p.stat().st_blocks * 512
    print(d, f"cache {f.name} complete ({len(ds)} episodes); freed {freed / 1e9:.1f} GB", flush=True)
