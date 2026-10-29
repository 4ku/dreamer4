"""Collect AlohaHandOver episodes for the Dreamer4 world model, written straight into the aloha-npy layout.

The demonstrator is the April state-only SAC expert (the fork's ``learning/train_jax_sac.py``, run 65, return 6.09 on
the env's own reward; success 0.977 under reward v2). It acts on the env's privileged 83-D state; the episodes record
what Dreamer sees: three cameras, the 14-D joint state, the applied (noisy) action and the reward-v2 labels of
:mod:`dreamer4.envs.aloha_handover_v2` (green target, success ends the episode, drops do not). One per-episode dial:
Gaussian noise of std sigma on the expert's action -- a share CLEAN_FRAC of the episodes is noise-free (the clones
learn from those), the rest draws sigma ~ U(NOISE_LO, NOISE_HI). The expert shrugs noise off (success 0.96 / 0.88 /
0.77 at sigma 0.1 / 0.3 / 0.5), so a second dial makes it fail and recover: a share GLITCH_FRAC of the episodes gets
one burst of uniformly random actions, GLITCH_LEN steps long, starting at a step in GLITCH_T (drops at the hand-over,
missed grasps, the box knocked over -- and what the expert does next). Glitched episodes are never cloned.

    SAC=<checkpoint dir> N=2000 NWORLD=64 OUT=<dir> python collect_handover.py   (in the fork's venv)

Layout (frames T+1 = 251 including the final one, actions / rewards T = 250):
    state.npy (N, T+1, 14) f32, action.npy (N, T, 14) f32, reward.npy (N, T) f32,
    pixels_view_0|1|2.npy (N, T+1, 128, 128, 3) u8, manifest.json (each episode's length, end, noise, return),
    phi.npy (N, T+1) f32 and contact.npy (N, T) u8: the reward's ingredients, to re-price it without the simulator;
    privileged.npy (N, T+1, 83) f32: the env's own observation (joints, velocities, box, grippers, box - target, time)
    -- enough to rebuild a reward of another shape offline

RELABEL=1 with OUT = an existing dataset (and its N / NWORLD / SEED / dials) replays it, checks the replay against the
recorded states and actions, and rewrites only reward.npy, phi.npy, contact.npy and the manifest. The replay must be
bit-exact: on a GPU shared with other jobs MJX diverges within a batch (state 0.25 off, measured), so re-label only on
an idle GPU, or rebuild offline from privileged.npy.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.sac import checkpoint as sac_ckpt

N = int(os.environ.get("N", 2000))
NWORLD = int(os.environ.get("NWORLD", 64))
OUT = pathlib.Path(os.environ["OUT"])
SAC = os.path.abspath(os.environ["SAC"])
SEED = int(os.environ.get("SEED", 0))
CLEAN_FRAC = float(os.environ.get("CLEAN_FRAC", 0.5))
NOISE_LO, NOISE_HI = float(os.environ.get("NOISE_LO", 0.05)), float(os.environ.get("NOISE_HI", 0.5))
GLITCH_FRAC = float(os.environ.get("GLITCH_FRAC", 0.0))
GLITCH_LEN = [int(x) for x in os.environ.get("GLITCH_LEN", "10,50").split(",")]
GLITCH_T = [int(x) for x in os.environ.get("GLITCH_T", "20,150").split(",")]
# KICK_FRAC: a share of the episodes where the box gets one horizontal push (KICK_SPEED m/s, random direction) the
# moment the left gripper first reaches it (closer than KICK_DIST, box still on the table): the first grasp closes on
# air and the expert has to re-grasp. Every recorded action stays the expert's own, so these corrections are cloned.
KICK_FRAC = float(os.environ.get("KICK_FRAC", 0.0))
KICK_SPEED = [float(x) for x in os.environ.get("KICK_SPEED", "0.2,0.5").split(",")]
KICK_DIST = float(os.environ.get("KICK_DIST", 0.05))   # the gripper site is ~3.8 cm from the box at the grasp
# OFF_FRAC: a share of the episodes where the expert sees the box shifted horizontally by OFF (m, random direction) from
# step OFF_FROM until OFF_UNTIL: its first grasp lands off-centre (some hold, some miss), then it sees the true box and
# re-grasps. The world model learns which grasps hold; every recorded action is still the expert's. A window after the
# lift (OFF_FROM ~38) moves the right hand's grasp instead: near-miss hand-overs.
OFF_FRAC = float(os.environ.get("OFF_FRAC", 0.0))
OFF = [float(x) for x in os.environ.get("OFF", "0.01,0.03").split(",")]
OFF_FROM = int(os.environ.get("OFF_FROM", 0))
OFF_UNTIL = int(os.environ.get("OFF_UNTIL", 45))
BOX_OBS = ((16, 17), (49, 50), (52, 53), (79, 80))   # the box's x, y in the 83-D obs: qpos, top, bottom, box - target
STATS_FROM = os.environ.get("STATS_FROM", "")    # copy this dataset's proprio standardisation
RELABEL = os.environ.get("RELABEL", "") == "1"   # replay OUT's episodes (same N / NWORLD / SEED / dials), check them
                                                 # against its states and actions, rewrite only rewards + manifest
REWARD_CONSTANTS = ("BONUS", "TIME", "TABLE")    # env vars overriding reward-v2 constants (recorded in the manifest)
CAM_RES_HW = (128, 128)
CAMERAS = ("view_0", "view_1", "view_2")


def load_hv2():
    path = pathlib.Path(__file__).resolve().parents[2] / "dreamer4" / "envs" / "aloha_handover_v2.py"
    spec = importlib.util.spec_from_file_location("aloha_handover_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in REWARD_CONSTANTS:
        if name in os.environ:
            setattr(mod, name, float(os.environ[name]))
    return mod


def episode_ends(done, ok):
    """(m, T) v2 done / success flags -> per-episode length (transitions) and how it ended."""
    T = done.shape[1]
    first = np.where(done.any(1), done.argmax(1), T - 1)
    end = np.where(ok.any(1), "success", np.where(done.any(1), "out", "time"))
    return first + 1, end


def make_rollout_fn(hv2, env, nworld):
    policy = sac_ckpt.load_policy(SAC, deterministic=True)
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)
    proprio = jax.vmap(lambda d: hv2.proprio(env, d))
    init_aux = jax.vmap(lambda s: hv2.init_aux(env, s))
    v2_step = jax.vmap(lambda s, a: hv2.step_reward(env, s, a))
    potential = jax.vmap(lambda s: hv2.potential(env, s.data, s.info))
    u8 = lambda x: (jp.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(jp.uint8)
    T = hv2.EPISODE_STEPS

    def frames(state):
        return {c: u8(state.obs[f"pixels/{c}"]) for c in CAMERAS}

    box_dof = int(env._mj_model.jnt_dofadr[env._mj_model.body_jntadr[env._box_body]])

    def kick(state, kick_v, kicked, t):
        """Push the box of the worlds whose left gripper has just reached it (once per episode)."""
        box = state.data.xpos[:, env._box_body]
        d_lg = jp.linalg.norm(state.data.site_xpos[:, env._left_gripper_site] - box, axis=-1)
        fire = (jp.linalg.norm(kick_v, axis=-1) > 0) & (kicked < 0) & (d_lg < KICK_DIST) & (box[:, 2] < 0.045)
        qvel = state.data.qvel.at[:, box_dof:box_dof + 2].set(
            jp.where(fire[:, None], kick_v, state.data.qvel[:, box_dof:box_dof + 2]))
        return state.replace(data=state.data.replace(qvel=qvel)), jp.where(fire, t, kicked)

    def seen(obs, off, t):
        """The expert's view: the box shifted by ``off`` (nworld, 2) from OFF_FROM until OFF_UNTIL."""
        on = (t >= OFF_FROM) & (t < OFF_UNTIL)
        for ix, iy in BOX_OBS:
            obs = obs.at[:, ix].add(jp.where(on, off[:, 0], 0.0)).at[:, iy].add(jp.where(on, off[:, 1], 0.0))
        return obs

    def rollout(rngs, sigma, g_start, g_len, kick_v, off, noise_key):
        state = reset(rngs)
        k_noise, k_glitch = jax.random.split(noise_key)
        noise = jax.random.normal(k_noise, (T, nworld, 14)) * sigma[None, :, None]
        wild = jax.random.uniform(k_glitch, (T, nworld, 14), minval=-1.0, maxval=1.0)
        t = jp.arange(T)[:, None]
        glitch = (t >= g_start[None]) & (t < (g_start + g_len)[None])          # (T, nworld)
        aux = init_aux(state)

        def body_fn(carry, xs):
            state, aux, kicked = carry
            eps, rand, g, ti = xs
            state, kicked = kick(state, kick_v, kicked, ti)
            clean, _ = policy(seen(state.obs["state"], off, ti), jax.random.PRNGKey(0))
            action = jp.where(g[:, None], rand, jp.clip(clean + eps, -1.0, 1.0))   # recorded = applied
            nxt = step(state, action)
            out = {"state": proprio(state.data), "priv": state.obs["state"], "action": action, **frames(state)}
            out["reward"], out["v2_done"], out["v2_ok"], aux = v2_step(nxt, aux)
            out["phi"], out["contact"] = aux["phi"], potential(nxt)[1]
            return (nxt, aux, kicked), out

        phi0 = aux["phi"]
        kicked0 = jp.full((nworld,), -1, jp.int32)
        (final, _, kicked), traj = jax.lax.scan(body_fn, (state, aux, kicked0),
                                                (noise, wild, glitch, jp.arange(T, dtype=jp.int32)))
        return traj, {"state": proprio(final.data), "priv": final.obs["state"], "phi0": phi0, "kicked": kicked,
                      **frames(final)}

    return jax.jit(rollout), T


def main():
    hv2 = load_hv2()
    env = hv2.make_env({"vision": True, "vision_config.nworld": NWORLD, "vision_config.cam_res": CAM_RES_HW,
                        "impl": "warp"})
    rollout, T = make_rollout_fn(hv2, env, NWORLD)
    OUT.mkdir(parents=True, exist_ok=True)
    mm = lambda name, shape, dtype: np.lib.format.open_memmap(OUT / name, mode="w+", dtype=dtype, shape=shape)
    arr = {"reward": mm("reward.npy", (N, T), np.float32),
           "phi": mm("phi.npy", (N, T + 1), np.float32),          # phi before the first step and after each one
           "contact": mm("contact.npy", (N, T), np.uint8)}        # a finger on the table after each step
    if RELABEL:
        old = {"state": np.load(OUT / "state.npy", mmap_mode="r"), "action": np.load(OUT / "action.npy", mmap_mode="r")}
    else:
        d_priv = env.observation_size["state"]
        d_priv = int(d_priv[0]) if isinstance(d_priv, (tuple, list)) else int(d_priv)
        arr.update({"state": mm("state.npy", (N, T + 1, 14), np.float32),
                    "priv": mm("privileged.npy", (N, T + 1, d_priv), np.float32),
                    "action": mm("action.npy", (N, T, 14), np.float32),
                    **{c: mm(f"pixels_{c}.npy", (N, T + 1, *CAM_RES_HW, 3), np.uint8) for c in CAMERAS}})
    rng = np.random.default_rng(SEED)
    episodes, n, b, t_start = [], 0, 0, time.time()
    while n < N:
        t0 = time.time()
        m = min(NWORLD, N - n)
        sigma = np.where(rng.random(NWORLD) < CLEAN_FRAC, 0.0,
                         rng.uniform(NOISE_LO, NOISE_HI, NWORLD)).astype(np.float32)
        g_len = np.where(rng.random(NWORLD) < GLITCH_FRAC,
                         rng.integers(GLITCH_LEN[0], GLITCH_LEN[1] + 1, NWORLD), 0).astype(np.int32)
        g_start = rng.integers(GLITCH_T[0], GLITCH_T[1] + 1, NWORLD).astype(np.int32)
        ang = rng.uniform(0, 2 * np.pi, NWORLD)
        speed = np.where(rng.random(NWORLD) < KICK_FRAC, rng.uniform(KICK_SPEED[0], KICK_SPEED[1], NWORLD), 0.0)
        kick_v = (speed[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)).astype(np.float32)
        ang_o = rng.uniform(0, 2 * np.pi, NWORLD)
        mag = np.where(rng.random(NWORLD) < OFF_FRAC, rng.uniform(OFF[0], OFF[1], NWORLD), 0.0)
        off = (mag[:, None] * np.stack([np.cos(ang_o), np.sin(ang_o)], 1)).astype(np.float32)
        keys = jax.random.split(jax.random.PRNGKey(SEED * 100003 + b), NWORLD + 1)
        traj, final = rollout(keys[:NWORLD], jp.asarray(sigma), jp.asarray(g_start), jp.asarray(g_len),
                              jp.asarray(kick_v), jp.asarray(off), keys[-1])
        traj = {k: np.swapaxes(np.asarray(v), 0, 1)[:m] for k, v in traj.items()}   # (m, T, ...)
        final = {k: np.asarray(v)[:m] for k, v in final.items()}
        sl = slice(n, n + m)
        if RELABEL:
            d_state = float(np.abs(np.asarray(old["state"][sl][:, :T]) - traj["state"]).max())
            d_action = float(np.abs(np.asarray(old["action"][sl]) - traj["action"]).max())
            if max(d_state, d_action) > 1e-3:
                raise SystemExit(f"replay of batch {b + 1} differs from the recorded episodes: state {d_state:.2e} "
                                 f"action {d_action:.2e} (same N / NWORLD / SEED / dials?)")
        else:
            arr["state"][sl] = np.concatenate([traj["state"], final["state"][:, None]], 1)
            arr["priv"][sl] = np.concatenate([traj["priv"], final["priv"][:, None]], 1)
            for c in CAMERAS:
                arr[c][sl] = np.concatenate([traj[c], final[c][:, None]], 1)
            arr["action"][sl] = traj["action"]
        arr["reward"][sl] = traj["reward"]
        arr["phi"][sl] = np.concatenate([final["phi0"][:, None], traj["phi"]], 1)
        arr["contact"][sl] = traj["contact"] > 0.5
        ret = traj["reward"].sum(1)
        length, end = episode_ends(traj["v2_done"], traj["v2_ok"])
        success = end == "success"
        for i in range(m):
            rec = {"noise": round(float(sigma[i]), 3), "success": bool(success[i]),
                   "return": round(float(ret[i]), 3), "length": int(length[i]), "end": str(end[i])}
            if g_len[i]:
                rec["glitch"] = [int(g_start[i]), int(g_len[i])]
                rec["noise"] = max(rec["noise"], 1.0)        # never cloned (the adapter clones noise-free ones)
            if speed[i] > 0:
                rec["kick"] = [int(final["kicked"][i]), round(float(speed[i]), 3)]   # step (-1: never fired), m/s
            if mag[i] > 0:
                rec["off"] = [round(float(off[i, 0]), 4), round(float(off[i, 1]), 4)]   # the expert's box offset (m)
            episodes.append(rec)
        n += m
        b += 1
        clean = sigma[:m] == 0
        print(f"batch {b}: {m} episodes, success {success.mean():.2f} (clean {success[clean].mean() if clean.any() else float('nan'):.2f}), "
              f"return {ret.mean():.2f}, success step {length[success].mean() if success.any() else float('nan'):.0f} | "
              f"{n}/{N} | {time.time() - t0:.1f} s", flush=True)
    for a in arr.values():
        a.flush()
    flat = np.asarray(old["state"] if RELABEL else arr["state"], np.float64).reshape(-1, 14)
    mean, std = flat.mean(0).round(6).tolist(), flat.std(0).round(6).tolist()
    if STATS_FROM:
        ref = json.loads((pathlib.Path(STATS_FROM) / "manifest.json").read_text())
        mean, std = ref["state_mean"], ref["state_std"]
    manifest = {"format": "aloha-npy", "cameras": list(CAMERAS), "downscale": 1, "n_episodes": N,
                "state_mean": mean, "state_std": std,
                "env": {"task": "handover", "reward": "v2", "cam_res": list(CAM_RES_HW), "max_steps": T},
                "reward_v2": {"bonus": hv2.BONUS, "time": hv2.TIME, "table": hv2.TABLE, "hold": hv2.HOLD},
                "collector": {"demonstrator": SAC, "clean_frac": CLEAN_FRAC, "noise": [NOISE_LO, NOISE_HI],
                              "glitch_frac": GLITCH_FRAC, "glitch_len": GLITCH_LEN, "glitch_t": GLITCH_T,
                              "kick_frac": KICK_FRAC, "kick_speed": KICK_SPEED, "kick_dist": KICK_DIST,
                              "off_frac": OFF_FRAC, "off": OFF, "off_from": OFF_FROM, "off_until": OFF_UNTIL,
                              "target": os.environ.get("HANDOVER_TARGET", "random"),
                              "seed": SEED, "relabelled": bool(RELABEL)},
                "episodes": episodes}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    s = np.array([e["success"] for e in episodes])
    r = np.array([e["return"] for e in episodes])
    print(f"DONE {N} episodes in {(time.time() - t_start) / 60:.1f} min: success {s.mean():.3f}, "
          f"return {r.mean():.2f} (max {r.max():.2f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
