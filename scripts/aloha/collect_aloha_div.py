"""DIVERSE AlohaPick episodes for world-model pretraining, written straight into the aloha-npy layout.

The scripted IK expert of collect_aloha_npy.py (approach above the box, descend, close, lift) with every
dial that makes a pick go wrong in a different way, so the world model sees the failure modes a learned
policy produces (a box gripped off-centre that tilts or slips, a late or early close, a drop) and not only
clean lifts:
  speed    -- every phase boundary scaled by s ~ U(SPEED_LO, SPEED_HI);
  offset   -- the grasp target moved by N(0, OFFSET_XY) in x and y and U(OFFSET_Z_LO, OFFSET_Z_HI) in z;
  close    -- the gripper closes CLOSE_JITTER steps early or late at most (uniform);
  noise    -- time-correlated (Ornstein-Uhlenbeck) noise on the right arm: stationary std
              sigma ~ U(NOISE_LO, NOISE_HI) for a share 1 - CLEAN_FRAC of the episodes, correlation per step
              (1 - theta) with theta ~ U(0.05, 0.5);
  release  -- with probability RELEASE_P the gripper re-opens at a random step after the close (a drop);
  random   -- a share RANDOM_FRAC of the episodes has no expert at all: correlated noise only (exploration).
Every episode is kept; the manifest records each episode's dials, success and return, and marks only clean
expert successes (no offset, noise, jitter, release) as demonstrations (noise 0), the rest noise 1.

    N=2000 NWORLD=128 OUT=/mnt/ssd1/aloha/data/aloha_div_2000 BASE=<the mix> python collect_aloha_div.py

BASE: a dataset whose proprio standardisation is copied (needed when the two are merged).

Layout as collect_aloha_npy.py (frames T+1 = 121, actions/rewards T = 120).
"""
from __future__ import annotations

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
from mujoco import mjx

from mujoco_playground import manipulation

N = int(os.environ.get("N", 2000))
NWORLD = int(os.environ.get("NWORLD", 128))
OUT = pathlib.Path(os.environ["OUT"])
SEED = int(os.environ.get("SEED", 7))
SPEED_LO, SPEED_HI = float(os.environ.get("SPEED_LO", 0.5)), float(os.environ.get("SPEED_HI", 1.2))
OFFSET_XY = float(os.environ.get("OFFSET_XY", 0.012))
OFFSET_Z_LO, OFFSET_Z_HI = float(os.environ.get("OFFSET_Z_LO", -0.003)), float(os.environ.get("OFFSET_Z_HI", 0.02))
CLOSE_JITTER = int(os.environ.get("CLOSE_JITTER", 8))
CLEAN_FRAC = float(os.environ.get("CLEAN_FRAC", 0.3))
NOISE_LO, NOISE_HI = float(os.environ.get("NOISE_LO", 0.05)), float(os.environ.get("NOISE_HI", 0.6))
RELEASE_P = float(os.environ.get("RELEASE_P", 0.2))
RANDOM_FRAC = float(os.environ.get("RANDOM_FRAC", 0.1))
DIAL_FRAC = float(os.environ.get("DIAL_FRAC", 0.7))   # share of expert episodes with offset / jitter
CAM_RES_HW = (128, 128)
CAMERAS = ("view_0", "view_2")
ACTION_SCALE = 0.015
PHASE_A, PHASE_B, PHASE_C = 30, 50, 65      # the fork's expert: approach / descend / close, then lift
RIGHT_ARM_JOINTS = ("right/waist", "right/shoulder", "right/elbow",
                    "right/forearm_roll", "right/wrist_angle", "right/wrist_rotate")


def make_rollout_fn(env, nworld):
    mj_model, mjx_model = env.mj_model, env.mjx_model
    site = mj_model.site("right/gripper").id
    body = int(mj_model.site_bodyid[site])
    dofs = jp.array([mj_model.jnt_dofadr[mj_model.joint(j).id] for j in RIGHT_ARM_JOINTS])
    box_qadr = int(env._box_qadr)
    ep_len = int(env._config.episode_length)

    def policy_one(state, box0, grasp_pos, t, pa, pb, pc, offset, t_rel, expert):
        aim = box0 + offset
        target = jp.where(t < pa, aim + jp.array([0.0, 0.0, 0.10]),
                          jp.where(t < pc, aim + jp.array([0.0, 0.0, 0.005]),
                                   grasp_pos + jp.array([0.0, 0.0, 0.10])))
        eef = state.data.site_xpos[site]
        jacp, _ = mjx.jac(mjx_model, state.data, eef, body)
        J = jacp.T[:, dofs]
        dq = J.T @ jp.linalg.solve(J @ J.T + 1e-3 * jp.eye(3), 0.5 * (target - eef))
        action = jp.zeros(14, dtype=jp.float32).at[7:13].set(jp.clip(dq / ACTION_SCALE, -1.0, 1.0))
        finger = jp.where((t < pb) | (t >= t_rel), 0.04, 0.0)
        action = action.at[13].set(jp.clip((finger - state.data.ctrl[13]) / ACTION_SCALE, -1.0, 1.0))
        box_now = state.data.qpos[box_qadr:box_qadr + 3]
        return action * expert, jp.where(t == pc - 1, box_now, grasp_pos)

    policy = jax.vmap(policy_one, in_axes=(0, 0, 0, None, 0, 0, 0, 0, 0, 0))
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)
    kin = jax.vmap(lambda d: mjx.kinematics(mjx_model, d))
    u8 = lambda x: (jp.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(jp.uint8)

    def rollout(rngs, speed, sigma, theta, offset, jitter, t_rel, expert, noise_key):
        state = reset(rngs)
        state = state.replace(data=kin(state.data))
        box0 = state.data.qpos[:, box_qadr:box_qadr + 3]
        pa = jp.round(PHASE_A * speed).astype(jp.int32)
        pb = jp.round(PHASE_B * speed).astype(jp.int32) + jitter
        pc = jp.maximum(jp.round(PHASE_C * speed).astype(jp.int32), pb + 1)
        white = jax.random.normal(noise_key, (ep_len, nworld, 14))
        white = white.at[:, :, :7].set(0.0)                       # the left arm stays frozen

        def body_fn(carry, xs):
            state, grasp, eps = carry
            t, w = xs
            # OU noise with stationary std sigma: eps <- (1 - theta) eps + sqrt(1 - (1 - theta)^2) sigma w
            eps = (1.0 - theta)[:, None] * eps + jp.sqrt(1.0 - (1.0 - theta) ** 2)[:, None] * sigma[:, None] * w
            action, grasp = policy(state, box0, grasp, t, pa, pb, pc, offset, t_rel, expert)
            action = jp.clip(action + eps, -1.0, 1.0)
            nxt = step(state, action)
            out = {"state": state.obs["state"], "view_0": u8(state.obs["pixels/view_0"]),
                   "view_2": u8(state.obs["pixels/view_2"]), "action": action, "reward": nxt.reward,
                   "box_z": nxt.data.qpos[:, box_qadr + 2],
                   "grasping": nxt.info["grasping"].astype(jp.float32)}
            return (nxt, grasp, eps), out

        init = (state, jp.zeros((nworld, 3)), jp.zeros((nworld, 14)))
        (final, _, _), traj = jax.lax.scan(body_fn, init, (jp.arange(ep_len), white))
        final_obs = {"state": final.obs["state"], "view_0": u8(final.obs["pixels/view_0"]),
                     "view_2": u8(final.obs["pixels/view_2"])}
        return traj, final_obs

    return jax.jit(rollout), ep_len


def main():
    env = manipulation.load("AlohaPick", config_overrides={
        "vision": True, "vision_config.nworld": NWORLD, "vision_config.cam_res": CAM_RES_HW})
    rollout, T = make_rollout_fn(env, NWORLD)
    OUT.mkdir(parents=True, exist_ok=True)
    mm = lambda name, shape, dtype: np.lib.format.open_memmap(OUT / name, mode="w+", dtype=dtype, shape=shape)
    arr = {"state": mm("state.npy", (N, T + 1, 14), np.float32),
           "action": mm("action.npy", (N, T, 14), np.float32),
           "reward": mm("reward.npy", (N, T), np.float32),
           **{c: mm(f"pixels_{c}.npy", (N, T + 1, *CAM_RES_HW, 3), np.uint8) for c in CAMERAS}}
    rng = np.random.default_rng(SEED)
    episodes, n, b, t_start = [], 0, 0, time.time()
    while n < N:
        t0 = time.time()
        m = min(NWORLD, N - n)
        W = NWORLD
        speed = rng.uniform(SPEED_LO, SPEED_HI, W).astype(np.float32)
        expert = (rng.random(W) >= RANDOM_FRAC).astype(np.float32)
        dial = (rng.random(W) < DIAL_FRAC) & (expert > 0)
        offset = np.zeros((W, 3), np.float32)
        offset[:, :2] = rng.normal(0.0, OFFSET_XY, (W, 2))
        offset[:, 2] = rng.uniform(OFFSET_Z_LO, OFFSET_Z_HI, W)
        offset[~dial] = 0.0
        jitter = np.where(dial, rng.integers(-CLOSE_JITTER, CLOSE_JITTER + 1, W), 0).astype(np.int32)
        noisy = (rng.random(W) >= CLEAN_FRAC) | (expert == 0)
        sigma = np.where(noisy, rng.uniform(NOISE_LO, NOISE_HI, W), 0.0).astype(np.float32)
        sigma[expert == 0] = rng.uniform(0.3, 0.8, int((expert == 0).sum()))
        theta = rng.uniform(0.05, 0.5, W).astype(np.float32)
        pc_est = np.round(PHASE_C * speed).astype(int)
        release = (rng.random(W) < RELEASE_P) & (expert > 0)
        t_rel = np.where(release, [rng.integers(min(p + 5, T - 11), T - 10) for p in pc_est], 10 * T).astype(np.int32)
        keys = jax.random.split(jax.random.PRNGKey(SEED * 100003 + b), W + 1)
        traj, final = rollout(keys[:W], jp.asarray(speed), jp.asarray(sigma), jp.asarray(theta),
                              jp.asarray(offset), jp.asarray(jitter), jp.asarray(t_rel), jp.asarray(expert),
                              keys[-1])
        traj = {k: np.swapaxes(np.asarray(v), 0, 1)[:m] for k, v in traj.items()}   # (m, T, ...)
        final = {k: np.asarray(v)[:m] for k, v in final.items()}
        sl = slice(n, n + m)
        arr["state"][sl] = np.concatenate([traj["state"], final["state"][:, None]], 1)
        for c in CAMERAS:
            arr[c][sl] = np.concatenate([traj[c], final[c][:, None]], 1)
        arr["action"][sl] = traj["action"]
        arr["reward"][sl] = traj["reward"]
        success = (traj["box_z"][:, -1] > 0.10) & (traj["grasping"][:, -1] > 0.5)
        ret = traj["reward"].sum(1)
        for i in range(m):
            clean = expert[i] > 0 and not dial[i] and sigma[i] == 0 and not release[i]
            episodes.append({"speed": round(float(speed[i]), 3), "noise": 0.0 if clean else 1.0,
                             "sigma": round(float(sigma[i]), 3), "theta": round(float(theta[i]), 3),
                             "offset": [round(float(x), 4) for x in offset[i]], "jitter": int(jitter[i]),
                             "release": int(t_rel[i]) if release[i] else -1, "expert": bool(expert[i] > 0),
                             "success": bool(success[i]), "return": round(float(ret[i]), 2)})
        n += m
        b += 1
        print(f"batch {b}: {m} episodes, success {success.mean():.2f}, return {ret.mean():.1f} | "
              f"{n}/{N} | {time.time() - t0:.1f} s", flush=True)
    for a in arr.values():
        a.flush()
    if os.environ.get("BASE"):
        # merged with BASE (e.g. the scripted mix), every part must standardise the proprio alike
        base = json.loads((pathlib.Path(os.environ["BASE"]) / "manifest.json").read_text())
        mean, std = base["state_mean"], base["state_std"]
    else:
        flat = np.asarray(arr["state"], np.float64).reshape(-1, 14)
        mean, std = flat.mean(0).round(6).tolist(), flat.std(0).round(6).tolist()
    manifest = {"format": "aloha-npy", "cameras": list(CAMERAS), "downscale": 1, "n_episodes": N,
                "state_mean": mean, "state_std": std,
                "env": {"cam_res": list(CAM_RES_HW), "max_steps": T},
                "collector": {"script": "collect_aloha_div.py", "speed": [SPEED_LO, SPEED_HI],
                              "offset_xy": OFFSET_XY, "offset_z": [OFFSET_Z_LO, OFFSET_Z_HI],
                              "close_jitter": CLOSE_JITTER, "clean_frac": CLEAN_FRAC,
                              "noise": [NOISE_LO, NOISE_HI], "release_p": RELEASE_P,
                              "random_frac": RANDOM_FRAC, "dial_frac": DIAL_FRAC, "seed": SEED},
                "episodes": episodes}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    s = np.array([e["success"] for e in episodes])
    r = np.array([e["return"] for e in episodes])
    print(f"DONE {N} episodes in {(time.time() - t_start) / 60:.1f} min: success {s.mean():.3f}, "
          f"return {r.mean():.1f} (max {r.max():.1f}), clean demonstrations "
          f"{sum(e['noise'] == 0 for e in episodes)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
