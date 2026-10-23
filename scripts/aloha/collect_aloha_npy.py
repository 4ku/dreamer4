"""Collect AlohaPick episodes for the Dreamer4 world model, written straight into the aloha-npy layout.

The fork's scripted IK expert (scripts/collect_aloha_pick_expert.py: approach above the box, descend, close,
lift), batched over NWORLD worlds in one JIT'd scan, with two per-episode dials so the data covers more
than one way to pick:
  speed  -- every phase boundary scaled by s ~ U(SPEED_LO, SPEED_HI) (s < 1: an earlier lift, a higher return);
  noise  -- Gaussian noise of std sigma added to the right-arm actions; a share CLEAN_FRAC of the
            episodes is noise-free, the rest draws sigma ~ U(NOISE_LO, NOISE_HI).
Every episode is kept (failures teach the world model what a bad action does); the manifest records
speed, noise, success and return per episode, and the adapter clones only clean successes.

    N=2000 NWORLD=128 OUT=/mnt/dsk1/iefremov/aloha/data/aloha_mix_2000 python collect_aloha_npy.py

Layout (frames T+1 = 121 including the final one, actions/rewards T = 120):
    state.npy (N, T+1, 14) f32, action.npy (N, T, 14) f32, reward.npy (N, T) f32,
    pixels_view_0.npy / pixels_view_2.npy (N, T+1, 128, 128, 3) u8, manifest.json
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
SEED = int(os.environ.get("SEED", 0))
SPEED_LO, SPEED_HI = float(os.environ.get("SPEED_LO", 0.5)), float(os.environ.get("SPEED_HI", 1.2))
CLEAN_FRAC = float(os.environ.get("CLEAN_FRAC", 0.5))
NOISE_LO, NOISE_HI = float(os.environ.get("NOISE_LO", 0.05)), float(os.environ.get("NOISE_HI", 0.5))
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

    def policy_one(state, box0, grasp_pos, t, pa, pb, pc):
        target = jp.where(t < pa, box0 + jp.array([0.0, 0.0, 0.10]),
                          jp.where(t < pc, box0 + jp.array([0.0, 0.0, 0.005]),
                                   grasp_pos + jp.array([0.0, 0.0, 0.10])))
        eef = state.data.site_xpos[site]
        jacp, _ = mjx.jac(mjx_model, state.data, eef, body)
        J = jacp.T[:, dofs]
        dq = J.T @ jp.linalg.solve(J @ J.T + 1e-3 * jp.eye(3), 0.5 * (target - eef))
        action = jp.zeros(14, dtype=jp.float32).at[7:13].set(jp.clip(dq / ACTION_SCALE, -1.0, 1.0))
        finger = jp.where(t < pb, 0.04, 0.0)
        action = action.at[13].set(jp.clip((finger - state.data.ctrl[13]) / ACTION_SCALE, -1.0, 1.0))
        box_now = state.data.qpos[box_qadr:box_qadr + 3]
        return action, jp.where(t == pc - 1, box_now, grasp_pos)

    policy = jax.vmap(policy_one, in_axes=(0, 0, 0, None, 0, 0, 0))
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)
    kin = jax.vmap(lambda d: mjx.kinematics(mjx_model, d))
    u8 = lambda x: (jp.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(jp.uint8)

    def rollout(rngs, speed, sigma, noise_key):
        state = reset(rngs)
        state = state.replace(data=kin(state.data))
        box0 = state.data.qpos[:, box_qadr:box_qadr + 3]
        pa = jp.round(PHASE_A * speed).astype(jp.int32)
        pb = jp.round(PHASE_B * speed).astype(jp.int32)
        pc = jp.round(PHASE_C * speed).astype(jp.int32)
        noise = jax.random.normal(noise_key, (ep_len, nworld, 14)) * sigma[None, :, None]
        noise = noise.at[:, :, :7].set(0.0)                       # the left arm stays frozen

        def body_fn(carry, xs):
            state, grasp = carry
            t, eps = xs
            action, grasp = policy(state, box0, grasp, t, pa, pb, pc)
            action = jp.clip(action + eps, -1.0, 1.0)
            nxt = step(state, action)
            out = {"state": state.obs["state"], "view_0": u8(state.obs["pixels/view_0"]),
                   "view_2": u8(state.obs["pixels/view_2"]), "action": action, "reward": nxt.reward,
                   "box_z": nxt.data.qpos[:, box_qadr + 2],
                   "grasping": nxt.info["grasping"].astype(jp.float32)}
            return (nxt, grasp), out

        (final, _), traj = jax.lax.scan(body_fn, (state, jp.zeros((nworld, 3))), (jp.arange(ep_len), noise))
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
        speed = rng.uniform(SPEED_LO, SPEED_HI, NWORLD).astype(np.float32)
        sigma = np.where(rng.random(NWORLD) < CLEAN_FRAC, 0.0,
                         rng.uniform(NOISE_LO, NOISE_HI, NWORLD)).astype(np.float32)
        keys = jax.random.split(jax.random.PRNGKey(SEED * 100003 + b), NWORLD + 1)
        traj, final = rollout(keys[:NWORLD], jp.asarray(speed), jp.asarray(sigma), keys[-1])
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
            episodes.append({"speed": round(float(speed[i]), 3), "noise": round(float(sigma[i]), 3),
                             "success": bool(success[i]), "return": round(float(ret[i]), 2)})
        n += m
        b += 1
        print(f"batch {b}: {m} episodes, success {success.mean():.2f}, return {ret.mean():.1f} "
              f"(clean {ret[sigma[:m] == 0].mean() if (sigma[:m] == 0).any() else float('nan'):.1f}) | "
              f"{n}/{N} | {time.time() - t0:.1f} s", flush=True)
    for a in arr.values():
        a.flush()
    flat = np.asarray(arr["state"], np.float64).reshape(-1, 14)
    manifest = {"format": "aloha-npy", "cameras": list(CAMERAS), "downscale": 1, "n_episodes": N,
                "state_mean": flat.mean(0).round(6).tolist(), "state_std": flat.std(0).round(6).tolist(),
                "env": {"cam_res": list(CAM_RES_HW), "max_steps": T},
                "collector": {"speed": [SPEED_LO, SPEED_HI], "clean_frac": CLEAN_FRAC,
                              "noise": [NOISE_LO, NOISE_HI], "seed": SEED},
                "episodes": episodes}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    s = np.array([e["success"] for e in episodes])
    r = np.array([e["return"] for e in episodes])
    print(f"DONE {N} episodes in {(time.time() - t_start) / 60:.1f} min: success {s.mean():.3f}, "
          f"return {r.mean():.1f} (max {r.max():.1f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
