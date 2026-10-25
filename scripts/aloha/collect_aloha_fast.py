"""Collect FAST AlohaPick picks for the world model: a closed-loop scripted expert, in the aloha-npy layout.

The fork's expert (collect_aloha_npy.py) moves on to the next phase at fixed times scaled by a speed dial; its fast
settings close the fingers before the gripper is down at the box and almost always fail (speed 0.5-0.6: 1 % success,
no clean success above return 81). The world model trained on that data learns "fast = fail" and dreams the faster
grasps of a policy as failures. This expert moves on when the gripper is where it should be:
  0 to a pre-grasp point PRE above the box (until within TOL_A)  ->  1 down at the box (until within TOL_B)  ->
  2 close the fingers for N_CLOSE steps  ->  3 lift,
with a per-episode IK gain, pre-grasp height and tolerances, so the data covers fast AND slow successful picks and
the near misses between them. The joint-velocity limit, not the gain, sets the speed; a low pre-grasp point is the
short path (PRE 0.02-0.05: success 0.79, lift at step 41 median, 23 at best; PRE 0.08-0.10: 0.53, step 53).

Episodes are written with noise = 1.0, the convention that keeps them out of phase-2 cloning (the adapter clones
clean recordings only): they teach the world model, the copycat stays the one cloned from the original demos. The
proprio standardisation is copied from --base so a merged dataset reads one convention.

    BASE=<base dataset dir> N=1024 NWORLD=128 OUT=<new dir> python collect_aloha_fast.py
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

N = int(os.environ.get("N", 1024))
NWORLD = int(os.environ.get("NWORLD", 128))
OUT = pathlib.Path(os.environ["OUT"])
BASE = pathlib.Path(os.environ["BASE"])
SEED = int(os.environ.get("SEED", 1))
GAIN_LO, GAIN_HI = float(os.environ.get("GAIN_LO", 1.0)), float(os.environ.get("GAIN_HI", 4.0))
PRE_LO, PRE_HI = float(os.environ.get("PRE_LO", 0.01)), float(os.environ.get("PRE_HI", 0.06))   # pre-grasp height
TOL_A_LO, TOL_A_HI = float(os.environ.get("TOL_A_LO", 0.02)), float(os.environ.get("TOL_A_HI", 0.06))
TOL_B_LO, TOL_B_HI = float(os.environ.get("TOL_B_LO", 0.006)), float(os.environ.get("TOL_B_HI", 0.02))
CLOSE_LO, CLOSE_HI = int(os.environ.get("CLOSE_LO", 3)), int(os.environ.get("CLOSE_HI", 12))
CAM_RES_HW = (128, 128)
CAMERAS = ("view_0", "view_2")
ACTION_SCALE = 0.015
RIGHT_ARM_JOINTS = ("right/waist", "right/shoulder", "right/elbow",
                    "right/forearm_roll", "right/wrist_angle", "right/wrist_rotate")


def make_rollout_fn(env, nworld):
    mj_model, mjx_model = env.mj_model, env.mjx_model
    site = mj_model.site("right/gripper").id
    body = int(mj_model.site_bodyid[site])
    dofs = jp.array([mj_model.jnt_dofadr[mj_model.joint(j).id] for j in RIGHT_ARM_JOINTS])
    box_qadr = int(env._box_qadr)
    ep_len = int(env._config.episode_length)

    def policy_one(state, box0, grasp, phase, count, gain, pre, tol_a, tol_b, n_close):
        above, down = box0 + jp.array([0.0, 0.0, 1.0]) * pre, box0 + jp.array([0.0, 0.0, 0.005])
        eef = state.data.site_xpos[site]
        box_now = state.data.qpos[box_qadr:box_qadr + 3]
        # Closed loop: a phase ends when the gripper is where it should be, not at a fixed time.
        nxt = jp.where((phase == 0) & (jp.linalg.norm(eef - above) < tol_a), 1, phase)
        nxt = jp.where((nxt == 1) & (jp.linalg.norm(eef - down) < tol_b), 2, nxt)
        grasp = jp.where((phase < 2) & (nxt == 2), box_now, grasp)          # where the box was when closing began
        count = jp.where(nxt == 2, count + 1, 0)
        nxt = jp.where((nxt == 2) & (count >= n_close), 3, nxt)
        target = jp.where(nxt == 0, above, jp.where(nxt < 3, down, grasp + jp.array([0.0, 0.0, 0.10])))
        jacp, _ = mjx.jac(mjx_model, state.data, eef, body)
        J = jacp.T[:, dofs]
        dq = J.T @ jp.linalg.solve(J @ J.T + 1e-3 * jp.eye(3), gain * (target - eef))
        action = jp.zeros(14, dtype=jp.float32).at[7:13].set(jp.clip(dq / ACTION_SCALE, -1.0, 1.0))
        finger = jp.where(nxt < 2, 0.04, 0.0)
        action = action.at[13].set(jp.clip((finger - state.data.ctrl[13]) / ACTION_SCALE, -1.0, 1.0))
        return action, grasp, nxt, count

    policy = jax.vmap(policy_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)
    kin = jax.vmap(lambda d: mjx.kinematics(mjx_model, d))
    u8 = lambda x: (jp.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(jp.uint8)

    def rollout(rngs, gain, pre, tol_a, tol_b, n_close):
        state = reset(rngs)
        state = state.replace(data=kin(state.data))
        box0 = state.data.qpos[:, box_qadr:box_qadr + 3]

        def body_fn(carry, t):
            state, grasp, phase, count = carry
            action, grasp, phase, count = policy(state, box0, grasp, phase, count, gain, pre, tol_a, tol_b,
                                                 n_close)
            nxt = step(state, action)
            out = {"state": state.obs["state"], "view_0": u8(state.obs["pixels/view_0"]),
                   "view_2": u8(state.obs["pixels/view_2"]), "action": action, "reward": nxt.reward,
                   "box_z": nxt.data.qpos[:, box_qadr + 2],
                   "grasping": nxt.info["grasping"].astype(jp.float32), "phase": phase}
            return (nxt, grasp, phase, count), out

        zeros = jp.zeros(nworld, jp.int32)
        (final, _, _, _), traj = jax.lax.scan(body_fn, (state, jp.zeros((nworld, 3)), zeros, zeros),
                                              jp.arange(ep_len))
        final_obs = {"state": final.obs["state"], "view_0": u8(final.obs["pixels/view_0"]),
                     "view_2": u8(final.obs["pixels/view_2"])}
        return traj, final_obs

    return jax.jit(rollout), ep_len


def main():
    env = manipulation.load("AlohaPick", config_overrides={
        "vision": True, "vision_config.nworld": NWORLD, "vision_config.cam_res": CAM_RES_HW})
    rollout, T = make_rollout_fn(env, NWORLD)
    base = json.loads((BASE / "manifest.json").read_text())
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
        gain = rng.uniform(GAIN_LO, GAIN_HI, NWORLD).astype(np.float32)
        pre = rng.uniform(PRE_LO, PRE_HI, NWORLD).astype(np.float32)
        tol_a = rng.uniform(TOL_A_LO, TOL_A_HI, NWORLD).astype(np.float32)
        tol_b = rng.uniform(TOL_B_LO, TOL_B_HI, NWORLD).astype(np.float32)
        n_close = rng.integers(CLOSE_LO, CLOSE_HI + 1, NWORLD).astype(np.int32)
        keys = jax.random.split(jax.random.PRNGKey(SEED * 100003 + b), NWORLD)
        traj, final = rollout(keys, *(jp.asarray(x) for x in (gain, pre, tol_a, tol_b, n_close)))
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
        lift_t = np.where((traj["phase"] == 3).any(1), (traj["phase"] == 3).argmax(1), -1)
        for i in range(m):
            episodes.append({"source": "fast_expert", "gain": round(float(gain[i]), 3),
                             "pre": round(float(pre[i]), 3),
                             "tol_a": round(float(tol_a[i]), 4), "tol_b": round(float(tol_b[i]), 4),
                             "n_close": int(n_close[i]), "lift_step": int(lift_t[i]), "speed": 1.0, "noise": 1.0,
                             "success": bool(success[i]), "return": round(float(ret[i]), 2)})
        n += m
        b += 1
        print(f"batch {b}: {m} episodes, success {success.mean():.2f}, return {ret.mean():.1f} "
              f"(successes {ret[success].mean() if success.any() else float('nan'):.1f}, "
              f"max {ret.max():.1f}) | {n}/{N} | {time.time() - t0:.1f} s", flush=True)
    for a in arr.values():
        a.flush()
    manifest = {k: base[k] for k in ("format", "cameras", "downscale", "env", "state_mean", "state_std")}
    manifest.update({"n_episodes": N, "episodes": episodes,
                     "collector": {"expert": "closed-loop fast", "gain": [GAIN_LO, GAIN_HI],
                                   "pre": [PRE_LO, PRE_HI],
                                   "tol_a": [TOL_A_LO, TOL_A_HI], "tol_b": [TOL_B_LO, TOL_B_HI],
                                   "n_close": [CLOSE_LO, CLOSE_HI], "seed": SEED,
                                   "proprio_stats_from": str(BASE)}})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    s = np.array([e["success"] for e in episodes])
    r = np.array([e["return"] for e in episodes])
    print(f"DONE {N} episodes in {(time.time() - t_start) / 60:.1f} min: success {s.mean():.3f}, "
          f"return {r.mean():.1f} (successes {r[s].mean() if s.any() else float('nan'):.1f}, max {r.max():.1f})",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
