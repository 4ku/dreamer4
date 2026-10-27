"""AlohaHandOver for Dreamer 4: the env as the agent sees it, and reward v2 (JAX, MuJoCo Playground side).

MuJoCo Playground's AlohaHandOver: the left arm picks the box up on the left half of the table, hands it over at the
midline, and the right arm carries it to a random target and holds it there; 250 steps of 0.02 s, 14-D delta-joint
actions on both arms. Its own reward pays only the best progress so far (a ratchet: mistakes are free), adds a small
bonus every step the end state is held, and the env has no success flag at all. Two changes to the env:

* the target marker is a box of the box's own size, drawn in the box's own red; the batch renderer ignores its alpha,
  so a pixel policy would see two identical red boxes. It is recoloured green;
* dropping the box no longer ends the episode (``no_drop_termination``): only success, leaving the bounds or a NaN
  do, so an early end can never be chosen to stop paying the time cost.

Reward v2, as AlohaPick's (:mod:`aloha_reward_v2`):

    success  = a real hand-over, held for HOLD consecutive steps: the LEFT gripper held the box off the table earlier
               in the episode, the box is within GOAL of the target, the RIGHT gripper is on it (GRIP) and closed
               (CLOSED), the left gripper is away from it (AWAY) and back on its own half (x < 0)
               -> + BONUS, and the episode ends
    progress = phi(s') - phi(s), phi = (gripper_box + 4 box_handover + 8 handover_target) / 22: the env's own terms
               and weights, scaled to [0, 1]; potential-based, so nothing is earned by holding or by going back and
               forth
    time     = - TIME per step;  contact = - TABLE per step with a finger on the table
    leaving the bounds or a NaN ends the episode and charges the time it skipped

Checked on the April state-only SAC expert (256 episodes): success 0.977, at step 88 (median); exactly the episodes
the old "return >= 4 and never terminated" rule kept; every success preceded by a left-hand lift (step 32).
"""
import os

import jax.numpy as jp
from mujoco import mjx

BONUS, HOLD = 1.0, 5
# HANDOVER_TIME / HANDOVER_TABLE re-price the two costs (a whole pipeline sets them once: the collectors, the eval /
# collection env worker and the trainers all read this module)
TIME = float(os.environ.get("HANDOVER_TIME", 0.01))
TABLE = float(os.environ.get("HANDOVER_TABLE", 0.1))
GOAL, GRIP, CLOSED, AWAY, LIFTED = 0.05, 0.05, 0.028, 0.08, 0.05
TERMINAL_REWARD = 0.5
TARGET_RGBA = (0.1, 0.8, 0.1, 1.0)
EPISODE_STEPS = 250


def make_env(config_overrides=None):
    """MuJoCo Playground's AlohaHandOver with the green target and no drop termination."""
    from mujoco_playground._src.manipulation.aloha import handover

    class HandOverGreenTarget(handover.HandOver):
        def _init_vision(self):
            # before the render context is built from the model's colours
            geom = self._mj_model.body_geomadr[self._mj_model.body("mocap_target").id]
            self._mj_model.geom_rgba[geom] = TARGET_RGBA
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
            super()._init_vision()

    return HandOverGreenTarget(config=handover.default_config(),
                               config_overrides={"no_drop_termination": True, **(config_overrides or {})})


def proprio(env, data):
    """The 14-D joint state, laid out as the action vector (as AlohaPick's): left arm 6, left gripper gap, right arm
    6, right gripper gap. The env's own observation (83-D, with the box and the target) is privileged."""
    arm = data.qpos[env._arm_qadr]
    f = data.qpos[env._finger_qposadr]
    return jp.concatenate([arm[:6], (f[0] + f[1])[None], arm[6:], (f[2] + f[3])[None]])


def potential(env, data, info):
    """phi in [0, 1] and the table-contact flag of one world (the env's reward terms)."""
    raw = env._get_reward(data, dict(info))
    phi = (raw["gripper_box"] + 4.0 * raw["box_handover"] + 8.0 * raw["handover_target"]) / 22.0
    return phi, 1.0 - raw["no_table_collision"]


def hand_flags(env, data, info):
    """(left gripper holds the box off the table, the success pose) of one world."""
    box = data.xpos[env._box_body]
    lg, rg = data.site_xpos[env._left_gripper_site], data.site_xpos[env._right_gripper_site]
    f = data.qpos[env._finger_qposadr]
    d_lg, d_rg = jp.linalg.norm(lg - box), jp.linalg.norm(rg - box)
    left_hold = (d_lg < GRIP) & (0.5 * (f[0] + f[1]) < CLOSED) & (box[2] > LIFTED)
    at_goal = ((jp.linalg.norm(box - info["target_pos"]) < GOAL) & (d_rg < GRIP) & (0.5 * (f[2] + f[3]) < CLOSED)
               & (d_lg > AWAY) & (lg[0] < 0.0))
    return left_hold, at_goal


def init_aux(env, state):
    """Per-world bookkeeping right after a reset."""
    phi, _ = potential(env, state.data, state.info)
    return {"phi": phi, "held": jp.zeros((), jp.int32), "left_held": jp.zeros((), bool),
            "t": jp.zeros((), jp.int32), "ended": jp.zeros((), bool)}


def step_reward(env, next_state, aux):
    """v2 reward, done and success for the transition into ``next_state`` (one world). After the episode has ended
    every later step pays 0 and reports done (lockstep batches keep stepping the physics)."""
    phi, contact = potential(env, next_state.data, next_state.info)
    left_hold, at_goal = hand_flags(env, next_state.data, next_state.info)
    left_held = aux["left_held"] | left_hold
    held = jp.where(at_goal & left_held, aux["held"] + 1, 0)
    success = held >= HOLD
    t = aux["t"] + 1
    quit_early = (next_state.done > 0.5) & ~success
    reward = ((phi - aux["phi"]) - TIME - TABLE * contact + BONUS * success
              - TIME * (EPISODE_STEPS - t) * quit_early)
    done = success | (next_state.done > 0.5)
    reward = jp.where(aux["ended"], 0.0, reward)
    success = success & ~aux["ended"]
    new = {"phi": phi, "held": held, "left_held": left_held, "t": t, "ended": aux["ended"] | done}
    return reward, aux["ended"] | done, success, new
