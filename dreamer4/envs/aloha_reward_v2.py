"""AlohaPick reward v2: success ends the episode; progress, not holding, is paid (JAX, MuJoCo Playground side).

The env's own reward (v1) pays ~1.09 per step while the box is held up and stops growing at 10 cm, and success is
checked only at the last of 120 steps -- so a return-maximising policy lifts early, hovers just under the 10 cm
line and fails (2026-10-26: at loop iteration 8, 15 % of the episodes ended with the box held at ~9-10 cm). v2:

    success  = box above 10 cm and grasped for HOLD consecutive steps -> + BONUS, the episode ends
    progress = phi(s') - phi(s), phi = 0.1 approach + 0.2 grasp + 0.7 grasp * lift_frac (the v1 terms, in [0, 1])
    time     = - TIME per step
    contact  = - TABLE per step with a finger on the table

The progress term is potential-based: over an episode it sums to phi(end) - phi(start) <= 1 whatever the timing,
so nothing is earned by holding or hovering, and the time cost makes an earlier success strictly better. A dropped
box or the time limit ends the episode without the bonus. BONUS stays within the reward head's two-hot range
(+-1.72 with the default bins), and a step paying >= TERMINAL_REWARD is the success step.
"""
import jax.numpy as jp

BONUS, TIME, TABLE, HOLD, LINE = 1.0, 0.01, 0.1, 5, 0.10
TERMINAL_REWARD = 0.5


def potential(env, data, info):
    """phi in [0, 1], the table-contact flag, and the grasp flag of one world (the env's v1 reward terms)."""
    raw = env._get_reward(data, dict(info))
    phi = 0.1 * raw["approach"] + 0.2 * raw["grasp"] + 0.7 * raw["lift"]
    return phi, 1.0 - raw["no_table_collision"], raw["grasp"]


def init_aux(env, state):
    """Per-world bookkeeping right after a reset."""
    phi, _, _ = potential(env, state.data, state.info)
    return {"phi": phi, "held": jp.zeros((), jp.int32), "ended": jp.zeros((), bool)}


def step_reward(env, next_state, aux):
    """v2 reward, done and success for the transition into ``next_state`` (one world). After the episode has ended
    every later step pays 0 and reports done (lockstep batches keep stepping the physics)."""
    phi, contact, grasp = potential(env, next_state.data, next_state.info)
    z = next_state.data.xpos[env._box_body][2]
    held = jp.where((z > LINE) & (grasp > 0.5), aux["held"] + 1, 0)
    success = held >= HOLD
    reward = (phi - aux["phi"]) - TIME - TABLE * contact + BONUS * success
    done = success | (next_state.done > 0.5)
    reward = jp.where(aux["ended"], 0.0, reward)
    success = success & ~aux["ended"]
    new = {"phi": phi, "held": held, "ended": aux["ended"] | done}
    return reward, aux["ended"] | done, success, new
