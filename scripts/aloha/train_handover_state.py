"""State-only SAC on AlohaHandOver (MuJoCo Playground fork venv): does an RL agent trained from scratch on reward v2
learn the hand-over -- or find a loophole the success test misses? And does it learn more reliably than on the env's
own reward? The env is the privileged-state one (83-D observation, no cameras). Training pays reward v2 or the env's
own reward (v1); evaluation always reports the v2 success rate (``eval/episode_success``) and episode length.
Hyper-parameters default to the April run 65, the one seed (of six) that solved the env's own reward (return 6.09):
lr 5e-5, reward scaling 2, discount 0.97, replay 2M, 512 envs, 32 gradient updates per step, MLP 256 x 256.

    python train_handover_state.py <logdir> [num_timesteps] [seed] [v2|v1]
"""
import functools
import importlib.util
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp

if "device_put_replicated" not in vars(jax):
    # brax's trainers still call it; JAX 0.10 (pinned for the batch renderer) removed it. Drop-in for local devices.
    def _device_put_replicated(x, devices):
        sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, ("d",)), jax.sharding.PartitionSpec("d"))
        return jax.tree.map(lambda a: jax.device_put(jp.stack([jp.asarray(a)] * len(devices)), sharding), x)
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac_train
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aloha import handover

LOGDIR = pathlib.Path(sys.argv[1]).resolve()
STEPS = int(float(sys.argv[2])) if len(sys.argv) > 2 else 100_000_000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MODE = sys.argv[4] if len(sys.argv) > 4 else "v2"
spec = importlib.util.spec_from_file_location(
    "hv2", pathlib.Path(__file__).resolve().parents[2] / "dreamer4" / "envs" / "aloha_handover_v2.py")
hv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hv2)
# reward-v2 constants under test (TABLE=0.01 python ...): the step reward reads them when it is traced
for name in ("TABLE", "TIME", "BONUS"):
    if name in os.environ:
        setattr(hv2, name, float(os.environ[name]))
AUX = ("phi", "held", "left_held", "t", "ended")


class HandOverV2(handover.HandOver):
    """The env with the reward-v2 bookkeeping in ``info``; pays v2 (or, MODE v1, its own reward) and always reports
    the v2 success in ``metrics`` (brax's evaluator sums a metric over the episode)."""

    def reset(self, rng):
        s = super().reset(rng)
        aux = hv2.init_aux(self, s)
        s.info.update({f"v2_{k}": v for k, v in aux.items()})
        s.metrics.update(success=jp.zeros(()), phi=aux["phi"])
        return s

    def step(self, state, action):
        aux = {k: state.info[f"v2_{k}"] for k in AUX}
        nxt = super().step(state, action)
        reward, done, ok, aux = hv2.step_reward(self, nxt, aux)
        nxt.info.update({f"v2_{k}": v for k, v in aux.items()})
        nxt.metrics.update(success=ok.astype(jp.float32), phi=aux["phi"])
        if MODE == "v1":
            return nxt                         # the env's own reward and termination; success is only measured
        return nxt.replace(reward=reward, done=done.astype(jp.float32))


def make_env():
    # v1 keeps the env's own drop termination; v2 drops it (hv2.make_env's choice)
    over = {"impl": "warp"} if MODE == "v1" else {"impl": "warp", "no_drop_termination": True}
    return HandOverV2(config=handover.default_config(), config_overrides=over)


def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (LOGDIR / "checkpoints").mkdir(exist_ok=True)
    log = open(LOGDIR / "evals.jsonl", "a", buffering=1)
    t0 = time.time()

    def progress(num_steps, metrics):
        if "eval/episode_reward" not in metrics:
            return
        row = {"steps": int(num_steps), "minutes": round((time.time() - t0) / 60, 1),
               "reward": float(metrics["eval/episode_reward"]),
               "success": float(metrics.get("eval/episode_success", float("nan"))),
               "length": float(metrics.get("eval/avg_episode_length", float("nan")))}
        log.write(json.dumps(row) + "\n")
        print(f"[{row['minutes']:7.1f} min] {num_steps:>11d} steps | return {row['reward']:7.3f} | v2 success "
              f"{row['success']:.3f} | episode length {row['length']:.0f}", flush=True)

    env, eval_env = make_env(), make_env()
    train = functools.partial(
        sac_train.train,
        network_factory=functools.partial(sac_networks.make_sac_networks, hidden_layer_sizes=(256, 256)),
        num_timesteps=STEPS, episode_length=hv2.EPISODE_STEPS, action_repeat=1, num_envs=512, num_eval_envs=64,
        learning_rate=5e-5, discounting=0.97, seed=SEED, batch_size=512, num_evals=50,
        normalize_observations=True, reward_scaling=2.0, tau=0.005, min_replay_size=8192,
        max_replay_size=2_000_000, grad_updates_per_step=32, deterministic_eval=True,
        wrap_env_fn=wrapper.wrap_for_brax_training, progress_fn=progress, eval_env=eval_env,
        checkpoint_logdir=str(LOGDIR / "checkpoints"))
    print(f"SAC on AlohaHandOver, reward {MODE} (bonus {hv2.BONUS}, time {hv2.TIME}, table {hv2.TABLE}), seed {SEED}, "
          f"{STEPS} steps -> {LOGDIR}", flush=True)
    train(environment=env)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
