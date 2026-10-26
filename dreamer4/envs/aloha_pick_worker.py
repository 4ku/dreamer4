"""Worker process: N AlohaPick worlds (MuJoCo Playground, vision) stepped in lockstep.

Runs in the MuJoCo Playground environment (JAX + MJX + the batch renderer), NOT in the training
environment: :class:`dreamer4.envs.aloha_pick.AlohaPickBatch` starts it as a subprocess and talks to
it over stdin / stdout with pickled messages. Standalone on purpose -- it imports nothing from
dreamer4, so the two Python environments never have to coexist in one process.

Protocol (one pickled object per message, length-prefixed):
    ("reset", seeds (N,) int)        -> {"state": (N,14) f32, "<camera>": (N,H,W,3) u8 ...}
    ("step", actions (N,14) f32)     -> same obs dict + "reward" (N,) f32, "done" (N,) bool,
                                        "success" (N,) bool
    ("close",)                       -> exits
"""
import os
import pickle
import struct
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")


def read_msg(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    if n > 1 << 31:
        raise RuntimeError(f"AlohaPick worker protocol out of sync (message length {n}): something "
                           f"other than the protocol wrote to the stream")
    return pickle.loads(stream.read(n))


def write_msg(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def main():
    # The protocol owns the real stdout. Keep a private copy of it and point fd 1 at stderr BEFORE any
    # library loads, so nothing they print -- from Python or from C (XLA, warp) -- lands in the message
    # stream; a stray line there once desynchronised the parent mid-evaluation.
    outp = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    nworld, res, cameras = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(",")
    lift_target = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
    reward_mode = sys.argv[5] if len(sys.argv) > 5 else "v1"
    import jax
    import jax.numpy as jp
    import numpy as np
    from mujoco_playground import manipulation

    env = manipulation.load("AlohaPick", config_overrides={
        "vision": True, "vision_config.nworld": nworld, "vision_config.cam_res": (res, res),
        "box_z_lift_target": lift_target})
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    if reward_mode == "v2":
        import aloha_reward_v2 as v2          # this file's directory is on sys.path (it runs as a script)
        init_aux = jax.jit(jax.vmap(lambda s: v2.init_aux(env, s)))
        v2_step = jax.jit(jax.vmap(lambda s, a: v2.step_reward(env, s, a)))
    aux = None
    u8 = lambda x: np.asarray((jp.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(jp.uint8))

    def obs_dict(state):
        out = {"state": np.asarray(state.obs["state"], np.float32)}
        for c in cameras:
            out[c] = u8(state.obs[f"pixels/{c}"])
        return out

    inp = sys.stdin.buffer
    state = None
    while True:
        msg = read_msg(inp)
        if msg is None or msg[0] == "close":
            return
        if msg[0] == "reset":
            keys = jp.stack([jax.random.PRNGKey(int(s)) for s in msg[1]])
            state = reset(keys)
            if reward_mode == "v2":
                aux = init_aux(state)
            write_msg(outp, obs_dict(state))
        elif msg[0] == "step":
            state = step(state, jp.asarray(msg[1], jp.float32))
            out = obs_dict(state)
            if reward_mode == "v2":
                r, done, ok, aux = v2_step(state, aux)
                out["reward"] = np.asarray(r, np.float32)
                out["done"] = np.asarray(done, bool)
                out["success"] = np.asarray(ok, bool)
            else:
                out["reward"] = np.asarray(state.reward, np.float32)
                out["done"] = np.asarray(state.done, bool)
                out["success"] = np.asarray(state.metrics["success"]) > 0.5
            write_msg(outp, out)


if __name__ == "__main__":
    main()
