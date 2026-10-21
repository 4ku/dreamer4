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
    return pickle.loads(stream.read(n))


def write_msg(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def main():
    nworld, res, cameras = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(",")
    import jax
    import jax.numpy as jp
    import numpy as np
    from mujoco_playground import manipulation

    env = manipulation.load("AlohaPick", config_overrides={
        "vision": True, "vision_config.nworld": nworld, "vision_config.cam_res": (res, res)})
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    u8 = lambda x: np.asarray((jp.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(jp.uint8))

    def obs_dict(state):
        out = {"state": np.asarray(state.obs["state"], np.float32)}
        for c in cameras:
            out[c] = u8(state.obs[f"pixels/{c}"])
        return out

    inp, outp = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                     # nothing else may write to the protocol stream
    state = None
    while True:
        msg = read_msg(inp)
        if msg is None or msg[0] == "close":
            return
        if msg[0] == "reset":
            keys = jp.stack([jax.random.PRNGKey(int(s)) for s in msg[1]])
            state = reset(keys)
            write_msg(outp, obs_dict(state))
        elif msg[0] == "step":
            state = step(state, jp.asarray(msg[1], jp.float32))
            out = obs_dict(state)
            out["reward"] = np.asarray(state.reward, np.float32)
            out["done"] = np.asarray(state.done, bool)
            out["success"] = np.asarray(state.metrics["success"]) > 0.5
            write_msg(outp, out)


if __name__ == "__main__":
    main()
