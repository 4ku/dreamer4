"""Roll a trained latent policy out in a REAL environment and score it.

Phases 2 and 3 both select their checkpoints with :func:`evaluate`, and a dream never scores
itself: offline metrics -- clone accuracy, reward MAE -- only say the model fits the data, not that
the policy reaches goals.

Nothing here is tied to one environment. Any Gymnasium env works as long as its observation is an
image the tokenizer can encode. Everything domain-specific -- which env to open, how to read its
proprio -- comes from the dataset the agent was trained on, through the ``env_spec`` /
``proprio_from_info`` hooks of :class:`dreamer4.data.base.EpisodeVideoDataset`.

One step of the loop: the env hands over a frame; the frozen tokenizer encodes it with its sliding
history window (exactly the encoding training used); proprio is read from the env's own ``info`` --
the exact state, never inferred from the render; the policy turns the latent into an action; the
env steps. Nothing is decoded: every frame here is real.

The policy is :class:`dreamer4.models.world_model.AgentPolicy`: heads on agent tokens inside the
world model, so its action depends on the episode's context window and it has to be told when an
episode restarts (``reset_rows``). That way the context a policy has here is the context it had in
training -- the thing an offline metric cannot check and a stale window silently breaks.

Optional ``info`` keys, both with a fallback so a plain Gymnasium env works unchanged:
  ``is_success``                episode outcome; falls back to ``terminated``;
  ``optimal_steps_from_start``  length of the shortest solution from the reset state; when the env
                                does not publish it, the optimality ratio is reported as NaN.

The one policy-space assumption is discrete actions: the head emits logits over ``n_actions``.
"""
import numpy as np
import torch


def make_env(spec):
    """Open the environment an ``env_spec`` names.

    Args:
        spec: ``{"id": ..., "kwargs": {...}}``, as produced by
            :meth:`EpisodeVideoDataset.env_spec`.

    Returns:
        The env built by ``gymnasium.make``, so any registered env works.
    """
    if not spec or not spec.get("id"):
        raise ValueError("no env_spec: this dataset has no live environment to evaluate in")
    import gymnasium as gym
    return gym.make(spec["id"], **spec.get("kwargs", {}))


@torch.no_grad()
def evaluate(net, envs, *, tok, proprio_from_info, n, seed, greedy, device):
    """Play ``n`` episodes across ``envs`` in parallel (one batched policy forward per step).

    Episodes are seeded consecutively from ``seed``, so two checkpoints scored with the same seed
    meet the same layouts, and a held-out seed keeps the evaluation off the training distribution.

    Args:
        net:               the policy being scored: an
                           :class:`~dreamer4.models.world_model.AgentPolicy`, which exposes
                           ``act(z, proprio, greedy=..., prev_action=...)`` and
                           ``reset_rows(rows)``.
        envs:              already-built envs, stepped in lockstep; more envs, fewer forwards.
        tok:               frozen tokenizer, used only to encode the frames the envs return.
        proprio_from_info: turns an env ``info`` dict into the proprio vector the policy reads.
        n:                 episodes to finish before returning.
        seed:              seed of the first episode; the next ones take ``seed + 1``, ``+ 2``, ...
        greedy:            argmax the policy logits instead of sampling them.
        device:            device the policy runs on.

    Returns:
        ``(success_rate, mean_steps, steps_over_optimal, mean_return)``. The third is the metric to read: mean
        episode length is not comparable across policies, because a policy that fails more has its
        failures capped at the env's step limit while a policy that succeeds more pays the real
        path length -- the two move in opposite directions and the mean hides both. Optimality is
        steps / the shortest solution, over SOLVED episodes only, taken from the env's own
        ``optimal_steps_from_start``; 1.000 is perfect, and NaN when no solved episode carried one.
        The mean return is the env's own episode return -- what separates policies that all
        succeed (e.g. how long a picked object is held, on a dense-reward arm task).
    """
    was_training = net.training
    # Eval mode first: anything the network does differently while training would silently score
    # a different network than the one being shipped. The training mode is restored on return.
    net.eval()
    B = len(envs)
    hist = tok.history
    succ, steps, ratios, returns, done_n, nxt = [], [], [], [], 0, 0
    obs_l, info_l, ln, ret = [None] * B, [None] * B, [0] * B, [0.0] * B
    opt = [0] * B
    frames = [None] * B
    # Exactly the episodes seed .. seed+n-1 are scored, every one of them to its end. Stopping at
    # the n-th FINISHED episode instead leaves up to len(envs) episodes in flight, and those are
    # mostly the long ones -- a failure runs to the step limit -- so the success rate comes out
    # too high, by a lot when n is only a few times len(envs).
    active = [False] * B
    for i in range(B):
        obs_l[i], info_l[i] = envs[i].reset(seed=seed + min(nxt, n - 1))
        active[i] = nxt < n
        nxt += 1
        opt[i] = int(info_l[i].get("optimal_steps_from_start", 0))
        frames[i] = [obs_l[i]] * hist
    net.reset_rows(None)
    torch.manual_seed(seed)
    prev_a = None
    while done_n < n:
        vid = np.stack([np.stack(f[-hist:]) for f in frames])            # (B, hist, H, W, C)
        z = tok.encode_frames(vid)[:, -1].float().to(device)             # (B, N, D)
        pr = torch.as_tensor(np.stack([proprio_from_info(i) for i in info_l]),
                             dtype=torch.float32, device=device)
        a = net.act(z, pr, greedy=greedy, prev_action=prev_a)
        prev_a = a
        env_a = net.env_action(a)
        restarted = []
        for i in range(B):
            if not active[i]:
                continue                                  # an idle row: no seed left for it
            obs_l[i], r, te, tr, info_l[i] = envs[i].step(env_a[i])
            frames[i].append(obs_l[i])
            ln[i] += 1
            ret[i] += float(r)
            if te or tr:
                ok = bool(info_l[i].get("is_success", te))
                succ.append(ok)
                steps.append(ln[i])
                returns.append(ret[i])
                if ok and opt[i] > 0:
                    ratios.append(ln[i] / opt[i])
                ln[i] = 0
                ret[i] = 0.0
                done_n += 1
                if nxt < n:
                    obs_l[i], info_l[i] = envs[i].reset(seed=seed + nxt)
                    nxt += 1
                    opt[i] = int(info_l[i].get("optimal_steps_from_start", 0))
                    frames[i] = [obs_l[i]] * hist
                    restarted.append(i)
                else:
                    active[i] = False
        if restarted:
            # The rows that just started a new episode must not inherit the context of the old one.
            net.reset_rows(restarted)
    net.train(was_training)
    return (float(np.mean(succ)), float(np.mean(steps)),
            float(np.mean(ratios)) if ratios else float("nan"), float(np.mean(returns)))


def main(argv=None):
    """Score a phase-2 or phase-3 checkpoint, greedy and sampled.

        python -m dreamer4.agent_eval --ckpt runs/x/rl/checkpoints/best.pt \\
            --dyn runs/x/dyn/checkpoints/latest.pt --n 2000 --seed 900000 --json runs/x/heldout.json

    The policy reads the transformer weights stored in the agent checkpoint (phase 2 finetunes
    them) on top of the world model's tokenizer and config. Seeds 900000+ are layouts no phase
    has seen.
    """
    import argparse
    import json
    import time
    from pathlib import Path

    from dreamer4.models.agent_heads import build_agent_heads
    from dreamer4.models.world_model import AgentPolicy, WorldModel

    ap = argparse.ArgumentParser(description="Score an agent checkpoint in the real environment.")
    ap.add_argument("--ckpt", required=True, help="phase-2 or phase-3 checkpoint")
    ap.add_argument("--dyn", required=True, help="the phase-1b world model it was trained on")
    ap.add_argument("--n", type=int, default=2000, help="episodes (the first n seeds from --seed)")
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--envs", type=int, default=64, help="environments played in parallel")
    ap.add_argument("--json", default="", help="also write the result here")
    cfg = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    world = WorldModel(cfg.dyn, device=dev, n_agent=int(ckpt["n_agent"]))
    world.load_dynamics_weights(ckpt["dyn"])
    heads = build_agent_heads(ckpt["agent_cfg"]).to(dev)
    heads.load_state_dict(ckpt["agent"])
    policy = AgentPolicy(world.dyn.eval(), heads.eval(), window=int(ckpt["clip_T"]),
                         tau_ctx=world.tau_ctx, k_max=world.k_max).eval()
    envs = [make_env(world.env_spec) for _ in range(cfg.envs)]
    kw = dict(tok=world.tok, proprio_from_info=world.proprio_from_info, n=cfg.n, seed=cfg.seed,
              device=dev)
    greedy = evaluate(policy, envs, greedy=True, **kw)
    sampled = evaluate(policy, envs, greedy=False, **kw)
    res = {"ckpt": cfg.ckpt, "n": cfg.n, "seed": cfg.seed, "train_step": ckpt.get("step"),
           "greedy_sr": greedy[0], "greedy_opt": greedy[2], "greedy_return": greedy[3],
           "sampled_sr": sampled[0], "sampled_opt": sampled[2], "sampled_return": sampled[3],
           "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    if cfg.json:
        Path(cfg.json).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
