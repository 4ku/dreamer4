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
        ``(success_rate, mean_steps, steps_over_optimal)``. The third is the metric to read: mean
        episode length is not comparable across policies, because a policy that fails more has its
        failures capped at the env's step limit while a policy that succeeds more pays the real
        path length -- the two move in opposite directions and the mean hides both. Optimality is
        steps / the shortest solution, over SOLVED episodes only, taken from the env's own
        ``optimal_steps_from_start``; 1.000 is perfect, and NaN when no solved episode carried one.
    """
    was_training = net.training
    # Eval mode first: anything the network does differently while training would silently score
    # a different network than the one being shipped. The training mode is restored on return.
    net.eval()
    B = len(envs)
    hist = tok.history
    succ, steps, ratios, done_n, nxt = [], [], [], 0, 0
    obs_l, info_l, ln = [None] * B, [None] * B, [0] * B
    opt = [0] * B
    frames = [None] * B
    for i in range(B):
        obs_l[i], info_l[i] = envs[i].reset(seed=seed + nxt)
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
        restarted = []
        for i in range(B):
            obs_l[i], _, te, tr, info_l[i] = envs[i].step(int(a[i]))
            frames[i].append(obs_l[i])
            ln[i] += 1
            if te or tr:
                ok = bool(info_l[i].get("is_success", te))
                succ.append(ok)
                steps.append(ln[i])
                if ok and opt[i] > 0:
                    ratios.append(ln[i] / opt[i])
                ln[i] = 0
                done_n += 1
                obs_l[i], info_l[i] = envs[i].reset(seed=seed + nxt)
                nxt += 1
                opt[i] = int(info_l[i].get("optimal_steps_from_start", 0))
                frames[i] = [obs_l[i]] * hist
                restarted.append(i)
        if restarted:
            # The rows that just started a new episode must not inherit the old one's context.
            net.reset_rows(restarted)
    net.train(was_training)
    return (float(np.mean(succ)), float(np.mean(steps)),
            float(np.mean(ratios)) if ratios else float("nan"))
