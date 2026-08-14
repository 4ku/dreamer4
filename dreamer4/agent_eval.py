"""
Run a trained agent in a REAL environment and score it.

This is the number that decides whether a run ships. Offline metrics (BC
accuracy, reward MAE) only say the model fits the data; they cannot say the
policy actually reaches goals. Only playing the game answers that — so this
module is to phase 2/3 what the windowed rollout gate is to phase 1.

Nothing here is tied to one environment. Any **Gymnasium** env works as
long as its observation is an image the tokenizer can encode and its action
space matches the policy head (discrete or continuous). Everything
domain-specific — which env to open, how to read its proprio — comes from
the dataset the agent was trained on, via the ``env_spec`` /
``proprio_from_info`` hooks in :mod:`dreamer4.data.base`. Gridworld is simply
the first dataset to implement them.

What is in here
---------------
``AgentPolicy``        pixels in -> action out. The deployment loop.
``make_env``           open the env a dataset/checkpoint names (``gymnasium.make``).
``evaluate_policy``    play N seeded episodes, return metrics + per-episode rows.
``evaluate_random``    the same with random actions (``action_space.sample()``):
                       the floor any policy must clear.
``summarize_records``  per-episode rows -> success rate, return, steps/optimal.
``main``               the ``dreamer4-eval-agent`` CLI (evaluate a saved checkpoint).

The trainer calls ``evaluate_policy`` / ``evaluate_random`` itself at the end
of every run, so ``final.json`` always carries online numbers.

One step of the deployment loop
-------------------------------
1. the env hands over an RGB frame;
2. the frozen tokenizer encodes it into one latent, using its sliding
   ``history`` window of past frames. A model with a proprio stream reads it
   from the env's own ``info`` — the exact state, never inferred from the
   render (the dataset supplies that reader and guarantees it matches the
   convention training used);
3. take the last ``window`` latents, corrupt them all to ``tau_ctx``, and
   attach the actions that led into them (start flag on the first slot, the
   same alignment ``windowed_rollout`` uses);
4. one forward pass of the dynamics transformer in ``wm_agent`` mode;
5. read the agent token at slot ``POLICY_SLOT`` of the LAST position and turn
   it into an action — ``greedy=True`` takes the distribution's mode,
   ``greedy=False`` samples. On gridworld **sampling scores ~5 points higher**
   (argmax gets stuck in two-step loops in some layouts), so it is the mode to
   deploy; the CLI and the trainer measure both;
6. step the env, repeat.

Steps 2-4 are deliberately identical to the agent forward in training
(``train_agent``), so the policy acts on exactly the kind of embedding it
learned on.

**No denoising here.** K, the denoise-step count, never appears: the world
model only denoises when it has to *invent* a frame, and here every frame is
real. K comes back in phase 3, where the agent acts inside dreams.

Metrics reported
----------------
``success_rate``, ``mean_return``, ``timeout_rate``, ``mean_steps``. An
episode counts as a success from ``info["is_success"]`` (the Gymnasium
convention) or, failing that, from ``terminated`` — i.e. the episode ended
by reaching a terminal state rather than running out of time.

``steps_over_optimal`` appears only when the env reports a best-case episode
length in ``info["optimal_steps_from_start"]``: steps taken divided by the
shortest possible path, averaged over successes, so 1.0 means perfect
routing. Episode seeds default to 1e6+, far outside the training datasets'
seed range.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from dreamer4.models.dynamics import DynamicsModel, _corrupt_past, _log2_int
from dreamer4.models.agent import (POLICY_SLOT, REWARD_SLOT, PolicyHead,
                                   TaskEncoder)
from dreamer4.models.tokenizer import FrozenTokenizer


class AgentPolicy:
    """
    The deployed agent: pixels in, action out (see module docstring for the
    loop and why it matches training bit-for-bit). Environment-agnostic.

    Args:
        dynamics:     Finetuned DynamicsModel (n_agent > 0, wm_agent mode).
        task_encoder: TaskEncoder producing the agent-token inputs.
        policy_head:  PolicyHead reading slot ``POLICY_SLOT``; its
                      ``action_type`` decides what ``act`` returns.
        tokenizer:    FrozenTokenizer (same codec the model was trained on).
        window:       Sliding context window (= the trained seq_len).
        tau_ctx:      Context corruption level (the trained value).
        task_id:      Task index (single-task datasets: 0).
        greedy:       True = distribution mode; False = sample.
        proprio_from_info: ``info -> (D_p,) float32`` — the env's OWN state,
                      REQUIRED iff the dynamics model has a proprio stream.
                      Pass the training dataset's ``proprio_from_info``, which
                      is what keeps deployment on the same convention the
                      model was trained with.
    """

    def __init__(self, *, dynamics: DynamicsModel, task_encoder: TaskEncoder,
                 policy_head: PolicyHead, tokenizer: FrozenTokenizer,
                 window: int = 4, tau_ctx: float = 0.1, task_id: int = 0,
                 greedy: bool = True,
                 proprio_from_info: Optional[Callable[[Dict], np.ndarray]] = None,
                 reward_head=None, device: str | torch.device = "cuda"):
        if dynamics.d_proprio is not None and proprio_from_info is None:
            raise ValueError(
                "this model has a proprio stream, so proprio_from_info is "
                "required — pass the training dataset's proprio_from_info "
                "(an env that cannot report its state cannot host this model)")
        self.proprio_from_info = proprio_from_info
        # optional: lets act_detailed report what the reward head predicts
        self.reward_head = None if reward_head is None else reward_head.eval()
        self.dynamics = dynamics.eval()
        self.task_encoder = task_encoder.eval()
        self.policy_head = policy_head.eval()
        self.tokenizer = tokenizer
        self.window = int(window)
        self.tau_ctx = float(tau_ctx)
        self.greedy = bool(greedy)
        self.device = torch.device(device)
        # width of one action slot in the model's input (see align_actions:
        # the dataset action vector plus the start flag)
        self.action_vec_dim = (policy_head.action_dim
                               if policy_head.action_type == "continuous"
                               else policy_head.action_dim * policy_head.num_categories)
        k_max = dynamics.k_max
        self._emax = _log2_int(k_max)
        self._ctx_sig = min(int(round((1.0 - tau_ctx) * k_max)), k_max)
        with torch.no_grad():
            task = torch.full((1,), task_id, dtype=torch.long, device=self.device)
            # (1, 1, n_agent, d_model) — broadcast over the window at act()
            self._agent_in = task_encoder(task)
        self.reset()

    def reset(self) -> None:
        """Clear the frame/action history. Call at every episode start."""
        self._frames: List[np.ndarray] = []
        self._latents: List[torch.Tensor] = []
        self._action_vecs: List[torch.Tensor] = []
        self._proprios: List[np.ndarray] = []

    def _record_proprio(self, info: Optional[Dict]) -> None:
        """Append this step's proprio, read from the env's own state."""
        prop = None if info is None else self.proprio_from_info(info)
        if prop is None:
            raise ValueError(
                "no proprio for this step: the env reported no usable info "
                "(pass info to act(), and check the dataset's "
                "proprio_from_info)")
        self._proprios.append(np.asarray(prop, np.float32))

    @torch.no_grad()
    def _agent_slots(self, obs: np.ndarray,
                     info: Optional[Dict]) -> torch.Tensor:
        """Run one deployment forward and return this step's agent-token
        outputs ``(n_agent, d_model)`` — the heads' inputs."""
        self._frames.append(obs)
        if self.dynamics.d_proprio is not None:
            self._record_proprio(info)
        hist = self._frames[-self.tokenizer.history:]
        z_t = self.tokenizer.encode_frames(np.stack(hist)[None])[0, -1]
        self._latents.append(z_t)

        w = min(len(self._latents), self.window)
        z_win = torch.stack(self._latents[-w:], 0)[None]        # (1, w, Nz, Dz)

        # per-window action alignment with the start flag, exactly like
        # windowed_rollout: position 0 = "nothing led into this window"
        aligned = torch.zeros(1, w, self.action_vec_dim + 1, device=self.device)
        aligned[0, 0, -1] = 1.0
        for j in range(1, w):
            prev = self._action_vecs[len(self._action_vecs) - (w - 1) + (j - 1)]
            aligned[0, j, :self.action_vec_dim] = prev

        step_idx = torch.full((1, w), self._emax, device=self.device,
                              dtype=torch.long)
        signal_idx = torch.full((1, w), self._ctx_sig, device=self.device,
                                dtype=torch.long)
        z_c = _corrupt_past(z_win, self.tau_ctx)
        agent_in = self._agent_in.expand(1, w, -1, -1)

        kwargs = {}
        if self.dynamics.d_proprio is not None:
            prop = torch.as_tensor(np.stack(self._proprios[-w:]))[None].to(
                self.device)
            kwargs["proprio_noisy"] = _corrupt_past(prop, self.tau_ctx)
        out = self.dynamics(aligned, step_idx, signal_idx, z_c,
                            agent_tokens=agent_in, **kwargs)
        return out[-1][0, -1]                              # (n_agent, d_model)

    def _commit(self, action: torch.Tensor) -> Any:
        """Record the chosen action for the next window and hand back the
        env-ready form."""
        self._action_vecs.append(
            self.policy_head.action_to_vector(action).reshape(-1))
        arr = action.reshape(-1).cpu().numpy()
        if self.policy_head.action_type == "discrete" and arr.size == 1:
            return int(arr[0])                     # Discrete(n) wants a scalar
        return arr

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> Any:
        """
        One env step: ``(H,W,C)`` uint8 observation (plus the env's ``info``,
        when available, for exact proprio) -> an action ready for
        ``env.step`` — a python int for a single discrete action, else a
        numpy array.
        """
        return self.act_detailed(obs, info)[0]

    @torch.no_grad()
    def act_detailed(self, obs: np.ndarray, info: Optional[Dict] = None,
                     *, override: Optional[int] = None
                     ) -> Tuple[Any, Dict[str, Any]]:
        """
        :meth:`act` plus what the heads saw — for playgrounds, debugging, and
        the phase-3 calibration probe.

        Args:
            override: take THIS action instead of the policy's (the history
                      still records it, so the model keeps a truthful past).
                      Lets a human drive and then hand control back.

        Returns:
            ``(env_action, {"probs", "reward_pred", "continue_pred",
            "chosen"})``. ``probs`` is None for continuous policies.
        """
        slots = self._agent_slots(obs, info)
        params = self.policy_head(slots[POLICY_SLOT], head_idx=0)
        discrete = self.policy_head.action_type == "discrete"

        if override is not None:
            action = torch.as_tensor([override], device=self.device).long()
        else:
            action = (self.policy_head.mode(params) if self.greedy
                      else self.policy_head.sample(params)[0])

        probs = None
        if discrete:
            logits = params.reshape(self.policy_head.action_dim,
                                    self.policy_head.num_categories)
            probs = torch.softmax(logits, -1)[0].cpu().numpy().tolist()
        reward_pred = float(self.reward_head.predict(slots[REWARD_SLOT])) \
            if self.reward_head is not None else None
        diag = {
            "probs": probs,
            "reward_pred": reward_pred,
            # the phase-3 dream-stop rule, shown live
            "continue_pred": (None if reward_pred is None
                              else float(reward_pred <= 0.5)),
            "chosen": int(action.reshape(-1)[0]) if discrete else None,
            "overridden": override is not None,
        }
        return self._commit(action), diag


def make_env(spec: Optional[Dict], **overrides):
    """
    Open the environment named by an ``env_spec`` — ``{"id": ..., "kwargs":
    {...}}``, as produced by ``EpisodeVideoDataset.env_spec`` and stored in
    agent checkpoints. Uses ``gymnasium.make``, so any registered env works.
    """
    if not spec or not spec.get("id"):
        raise ValueError("no env_spec: this dataset/checkpoint has no live "
                         "environment to evaluate in")
    import gymnasium as gym
    return gym.make(spec["id"], **{**spec.get("kwargs", {}), **overrides})


def _rollout(env, act_fn, *, episodes: int, seed: int,
             on_reset: Optional[Callable[[], None]] = None) -> List[Dict]:
    """Play ``episodes`` seeded episodes, one record each. ``act_fn(obs)``
    picks the action from the observation and the env's ``info``;
    ``on_reset`` (the policy's history clear) runs before each episode."""
    records: List[Dict] = []
    for i in range(episodes):
        if on_reset is not None:
            on_reset()
        obs, info = env.reset(seed=seed + i)
        ep_return, steps, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            action = act_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            steps += 1
        records.append({
            "seed": seed + i,
            # Gymnasium convention; without it, "ended, not timed out"
            "success": bool(info.get("is_success", terminated)),
            "steps": steps,
            "return": ep_return,
            "optimal_steps": info.get("optimal_steps_from_start"),
        })
    return records


def evaluate_policy(
    policy: AgentPolicy,
    env,
    *,
    episodes: int,
    seed: int = 1000000,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Roll ``policy`` for ``episodes`` seeded episodes of ``env`` (any
    Gymnasium env; build one with :func:`make_env`).

    Returns ``(metrics, per-episode records)`` — see the module docstring for
    what the metrics mean.
    """
    records = _rollout(env, policy.act, episodes=episodes, seed=seed,
                       on_reset=policy.reset)
    return summarize_records(records), records


def evaluate_random(
    env,
    *,
    episodes: int,
    seed: int = 1000000,
) -> Dict[str, float]:
    """The floor baseline: ``action_space.sample()`` on the same seeded
    episodes any policy is measured on."""
    env.action_space.seed(seed)
    return summarize_records(
        _rollout(env, lambda _obs, _info: env.action_space.sample(),
                 episodes=episodes, seed=seed))


def summarize_records(records: List[Dict]) -> Dict[str, float]:
    """Aggregate per-episode records into the standard online metrics."""
    if not records:
        return {"episodes": 0}
    succ = np.array([r["success"] for r in records], dtype=bool)
    steps = np.array([r["steps"] for r in records], dtype=np.float64)
    out = {
        "episodes": len(records),
        "success_rate": float(succ.mean()),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "timeout_rate": float((~succ).mean()),
    }
    if succ.any():
        out["mean_steps"] = float(steps[succ].mean())
        # only when the env reports a best case to compare against
        opt = [r.get("optimal_steps") for r in records]
        if all(o is not None for o in opt):
            opt = np.maximum(np.array(opt, dtype=np.float64), 1.0)
            out["steps_over_optimal"] = float((steps[succ] / opt[succ]).mean())
    return out


def main(argv=None) -> None:
    """CLI: evaluate a saved phase-2 agent checkpoint online."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", required=True,
                        help="train_agent checkpoint (checkpoints/latest.pt)")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--mode", choices=("greedy", "sample", "both"),
                        default="both")
    parser.add_argument("--random-baseline", action="store_true",
                        help="also roll the uniform-random floor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="", help="optional JSON output path")
    args = parser.parse_args(argv)

    # lazy: train_agent imports this module
    from dreamer4.train.train_agent import load_agent_policy
    results: Dict[str, Dict[str, float]] = {}
    modes = ("greedy", "sample") if args.mode == "both" else (args.mode,)
    env = None
    for mode in modes:
        policy, env_spec = load_agent_policy(
            args.ckpt, device=args.device, greedy=(mode == "greedy"))
        env = env or make_env(env_spec)
        if mode == "sample":
            torch.manual_seed(args.seed)
        metrics, _ = evaluate_policy(policy, env, episodes=args.episodes,
                                     seed=args.seed)
        results[mode] = metrics
        print(f"[{mode}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                       else f"{k}={v}"
                                       for k, v in metrics.items()))
    if args.random_baseline:
        results["random"] = evaluate_random(env or make_env(
            load_agent_policy(args.ckpt, device=args.device)[1]),
            episodes=args.episodes, seed=args.seed)
        print("[random] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                      else f"{k}={v}"
                                      for k, v in results["random"].items()))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
