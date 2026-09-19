# Dreamer 4 — PyTorch

A PyTorch implementation of the Dreamer 4 training pipeline
([Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527)):
learn a world model from recorded video, then improve a policy **inside** that
world model. Four phases, each its own trainer:

| phase | module | what it produces |
|---|---|---|
| 1a | `dreamer4.train.train_tokenizer` | causal video tokenizer — frames to a small latent per timestep |
| 1b | `dreamer4.train.train_dynamics` | world model over those latents, trained with shortcut forcing; the bootstrap term that makes **one** denoising step enough (K=1) is ramped in during the same run |
| 2 | `dreamer4.train.train_heads` | agent finetuning — policy and reward heads on agent tokens **inside** the world model, trained together with the video-prediction loss |
| 3 | `dreamer4.train.train_pmpo` | PMPO reinforcement learning purely in imagination — the environment is never stepped during training |

Phases 1a-1b are configured by the dataclass tree in `dreamer4/train/config.py`
(defaults < `--config file.yaml` < CLI flags); the resolved config is written
to `<out>/config.yaml` and embedded in every checkpoint. Phases 2-3 take plain
flags — their architecture arrives inside the checkpoint they start from.

## Layout

```
dreamer4/
  data/            unified episode format + one module per storage format
  models/          tokenizer, dynamics model, shared transformer backbone,
                   agent_heads.py (policy / reward / value heads on agent tokens),
                   world_model.py (a trained dynamics checkpoint as one object,
                   and the agent policy that acts through it)
  train/           the four trainers, their objectives and shared plumbing
  dynamics_eval.py open-loop rollout gate for the world model
  agent_eval.py    real-environment rollout of a trained policy
scripts/
  train_gridworld.sh   the whole chain, end to end
```

## Install

```bash
pip install -e .                     # the library
pip install -e ../gridworld          # the example environment (see below)
```

`gridworld` is the toy environment these instructions reproduce on. It is a
separate package that lives **next to** this repository and is not a dependency
of the library: `dreamer4.data.gridworld` imports it lazily. Optional extras:
`.[lerobot]` (parquet + video decoding, for LeRobot datasets), `.[dino]`
(DINOv3 perceptual loss), `.[dev]` (pytest, ruff).

## The dataset contract — adding an environment

A dataset is an indexed collection of episodes. Trainers only ever ask for
fixed-length windows and always get the same payload:

```python
{"video":     (T, H, W, C) uint8,      # one frame per timestep
 "proprio":   (T, D_p)     float32,    # optional; the adapter owns the scaling
 "actions":   (T-1, D_a)   float32,    # optional; a[t] drives t -> t+1
 "rewards":   (T-1,)       float32,    # optional; aligned with actions
 "terminals": (T-1,)       bool}       # optional; aligned with actions
```

To add an environment, subclass `EpisodeVideoDataset`
(`dreamer4/data/base.py`), implement `recognizes` / `from_path` / `__len__` /
`episode_frames` / `_load_clip`, and register the class in
`dreamer4.data.ADAPTERS`. `open_video_dataset` then finds it by asking each
adapter whether it recognises the directory.

Everything an environment knows about itself is reached through optional hooks
on that same class; each has a safe default, so an offline dataset trains
without the parts that need a live environment:

| hook | what the trainers use it for |
|---|---|
| `episode_meta(i)` | the collector's per-episode quality signals |
| `bc_weight(i)` | which episodes may be imitated (phase 2 clones those with `> 0`) |
| `env_spec()` | `{"id", "kwargs"}` for `gymnasium.make` — real-environment evaluation in phases 2-3 |
| `continues_from_reward(r)` | the domain's terminal rule; a dream has no recorded terminals, so phase 3 stops on this |
| `proprio_from_info(info)` | one step's proprio from a live env, in the same convention the clips carry |
| `eval_metrics` / `gate` | domain-specific reconstruction metrics and the pass/fail gate for phases 1a-1b |

`dreamer4/data/gridworld.py` is the worked example, and
`dreamer4/data/lerobot.py` is a second storage format (multi-camera video +
parquet).

## The agent

As in the paper (Section 3.3), the policy is not a network beside the world
model but a set of heads **inside** it. Agent tokens are interleaved into the
dynamics transformer as one more modality; they receive a task embedding as
input and attend to every other modality while **nothing attends back**, so the
world model never predicts the future from what the agent intends. The policy
and reward are small MLPs on the agent token's output embedding `h_t`, with one
output layer per multi-token-prediction distance (`--mtp`, the paper's L = 8).
Because attention is block-causal in time, `h_t` depends on the whole context
window, so phase 2 trains on clips.

**Phase 2** finetunes the whole transformer: the video-prediction loss keeps
running on noisy representations, the agent loss (behavioral cloning and reward
prediction, Eq. 9) is added to it, and every term is RMS-normalised. Half of each
agent batch comes from demonstrations and half from all episodes; the cloning
term applies only to the demonstration half (Section 4.1).

**Phase 3** freezes the transformer and trains only the policy and value heads.
Imagined rollouts start from recorded contexts, one rollout per context. The
value head is a symexp two-hot distribution trained by cross-entropy on the
lambda-return (Eq. 10); the policy is trained with PMPO (Eq. 11) — the sign of
the advantage, `alpha = 0.5` between the positive and negative sets, and a fixed
`beta = 0.3` on the reverse KL to the behavioral prior.

The same context-window convention is used by phase 2, by imagination and by the
live evaluator: a `--clip_T`-slot window ending at the current frame, left-padded
by repeating the episode's first frame, with the action start flag at slot 0 and
at every padded slot, and the action in slot `j` being the one that led **into**
that slot's frame.

## Deviations from the paper

| what | paper | here | why |
|---|---|---|---|
| context length | C = 192 frames | `--clip_T` = the window the dynamics model was trained on (`--data.seq_len`, 4 on the gridworld recipe) | the policy's context is the context imagination dreams with, and the dynamics only follows the actions inside window lengths it was trained on (with 7+ past frames the dreamed player is in the wrong cell on >90 % of the steps); imagination needs one slot spare for the frame being denoised |
| tokenizer bottleneck | 512 latent tokens x 16 channels, masked autoencoding | 16 x 16, no masking | sized for a toy environment and one GPU |
| proprio stream | not present | one token per frame, denoised jointly with the latents | the agent's own state (the player's position on the gridworld) |
| bootstrap term | part of shortcut forcing | ramped in during phase 1b (`--objective.bootstrap_start_frac`, `--objective.bootstrap_ramp_frac`) | distilling two half-steps into one needs a model that can already take a half-step |

## Choices the paper leaves open

The paper has no hyperparameter table, and Section 3.3 leaves the following
unstated. Each line is what this implementation picked.

| open point | choice here |
|---|---|
| number of agent tokens per frame | `--n_agent 1`; more are concatenated into `h_t` |
| position of the agent tokens in the block | last, so every world token keeps its phase-1 positional index and a phase-1 checkpoint loads unchanged |
| which layer produces `h_t` | the final one |
| which denoising pass supplies `h_t` in imagination | the pass that commits the finished frame at the context signal level `1 - tau_ctx` |
| whether block `t` carries `a_t` or `a_{t-1}` | `a_{t-1}`, the action that led into frame `t`; carrying `a_t` would hand the policy the action it is asked to predict |
| lambda | `--lam 0.95` |
| two-hot bins for reward and value | 255 bins over symlog [-1, 1]; over wide bins a small mass on a far bin moves the decoded value a lot |
| how `c_t` is produced | the dataset's terminal rule applied to the reward head's distribution: `P(terminal) = sum_b softmax(logits)_b * (1 - continues(bin_b))` |
| imagination horizon | the longest recorded episode (`--horizon 0`) |
| whether the value head uses MTP | no |
| optimisation | phase 2: AdamW, `--lr 5e-4` heads, `--dyn_lr 5e-5` transformer, cosine decay, grad clip 1.0. Phase 3: AdamW, `--lr 3e-4` policy, `--value_lr 1e-3` value, linear decay, grad clip 0.5 |
| whether the behavioral prior is refreshed | no — a frozen copy taken at the start of phase 3 |

## Reproducing the gridworld run

16x16 grid, 10 000 mixed-quality episodes, 40-step cap.

```bash
# empty maze — the whole chain on one RTX 4090
COLLECT=1 ./scripts/train_gridworld.sh

# fewer demonstrations
BC_FRAC=0.2 HEADS_STEPS=2000 HEADS_EXTRA="--eval_every 250" \
  OUT=runs/fifth ./scripts/train_gridworld.sh

# obstacles — 0 to 25 % of the cells are walls, drawn per episode
COLLECT=1 DATA=data/gw_obs_10k OUT=runs/gw_obs DENSITY="0,0.25" \
  HEADS_STEPS=8000 ./scripts/train_gridworld.sh
```

The script is resumable (a phase whose `<out>/final.json` exists is skipped),
and every path, GPU, seed and step count is an environment variable;
`HEADS_EXTRA` / `PMPO_EXTRA` pass further flags to phases 2 / 3. The commands it
runs:

```bash
# step 0 — dataset (companion `gridworld` package)
gridworld-collect --out data/gw_noobs_10k --n-episodes 10000 --size 16 \
  --obstacle-density 0 --max-steps 40 --shard-size 1000 --base-seed 4242 \
  --step-penalty 0.01 --goal-reward 1.0 --stickiness 0.0

# 1a — tokenizer
python -m dreamer4.train.train_tokenizer \
  --data.path data/gw_noobs_10k --out runs/gridworld/tok \
  --steps 16000 --seed 1 --resume True

# 1b — world model (these flags are the trainer defaults): bootstrap fraction 0
#      until step 24 000, 0 -> 0.5 by 25 500; LR 3e-4 -> 5e-5 over the 1 000 steps
#      BEFORE that ramp opens
python -m dreamer4.train.train_dynamics \
  --data.path data/gw_noobs_10k \
  --tokenizer.ckpt runs/gridworld/tok/checkpoints/latest.pt \
  --out runs/gridworld/dyn --steps 30000 --optim.lr 3e-4 --optim.lr_final 5e-5 \
  --optim.lr_decay_steps 1000 --objective.bootstrap_frac 0.5 \
  --objective.bootstrap_start_frac 0.8 --objective.bootstrap_ramp_frac 0.05 \
  --seed 1 --resume True

# 2 — agent finetuning
python -m dreamer4.train.train_heads \
  --dyn runs/gridworld/dyn/checkpoints/latest.pt --out runs/gridworld/heads \
  --steps 6000 --bc_frac 1.0 --seed 1

# 3 — PMPO in imagination
PYTORCH_ALLOC_CONF=expandable_segments:True python -m dreamer4.train.train_pmpo \
  --dyn runs/gridworld/dyn/checkpoints/latest.pt \
  --init_from runs/gridworld/heads/checkpoints/best.pt --out runs/gridworld/rl \
  --K 1 --batch 32 --steps 3200 --eval_every 800 --eval_n 1000 --seed 1
```

Phase 1 has its own gate, checked at the end of each run: the tokenizer must
place the sprites within one cell, and the world model must survive a 39-step
open-loop rollout at both K=4 and K=1.

## Results

Empty maze, held-out evaluation over 2 000 episodes (seed 900000, disjoint from
the training data), one training seed, one RTX 4090. 5 349 episodes are eligible
as demonstrations; the proprio stream carries the player position only, so the
goal has to be seen in the latents. Success rate, and path length / shortest path
over the solved episodes:

| | greedy | sampled |
|---|---|---|
| phase 2, 6 000 steps (7 min) | **1.000** / 1.003 | 0.995 / 1.54 |
| phase 3, 400 updates (+4 min) | **1.000** / 1.002 | **1.000** / 1.028 |
| phase 3, 3 200 updates | **1.000** / 1.002 | **1.000** / 1.018 |

Measured 2026-09-19. The 0.9935 / 0.9960 (path 1.078) this table carried before
came from a policy context of `max_T - 1 = 11` frames on a world model trained on
4-frame windows -- which phase 3 also dreamed with (the dreamed player was in the
wrong cell on 94 % of the steps) -- and from a PMPO term that included the action
"taken" at a dream's terminal frame. With both fixed, the same recipe gives the
numbers above.

The obstacle maze is not solved by this recipe. There the phase-2 finetune costs
the transformer its rollout accuracy (the dreamed player is in the wrong cell on
63 % of the steps against 1 % before), so phase 3 has no usable simulator. Cloning
alone reaches about 0.87 greedy (phase 2 run as `--dyn_weight 0 --dyn_lr 2e-4
--batch 96`, which is not the paper's objective), and the remaining failures are a
greedy policy pushing into a wall it does not see. The open problem is a phase 2
that keeps the world model.
