#!/usr/bin/env bash
# Reproduce the full Dreamer4 training chain on the 16x16 gridworld:
# tokenizer -> world model -> agent finetuning -> PMPO in imagination.
# One RTX 4090; see the README results table.
#
# TWO RECIPES, same chain, two environment variables apart:
#
#   empty maze  DENSITY=0         HEADS_STEPS=4000
#   obstacles   DENSITY="0,0.25"  HEADS_STEPS=8000   (not re-measured with the
#                                                     current phase-2/3 defaults)
#
# Everything is an environment variable with a default, so a different
# dataset, run dir, GPU or seed needs no edit:
#
#   DATA=data/gw_noobs_10k  dataset directory (see COLLECT below)
#   OUT=runs/gridworld      run root; each phase writes OUT/<phase>
#   GPU=0                   CUDA device index
#   SEED=1                  seed for every phase
#   BC_FRAC=1.0             fraction of the eligible demonstrations phase 2
#                           may clone (e.g. 0.2 clones a fifth of them)
#   HEADS_STEPS=4000        phase-2 steps
#   PMPO_STEPS=1000         phase-3 policy updates
#   TOK=""                  path of an existing tokenizer run dir; when set,
#                           phase 1a is skipped and that tokenizer is reused
#   COLLECT=0               1 = collect DATA first if it does not exist
#   DENSITY=0               obstacle density of the step-0 collection: a fixed
#                           fraction, or "lo,hi" sampled per episode. 0 is the
#                           empty maze of the results table; "0,0.25" collects
#                           the obstacle variant of the same task
#   HEADS_EXTRA=""          extra flags appended to the phase-2 command, e.g.
#                           HEADS_EXTRA="--eval_every 250" for a short run
#   PMPO_EXTRA=""           extra flags appended to the phase-3 command
#
# Resumable: a phase whose <out>/final.json exists is skipped, so re-running
# the script after an interruption continues where it stopped (each trainer
# also resumes from its own latest checkpoint).
#
#   COLLECT=1 ./scripts/train_gridworld.sh
#   COLLECT=1 OUT=runs/gw BC_FRAC=0.5 HEADS_STEPS=10000 ./scripts/train_gridworld.sh
#
# Few demonstrations (214 of them: cloning no longer solves the maze, phase 3 does;
# 100 % success, path / shortest < 1.01, 10 min for phases 2-3 -- README, results):
#
#   BC_FRAC=0.04 OUT=runs/fewdemo ./scripts/train_gridworld.sh
#
# Phases 2 and 3 evaluate in the real env only once, at their end (that is how
# the README's wall-clock numbers were taken); for curves pass e.g.
# HEADS_EXTRA="--eval_every 1000" PMPO_EXTRA="--eval_every 200".
#   COLLECT=1 DATA=data/gw_obs_10k OUT=runs/gw_obs DENSITY="0,0.25" \
#     HEADS_STEPS=8000 ./scripts/train_gridworld.sh

set -euo pipefail

# Run from the repo root: every path below is relative to it, and `python -m`
# then finds the `dreamer4` package without any PYTHONPATH juggling.
cd "$(dirname "$0")/.."

DATA=${DATA:-data/gw_noobs_10k}
OUT=${OUT:-runs/gridworld}
GPU=${GPU:-0}
SEED=${SEED:-1}
BC_FRAC=${BC_FRAC:-1.0}
HEADS_STEPS=${HEADS_STEPS:-4000}
PMPO_STEPS=${PMPO_STEPS:-1000}
TOK=${TOK:-}
COLLECT=${COLLECT:-0}
DENSITY=${DENSITY:-0}
PYTHON=${PYTHON:-python}
# Split on whitespace like a shell command line would; empty stays an empty array.
read -r -a HEADS_EXTRA <<<"${HEADS_EXTRA:-}"
read -r -a PMPO_EXTRA <<<"${PMPO_EXTRA:-}"

export CUDA_VISIBLE_DEVICES=$GPU

# Skip a phase that already finished.
done_phase() { [ -f "$1/final.json" ]; }

# --- step 0: dataset (companion `gridworld` package, not a dependency) ------
# 10 000 mixed-quality episodes, 40-step cap, obstacle density DENSITY. The
# collector dials expert->random per episode; phase 2 clones only the clean
# successful ones (dreamer4/data/gridworld.py::bc_weight). The density is
# written into the manifest, and that is what the real-environment evaluation
# of phases 2-3 rebuilds the env from -- so the policy is always scored on the
# maze distribution it was trained on.
if [ "$COLLECT" = "1" ] && [ ! -d "$DATA" ]; then
  echo "=== step 0: collecting $DATA ==="
  gridworld-collect --out "$DATA" --n-episodes 10000 --size 16 \
    --obstacle-density "$DENSITY" --max-steps 40 --shard-size 1000 --base-seed 4242 \
    --step-penalty 0.01 --goal-reward 1.0 --stickiness 0.0
fi

# --- phase 1a: causal video tokenizer --------------------------------------
if [ -n "$TOK" ]; then
  echo "=== phase 1a: reusing tokenizer $TOK ==="
elif done_phase "$OUT/tok"; then
  TOK=$OUT/tok
  echo "=== phase 1a: already done ($TOK) ==="
else
  echo "=== phase 1a: tokenizer ==="
  $PYTHON -m dreamer4.train.train_tokenizer \
    --data.path "$DATA" --out "$OUT/tok" --steps 16000 --seed "$SEED" --resume True
  TOK=$OUT/tok
fi

# --- phase 1b: dynamics (world model) --------------------------------------
# ONE run. The bootstrap term (two half-steps distilled into one) is what makes
# single-step sampling legal -- imagination cannot afford K=4 per dreamed frame
# -- but it can only distill a model that already exists, so its batch fraction
# stays 0 for the first 24 000 steps, ramps to 0.5 over the next 1 500 and is
# held there to 30 000. The LR walks from 3e-4 down to 5e-5 over the 1 000
# steps BEFORE the ramp opens, and the loss normalizer is re-seeded once where
# it opens: decaying the LR across the ramp instead diverges on the obstacle
# maze. These are the trainer defaults; they are spelled out to pin the recipe.
if done_phase "$OUT/dyn"; then
  echo "=== phase 1b: already done ==="
else
  echo "=== phase 1b: dynamics ==="
  $PYTHON -m dreamer4.train.train_dynamics \
    --data.path "$DATA" \
    --tokenizer.ckpt "$TOK/checkpoints/latest.pt" \
    --out "$OUT/dyn" --steps 30000 --optim.lr 3e-4 --optim.lr_final 5e-5 \
    --optim.lr_decay_steps 1000 --objective.bootstrap_frac 0.5 \
    --objective.bootstrap_start_frac 0.8 --objective.bootstrap_ramp_frac 0.05 \
    --seed "$SEED" --resume True
fi

# --- phase 2: agent finetuning ---------------------------------------------
if done_phase "$OUT/heads"; then
  echo "=== phase 2: already done ==="
else
  echo "=== phase 2: agent finetuning ==="
  $PYTHON -m dreamer4.train.train_heads \
    --dyn "$OUT/dyn/checkpoints/latest.pt" --out "$OUT/heads" \
    --steps "$HEADS_STEPS" --eval_every "$HEADS_STEPS" \
    --bc_frac "$BC_FRAC" --seed "$SEED" ${HEADS_EXTRA[@]+"${HEADS_EXTRA[@]}"}
fi

# --- phase 3: PMPO in imagination ------------------------------------------
# expandable_segments keeps the allocator from fragmenting over the variable
# dream horizons.
if done_phase "$OUT/rl"; then
  echo "=== phase 3: already done ==="
else
  echo "=== phase 3: PMPO in imagination ==="
  PYTORCH_ALLOC_CONF=expandable_segments:True $PYTHON -m dreamer4.train.train_pmpo \
    --dyn "$OUT/dyn/checkpoints/latest.pt" \
    --init_from "$OUT/heads/checkpoints/latest.pt" --out "$OUT/rl" \
    --steps "$PMPO_STEPS" --eval_every "$PMPO_STEPS" --eval_n 1000 \
    --seed "$SEED" ${PMPO_EXTRA[@]+"${PMPO_EXTRA[@]}"}
fi

echo "=== done: $OUT ==="
