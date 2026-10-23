#!/usr/bin/env bash
# The general AlohaPick recipe end to end, one GPU:
#   1. the online loop from a phase-2 clone: imagination PMPO (horizon 32) -> 128 real episodes -> world-model refresh
#      (50 % online video windows and dream starts, the policy clones the top 25 % of all real episodes by return,
#      every 50th PMPO update learns from the newest real episodes), ITERS iterations;
#   2. pick the iteration whose collection had the best return;
#   3. real-env PMPO polish of that policy, 32 episodes per update, POLISH updates;
#   4. held-out evaluation (512 episodes, seeds 900000+) of the polished best checkpoint.
#   scripts/aloha/recipe.sh <gpu> <tag> <phase-2 clone ckpt> <world-model run> [iterations] [polish updates]
# Env: A (workspace with data/ and runs/), PY (the dreamer4 venv's python), SEED (first collection seed).
# From a 500-step clone (held-out greedy 0.60 / 41.7) it reached 0.97 / 96.1 held-out (2026-10-23); continuing the
# polish with 64 episodes per update at LR 3e-4 took it to 0.98 / 99.4 (sampled).
set -u
A=${A:-/mnt/dsk1/iefremov/aloha}
PY=${PY:-/mnt/dsk1/iefremov/dreamer/dreamer4/.venv/bin/python}
export A PY ALOHA_PYTHON=$A/mujoco_playground/.venv/bin/python PYTORCH_ALLOC_CONF=expandable_segments:True
GPU=$1 TAG=$2 CLONE=$3 BASE=$4 ITERS=${5:-10} POLISH=${6:-1000}
HERE=$(cd "$(dirname "$0")" && pwd)
stamp() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }
WM=$A/runs/$BASE/checkpoints/latest.pt
R=$A/runs/dyna_$TAG

stamp "1. online loop ($ITERS iterations)"
HORIZON=32 ONLINE_FRAC=0.5 START_FRAC=0.5 K=1000 BC_TOP=0.25 REAL_EVERY=50 SEED=${SEED:-27000000} \
  bash "$HERE/dyna_loop.sh" "$GPU" "$TAG" "$CLONE" "$BASE" "$ITERS" || { stamp "loop FAILED"; exit 1; }

BEST=$(tail -n +2 "$R/progress.tsv" | sort -t$'\t' -k4 -g | tail -1 | cut -f1)
stamp "2. best iteration: $BEST ($(grep -P "^$BEST\t" "$R/progress.tsv" | cut -f3,4))"

stamp "3. real-env polish of rl_$BEST, $POLISH updates x 32 episodes"
cd "${REPO:-$HERE/../..}"
CUDA_VISIBLE_DEVICES=$GPU $PY -m dreamer4.train.train_pmpo --dyn "$WM" --init_from "$R/rl_$BEST/checkpoints/latest.pt" \
  --out "$A/runs/polish_$TAG" --seed 1 --steps "$POLISH" --real_env 32 --lr_final 3e-5 --adv_drop_final 0.5 \
  --clip_split 1 --eval_every 200 --eval_n 128 --eval_envs 64 --log_every 20 --beta 0.1 || { stamp "polish FAILED"; exit 1; }

stamp "4. held-out of the polished best"
bash "$HERE/heldout_queue.sh" "$GPU" "recipe_$TAG=$A/runs/polish_$TAG/checkpoints/best.pt@$WM"
stamp "RECIPE DONE"
