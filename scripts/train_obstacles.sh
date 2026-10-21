#!/usr/bin/env bash
# The obstacle-maze recipe on ONE RTX 4090, about 3.1 h end to end:
# tokenizer -> world model (width 512, bootstrapped in one run) -> agent finetuning
# (two seeds, the better one on validation layouts) -> PMPO in imagination
# (16 000 updates) -> held-out evaluation. Paper architecture throughout.
#
# Measured 2026-10-21, three seeds, held-out n = 2 000 (seeds 900000+, layouts no
# phase has seen): success 0.957 / 0.944 / 0.953, path / shortest 1.070 / 1.099 /
# 1.078; 3.05-3.10 h each (tokenizer 8 min, world model 26, phase 2 15, phase 3
# 134). Those runs ended a dream when the reward head gave a terminal probability
# above 0.3; this script keeps the trainer default 0.5. README, "Obstacle maze".
#
# Environment variables, with defaults:
#   DATA=data/gw_obs_10k   dataset directory; COLLECT=1 collects it first if missing
#   OUT=runs/obstacles     run root; each phase writes OUT/<phase>, times go to OUT/timing.log
#   GPU=0  SEED=1  PYTHON=python
#   PMPO_STEPS=16000       phase-3 policy updates
#   TOK=""                 an existing tokenizer run dir to reuse (phase 1a is skipped)
#
# Resumable: a phase whose <out>/final.json exists is skipped.
#
#   COLLECT=1 ./scripts/train_obstacles.sh
#   SEED=2 OUT=runs/obstacles_s2 GPU=1 ./scripts/train_obstacles.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data/gw_obs_10k}
OUT=${OUT:-runs/obstacles}
GPU=${GPU:-0}
SEED=${SEED:-1}
PMPO_STEPS=${PMPO_STEPS:-16000}
TOK=${TOK:-}
COLLECT=${COLLECT:-0}
PYTHON=${PYTHON:-python}

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT"
done_phase() { [ -f "$1/final.json" ]; }
stamp() { echo "$1: $(( ($(date +%s) - $2) / 60 )) min" | tee -a "$OUT/timing.log"; }
T0=$(date +%s)

# --- step 0: dataset --------------------------------------------------------
# 10 000 mixed-quality episodes, 0-25 % of the cells walls (drawn per episode),
# 40-step cap. Base seed 0 is the set the README numbers were measured on.
if [ "$COLLECT" = "1" ] && [ ! -d "$DATA" ]; then
  gridworld-collect --out "$DATA" --n-episodes 10000 --size 16 \
    --obstacle-density "0,0.25" --max-steps 40 --shard-size 1000 --base-seed 0 \
    --step-penalty 0.01 --goal-reward 1.0 --stickiness 0.0
fi

# --- phase 1a: tokenizer (repo defaults) ------------------------------------
t=$(date +%s)
if [ -n "$TOK" ]; then
  echo "=== phase 1a: reusing tokenizer $TOK ==="
elif done_phase "$OUT/tok"; then
  TOK=$OUT/tok
else
  $PYTHON -m dreamer4.train.train_tokenizer \
    --data.path "$DATA" --out "$OUT/tok" --steps 16000 --seed "$SEED" --resume True
  TOK=$OUT/tok
  stamp "1a tokenizer" "$t"
fi

# --- phase 1b: world model --------------------------------------------------
# Width 512 (8 heads), depth 8, ONE run of 30 000 steps. The bootstrap term ramps
# in at 24 000 with an EMA-copy teacher (trainer defaults); it is what makes
# single-step (K=1) dreams accurate, and with the old online teacher it blew up
# every model wider than 256.
t=$(date +%s)
if ! done_phase "$OUT/dyn"; then
  $PYTHON -m dreamer4.train.train_dynamics \
    --data.path "$DATA" --tokenizer.ckpt "$TOK/checkpoints/latest.pt" \
    --out "$OUT/dyn" --steps 30000 --model.d_model 512 --model.n_heads 8 \
    --seed "$SEED" --resume True
  stamp "1b world model" "$t"
fi
WM=$OUT/dyn/checkpoints/latest.pt

# Dreams take K=1 denoising step when the 39-step K=1 rollout gate holds (player
# error <= 0.1 cell), else K=4.
K=$($PYTHON - "$OUT/dyn/final.json" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))["gate"]
err = [r["player_err"] for r in rows if r["horizon"] == 39 and r["K"] == 1]
print(1 if err and err[0] <= 0.1 else 4)
EOF
)
echo "dreams at K=$K" | tee -a "$OUT/timing.log"

# --- phase 2: agent finetuning, best of two seeds ---------------------------
# Video loss at weight 10 (keeps the world model while the agent tokens learn),
# head MLPs 1024 wide. Cloning quality varies by ~5 points between seeds and
# phase 3 cannot repair features phase 2 got wrong, so two seeds run and the one
# with the higher greedy success on validation layouts (seeds 31337+, n = 500:
# neither training nor held-out) goes on.
t=$(date +%s)
if ! done_phase "$OUT/heads"; then
  best=-1
  pick=""
  for s in "$SEED" $((SEED + 100)); do
    h=$OUT/heads_s$s
    done_phase "$h" || $PYTHON -m dreamer4.train.train_heads \
      --dyn "$WM" --out "$h" --seed "$s" --dyn_weight 10 --d_hidden 1024 \
      --eval_every 1000000 --final_eval_n 200
    $PYTHON -m dreamer4.agent_eval --ckpt "$h/checkpoints/latest.pt" --dyn "$WM" \
      --n 500 --seed 31337 --json "$h/val.json"
    sr=$($PYTHON -c "import json, sys; print(json.load(open(sys.argv[1]))['greedy_sr'])" "$h/val.json")
    if $PYTHON -c "import sys; sys.exit(float(sys.argv[1]) <= float(sys.argv[2]))" "$sr" "$best"; then
      best=$sr
      pick=$h
    fi
  done
  ln -sfn "$(basename "$pick")" "$OUT/heads"
  stamp "2 agent finetuning (kept $(basename "$pick"), validation success $best)" "$t"
fi

# --- phase 3: PMPO in imagination ------------------------------------------
# Rates decay to 3e-5; the |A|-drop ends at 0.5 (a harsher drop lengthens paths);
# policy and value gradients are clipped separately. The in-run evaluation every
# 1 000 updates (validation layouts) keeps best.pt.
t=$(date +%s)
if ! done_phase "$OUT/rl"; then
  PYTORCH_ALLOC_CONF=expandable_segments:True $PYTHON -m dreamer4.train.train_pmpo \
    --dyn "$WM" --init_from "$OUT/heads/checkpoints/latest.pt" --out "$OUT/rl" \
    --seed "$SEED" --steps "$PMPO_STEPS" --K "$K" --lr_final 3e-5 --adv_drop_final 0.5 \
    --clip_split 1 --eval_every 1000 --eval_n 500
  stamp "3 PMPO" "$t"
fi
stamp "this invocation" "$T0"

# --- held-out evaluation of the validation-picked checkpoint ---------------
$PYTHON -m dreamer4.agent_eval --ckpt "$OUT/rl/checkpoints/best.pt" --dyn "$WM" \
  --n 2000 --seed 900000 --json "$OUT/heldout.json"
echo "=== done: $OUT (held-out result in $OUT/heldout.json) ==="
