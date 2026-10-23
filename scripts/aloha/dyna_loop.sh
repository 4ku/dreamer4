#!/usr/bin/env bash
# AlohaPick online loop with SMALL real collections: imagination PMPO -> collect a few episodes with the latest policy
# -> refresh the world model on all data (the transformer keeps its video-prediction loss, the reward and value heads
# see the new episodes, the policy clones its best real episodes or is distilled on its newest) -> imagination again.
#   scripts/aloha/dyna_loop.sh <gpu> <tag> <start ckpt (phase 2/3)> <world-model run> [iterations]
# recipe.sh runs it with the settings that worked (HORIZON 32, ONLINE_FRAC 0.5, START_FRAC 0.5, K 1000, BC_TOP 0.25,
# REAL_EVERY 50); the defaults here are the plain loop.
# Env: A (workspace with data/ and runs/), PY (the dreamer4 venv's python), REPO (this repository, default: the
# checkout this script lives in), U = PMPO updates per iteration (250), M = episodes collected per iteration (128),
# K = refresh steps (500), SEED = first collection seed (3000000), BETA (0.1); the refresh runs at the gentle rates
# RLR (heads, 1e-4) / RDLR (transformer, 5e-5) with a WARM-step ramp (100): at phase 2's rates a fresh optimizer
# jolts the trained policy (10 steps: sampled success 0.88 -> 0.61; 300 gentle steps: 0.88 -> 0.84).
# Writes runs/dyna_<tag>/{rl_<i>,wm_<i>} and data/dyna_<tag>_<i>; one summary line per iteration in
# runs/dyna_<tag>/progress.tsv (iteration, episodes so far, success and return of the collected episodes).
set -u
A=${A:-/mnt/dsk1/iefremov/aloha}
PY=${PY:-/mnt/dsk1/iefremov/dreamer/dreamer4/.venv/bin/python}
export ALOHA_PYTHON=$A/mujoco_playground/.venv/bin/python PYTORCH_ALLOC_CONF=expandable_segments:True
GPU=$1 TAG=$2 CK=$3 BASE=$4 ITERS=${5:-12}
U=${U:-250} M=${M:-128} K=${K:-500} SEED=${SEED:-3000000} BETA=${BETA:-0.1}
RLR=${RLR:-1e-4} RDLR=${RDLR:-5e-5} WARM=${WARM:-100}
HORIZON=${HORIZON:-}                       # imagination horizon; empty = the episode length (phase 3's default)
ONLINE_FRAC=${ONLINE_FRAC:-0}              # share of the refresh's video windows from the online collections
START_FRAC=${START_FRAC:-0}                # share of the dream starts from the online collections
BC_TOP=${BC_TOP:-}                         # refresh cloning: empty = distil the newest collection; Q = clone the
                                           # top Q of all online episodes by return (self-imitation)
if [ -n "$BC_TOP" ]; then BCARGS="--bc_online_top $BC_TOP"; else BCARGS="--bc_last_part 1"; fi
REAL_EVERY=${REAL_EVERY:-}                 # N: every N-th PMPO update learns from the previous collection
START_ITER=${START_ITER:-1}                # resume: collections 1 .. START_ITER-1 are reused; a stage whose output
                                           # already exists (rl_i, the collection, wm_i) is skipped
export CUDA_VISIBLE_DEVICES=$GPU
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${REPO:-$HERE/../..}"
stamp() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
WM=$A/runs/$BASE/checkpoints/latest.pt
R=$A/runs/dyna_$TAG
mkdir -p "$R"
PARTS=$($PY -c "import torch, sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['config']['data']['path'])" "$WM") \
  || fail "reading $BASE"
FIRST=${PARTS%%,*}
[ -f "$R/progress.tsv" ] || printf "iter\tepisodes\tsuccess\treturn\tutc\n" > "$R/progress.tsv"

for j in $(seq 1 $((START_ITER - 1))); do PARTS=$PARTS,$A/data/dyna_${TAG}_$j; done
for i in $(seq "$START_ITER" "$ITERS"); do
  stamp "iteration $i: PMPO $U updates from $CK"
  REALARGS=""
  if [ -n "$REAL_EVERY" ] && [ "$i" -gt 1 ]; then
    REALARGS="--real_data $A/data/dyna_${TAG}_$((i - 1)) --real_every $REAL_EVERY"
  fi
  if [ ! -f "$R/rl_$i/checkpoints/latest.pt" ]; then
    $PY -m dreamer4.train.train_pmpo --dyn "$WM" --data "$PARTS" --init_from "$CK" --out "$R/rl_$i" --seed "$i" $REALARGS \
      --steps "$U" --K 1 --lr_final 3e-5 --adv_drop_final 0.5 --clip_split 1 --eval_every 1000000 --eval_n 64 \
      --eval_envs 64 --beta "$BETA" ${HORIZON:+--horizon "$HORIZON"} --start_online_frac "$START_FRAC" \
      || fail "PMPO $i"
  fi
  CK=$R/rl_$i/checkpoints/latest.pt

  stamp "iteration $i: collect $M episodes"
  OUT=$A/data/dyna_${TAG}_$i
  if [ ! -f "$OUT/manifest.json" ]; then
    PYTHONPATH=. $PY "$HERE/collect_policy.py" --ckpt "$CK" --dyn "$WM" --base "$FIRST" --out "$OUT" --n "$M" \
      --envs 64 --seed $((SEED + 1000 * i)) | tee "$R/collect_$i.log" || fail "collection $i"
  fi
  if ! cut -f1 "$R/progress.tsv" | grep -qx "$i"; then
    $PY -c "import json, sys; e = json.load(open(sys.argv[1] + '/manifest.json'))['episodes']
print('%d\t%d\t%.3f\t%.2f' % (int(sys.argv[2]), int(sys.argv[3]) * int(sys.argv[2]),
      sum(x['success'] for x in e) / len(e), sum(x['return'] for x in e) / len(e)), end='')" "$OUT" "$i" "$M" \
      >> "$R/progress.tsv" && printf "\t%s\n" "$(date -u +%H:%M)" >> "$R/progress.tsv"
  fi
  PARTS=$PARTS,$OUT

  stamp "iteration $i: refresh the world model and heads, $K steps"
  [ -f "$R/wm_$i/checkpoints/latest.pt" ] || $PY -m dreamer4.train.train_heads --dyn "$WM" --data "$PARTS" --init_from "$CK" --out "$R/wm_$i" --seed 1 \
    --steps "$K" --val_frac 0 $BCARGS --dyn_weight 10 --fwd_weight 0 --lr "$RLR" --dyn_lr "$RDLR" \
    --dyn_warmup "$WARM" --dyn_online_frac "$ONLINE_FRAC" --eval_every 1000000 --final_eval_n 64 --eval_envs 64 \
    || fail "refresh $i"
  CK=$R/wm_$i/checkpoints/latest.pt
done
stamp "LOOP DONE"
