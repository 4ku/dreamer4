#!/usr/bin/env bash
# AlohaPick online loop with SMALL real collections: imagination PMPO -> collect a few episodes with the latest policy
# -> refresh the world model on all data (the transformer keeps its video-prediction loss, the reward and value heads
# see the new episodes, the policy clones its best real episodes or is distilled on its newest) -> imagination again.
#   scripts/aloha/dyna_loop.sh <gpu> <tag> <start ckpt (phase 2/3)> <world-model run> [iterations, 0 = no limit]
# Stopping: after the iterations, when the greedy held-out success of the newest policy reaches TARGET_SR in TARGET_N
# consecutive iterations (each one is tested on EVAL_N held-out episodes, results in runs/dyna_<tag>/heldout.tsv), or
# at the next iteration boundary once runs/dyna_<tag>/STOP exists. runs/dyna_<tag>/control.env, if present, is sourced
# at the start of every iteration: a running loop's settings (ITERS, TARGET_SR, EVAL_N, M, ...) can be changed there.
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
RBATCH=${RBATCH:-}                         # refresh clips per step (empty = train_heads' default; a bigger world
                                           # model may need fewer to fit next to the evaluation workers)
HORIZON=${HORIZON:-}                       # imagination horizon; empty = the episode length (phase 3's default)
ONLINE_FRAC=${ONLINE_FRAC:-0}              # share of the refresh's video windows from the online collections
START_FRAC=${START_FRAC:-0}                # share of the dream starts from the online collections
BC_TOP=${BC_TOP:-}                         # refresh cloning: empty = distil the newest collection; Q = clone the
                                           # top Q of all online episodes by return (self-imitation)
if [ -n "$BC_TOP" ]; then BCARGS="--bc_online_top $BC_TOP"; else BCARGS="--bc_last_part 1"; fi
REAL_EVERY=${REAL_EVERY:-}                 # N: every N-th PMPO update learns from the previous collection
COLLECT_T=${COLLECT_T:-1.0}                # sampling temperature of the real collections (> 1: more varied)
PRIOR=${PRIOR:-}                           # FIXED KL-prior checkpoint for every PMPO stage (empty = its warm start)
PLR=${PLR:-1e-3}                           # PMPO learning rate (policy and value) at the start of every stage
BC_LOSS_W=${BC_LOSS_W:-1}                  # weight of the cloning loss in the refresh
DISTILL=${DISTILL:-0}                      # 1: the refresh keeps the policy by matching its pre-refresh action
                                           # distribution instead of cloning recorded actions (BC_TOP ignored)
if [ "$DISTILL" = 1 ]; then BCARGS="--bc_last_part 1 --distill 1"; fi
EXTRA_BASE=${EXTRA_BASE:-}                 # more offline data dirs (comma-separated) next to the world model's own:
                                           # part of the base in every stage, never counted as online
EXTRA_ONLINE=${EXTRA_ONLINE:-}             # offline data dirs counted as ONLINE: the refresh's video windows and the
                                           # dream starts draw from them with the collections (HandOver: 1 024 expert
                                           # grasps 2-12 mm off teach the world model which grasps hold; only then
                                           # does RL in dreams fix the grasp, no lift 0.21 -> 0.08)
PMPO_ONLINE=${PMPO_ONLINE:-0}              # 1: the PMPO stages read only the online parts. They use the data for
                                           # nothing but dream starts, and with START_FRAC 1 every start is online:
                                           # the base's latents (~5 GB for the HandOver base) would only fill the GPU
EVAL_ENVS=${EVAL_ENVS:-64}                 # env worlds of the stages' evaluations (fewer = less GPU memory)
REFRESH_EVAL_N=${REFRESH_EVAL_N:-64}       # the refresh's final real evaluation (0: none and no env workers; the
                                           # held-out EVAL_N measures every policy anyway)
REWARD_OFFSET=${REWARD_OFFSET:-0}          # added to every recorded reward the refresh's reward head learns (dreams
                                           # and RL inherit it; evaluation keeps the env's reward): +0.007 = HandOver
                                           # time cost -0.003 instead of -0.01 per step
RECENT_FRAC=${RECENT_FRAC:-0}              # > 0: that share of the refresh's video windows from the newest
RECENT_PARTS=${RECENT_PARTS:-2}            # RECENT_PARTS collections (the current policy's own episodes; uniform
                                           # sampling leaves them ~5 % next to near-miss and older data)
KD=${KD:-1}                                # denoising steps per dreamed frame in the PMPO stages: with 4 the
                                           # failure-weighted world model dreams more of the risky grasps' failures
                                           # (one policy: held 0.97 -> 0.91, real 0.81; mixed outcomes 19 -> 41 %)
FAIL_W=${FAIL_W:-1}                        # failed episodes' video windows sampled FAIL_W x in the refresh: a world
                                           # model trained on mostly successful grasps dreams every uncertain grasp as
                                           # a success; 3 makes it admit the policy's failures (dreamed held 0.97 ->
                                           # 0.81 for a policy that holds the box in 0.78 of its real episodes)
ACCEPT_TOL=${ACCEPT_TOL:-}                 # with EVAL_N: when an RL stage's policy returns more than ACCEPT_TOL less
                                           # than the policy it started from (both played on the same EVAL_N held-out
                                           # seeds in one run), the stage is REJECTED: it still collects (the world
                                           # model learns from exactly where its dreams were wrong), but the next
                                           # refresh starts from the stage's start checkpoint again. Decisions in
                                           # runs/dyna_<tag>/accept.tsv. Empty = every stage is kept
START_ITER=${START_ITER:-1}                # resume: collections 1 .. START_ITER-1 are reused; a stage whose output
                                           # already exists (rl_i, the collection, wm_i) is skipped
LAST_REFRESH=${LAST_REFRESH:-1}            # 0: no refresh after the last collection (no PMPO stage would use it)
EVAL_N=${EVAL_N:-0}                        # > 0: greedy held-out episodes played by every new policy
TARGET_SR=${TARGET_SR:-}                   # stop once that held-out success is reached TARGET_N times in a row
TARGET_N=${TARGET_N:-2}
CAL_N=${CAL_N:-0}                          # > 0: after every refresh the refreshed policy plays CAL_N whole episodes
                                           # in its world model (dream_eval.py, K = KD, sampled actions); their
                                           # success next to the real success of the collection it learned from goes
                                           # to runs/dyna_<tag>/calib.tsv -- is the world model honest about the policy?
CAL_H=${CAL_H:-}                           # dreamed episode length (empty = the env's time limit)
GATE=${GATE:-}                             # with CAL_N: an iteration runs RL only if the last dreamed success was
                                           # within GATE of the real one; otherwise it keeps the policy and only
                                           # collects + refreshes (an RL stage in a dishonest world model is exploited)
GATE_RET=${GATE_RET:-}                     # the same test on the return (what RL climbs): |dreamed - real| <= GATE_RET
export LC_ALL=C                            # numbers in awk / sort: a ru_RU LC_NUMERIC reads 0.99 as 0
export CUDA_VISIBLE_DEVICES=$GPU
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${REPO:-$HERE/../..}"
stamp() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
WM=$A/runs/$BASE/checkpoints/latest.pt
R=$A/runs/dyna_$TAG
mkdir -p "$R"
if [ -n "${BASE_DATA:-}" ]; then              # e.g. the reward-v2 re-labelling of the world model's own data:
  PARTS=$BASE_DATA                            # its first dir sets the env and the terminal rule of every stage
else
  PARTS=$($PY -c "import torch, sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['config']['data']['path'])" "$WM") \
    || fail "reading $BASE"
fi
FIRST=${PARTS%%,*}
if [ -z "$CAL_H" ]; then
  CAL_H=$($PY -c "import json, sys; print(json.load(open(sys.argv[1] + '/manifest.json'))['env']['max_steps'])" "$FIRST") \
    || fail "reading the time limit of $FIRST"
fi
if [ -n "$EXTRA_BASE" ]; then PARTS=$PARTS,$EXTRA_BASE; fi
NBASE=$(echo "$PARTS" | tr ',' '\n' | wc -l)
if [ -n "$EXTRA_ONLINE" ]; then PARTS=$PARTS,$EXTRA_ONLINE; fi
[ -f "$R/progress.tsv" ] || printf "iter\tepisodes\tsuccess\treturn\tutc\n" > "$R/progress.tsv"
[ -f "$R/heldout.tsv" ] || printf "iter\tsuccess\tsteps_success\treturn\tutc\n" > "$R/heldout.tsv"
[ "$CAL_N" -gt 0 ] && [ ! -f "$R/calib.tsv" ] \
  && printf "iter\treal_success\tdream_success\tgap\tutc\treal_return\tdream_return\treturn_gap\n" > "$R/calib.tsv"

for j in $(seq 1 $((START_ITER - 1))); do PARTS=$PARTS,$A/data/dyna_${TAG}_$j; done
i=$START_ITER
while :; do
  if [ -f "$R/control.env" ]; then . "$R/control.env"; fi
  if [ "$ITERS" -gt 0 ] && [ "$i" -gt "$ITERS" ]; then break; fi
  rl=1
  if [ "$U" = 0 ]; then rl=0; fi
  if [ -n "$GATE" ] && [ "$i" -gt 1 ]; then
    gap=$(awk -F'\t' -v k=$((i - 1)) '$1 == k {g = $4} END {print g}' "$R/calib.tsv" 2>/dev/null)
    if [ -z "$gap" ] || awk -v g="$gap" -v t="$GATE" 'BEGIN {exit !(g > t || g < -t)}'; then rl=0; fi
  fi
  if [ -n "$GATE_RET" ] && [ "$i" -gt 1 ]; then
    rgap=$(awk -F'\t' -v k=$((i - 1)) '$1 == k {g = $8} END {print g}' "$R/calib.tsv" 2>/dev/null)
    if [ -z "$rgap" ] || awk -v g="$rgap" -v t="$GATE_RET" 'BEGIN {exit !(g > t || g < -t)}'; then rl=0; fi
    gap="${gap:-} return ${rgap:-unknown}"
  fi
  if [ "$rl" = 0 ] && [ ! -f "$R/rl_$i/checkpoints/latest.pt" ]; then
    stamp "iteration $i: no RL (U $U, dreamed - real success ${gap:-unknown}, gate ${GATE:-none}): the policy is kept"
    mkdir -p "$R/rl_$i/checkpoints" && cp "$CK" "$R/rl_$i/checkpoints/latest.pt" || fail "keeping the policy $i"
  fi
  stamp "iteration $i: PMPO $U updates from $CK"
  STAGE_START=$CK
  REALARGS=""
  if [ -n "$REAL_EVERY" ] && [ "$i" -gt 1 ]; then
    REALARGS="--real_data $A/data/dyna_${TAG}_$((i - 1)) --real_every $REAL_EVERY"
  fi
  if [ ! -f "$R/rl_$i/checkpoints/latest.pt" ]; then
    PDATA=$PARTS PBASE=$NBASE
    if [ "$PMPO_ONLINE" = 1 ] && [ "$(echo "$PARTS" | tr ',' '\n' | wc -l)" -gt "$NBASE" ]; then
      PDATA=$(echo "$PARTS" | cut -d, -f$((NBASE + 1))-) PBASE=0
    fi
    $PY -m dreamer4.train.train_pmpo --dyn "$WM" --data "$PDATA" --init_from "$CK" --out "$R/rl_$i" --seed "$i" $REALARGS \
      --steps "$U" --K "$KD" --lr "$PLR" --value_lr "$PLR" --lr_final 3e-5 --adv_drop_final 0.5 --clip_split 1 \
      --eval_every 1000000 --eval_n 64 ${PRIOR:+--prior_from "$PRIOR"} \
      --eval_envs "$EVAL_ENVS" --beta "$BETA" ${HORIZON:+--horizon "$HORIZON"} --start_online_frac "$START_FRAC" \
      --base_parts "$PBASE" \
      || fail "PMPO $i"
  fi
  CK=$R/rl_$i/checkpoints/latest.pt
  if [ "$EVAL_N" -gt 0 ] && { [ "$rl" = 1 ] || [ "$i" = 1 ]; } && ! cut -f1 "$R/heldout.tsv" | grep -qx "$i"; then
    stamp "iteration $i: greedy held-out, $EVAL_N episodes"
    PYTHONPATH=. $PY "$HERE/eval_margin.py" --dyn "$WM" --data "$FIRST" --out "$R/rl_$i/heldout.json" --n "$EVAL_N" \
      "rl_$i=$CK" > "$R/rl_$i/heldout.log" 2>&1 || fail "held-out $i"
    $PY -c "import json, sys; r = json.load(open(sys.argv[1]))['results']['rl_' + sys.argv[2]]
print('%s\t%.4f\t%.1f\t%.3f' % (sys.argv[2], r['success'], r['steps_success'], r['return']), end='')" \
      "$R/rl_$i/heldout.json" "$i" >> "$R/heldout.tsv" && printf "\t%s\n" "$(date -u +%H:%M)" >> "$R/heldout.tsv"
    stamp "iteration $i: held-out $(tail -n 1 "$R/heldout.tsv" | cut -f2-4)"
  fi
  verdict=""
  if [ -n "$ACCEPT_TOL" ] && [ "$EVAL_N" -gt 0 ] && [ "$STAGE_START" != "$CK" ] && [ -f "$STAGE_START" ]; then
    [ -f "$R/accept.tsv" ] || printf "iter\tverdict\treturn\tstart_return\tsuccess\tstart_success\tutc\n" > "$R/accept.tsv"
    if ! cut -f1 "$R/accept.tsv" | grep -qx "$i"; then
      # the stage's policy and the policy it started from on the same held-out seeds, in ONE run: two separate
      # 128-episode runs of the same policy differ by up to ~0.2 return (the physics is not bit-exact across batches)
      stamp "iteration $i: acceptance -- rl_$i against its start, $EVAL_N greedy episodes each"
      PYTHONPATH=. $PY "$HERE/eval_margin.py" --dyn "$WM" --data "$FIRST" --out "$R/rl_$i/accept.json" --n "$EVAL_N" \
        "start_$i=$STAGE_START" "rl_$i=$CK" > "$R/rl_$i/accept.log" 2>&1 || fail "acceptance $i"
      $PY -c "import json, sys; r = json.load(open(sys.argv[1]))['results']; i, tol = sys.argv[2], float(sys.argv[3])
c, s = r['rl_' + i], r['start_' + i]
print('%s\t%s\t%.3f\t%.3f\t%.4f\t%.4f' % (i, 'rejected' if c['return'] < s['return'] - tol else 'kept', c['return'],
      s['return'], c['success'], s['success']), end='')" "$R/rl_$i/accept.json" "$i" "$ACCEPT_TOL" >> "$R/accept.tsv" \
        && printf "\t%s\n" "$(date -u +%H:%M)" >> "$R/accept.tsv"
      stamp "iteration $i: return / start return / success / start success $(tail -n 1 "$R/accept.tsv" | cut -f3-6): rl_$i $(tail -n 1 "$R/accept.tsv" | cut -f2)"
    fi
    verdict=$(awk -F'\t' -v k="$i" '$1 == k {v = $2} END {print v}' "$R/accept.tsv")
  fi
  if [ -n "$TARGET_SR" ] && [ "$EVAL_N" -gt 0 ]; then
    hits=$(tail -n "$TARGET_N" "$R/heldout.tsv" | awk -F'\t' -v t="$TARGET_SR" '$1 != "iter" && $2 >= t {k++} END {print k + 0}')
    if [ "$hits" -ge "$TARGET_N" ]; then
      stamp "TARGET REACHED: held-out success >= $TARGET_SR in $TARGET_N iterations in a row"
      break
    fi
  fi

  stamp "iteration $i: collect $M episodes"
  OUT=$A/data/dyna_${TAG}_$i
  if [ ! -f "$OUT/manifest.json" ]; then
    PYTHONPATH=. $PY "$HERE/collect_policy.py" --ckpt "$CK" --dyn "$WM" --base "$FIRST" --out "$OUT" --n "$M" \
      --envs 64 --seed $((SEED + 1000 * i)) --temperature "$COLLECT_T" | tee "$R/collect_$i.log"
    [ -f "$OUT/manifest.json" ] || fail "collection $i"     # the pipe through tee hides the collector's exit code
  fi
  if ! cut -f1 "$R/progress.tsv" | grep -qx "$i"; then
    $PY -c "import json, sys; e = json.load(open(sys.argv[1] + '/manifest.json'))['episodes']
print('%d\t%d\t%.3f\t%.2f' % (int(sys.argv[2]), int(sys.argv[3]) * int(sys.argv[2]),
      sum(x['success'] for x in e) / len(e), sum(x['return'] for x in e) / len(e)), end='')" "$OUT" "$i" "$M" \
      >> "$R/progress.tsv" && printf "\t%s\n" "$(date -u +%H:%M)" >> "$R/progress.tsv"
  fi
  if [ "${verdict:-}" = rejected ]; then CK=$STAGE_START; fi
  PARTS=$PARTS,$OUT
  last=0
  if [ -f "$R/STOP" ] || [ "$i" = "$ITERS" ]; then last=1; fi
  if [ "$last" = 1 ] && [ "$LAST_REFRESH" = 0 ]; then
    break
  fi

  stamp "iteration $i: refresh the world model and heads, $K steps"
  [ -f "$R/wm_$i/checkpoints/latest.pt" ] || $PY -m dreamer4.train.train_heads --dyn "$WM" --data "$PARTS" --init_from "$CK" --out "$R/wm_$i" --seed 1 \
    --steps "$K" ${RBATCH:+--batch "$RBATCH"} --val_frac 0 $BCARGS --bc_loss_weight "$BC_LOSS_W" --dyn_weight 10 --fwd_weight 0 --lr "$RLR" \
    --dyn_lr "$RDLR" --dyn_warmup "$WARM" --dyn_online_frac "$ONLINE_FRAC" --eval_every 1000000 --final_eval_n "$REFRESH_EVAL_N" \
    --eval_envs "$EVAL_ENVS" --base_parts "$NBASE" --dyn_fail_weight "$FAIL_W" \
    --dyn_recent_parts "$RECENT_PARTS" --dyn_recent_frac "$RECENT_FRAC" --reward_offset "$REWARD_OFFSET" \
    || fail "refresh $i"
  CK=$R/wm_$i/checkpoints/latest.pt
  if [ "$CAL_N" -gt 0 ] && ! cut -f1 "$R/calib.tsv" | grep -qx "$i"; then
    stamp "iteration $i: $CAL_N whole episodes of the refreshed policy in its world model"
    # dreams start from the first frames of the episodes this refresh learned from: the policy's own start states
    PYTHONPATH=. $PY "$HERE/dream_eval.py" --dyn "$WM" --data "$OUT" --out "$R/wm_$i/dream_eval.json" --n "$CAL_N" \
      --batch 128 --horizon "$CAL_H" --K "$KD" --seed "$i" "wm_$i=$CK" > "$R/wm_$i/dream_eval.log" 2>&1 \
      || fail "dream eval $i"
    $PY -c "import json, sys; r = json.load(open(sys.argv[1]))['results']['wm_' + sys.argv[2]]
e = json.load(open(sys.argv[3] + '/manifest.json'))['episodes']; n = len(e)
s, ret = sum(x['success'] for x in e) / n, sum(x['return'] for x in e) / n
d, dr = r['dream_success'], r['dream_return']
print('%s\t%.3f\t%.3f\t%.3f\t%s\t%.3f\t%.3f\t%.3f' % (sys.argv[2], s, d, d - s, sys.argv[4], ret, dr, dr - ret))" \
      "$R/wm_$i/dream_eval.json" "$i" "$OUT" "$(date -u +%H:%M)" >> "$R/calib.tsv" || fail "calibration row $i"
    stamp "iteration $i: real vs dreamed success $(tail -n 1 "$R/calib.tsv" | cut -f2-4), return $(tail -n 1 "$R/calib.tsv" | cut -f6-8)"
  fi
  if [ "$last" = 1 ]; then break; fi
  i=$((i + 1))
done
stamp "LOOP DONE"
