#!/usr/bin/env bash
# Held-out evaluations (n = 512, seeds 900000+, greedy and sampled) of AlohaPick policies, one after another.
#   scripts/aloha/heldout_queue.sh <gpu> <name>=<ckpt>@<world model> ...
# Each writes $A/runs/heldout/<name>.json; an existing result is skipped. Only these numbers rank policies: the
# in-run evaluations (n = 128, seeds 31337+) run 2-5 points optimistic.
set -u
A=${A:-/mnt/dsk1/iefremov/aloha}
PY=${PY:-/mnt/dsk1/iefremov/dreamer/dreamer4/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=$1 ALOHA_PYTHON=$A/mujoco_playground/.venv/bin/python
shift
mkdir -p "$A/runs/heldout"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${REPO:-$HERE/../..}"
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; ck=${rest%@*}; dyn=${rest#*@}
  out=$A/runs/heldout/$name.json
  [ -f "$out" ] && continue
  echo "[$(date -u +%H:%M:%S) UTC] $name"
  $PY -m dreamer4.agent_eval --ckpt "$ck" --dyn "$dyn" --n 512 --seed 900000 --envs 64 --json "$out" 2>&1 \
    | grep -v -i "warn\|Module" | tail -1
done
echo "[$(date -u +%H:%M:%S) UTC] QUEUE DONE"
