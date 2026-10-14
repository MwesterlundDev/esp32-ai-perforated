#!/usr/bin/env bash
# Head-dendrite experiment: capacity test, real run, then a control matched to the
# real run's length. Stops at the first failure. Log: runs/full-experiment.log
#
#   caffeinate -i bash scripts/run_pai_full_experiment.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs

PY=.venv-pai/bin/python
SCRIPT=research/tinystories/train_pai_full.py
COMMON=(--arm ple --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128
        --target-core 560000 --batch-size 16 --seq-len 256 --seed 0)

run() { PYTHONPATH=src "$PY" "$SCRIPT" "${COMMON[@]}" "$@"; }

{
  echo "=== 1/3 capacity test ==="
  run --tag cap --capacity-test

  echo "=== 2/3 real run (dendrite on the head) ==="
  run --tag head

  EPOCHS=$("$PY" -c "import json;print(json.load(open('runs/pai-full-pai-ple-head-s0.json'))['epochs_run'])")
  echo "=== 3/3 control, matched to $EPOCHS epochs ==="
  run --tag ctl --control --control-epochs "$EPOCHS"

  echo "=== all done ==="
} 2>&1 | tee runs/full-experiment.log
