#!/usr/bin/env bash
# Head-dendrite experiment with Perforated Backpropagation (needs a license).
# Real run, then a control matched to its length. Log: runs/pb-experiment.log
#
#   export PAIEMAIL=...     # your perforatedai.com account email
#   export PAITOKEN=...     # your token; never commit it or paste it into a chat
#   caffeinate -i bash scripts/run_pai_pb_experiment.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs

if [ -z "${PAIEMAIL:-}" ] || [ -z "${PAITOKEN:-}" ]; then
  echo "PAIEMAIL and PAITOKEN are not set. Export them in this shell first (see the" >&2
  echo "header of this script); the run would otherwise stop at a license prompt." >&2
  exit 1
fi

PY=.venv-pai/bin/python
SCRIPT=research/tinystories/train_pai_full.py
COMMON=(--arm ple --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128
        --target-core 560000 --batch-size 16 --seq-len 256 --seed 0)

run() { PYTHONPATH=src "$PY" "$SCRIPT" "${COMMON[@]}" "$@"; }

{
  echo "=== 1/2 Perforated Backpropagation, dendrite on the head ==="
  run --pb --tag pb

  EPOCHS=$("$PY" -c "import json;print(json.load(open('runs/pai-full-pai-ple-pb-s0.json'))['epochs_run'])")
  echo "=== 2/2 control, matched to $EPOCHS epochs ==="
  run --tag ctlpb --control --control-epochs "$EPOCHS"

  echo "=== all done ==="
} 2>&1 | tee runs/pb-experiment.log
