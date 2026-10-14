#!/usr/bin/env bash
# Core-layer dendrite experiment: blocks.5 FFN gate (the c_fc analogue) and down
# (the c_proj analogue). Open-source dendrites first, then Perforated
# Backpropagation if PAIEMAIL and PAITOKEN are set. Compare against the existing
# no-dendrite control runs/pai-full-ctl-ple-ctlpb-s0.json (24 epochs, same seed);
# a new control is run only if a dendrite run goes longer than that.
# Log: runs/core-experiment.log
#
#   caffeinate -i bash scripts/run_pai_core_experiment.sh
#   # for the PB part: export PAIEMAIL=... PAITOKEN=... first (never commit them)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs

PY=.venv-pai/bin/python
SCRIPT=research/tinystories/train_pai_full.py
TARGETS=".blocks.5.ffn.gate,.blocks.5.ffn.down"
COMMON=(--arm ple --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128
        --target-core 560000 --batch-size 16 --seq-len 256 --seed 0
        --targets "$TARGETS")
CTL=runs/pai-full-ctl-ple-ctlpb-s0.json

run() { PYTHONPATH=src "$PY" "$SCRIPT" "${COMMON[@]}" "$@"; }
epochs_of() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['epochs_run'])" "$1"; }

if [ ! -f "$CTL" ]; then
  echo "missing $CTL (the 24-epoch control); run scripts/run_pai_pb_experiment.sh first" >&2
  exit 1
fi
CTL_EPOCHS=$(epochs_of "$CTL")

{
  echo "=== 1/2 open-source dendrites on $TARGETS ==="
  run --tag core
  N1=$(epochs_of runs/pai-full-pai-ple-core-s0.json)

  N2=0
  if [ -n "${PAIEMAIL:-}" ] && [ -n "${PAITOKEN:-}" ]; then
    echo "=== 2/2 Perforated Backpropagation on $TARGETS ==="
    run --pb --tag corepb
    N2=$(epochs_of runs/pai-full-pai-ple-corepb-s0.json)
  else
    echo "=== 2/2 skipped: PAIEMAIL / PAITOKEN not set ==="
  fi

  NEED=$(( N1 > N2 ? N1 : N2 ))
  if [ "$NEED" -gt "$CTL_EPOCHS" ]; then
    echo "=== extra control: dendrite runs took $NEED epochs, existing control has $CTL_EPOCHS ==="
    run --tag ctlcore --control --control-epochs "$NEED"
  fi
  echo "=== all done ==="
} 2>&1 | tee runs/core-experiment.log
