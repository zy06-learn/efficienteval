#!/usr/bin/env bash
# Main experiment with reselected hyperparameters and end-to-end latency.
# Usage:  run_part1c_full_v1.sh <run_dir> [smoke]
set -uo pipefail

# AFR_ROOT names the repository code root. The default is derived from this script's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
AFR_ROOT="${AFR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

RUN_DIR="${1:?usage: run_part1c_full_v1.sh <run_dir> [smoke]}"
SMOKE="${2:-0}"
PY=${AFR_PYTHON:-python3}
SRC=${AFR_ROOT}/experiments
SCRIPT="$SRC/08_routing_code/part1c_main_full_v1.py"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/04_provenance"
export V3_RUN_DIR="$RUN_DIR"
export V3_SMOKE="$SMOKE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

{
  echo "launcher_pid=$$"
  echo "run_dir=$RUN_DIR"
  echo "smoke=$SMOKE"
  echo "script=$SCRIPT"
  echo "script_sha256=$(sha256sum "$SCRIPT" | cut -d' ' -f1)"
  echo "launcher_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
  echo "started=$(date -Is)"
  echo "host=$(hostname)  nproc=$(nproc)"
} > "$RUN_DIR/04_provenance/LAUNCH.txt"

for stage in preflight hpselect protoB protoA report; do
  if [ -f "$RUN_DIR/logs/$stage.done" ]; then
    echo "[launcher] $stage already done, skipping" | tee -a "$RUN_DIR/logs/launcher.log"
    continue
  fi
  echo "[launcher] $(date -Is) starting $stage" | tee -a "$RUN_DIR/logs/launcher.log"
  stdbuf -oL -eL "$PY" "$SCRIPT" "$stage" > "$RUN_DIR/logs/$stage.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[launcher] $(date -Is) STAGE FAILED: $stage (rc=$rc)" | tee -a "$RUN_DIR/logs/launcher.log"
    echo "{\"status\":\"failed\",\"stage\":\"$stage\",\"rc\":$rc,\"at\":\"$(date -Is)\"}" \
        > "$RUN_DIR/04_provenance/FAILED.json"
    tail -30 "$RUN_DIR/logs/$stage.log" | tee -a "$RUN_DIR/logs/launcher.log"
    exit $rc
  fi
  touch "$RUN_DIR/logs/$stage.done"
  echo "[launcher] $(date -Is) finished $stage" | tee -a "$RUN_DIR/logs/launcher.log"
done

echo "PART1C_ALL_DONE $(date -Is)" | tee -a "$RUN_DIR/logs/launcher.log"
