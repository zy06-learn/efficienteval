#!/usr/bin/env bash
# Part 2 ablation, both protocols. Usage: run_part2_ablation_v1.sh <run_dir> [smoke]
set -uo pipefail

RUN_DIR="${1:?usage: run_part2_ablation_v1.sh <run_dir> [smoke]}"
SMOKE="${2:-0}"
PY=${AFR_PYTHON:-python3}
SCRIPT=${AFR_ROOT:-/home/zeyu/projects/adaptive-faithfulness-router-v2}/paper_v3/DELIVERABLE/08_scripts/part2_ablation_v1.py

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/04_provenance"
export V3_RUN_DIR="$RUN_DIR" V3_SMOKE="$SMOKE" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""

{
  echo "launcher_pid=$$"
  echo "run_dir=$RUN_DIR  smoke=$SMOKE"
  echo "script_sha256=$(sha256sum "$SCRIPT" | cut -d' ' -f1)"
  echo "launcher_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
  echo "started=$(date -Is)  host=$(hostname)  nproc=$(nproc)"
} > "$RUN_DIR/04_provenance/LAUNCH.txt"

for stage in protoB protoA report; do
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

echo "PART2_ALL_DONE $(date -Is)" | tee -a "$RUN_DIR/logs/launcher.log"
