#!/usr/bin/env bash
# Wait for Part 3 phase 2 to finish, then run Part 4 in its own run directory.
# Part 4 never writes into Part 3's tree; the only thing it reads from elsewhere is the frozen
# reference contract. Usage: chain_part4.sh
set -uo pipefail

BASE=${AFR_ROOT:-/home/zeyu/projects/adaptive-faithfulness-router-v2}/experiments
P3=$BASE/runs/part3_extended_v1
P4=$BASE/runs/part4_cascade_v1
PY=${AFR_PYTHON:-python3}
SCRIPT=$BASE/08_routing_code/part4_cascade_v1.py

mkdir -p "$P4/logs" "$P4/results" "$P4/04_provenance"
CHAIN_LOG="$P4/logs/chain.log"
echo "[chain] $(date -Is) waiting for part3 phase2" > "$CHAIN_LOG"

# poll for the phase-2 completion marker, with a hard ceiling so this can never hang forever
DEADLINE=$(( $(date +%s) + 8*3600 ))
while true; do
  if grep -q "PART3_phase2_DONE" "$P3/logs/launcher.log" 2>/dev/null; then
    echo "[chain] $(date -Is) part3 phase2 finished, starting part4" >> "$CHAIN_LOG"
    break
  fi
  if grep -q "STAGE FAILED" "$P3/logs/launcher.log" 2>/dev/null; then
    echo "[chain] $(date -Is) part3 phase2 FAILED; running part4 anyway (independent)" >> "$CHAIN_LOG"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[chain] $(date -Is) deadline reached without a marker; starting part4" >> "$CHAIN_LOG"
    break
  fi
  sleep 120
done

export V3_RUN_DIR="$P4" V3_SMOKE=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""

{
  echo "chain_pid=$$"
  echo "run_dir=$P4"
  echo "script_sha256=$(sha256sum "$SCRIPT" | cut -d' ' -f1)"
  echo "chain_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
  echo "started=$(date -Is)  host=$(hostname)  nproc=$(nproc)"
} > "$P4/04_provenance/LAUNCH.txt"

for stage in protoB reportB protoA reportA; do
  if [ -f "$P4/logs/$stage.done" ]; then
    echo "[chain] $stage already done, skipping" >> "$CHAIN_LOG"
    continue
  fi
  echo "[chain] $(date -Is) starting $stage" >> "$CHAIN_LOG"
  stdbuf -oL -eL "$PY" "$SCRIPT" "$stage" > "$P4/logs/$stage.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[chain] $(date -Is) STAGE FAILED: $stage (rc=$rc)" >> "$CHAIN_LOG"
    tail -30 "$P4/logs/$stage.log" >> "$CHAIN_LOG"
    exit $rc
  fi
  touch "$P4/logs/$stage.done"
  echo "[chain] $(date -Is) finished $stage" >> "$CHAIN_LOG"
done

echo "PART4_ALL_DONE $(date -Is)" >> "$CHAIN_LOG"
