#!/usr/bin/env bash
# Chain for exp1_paperconfig_20260929.py. Usage: run_exp1_paperconfig_20260929.sh
set -euo pipefail
PY=/home/zeyu/projects/NLP_ilp_dpp_mmr_experiment/.venv/bin/python
SRC=/home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3
S=$SRC/exp1_paperconfig_20260929.py
R=$SRC/runs/exp1_paperconfig_20260929
L=$R/logs
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 EXP1P_RUN_DIR=$R
cd $SRC
echo "launch $(date -u +%FT%TZ) script_sha256=$(sha256sum $S | cut -d' ' -f1)" >> $R/FREEZE_STAMP.txt
$PY $S gate > $L/gate.log 2>&1
screen_seed () {
  $PY $S screen $1 0 2 > $L/screen_seed$1_0.log 2>&1 &
  $PY $S screen $1 1 2 > $L/screen_seed$1_1.log 2>&1 &
  wait
}
screen_seed 17
$PY $S fixed > $L/fixed.log 2>&1
$PY $S finals > $L/finals.log 2>&1
$PY $S apply > $L/apply_primary.log 2>&1
screen_seed 29
screen_seed 43
$PY $S apply > $L/apply.log 2>&1
echo COMPLETE > $L/DONE
