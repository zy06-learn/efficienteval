#!/usr/bin/env bash
# Invoke via nohup; preserve all prior logs and resume split caches.
set -u
cd /home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3 || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export V3_RUN_DIR=/home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3/runs/p6_table7_complete_20260917_v2
if [[ -f "$V3_RUN_DIR/DONE.json" ]]; then
    echo 'Already complete; nothing to run.'
    exit 0
fi
if [[ -f "$V3_RUN_DIR/PID" ]] && kill -0 "$(cat "$V3_RUN_DIR/PID")" 2>/dev/null; then
    echo 'Recorded process still running; refusing duplicate launch.'
    exit 1
fi
date -u '+%Y-%m-%dT%H:%M:%SZ' >> "$V3_RUN_DIR/LAUNCHES.txt"
printf '%s\n' 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 V3_RUN_DIR=runs/p6_table7_complete_20260917_v2 /home/zeyu/projects/NLP_ilp_dpp_mmr_experiment/.venv/bin/python -u p6_table7_complete.py' >> "$V3_RUN_DIR/LAUNCHES.txt"
/home/zeyu/projects/NLP_ilp_dpp_mmr_experiment/.venv/bin/python -u p6_table7_complete.py &
task_pid=$!
printf '%s\n' "$task_pid" > "$V3_RUN_DIR/PID"
wait "$task_pid"
task_exit=$?
printf '%s\n' "$task_exit" > "$V3_RUN_DIR/EXIT_CODE"
exit "$task_exit"
