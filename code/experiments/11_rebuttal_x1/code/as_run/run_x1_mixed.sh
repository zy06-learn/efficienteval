#!/usr/bin/env bash
# Usage: run_x1_bestfixed.sh <run_dir> [smoke]
set -euo pipefail
cd /home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export V3_RUN_DIR="$1"
[ "${2:-}" = smoke ] && export X1_SMOKE=1
mkdir -p "$V3_RUN_DIR"
{ date; hostname; git -C .. rev-parse HEAD 2>/dev/null || true; git -C .. status --short 2>/dev/null | head -20 || true;
  echo "cmd: $0 $*"; sha256sum x1_mixed_at_k.py; } > "$V3_RUN_DIR/RUN_INFO.txt"
/home/zeyu/projects/NLP_ilp_dpp_mmr_experiment/.venv/bin/python -u x1_mixed_at_k.py
echo "EXIT=$?"
