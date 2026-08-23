#!/usr/bin/env bash
# Single entry point for this release. Every stage reads AFR_ROOT and AFR_INPUTS, which are
# set here from the location of this script, so the repository can live anywhere.
#
#   ./reproduce.sh verify     manifest check + the bit-for-bit reference gate   (CPU, <1 min)
#   ./reproduce.sh main       stage 3, main experiment, both protocols          (CPU, hours)
#   ./reproduce.sh ablation   stage 3, core ablations
#   ./reproduce.sh extended   stage 3, extended ablations and per-corpus tables
#   ./reproduce.sh cascade    stage 3, cascade and alternative learners
#   ./reproduce.sh controls   stage 4, dataset control, few-shot curve, significance
#
# Stages 1 and 2 (corpus ingest and verifier scoring) need the corpora, GPU model weights,
# and a running vLLM server. They are documented in docs/reproducibility.md rather than
# wired in here, because they cannot run from this repository alone.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AFR_ROOT="$REPO/code"
export AFR_INPUTS="$AFR_ROOT/paper_v3/DELIVERABLE/00_inputs"
export AFR_PYTHON="${AFR_PYTHON:-$(command -v python3)}"
SCRIPTS="$AFR_ROOT/paper_v3/DELIVERABLE/08_scripts"
CONTROLS="$AFR_ROOT/paper_v3/DELIVERABLE/09_live_and_controls/code"
RUNS="$REPO/runs"

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 1 ] || usage

case "$1" in
  verify)
    exec "$AFR_PYTHON" -m pytest "$REPO/tests/test_reproduction.py" -v -s
    ;;
  main)
    exec bash "$SCRIPTS/run_part1c_full_v1.sh" "$RUNS/part1c_main_full_v1" "${2:-0}"
    ;;
  ablation)
    exec bash "$SCRIPTS/run_part2_ablation_v1.sh" "$RUNS/part2_ablation_v1" "${2:-0}"
    ;;
  extended)
    exec bash "$SCRIPTS/run_part3_v1.sh" "$RUNS/part3_extended_v1" "${2:-0}"
    ;;
  cascade)
    exec bash "$SCRIPTS/chain_part4.sh"
    ;;
  controls)
    # The dataset control and the few-shot curve are CPU only. The live Protocol B re-run
    # needs a GPU and a vLLM server; see docs/reproducibility.md for that one.
    mkdir -p "$RUNS/controls"
    export V3_RUN_DIR="$RUNS/controls"
    for stage in dataset_control ds_only fewshot fewshot_k0 sig_main; do
      echo "[reproduce] $stage"
      "$AFR_PYTHON" "$CONTROLS/$stage.py"
    done
    ;;
  *)
    usage
    ;;
esac
