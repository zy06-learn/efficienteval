#!/usr/bin/env bash
# Single entry point for this release. Every stage reads AFR_ROOT and AFR_INPUTS, which are
# set here from the location of this script, so the repository can live anywhere.
#
#   ./reproduce.sh verify     manifest check + the bit-for-bit reference gate   (CPU, <1 min)
#   ./reproduce.sh main       stage 3, main experiment, both protocols          (CPU, hours)
#   ./reproduce.sh ablation   stage 3, core ablations
#   ./reproduce.sh extended <phase1|phase2> [smoke]
#                             stage 3, extended ablations and per-corpus tables
#   ./reproduce.sh cascade    stage 3, cascade and alternative learners
#   ./reproduce.sh controls   stage 4, dataset control, few-shot curve, control intervals
#
# Stages 1 and 2 (corpus ingest and verifier scoring) need the corpora, GPU model weights,
# and a running vLLM server. They are documented in docs/reproducibility.md rather than
# wired in here, because they cannot run from this repository alone.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AFR_ROOT="$REPO/code"
export AFR_INPUTS="$AFR_ROOT/experiments/00_inputs"
export AFR_PYTHON="${AFR_PYTHON:-$(command -v python3)}"
SCRIPTS="$AFR_ROOT/experiments/08_routing_code"
CONTROLS="$AFR_ROOT/experiments/09_live_and_controls/code"
# chain_part4.sh looks for Part 3 under $AFR_ROOT/experiments/runs, so every stage writes there;
# a run directory below the repository root would leave cascade waiting for a marker that
# never appears.
RUNS="$AFR_ROOT/experiments/runs"

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
    # run_part3_v1.sh takes <run_dir> <phase1|phase2> [smoke]; there is no default phase.
    case "${2:-}" in
      phase1|phase2) ;;
      *) echo "usage: ./reproduce.sh extended <phase1|phase2> [smoke]" >&2; exit 1 ;;
    esac
    exec bash "$SCRIPTS/run_part3_v1.sh" "$RUNS/part3_extended_v1" "$2" "${3:-0}"
    ;;
  cascade)
    exec bash "$SCRIPTS/chain_part4.sh"
    ;;
  controls)
    # The dataset control and the few-shot curve are CPU only. The live Protocol B re-run
    # needs a GPU and a vLLM server; see docs/reproducibility.md for that one.
    mkdir -p "$RUNS/controls"
    export V3_RUN_DIR="$RUNS/controls"
    export SIG_OUT="$RUNS/controls" FS_OUT="$RUNS/controls"
    # sig_main covers the main tables; sig_controls covers the three control experiments.
    # fewshot_frac is the fraction grid the paper plots; fewshot and fewshot_k0 are the
    # superseded absolute-count sweep, kept because the archived curve came from them.
    for stage in dataset_control ds_only fewshot_frac sig_main sig_controls; do
      echo "[reproduce] $stage"
      "$AFR_PYTHON" "$CONTROLS/$stage.py"
    done
    ;;
  *)
    usage
    ;;
esac
