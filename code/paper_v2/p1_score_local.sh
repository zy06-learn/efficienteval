#!/usr/bin/env bash
# Stage 2a: the nine locally-hosted verifiers, scored on the P1 cohort.
# The six vLLM-served verifiers need a running server and are handled by p1_api.sh.
#
#   AFR_ROOT         repository code root (default: the parent of this file's directory)
#   AFR_PYTHON       interpreter with requirements-verifiers.txt installed (default: python3)
#   P1_SCORING_DIR   where scores are written (default: paper_v2/results/p1_scoring)
#
# Each verifier writes a status/<name>.done marker, so an interrupted run resumes rather
# than rescoring what already finished.
set -u
ROOT=${AFR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${AFR_PYTHON:-python3}
V2=$ROOT/paper_v2
CLI=$V2/ingest/verifier_cli.py
DIR=${P1_SCORING_DIR:-$V2/results/p1_scoring}
export PYTHONPATH=$ROOT:$V2${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$DIR"
cd "$V2" || exit 1

echo "=== prepare $(date +%H:%M:%S) ==="
$PY "$CLI" prepare --cohort "$V2/data/P1_SCORING_COHORT.parquet" --result-dir "$DIR" || exit 1

for v in factcc factkb lettuce_v2 hhem wecheck factcg minicheck_dbta minicheck_ft5 alignscore; do
  if [ -f "$DIR/status/${v}.done" ]; then echo "=== $v already done, skip ==="; continue; fi
  echo "=== $v start $(date +%H:%M:%S) ==="
  $PY "$CLI" score --verifier "$v" --result-dir "$DIR" --device cuda
  rc=$?
  echo "=== $v end rc=$rc $(date +%H:%M:%S) ==="
  [ $rc -eq 0 ] && mkdir -p "$DIR/status" && touch "$DIR/status/${v}.done"
done
echo "=== LOCAL VERIFIERS COMPLETE $(date +%H:%M:%S) ==="
