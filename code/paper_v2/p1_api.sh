#!/usr/bin/env bash
# Stage 2b: the six vLLM-served verifiers. One model is loaded at a time; qwen30_fast and
# qwen30_judge share weights, so they are scored back to back under a single load.
#
# The serving flags are part of the measurement, not incidental. --max-num-seqs 1 makes the
# reported latency strict batch-1, --max-model-len 16384 fixes the context budget every
# served verifier is measured under, and --enable-prefix-caching is what the paper's
# non-determinism note refers to: re-running a row can hit a warm prefix and return a score
# that differs in the last digits.
#
#   AFR_ROOT      repository code root (default: the parent of this file's directory)
#   AFR_PYTHON    interpreter with requirements-verifiers.txt installed (default: python3)
#   HF_HUB        HuggingFace hub cache holding the model snapshots
#   VLLM_API      OpenAI-compatible endpoint the server is brought up on
set -u
ROOT=${AFR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${AFR_PYTHON:-python3}
V2=$ROOT/paper_v2
HUB=${HF_HUB:-$HOME/.cache/huggingface/hub}
API=${VLLM_API:-http://127.0.0.1:8001/v1}
PORT=$(printf '%s' "$API" | sed -E 's|.*:([0-9]+)/.*|\1|')
SERVED=unified-summary-verifier
DIR=${P1_SCORING_DIR:-$V2/results/p1_scoring}
export PYTHONPATH=$ROOT:$V2${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$V2/logs" "$DIR"
cd "$V2" || exit 1

snap () { ls -d "$HUB/$1"/snapshots/*/ 2>/dev/null | head -1; }

# Both batches would otherwise contend for the single GB10: a vLLM server sized with
# --gpu-memory-utilization cannot coexist with an encoder that is already holding memory.
echo "=== waiting for the local-model batch to finish $(date +%H:%M:%S) ==="
while ps -eo cmd --no-headers | grep -q "[v]erifier_cli.py score"; do sleep 30; done
echo "=== local batch done, starting the vLLM sequence $(date +%H:%M:%S) ==="

serve_and_score () {
  local repo="$1"; shift
  local verifiers="$1"; shift
  local util="${1:-0.85}"
  local model; model="$(snap "$repo")"
  if [ -z "$model" ]; then echo "!!! snapshot not found for $repo under $HUB"; return 1; fi

  local all_done=1
  for v in ${verifiers//,/ }; do
    [ -f "$DIR/status/${v}.done" ] || all_done=0
  done
  if [ "$all_done" = "1" ]; then echo "=== $verifiers already done, skip ==="; return 0; fi

  echo "=== serving $repo  ($verifiers)  $(date +%H:%M:%S) ==="
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$model" --served-model-name "$SERVED" \
      --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization "$util" --max-model-len 16384 --max-num-seqs 1 \
      --enable-prefix-caching > "$V2/logs/vllm_$(basename "$repo").log" 2>&1 &
  local pid=$!
  local ready=0
  for _ in $(seq 1 360); do
    curl -fsS "$API/models" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$pid" 2>/dev/null || { echo "!!! server exited early for $repo"; break; }
    sleep 5
  done
  if [ "$ready" != "1" ]; then
    echo "!!! $repo never became ready; see logs/vllm_$(basename "$repo").log"
    kill "$pid" 2>/dev/null; sleep 10; return 1
  fi
  echo "=== ready, scoring $verifiers $(date +%H:%M:%S) ==="
  TOKENIZER_PATH="$model" "$PY" p1_score.py "$verifiers" "$API"
  echo "=== stopping the server $(date +%H:%M:%S) ==="
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 "$pid" 2>/dev/null
  sleep 20
}

# The order below is the order the study ran in. A repeated line is a no-op on a resumed run,
# because serve_and_score skips a verifier whose status marker already exists.
serve_and_score models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8                     qwen30_fast                        0.90
serve_and_score models--ibm-granite--granite-guardian-3.2-3b-a800m                granite_guardian_3_2_3b_a800m      0.60
# granite_guardian_3_1_2b is the third member of the frozen pool. Its scores were produced in
# the earlier unified-v1 round under a separate launcher that is not part of this release; the
# entry is restated here so that stage 2 covers all six served verifiers under one contract.
# Pinned revision: ibm-granite/granite-guardian-3.1-2b@81145486e85c6c82c01e759c0356d9d6da4d21a5
serve_and_score models--ibm-granite--granite-guardian-3.1-2b                      granite_guardian_3_1_2b            0.60
serve_and_score models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8                     qwen30_fast,qwen30_judge           0.90
serve_and_score models--ibm-granite--granite-guardian-3.2-8b-factuality-detection granite_guardian_3_2_8b_factuality 0.85
# No --enable-lora flag is passed here, and none was passed in the study. This arm therefore
# ran on base granite-4.1-3b weights with no adapter attached, which is why the paper reports
# it as Granite-4.1-3B rather than as a factuality LoRA, and why its AUROC sits below chance.
# The verifier key keeps its original spelling because the frozen score files are named after
# it; renaming it here would silently detach this arm from its own scores.
serve_and_score models--ibm-granite--granite-4.1-3b       granite_guardian_4_1_3b_factuality_lora 0.60

echo "=== API VERIFIERS COMPLETE $(date +%H:%M:%S) ==="
ls "$DIR/status/" | tr '\n' ' '
