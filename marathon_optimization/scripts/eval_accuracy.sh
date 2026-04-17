#!/usr/bin/env bash
set -euo pipefail

# eval_accuracy.sh — deterministic GSM8K accuracy gate for marathon harness
#
# The orchestrator passes these env vars:
#   PORT        — inference server port (default 8888)
#   MODEL       — model path or name (e.g. /shared_nfs/models/GLM-5-FP8)
#   RESULTS_DIR — where to write eval artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_DIR="${SCRIPT_DIR}/../InferenceX"

if [[ ! -f "${INFERENCEX_DIR}/benchmarks/benchmark_lib.sh" ]]; then
    echo "ERROR: benchmark_lib.sh not found at ${INFERENCEX_DIR}/benchmarks/benchmark_lib.sh" >&2
    exit 1
fi

source "${INFERENCEX_DIR}/benchmarks/benchmark_lib.sh"

export PORT="${PORT:-8888}"
export MODEL_NAME="${MODEL:-}"
export MODEL="${MODEL:-}"
export EVAL_TASK="gsm8k"
export EVAL_RESULT_DIR="${RESULTS_DIR:-$(mktemp -d /tmp/eval_accuracy-XXXXXX)}"
export NUM_FEWSHOT=2

# Verify server is reachable before running eval
if ! curl --output /dev/null --silent --fail "http://0.0.0.0:${PORT}/health"; then
    echo "ERROR: Server not healthy at port ${PORT}" >&2
    exit 1
fi

echo "Running GSM8K accuracy evaluation on port ${PORT}, model ${MODEL_NAME}..."
run_lm_eval --port "${PORT}" --task gsm8k --num-fewshot "${NUM_FEWSHOT}" --results-dir "${EVAL_RESULT_DIR}"
eval_exit=$?

if [[ $eval_exit -ne 0 ]]; then
    echo "ERROR: lm_eval exited with code ${eval_exit}" >&2
    exit $eval_exit
fi

# Parse and print accuracy from lm_eval JSON results for the orchestrator to consume.
# The orchestrator regex looks for: gsm8k.*acc...|<number> OR accuracy <number>
RESULTS_JSON=$(find "${EVAL_RESULT_DIR}" -name "results*.json" -type f 2>/dev/null | head -1)
if [[ -n "${RESULTS_JSON}" ]]; then
    python3 -c "
import json, sys
data = json.load(open('${RESULTS_JSON}'))
results = data.get('results', {})
for task, metrics in results.items():
    strict = metrics.get('exact_match,strict-match')
    flex = metrics.get('exact_match,flexible-extract')
    if strict is not None:
        pct = strict * 100 if strict <= 1 else strict
        print(f'gsm8k acc strict-match: {pct:.2f}')
    if flex is not None:
        pct = flex * 100 if flex <= 1 else flex
        print(f'gsm8k acc flexible-extract: {pct:.2f}')
    if strict is not None:
        pct = strict * 100 if strict <= 1 else strict
        print(f'accuracy: {pct:.2f}')
        sys.exit(0)
    elif flex is not None:
        pct = flex * 100 if flex <= 1 else flex
        print(f'accuracy: {pct:.2f}')
        sys.exit(0)
print('WARNING: no accuracy metric found in results')
" 2>&1
else
    echo "WARNING: no results JSON found in ${EVAL_RESULT_DIR}"
fi

echo "Eval artifacts saved to: ${EVAL_RESULT_DIR}"
