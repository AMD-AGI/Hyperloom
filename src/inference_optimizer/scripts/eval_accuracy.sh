#!/usr/bin/env bash
# scripts/eval_accuracy.sh — DESIGN §7.5.2.
#
# Required env:
#   MODEL         — model path / hub id
#   PORT          — server port
#   RESULTS_DIR   — where eval_summary_<task>.json gets written
#
# Optional env:
#   EVAL_TASK     — gsm8k | mmlu | ...               (default gsm8k)
#   NUM_FEWSHOT   — fewshot count                    (default 5)
#   DRY_RUN_MOCK  — when "1", skip real eval and write a deterministic
#                   summary so unit tests can exercise the gate logic.
#
# The accuracy_gate.run_gsm8k helper invokes this script and reads
# ``$RESULTS_DIR/eval_summary_${EVAL_TASK}.json``.
set -uo pipefail

: "${MODEL:?MODEL must be set}"
: "${PORT:?PORT must be set}"
: "${RESULTS_DIR:?RESULTS_DIR must be set}"

EVAL_TASK="${EVAL_TASK:-gsm8k}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
DRY_RUN_MOCK="${DRY_RUN_MOCK:-0}"

mkdir -p "$RESULTS_DIR"

if [ "$DRY_RUN_MOCK" = "1" ]; then
  cat > "$RESULTS_DIR/eval_summary_${EVAL_TASK}.json" <<JSON
{
  "task": "$EVAL_TASK",
  "num_fewshot": $NUM_FEWSHOT,
  "model": "$MODEL",
  "score": 0.71,
  "mocked": true
}
JSON
  echo "DRY_RUN_MOCK: wrote $RESULTS_DIR/eval_summary_${EVAL_TASK}.json"
  exit 0
fi

echo "[eval_accuracy.sh] real path requires lm-evaluation-harness — set DRY_RUN_MOCK=1 for tests"
exit 1
