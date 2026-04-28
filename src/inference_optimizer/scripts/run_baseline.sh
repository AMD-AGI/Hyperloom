#!/usr/bin/env bash
# scripts/run_baseline.sh — IR-3 entry point for baseline / bench / integrate.
#
# Required env:
#   MODEL        — model path / hub id
#   TP           — tensor-parallel size
#   PORT         — server port (e.g. 8080)
#   OUT_DIR      — where to write metrics.json (must exist; we mkdir -p)
#
# Optional env:
#   FRAMEWORK    — sglang | vllm                     (default sglang)
#   PROFILE      — "1" to enable profiling, unset otherwise
#   DRY_RUN_MOCK — when set to "1" we skip the real launch and write a
#                  deterministic metrics.json so unit tests can exercise
#                  the rest of the pipeline without GPUs.
#
# Exit 0 → metrics.json written; exit non-zero → caller should mark task failed.
set -uo pipefail

: "${MODEL:?MODEL must be set}"
: "${TP:?TP must be set}"
: "${PORT:?PORT must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"

FRAMEWORK="${FRAMEWORK:-sglang}"
DRY_RUN_MOCK="${DRY_RUN_MOCK:-0}"

mkdir -p "$OUT_DIR"

if [ "$DRY_RUN_MOCK" = "1" ]; then
  cat > "$OUT_DIR/metrics.json" <<JSON
{
  "model": "$MODEL",
  "framework": "$FRAMEWORK",
  "tp": $TP,
  "tput_per_gpu": 5000.0,
  "p50_latency_ms": 12.5,
  "p95_latency_ms": 28.0,
  "mocked": true
}
JSON
  echo "DRY_RUN_MOCK: wrote $OUT_DIR/metrics.json"
  exit 0
fi

# Real path — production sandbox only.
echo "[run_baseline.sh] would launch $FRAMEWORK on port $PORT for $MODEL (TP=$TP)"
echo "[run_baseline.sh] real implementation lives in the sprint repository"
echo "[run_baseline.sh] set DRY_RUN_MOCK=1 to use the test fixture path"
exit 1
