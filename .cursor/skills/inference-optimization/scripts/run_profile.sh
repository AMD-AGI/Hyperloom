#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ============================================================
# Inference Optimization — Profiling (separate from baseline)
# Supports both SGLang and vLLM serving frameworks.
# Requires: server already running (from run_baseline.sh)
#
# Required env vars: MODEL, CONC, ISL, OSL, INFERENCEX_PATH
# Optional: PORT (default 8888), RESULT_DIR, TRACE_DIR,
#           FRAMEWORK (sglang|vllm, default sglang)
# ============================================================

USER_PORT="${PORT:-}"
USER_RESULT_DIR="${RESULT_DIR:-}"
USER_TRACE_DIR="${TRACE_DIR:-}"
USER_MODEL="${MODEL:-}"
USER_CONC="${CONC:-}"
USER_ISL="${ISL:-}"
USER_OSL="${OSL:-}"
USER_INFERENCEX_PATH="${INFERENCEX_PATH:-}"
USER_FRAMEWORK="${FRAMEWORK:-}"

CONTEXT_CANDIDATE="${RUN_CONTEXT_FILE:-}"
if [ -z "$CONTEXT_CANDIDATE" ] && [ -n "$USER_RESULT_DIR" ] && [ -f "$USER_RESULT_DIR/run_context.env" ]; then
    CONTEXT_CANDIDATE="$USER_RESULT_DIR/run_context.env"
fi

if [ -n "$CONTEXT_CANDIDATE" ] && [ -f "$CONTEXT_CANDIDATE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$CONTEXT_CANDIDATE"
    set +a
fi

[ -n "$USER_MODEL" ] && MODEL="$USER_MODEL"
[ -n "$USER_CONC" ] && CONC="$USER_CONC"
[ -n "$USER_ISL" ] && ISL="$USER_ISL"
[ -n "$USER_OSL" ] && OSL="$USER_OSL"
[ -n "$USER_INFERENCEX_PATH" ] && INFERENCEX_PATH="$USER_INFERENCEX_PATH"
[ -n "$USER_PORT" ] && PORT="$USER_PORT"
[ -n "$USER_RESULT_DIR" ] && RESULT_DIR="$USER_RESULT_DIR"
[ -n "$USER_TRACE_DIR" ] && TRACE_DIR="$USER_TRACE_DIR"
[ -n "$USER_FRAMEWORK" ] && FRAMEWORK="$USER_FRAMEWORK"

FRAMEWORK="${FRAMEWORK:-sglang}"
PORT=${PORT:-8888}
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
NFS_ROOT="${NFS_ROOT:-/wekafs}"
RESULT_DIR="${RESULT_DIR:-${NFS_ROOT}/inference-optimization/results/$TIMESTAMP}"
TRACE_DIR="${TRACE_DIR:-}"

: "${MODEL:?MODEL env var required (or load via RUN_CONTEXT_FILE)}"
: "${CONC:?CONC env var required (or load via RUN_CONTEXT_FILE)}"
: "${ISL:?ISL env var required (or load via RUN_CONTEXT_FILE)}"
: "${OSL:?OSL env var required (or load via RUN_CONTEXT_FILE)}"
: "${INFERENCEX_PATH:?INFERENCEX_PATH env var required (or load via RUN_CONTEXT_FILE)}"

if [ -z "$TRACE_DIR" ]; then
    echo "ERROR: TRACE_DIR is unknown."
    echo "Provide either:"
    echo "  1. RUN_CONTEXT_FILE=<baseline RESULT_DIR>/run_context.env"
    echo "  2. RESULT_DIR=<baseline RESULT_DIR>  (script will auto-load run_context.env)"
    echo "  3. TRACE_DIR=\${NFS_ROOT}/inference-optimization/traces/<timestamp>"
    exit 1
fi

profile_cleanup() {
    curl -s -X POST "http://0.0.0.0:${PORT}/stop_profile" 2>/dev/null \
        || curl -s "http://0.0.0.0:${PORT}/stop_profile" 2>/dev/null \
        || true
    unset PROFILE SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR 2>/dev/null || true
}
trap profile_cleanup EXIT INT TERM

mkdir -p "$RESULT_DIR" "$TRACE_DIR"

if ! curl -s "http://0.0.0.0:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server not running on port $PORT."
    if [ "$FRAMEWORK" = "vllm" ]; then
        echo "Run run_baseline.sh first with FRAMEWORK=vllm, or start vllm serve with VLLM_TORCH_PROFILER_DIR set."
    else
        echo "Run run_baseline.sh first, or start server with SGLANG_TORCH_PROFILER_DIR set."
    fi
    exit 1
fi

check_benchmark_lib "$INFERENCEX_PATH"
cd "$INFERENCEX_PATH"

echo "============================================================"
echo "Profiling Run"
echo "Framework: $FRAMEWORK"
echo "Model: $MODEL | CONC=$CONC ISL=$ISL OSL=$OSL"
if [ -n "$CONTEXT_CANDIDATE" ]; then
    echo "Context: $CONTEXT_CANDIDATE"
fi
echo "Traces will be saved to: $TRACE_DIR"
echo "============================================================"

echo "[1/5] Starting profiler via /start_profile endpoint..."
PROFILE_START=$(curl -s -X POST "http://0.0.0.0:$PORT/start_profile" 2>&1 \
    || curl -s "http://0.0.0.0:$PORT/start_profile" 2>&1)
if echo "$PROFILE_START" | grep -qi "error\|not supported\|not in progress"; then
    if [ "$FRAMEWORK" = "vllm" ]; then
        echo "WARNING: Server may not have VLLM_TORCH_PROFILER_DIR set."
    else
        echo "WARNING: Server may not have SGLANG_TORCH_PROFILER_DIR set."
    fi
    echo "Profiling may fail. Consider restarting server with the profiler env var."
fi

echo "[2/5] Running profiling benchmark (num_prompts=$CONC)..."
export RANDOM_RANGE_RATIO=1.0
export RESULT_FILENAME="profile_run"
export PROFILE=1

run_benchmark_serving \
    --model "$MODEL" --port "$PORT" --backend vllm \
    --input-len "$ISL" --output-len "$OSL" --random-range-ratio 1.0 \
    --num-prompts "$CONC" --max-concurrency "$CONC" \
    --result-filename profile_run --result-dir "$RESULT_DIR/" \
    --trust-remote-code

unset PROFILE

echo "[3/5] Stopping profiler (trace serialization + NFS write may take 30-120s)..."
curl -s -X POST "http://0.0.0.0:$PORT/stop_profile" 2>/dev/null \
    || curl -s "http://0.0.0.0:$PORT/stop_profile" 2>/dev/null \
    || echo "WARNING: stop_profile failed"

echo ""
echo "[4/5] Waiting for traces to be written to NFS..."
sleep 15

PREV_SIZE=0
TRACE_COUNT=0
TP=${TP:-1}
for i in $(seq 1 60); do
    TRACE_COUNT=$(ls "$TRACE_DIR"/*.json.gz 2>/dev/null | wc -l || echo 0)
    CUR_SIZE=$(du -sb "$TRACE_DIR" 2>/dev/null | awk '{print $1}' || echo 0)
    if [ "$TRACE_COUNT" -ge "$TP" ] && [ "$CUR_SIZE" = "$PREV_SIZE" ] && [ "$CUR_SIZE" -gt 0 ]; then
        echo "Traces stable at ${TRACE_COUNT} files"
        break
    fi
    PREV_SIZE=$CUR_SIZE
    sleep 5
done

echo "[5/5] Filtering traces for TraceLens..."
TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*TP-0*.json.gz 2>/dev/null | head -1 || true)
if [ -z "$TRACE_FOR_ANALYSIS" ]; then
    TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*.json.gz 2>/dev/null | head -1 || true)
fi

if [ -n "$TRACE_FOR_ANALYSIS" ]; then
    FILTERED_TRACE="${TRACE_DIR}/filtered-TP-0.trace.json.gz"
    filter_trace "$TRACE_FOR_ANALYSIS" "$FILTERED_TRACE" 2>&1 || echo "WARNING: Trace filtering failed"
    TRACE_FOR_ANALYSIS="$FILTERED_TRACE"
fi

unset SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR 2>/dev/null || true

if [ "$TRACE_COUNT" -gt 0 ]; then
    echo ""
    echo "Traces ($TRACE_COUNT files) saved to $TRACE_DIR:"
    ls -lh "$TRACE_DIR"/*.json.gz 2>/dev/null | head -5
    TOTAL_SIZE=$(du -sh "$TRACE_DIR" | awk '{print $1}')
    echo "Total trace size: $TOTAL_SIZE"
    echo ""
    echo "For TraceLens: ${TRACE_FOR_ANALYSIS:-not found}"
else
    echo "WARNING: No trace files found in $TRACE_DIR after 60s"
    if [ "$FRAMEWORK" = "vllm" ]; then
        echo "Server may need VLLM_TORCH_PROFILER_DIR=$TRACE_DIR"
    else
        echo "Server may need SGLANG_TORCH_PROFILER_DIR=$TRACE_DIR"
    fi
fi

echo ""
echo "=== Profiling complete. ==="
echo "FRAMEWORK=$FRAMEWORK"
echo "TRACE_DIR=$TRACE_DIR"
