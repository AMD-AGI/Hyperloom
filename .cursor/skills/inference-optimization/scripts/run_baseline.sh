#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ============================================================
# Inference Optimization — Baseline + Profile (single server launch)
#
# Supports both SGLang and vLLM serving frameworks.
# Runs baseline benchmark (clean, no profiling overhead), then
# activates profiler via HTTP endpoint for trace collection.
#
# Required env vars: MODEL, TP, CONC, ISL, OSL, INFERENCEX_PATH
# Optional: PORT (default 8888), RESULT_DIR, TRACE_DIR,
#           NUM_PROMPTS_MULTIPLIER (default 3),
#           FRAMEWORK (sglang|vllm, default sglang),
#           SGLANG_EXTRA_ARGS, VLLM_EXTRA_ARGS
# ============================================================

unset PROFILE
KEEP_SERVER=0
baseline_cleanup() {
    unset PROFILE SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR 2>/dev/null || true
    if [ "${KEEP_SERVER:-0}" != "1" ]; then
        kill_server 2>/dev/null || true
    fi
}
trap baseline_cleanup EXIT INT TERM

: "${MODEL:?MODEL env var required}"
: "${TP:?TP env var required}"
: "${CONC:?CONC env var required}"
: "${ISL:?ISL env var required}"
: "${OSL:?OSL env var required}"
: "${INFERENCEX_PATH:?INFERENCEX_PATH env var required}"

FRAMEWORK="${FRAMEWORK:-sglang}"
PORT=${PORT:-8888}
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
RESULT_DIR="${RESULT_DIR:-/wekafs/inference-optimization/results/${TIMESTAMP}}"
TRACE_DIR="${TRACE_DIR:-/wekafs/inference-optimization/traces/${TIMESTAMP}}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-3}"
NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULTIPLIER))
RESULT_FILENAME="baseline_${FRAMEWORK}_tp${TP}_conc${CONC}_isl${ISL}_osl${OSL}"
CONTEXT_FILE="${RESULT_DIR}/run_context.env"

mkdir -p "$RESULT_DIR" "$TRACE_DIR"

write_run_context() {
    cat > "$CONTEXT_FILE" <<EOF
MODEL=$(printf '%q' "$MODEL")
TP=$(printf '%q' "$TP")
CONC=$(printf '%q' "$CONC")
ISL=$(printf '%q' "$ISL")
OSL=$(printf '%q' "$OSL")
INFERENCEX_PATH=$(printf '%q' "$INFERENCEX_PATH")
PORT=$(printf '%q' "$PORT")
FRAMEWORK=$(printf '%q' "$FRAMEWORK")
RESULT_DIR=$(printf '%q' "$RESULT_DIR")
TRACE_DIR=$(printf '%q' "$TRACE_DIR")
NUM_PROMPTS_MULTIPLIER=$(printf '%q' "$NUM_PROMPTS_MULTIPLIER")
RESULT_FILENAME=$(printf '%q' "$RESULT_FILENAME")
SGLANG_EXTRA_ARGS=$(printf '%q' "${SGLANG_EXTRA_ARGS:-}")
VLLM_EXTRA_ARGS=$(printf '%q' "${VLLM_EXTRA_ARGS:-}")
SERVER_PID=$(printf '%q' "${SERVER_PID:-}")
TRACE_FOR_ANALYSIS=$(printf '%q' "${TRACE_FOR_ANALYSIS:-}")
EOF
}

write_run_context

check_benchmark_lib "$INFERENCEX_PATH"
cd "$INFERENCEX_PATH"

echo "============================================================"
echo "Baseline + Profile (single server launch)"
echo "Framework: $FRAMEWORK"
echo "Model: $MODEL | TP=$TP CONC=$CONC ISL=$ISL OSL=$OSL"
echo "Prompts: $NUM_PROMPTS (${NUM_PROMPTS_MULTIPLIER}x CONC)"
echo "Results: $RESULT_DIR"
echo "Traces:  $TRACE_DIR"
echo "============================================================"

# --- Kill stale processes ---
kill_server

GPU_FREE_THRESHOLD="${GPU_FREE_THRESHOLD:-0.85}"
GPU_CHECK_RETRIES="${GPU_CHECK_RETRIES:-2}"
export GPU_FREE_THRESHOLD GPU_CHECK_RETRIES
check_gpu_memory || exit 1

# --- Launch server WITH profiler dir (but profiler NOT active yet) ---
echo ""
echo "[1/4] Starting $FRAMEWORK server (with profiler dir pre-configured)..."

if [ "$FRAMEWORK" = "vllm" ]; then
    export VLLM_TORCH_PROFILER_DIR="$TRACE_DIR"
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
    write_run_context

    vllm serve "$MODEL" \
        --host 0.0.0.0 --port $PORT --tensor-parallel-size $TP \
        --trust-remote-code \
        --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL:-0.85} --disable-log-stats \
        $VLLM_EXTRA_ARGS \
        > "$RESULT_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    write_run_context
    wait_for_health "$PORT" "$RESULT_DIR/server.log" "$SERVER_PID"
else
    export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
    export SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"
    SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
    write_run_context

    if [ -n "${MEM_FRACTION:-}" ]; then
        : # user explicitly set MEM_FRACTION, use it
    elif echo "$SGLANG_EXTRA_ARGS" | grep -q "\-\-enable-torch-compile"; then
        MEM_FRACTION=0.6
    else
        MEM_FRACTION=0.8
    fi

    python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --host=0.0.0.0 --port $PORT --tensor-parallel-size $TP \
        --trust-remote-code \
        --mem-fraction-static "$MEM_FRACTION" --disable-radix-cache \
        --cuda-graph-max-bs "$CONC" \
        $SGLANG_EXTRA_ARGS \
        > "$RESULT_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    write_run_context
    wait_for_health "$PORT" "$RESULT_DIR/server.log" "$SERVER_PID"
fi

# --- Phase 2: Baseline benchmark (profiler NOT active, clean numbers) ---
echo ""
echo "[2/4] Running baseline benchmark ($NUM_PROMPTS prompts, no profiling)..."
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"
export RANDOM_RANGE_RATIO RESULT_FILENAME

run_benchmark_serving \
    --model "$MODEL" --port "$PORT" --backend vllm \
    --input-len "$ISL" --output-len "$OSL" --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" --result-dir "$RESULT_DIR/" \
    --trust-remote-code

echo ""
python3 -c "
import json
with open('$RESULT_DIR/${RESULT_FILENAME}.json') as f:
    d = json.load(f)
print(f'=== Baseline Results ($FRAMEWORK) ===')
print(f'Output throughput: {d[\"output_throughput\"]:.2f} tok/s')
print(f'TPOT: {d[\"mean_tpot_ms\"]:.2f} ms')
print(f'TTFT: {d[\"mean_ttft_ms\"]:.2f} ms')
print(f'Completed: {d[\"completed\"]}/{d[\"num_prompts\"]}')
"

# --- Phase 3: Profiling (activate via HTTP, same server, no restart) ---
echo ""
echo "[3/4] Activating profiler via /start_profile..."
curl -s -X POST "http://0.0.0.0:$PORT/start_profile" 2>/dev/null \
    || curl -s "http://0.0.0.0:$PORT/start_profile" 2>/dev/null \
    || echo "WARNING: start_profile failed"

PROFILE_PROMPTS=$((CONC < 16 ? CONC : 16))
echo "Running profiling benchmark ($PROFILE_PROMPTS prompts, reduced for trace size)..."
export RESULT_FILENAME="profile_run"
python3 "$INFERENCEX_PATH/utils/bench_serving/benchmark_serving.py" \
    --model "$MODEL" --backend vllm --base-url "http://0.0.0.0:$PORT" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" --num-prompts "$PROFILE_PROMPTS" \
    --max-concurrency "$PROFILE_PROMPTS" --request-rate inf --ignore-eos \
    --num-warmups 0 --save-result \
    --result-dir "$RESULT_DIR/" --result-filename profile_run \
    --trust-remote-code

echo "Stopping profiler (trace serialization + NFS write may take 30-120s)..."
curl -s -X POST "http://0.0.0.0:$PORT/stop_profile" 2>/dev/null \
    || curl -s "http://0.0.0.0:$PORT/stop_profile" 2>/dev/null \
    || echo "WARNING: stop_profile failed"

# --- Wait for traces to be fully written ---
echo ""
echo "[4/4] Waiting for traces to be written to NFS..."
sleep 15

PREV_SIZE=0
TRACE_COUNT=0
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

# Filter traces — try TP-0 first, fall back to any .json.gz
TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*TP-0*.json.gz 2>/dev/null | head -1 || true)
if [ -z "$TRACE_FOR_ANALYSIS" ]; then
    TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*.json.gz 2>/dev/null | head -1 || true)
fi

if [ -n "$TRACE_FOR_ANALYSIS" ]; then
    echo "Filtering trace (removing python_function events for TraceLens)..."
    FILTERED_TRACE="${TRACE_DIR}/filtered-TP-0.trace.json.gz"
    filter_trace "$TRACE_FOR_ANALYSIS" "$FILTERED_TRACE" 2>&1 || echo "WARNING: Trace filtering failed"
    TRACE_FOR_ANALYSIS="$FILTERED_TRACE"
fi
write_run_context

if [ "$TRACE_COUNT" -gt 0 ]; then
    echo "Traces ($TRACE_COUNT files) saved to $TRACE_DIR:"
    ls -lh "$TRACE_DIR"/*.json.gz | head -5
    echo "..."
    TOTAL_SIZE=$(du -sh "$TRACE_DIR" | awk '{print $1}')
    echo "Total trace size: $TOTAL_SIZE"
    echo ""
    if [ -z "$TRACE_FOR_ANALYSIS" ] || [ ! -f "$TRACE_FOR_ANALYSIS" ]; then
        TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/filtered-TP-0*.json.gz 2>/dev/null | head -1)
    fi
    if [ -z "$TRACE_FOR_ANALYSIS" ] || [ ! -f "$TRACE_FOR_ANALYSIS" ]; then
        TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*TP-0*.json.gz 2>/dev/null | head -1)
    fi
    if [ -z "$TRACE_FOR_ANALYSIS" ] || [ ! -f "$TRACE_FOR_ANALYSIS" ]; then
        TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*.json.gz 2>/dev/null | head -1)
    fi
    echo "For TraceLens: $TRACE_FOR_ANALYSIS"
else
    echo "WARNING: No trace files found in $TRACE_DIR after 60s"
fi
write_run_context

KEEP_SERVER=1
echo ""
echo "============================================================"
echo "=== Complete. Server still running (PID=$SERVER_PID). ==="
echo "FRAMEWORK=$FRAMEWORK"
echo "RESULT_DIR=$RESULT_DIR"
echo "TRACE_DIR=$TRACE_DIR"
echo "RUN_CONTEXT=$CONTEXT_FILE"
echo "SERVER_PID=$SERVER_PID"
echo "============================================================"
