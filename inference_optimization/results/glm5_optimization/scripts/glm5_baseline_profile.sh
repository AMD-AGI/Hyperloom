#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization"
INFERENCEX_PATH="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX"
SKILL_SCRIPTS="/root/.cursor/skills/inference-optimization/scripts"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)

export RESULT_DIR="${WORK_DIR}/results/${TIMESTAMP}"
export TRACE_DIR="${WORK_DIR}/traces/${TIMESTAMP}"
mkdir -p "$RESULT_DIR" "$TRACE_DIR"

export MODEL="/shared_nfs/limou/models/zai-org/GLM-5-FP8"
export TP=4
export CONC="${CONC:-64}"
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export PORT=8888
export FRAMEWORK=sglang

export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1
export SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"

source "$SKILL_SCRIPTS/common.sh"

echo "============================================================"
echo " GLM-5-FP8 Baseline + Profile"
echo "============================================================"
echo "  MODEL : $MODEL"
echo "  TP=$TP  CONC=$CONC  ISL=$ISL  OSL=$OSL"
echo "  RESULT_DIR: $RESULT_DIR"
echo "  TRACE_DIR:  $TRACE_DIR"
echo "============================================================"

kill_server
sleep 5

GPU_FREE_THRESHOLD=0.50
GPU_CHECK_RETRIES=2
export GPU_FREE_THRESHOLD GPU_CHECK_RETRIES
check_gpu_memory || { echo "WARNING: GPU memory check failed, proceeding anyway"; }

echo ""
echo "[1/5] Launching SGLang server with GLM5 best config..."

SERVER_LOG="$RESULT_DIR/server_baseline.log"

python3 -m sglang.launch_server \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend aiter \
    --cuda-graph-max-bs "$CONC" \
    --disable-radix-cache \
    --model-path "$MODEL" \
    --served-model-name "zai-org/GLM-5-FP8" \
    --host=0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_health "$PORT" "$SERVER_LOG" "$SERVER_PID" 600

echo ""
echo "[2/5] Running baseline benchmark (no profiling)..."

check_benchmark_lib "$INFERENCEX_PATH"
cd "$INFERENCEX_PATH"

NUM_PROMPTS=$((CONC * 3))
RESULT_FILENAME="baseline_sglang_tp${TP}_conc${CONC}_isl${ISL}_osl${OSL}"

export RANDOM_RANGE_RATIO=1.0

run_benchmark_serving \
    --model "zai-org/GLM-5-FP8" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$RESULT_DIR/" \
    --trust-remote-code

echo ""
python3 -c "
import json
with open('$RESULT_DIR/${RESULT_FILENAME}.json') as f:
    d = json.load(f)
print('=== Baseline Results ===')
print(f'Output throughput: {d[\"output_throughput\"]:.2f} tok/s')
print(f'TPOT: {d[\"mean_tpot_ms\"]:.2f} ms')
print(f'TTFT: {d[\"mean_ttft_ms\"]:.2f} ms')
print(f'Completed: {d[\"completed\"]}/{d[\"num_prompts\"]}')
"

echo ""
echo "[3/5] Activating profiler..."
curl -s -X POST "http://0.0.0.0:$PORT/start_profile" 2>/dev/null \
    || curl -s "http://0.0.0.0:$PORT/start_profile" 2>/dev/null \
    || echo "WARNING: start_profile failed"

PROFILE_PROMPTS=$((CONC < 16 ? CONC : 16))
echo "Running profiling benchmark ($PROFILE_PROMPTS prompts)..."

python3 "$INFERENCEX_PATH/utils/bench_serving/benchmark_serving.py" \
    --model "zai-org/GLM-5-FP8" \
    --backend vllm \
    --base-url "http://0.0.0.0:$PORT" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$PROFILE_PROMPTS" \
    --max-concurrency "$PROFILE_PROMPTS" \
    --request-rate inf \
    --ignore-eos \
    --num-warmups 0 \
    --save-result \
    --result-dir "$RESULT_DIR/" \
    --result-filename profile_run \
    --trust-remote-code

echo "Stopping profiler..."
curl -s -X POST "http://0.0.0.0:$PORT/stop_profile" 2>/dev/null \
    || curl -s "http://0.0.0.0:$PORT/stop_profile" 2>/dev/null \
    || echo "WARNING: stop_profile failed"

echo ""
echo "[4/5] Waiting for traces..."
sleep 15

PREV_SIZE=0
for i in $(seq 1 60); do
    TRACE_COUNT=$(ls "$TRACE_DIR"/*.json.gz 2>/dev/null | wc -l || echo 0)
    CUR_SIZE=$(du -sb "$TRACE_DIR" 2>/dev/null | awk '{print $1}' || echo 0)
    if [ "$TRACE_COUNT" -ge "$TP" ] && [ "$CUR_SIZE" = "$PREV_SIZE" ] && [ "$CUR_SIZE" -gt 0 ]; then
        echo "Traces stable at ${TRACE_COUNT} files"
        break
    fi
    PREV_SIZE=$CUR_SIZE
    echo "  Waiting... ($TRACE_COUNT files, ${CUR_SIZE} bytes)"
    sleep 5
done

echo ""
echo "[5/5] Filtering traces..."
TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*TP-0*.json.gz 2>/dev/null | head -1 || true)
if [ -z "$TRACE_FOR_ANALYSIS" ]; then
    TRACE_FOR_ANALYSIS=$(ls "$TRACE_DIR"/*.json.gz 2>/dev/null | head -1 || true)
fi

if [ -n "$TRACE_FOR_ANALYSIS" ]; then
    FILTERED_TRACE="${TRACE_DIR}/filtered-TP-0.trace.json.gz"
    filter_trace "$TRACE_FOR_ANALYSIS" "$FILTERED_TRACE" 2>&1 || echo "WARNING: Trace filtering failed"
    echo "Filtered trace: $FILTERED_TRACE"
fi

cat > "$RESULT_DIR/run_context.env" <<EOF
MODEL=$MODEL
TP=$TP
CONC=$CONC
ISL=$ISL
OSL=$OSL
PORT=$PORT
FRAMEWORK=$FRAMEWORK
RESULT_DIR=$RESULT_DIR
TRACE_DIR=$TRACE_DIR
SERVER_PID=$SERVER_PID
RESULT_FILENAME=$RESULT_FILENAME
FILTERED_TRACE=${FILTERED_TRACE:-}
TIMESTAMP=$TIMESTAMP
EOF

echo ""
echo "============================================================"
echo "=== GLM5 Baseline + Profile Complete ==="
echo "Results: $RESULT_DIR"
echo "Traces:  $TRACE_DIR"
echo "Server still running (PID=$SERVER_PID)"
echo "============================================================"
