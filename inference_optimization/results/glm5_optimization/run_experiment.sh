#!/bin/bash
# Quick experiment runner - launches server, benchmarks, kills server
# Usage: ./run_experiment.sh <experiment_name> <extra_server_args...>

set -e
EXPERIMENT_NAME="${1:?Usage: $0 <experiment_name> [extra_server_args...]}"
shift
EXTRA_ARGS="$@"

RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v2"
mkdir -p "$RESULTS_DIR"

cd /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX
source benchmarks/benchmark_lib.sh

export MODEL="zai-org/GLM-5-FP8"
export RANDOM_RANGE_RATIO=0.8
export PORT=8888

export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384

SERVER_LOG="/workspace/server_${EXPERIMENT_NAME}.log"

echo "==========================================="
echo "EXPERIMENT: $EXPERIMENT_NAME"
echo "EXTRA ARGS: $EXTRA_ARGS"
echo "==========================================="

python3 -m sglang.launch_server \
    --disable-radix-cache \
    --model-path models/$MODEL \
    --served-model-name $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    $EXTRA_ARGS \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Run ISL=1024 OSL=1024 benchmark
export ISL=1024 OSL=1024
# Determine CONC from experiment name or default 64
CONC=${BENCH_CONC:-64}
export CONC
export RESULT_FILENAME="${EXPERIMENT_NAME}_isl${ISL}_osl${OSL}_conc${CONC}"

echo "Running benchmark: ISL=$ISL OSL=$OSL CONC=$CONC"
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 3))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$RESULTS_DIR/"

echo ""
grep "Output token throughput\|Total Token throughput\|Successful\|Benchmark duration" "$RESULTS_DIR/${RESULT_FILENAME}.log" 2>/dev/null || true
echo ""
echo "EXPERIMENT $EXPERIMENT_NAME DONE"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 5
