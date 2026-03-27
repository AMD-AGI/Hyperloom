#!/bin/bash
set -e

cd /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX
source benchmarks/benchmark_lib.sh

export MODEL="zai-org/GLM-5-FP8"
export TP=4
export RANDOM_RANGE_RATIO=0.8
export PORT=8888

export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384

RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results"
mkdir -p "$RESULTS_DIR"

CONC=64
SERVER_LOG=/workspace/server.log

echo "Starting server with optimized config..."
python3 -m sglang.launch_server \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --cuda-graph-max-bs $CONC \
    --disable-radix-cache \
    --model-path models/$MODEL \
    --served-model-name $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85 \
    --num-continuous-decode-steps 8 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    > $SERVER_LOG 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

declare -a CONFIGS=(
    "1024 1024 64"
    "1024 8192 32"
    "8192 1024 16"
)

for config in "${CONFIGS[@]}"; do
    read -r ISL OSL CONC <<< "$config"
    export ISL OSL CONC
    RESULT_FILENAME="sweep_isl${ISL}_osl${OSL}_conc${CONC}"
    export RESULT_FILENAME

    echo ""
    echo "=========================================="
    echo "Running ISL=$ISL OSL=$OSL CONC=$CONC"
    echo "=========================================="

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

    echo "ISL=$ISL OSL=$OSL CONC=$CONC done."
done

echo ""
echo "All sweep benchmarks complete!"
kill $SERVER_PID 2>/dev/null || true
