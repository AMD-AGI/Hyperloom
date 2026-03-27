#!/bin/bash
# Rapid single-experiment runner for per-GPU optimization
# Usage: EXPERIMENT=name EXTRA_SERVER_ARGS="..." EXTRA_ENV="..." bash rapid_experiment.sh
set -e

EXPERIMENT="${EXPERIMENT:?Set EXPERIMENT=name}"
RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v3"
mkdir -p "$RESULTS_DIR"

cd /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX
source benchmarks/benchmark_lib.sh

export MODEL="zai-org/GLM-5-FP8"
export RANDOM_RANGE_RATIO=0.8
export PORT=8888

# Default env vars (can be overridden by EXTRA_ENV)
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384

# Apply any extra env vars
eval "$EXTRA_ENV"

SERVER_LOG="/tmp/server_${EXPERIMENT}.log"
TP=${TP:-4}
CONC=${CONC:-64}
ISL=${ISL:-1024}
OSL=${OSL:-1024}

echo "=== EXPERIMENT: $EXPERIMENT ==="
echo "  TP=$TP CONC=$CONC ISL=$ISL OSL=$OSL"
echo "  EXTRA_SERVER_ARGS: ${EXTRA_SERVER_ARGS:-none}"
echo "  EXTRA_ENV: ${EXTRA_ENV:-none}"

pkill -9 -f sglang 2>/dev/null || true
sleep 5

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
    $EXTRA_SERVER_ARGS \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

RESULT_FILENAME="${EXPERIMENT}_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"
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

if [ -f "$RESULTS_DIR/${RESULT_FILENAME}.json" ]; then
    python3 -c "
import json
d = json.load(open('$RESULTS_DIR/${RESULT_FILENAME}.json'))
total = d['total_token_throughput']
output = d['output_throughput']
tpot = d['mean_tpot_ms']
per_gpu = total / $TP
print(f'  Total tok/s:    {total:.2f}')
print(f'  Per-GPU tok/s:  {per_gpu:.2f}')
print(f'  Output tok/s:   {output:.2f}')
print(f'  Mean TPOT:      {tpot:.2f} ms')
baseline = 1403.43
delta = (total - baseline) / baseline * 100
print(f'  vs TP=4 baseline (1403): {delta:+.2f}%')
"
fi

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 5
echo "=== DONE: $EXPERIMENT ==="
