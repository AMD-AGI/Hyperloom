#!/bin/bash
# Replication of best config: DP=2/TP=4, ds=16, fused-moe-sum-all-reduce, CONC=128
# Original result: 2794.41 total tok/s (dp2_tp4_ds16_fusedmoe_isl1024_osl1024_conc128.json)
# This script writes to results_replication/ — does NOT overwrite any existing results.
set -e

RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_replication"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="replicate_dp2_tp4_ds16_fusedmoe_conc128_${TIMESTAMP}"
SERVER_LOG="/tmp/server_${EXPERIMENT_NAME}.log"

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

echo "==========================================="
echo "REPLICATION: $EXPERIMENT_NAME"
echo "Target: 2794.41 total tok/s"
echo "Config: DP=2/TP=4, ds=16, fused-moe, CONC=128"
echo "Results dir: $RESULTS_DIR"
echo "Server log: $SERVER_LOG"
echo "==========================================="

pkill -9 -f sglang 2>/dev/null || true
sleep 3

echo "[$(date)] Launching server..."
python3 -m sglang.launch_server \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --cuda-graph-max-bs 128 \
    --disable-radix-cache \
    --model-path models/$MODEL \
    --served-model-name $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85 \
    --num-continuous-decode-steps 16 \
    --enable-fused-moe-sum-all-reduce \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "[$(date)] Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
echo "[$(date)] Server ready."

ISL=1024
OSL=1024
CONC=128
RESULT_FILENAME="${EXPERIMENT_NAME}"

echo ""
echo "[$(date)] Running benchmark: ISL=$ISL OSL=$OSL CONC=$CONC"
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
echo "==========================================="
echo "[$(date)] REPLICATION COMPLETE"
echo "==========================================="

if [ -f "$RESULTS_DIR/${RESULT_FILENAME}.json" ]; then
    python3 -c "
import json
d = json.load(open('$RESULTS_DIR/${RESULT_FILENAME}.json'))
total = d['total_token_throughput']
output = d['output_throughput']
tpot = d['mean_tpot_ms']
ttft = d['mean_ttft_ms']
completed = d['completed']
num = d['num_prompts']
print(f'')
print(f'  Output throughput:  {output:.2f} tok/s')
print(f'  Total throughput:   {total:.2f} tok/s')
print(f'  Mean TPOT:          {tpot:.2f} ms')
print(f'  Mean TTFT:          {ttft:.2f} ms')
print(f'  Completed:          {completed}/{num}')
print(f'')
print(f'  ORIGINAL:           2794.41 total tok/s')
print(f'  REPLICATION:        {total:.2f} total tok/s')
diff_pct = (total - 2794.41) / 2794.41 * 100
print(f'  DELTA:              {diff_pct:+.2f}%')
if abs(diff_pct) < 5:
    print(f'  STATUS:             REPLICATED (within 5% tolerance)')
else:
    print(f'  STATUS:             MISMATCH (>{5}% deviation)')
"
fi

echo ""
echo "Killing server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 5
echo "Done."
