#!/bin/bash
# Batch experiment runner for per-GPU optimization
# Tests multiple configurations sequentially, logs results summary
set -e

RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v3"
SUMMARY="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v3/experiment_summary.txt"
mkdir -p "$RESULTS_DIR"

cd /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX
source benchmarks/benchmark_lib.sh

export MODEL="zai-org/GLM-5-FP8"
export RANDOM_RANGE_RATIO=0.8
export PORT=8888

BASE_ENV="export SGLANG_USE_AITER=1; export RCCL_MSCCL_ENABLE=0; export SGLANG_ROCM_FUSED_DECODE_MLA=0; export ROCM_QUICK_REDUCE_QUANTIZATION=INT4; export SAFETENSORS_FAST_GPU=1; export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384"

BASE_ARGS="--nsa-prefill-backend tilelang --nsa-decode-backend tilelang --cuda-graph-max-bs 64 --disable-radix-cache --model-path models/$MODEL --served-model-name $MODEL --host=0.0.0.0 --port $PORT --tensor-parallel-size 4 --trust-remote-code --tool-call-parser glm47 --reasoning-parser glm45 --mem-fraction-static 0.85 --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}"

TP=4
CONC=64
ISL=1024
OSL=1024
BASELINE=1403.43

echo "=== BATCH EXPERIMENTS $(date) ===" | tee "$SUMMARY"
echo "Baseline: ${BASELINE} total tok/s (TP=4, ds=8, CONC=64)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_one_experiment() {
    local NAME="$1"
    local EXTRA_ARGS="$2"
    local EXTRA_ENV_CMD="$3"
    local DS="$4"  # decode steps
    
    echo "========================================" | tee -a "$SUMMARY"
    echo "[$(date)] Starting: $NAME" | tee -a "$SUMMARY"
    echo "  Extra args: $EXTRA_ARGS" | tee -a "$SUMMARY"
    echo "  Extra env:  $EXTRA_ENV_CMD" | tee -a "$SUMMARY"
    echo "  Decode steps: $DS" | tee -a "$SUMMARY"
    
    eval "$BASE_ENV"
    eval "$EXTRA_ENV_CMD" 2>/dev/null || true
    
    pkill -9 -f sglang 2>/dev/null || true
    sleep 5
    
    local SERVER_LOG="/tmp/server_${NAME}.log"
    
    python3 -m sglang.launch_server \
        $BASE_ARGS \
        --num-continuous-decode-steps $DS \
        $EXTRA_ARGS \
        > "$SERVER_LOG" 2>&1 &
    
    local SERVER_PID=$!
    
    if ! wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" 2>/dev/null; then
        echo "  FAILED: Server did not start" | tee -a "$SUMMARY"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        sleep 5
        return
    fi
    
    local RESULT_FILENAME="${NAME}_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"
    
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
baseline = $BASELINE
delta = (total - baseline) / baseline * 100
print(f'  Total tok/s:    {total:.2f}')
print(f'  Per-GPU tok/s:  {per_gpu:.2f}')
print(f'  Output tok/s:   {output:.2f}')
print(f'  Mean TPOT:      {tpot:.2f} ms')
print(f'  vs baseline:    {delta:+.2f}%')
" | tee -a "$SUMMARY"
    else
        echo "  FAILED: No result file" | tee -a "$SUMMARY"
    fi
    
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
    echo "" | tee -a "$SUMMARY"
}

# Exp 3: Higher decode steps = 32
run_one_experiment "ds32" "" "" 32

# Exp 4: Higher decode steps = 64
run_one_experiment "ds64" "" "" 64

# Exp 5: Fused MoE sum all-reduce + ds=16
run_one_experiment "ds16_fusedmoe" "--enable-fused-moe-sum-all-reduce" "" 16

# Exp 6: Fused MoE + ds=32
run_one_experiment "ds32_fusedmoe" "--enable-fused-moe-sum-all-reduce" "" 32

# Exp 7: Fused MoE + ds=64
run_one_experiment "ds64_fusedmoe" "--enable-fused-moe-sum-all-reduce" "" 64

# Exp 8: Mixed chunk scheduling + ds=16
run_one_experiment "ds16_mixedchunk" "--enable-mixed-chunk --enable-fused-moe-sum-all-reduce" "" 16

# Exp 9: NCCL tuning (more channels)
run_one_experiment "ds16_nccl32ch" "--enable-fused-moe-sum-all-reduce" "export NCCL_MIN_NCHANNELS=32" 16

# Exp 10: mem-fraction 0.90 (bigger batch)
run_one_experiment "ds16_mem90" "--enable-fused-moe-sum-all-reduce --mem-fraction-static 0.90" "" 16

# Exp 11: cuda-graph-max-bs 128 + higher conc
run_one_experiment "ds16_cgbs128" "--enable-fused-moe-sum-all-reduce --cuda-graph-max-bs 128" "" 16

# Exp 12: triton MoE runner backend
run_one_experiment "ds16_moe_triton" "--enable-fused-moe-sum-all-reduce --moe-runner-backend triton" "" 16

# Exp 13: aiter decode backend for NSA
run_one_experiment "ds16_nsa_aiter" "--enable-fused-moe-sum-all-reduce --nsa-decode-backend aiter" "" 16

echo "========================================" | tee -a "$SUMMARY"
echo "[$(date)] ALL EXPERIMENTS COMPLETE" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "=== TOP RESULTS ===" | tee -a "$SUMMARY"
python3 -c "
import json, glob, os
results = []
for f in sorted(glob.glob('$RESULTS_DIR/*.json')):
    d = json.load(open(f))
    name = os.path.basename(f).replace('.json','')
    total = d['total_token_throughput']
    delta = (total - $BASELINE) / $BASELINE * 100
    results.append((total, delta, name, d['mean_tpot_ms']))

results.sort(key=lambda x: -x[0])
print(f'{'Name':50s} {'Total tok/s':>12s} {'Delta':>8s} {'TPOT ms':>10s}')
print('-'*82)
for total, delta, name, tpot in results:
    print(f'{name:50s} {total:12.2f} {delta:+7.2f}% {tpot:10.2f}')
" | tee -a "$SUMMARY"
