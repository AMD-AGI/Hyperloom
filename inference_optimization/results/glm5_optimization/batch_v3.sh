#!/bin/bash
# Batch experiments v3: systematic per-GPU optimization
# Each experiment restarts the server with different args/env, runs benchmark, logs results
# Tuned GEMMs are now in /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv
set -euo pipefail

RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v3"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/batch_v3_summary.txt"
echo "========== BATCH V3 EXPERIMENTS $(date) ==========" > "$SUMMARY"

cd /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX
source benchmarks/benchmark_lib.sh

export MODEL="zai-org/GLM-5-FP8"
export RANDOM_RANGE_RATIO=0.8
export PORT=8888

BASE_ENV=(
    SGLANG_USE_AITER=1
    RCCL_MSCCL_ENABLE=0
    SGLANG_ROCM_FUSED_DECODE_MLA=0
    ROCM_QUICK_REDUCE_QUANTIZATION=INT4
    SAFETENSORS_FAST_GPU=1
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
)

run_experiment() {
    local name="$1"
    local tp="${2:-4}"
    local conc="${3:-64}"
    local ds="${4:-8}"
    local extra_server_args="${5:-}"
    local extra_env="${6:-}"
    local mem_frac="${7:-0.85}"

    echo ""
    echo "================================================================"
    echo "  EXP: $name | TP=$tp CONC=$conc DS=$ds MEM=$mem_frac"
    echo "  EXTRA_ARGS: $extra_server_args"
    echo "  EXTRA_ENV: $extra_env"
    echo "================================================================"
    echo "EXP: $name | TP=$tp CONC=$conc DS=$ds" >> "$SUMMARY"

    pkill -9 -f sglang 2>/dev/null || true
    sleep 5

    for var in "${BASE_ENV[@]}"; do export "$var"; done
    eval "$extra_env" 2>/dev/null || true

    local server_log="/tmp/server_${name}.log"
    python3 -m sglang.launch_server \
        --nsa-prefill-backend tilelang \
        --nsa-decode-backend tilelang \
        --cuda-graph-max-bs "$conc" \
        --disable-radix-cache \
        --model-path "models/$MODEL" \
        --served-model-name "$MODEL" \
        --host=0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size "$tp" \
        --trust-remote-code \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --mem-fraction-static "$mem_frac" \
        --num-continuous-decode-steps "$ds" \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
        $extra_server_args \
        > "$server_log" 2>&1 &

    local spid=$!
    if ! wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$spid"; then
        echo "  FAILED: server did not start" | tee -a "$SUMMARY"
        kill $spid 2>/dev/null || true
        return 1
    fi

    local fname="${name}_tp${tp}_isl1024_osl1024_conc${conc}"
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len 1024 \
        --output-len 1024 \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$((conc * 3))" \
        --max-concurrency "$conc" \
        --result-filename "$fname" \
        --result-dir "$RESULTS_DIR/"

    if [ -f "$RESULTS_DIR/${fname}.json" ]; then
        python3 -c "
import json, sys
d = json.load(open('$RESULTS_DIR/${fname}.json'))
total = d['total_token_throughput']
output = d['output_throughput']
tpot = d['mean_tpot_ms']
per_gpu = total / $tp
baseline = 1403.43
delta = (total - baseline) / baseline * 100
line = f'  total={total:.1f} per_gpu={per_gpu:.1f} out={output:.1f} tpot={tpot:.1f}ms delta={delta:+.1f}%'
print(line)
with open('$SUMMARY', 'a') as f:
    f.write(line + '\n')
"
    else
        echo "  NO RESULT FILE" | tee -a "$SUMMARY"
    fi

    kill $spid 2>/dev/null || true
    wait $spid 2>/dev/null || true
    sleep 3
}

# ========== EXPERIMENTS ==========

# EXP 1: Tuned GEMMs + fused-moe-sum-all-reduce + ds=16 (combine best known config at TP=4)
run_experiment "tuned_fusedmoe_ds16" 4 64 16 "--enable-fused-moe-sum-all-reduce"

# EXP 2: Same but higher concurrency 128
run_experiment "tuned_fusedmoe_ds16_conc128" 4 128 16 "--enable-fused-moe-sum-all-reduce --cuda-graph-max-bs 128"

# EXP 3: ds=32 + fused moe (more batching of decode steps)
run_experiment "tuned_fusedmoe_ds32" 4 64 32 "--enable-fused-moe-sum-all-reduce"

# EXP 4: ds=64 + fused moe
run_experiment "tuned_fusedmoe_ds64" 4 64 64 "--enable-fused-moe-sum-all-reduce"

# EXP 5: Triton MoE backend (different kernel implementation)
run_experiment "tuned_moe_triton_ds16" 4 64 16 "--enable-fused-moe-sum-all-reduce --moe-runner-backend triton"

# EXP 6: Mixed chunk scheduling (overlap prefill+decode)
run_experiment "tuned_mixedchunk_ds16" 4 64 16 "--enable-fused-moe-sum-all-reduce --enable-mixed-chunk"

# EXP 7: NSA aiter decode backend
run_experiment "tuned_nsa_aiter_ds16" 4 64 16 "--enable-fused-moe-sum-all-reduce --nsa-decode-backend aiter"

# EXP 8: mem-fraction 0.90 (larger KV cache = bigger batches)
run_experiment "tuned_mem90_ds16" 4 64 16 "--enable-fused-moe-sum-all-reduce" "" "0.90"

# EXP 9: All-reduce fusion + fused moe + ds=16 (combine the two best NCCL opts)
run_experiment "tuned_allreduce_fusedmoe_ds16" 4 64 16 "--enable-fused-moe-sum-all-reduce --enable-aiter-allreduce-fusion"

# EXP 10: NCCL tuning - more channels
run_experiment "tuned_nccl32ch_ds16" 4 64 16 "--enable-fused-moe-sum-all-reduce" "export NCCL_MIN_NCHANNELS=32"

# EXP 11: Best combo with ds=32 and conc=128
run_experiment "tuned_fusedmoe_ds32_conc128" 4 128 32 "--enable-fused-moe-sum-all-reduce --cuda-graph-max-bs 128"

# EXP 12: Aggressive - combine everything promising (allreduce fusion + fused moe + mixed chunk + ds=16)
run_experiment "tuned_kitchen_sink_ds16" 4 64 16 \
    "--enable-fused-moe-sum-all-reduce --enable-aiter-allreduce-fusion --enable-mixed-chunk"

echo ""
echo "========== ALL EXPERIMENTS COMPLETE =========="
echo ""
cat "$SUMMARY"
