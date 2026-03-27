#!/bin/bash
# GLM-5-FP8 Novel Optimization Experiments — Batch v4
# Focus: Communication reduction, overlap, and scheduling improvements
# Baseline: 1,630 tok/s (combined_nsa_mixed_ds16 = NSA aiter + mixed-chunk + ds16)
set -e

RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v4"
mkdir -p "$RESULTS_DIR"

cd /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/InferenceX
source benchmarks/benchmark_lib.sh

export MODEL="zai-org/GLM-5-FP8"
export RANDOM_RANGE_RATIO=0.8
export PORT=8888

# Default env vars
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384

TP=4
CONC=64
ISL=1024
OSL=1024

# Best config from v3 (our new baseline)
BEST_ARGS="--nsa-decode-backend aiter --enable-mixed-chunk --num-continuous-decode-steps 16"
BASE_SERVER_ARGS="--nsa-prefill-backend tilelang --cuda-graph-max-bs $CONC --disable-radix-cache --mem-fraction-static 0.85 --trust-remote-code --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 8}'"

run_single_experiment() {
    local EXP_NAME="$1"
    local EXTRA_ARGS="$2"
    local EXTRA_ENV_CMD="$3"
    
    echo ""
    echo "=================================================================="
    echo "=== EXPERIMENT: $EXP_NAME ==="
    echo "  EXTRA_ARGS: ${EXTRA_ARGS:-none}"
    echo "  EXTRA_ENV: ${EXTRA_ENV_CMD:-none}"
    echo "=================================================================="
    
    pkill -9 -f sglang 2>/dev/null || true
    sleep 8
    
    # Apply extra env vars
    eval "$EXTRA_ENV_CMD"
    
    SERVER_LOG="/tmp/server_${EXP_NAME}.log"
    
    python3 -m sglang.launch_server \
        --nsa-prefill-backend tilelang \
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
        $BEST_ARGS \
        $EXTRA_ARGS \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
        > "$SERVER_LOG" 2>&1 &
    
    SERVER_PID=$!
    
    if ! wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"; then
        echo "  FAILED: Server did not start for $EXP_NAME"
        echo "$EXP_NAME,FAILED,0,0,0,Server startup failed" >> "$RESULTS_DIR/v4_summary.csv"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        sleep 5
        return 1
    fi
    
    RESULT_FILENAME="${EXP_NAME}_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"
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
ttft = d['mean_ttft_ms']
per_gpu = total / $TP
# Compare against best v3 baseline
best_v3 = 1630.2
delta = (total - best_v3) / best_v3 * 100
print(f'  Total tok/s:    {total:.2f}')
print(f'  Per-GPU tok/s:  {per_gpu:.2f}')
print(f'  Output tok/s:   {output:.2f}')
print(f'  Mean TPOT:      {tpot:.2f} ms')
print(f'  Mean TTFT:      {ttft:.2f} ms')
print(f'  vs best v3 (1630): {delta:+.2f}%')
# Append to CSV summary
with open('$RESULTS_DIR/v4_summary.csv', 'a') as f:
    f.write(f'$EXP_NAME,{total:.2f},{per_gpu:.2f},{tpot:.2f},{ttft:.2f},{delta:+.2f}%\n')
"
    else
        echo "  WARNING: No result file found"
        echo "$EXP_NAME,NO_RESULT,0,0,0,No result file" >> "$RESULTS_DIR/v4_summary.csv"
    fi
    
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
    echo "=== DONE: $EXP_NAME ==="
}

# Header for CSV
echo "experiment,total_tok_s,per_gpu_tok_s,tpot_ms,ttft_ms,delta_vs_best_v3" > "$RESULTS_DIR/v4_summary.csv"

echo "Starting GLM-5-FP8 Novel Optimization Batch v4"
echo "Baseline: 1,630 tok/s (combined_nsa_mixed_ds16)"
echo "All experiments use: NSA aiter + mixed-chunk + ds16 as base"
echo ""

# ==========================================
# EXP 1: Reproduce baseline (sanity check)
# ==========================================
run_single_experiment "v4_baseline_repro" "" ""

# ==========================================
# EXP 2: FP8 KV cache — halves KV memory, allows more batch headroom
# ==========================================
run_single_experiment "v4_fp8_kv_cache" "--kv-cache-dtype fp8_e4m3" ""

# ==========================================
# EXP 3: Schedule conservativeness 0.5 — more aggressive batching
# ==========================================
run_single_experiment "v4_sched_conserv_05" "--schedule-conservativeness 0.5" ""

# ==========================================
# EXP 4: Schedule conservativeness 0.3
# ==========================================
run_single_experiment "v4_sched_conserv_03" "--schedule-conservativeness 0.3" ""

# ==========================================
# EXP 5: Higher decode steps 32 — more decode per schedule round
# ==========================================
run_single_experiment "v4_ds32" "--num-continuous-decode-steps 32" ""

# ==========================================
# EXP 6: Higher decode steps 64 with mixed chunk
# ==========================================
run_single_experiment "v4_ds64" "--num-continuous-decode-steps 64" ""

# ==========================================
# EXP 7: RCCL algorithm = Tree (may be faster for TP=4)
# ==========================================
run_single_experiment "v4_rccl_tree" "" "export NCCL_ALGO=Tree"

# ==========================================
# EXP 8: RCCL algorithm = Ring
# ==========================================
run_single_experiment "v4_rccl_ring" "" "export NCCL_ALGO=Ring"

# ==========================================
# EXP 9: Quick reduce INT8 instead of INT4 (less quantization noise)
# ==========================================
run_single_experiment "v4_qr_int8" "" "export ROCM_QUICK_REDUCE_QUANTIZATION=INT8"

# ==========================================
# EXP 10: Quick reduce FP (no quantization)
# ==========================================
run_single_experiment "v4_qr_fp" "" "export ROCM_QUICK_REDUCE_QUANTIZATION=FP"

# ==========================================
# EXP 11: Disable custom allreduce — force NCCL/RCCL
# ==========================================
run_single_experiment "v4_no_custom_ar" "--disable-custom-all-reduce" ""

# ==========================================
# EXP 12: Aiter allreduce fusion + higher decode steps
# ==========================================
run_single_experiment "v4_ar_fusion_ds32" "--enable-aiter-allreduce-fusion --num-continuous-decode-steps 32" ""

# ==========================================
# EXP 13: Chunked prefill size 65536 (smaller chunks with mixed-chunk)
# ==========================================
run_single_experiment "v4_chunk_65536" "--chunked-prefill-size 65536" ""

# ==========================================
# EXP 14: Chunked prefill size 32768
# ==========================================
run_single_experiment "v4_chunk_32768" "--chunked-prefill-size 32768" ""

# ==========================================
# EXP 15: Memory fraction 0.90 + FP8 KV — maximize batch capacity
# ==========================================
run_single_experiment "v4_mem90_fp8kv" "--mem-fraction-static 0.90 --kv-cache-dtype fp8_e4m3" ""

# ==========================================
# EXP 16: RCCL with more channels (64) + Tree algorithm
# ==========================================
run_single_experiment "v4_rccl_tree_64ch" "" "export NCCL_ALGO=Tree; export NCCL_MIN_NCHANNELS=64"

echo ""
echo "=================================================================="
echo "=== ALL v4 EXPERIMENTS COMPLETE ==="
echo "=================================================================="
echo ""
echo "Summary:"
cat "$RESULTS_DIR/v4_summary.csv"
