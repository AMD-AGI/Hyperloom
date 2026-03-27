#!/bin/bash
# Combined best optimizations experiment
# Run after resume_on_new_node.sh
set -euo pipefail

RESULTS_DIR="/shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization/results_v3"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/combined_summary.txt"
echo "========== COMBINED BEST EXPERIMENTS $(date) ==========" > "$SUMMARY"

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

run_one() {
    local name="$1"
    local ds="${2:-16}"
    local extra_args="${3:-}"
    local extra_env="${4:-}"
    local conc="${5:-64}"

    echo ""
    echo "======== $name (DS=$ds CONC=$conc) ========"
    echo "  ARGS: $extra_args"
    echo "  ENV: $extra_env"
    echo "$name DS=$ds CONC=$conc" >> "$SUMMARY"

    pkill -9 -f sglang 2>/dev/null || true
    sleep 5

    eval "$extra_env" 2>/dev/null || true

    local log="/tmp/server_${name}.log"
    python3 -m sglang.launch_server \
        --nsa-prefill-backend tilelang \
        --nsa-decode-backend aiter \
        --cuda-graph-max-bs $conc \
        --disable-radix-cache \
        --model-path models/$MODEL \
        --served-model-name $MODEL \
        --host=0.0.0.0 --port $PORT \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --tool-call-parser glm47 --reasoning-parser glm45 \
        --mem-fraction-static 0.85 \
        --num-continuous-decode-steps $ds \
        --enable-mixed-chunk \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
        $extra_args \
        > "$log" 2>&1 &

    local spid=$!
    wait_for_server_ready --port "$PORT" --server-log "$log" --server-pid "$spid"

    local fname="${name}_tp4_isl1024_osl1024_conc${conc}"
    run_benchmark_serving \
        --model "$MODEL" --port "$PORT" --backend vllm \
        --input-len 1024 --output-len 1024 \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$((conc * 3))" --max-concurrency "$conc" \
        --result-filename "$fname" --result-dir "$RESULTS_DIR/"

    if [ -f "$RESULTS_DIR/${fname}.json" ]; then
        python3 -c "
import json
d = json.load(open('$RESULTS_DIR/${fname}.json'))
total = d['total_token_throughput']
per_gpu = total / 4
tpot = d['mean_tpot_ms']
out = d['output_throughput']
baseline = 1403.43
delta = (total - baseline) / baseline * 100
line = f'  total={total:.1f} per_gpu={per_gpu:.1f} out={out:.1f} tpot={tpot:.1f}ms delta={delta:+.1f}%'
print(line)
with open('$SUMMARY', 'a') as f:
    f.write(line + '\n')
"
    fi
    kill $spid 2>/dev/null || true
    wait $spid 2>/dev/null || true
    sleep 3
}

# ---------- EXPERIMENTS ----------

# 1. Combined best: NSA aiter + mixed-chunk + ds=16 (both winners together)
run_one "combined_aiter_mixed_ds16" 16

# 2. Same with ds=8 (original baseline decode steps)
run_one "combined_aiter_mixed_ds8" 8

# 3. Same with ds=64 (aggressive batching)
run_one "combined_aiter_mixed_ds64" 64

# 4. Combined + allreduce fusion
run_one "combined_allreduce_ds16" 16 "--enable-aiter-allreduce-fusion"

# 5. Combined + NCCL tuning
run_one "combined_nccl32ch_ds16" 16 "" "export NCCL_MIN_NCHANNELS=32"

# 6. Combined + NCCL Ring algo
run_one "combined_nccl_ring_ds16" 16 "" "export NCCL_ALGO=Ring"

# 7. Combined + NCCL Tree algo
run_one "combined_nccl_tree_ds16" 16 "" "export NCCL_ALGO=Tree"

# 8. Combined + mscclpp
run_one "combined_mscclpp_ds16" 16 "--enable-mscclpp"

echo ""
echo "========== ALL COMBINED EXPERIMENTS COMPLETE =========="
cat "$SUMMARY"
