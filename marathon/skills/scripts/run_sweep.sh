#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ============================================================
# Inference Optimization — Parameter Sweep
# Supports both SGLang and vLLM serving frameworks.
#
# Required env vars: MODEL, TP, INFERENCEX_PATH
# Optional: CONC_VALUES, ISL_OSL_CONFIGS, PORT, RESULT_DIR,
#           NUM_PROMPTS_MULTIPLIER, MAX_OUTPUT_TOKENS (default 2M),
#           FRAMEWORK (sglang|vllm, default sglang),
#           SGLANG_EXTRA_ARGS, VLLM_EXTRA_ARGS
# ============================================================

unset PROFILE SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR

: "${MODEL:?MODEL env var required}"
: "${TP:?TP env var required}"
: "${INFERENCEX_PATH:?INFERENCEX_PATH env var required}"

FRAMEWORK="${FRAMEWORK:-sglang}"
PORT=${PORT:-8888}
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
RESULT_DIR="${RESULT_DIR:-/shared_nfs/inference-optimization/results/sweep_$TIMESTAMP}"
CONC_VALUES="${CONC_VALUES:-4 16 64}"
ISL_OSL_CONFIGS="${ISL_OSL_CONFIGS:-1024:1024 8192:1024 1024:8192}"
MAX_CONC=$(echo "$CONC_VALUES" | tr ' ' '\n' | sort -n | tail -1)
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2000000}"

mkdir -p "$RESULT_DIR"

check_benchmark_lib "$INFERENCEX_PATH"
cd "$INFERENCEX_PATH"

sweep_error_cleanup() {
    kill_server 2>/dev/null || true
}
trap sweep_error_cleanup ERR
trap 'kill_server 2>/dev/null || true; exit 130' INT
trap 'kill_server 2>/dev/null || true; exit 143' TERM

echo "framework	conc	isl	osl	output_tput	total_tput	ttft_ms	tpot_ms	itl_ms	e2el_ms	status	description" > "$RESULT_DIR/results.tsv"

launch_server() {
    local LOG_FILE="$1"
    if [ "$FRAMEWORK" = "vllm" ]; then
        VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
        vllm serve "$MODEL" \
            --host 0.0.0.0 --port $PORT --tensor-parallel-size $TP \
            --trust-remote-code \
            --gpu-memory-utilization 0.85 --disable-log-stats \
            $VLLM_EXTRA_ARGS \
            > "$LOG_FILE" 2>&1 &
        SERVER_PID=$!
        wait_for_health "$PORT" "$LOG_FILE" "$SERVER_PID"
    else
        export SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
        SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
        if [ -n "${MEM_FRACTION:-}" ]; then
            : # user explicitly set MEM_FRACTION
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
            --cuda-graph-max-bs $MAX_CONC \
            $SGLANG_EXTRA_ARGS \
            > "$LOG_FILE" 2>&1 &
        SERVER_PID=$!
        wait_for_health "$PORT" "$LOG_FILE" "$SERVER_PID"
    fi
}

compute_num_prompts() {
    local conc=$1 osl=$2
    if [ -n "${NUM_PROMPTS_MULTIPLIER:-}" ]; then
        echo $((conc * NUM_PROMPTS_MULTIPLIER))
        return
    fi
    if [ "$osl" -gt 4096 ]; then
        echo $((conc * 2))
    elif [ "$osl" -gt 1024 ]; then
        echo $((conc * 3))
    else
        echo $((conc * 5))
    fi
}

CONFIG_LIST=""
for ISL_OSL in $ISL_OSL_CONFIGS; do
    IFS=':' read -r ISL OSL <<< "$ISL_OSL"
    for CONC in $CONC_VALUES; do
        NP=$(compute_num_prompts "$CONC" "$OSL")
        COST=$((NP * OSL))
        CONFIG_LIST="${CONFIG_LIST}${COST} ${ISL} ${OSL} ${CONC} ${NP}
"
    done
done
SORTED_CONFIGS=$(echo "$CONFIG_LIST" | sort -n)

TOTAL_CONFIGS=$(echo "$SORTED_CONFIGS" | grep -c '[0-9]')

echo "============================================================"
echo "Parameter Sweep (single-server, smart-ordered)"
echo "Framework: $FRAMEWORK | Model: $MODEL | TP=$TP"
echo "CONC values: $CONC_VALUES"
echo "ISL/OSL configs: $ISL_OSL_CONFIGS"
echo "Total configs: $TOTAL_CONFIGS | MAX_OUTPUT_TOKENS: $MAX_OUTPUT_TOKENS"
echo "Results: $RESULT_DIR"
echo "============================================================"
echo ""
echo "Execution order (short → long):"
echo "$SORTED_CONFIGS" | while IFS=' ' read -r COST ISL OSL CONC NP; do
    [ -z "$COST" ] && continue
    if [ "$COST" -gt "$MAX_OUTPUT_TOKENS" ]; then
        echo "  SKIP  ISL=$ISL OSL=$OSL CONC=$CONC (${COST} output tokens > limit)"
    else
        echo "  RUN   ISL=$ISL OSL=$OSL CONC=$CONC n=$NP (${COST} output tokens)"
    fi
done
echo ""

kill_server
launch_server "$RESULT_DIR/server.log"

DONE=0
SWEEP_START=$(date +%s)

while IFS=' ' read -r COST ISL OSL CONC NUM_PROMPTS; do
    [ -z "$COST" ] && continue

    if [ "$COST" -gt "$MAX_OUTPUT_TOKENS" ]; then
        echo "--- SKIP ISL=$ISL OSL=$OSL CONC=$CONC (${COST} output tokens > MAX_OUTPUT_TOKENS=$MAX_OUTPUT_TOKENS) ---"
        echo "$FRAMEWORK	$CONC	$ISL	$OSL	-	-	-	-	-	-	skipped	output_tokens=${COST} > limit=${MAX_OUTPUT_TOKENS}" >> "$RESULT_DIR/results.tsv"
        continue
    fi

    DONE=$((DONE + 1))
    ELAPSED=$(( $(date +%s) - SWEEP_START ))
    RESULT_FILENAME="${FRAMEWORK}_tp${TP}_conc${CONC}_isl${ISL}_osl${OSL}"
    echo "--- [$DONE/$TOTAL_CONFIGS +${ELAPSED}s] $RESULT_FILENAME (n=$NUM_PROMPTS, ~${COST} output tokens) ---"

    export RANDOM_RANGE_RATIO=1.0 RESULT_FILENAME
    # --backend vllm: InferenceX benchmark uses "vllm" for any OpenAI-compatible API (both SGLang and vLLM)
    run_benchmark_serving \
        --model "$MODEL" --port "$PORT" --backend vllm \
        --input-len "$ISL" --output-len "$OSL" --random-range-ratio 1.0 \
        --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
        --trust-remote-code \
        --result-filename "$RESULT_FILENAME" --result-dir "$RESULT_DIR/"

    python3 -c "
import json
with open('$RESULT_DIR/${RESULT_FILENAME}.json') as f:
    d = json.load(f)
line = f'$FRAMEWORK\t$CONC\t$ISL\t$OSL\t{d[\"output_throughput\"]:.2f}\t{d[\"total_token_throughput\"]:.2f}\t{d[\"mean_ttft_ms\"]:.2f}\t{d[\"mean_tpot_ms\"]:.2f}\t{d[\"mean_itl_ms\"]:.2f}\t{d[\"mean_e2el_ms\"]:.2f}\tswept\tCONC=$CONC ISL=$ISL OSL=$OSL (n=$NUM_PROMPTS)'
print(line)
with open('$RESULT_DIR/results.tsv', 'a') as f:
    f.write(line + '\n')
"
done <<< "$SORTED_CONFIGS"

kill_server

TOTAL_ELAPSED=$(( $(date +%s) - SWEEP_START ))
echo ""
echo "=== SWEEP COMPLETE in ${TOTAL_ELAPSED}s ==="
echo "Results: $RESULT_DIR/results.tsv"
cat "$RESULT_DIR/results.tsv"
