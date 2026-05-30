#!/usr/bin/env bash
# MiniMax-M2.5 FP8 — Marathon-Optimized Launch Script for MI300X
#
# Ported from Agentic-InferenceX/MiniMax-M2.5-marathon-optimized/scripts/
# Includes: aiter patches, FMoE tuning CSVs, and optimized env vars.
#
# Prerequisites:
#   - 8x MI300X GPUs (TP=8 for full BF16/FP8 model)
#   - vLLM with ROCm support installed
#   - aiter package installed
#
# Usage:
#   MODEL_PATH=/wekafs/models/MiniMaxAI-MiniMax-M2.5 bash examples/launch_minimax_m2.5_marathon.sh
#   MODEL_PATH=/wekafs/models/MiniMaxAI-MiniMax-M2.5 TP=2 PORT=8888 bash examples/launch_minimax_m2.5_marathon.sh
#
# Environment variables:
#   MODEL_PATH  — Path to MiniMax-M2.5 model weights (required)
#   TP          — Tensor parallelism degree (default: 8)
#   PORT        — Server port (default: 8000)
#   MAX_LEN     — Max sequence length (default: 4096)
#   BACKGROUND  — Set to 1 to launch in background and wait for health

set -euo pipefail

MODEL="${MODEL_PATH:?Set MODEL_PATH to the MiniMax-M2.5 weights directory}"
TP="${TP:-8}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
BACKGROUND="${BACKGROUND:-0}"

# ── Detect aiter installation ────────────────────────────────────
AITER_DIR="$(python3 -c 'import aiter, os; print(os.path.dirname(aiter.__file__))' 2>/dev/null || echo '')"
if [ -z "$AITER_DIR" ]; then
    echo "WARNING: aiter not found. Running without aiter optimizations."
    echo "         Install aiter for +47% throughput on MoE kernels."
fi

# ── Marathon environment flags ───────────────────────────────────
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export HIP_FORCE_DEV_KERNARG=1

# Disable v1 engine if flash_attn has ABI issues with this container
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_FLASH_ATTN}"

# ── Apply FMoE tuning CSVs if available ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/../vendor/inferencex/patches/minimax-m2.5"

if [ -n "$AITER_DIR" ] && [ -d "$PATCH_DIR" ]; then
    MODEL_CONFIGS_DIR="$AITER_DIR/configs/model_configs"
    mkdir -p "$MODEL_CONFIGS_DIR" 2>/dev/null || true
    for csv in "$PATCH_DIR"/*.csv; do
        [ -f "$csv" ] || continue
        dst="$MODEL_CONFIGS_DIR/$(basename "$csv")"
        if [ ! -f "$dst" ]; then
            cp "$csv" "$dst"
            echo "Installed tuning config: $(basename "$csv")"
        fi
    done
fi

# ── Build launch command ─────────────────────────────────────────
CMD=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization 0.95
    --max-model-len "$MAX_LEN"
    --block-size 32
    --kv-cache-dtype fp8_e4m3
    --trust-remote-code
    --enforce-eager
    --disable-log-requests
    --no-enable-prefix-caching
    --max-num-seqs 64
)

echo "============================================"
echo " MiniMax-M2.5 — Marathon Optimized (MI300X)"
echo " Model: $MODEL"
echo " TP=$TP  Port=$PORT  MaxLen=$MAX_LEN"
echo " AITER=$([ -n "$AITER_DIR" ] && echo ON || echo OFF)"
echo " TRITON_ROPE=1  SHUFFLE_KV=1"
echo " Engine: v${VLLM_USE_V1:-0}  Attn: $VLLM_ATTENTION_BACKEND"
echo "============================================"

if [ "$BACKGROUND" -eq 1 ]; then
    LOG="/tmp/minimax_m25_marathon_server.log"
    nohup "${CMD[@]}" > "$LOG" 2>&1 &
    PID=$!
    echo "Server PID: $PID (log: $LOG)"
    echo "Waiting for health..."
    for i in $(seq 1 90); do
        if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            echo "Server healthy after $((i * 10))s"
            exit 0
        fi
        sleep 10
    done
    echo "ERROR: Server not healthy after 900s — check $LOG"
    exit 1
else
    exec "${CMD[@]}"
fi
