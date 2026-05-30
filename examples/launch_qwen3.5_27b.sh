#!/usr/bin/env bash
# Launch script for Qwen3.5-27B on MI300X (TP=8, BF16, vLLM)
#
# Usage:
#   arbor dfs \
#     --model /wekafs/models/Qwen-Qwen3.5-27B \
#     --launch-script examples/launch_qwen3.5_27b.sh

set -euo pipefail

MODEL="${ARBOR_MODEL_PATH:?Set ARBOR_MODEL_PATH or pass --model to arbor}"
PORT="${ARBOR_PORT:-8888}"
TP="${ARBOR_TP:-8}"
GPUS="${ARBOR_SERVING_GPUS:-0,1,2,3,4,5,6,7}"

export ROCR_VISIBLE_DEVICES="$GPUS"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

export VLLM_ROCM_USE_AITER=1
export HIP_FORCE_DEV_KERNARG=1

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.45 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    "$@"
