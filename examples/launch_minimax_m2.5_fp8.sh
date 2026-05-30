#!/usr/bin/env bash
# Launch script for MiniMax-M2.5 FP8 on MI300X (vLLM)
# Mirrors InferenceX benchmarks/single_node/minimaxm2.5_fp8_mi300x.sh
#
# InferenceX recipe flags:
#   VLLM_ROCM_USE_AITER=1
#   --gpu-memory-utilization 0.95
#   --block-size 32
#   --disable-log-requests
#   --no-enable-prefix-caching
#   --trust-remote-code

set -euo pipefail

MODEL="${ARBOR_MODEL_PATH:?Set ARBOR_MODEL_PATH or pass --model to arbor}"
PORT="${ARBOR_PORT:-8888}"
TP="${ARBOR_TP:-2}"
GPUS="${ARBOR_SERVING_GPUS:-0,1}"

export ROCR_VISIBLE_DEVICES="$GPUS"
export HIP_VISIBLE_DEVICES="$GPUS"
unset CUDA_VISIBLE_DEVICES

export VLLM_ROCM_USE_AITER=1
export HIP_FORCE_DEV_KERNARG=1

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --block-size 32 \
    --max-num-seqs 64 \
    --no-enable-prefix-caching \
    "$@"
