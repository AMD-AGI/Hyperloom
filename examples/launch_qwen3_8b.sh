#!/usr/bin/env bash
# Launch script for Qwen3-8B on MI300X (TP=2, BF16, vLLM)
# Uses InferenceX-style optimization flags:
#   AITER, unified attention, fuse_rope_kvcache, inductor graph partition
#
# Usage:
#   arbor dfs \
#     --model /wekafs/models/Qwen-Qwen3-8B \
#     --launch-script examples/launch_qwen3_8b.sh \
#     --tp 2 --serving-gpus 0,1

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
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    -cc.pass_config.fuse_rope_kvcache=True \
    -cc.use_inductor_graph_partition=True \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --block-size 64 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    "$@"
