#!/usr/bin/env bash
# Launch script for GPT-OSS 20B MXFP4 on MI300X (vLLM)
# Mirrors InferenceX benchmarks/single_node/gptoss_fp4_mi300x.sh
#
# InferenceX recipe flags:
#   VLLM_ROCM_USE_AITER=1
#   AMDGCN_USE_BUFFER_OPS=0
#   VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
#   --attention-backend ROCM_AITER_UNIFIED_ATTN
#   -cc.pass_config.fuse_rope_kvcache=True -cc.use_inductor_graph_partition=True
#   --gpu-memory-utilization 0.95, --block-size 64, --no-enable-prefix-caching

set -euo pipefail

MODEL="${ARBOR_MODEL_PATH:?Set ARBOR_MODEL_PATH or pass --model to arbor}"
PORT="${ARBOR_PORT:-8888}"
TP="${ARBOR_TP:-2}"
GPUS="${ARBOR_SERVING_GPUS:-0,1}"

export ROCR_VISIBLE_DEVICES="$GPUS"
export HIP_VISIBLE_DEVICES="$GPUS"
unset CUDA_VISIBLE_DEVICES

export VLLM_ROCM_USE_AITER=1
export AMDGCN_USE_BUFFER_OPS=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
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
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --block-size 64 \
    --max-num-seqs 64 \
    --no-enable-prefix-caching \
    "$@"
