#!/usr/bin/env bash
# Launch script for MiniMax-M2.5 on MI300X (vLLM).
# Patched v5: combines proven v3 CLI flags (EP/fp8KV/chunked-prefill/async/
# bumped batched tokens) with v4 env additions from the documented +40% recipe
# (QuickReduce INT4, AITER unified attn, GPU_MAX_HW_QUEUES=2, MOE_PADDING=0).
#
# v3 alone measured: 1885.86 tok/s (+14.0% over 1654.6 baseline).
# v4 env additions are documented to give additional headroom on top.

set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
GPUS="${GPUS:-0,1}"

# --- GPU visibility ---
export CUDA_VISIBLE_DEVICES="$GPUS"
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true

# --- Container-specific workarounds (ABI shim) ---
if [ -f /usr/local/lib/libhip_compat_shim.so ]; then
    TORCH_LIB="/usr/local/lib/python3.12/dist-packages/torch/lib"
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/usr/local/lib/libhip_compat_shim.so:${TORCH_LIB}/libtorch_hip.so"
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

export CU_NUM="${CU_NUM:?CU_NUM not set — run via hyperloom or set manually}"
export GPU_ARCHS="${GPU_ARCHS:?GPU_ARCHS not set — run via hyperloom or set manually}"

# --- AITER ROCm kernel toggles (proven baseline from v3) ---
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1

# --- v4 additions: documented winning recipe env vars (TRACE_ANALYSIS_MiniMax-M2.5.md) ---
# A002: QuickReduce INT4 for large AllReduces (+7.7% reported)
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
# A012-companion: AITER unified attention backend
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
# A036: reduce HW queue contention
export GPU_MAX_HW_QUEUES=2
# A037: eliminate wasted MoE padding
export VLLM_ROCM_MOE_PADDING=0

# --- HIP runtime tunings (lower launch latency) ---
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export AITER_ROCM_ARCH="gfx942;gfx950"

# --- Serve (v3 CLI flags — proven to give +14%) ---
exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --compilation-config '{"use_inductor_graph_partition":true,"fast_moe_cold_start":true}' \
    "$@"
