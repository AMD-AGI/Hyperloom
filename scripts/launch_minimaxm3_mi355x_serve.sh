#!/usr/bin/env bash
# Serve-ONLY launch script for Hyperloom (`hyperloom optimize --launch-script ...`).
# Starts MiniMax-M3 MXFP8 on MI355X in the FOREGROUND and exposes /health.
# AITER levers enabled (shared-expert fusion + AITER RMSNorm), source-verified
# against the live vLLM install (v0.22.1rc1.dev490, rocm723) by Hyperloom agents.
#
# Hyperloom passes these via env (see hyperloom/server.py::_build_env):
#   MODEL_PATH, PORT, TP, GPUS, SESSION_DIR, GPU_TYPE  (+ CU_NUM/GPU_ARCHS)
set -eo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
PORT="${PORT:-8951}"
TP="${TP:-4}"
GPUS="${GPUS:-0,1,2,3}"

# Single GPU mask only (ROCR); setting ROCR+HIP together double-masks to nothing.
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$GPUS}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES 2>/dev/null || true

# Pin single-arch so the MXFP8 HIP kernels don't JIT-fail -> Triton fallback / rank divergence.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"

# ---------------------------------------------------------------------------
# AITER shared-expert fusion. Empirically verified gating in THIS build:
#   is_fusion_moe_shared_experts_enabled() = is_fused_moe_enabled() AND FSE
#   is_fused_moe_enabled()                 = VLLM_ROCM_USE_AITER AND VLLM_ROCM_USE_AITER_MOE
# So fusion requires ALL THREE: AITER=1, AITER_MOE=1, FSE=1.
# Pin MHA/MLA off (keep TRITON_ATTN) and LINEAR off to avoid accuracy regressors;
# MXFP8 linear is a no-op for AITER anyway (native dot_scaled kernels).
# ---------------------------------------------------------------------------
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-1}"
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
export VLLM_ROCM_USE_AITER_MHA="${VLLM_ROCM_USE_AITER_MHA:-0}"
export VLLM_ROCM_USE_AITER_MLA="${VLLM_ROCM_USE_AITER_MLA:-0}"
export VLLM_ROCM_USE_AITER_LINEAR="${VLLM_ROCM_USE_AITER_LINEAR:-0}"

# Workload-derived context length (isl+osl+256), matching the InferenceX sweep.
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 256))}"

echo "[launch] MiniMax-M3 serve (AITER: shared-expert fusion + RMSNorm)"
echo "[launch] AITER FSE=$VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS master=$VLLM_ROCM_USE_AITER moe=$VLLM_ROCM_USE_AITER_MOE rmsnorm=$VLLM_ROCM_USE_AITER_RMSNORM (mha/mla/linear off)"
echo "[launch] MODEL=$MODEL PORT=$PORT TP=$TP ROCR=$ROCR_VISIBLE_DEVICES MAX_MODEL_LEN=$MAX_MODEL_LEN"

exec vllm serve "$MODEL" --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
    --attention-backend "${ATTENTION_BACKEND:-TRITON_ATTN}" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice \
    "$@"
