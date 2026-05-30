#!/usr/bin/env bash
# DeepSeek-R1-0528 on vLLM (MI300X, FP8, TP=8)
# Matching the marathon-session 20260430-075928 config
set -euo pipefail

MODEL_PATH="${ARBOR_MODEL_PATH:-/wekafs/models/DeepSeek-R1-0528}"
PORT="${ARBOR_PORT:-8888}"
TP="${ARBOR_TP:-8}"

export VLLM_ROCM_USE_AITER=1
export HSA_NO_SCRATCH_RECLAIM=1
unset VLLM_ROCM_MOE_N_SPLIT_SCHEDULE 2>/dev/null || true

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --attention-backend ROCM_AITER_MLA \
  --enable-chunked-prefill \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  "$@"
