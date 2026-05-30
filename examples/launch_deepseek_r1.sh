#!/bin/bash
# Launch script for DeepSeek-R1-0528 on MI300X (TP=8)
# Arbor injects: ARBOR_MODEL_PATH, ARBOR_PORT, ARBOR_TP, ARBOR_SERVING_GPUS
#
# Optimization history:
#   v1 (baseline): AITER disabled, no TRITON_ROPE, no QR threshold, HSA_NO_SCRATCH_RECLAIM=1
#                  Throughput: ~1296 tok/s (pre-AITER)
#
#   v2 (20260518-session-1): AITER=1, TRITON_ROPE=1, QR INT4, GPU_MAX_HW_QUEUES=2,
#                  fuse_rope_kvcache, use_inductor_graph_partition,
#                  remove HSA_NO_SCRATCH_RECLAIM
#                  gpu-memory-utilization=0.95 (NOTE: OOM crash observed in some runs
#                  with benchmark_combo_kernel=True; added benchmark_combo_kernel=False fix)
#                  Throughput: 1686.33 tok/s (+30.1% over baseline)
#
#   v3 (20260518-231205): QR bf16 TP=8 bug fixed in quick_all_reduce.py line 57.
#                  bfloat16 TP=8 threshold changed from [2048,2048,2048] MB → [4,4,2] MB.
#                  QR INT4 now fires natively for bf16 prefill tensors >4 MB.
#                  (With CAST_BF16_TO_FP16=1: QR fires via fp16 path at >2 MB — already working)
#                  PENDING TEST: gpu-memory-utilization=0.95 + benchmark_combo_kernel=False
#                  (0.95 showed +18.5% vs 0.45 in session 20260518-222818)

VLLM_ROCM_USE_AITER=1 \
HIP_FORCE_DEV_KERNARG=1 \
VLLM_ROCM_USE_AITER_TRITON_ROPE=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
AMDGCN_USE_BUFFER_OPS=1 \
GPU_MAX_HW_QUEUES=2 \
python3 -m vllm.entrypoints.openai.api_server \
  --model "${ARBOR_MODEL_PATH:-/wekafs/models/DeepSeek-R1-0528}" \
  --tensor-parallel-size "${ARBOR_TP:-8}" \
  --host 0.0.0.0 \
  --port "${ARBOR_PORT:-8888}" \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --attention-backend ROCM_AITER_MLA \
  --enable-chunked-prefill \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8_e4m3 \
  -cc.pass_config.fuse_rope_kvcache=True \
  -cc.use_inductor_graph_partition=True \
  -cc.inductor_compile_config.benchmark_combo_kernel=False
