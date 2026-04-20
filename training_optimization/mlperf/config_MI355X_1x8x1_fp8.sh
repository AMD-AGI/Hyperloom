#!/bin/bash
# =============================================================================
# MLPerf GPT-OSS-20B Configuration for MI355X (1 node, 8 GPUs)
# =============================================================================

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
export DGXSYSTEM=MI355X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
export PRIMUS_PATH=/workspace/Primus
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP=/root/mlperf_primus/conf/gpt_oss_20B-pretrain-fp8.yaml
export EXP_NAME=gpt_oss_20b_20260407
export DATADIR=/shared_nfs/huangwei/gpt_oss_20b/data
export DATA_PATH=/shared_nfs/huangwei/gpt_oss_20b/data
export CONT=tasimage/primus:gpt_oss_20b_training_5.1
export MODELDIR=/shared_nfs/huangwei/gpt_oss_20b/model
export LOGDIR=/root/mlperf_primus/logs
export CLEAR_CACHES=0
export HF_TOKEN="hf_REDACTED"
export WANDB_API_KEY="your_api_key"
#export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
export PRIMUS_MICRO_BATCH_SIZE=2
export PRIMUS_GLOBAL_BATCH_SIZE=32
export PRIMUS_LR=4.0e-4
export PRIMUS_MIN_LR=4.0e-5 # Set to 10% of max LR
export PRIMUS_TRAIN_ITERS=1200000      # 1.2M iters × 16 GBS = 19.2B samples
export PRIMUS_LR_WARMUP_ITERS=128
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS)) # 1200000 - 128 = 1199872
# export SEED=30279

# Evaluation frequency (sample-based, adjusts automatically with GBS)
export EVAL_SAMPLES_INTERVAL=12288   # Evaluate every 12,288 samples
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))  # Auto-computed

# -----------------------------------------------------------------------------
# Optimizations
# -----------------------------------------------------------------------------
export PRIMUS_APPLY_ROPE_FUSION=True
export PRIMUS_FP8_RECIPE=hybrid
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TORCH_NCCL_HIGH_PRIORITY=1
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

# -----------------------------------------------------------------------------
# MLPerf Logging
# -----------------------------------------------------------------------------
export MLLOG_TRAIN_LOSS_LOG_FREQ=32
export MLLOG_TARGET_EVAL_LOSS=3.34
export MLLOG_OUTPUT_FILE=/root/mlperf_primus/outputs/output.log
export MLLOG_SAVE_TO_FILE=0
export MLLOG_SUBMISSION_BENCHMARK=gpt-oss-20b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X

# -----------------------------------------------------------------------------
# TE Configuration
# -----------------------------------------------------------------------------
export NVTE_ROCM_ENABLE_MXFP8=0
# FP8 has a lot of small kernels where the _cast_tranpose_triton can be a bottleneck. Enable the optimized version which merges the cast and transpose into one kernel and is further optimized for AMD GPUs
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
