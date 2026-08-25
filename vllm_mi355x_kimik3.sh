#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie vLLM benchmark script for Kimi-K3 on MI355X.
#
# Derived from InferenceX benchmarks/vllm_mi355x.sh. The Magpie contract
# (phases, env checks, profiler args, server lifecycle, eval hand-off) is
# unchanged; the additions are the Kimi-K3 reference recipe's envs and serve
# flags, each behind an override so the optimizer can still A/B them.
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only writes PID to MAGPIE_SERVER_PID_FILE then disowns and exits.
#
# Remote server (BENCHMARK_BASE_URL): when set, the client phase points
# benchmark_serving at an external vLLM-compatible HTTP endpoint
# instead of localhost:$PORT, and forces PHASE=client (no local server
# launch). See vllm_mi300x.sh for the full contract.

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"
# shellcheck source=magpie_bench_remote_compat.sh
[[ -f "$(dirname "$0")/magpie_bench_remote_compat.sh" ]] && source "$(dirname "$0")/magpie_bench_remote_compat.sh"

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
  if [[ "$PHASE" != "client" ]]; then
    echo "[vllm_mi355x_kimik3] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
    PHASE=client
  fi
fi

if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL TP
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
fi

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

# MI355X specific: Check MEC firmware version for RCCL memory reclaim
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# ROCR_VISIBLE_DEVICES already re-indexes visible GPUs to 0..N-1, so HIP
# must use the logical range, not the original physical ids.
if [ -n "$ROCR_VISIBLE_DEVICES" ] && [ -z "$HIP_VISIBLE_DEVICES" ]; then
    n=$(echo "$ROCR_VISIBLE_DEVICES" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
fi

# vLLM optimizations for MI355X
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}

# Kimi-K3 reference recipe envs, boot-verified on this stack (gfx950, vLLM
# 0.1.dev19253). VLLM_USE_RUST_FRONTEND is omitted: this build ships no
# vllm-rs binary, so enabling it aborts startup.
export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-1}
# The 1.4 TiB checkpoint plus aiter JIT can exceed the stock engine-ready wait.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}

# Validated by Hyperloom session Kimi-K3_20260805T091405Z: 812.2 -> 888.8
# tok/s/GPU (+9.4%) at ISL8192/OSL1024/conc64/TP8, gsm8k 0.9689 -> 0.9704.
#
# The next two are an ATOMIC PAIR — enabling only the first is worse than
# leaving both off. A8W4=1 flips the routed-expert MoE to gate_mode=INTERLEAVE,
# and aiter's INTERLEAVE branch selects fp8 activation only when M >=
# AITER_BF16_FP8_MOE_BOUND (default 256). At the default bound that means fp8
# for prefill and bf16 for decode; on this checkpoint the mixed path generates
# output that never emits EOS, so the accuracy eval runs to the full max_tokens
# budget on every sample and the benchmark dies on a timeout. bound=0 puts
# decode and prefill on the same a8w4 kernel and the output is well-formed.
export AITER_SITUV2_A8W4=${AITER_SITUV2_A8W4:-1}
export AITER_BF16_FP8_MOE_BOUND=${AITER_BF16_FP8_MOE_BOUND:-0}
# qr_all_reduce is ~8.7% of GPU time at TP8. INT4 measured best; INT6 and INT8
# landed lower, and FP8 could not serve.
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

# Build profiler args for vLLM >= 0.15 (env var VLLM_TORCH_PROFILER_DIR is deprecated)
PROFILER_ARGS=()
if [[ "${PROFILE:-}" == "1" ]]; then
  TRACE_DIR="${VLLM_TORCH_PROFILER_DIR:-$WORKSPACE_DIR/torch_trace}"
  mkdir -p "$TRACE_DIR"
  PROFILER_ARGS+=(--profiler-config.profiler torch)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_dir "$TRACE_DIR")
  PROFILER_ARGS+=(--profiler-config.torch_profiler_record_shapes True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_with_memory True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_with_flops True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_use_gzip True)
fi

# Serve flags from the reference recipe that this stack rejects, all confirmed
# by boot test and therefore left off the serve line:
#   --kv-cache-dtype fp8   selects aiter's mla_gluon bh16bn128 regime, which
#                          asserts batch_size == 1 and so cannot serve.
#   --attention-config     its FLASHINFER MLA prefill backend requires a
#                          compute capability gfx950 does not report.
#   --load-format fastsafetensors  not among this build's accepted values.
#
# Speculative decoding needs the draft checkpoint present; unset by default so
# a missing draft model cannot fail the run.
SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-}

# Optional extras: only the flags whose value is conditional live here, so the
# recipe's fixed flags stay literal on the serve line where --reference-script
# can lift them (it skips any token still containing "$").
RECIPE_EXTRA_ARGS=()
[[ -n "$SPECULATIVE_CONFIG" ]] && RECIPE_EXTRA_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")

# Multi-node is opt-in; NNODES=1 keeps the stock single-node launch.
NNODES=${NNODES:-1}
if [[ "$NNODES" -gt 1 ]]; then
  RECIPE_EXTRA_ARGS+=(--nnodes "$NNODES" --node-rank "${NODE_RANK:-0}")
fi

set -x
if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  setsid vllm serve $MODEL --port $PORT \
    --tensor-parallel-size=$TP \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --enable-prefix-caching \
    --moe-backend auto \
    --load-format auto \
    --max-num-seqs 512 \
    --max-cudagraph-capture-size 256 \
    --max-num-batched-tokens 16384 \
    "${RECIPE_EXTRA_ARGS[@]}" \
    "${PROFILER_ARGS[@]}" \
    $EXTRA_VLLM_ARGS > $SERVER_LOG 2>&1 &

  SERVER_PID=$!
  if [[ "$PHASE" == "all" ]]; then
    trap 'magpie_stop_benchmark_server_stack "$SERVER_PID"' EXIT INT TERM
  fi

  wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

  if [[ "$PHASE" == "server" ]]; then
    if [[ -z "${MAGPIE_SERVER_PID_FILE:-}" ]]; then
      echo "ERROR: MAGPIE_SERVER_PID_FILE must be set for MAGPIE_RUN_PHASE=server" >&2
      kill -TERM "-$SERVER_PID" 2>/dev/null || true
      exit 3
    fi
    printf '%s\n' "$SERVER_PID" > "$MAGPIE_SERVER_PID_FILE"
    disown "$SERVER_PID" 2>/dev/null || true
    exit 0
  fi
fi

SERVER_MONITOR_ARGS=()
if [[ -n "${SERVER_PID:-}" ]]; then
  SERVER_MONITOR_ARGS+=(--server-pid "$SERVER_PID")
fi

if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
    SERVER_MONITOR_ARGS=()
    magpie_run_benchmark_serving_remote_direct trust || exit $?
  else
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts ${NUM_PROMPTS:-$(( $CONC * 10 ))} \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir "$WORKSPACE_DIR/" \
        "${SERVER_MONITOR_ARGS[@]}" \
        --trust-remote-code || exit $?
  fi
fi

if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
    if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
        if declare -F magpie_run_eval_remote_direct &>/dev/null; then
            magpie_run_eval_remote_direct || exit $?
        else
            echo "[vllm_mi355x_kimik3] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        run_eval --framework lm-eval --port "$PORT" || exit $?
        append_lm_eval_summary
    fi
fi
set +x
