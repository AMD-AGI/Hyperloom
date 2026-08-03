#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORLDMIRROR_REPO_URL="${WORLDMIRROR_REPO_URL:-https://github.com/Tencent-Hunyuan/HY-World-2.0.git}"
WORLDMIRROR_DIR="${WORLDMIRROR_DIR:-${WORLDMIRROR_REPO_PATH:-}}"
if [ -z "${WORLDMIRROR_DIR}" ]; then
    _CACHE_ROOT="${HYPERLOOM_CACHE_DIR:-${HOME}/.cache/hyperloom}"
    WORLDMIRROR_DIR="${_CACHE_ROOT}/HY-World-2.0"
fi

if [ ! -d "${WORLDMIRROR_DIR}" ]; then
    mkdir -p "$(dirname "${WORLDMIRROR_DIR}")"
    git clone --depth 1 "${WORLDMIRROR_REPO_URL}" "${WORLDMIRROR_DIR}"
fi

export WORLDMIRROR_DIR
export WORLDMIRROR_REPO_PATH="${WORLDMIRROR_DIR}"
export WORLDMIRROR_BENCH="${WORLDMIRROR_BENCH:-${_SCRIPT_DIR}/worldmirror_bench.py}"

_SCRIPT_NAME="$(basename "$0")"
_SCRIPT_RUNNER="${_SCRIPT_NAME#worldmirror_}"
_SCRIPT_RUNNER="${_SCRIPT_RUNNER%.sh}"
export RUNNER_TYPE="${RUNNER_TYPE:-${_SCRIPT_RUNNER:-mi355x}}"

if [ ! -f "${WORLDMIRROR_BENCH}" ]; then
    echo "[worldmirror] benchmark driver not found. Set WORLDMIRROR_BENCH." >&2
    exit 2
fi

unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    n=$(echo "${ROCR_VISIBLE_DEVICES}" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1)))
fi
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${WORLDMIRROR_DIR}:${PYTHONPATH:-}"

RESULT_DIR="${RESULT_DIR:?RESULT_DIR must be set by Magpie}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
MODEL_PATH="${MODEL:?MODEL must be set}"
TP_DEG="${TP:-1}"
TARGET_SIZE="${WM_TARGET_SIZE:-518}"
WARMUP="${WM_WARMUP_CALLS:-3}"
ITERATIONS="${WM_NUM_ITERATIONS:-5}"
SCENES="${WM_SCENES:-}"
INPUT_PATH="${WM_INPUT_PATH:-}"
REL_MAX="${WM_QUALITY_REL_MAX:-0.2}"
PYTHON_BIN="${WORLDMIRROR_PYTHON:-python3}"

PROFILE_ARGS=()
if [ "${PROFILE:-0}" = "1" ]; then
    PROFILE_DIR="${VLLM_TORCH_PROFILER_DIR:-${RESULT_DIR}/torch_trace}"
    mkdir -p "${PROFILE_DIR}"
    PROFILE_ARGS+=(--profile-dir "${PROFILE_DIR}")
fi

PRECISION_FLAG=()
if [ "${PRECISION:-bf16}" = "bf16" ]; then
    PRECISION_FLAG+=(--enable-bf16)
fi

FSDP_FLAG=()
if [ "${WM_USE_FSDP:-0}" = "1" ] || [ "${TP_DEG}" -gt 1 ]; then
    FSDP_FLAG+=(--use-fsdp)
fi

CMD=(
    "${PYTHON_BIN}" "${WORLDMIRROR_BENCH}"
    --worldmirror-dir "${WORLDMIRROR_DIR}"
    --model-path "${MODEL_PATH}"
    --result-dir "${RESULT_DIR}"
    --result-filename "${RESULT_FILENAME}"
    --target-size "${TARGET_SIZE}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --scenes "${SCENES}"
    --quality-ref "${XDIT_QUALITY_REF:-}"
    --quality-ref-write "${XDIT_QUALITY_REF_WRITE:-}"
    --quality-rel-max "${REL_MAX}"
    "${PRECISION_FLAG[@]}"
    "${FSDP_FLAG[@]}"
    "${PROFILE_ARGS[@]}"
)

if [ -n "${INPUT_PATH}" ]; then
    CMD+=(--input-path "${INPUT_PATH}")
fi

if [ "${TP_DEG}" -gt 1 ]; then
    TORCHRUN="${WORLDMIRROR_TORCHRUN:-}"
    [ -n "${TORCHRUN}" ] || TORCHRUN="$(command -v torchrun)"
    exec "${TORCHRUN}" --nproc_per_node="${TP_DEG}" --master_port="${WORLDMIRROR_MASTER_PORT:-29543}" "${CMD[@]}"
fi

exec "${CMD[@]}"
