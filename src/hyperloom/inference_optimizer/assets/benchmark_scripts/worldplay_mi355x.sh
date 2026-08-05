#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORLDPLAY_REPO_URL="${WORLDPLAY_REPO_URL:-https://github.com/Tencent-Hunyuan/HY-WorldPlay.git}"
# Prefixed names first, then the framework-agnostic form: a session is
# single-framework, so an operator should not need to know the name to point at
# the checkout.
WORLDPLAY_DIR="${WORLDPLAY_DIR:-${WORLDPLAY_REPO_PATH:-${FRAMEWORK_REPO_PATH:-}}}"
if [ -z "${WORLDPLAY_DIR}" ]; then
    _CACHE_ROOT="${HYPERLOOM_CACHE_DIR:-${HOME}/.cache/hyperloom}"
    WORLDPLAY_DIR="${_CACHE_ROOT}/HY-WorldPlay"
fi

if [ ! -d "${WORLDPLAY_DIR}" ]; then
    mkdir -p "$(dirname "${WORLDPLAY_DIR}")"
    git clone --depth 1 "${WORLDPLAY_REPO_URL}" "${WORLDPLAY_DIR}"
fi

export WORLDPLAY_DIR
export WORLDPLAY_REPO_PATH="${WORLDPLAY_DIR}"
export WORLDPLAY_BENCH="${WORLDPLAY_BENCH:-${_SCRIPT_DIR}/bench_fps.py}"
_SCRIPT_NAME="$(basename "$0")"
_SCRIPT_RUNNER="${_SCRIPT_NAME#worldplay_}"
_SCRIPT_RUNNER="${_SCRIPT_RUNNER%.sh}"
export RUNNER_TYPE="${RUNNER_TYPE:-${_SCRIPT_RUNNER:-mi355x}}"

if [ ! -f "${WORLDPLAY_BENCH}" ]; then
    for _candidate in \
        "${WORLDPLAY_DIR}/hyperloom_bench/bench_fps.py" \
        "${WORLDPLAY_DIR}/bench/bench_fps.py" \
        "${WORLDPLAY_DIR}/benchmark/bench_fps.py"; do
        if [ -f "${_candidate}" ]; then
            export WORLDPLAY_BENCH="${_candidate}"
            break
        fi
    done
fi

if [ ! -f "${WORLDPLAY_BENCH}" ]; then
    echo "[worldplay] benchmark driver not found. Set WORLDPLAY_BENCH." >&2
    exit 2
fi

exec bash "${_SCRIPT_DIR}/worldplay_bench_common.sh"
