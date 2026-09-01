#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compatibility entry point for the registered KernelForge E2E workload
# template. KernelForge now lives inside Hyperloom and its runnable examples are
# package data, while the existing template invokes this historical root path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${KERNELFORGE_PYTHON:-python3}"

# The retired standalone distribution carried these dependencies at top level;
# Hyperloom keeps them in explicit extras so ordinary installs stay lightweight.
"$PYTHON" -m pip install --quiet -e "${ROOT}[forge,forge-profiling]"

exec bash "${ROOT}/src/kernelforge/data/examples/triton-softmax-forge-loop/run_example.sh" "$@"
