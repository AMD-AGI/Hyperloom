#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# install.sh <tool-cache-dir>: stage the tools an inference_suite round runs into <tool-cache-dir> and print the
# manifest's sha256 (python -m inference_suite stage). Re-running on a staged cache verifies it and stages nothing.
# PYTHON (python3) must be the interpreter rounds run under: lm-eval is installed into it from the runner hash lock
# the inference_suite beside this script ships. That package is the one that stages, so the pins it ships are the
# ones staged.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: install.sh <tool-cache-dir>" >&2
    exit 2
fi
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="${package_root}${PYTHONPATH:+:${PYTHONPATH}}" exec "${PYTHON:-python3}" -m inference_suite stage --tool-cache "$1"
