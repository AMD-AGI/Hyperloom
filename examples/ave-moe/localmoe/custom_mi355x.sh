#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ave_source="$(cd -- "${FRAMEWORK_REPO_PATH:?Set FRAMEWORK_REPO_PATH to the AveLang checkout}" && pwd)"
export PYTHONPATH="${ave_source}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd -- "$ave_source"
exec python "$example_dir/driver.py" --report "${RESULT_DIR:?Set RESULT_DIR}/${RESULT_FILENAME:-inferencex_result}.json"
