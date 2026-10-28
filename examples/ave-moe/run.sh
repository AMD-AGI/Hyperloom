#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
suite=localmoe
if (( $# )) && [[ "$1" == localmoe || "$1" == megamoe ]]; then
  suite="$1"
  shift
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash examples/ave-moe/run.sh [localmoe|megamoe] [workspace] [--max-steps N]"
  exit 0
fi
workspace="$HOME/ave-moe-optimization"
if (( $# )) && [[ "$1" != -* ]]; then
  workspace="$1"
  shift
fi
workspace="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$workspace")"
python_exe="$(python -c 'import sys; print(sys.executable)')"
export PATH="$(dirname -- "$python_exe"):$PATH"
gpus=1
if [[ "$suite" == megamoe ]]; then
  gpus=8
fi
default_devices="$(seq -s, 0 "$((gpus - 1))")"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$default_devices}"
cd -- "$example_dir/../.."

export INFERENCE_OPTIMIZER_RAY_EXEC=0
export HYPERLOOM_BENCHMARK_BACKEND=bypass
export KERNEL_OPT_BACKEND_ORDER=ave
export AVE_PROMPTS_FILE="$workspace/$suite/prompts.json"
export FRAMEWORK_REPO_PATH="$workspace/$suite/avelang"
export CLAUDE_MODEL=sonnet
export CLAUDE_CODE_EFFORT_LEVEL="${CLAUDE_CODE_EFFORT_LEVEL:-medium}"
export USER_DATA_PATH="$workspace/sessions"
export HYPERLOOM_RUNTIME_DIR="$USER_DATA_PATH/runtime"
export KERNEL_AGENT_ENV="$HYPERLOOM_RUNTIME_DIR/kernel-agent.env.sh"

"$python_exe" -m pip install \
  'cmake>=3.25' ninja numpy pytest \
  'TraceLens[kernel_source] @ git+https://github.com/AMD-AGI/TraceLens.git@647cbf2702687a656f2480e07517824369898aa9' \
  -e '.[runtime]'
"$python_exe" "$example_dir/prepare.py" "$workspace" --suite "$suite" "$@"
max_hours="$("$python_exe" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["steps"]) + 1)' "$AVE_PROMPTS_FILE")"
PYTHON="$python_exe" INFERENCE_OPTIMIZER_FORCE_PYTHON=1 FRAMEWORK=custom SKIP_RAY_START=1 \
  bash src/hyperloom/inference_optimizer/assets/install.sh

exec "$python_exe" -m hyperloom optimize \
  --framework custom \
  --framework-path "$FRAMEWORK_REPO_PATH" \
  --benchmark-scripts-dir "$example_dir/$suite" \
  --model "$example_dir/$suite" \
  --gpu-type mi355x --tp "$gpus" --max-hours "$max_hours" --tick-interval-sec 60 \
  --no-framework-agent --no-research-scout --no-static-recon \
  --no-target-advisory --no-enable-roofline --no-enable-conc-sweep \
  --enablement off
