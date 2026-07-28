#!/usr/bin/env bash
# Kernel-only Qwen3-14B-FP8 run on the -sglang worktrees, forge backend.
set -euo pipefail

REPO=/primus/data/xiaofei/Hyperloom-sglang
FORGE=/primus/data/xiaofei/KernelForge-sglang

export PATH="/opt/venv/bin:${PATH}"
set -a
source "${REPO}/.env"
set +a
export KERNEL_AGENT_ENV="${REPO}/runtime-env/kernel-agent.env.sh"
export FORGE_PATH="${FORGE}"
export KERNEL_FORGE_ROOT="${FORGE}"
export PYTHONPATH="${REPO}/src:${REPO}:${FORGE}/src:/primus/xiaofei/Magpie:/sgl-workspace/mori:/sgl-workspace/aiter"
export GPU_TARGET="${GPU_TARGET:-gfx950}"
export KERNEL_OPT_BACKEND_ORDER=forge
# forge per-task-GROUP aiter tuner timeout (≈per shape). Old default 120s killed
# the first group of every shape (first-run JIT ~44s/module + serial baton-lock
# builds on gfx950/rocm-7.2.0) -> all shapes wrongly flagged "GPU hang". Measured:
# largest-K group extrapolates to ~1750s, so 1800s was too tight; 3600s is the
# forge default now too, kept explicit here for clarity.
export FORGE_TUNE_TASK_TIMEOUT="${FORGE_TUNE_TASK_TIMEOUT:-3600}"
# Pin the tuner's aiter to the serving aiter so tuned configs match runtime.
export AITER_ROOT_DIR="${AITER_ROOT_DIR:-/sgl-workspace/aiter}"

TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M)}"
LOG=/primus/xiaofei/logs/qwen3-14b-fp8-kernel-4h-forge-${TS}.log
INFO=/primus/xiaofei/logs/qwen3-14b-fp8-kernel-4h-forge-${TS}.launch.json

cd "${REPO}"
exec python3 -m hyperloom.inference_optimizer.cli optimize \
  --model /primus/models/Qwen3-14B-FP8 \
  --framework sglang \
  --gpu-type mi355x \
  --tp 1 --conc 64 --isl 1024 --osl 1024 \
  --precision fp8 \
  --max-hours 4 \
  --no-explore \
  --no-framework-agent \
  --no-enable-conc-sweep \
  --launch-info-file "${INFO}" \
  >> "${LOG}" 2>&1
