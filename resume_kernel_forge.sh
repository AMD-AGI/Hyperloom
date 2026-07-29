#!/usr/bin/env bash
# RESUME (never a fresh session) the kernel-only Qwen3-14B-FP8 forge run on the
# -sglang worktrees, so the worktree code fixes (Fix C idle wind-down, 3600s
# gemm timeout, B/D) take effect on the SAME session 20260728T100852Z.
set -euo pipefail

REPO=/primus/data/xiaofei/Hyperloom-sglang
FORGE=/primus/data/xiaofei/KernelForge-sglang
SESSION_DIR=/primus/xiaofei/sessions/Qwen3-14B-FP8/20260728T100852Z

export PATH="/opt/venv/bin:${PATH}"
set -a
source "${REPO}/.env"          # USER_DATA_PATH=/primus/xiaofei/sessions (do NOT modify)
set +a
export KERNEL_AGENT_ENV="${REPO}/runtime-env/kernel-agent.env.sh"
export FORGE_PATH="${FORGE}"
export KERNEL_FORGE_ROOT="${FORGE}"
export PYTHONPATH="${REPO}/src:${REPO}:${FORGE}/src:/primus/xiaofei/Magpie:/sgl-workspace/mori:/sgl-workspace/aiter"
export GPU_TARGET="${GPU_TARGET:-gfx950}"
export KERNEL_OPT_BACKEND_ORDER=forge
export FORGE_TUNE_TASK_TIMEOUT="${FORGE_TUNE_TASK_TIMEOUT:-3600}"
export AITER_ROOT_DIR="${AITER_ROOT_DIR:-/sgl-workspace/aiter}"

TS="$(date -u +%Y%m%d-%H%M)"
LOG=/primus/xiaofei/logs/qwen3-14b-fp8-kernel-4h-forge-resume-${TS}.log

cd "${REPO}"
echo "resuming session ${SESSION_DIR} -> log ${LOG}"
exec python3 -m hyperloom.inference_optimizer.cli optimize \
  --resume \
  --resume-from "${SESSION_DIR}" \
  --framework sglang \
  --gpu-type mi355x \
  --max-hours 4 \
  --no-explore \
  --no-framework-agent \
  --no-enable-conc-sweep \
  >> "${LOG}" 2>&1
