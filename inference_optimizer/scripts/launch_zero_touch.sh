#!/usr/bin/env bash
# Zero-touch Hyperloom launch: install → env → preflight → coordinator.
# Pinned Hyperloom commit: 99722442b0449fc873779a3f1dee4954e63b8dff (main #499)
export HYPERLOOM_PIN="${HYPERLOOM_PIN:-99722442b0449fc873779a3f1dee4954e63b8dff}"
#
# Usage:
#   export USER_DATA_PATH=/home/<USER>/PROJECTS/runs/gptoss120b_<TS>
#   export REPO_ROOT=/home/<USER>/PROJECTS/v05/Hyperloom
#   source "$REPO_ROOT/.env"   # SAFE_API_KEY, TRACELENS_ROOT, GEAK_REF, ...
#   bash "$REPO_ROOT/inference_optimizer/scripts/launch_zero_touch.sh" [optimize args...]
#
# All extra args are forwarded to ``python3 -m inference_optimizer.cli optimize``.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_script_dir}/../.." && pwd)}"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
export USER_DATA_PATH
export HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
export KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"

log() { echo "[zero-touch] $*"; }
die() { echo "[zero-touch ERROR] $*" >&2; exit 1; }

# Zero-touch defaults (must precede install.sh — install gates on PATCH_MAGPIE).
export HYPERLOOM_KERNELOPT_RAW_TRACE_FALLBACK="${HYPERLOOM_KERNELOPT_RAW_TRACE_FALLBACK:-1}"
export HYPERLOOM_AUTO_INTEGRATE="${HYPERLOOM_AUTO_INTEGRATE:-1}"
export ALLOW_GEAK_MULTIGPU="${ALLOW_GEAK_MULTIGPU:-1}"
export HYPERLOOM_GEAK_COST_LIMIT="${HYPERLOOM_GEAK_COST_LIMIT:-0.0}"
export HYPERLOOM_FUSED_MOE_GEAK_BUDGET_MIN="${HYPERLOOM_FUSED_MOE_GEAK_BUDGET_MIN:-240}"
export GEAK_SKIP_PREPROCESS_PROFILE="${GEAK_SKIP_PREPROCESS_PROFILE:-1}"
export GEAK_BENCHMARK_WARMUP="${GEAK_BENCHMARK_WARMUP:-10}"
export GEAK_BENCHMARK_ITERATIONS="${GEAK_BENCHMARK_ITERATIONS:-40}"
export GEAK_BENCH_TIMEOUT="${GEAK_BENCH_TIMEOUT:-300}"
export GEAK_FULL_TOTAL_S="${GEAK_FULL_TOTAL_S:-10800}"
export AITER_REBUILD="${AITER_REBUILD:-1}"
export GEAK_REF="${GEAK_REF:-gwiab-scheduler}"
export GEAK_REPO="${GEAK_REPO:-/home/sapmajum/PROJECTS/v05/GEAK}"
export GEAK_RUN_MODE="${GEAK_RUN_MODE:-full}"
export KERNEL_AGENT_NUM_GPUS="${KERNEL_AGENT_NUM_GPUS:-4}"
export KERNEL_OPT_MAX_PARALLEL="${KERNEL_OPT_MAX_PARALLEL:-1}"
export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-geak}"
export INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT="${INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT:-1}"
export GEAK_MIN_PARALLEL_WORKERS="${GEAK_MIN_PARALLEL_WORKERS:-4}"
export GEAK_WORKERS_PER_GPU="${GEAK_WORKERS_PER_GPU:-3}"
export GEAK_FULL_MAX_ROUNDS="${GEAK_FULL_MAX_ROUNDS:-5}"
export HYPERLOOM_KERNEL_MAX_TURNS="${HYPERLOOM_KERNEL_MAX_TURNS:-40}"
export KERNEL_AGENT_BUILD_GEAK_RAG_INDEX="${KERNEL_AGENT_BUILD_GEAK_RAG_INDEX:-1}"
export HYPERLOOM_KERNEL_AGENT_ROOT="${REPO_ROOT}/kernel-agent"
export KERNEL_AGENT_ROOT="${REPO_ROOT}/kernel-agent"
unset GEAK_CONFIG
export HYPERLOOM_DISABLE_MOE_TILE_HACK="${HYPERLOOM_DISABLE_MOE_TILE_HACK:-0}"
export HYPERLOOM_ZERO_TOUCH_MINIMAL_EXPLORE="${HYPERLOOM_ZERO_TOUCH_MINIMAL_EXPLORE:-1}"
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-2400}"
export PATCH_MAGPIE="${PATCH_MAGPIE:-0}"
export HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT="${HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT:-0}"
export INFERENCE_OPTIMIZER_STEADY_STATE_MODE="${INFERENCE_OPTIMIZER_STEADY_STATE_MODE:-mixed}"

# Auth-proxy routing MUST precede install.sh — install writes geak-config/local.yaml
# from OPENAI_BASE_URL / GEAK_BASE_URL. Zero-touch always routes GEAK litellm
# through the local auth-proxy on :4002 (see repro guide §2).
export INFERENCE_OPTIMIZER_CATALOG_PROBE_URL="${INFERENCE_OPTIMIZER_CATALOG_PROBE_URL:-http://127.0.0.1:4002/v1}"
export ANTHROPIC_BASE_URL="http://127.0.0.1:4002"
export OPENAI_BASE_URL="http://127.0.0.1:4002/v1"
export GEAK_BASE_URL="http://127.0.0.1:4002/v1"

mkdir -p "$USER_DATA_PATH" "${USER_DATA_PATH}/optimizer_runs"

log "USER_DATA_PATH=$USER_DATA_PATH"
log "running install.sh"
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"

if [ ! -f "$KERNEL_AGENT_ENV" ]; then
  die "kernel-agent env missing after install: $KERNEL_AGENT_ENV"
fi
# shellcheck disable=SC1090
source "$KERNEL_AGENT_ENV"

log "running local_setup.sh"
bash "$REPO_ROOT/inference_optimizer/scripts/local_setup.sh"
if [ -f "${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh" ]; then
  # shellcheck disable=SC1091
  source "${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh"
fi

# Auth-proxy (local :4002) is required for catalog probe + GEAK litellm routing.
export INFERENCE_OPTIMIZER_CATALOG_PROBE_URL="${INFERENCE_OPTIMIZER_CATALOG_PROBE_URL:-http://127.0.0.1:4002/v1}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:4002}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:4002/v1}"

ulimit -n 65536 || log "WARN: could not raise ulimit -n (continuing)"

_kill_orphan_raylets() {
  for pattern in raylet gcs_server 'ray::Raylet'; do
    pkill -TERM -f "$pattern" >/dev/null 2>&1 || true
  done
  sleep 1
  for pattern in raylet gcs_server 'ray::Raylet'; do
    pkill -KILL -f "$pattern" >/dev/null 2>&1 || true
  done
}

if command -v ray >/dev/null 2>&1; then
  ray stop --force >/dev/null 2>&1 || true
  sleep 2
  _kill_orphan_raylets
  rm -rf /tmp/ray/session_* /tmp/ray 2>/dev/null || true
  if ! ray status >/dev/null 2>&1; then
    num_gpus="$(python3 - <<'PY' 2>/dev/null || echo 0
try:
    import torch
    print(torch.cuda.device_count() or 0)
except Exception:
    print(0)
PY
)"
    log "starting ray head (--num-gpus=${num_gpus})"
    ray start --head --disable-usage-stats \
      --num-gpus="${num_gpus}" --include-dashboard=false >/dev/null \
      || die "ray start failed"
  fi
  export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
fi

log "preflight: python imports"
python3 - <<'PY' || die "preflight import check failed (Magpie/litellm/fastmcp)"
import importlib
for mod in ("Magpie", "litellm", "fastmcp"):
    importlib.import_module(mod)
print("preflight imports ok")
PY

if curl -sf "http://127.0.0.1:4002/v1/models" >/dev/null 2>&1; then
  log "preflight: auth-proxy OK (4002/v1/models)"
else
  log "WARN: auth-proxy not reachable at http://127.0.0.1:4002/v1/models"
fi

# Zero-touch defaults (override via env before launch if needed).
# (Moved above install.sh — duplicate exports here are idempotent.)
export HYPERLOOM_KERNELOPT_RAW_TRACE_FALLBACK="${HYPERLOOM_KERNELOPT_RAW_TRACE_FALLBACK:-1}"
export HYPERLOOM_AUTO_INTEGRATE="${HYPERLOOM_AUTO_INTEGRATE:-1}"
export ALLOW_GEAK_MULTIGPU="${ALLOW_GEAK_MULTIGPU:-1}"
export HYPERLOOM_GEAK_COST_LIMIT="${HYPERLOOM_GEAK_COST_LIMIT:-0.0}"
export HYPERLOOM_FUSED_MOE_GEAK_BUDGET_MIN="${HYPERLOOM_FUSED_MOE_GEAK_BUDGET_MIN:-240}"
export GEAK_SKIP_PREPROCESS_PROFILE="${GEAK_SKIP_PREPROCESS_PROFILE:-1}"
export GEAK_BENCHMARK_WARMUP="${GEAK_BENCHMARK_WARMUP:-10}"
export GEAK_BENCHMARK_ITERATIONS="${GEAK_BENCHMARK_ITERATIONS:-40}"
export GEAK_BENCH_TIMEOUT="${GEAK_BENCH_TIMEOUT:-300}"
export GEAK_FULL_TOTAL_S="${GEAK_FULL_TOTAL_S:-10800}"
export AITER_REBUILD="${AITER_REBUILD:-1}"
export GEAK_REF="${GEAK_REF:-gwiab-scheduler}"
export GEAK_REPO="${GEAK_REPO:-/home/sapmajum/PROJECTS/v05/GEAK}"
export GEAK_RUN_MODE="${GEAK_RUN_MODE:-full}"
export KERNEL_AGENT_NUM_GPUS="${KERNEL_AGENT_NUM_GPUS:-4}"
export KERNEL_OPT_MAX_PARALLEL="${KERNEL_OPT_MAX_PARALLEL:-1}"
export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-geak}"
export INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT="${INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT:-1}"
export GEAK_MIN_PARALLEL_WORKERS="${GEAK_MIN_PARALLEL_WORKERS:-4}"
export GEAK_WORKERS_PER_GPU="${GEAK_WORKERS_PER_GPU:-3}"
export GEAK_FULL_MAX_ROUNDS="${GEAK_FULL_MAX_ROUNDS:-5}"
export HYPERLOOM_KERNEL_MAX_TURNS="${HYPERLOOM_KERNEL_MAX_TURNS:-40}"
export KERNEL_AGENT_BUILD_GEAK_RAG_INDEX="${KERNEL_AGENT_BUILD_GEAK_RAG_INDEX:-1}"
export HYPERLOOM_KERNEL_AGENT_ROOT="${REPO_ROOT}/kernel-agent"
export KERNEL_AGENT_ROOT="${REPO_ROOT}/kernel-agent"
unset GEAK_CONFIG
export HYPERLOOM_DISABLE_MOE_TILE_HACK="${HYPERLOOM_DISABLE_MOE_TILE_HACK:-0}"
export HYPERLOOM_ZERO_TOUCH_MINIMAL_EXPLORE="${HYPERLOOM_ZERO_TOUCH_MINIMAL_EXPLORE:-1}"
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-2400}"
export PATCH_MAGPIE="${PATCH_MAGPIE:-0}"
export HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT="${HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT:-0}"
export INFERENCE_OPTIMIZER_STEADY_STATE_MODE="${INFERENCE_OPTIMIZER_STEADY_STATE_MODE:-mixed}"

# Per-run JIT cache isolation (#485 review point b): the integrate cache
# move-aside (apply_kernel_patch) operates on $TRITON_CACHE_DIR /
# $TORCHINDUCTOR_CACHE_DIR. Pinning both under USER_DATA_PATH scopes the
# move-aside to THIS run so it cannot disturb a co-tenant's compile cache on a
# shared node. Set here (after local-setup.env.sh) so it is the authoritative
# value the coordinator and its sglang subprocesses inherit.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${USER_DATA_PATH}/runtime/jit-cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${USER_DATA_PATH}/runtime/jit-cache/torchinductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_LOG="${USER_DATA_PATH}/optimizer_runs/zero_touch_${TS}.log"
log "mirror log → $RUN_LOG"

exec > >(tee -a "$RUN_LOG") 2>&1
log "launching coordinator (args: $*)"
cd "$REPO_ROOT"
exec python3 -m inference_optimizer.cli optimize "$@"
