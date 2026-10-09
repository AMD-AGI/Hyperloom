#!/usr/bin/env bash
# Drive this forge-loop task (AITER TP4 all-reduce) end to end.
#
# This is a *repository* task: the code under optimization lives in an existing
# AITER checkout spanning two files, not in this directory. It also needs four
# GPUs, so it differs from the single-file softmax examples in three ways:
#
#   * the scratch workspace is a git worktree of AITER (a /tmp copy would be a
#     multi-GB clone), created detached at a pinned baseline commit;
#   * JIT artifacts are redirected to a task-private AITER_JIT_DIR so candidate
#     builds never overwrite the shared checkout's modules;
#   * AITER_REBUILD is forced to 0 at the outer level — see the note below.
#
# Usage:
#   ./run_example.sh [WORKSPACE_DIR]
#
# Environment overrides (all optional):
#   AITER_SRC    existing AITER checkout to branch from (default: /sgl-workspace/aiter)
#   AITER_REF    baseline commit/ref                    (default: HEAD of AITER_SRC)
#   GPU_TARGET   gfx arch (default: autodetect via rocminfo)
#   NPROC        ranks / GPUs                           (default: 2)
#   SUITE        case set; 'default' derives its cases from NPROC,
#                tp4_wide / tp8_k3 carry a measured baseline (default: default)
#   MAX_HOURS    wall-clock budget in hours             (default: 2.0)
#   FORGE_MODEL  model name served by your gateway      (default: forge default)
#
# Prerequisites:
#   * Hyperloom installed so `kernelforge` is on PATH (pip install -e .)
#   * 4 GPUs on one node, fully connected (XGMI/P2P)
#   * An AITER checkout with its JIT modules already built
#   * Claude gateway configured: ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/tmp/forge_loop_aiter_ar_$(date +%s)}"
AITER_SRC="${AITER_SRC:-/sgl-workspace/aiter}"
NPROC="${NPROC:-2}"
SUITE="${SUITE:-default}"
# forge invokes the driver with no --shape, so the suite reaches the scored
# baseline and candidates only through the environment. Without this the
# campaign measures the derived default no matter what SUITE says.
export FORGE_COLLECTIVE_SUITE="$SUITE"
MAX_HOURS="${MAX_HOURS:-2.0}"
MODEL_ARGS=()
if [ -n "${FORGE_MODEL:-}" ]; then
  MODEL_ARGS=(--model "$FORGE_MODEL")
fi

detect_arch() {
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -oim1 'gfx[0-9a-f]\+' | tr '[:upper:]' '[:lower:]' || true
  fi
}
GPU_TARGET="${GPU_TARGET:-$(detect_arch)}"
GPU_TARGET="${GPU_TARGET:-gfx942}"

if ! command -v kernelforge >/dev/null 2>&1; then
  echo "error: 'kernelforge' not found on PATH. Install Hyperloom first:" >&2
  echo "         pip install -e /path/to/Hyperloom" >&2
  exit 1
fi
if [ ! -d "$AITER_SRC/.git" ]; then
  echo "error: AITER_SRC=$AITER_SRC is not a git checkout" >&2
  exit 1
fi

# A collective task is only meaningful with every rank present.
GPU_COUNT="$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
if [ "$GPU_COUNT" -lt "$NPROC" ]; then
  echo "error: need $NPROC GPUs, found $GPU_COUNT" >&2
  exit 1
fi

AITER_REF="${AITER_REF:-$(git -C "$AITER_SRC" rev-parse HEAD)}"

echo "==> AITER source   : $AITER_SRC @ ${AITER_REF:0:12}"
echo "==> Workspace      : $WORKSPACE"
echo "==> GPUs / arch    : $NPROC x mi300x/$GPU_TARGET"
echo "==> Case suite     : $SUITE"

# forge-loop stages every tracked modification with `git add -u` and reverts
# with `git revert HEAD`, so a dirty tree would be swept into the first
# candidate commit. A worktree at a pinned ref is guaranteed clean.
if [ ! -d "$WORKSPACE/.git" ] && [ ! -f "$WORKSPACE/.git" ]; then
  git -C "$AITER_SRC" worktree add -f --detach "$WORKSPACE" "$AITER_REF"
fi
cd "$WORKSPACE"
# forge-loop commits every kept candidate. A container without a git identity
# makes that commit fail, which the loop reports as a crashed candidate and
# reverts -- a real improvement then shows up as "Kept: 0".
#
# Supplied through the environment rather than `git config`: a worktree shares
# the main repository's config file, so writing there would overwrite the
# identity in the user's own AITER checkout. These names are only a fallback --
# an operator who already configured an identity keeps it.
if ! git config user.email >/dev/null 2>&1; then
  export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-forge-example@local}"
  export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-forge-example@local}"
fi
if ! git config user.name >/dev/null 2>&1; then
  export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-forge-example}"
  export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-forge-example}"
fi
BRANCH="forge-ar-$(basename "$WORKSPACE")"
git checkout -q -B "$BRANCH"
if [ -n "$(git status --porcelain)" ]; then
  echo "error: workspace is dirty; refusing to start" >&2
  git status --short >&2
  exit 1
fi

# The protected driver is untracked on purpose: `git add -u` never picks it up,
# and `git checkout -- .` (the loop's revert) never deletes it.
mkdir -p op_tests/multigpu_tests
cp "$EXAMPLE_DIR/driver.py" op_tests/multigpu_tests/forge_all_reduce_driver.py
DRIVER="$WORKSPACE/op_tests/multigpu_tests/forge_all_reduce_driver.py"

# Candidate JIT artifacts stay out of the shared checkout.
export AITER_JIT_DIR="${AITER_JIT_DIR:-$WORKSPACE/../$(basename "$WORKSPACE")-jit}"
mkdir -p "$AITER_JIT_DIR"
if [ -z "$(ls -A "$AITER_JIT_DIR" 2>/dev/null)" ]; then
  echo "==> Seeding JIT dir from $AITER_SRC (avoids a full from-scratch build)"
  cp -a "$AITER_SRC"/aiter/jit/*.so "$AITER_JIT_DIR"/ 2>/dev/null || true
fi
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

# CRITICAL: force AITER_REBUILD=0 for the whole loop.
#
# runner.py calls force_jit_rebuild(), which does
# os.environ.setdefault("AITER_REBUILD","1") — inherited by EVERY driver
# subprocess. AITER_REBUILD==1 makes aiter wipe the ninja cache and relink from
# scratch, and the in-process dedup list resets per subprocess, so all five
# validation stages plus bench plus baseline would each pay a full rebuild and
# blow past their timeouts. setdefault does not override an existing value, so
# exporting 0 here hands rebuild control to the driver, which recompiles once
# per source-hash change with AITER_REBUILD=2 (cache preserved).
export AITER_REBUILD=0

# QuickReduce would preempt custom all-reduce for large payloads.
unset AITER_QUICK_REDUCE_QUANTIZATION

echo "==> Driver self-check (correctness, then bench suite)"
python3 "$DRIVER" \
  --shape "target=raw,tp=$NPROC,rows=6,hidden=7168,dtype=bf16" \
  --mode smoke --snr-threshold 40
python3 "$DRIVER" \
  --shape "suite=$SUITE,tp=$NPROC,dtype=bf16" \
  --warmup 20 --iters 50 --bench-mode >/dev/null

echo "==> Launching forge-loop"
exec kernelforge forge-loop \
  "${MODEL_ARGS[@]}" \
  --workspace "$WORKSPACE" \
  --kernel "$WORKSPACE/csrc/include/custom_all_reduce.cuh" \
  --driver "$DRIVER" \
  --program-md-file "$EXAMPLE_DIR/program.md" \
  --task-type repository \
  --source-files "$WORKSPACE/csrc/include/custom_all_reduce.cuh,$WORKSPACE/aiter/dist/device_communicators/communicator_cuda.py" \
  --target-functions "CustomAllreduce::allreduce,dispatchFusedAllReduceRMSNorm" \
  --snr-threshold 40 \
  --gpu-target "$GPU_TARGET" \
  --gpu-type mi300x \
  --max-hours "$MAX_HOURS" \
  --git-branch "$BRANCH-opt" \
  --experiments-dir "$WORKSPACE/forge_experiments" \
  --nproc-per-node "$NPROC" \
  --bench-repeat 3
