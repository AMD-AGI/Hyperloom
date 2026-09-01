#!/usr/bin/env bash
# Drive the MoRI-EP dispatch/combine forge-loop task end to end.
#
# A distributed (8-GPU, single-node) multi-rank task: forge-loop tunes ONLY
# the launch config in mori_ep_config.py (block_num / warp_per_block /
# kernel_type / combine_zero_copy) -- never mori's C++/HIP kernel source. See
# local_knowledge/framework/mori/operators/ep_dispatch_combine/ for the full
# knowledge base (kernel-type decision tree, buffer modes, measured MI300X
# results across forge-loop rounds).
#
# Usage:
#   ./run_example.sh [WORKSPACE_DIR]
#
# WORKSPACE_DIR safety: if omitted, a fresh timestamped /tmp dir is used
# (always safe). If you pass an existing, non-empty directory (or one that's
# already a git repo), this script refuses to run there unless you set
# FORGE_ALLOW_EXISTING_WORKSPACE=1 -- it copies files into WORKSPACE_DIR,
# stages+commits them with git, and forge-loop checks out a branch there;
# pointing that at an arbitrary real repo risks overwriting files or
# polluting its history/branch state.
#
# Environment overrides (all optional):
#   GPU_TARGET   gfx arch (default: autodetect via rocminfo, else gfx942).
#                Also propagated to MORI_GPU_ARCHS (below) so mori loads the
#                matching precompiled binary instead of always gfx942.
#   MAX_HOURS    wall-clock budget in hours            (default: 1.0, CLI min)
#   FORGE_MODEL  model name served by your gateway     (default: forge default)
#   MORI_TOKENS_PER_RANK  tokens/rank for the bench shape (default: 4096)
#   MORI_HIDDEN_DIM       hidden dim for the bench shape  (default: 7168)
#   MORI_TOPK             experts/token for the bench shape (default: 8)
#   MORI_REPO_ROOT        path to a `mori` GIT CHECKOUT (not just the pip
#                          package -- the correctness gate reuses mori's own
#                          test-suite reference math)   (default: /work/mori)
#   KERNELFORGE_INCLUDE_MORI_KB  inject local_knowledge/framework/mori/ into
#                          the kernel backend's system prompt (default: 1 -- on for
#                          this example; set 0 to run the KB-ablation arm)
#   FORGE_ALLOW_EXISTING_WORKSPACE  set 1 to allow reusing a non-empty/
#                          already-git WORKSPACE_DIR (default: 0, refuse)
#
# Prerequisites:
#   * Hyperloom installed so `kernelforge` is on PATH (pip install -e .)
#   * 8 GPUs, mori installed (`python -c "import mori"`)
#   * A `mori` git checkout on disk for the correctness gate -- see
#     MORI_REPO_ROOT above (driver.py's module docstring has details)
#   * HSA_NO_SCRATCH_RECLAIM=1 (driver.py also sets this itself as a
#     belt-and-suspenders default, but it must take effect before HIP init)
#   * Claude gateway configured: ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/tmp/forge_loop_mori_ep_dispatch_combine_$(date +%s)}"

detect_arch() {
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -oim1 'gfx[0-9a-f]\+' | tr '[:upper:]' '[:lower:]' || true
  fi
}
GPU_TARGET="${GPU_TARGET:-$(detect_arch)}"
GPU_TARGET="${GPU_TARGET:-gfx942}"
MAX_HOURS="${MAX_HOURS:-1.0}"

if ! command -v kernelforge >/dev/null 2>&1; then
  echo "error: 'kernelforge' not found on PATH. Install Hyperloom first:" >&2
  echo "         pip install -e /path/to/Hyperloom" >&2
  exit 1
fi

if ! python3 -c "import mori" >/dev/null 2>&1; then
  echo "error: 'mori' is not importable in this Python environment." >&2
  exit 1
fi

# Refuse to touch an existing, non-empty (or already-git) directory unless
# explicitly told to. Never run this against a real project checkout by
# accident -- see the WORKSPACE_DIR note in the header comment.
if [ -e "$WORKSPACE" ]; then
  existing_contents="$(find "$WORKSPACE" -mindepth 1 -maxdepth 1 2>/dev/null || true)"
  if [ -n "$existing_contents" ] || [ -d "$WORKSPACE/.git" ]; then
    if [ "${FORGE_ALLOW_EXISTING_WORKSPACE:-0}" != "1" ]; then
      echo "error: '$WORKSPACE' already exists and is non-empty (or already a git repo)." >&2
      echo "       Refusing to run there by default: this script copies the 3 example" >&2
      echo "       files into WORKSPACE_DIR, commits them, and forge-loop checks out a" >&2
      echo "       'forge-optimize' branch there -- pointing that at an arbitrary existing" >&2
      echo "       directory risks overwriting same-named files or mutating a real repo." >&2
      echo "       Pass a new/empty directory (the default, timestamped /tmp path, always" >&2
      echo "       is one), or set FORGE_ALLOW_EXISTING_WORKSPACE=1 to proceed anyway." >&2
      exit 1
    fi
    echo "==> WARNING: reusing existing, non-empty workspace ($WORKSPACE) -- FORGE_ALLOW_EXISTING_WORKSPACE=1 set" >&2
  fi
fi

echo "==> Preparing scratch workspace: $WORKSPACE"
mkdir -p "$WORKSPACE"
cp "$EXAMPLE_DIR/mori_ep_config.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/driver.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/program.md" "$WORKSPACE/"

cd "$WORKSPACE"
if [ ! -d .git ]; then
  git init -q
  git config user.email "forge-example@local"
  git config user.name "forge-example"
fi
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.log
build/
forge_experiments/
EOF
# Stage ONLY the example's own files -- never `git add -A`: in a reused
# workspace (FORGE_ALLOW_EXISTING_WORKSPACE=1) that would sweep unrelated
# pre-existing changes into this "forge example" commit.
git add .gitignore mori_ep_config.py driver.py program.md
git commit -q -m "forge example: initial workspace (mori-ep dispatch/combine)" || true

export IS_SANDBOX="${IS_SANDBOX:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export MORI_TOKENS_PER_RANK="${MORI_TOKENS_PER_RANK:-4096}"
export MORI_HIDDEN_DIM="${MORI_HIDDEN_DIM:-7168}"
export MORI_TOPK="${MORI_TOPK:-8}"
# Propagate the SAME detected/overridden arch driver.py's MORI_GPU_ARCHS
# setdefault would otherwise silently default to gfx942 for (a gfx950 box
# would then load the wrong precompiled binary).
export MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-$GPU_TARGET}"
# On by default so this example's agent can actually read the mori KB card
# program.md points it at; set 0 to run the KB-ablation (no-KB) arm.
export KERNELFORGE_INCLUDE_MORI_KB="${KERNELFORGE_INCLUDE_MORI_KB:-1}"
echo "==> Shape: tokens_per_rank=$MORI_TOKENS_PER_RANK hidden_dim=$MORI_HIDDEN_DIM topk=$MORI_TOPK"
echo "==> MORI_GPU_ARCHS=$MORI_GPU_ARCHS KERNELFORGE_INCLUDE_MORI_KB=$KERNELFORGE_INCLUDE_MORI_KB"

MODEL_ARGS=()
if [ -n "${FORGE_MODEL:-}" ]; then
  MODEL_ARGS+=(--model "$FORGE_MODEL")
fi

echo "==> Launching forge-loop (gpu=$GPU_TARGET, max_hours=$MAX_HOURS)"
kernelforge forge-loop \
  --kernel "$WORKSPACE/mori_ep_config.py" \
  --driver "$WORKSPACE/driver.py" \
  --workspace "$WORKSPACE" \
  --experiments-dir "$WORKSPACE/forge_experiments" \
  --result-json "$WORKSPACE/forge_experiments/forge_result.json" \
  --program-md-file "$WORKSPACE/program.md" \
  --kernel-backend aiter \
  --gpu-target "$GPU_TARGET" \
  --snr-threshold 30.0 \
  --max-hours "$MAX_HOURS" \
  --git-branch forge-optimize \
  --target-functions "get_ep_launch_config,dispatch,combine" \
  --no-profiling \
  --no-prepare-task \
  "${MODEL_ARGS[@]}"
# --no-prepare-task: NOT because dispatch/combine "cannot" be graph-captured
# -- mori's own reference benchmark (tests/python/ops/bench_dispatch_combine.py)
# captures each rank's own dispatch()/combine() calls into per-rank
# torch.cuda.CUDAGraph()s and replays them, and the aiter KB explicitly
# documents MoRI-EP as "HIP-graph-capturable". What's actually true is
# narrower: this is an 8-PROCESS distributed job (mp.spawn, one graph per
# process), which doesn't fit --prepare-task's single-process
# graph_harness.py preflight assumption. driver.py's stdout contract
# (correctness/bench/profile modes) was verified by hand against
# examples/README.md §2 before this task was ever launched, so the
# preflight is redundant here regardless.
# --bench-mode's default (non-`--graph-mode`) path is still eager dispatch/
# combine calls, not graph replay -- see driver.py's module docstring for
# why, and its --graph-mode flag for the closer-to-production alternative.

echo "==> Done. Best config is checked out in: $WORKSPACE/mori_ep_config.py"
echo "    Iteration archive + profiles: $WORKSPACE/forge_experiments/"
echo "    Machine-readable result:      $WORKSPACE/forge_experiments/forge_result.json"
