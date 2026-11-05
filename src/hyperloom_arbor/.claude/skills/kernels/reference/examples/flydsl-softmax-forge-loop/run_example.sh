#!/usr/bin/env bash
# Drive this forge-loop task (FlyDSL softmax) end to end.
#
# forge-loop git-inits its workspace and edits the kernel IN PLACE, so this
# script copies the task out of the packaged example tree into a scratch workspace
# first (keeping the repo tree clean) — the isolate-then-run pattern any
# forge-loop caller should follow.
#
# Usage:
#   ./run_example.sh [WORKSPACE_DIR]
#
# Environment overrides (all optional):
#   GPU_TARGET   gfx arch (default: autodetect via rocminfo, else gfx950)
#   MAX_HOURS    wall-clock budget in hours            (default: 1.0)
#   FORGE_MODEL  model name served by your gateway     (default: forge default)
#
# Prerequisites:
#   * Hyperloom installed so `kernelforge` is on PATH (pip install -e .)
#   * A GPU with torch + FlyDSL available (`python -c "import flydsl"` works)
#   * Claude auth configured: ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN, or
#     CLAUDE_CODE_OAUTH_TOKEN for subscription billing
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/tmp/forge_loop_flydsl_softmax_$(date +%s)}"

detect_arch() {
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -oim1 'gfx[0-9a-f]\+' | tr '[:upper:]' '[:lower:]' || true
  fi
}
GPU_TARGET="${GPU_TARGET:-$(detect_arch)}"
GPU_TARGET="${GPU_TARGET:-gfx950}"
MAX_HOURS="${MAX_HOURS:-1.0}"

if ! command -v kernelforge >/dev/null 2>&1; then
  echo "error: 'kernelforge' not found on PATH. Install Hyperloom first:" >&2
  echo "         pip install -e /path/to/Hyperloom" >&2
  exit 1
fi

echo "==> Preparing scratch workspace: $WORKSPACE"
mkdir -p "$WORKSPACE"
cp "$EXAMPLE_DIR/softmax_kernel.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/driver.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/graph_harness.py" "$WORKSPACE/"   # measurement harness (protected)
cp "$EXAMPLE_DIR/program.md" "$WORKSPACE/"

# forge-loop's keep/revert relies on git; give it a repo with an initial commit.
# Build artifacts and the loop's own outputs stay untracked so a revert never
# fails on a dirtied tree.
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
git add -A
git commit -q -m "forge example: initial workspace" || true

# Two genuinely environmental vars (not forge config): IS_SANDBOX is required by
# the claude CLI under root; PYTHONUNBUFFERED makes the stream flush promptly.
export IS_SANDBOX="${IS_SANDBOX:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_ARGS=()
if [ -n "${FORGE_MODEL:-}" ]; then
  MODEL_ARGS+=(--model "$FORGE_MODEL")
fi

echo "==> Launching forge-loop (gpu=mi355x/$GPU_TARGET, max_hours=$MAX_HOURS)"
kernelforge forge-loop \
  --kernel "$WORKSPACE/softmax_kernel.py" \
  --driver "$WORKSPACE/driver.py" \
  --workspace "$WORKSPACE" \
  --experiments-dir "$WORKSPACE/forge_experiments" \
  --result-json "$WORKSPACE/forge_experiments/forge_result.json" \
  --program-md-file "$WORKSPACE/program.md" \
  --kernel-backend flydsl \
  --task-type flydsl2flydsl \
  --gpu-target "$GPU_TARGET" \
  --snr-threshold 30.0 \
  --max-hours "$MAX_HOURS" \
  --git-branch forge-optimize \
  --target-functions "build_softmax_module,softmax_kernel" \
  "${MODEL_ARGS[@]}"

echo "==> Done. Best kernel is checked out in: $WORKSPACE/softmax_kernel.py"
echo "    Iteration archive + profiles: $WORKSPACE/forge_experiments/"
echo "    Machine-readable result:      $WORKSPACE/forge_experiments/forge_result.json"
