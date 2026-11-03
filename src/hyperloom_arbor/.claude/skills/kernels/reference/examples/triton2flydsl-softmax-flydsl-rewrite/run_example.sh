#!/usr/bin/env bash
# Drive this triton2flydsl rewrite task (Triton softmax -> FlyDSL) end to end.
#
# `forge-rewrite-by-flydsl` git-inits its workspace and writes the FlyDSL kernel
# IN PLACE, so this script copies the task out of the packaged example tree into a
# scratch workspace first (keeping the repo tree clean) — the isolate-then-run
# pattern any caller should follow. The pipeline then runs:
#   ingest -> seed kernel.py -> measure source baseline (oracle)
#          -> PORT (correctness-only: translate softmax.py to FlyDSL kernel.py)
#          -> OPTIMIZE (forge-loop tunes the correct FlyDSL kernel)
#          -> report (source_ms vs flydsl_best_ms -> speedup)
#
# Usage:
#   ./run_example.sh [WORKSPACE_DIR]
#
# Environment overrides (all optional):
#   GPU_TARGET         gfx arch (default: autodetect via rocminfo, else gfx950)
#   MAX_PORT_ATTEMPTS  correctness-only port sessions before giving up (default: 3)
#   MAX_HOURS          OPTIMIZE: wall-clock budget in hours (min 1.0)   (default: 1.0)
#   FORGE_MODEL        model name served by your gateway                (default: forge default)
#
# Prerequisites:
#   * Hyperloom installed so `kernelforge` is on PATH (pip install -e .)
#   * A GPU with torch + Triton + FlyDSL available
#   * Claude auth configured: ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN, or
#     CLAUDE_CODE_OAUTH_TOKEN for subscription billing
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/tmp/triton2flydsl_softmax_flydsl_rewrite_$(date +%s)}"

detect_arch() {
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -oim1 'gfx[0-9a-f]\+' | tr '[:upper:]' '[:lower:]' || true
  fi
}
GPU_TARGET="${GPU_TARGET:-$(detect_arch)}"
GPU_TARGET="${GPU_TARGET:-gfx950}"
MAX_PORT_ATTEMPTS="${MAX_PORT_ATTEMPTS:-3}"
MAX_HOURS="${MAX_HOURS:-1.0}"

if ! command -v kernelforge >/dev/null 2>&1; then
  echo "error: 'kernelforge' not found on PATH. Install Hyperloom first:" >&2
  echo "         pip install -e /path/to/Hyperloom" >&2
  exit 1
fi

echo "==> Preparing scratch workspace: $WORKSPACE"
mkdir -p "$WORKSPACE"
cp "$EXAMPLE_DIR/softmax.py" "$WORKSPACE/"        # source kernel (ported FROM; protected)
cp "$EXAMPLE_DIR/driver.py" "$WORKSPACE/"         # measurement driver (protected)
cp "$EXAMPLE_DIR/graph_harness.py" "$WORKSPACE/"  # measurement harness (protected)
cp "$EXAMPLE_DIR/program.md" "$WORKSPACE/"        # agent guidance

# The pipeline git-inits the workspace itself and commits only the produced
# kernel, which it writes into its own .forge_rewrite/<attempt>/ directory; a
# .gitignore keeps build artifacts + experiment outputs untracked.
cd "$WORKSPACE"
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.log
build/
forge_experiments/
EOF

# Two genuinely environmental vars (not forge config): IS_SANDBOX is required by
# the claude CLI under root; PYTHONUNBUFFERED makes the stream flush promptly.
export IS_SANDBOX="${IS_SANDBOX:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_ARGS=()
if [ -n "${FORGE_MODEL:-}" ]; then
  MODEL_ARGS+=(--model "$FORGE_MODEL")
fi

echo "==> Launching forge-rewrite-by-flydsl (gpu=mi355x/$GPU_TARGET, port_attempts=$MAX_PORT_ATTEMPTS, max_hours=$MAX_HOURS)"
kernelforge forge-rewrite-by-flydsl \
  --source-kernel "$WORKSPACE/softmax.py" \
  --driver "$WORKSPACE/driver.py" \
  --logical-op-name softmax \
  --source-entry softmax \
  --target-functions "softmax,_softmax_kernel" \
  --workspace "$WORKSPACE" \
  --experiments-dir "$WORKSPACE/forge_experiments" \
  --result-json "$WORKSPACE/forge_experiments/forge_rewrite_result.json" \
  --flydsl-kernel-name kernel.py \
  --shapes-json '[{"M":256,"N":1024,"dtype":"f32"},{"M":4096,"N":1024,"dtype":"f32"}]' \
  --gpu-target "$GPU_TARGET" \
  --snr-threshold 30.0 \
  --max-port-attempts "$MAX_PORT_ATTEMPTS" \
  --max-hours "$MAX_HOURS" \
  "${MODEL_ARGS[@]}"

echo "==> Done. Ported FlyDSL kernel is under: $WORKSPACE/.forge_rewrite/<attempt>/kernel.py"
echo "    (the exact path is this run's temporary_paths in the result JSON)"
echo "    Iteration archive + profiles: $WORKSPACE/forge_experiments/"
echo "    Machine-readable result:      $WORKSPACE/forge_experiments/forge_rewrite_result.json"
