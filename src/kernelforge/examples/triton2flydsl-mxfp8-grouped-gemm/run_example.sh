#!/usr/bin/env bash
# Rewrite the SGLang Triton MXFP8 grouped GEMM to FlyDSL, then optimize it.
#
# Usage:
#   source /path/to/set_env.sh
#   ./run_example.sh [WORKSPACE_DIR]
#
# Optional overrides:
#   GPU_TARGET         gfx architecture (default: autodetect, then gfx950)
#   MAX_PORT_ATTEMPTS  correctness-only port sessions (default: 3)
#   MAX_HOURS          total rewrite budget, minimum 1 hour (default: 1.0)
#   FORGE_MODEL        model served by the configured gateway
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/tmp/triton2flydsl_mxfp8_grouped_gemm_$(date +%s)}"

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
  echo "       python3 -m pip install -e /path/to/Hyperloom" >&2
  exit 1
fi

echo "==> Preparing scratch workspace: $WORKSPACE"
mkdir -p "$WORKSPACE"
cp "$EXAMPLE_DIR/mxfp8_grouped_gemm.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/driver.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/graph_harness.py" "$WORKSPACE/"
cp "$EXAMPLE_DIR/session_cases.json" "$WORKSPACE/"
cp "$EXAMPLE_DIR/program.md" "$WORKSPACE/"
cp "$EXAMPLE_DIR/config.yaml" "$WORKSPACE/"

cat >"$WORKSPACE/.gitignore" <<'EOF'
__pycache__/
*.pyc
*.log
build/
forge_experiments/
EOF

export IS_SANDBOX="${IS_SANDBOX:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_ARGS=()
if [ -n "${FORGE_MODEL:-}" ]; then
  MODEL_ARGS+=(--model "$FORGE_MODEL")
fi

echo "==> Launching FlyDSL rewrite (gpu=mi355x/$GPU_TARGET, port_attempts=$MAX_PORT_ATTEMPTS, max_hours=$MAX_HOURS)"
kernelforge forge-rewrite-by-flydsl \
  --source-kernel "$WORKSPACE/mxfp8_grouped_gemm.py" \
  --driver "$WORKSPACE/driver.py" \
  --logical-op-name mxfp8_grouped_gemm \
  --source-entry _grouped_gemm_mxfp8 \
  --target-functions "_mxfp8_grouped_gemm_kernel,_grouped_gemm_mxfp8" \
  --workspace "$WORKSPACE" \
  --experiments-dir "$WORKSPACE/forge_experiments" \
  --result-json "$WORKSPACE/forge_experiments/forge_rewrite_result.json" \
  --flydsl-kernel-name kernel.py \
  --shapes-json '[{"tokens":1,"hidden":6144,"inter":384,"experts":128,"top_k":4,"dtype":"mxfp8"},{"tokens":64,"hidden":6144,"inter":384,"experts":128,"top_k":4,"dtype":"mxfp8"},{"tokens":16384,"hidden":6144,"inter":384,"experts":128,"top_k":4,"dtype":"mxfp8"}]' \
  --gpu-target "$GPU_TARGET" \
  --framework sglang \
  --snr-threshold 30.0 \
  --max-port-attempts "$MAX_PORT_ATTEMPTS" \
  --max-hours "$MAX_HOURS" \
  "${MODEL_ARGS[@]}"

echo "==> Done. FlyDSL kernel: $WORKSPACE/.forge_rewrite/<attempt>/kernel.py"
echo "    (the exact path is this run's temporary_paths in the result JSON)"
echo "    Rewrite result:     $WORKSPACE/forge_experiments/forge_rewrite_result.json"
