#!/bin/bash
# Bare-metal setup for Hyperloom local-local mode (no Docker required).
# Usage: bash deploy/local-local/setup-bare-metal.sh [--install-only | --start-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Defaults (override via env) ----
export GEAK_INSTALL_DIR="${GEAK_INSTALL_DIR:-/opt/geak}"
export INTELLIKIT_INSTALL_DIR="${INTELLIKIT_INSTALL_DIR:-/opt/intellikit}"
export TRACELENS_INSTALL_DIR="${TRACELENS_INSTALL_DIR:-/opt/TraceLens}"
export NFS_BASE_PATH="${NFS_BASE_PATH:-/tmp/geak-data}"
export DATABASE_PATH="${DATABASE_PATH:-${NFS_BASE_PATH}/geak.db}"
export TRACELENS_PORT="${TRACELENS_PORT:-8001}"
export GEAK_MCP_PORT="${GEAK_MCP_PORT:-8002}"
export MODE=local
export GEAK_LOCAL=true
export KERNEL_OPT_BACKENDS=geak
export FRAMEWORK="${FRAMEWORK:-sglang}"
export INFERENCEX_PATH="${REPO_ROOT}/inference_optimization/InferenceX"
export LLM_API_BASE="${LLM_API_BASE:-}"
export LLM_API_KEY="${LLM_API_KEY:-}"

PIDS=()

cleanup() {
    echo ""
    echo "[shutdown] stopping services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "[shutdown] done."
}
trap cleanup EXIT INT TERM

# ============================================================
#  Phase 1: Install dependencies
# ============================================================
do_install() {
    echo "============================================"
    echo "  Hyperloom — Bare-Metal Install"
    echo "============================================"

    # TraceLens
    if [ ! -d "$TRACELENS_INSTALL_DIR/TraceLens" ]; then
        echo "[install] copying TraceLens..."
        cp -r "${REPO_ROOT}/TraceLens-internal" "$TRACELENS_INSTALL_DIR"
    fi
    echo "[install] pip install TraceLens..."
    cd "$TRACELENS_INSTALL_DIR"
    pip install --no-cache-dir -e . > /dev/null
    pip install --no-cache-dir -r TraceLens/AgenticMode/MCPServer/requirements.txt > /dev/null

    # GEAK
    if [ ! -d "$GEAK_INSTALL_DIR/.git" ]; then
        echo "[install] cloning GEAK..."
        git clone --branch feature/xiaofei/claw --depth 1 \
            https://github.com/AMD-AGI/GEAK.git "$GEAK_INSTALL_DIR"
    fi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    echo "[install] pip install GEAK..."
    cd "$GEAK_INSTALL_DIR"
    pip install --no-cache-dir -e . > /dev/null
    pip install --no-cache-dir -r server/requirements.txt > /dev/null

    # intellikit
    if [ ! -d "$INTELLIKIT_INSTALL_DIR/.git" ]; then
        echo "[install] cloning intellikit..."
        git clone --depth 1 \
            https://github.com/AMDResearch/intellikit.git "$INTELLIKIT_INSTALL_DIR"
    fi
    echo "[install] pip install intellikit/metrix..."
    pip install --no-cache-dir -e "$INTELLIKIT_INSTALL_DIR/metrix/" > /dev/null

    mkdir -p "$NFS_BASE_PATH"
    echo "[install] done."
    echo ""
}

# ============================================================
#  Phase 2: Start services
# ============================================================
do_start() {
    # GPU inventory (prefer K8s/env allocation over hardware scan)
    if [ -n "${GPUS_PER_NODE:-}" ]; then
        GPU_COUNT="$GPUS_PER_NODE"
    elif [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
        GPU_COUNT=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    elif [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        GPU_COUNT=$(echo "$ROCR_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    elif command -v amd-smi &>/dev/null; then
        GPU_COUNT=$(amd-smi list 2>/dev/null | grep -c "^GPU:" || echo 0)
    elif command -v rocm-smi &>/dev/null; then
        GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "^[0-9]" || echo 0)
    else
        GPU_COUNT=0
    fi

    echo "============================================"
    echo "  Hyperloom — Bare-Metal Mode"
    echo "============================================"
    echo "  GPU:          ${GPU_COUNT} detected"
    echo "  Framework:    ${FRAMEWORK}"
    echo "  GEAK data:    ${NFS_BASE_PATH}"
    echo "  TraceLens:    :${TRACELENS_PORT}"
    echo "  GEAK MCP:     :${GEAK_MCP_PORT}"
    echo "  LLM API:      ${LLM_API_BASE:-<not set>}"
    echo "============================================"
    echo ""

    # TraceLens MCP
    echo "[start] TraceLens MCP on :${TRACELENS_PORT}..."
    TRACELENS_PORT="$TRACELENS_PORT" \
    TRACELENS_HOST=0.0.0.0 \
        python -m TraceLens.AgenticMode.MCPServer &
    PIDS+=($!)

    # GEAK MCP
    echo "[start] GEAK MCP on :${GEAK_MCP_PORT}..."
    cd "$GEAK_INSTALL_DIR"
    GEAK_LOCAL=true \
    HOST=0.0.0.0 \
    MCP_PORT="$GEAK_MCP_PORT" \
    NFS_BASE_PATH="$NFS_BASE_PATH" \
    DATABASE_PATH="$DATABASE_PATH" \
        python -m server.mcp.http_server &
    PIDS+=($!)

    echo ""
    echo "Services started (PIDs: ${PIDS[*]})"
    echo "  TraceLens → http://localhost:${TRACELENS_PORT}/mcp"
    echo "  GEAK      → http://localhost:${GEAK_MCP_PORT}/sse"
    echo ""

    wait -n
    EXIT_CODE=$?
    echo "[error] a service exited with code $EXIT_CODE"
    exit "$EXIT_CODE"
}

# ============================================================
#  Main
# ============================================================
case "${1:-all}" in
    --install-only) do_install ;;
    --start-only)   do_start ;;
    *)              do_install; do_start ;;
esac
