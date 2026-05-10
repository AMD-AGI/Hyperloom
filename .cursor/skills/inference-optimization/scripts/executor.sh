#!/usr/bin/env bash
# =============================================================================
# executor.sh — Execution backend abstraction layer
#
# Dispatches GPU-side commands to the correct backend:
#   - local: execute directly (eval)
#   - remote: submit to RayJob cluster via ray_submit.py
#
# Required env:
#   MODE             — "local" or "remote"
#   RAY_HEAD_ADDRESS — Ray client address (remote mode only), e.g. ray://<head>:10001
#
# Usage:
#   source executor.sh
#   exec_on_gpu "python3 -m sglang.launch_server --model $MODEL --tp $TP"
#   exec_on_gpu "bash \$SCRIPTS_DIR/run_baseline.sh"
# =============================================================================

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec_on_gpu() {
    local cmd="$1"
    local timeout="${2:-3600}"

    if [ "$MODE" = "local" ]; then
        eval "$cmd"
    elif [ "$MODE" = "remote" ]; then
        if [ -z "$RAY_HEAD_ADDRESS" ]; then
            echo "ERROR: RAY_HEAD_ADDRESS not set (required in remote mode)" >&2
            return 1
        fi
        python3 "$SCRIPTS_DIR/ray_submit.py" \
            --ray-address "$RAY_HEAD_ADDRESS" \
            --command "$cmd" \
            --timeout "$timeout"
    else
        echo "ERROR: Unknown MODE='$MODE'. Expected 'local' or 'remote'." >&2
        return 1
    fi
}

exec_on_gpu_bg() {
    local cmd="$1"

    if [ "$MODE" = "local" ]; then
        eval "$cmd" &
        echo $!
    elif [ "$MODE" = "remote" ]; then
        if [ -z "$RAY_HEAD_ADDRESS" ]; then
            echo "ERROR: RAY_HEAD_ADDRESS not set (required in remote mode)" >&2
            return 1
        fi
        python3 "$SCRIPTS_DIR/ray_submit.py" \
            --ray-address "$RAY_HEAD_ADDRESS" \
            --command "nohup $cmd > /dev/null 2>&1 & echo \$!" \
            --timeout 30
    else
        echo "ERROR: Unknown MODE='$MODE'. Expected 'local' or 'remote'." >&2
        return 1
    fi
}
