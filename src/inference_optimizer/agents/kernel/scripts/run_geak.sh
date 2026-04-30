#!/usr/bin/env bash
# run_geak.sh — submit one kernel candidate to GEAK via Ray.
#
# Wraps the package-shipped geak_ray_submit.py with sensible defaults
# (per-task timeout, --yolo). The kernel agent calls this in parallel
# (one per candidate) for IR-1.
#
# Usage:
#   bash run_geak.sh <kernel_file>
#
# Env (forwarded to geak_ray_submit.py):
#   GEAK_CONFIG, GEAK_MODEL_NAME, GEAK_API_KEY, GEAK_BASE_URL
#   AMD_LLM_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY (litellm fallback)
#   GEAK_GPUS_PER_TASK (default 1)
#
# Output: writes log under $RESULT_DIR/geak_<kernel_stem>.log if set,
# else streams to stdout. Exits with the geak_ray_submit.py rc.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: run_geak.sh <kernel_file>" >&2
    exit 2
fi

KERNEL_FILE="$1"
if [ ! -f "$KERNEL_FILE" ]; then
    echo "ERROR: kernel file not found: $KERNEL_FILE" >&2
    exit 1
fi

# Resolve the package's geak_ray_submit.py via the agent_pkg layout.
# AGENT_PKG_DIR is exported by the launcher's .env when present; fall
# back to deriving from this script's location.
AGENT_PKG_DIR="${AGENT_PKG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/..}"
PKG_ROOT="$(cd "$AGENT_PKG_DIR/../.." && pwd)"
SUBMIT_SCRIPT="$PKG_ROOT/scripts/geak_ray_submit.py"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "ERROR: geak_ray_submit.py not found at $SUBMIT_SCRIPT" >&2
    exit 1
fi

KERNEL_STEM="$(basename "$KERNEL_FILE" | sed 's/\.[^.]*$//')"
LOG_PATH="${RESULT_DIR:-/tmp}/geak_${KERNEL_STEM}.log"
mkdir -p "$(dirname "$LOG_PATH")"

echo "[$(date -Iseconds)] run_geak.sh: kernel=$KERNEL_FILE log=$LOG_PATH" >&2

python3 "$SUBMIT_SCRIPT" run -t "$KERNEL_FILE" --yolo \
    >> "$LOG_PATH" 2>&1
RC=$?
echo "[$(date -Iseconds)] run_geak.sh: rc=$RC" >&2
exit $RC
