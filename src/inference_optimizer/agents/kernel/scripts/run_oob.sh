#!/usr/bin/env bash
# run_oob.sh — submit one OOB (codex / claude / llm) round via Ray.
#
# Wraps the package-shipped oob_ray_submit.py. The kernel agent calls
# this in parallel with run_geak.sh for IR-1.
#
# Usage:
#   bash run_oob.sh <agent> <kernel_file> <prompt_file>
#
# Where <agent> is one of: codex / claude / llm.
#
# Env (forwarded to oob_ray_submit.py):
#   OOB_API_KEY, OOB_BASE_URL, OOB_LOCAL, OOB_CLI, OOB_HOME
#   ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
#   OPENAI_API_KEY, OPENAI_BASE_URL
#   OOB_GPUS_PER_TASK (default 1)
#
# Output: writes log under $RESULT_DIR/<agent>_<kernel_stem>.log if set.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: run_oob.sh <agent> <kernel_file> <prompt_file>" >&2
    exit 2
fi

OOB_AGENT="$1"
KERNEL_FILE="$2"
PROMPT_FILE="$3"
case "$OOB_AGENT" in
    codex|claude|llm) ;;
    *)
        echo "ERROR: agent must be one of codex/claude/llm, got $OOB_AGENT" >&2
        exit 2
        ;;
esac
if [ ! -f "$KERNEL_FILE" ]; then
    echo "ERROR: kernel file not found: $KERNEL_FILE" >&2
    exit 1
fi
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

AGENT_PKG_DIR="${AGENT_PKG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/..}"
PKG_ROOT="$(cd "$AGENT_PKG_DIR/../.." && pwd)"
SUBMIT_SCRIPT="$PKG_ROOT/scripts/oob_ray_submit.py"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "ERROR: oob_ray_submit.py not found at $SUBMIT_SCRIPT" >&2
    exit 1
fi

MAX_TURNS="${OOB_MAX_TURNS:-30}"
KERNEL_STEM="$(basename "$KERNEL_FILE" | sed 's/\.[^.]*$//')"
LOG_PATH="${RESULT_DIR:-/tmp}/${OOB_AGENT}_${KERNEL_STEM}.log"
mkdir -p "$(dirname "$LOG_PATH")"

echo "[$(date -Iseconds)] run_oob.sh: agent=$OOB_AGENT kernel=$KERNEL_FILE log=$LOG_PATH" >&2

python3 "$SUBMIT_SCRIPT" run \
    -a "$OOB_AGENT" \
    -p "@$PROMPT_FILE" \
    -f "$KERNEL_FILE" \
    --max-turns "$MAX_TURNS" \
    --no-live --json \
    >> "$LOG_PATH" 2>&1
RC=$?
echo "[$(date -Iseconds)] run_oob.sh: rc=$RC" >&2
exit $RC
