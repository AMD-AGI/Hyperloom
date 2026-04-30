#!/usr/bin/env bash
# inbox_scan.sh — triage helper. Tail the last N lines of every sibling
# agent's outbox.jsonl for cross-agent crash/stall correlation.
#
# Usage: bash scripts/inbox_scan.sh [N]
#   N defaults to 50.
set -u

N="${1:-50}"
SESSION_DIR="${SESSION_DIR:-$PWD}"

if [ ! -d "$SESSION_DIR/agents" ]; then
    echo "[inbox_scan] no agents/ dir under SESSION_DIR=$SESSION_DIR" >&2
    exit 0
fi

for d in "$SESSION_DIR"/agents/*/; do
    name="$(basename "$d")"
    if [ "$name" = "triage" ]; then
        continue   # skip self
    fi
    outbox="$d/outbox.jsonl"
    if [ -f "$outbox" ]; then
        echo "==== $name (tail -n $N) ===="
        tail -n "$N" "$outbox"
        echo ""
    else
        echo "==== $name (outbox.jsonl missing) ===="
        echo ""
    fi
done
