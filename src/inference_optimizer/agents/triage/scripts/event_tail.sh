#!/usr/bin/env bash
# event_tail.sh — triage helper. Pull the last N rows from the SQLite
# events table (ordered by seq DESC) so triage can grep recent bus
# activity without scanning JSONL files individually.
#
# Usage: bash scripts/event_tail.sh [N]
#   N defaults to 100.
set -u

N="${1:-100}"
SESSION_DIR="${SESSION_DIR:-$PWD}"
DB="$SESSION_DIR/storage/conductor.db"

if [ ! -f "$DB" ]; then
    echo "[event_tail] no SQLite at $DB" >&2
    exit 0
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
    echo "[event_tail] sqlite3 binary missing on PATH" >&2
    exit 0
fi

sqlite3 -readonly "$DB" \
    ".headers on" ".mode column" \
    "SELECT seq, ts, from_agent, to_agent, topic FROM events
     ORDER BY seq DESC LIMIT $N;"
