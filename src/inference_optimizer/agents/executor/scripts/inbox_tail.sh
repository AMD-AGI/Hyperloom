#!/usr/bin/env bash
# inbox_tail.sh — print the last N envelopes from $AGENT_DIR/inbox.jsonl
# in a compact, human-readable form.
#
# Usage:   bash $AGENT_PKG_DIR/scripts/inbox_tail.sh [N]
# Env:     AGENT_DIR (defaults to $SESSION_DIR/agents/executor)
#          SESSION_DIR (required if AGENT_DIR not set)
#
# Output (one line per envelope):
#   seq=42 ts=2026-04-29T07:01:23 from=conductor topic=decision kind=state_updated changes={current_tput: 482.3}
#
# This script is read-only — it never writes to inbox/outbox/cursors.

set -euo pipefail

N="${1:-10}"

# Resolve AGENT_DIR
if [ -z "${AGENT_DIR:-}" ]; then
    if [ -z "${SESSION_DIR:-}" ]; then
        echo "ERROR: AGENT_DIR or SESSION_DIR must be set" >&2
        exit 2
    fi
    AGENT_DIR="$SESSION_DIR/agents/executor"
fi

INBOX="$AGENT_DIR/inbox.jsonl"
if [ ! -f "$INBOX" ]; then
    echo "(inbox.jsonl not found at $INBOX — agent has not received any events yet)"
    exit 0
fi

python3 - "$INBOX" "$N" <<'PY'
import json, sys

path, n_str = sys.argv[1], sys.argv[2]
n = max(1, int(n_str))

with open(path, "r", encoding="utf-8") as fh:
    lines = [ln.strip() for ln in fh if ln.strip()]

tail = lines[-n:]
for raw in tail:
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  (unparseable line: {raw[:80]}...)")
        continue
    seq = e.get("seq", "?")
    ts = e.get("ts", "?")
    fr = e.get("from_agent", "?")
    topic = e.get("topic") or e.get("intent_type") or "?"
    payload = e.get("payload") or {}
    # Try to pull the most informative key
    kind = payload.get("kind", "")
    summary = payload.get("summary") or payload.get("body_md") or ""
    if isinstance(summary, str) and len(summary) > 60:
        summary = summary[:57] + "..."
    extras = []
    for k in ("changes", "action_name", "task_id", "severity"):
        if k in payload:
            v = payload[k]
            if isinstance(v, str) and len(v) > 40:
                v = v[:37] + "..."
            extras.append(f"{k}={v}")
    extras_str = " ".join(extras)
    parts = [f"seq={seq}", f"ts={ts[:19] if isinstance(ts, str) else ts}",
             f"from={fr}", f"topic={topic}"]
    if kind:
        parts.append(f"kind={kind}")
    if extras_str:
        parts.append(extras_str)
    if summary and not extras_str:
        parts.append(f"summary={summary!r}")
    print(" ".join(parts))
PY
