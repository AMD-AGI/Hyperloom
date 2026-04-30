# Wire Protocol — Quick Reference

> **Authoritative source**: `src/inference_optimizer/agents/PROTOCOL.md`
> in the parent of your skill dir. If `--add-dir` covers it (check via
> Glob), prefer reading that file. This page is the fallback inline
> summary for when PROTOCOL.md isn't visible.

## Envelope schema (one JSONL line)

```json
{
  "kind": "intent" | "message",
  "msg_id": "<uuid hex>",
  "seq":    <int>,
  "ts":     "<iso8601 utc microseconds>",
  "from_agent": "<agent name>",
  "to_agent":   "<agent name>" | "*" | "conductor",
  "payload":    { ... },

  // MESSAGE-only:
  "topic":       "<from TOPIC_ALLOWLIST>",
  "priority":    0..3,
  "in_reply_to": "<msg_id>" | null,

  // INTENT-only:
  "intent_type": "<from IntentType enum>"
}
```

## Per-restart Bash recipe (executor-flavoured)

```bash
LAST_SEQ=$(cat "$AGENT_DIR/inbox.jsonl.seq" 2>/dev/null || echo 0)

# Read every envelope newer than LAST_SEQ. Use python3 (jq + awk fiddly).
python3 - "$AGENT_DIR/inbox.jsonl" "$LAST_SEQ" <<'PY' > /tmp/new_inbox.jsonl
import json, sys
inbox, last = sys.argv[1], int(sys.argv[2])
try:
    with open(inbox, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("seq", 0) > last:
                print(json.dumps(obj))
except FileNotFoundError:
    pass
PY

# Reason over /tmp/new_inbox.jsonl. Then emit one or more intents:

emit_intent() {
    local intent_type=$1; shift
    local payload_json=$1; shift
    local in_reply_to=${1:-}
    local outbox="$AGENT_DIR/outbox.jsonl"
    local seq=$(($(wc -l < "$outbox" 2>/dev/null || echo 0) + 1))
    local msg_id=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
    local ts=$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat(timespec="microseconds"))')
    python3 - "$outbox" "$msg_id" "$seq" "$ts" "$intent_type" "$payload_json" "$in_reply_to" <<'PY'
import json, sys
outbox, msg_id, seq, ts, intent_type, payload, in_reply_to = sys.argv[1:]
env = {
    "kind": "intent",
    "msg_id": msg_id, "seq": int(seq), "ts": ts,
    "from_agent": __import__("os").environ.get("AGENT_NAME", "executor"),
    "to_agent": "conductor",
    "intent_type": intent_type,
    "payload": json.loads(payload),
}
if in_reply_to:
    env["in_reply_to"] = in_reply_to
with open(outbox, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(env) + "\n")
PY
}

# After handling all envelopes, persist the cursor:
NEW_SEQ=$(python3 -c 'import json,sys; m=0
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    try:
        s=json.loads(line).get("seq",0)
        if s>m: m=s
    except json.JSONDecodeError: pass
print(m)' /tmp/new_inbox.jsonl)
[ -n "$NEW_SEQ" ] && [ "$NEW_SEQ" -gt 0 ] && echo "$NEW_SEQ" > "$AGENT_DIR/inbox.jsonl.seq"
```

## In practice: use `emit_intent` MCP tool instead

In multi-cli mode the launcher injects an `emit_intent` MCP tool into
your Claude CLI. **Prefer the tool call** over the bash recipe — it's
cleaner and validated by PolicyGate without a Bash round-trip:

```
[Use the emit_intent MCP tool with input:]
{
  "intent_type": "delegate",
  "payload": {
    "action_name": "baseline",
    "params": {},
    "predicted_gain_pct": 0.0,
    "reason": "first measurement"
  }
}
```

The bash recipe above is for **debugging** or when you specifically
need to compose an envelope outside the MCP tool path (rare).

## Cursor invariants

- `inbox.jsonl.seq` — YOU write this, ONLY after successfully handling
  every envelope up to that seq. Resume = re-read inbox starting from
  this seq + 1.
- `outbox.jsonl` — each line gets a per-file monotonic seq (the bash
  helper bumps it via `wc -l`). The conductor assigns the global bus
  `seq` after ingest; that's invisible to you.
- `inbox.jsonl.mirrored` — Router-private; **never write** to it.
- `outbox.jsonl.cursor` — Router-private; **never write** to it.

## Stop signal

When `$SESSION_DIR/STOP_AGENT_executor` exists:

1. Finish the intent you're emitting *right now*.
2. Persist `inbox.jsonl.seq`.
3. Exit cleanly with `exit 0`.

The launcher's outer `while` loop honours the sentinel.
