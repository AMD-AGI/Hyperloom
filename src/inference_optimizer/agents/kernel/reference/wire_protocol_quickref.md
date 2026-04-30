# Wire Protocol — Quick Reference (Kernel Agent)

> Canonical source: `src/inference_optimizer/agents/PROTOCOL.md`. If
> Read can find that file, prefer it. This page is the inline
> fallback for kernel agent's specific request/response flow.

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

## Kernel agent's specific request/response flow

You see envelopes like this in `inbox.jsonl`:

```json
{
  "kind": "message",
  "msg_id": "fae3c1...",
  "seq": 42,
  "ts": "2026-04-29T07:30:00.000000+00:00",
  "from_agent": "executor",
  "to_agent": "kernel",
  "topic": "request",
  "priority": 2,
  "payload": {
    "kind": "select_kernels",
    "target_agent": "kernel",
    "params": {"trace_path": "/tmp/x.json.gz"},
    "reason": "have profile, need kernels to optimize"
  }
}
```

You respond by writing to `outbox.jsonl`:

```json
{
  "kind": "intent",
  "msg_id": "<new uuid>",
  "seq": 1,
  "ts": "<now>",
  "from_agent": "kernel",
  "to_agent": "conductor",
  "intent_type": "response",
  "payload": {
    "in_reply_to": "fae3c1...",   // <-- the request's msg_id
    "kind": "select_kernels_done",
    "status": "succeeded",
    "result": {"candidates": [...]}
  }
}
```

The conductor's `_handle_response` reverse-routes by `in_reply_to`,
addressing the response back to the executor (the request's original
sender).

## Per-restart Bash recipe (executor-flavoured, adapted)

```bash
LAST_SEQ=$(cat "$AGENT_DIR/inbox.jsonl.seq" 2>/dev/null || echo 0)

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

# Filter only request envelopes addressed to me.
python3 - <<'PY'
import json
for line in open("/tmp/new_inbox.jsonl"):
    e = json.loads(line)
    if e.get("topic") != "request": continue
    if e.get("to_agent") not in ("kernel", "*"): continue
    print(json.dumps(e))
PY
```

In practice, prefer the `emit_intent` MCP tool over the raw bash
recipe — it's validated by PolicyGate without round-trips.

## In practice: use `emit_intent` MCP tool

Reply by calling the MCP tool with this input:

```json
{
  "intent_type": "response",
  "payload": {
    "in_reply_to": "<request msg_id>",
    "kind": "select_kernels_done",
    "status": "succeeded",
    "result": {...}
  }
}
```

## Cursor invariants (same as executor)

- `inbox.jsonl.seq` — YOU write this, ONLY after handling all envelopes
  up to that seq.
- `outbox.jsonl` — append only.
- `inbox.jsonl.mirrored` / `outbox.jsonl.cursor` — Router-private.

## Stop signal

`$SESSION_DIR/STOP_AGENT_kernel` exists → finish current intent +
persist cursor + `exit 0`.
