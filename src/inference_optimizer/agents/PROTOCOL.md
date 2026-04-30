# Multi-CLI A2A Protocol — Agent Cookbook

Every agent CLI in `--transport multi-cli` follows the same wire
protocol. This file is the **canonical reference** every agent's
`system_prompt.md` points to. If something here disagrees with the role
prompt, the role prompt wins (it is allowed to add constraints, never
remove them).

## Files in your `$AGENT_DIR`

`$AGENT_DIR` resolves to `$SESSION_DIR/agents/<your-name>/` and is
exported into your environment by the launcher.

| file | who owns it | meaning |
| --- | --- | --- |
| `inbox.jsonl` | Router (write), agent (read) | bus events the Conductor routed to you, one envelope per line, monotonic `seq` |
| `inbox.jsonl.seq` | **agent** (read + write) | last envelope `seq` you have processed (advance after handling) |
| `inbox.jsonl.mirrored` | Router-private | last bus seq the Router mirrored. Do NOT touch. |
| `outbox.jsonl` | agent (write), Router (read) | intent envelopes you emit, one per line, per-file monotonic `seq` |
| `outbox.jsonl.cursor` | Router-private | byte offset already drained. Do NOT touch. |
| `conversation.jsonl` (Codex only) | launcher | prior turns reinjected on every restart |

## Envelope schema (canonical)

Every line of `inbox.jsonl` and `outbox.jsonl` is one JSON object with
this shape (pretty-printed for clarity; on disk it is one line):

```json
{
  "kind": "intent" | "message",
  "msg_id": "<uuid hex>",
  "seq": <int, per-file monotonic>,
  "ts": "<iso8601 utc, microseconds>",
  "from_agent": "<your-name>" | "conductor" | "clock" | other,
  "to_agent": "<target-name>" | "*" | "conductor",
  "payload": { /* per-intent or per-topic body */ },

  // MESSAGE-only:
  "topic": "event" | "alert" | "proposal" | "decision" | "reflection_tick" | ...,
  "priority": 0..3,
  "in_reply_to": "<msg_id>" | null,

  // INTENT-only:
  "intent_type": "send_message" | "delegate" | "propose_action" |
                 "objection" | "vote" | "update_state" |
                 "update_persona" | "ask_question" | "answer" | "alert"
}
```

Inbox envelopes use `kind="message"`. Outbox envelopes use `kind="intent"`.
PolicyGate (in the Conductor process) re-validates every outbox intent
against your role's `allowed_intents` — emitting a forbidden intent
type produces a `policy_denied` observation in your next inbox tick.

## Per-restart recipe (Bash, copy-paste safe)

```bash
# 1. Discover where you left off.
LAST_SEQ=$(cat "$AGENT_DIR/inbox.jsonl.seq" 2>/dev/null || echo 0)

# 2. Read every envelope newer than LAST_SEQ. We use python3 for the
#    JSON parsing because awk + jq combinations get fiddly with the
#    seq field type.
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

# 3. (Reason over /tmp/new_inbox.jsonl — that's your "what happened
#    since I last looked" tail.)

# 4. Emit one intent envelope per response. Helper:
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
    "from_agent": __import__("os").environ.get("AGENT_NAME", "?"),
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

# 5. Persist your cursor BEFORE exiting so the next restart skips the
#    envelopes you already handled.
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

## Worked example — Executor sees `run_started`

After the Conductor starts, your inbox tail will contain something like:

```json
{"kind":"message","msg_id":"abc...","seq":1,"ts":"2026-04-29T05:00:00Z","from_agent":"conductor","to_agent":"*","topic":"event","payload":{"kind":"run_started","mode":"marathon_multi_agent","model_path":"/.../model"},"priority":0}
```

The expected first response from the Executor is to delegate `baseline`:

```bash
emit_intent delegate '{
  "action_name": "baseline",
  "params": {},
  "predicted_gain_pct": 0.0,
  "reason": "first measurement; need baseline_tput before optimising"
}'
```

After you write this line, the Conductor picks it up within
`router-tick-s` (default 0.5s), PolicyGate accepts (executor role
allows `delegate`), and a `proposal` topic event lands on the bus +
a queued `delegate` task is created. Within the Conductor the
SubAgentRunner will dispatch `BaselineExecutor` which actually shells
out to `scripts/run_baseline.sh`.

## Stop signal

When `$SESSION_DIR/STOP_AGENT_<your-name>` exists:

* finish the intent you're emitting *right now*
* persist your cursor
* exit cleanly with `exit 0`

The launcher's outer `while` loop honours the sentinel and will not
re-enter `claude --print --continue` while the file is present.

## Hard rules carried over from the legacy reactor

* **PolicyGate is enforced in the Conductor process.** A `delegate`
  for an unknown action — or any intent your role isn't allowed to
  emit — will land as a `policy_denied` observation in your next
  inbox tick. Do not retry the same envelope; pivot.
* **Idempotency:** the Conductor de-dups `delegate(action_name, params)`
  pairs by (kind, agent, action_name, params) hash. Re-emitting an
  identical delegate after the task already terminated will produce a
  `delegate_dedup_to_terminal` event in your inbox — change params or
  pick a different action.
* **Never write to `inbox.jsonl.mirrored` or `outbox.jsonl.cursor`.**
  Those are Router-private bookkeeping. Touching them will desync
  your view of bus events.
