# Fleet KB customer demo

The customer demo uses one Fleet KB service on the Slack central server. Every
Hyperloom worker connects outbound to that service; workers do not maintain
writable KB replicas.

## Central server

Install the `demo/fleet-kb-service` Hyperloom-KB branch, put the service behind
the Slack server's TLS reverse proxy, and seed the reviewed Experience payload:

```bash
hyperloom-kb-fleet-serve \
  --host 127.0.0.1 \
  --port 8787 \
  --fleet-id customer-demo \
  --home /var/lib/hyperloom-fleet-kb \
  --declaration ./declarations/inference-recipe-v1.yaml
```

The Fleet KB demo branch automatically loads its bundled 59-Experience seed.
Repeated service starts are idempotent.

The process also needs `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`,
`LOCAL_KB_PLANNER_MODEL`, and `HYPERLOOM_FLEET_KB_TOKEN`.

## Worker launch

When Slack starts Hyperloom over SSH, inject:

```bash
export HYPERLOOM_KB_ENABLE=true
export HYPERLOOM_KB_DECL=/workspace/Hyperloom/examples/hyperloom-kb-inference.yaml
export HYPERLOOM_FLEET_KB_URL=https://slack-central.example/fleet-kb
export HYPERLOOM_FLEET_KB_TOKEN=...
export HYPERLOOM_FLEET_KB_ID=customer-demo
export HYPERLOOM_FLEET_KB_WORKER_ID="$(hostname)"
export HYPERLOOM_FLEET_KB_JOB_ID="$SLACK_JOB_ID"
export HYPERLOOM_FLEET_KB_THREAD_ID="$SLACK_THREAD_ID"
export HYPERLOOM_FLEET_KB_SPOOL="$SESSION_DIR/fleet-kb-spool"
```

During FRAMEWORK_AGENT, the orchestration prompt performs one Fleet read per
tick. A successful evidence block is inserted before the orchestration model
proposes work. Its rendered Experience references are recorded on the proposal
and preserved in the measured Experience.

At CLOSE, the existing Experience publisher uses the same SDK. When the Fleet
URL is configured, completed Experiences publish to the central service.
Failed network writes spool on the worker for idempotent retry.

## Slack event cursor

The Slack service polls:

```http
GET /v1/events?after=<last_sequence>&limit=100
Authorization: Bearer <token>
X-Hyperloom-Fleet-ID: customer-demo
```

Render `kb.read.completed` and `kb.experience.published` in the Hyperloom job
thread. Read events include signals, top Repeat Groups, actual rendered
Experience IDs, latency, and warnings. Write events include the change,
keep/revert outcome, measurements, content hash, and corpus sequence.

Persist the returned cursor only after Slack accepts the corresponding message.
Re-reading a cursor is safe because each event has a stable `event_id`.

## Demo failure behavior

- Read failure is advisory: Hyperloom continues without a KB prompt block.
- Write failure does not discard the Experience: the worker spool retains it.
- Slack failure does not roll back KB operations: the central SQLite outbox
  retains events.
- Operation IDs and Experience IDs are idempotent.

The demo central service uses canonical filesystem storage and SQLite. Those
are adapter boundaries; a later Postgres deployment does not change worker
APIs or Hyperloom prompt/write wiring.
