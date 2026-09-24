# Fleet KB customer demo

The customer demo uses one Fleet KB service on the Slack central server.
Workers catalog immutable Experiences there as `unverified`. Humans can
discover those records across runs, but Hyperloom can read only the
Experiences explicitly marked `verified_for_scope` for its own Run scope.

## Central server

Install the `demo/fleet-kb-service` Hyperloom-KB branch and put the service
behind the Slack server's TLS reverse proxy:

```bash
hyperloom-kb-fleet-serve \
  --host 127.0.0.1 \
  --port 8787 \
  --fleet-id customer-demo \
  --home /var/lib/hyperloom-fleet-kb \
  --declaration ./declarations/inference-recipe-v1.yaml
```

The service starts with an empty catalog. This keeps the demo trace explicit:
Run A creates however many measured Experiences its attempts produce; no older
or bundled corpus is loaded.

The process also needs `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`,
`LOCAL_KB_PLANNER_MODEL`, `HYPERLOOM_FLEET_KB_WORKER_TOKEN`, and a distinct
`HYPERLOOM_FLEET_KB_BOT_TOKEN`.

## Slack agent install

Install the Hyperloom demo branch into the workspace where the Slack server
starts Claude:

```bash
cd "$SLACK_CLAUDE_WORKSPACE"
python3 -m pip install --upgrade --target . \
  "git+https://github.com/AMD-AGI/Hyperloom.git@demo/fleet-kb-integration"
```

Restart Claude in that workspace and run the existing setup entry point:

```text
/hyperloom-setup
```

Setup installs/verifies the Hyperloom-KB product tools automatically when
needed. It does not ask for Fleet configuration and does not write Fleet token
placeholders. The existing optimizer Skill at `HYPERLOOM_SKILL_PATH` supplies
the per-Run Fleet overlay to every selected workload Skill.

## Worker launch

Install the matching Hyperloom-KB demo branch in the worker environment:

```bash
python3 -m pip install \
  "git+https://github.com/zili-amd/Hyperloom-KB.git@demo/fleet-kb-service"
```

When Slack starts Hyperloom over SSH, inject:

```bash
export HYPERLOOM_KB_ENABLE=true
export HYPERLOOM_KB_DECL=/workspace/Hyperloom/examples/hyperloom-kb-inference.yaml
export HYPERLOOM_FLEET_KB_URL=https://slack-central.example/fleet-kb
export HYPERLOOM_FLEET_KB_WORKER_TOKEN=...
export HYPERLOOM_FLEET_KB_ID=customer-demo
export HYPERLOOM_FLEET_KB_WORKER_ID="$(hostname)"
export HYPERLOOM_FLEET_KB_SCOPE_ID="$SLACK_JOB_ID"
export HYPERLOOM_FLEET_KB_JOB_ID="$SLACK_JOB_ID"
export HYPERLOOM_FLEET_KB_THREAD_ID="$SLACK_THREAD_ID"
export HYPERLOOM_FLEET_KB_SPOOL="${USER_DATA_PATH}/fleet-kb-spool/$SLACK_JOB_ID"

# Recommended MI325X example. The hardware probe wins if it reports otherwise.
export TARGET_GPU_TYPE=mi325x
```

Do not set `GPU_TYPE` from the example. Hyperloom persists the resolved real
board (`mi325x`) but may internally export `GPU_TYPE=mi300x` as the Magpie
runner label because MI300X and MI325X both use gfx942.

During FRAMEWORK_AGENT, the orchestration prompt performs one Fleet read for
each proposal-generation decision. Retries inside that decision reuse the same
read; a later decision re-reads the current shared corpus so writes from other
workers can become visible. A successful evidence block is inserted before the
orchestration model proposes work. Its rendered Experience references are
recorded on the proposal and preserved in the measured Experience. Historical
evidence may seed a proposal or current-best candidate, but the original Recipe
measurement remains the immutable gain-accounting baseline.

Before launching Run B, the Slack Bot uses its Bot credential to list the full
unverified Catalog, with optional relevance discovery. It marks chosen records
`verified_for_scope` only after an explicit user decision. The Catalog records
remain unverified, and Run C cannot inherit Run B's verification.

Hyperloom captures runtime `identity`, `workload`, `objective`,
`benchmark_baseline`, `current_best`, `observations`, `recent_results`, and
`already_tried`. The workflow does not construct search fields, weights, or a
QueryPlan; the central Planner derives them from this decision context.
Use the project `fleet-kb-integrate` Skill when adapting this boundary to
another optimization workflow.

At CLOSE, the existing Experience publisher uses the same SDK. When the Fleet
URL is configured, completed Experiences enter the central catalog as
unverified and unavailable to automated reads. Failed network writes spool on
the worker for idempotent retry.

## Slack event cursor

The Slack service polls:

```http
GET /v1/events?after=<last_sequence>&limit=100
Authorization: Bearer <token>
X-Hyperloom-Fleet-ID: customer-demo
```

Render `kb.experience.cataloged`, optional `kb.discovery.completed`,
`kb.experiences.verified_for_scope`, and `kb.read.completed` in the Hyperloom job
thread. Read events include signals, top Repeat Groups, actual rendered
Experience IDs, latency, and warnings. The accepted decision context is
retained as a future Test Case seed, but the Slack message renders only bounded
signal summaries. Write events include the change, keep/revert outcome,
measurements, content hash, and corpus sequence.

Persist the returned cursor only after Slack accepts the corresponding message.
Re-reading a cursor is safe because each event has a stable `event_id`.

## Demo failure behavior

- No scope verification produces a successful empty read.
- Read failure is advisory: Hyperloom continues without a KB prompt block.
- Write failure does not discard the Experience: the worker spool retains it.
- Slack failure does not roll back KB operations: the central SQLite outbox
  retains events.
- Operation IDs and Experience IDs are idempotent.

The demo central service uses canonical filesystem storage and SQLite. Those
are adapter boundaries; a later Postgres deployment does not change worker
APIs or Hyperloom prompt/write wiring.
