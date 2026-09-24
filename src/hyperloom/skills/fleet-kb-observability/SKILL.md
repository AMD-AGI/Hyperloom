---
name: fleet-kb-observability
description: Connects Fleet KB operation events to Slack job threads and explains which Experiences were retrieved, injected, and written. Use when implementing the Slack consumer, diagnosing missing KB messages, or auditing knowledge reuse.
---

# Fleet KB observability

Consume `GET /v1/events?after=<cursor>&limit=100` with the Fleet Bot token and ID.
Run `scripts/fetch_events.py` for a non-mutating connectivity check.

For the standalone demo bridge, also set `SLACK_BOT_TOKEN` and
`SLACK_CHANNEL_ID`. Set `HYPERLOOM_FLEET_KB_BOT_TOKEN` separately from the
worker token, then run:

```bash
python3 scripts/slack_bridge.py
```

The bridge uses Fleet `correlation.thread_id` as `thread_ts`, sends
`event_id` as Slack `client_msg_id`, and persists its event cursor in SQLite.

## Delivery rules

1. Keep one durable event cursor per fleet.
2. Deduplicate by `event_id`.
3. Route by `correlation.thread_id`; fall back to `job_id`.
4. Send the Slack message.
5. Advance the cursor only after Slack accepts the message.
6. Retry Slack failures without replaying the KB operation.

Do not put full prompt context, Experience reasoning, patches, or credentials
in chat. Link to an access-controlled detail view when needed.

## Human conversation flow

The central Slack bot owns language understanding; Fleet KB exposes explicit,
auditable tools.

1. After the user provides Run parameters, call
   `FleetBotClient.list_catalog()` and page through all `unverified` records.
2. Show bounded cards for every page. An optional natural-language relevance
   question may additionally call `discover()`, but discovery must not hide the
   complete Catalog review.
3. Do not verify anything from an ambiguous acknowledgement.
4. When the user explicitly chooses Experiences, call
   `FleetBotClient.verify_for_run()` with the displayed IDs, Slack user ID,
   reason, thread ID, and exact target `scope_id`.
5. Confirm each ID as `verified_for_scope`. The Catalog record remains
   `unverified`; another scope cannot inherit the verification.
6. Launch Hyperloom with `HYPERLOOM_FLEET_KB_SCOPE_ID` equal to that same
   target scope.

The reference tool is `scripts/fleet_conversation.py`:

```bash
python3 scripts/fleet_conversation.py catalog \
  --scope-id "$SLACK_JOB_ID" --limit 100

python3 scripts/fleet_conversation.py verify \
  --scope-id "$SLACK_JOB_ID" --actor-id "$SLACK_USER_ID" \
  --thread-id "$SLACK_THREAD_ID" \
  --experience-id "$EXP_1" --experience-id "$EXP_2" \
  --reason "$USER_MESSAGE"
```

## Read message

For `kb.read.completed`, render:

```text
KB read completed · worker=<worker_id> · run=<run_id>
Signals: <field/text summary and weights>
Eligible: <scope-verified Experience count>
Found: <Repeat Group count>
Injected: <rendered Experience IDs>
Omitted: <verified IDs that did not match>
Latency: <latency_ms> ms
Warnings: <warnings or none>
```

The event's `rendered_refs`, not every retrieved candidate, identify evidence
that actually entered the agent prompt.

Each terminal read event also retains `request_seed.decision` and
`request_seed.context` exactly as accepted by the Fleet API. Treat this as a
durable candidate seed for future Planner Test Cases. Do not copy the full seed
into Slack; the bridge renders only the bounded Planner signal summary.

For unavailable/failed reads, show a warning and state that Hyperloom
continued without KB evidence.

## Write message

For `kb.experience.cataloged`, render:

```text
KB Experience <created|unchanged> · unverified
Change: <change_summary>
Decision: <keep|revert>
Measurement: <baseline_value> → <outcome_value>
Experience: <experience_id>
```

If the worker receipt says `spooled`, show that the measurement is durable on
the worker but not yet visible to fleet reads.

## Reuse chain

When possible, show:

```text
read_id → rendered_refs → proposal/attempt → new Experience → outcome
```

This distinguishes retrieval from actual prompt exposure and measured
follow-up.

## Diagnosis

- No events: verify `/health`, token, Fleet ID header, and cursor.
- Reads but no Slack messages: inspect thread/job correlation and Slack retry
  state.
- Read events but empty refs: inspect Planner signals and ranked groups.
- Writes remain spooled: restore central connectivity and run
  `FleetKBClient.flush_spool()` on the worker.
- Duplicate Slack messages: repair event-id dedup before changing Fleet KB.
