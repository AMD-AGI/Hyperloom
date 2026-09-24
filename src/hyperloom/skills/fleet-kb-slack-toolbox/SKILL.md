---
name: fleet-kb-slack-toolbox
description: Gives an existing Slack agent the exact Fleet KB tools and state machine for listing unverified Experiences, recording run-scoped verification, launching a Hyperloom Worker job, and reporting events. Use on the Slack central server after Fleet KB service setup is healthy.
---

# Fleet KB Slack toolbox

This Skill belongs on the Slack central server. Assume the Slack runtime already
knows its Worker execution environment and already receives user messages. Do
not deploy Fleet KB here and do not ask users to call APIs.

## Install the tool package

Use Hyperloom-KB branch `demo/fleet-kb-service` in the Slack tool environment:

```bash
python3 -m pip install --upgrade \
  "git+https://github.com/zili-amd/Hyperloom-KB.git@demo/fleet-kb-service"
```

The tool process receives only:

```bash
export HYPERLOOM_FLEET_KB_URL=...
export HYPERLOOM_FLEET_KB_BOT_TOKEN=...
export HYPERLOOM_FLEET_KB_ID=customer-demo
```

Do not expose the Bot token in a model prompt or tool result.

## Register these tools

```python
from hyperloom_kb import SlackFleetTools

fleet = SlackFleetTools.from_env()

list_kb_experiences = fleet.list_kb_experiences
discover_kb_experiences = fleet.discover_kb_experiences
verify_kb_experiences_for_scope = fleet.verify_kb_experiences_for_scope
list_scope_verified_experiences = fleet.list_scope_verified_experiences
list_kb_events = fleet.list_kb_events
```

Contracts:

- `list_kb_experiences(scope_id, after, limit)` pages through every
  `unverified` Catalog record. Continue until `has_more=false`.
- `discover_kb_experiences(...)` is optional semantic relevance search. It
  never replaces complete Catalog listing.
- `verify_kb_experiences_for_scope(...)` records the user's exact IDs,
  `verified_by`, `verified_at`, reason, and target scope.
- `list_scope_verified_experiences(scope_id)` is the launch gate.
- `list_kb_events(after, limit)` drives Slack status delivery.

If direct Python tool registration is unavailable, use
`scripts/toolbox_cli.py` with the equivalent `catalog`, `discover`, `verify`,
`verifications`, and `events` commands.

## Thread state

Keep this server-side state per Slack thread:

```json
{
  "scope_id": "the planned Slack job id",
  "run_parameters": {},
  "catalog_cursor": 0,
  "displayed_experience_ids": [],
  "verified_experience_ids": [],
  "worker_job_id": ""
}
```

The scope is created before Catalog review and is reused unchanged for
verification and Worker launch. Never infer a scope from an Experience ID.

## Conversation behavior

When the user provides run parameters:

1. Allocate the planned job/scope ID.
2. Page through `list_kb_experiences` and show all unverified records in
   bounded Slack messages.
3. Include Experience ID, source Run, identity/workload summary, change,
   KEEP/REVERT, and measurement.
4. State that nothing is usable until the user chooses it for this scope.

When the user chooses IDs:

1. Accept only IDs actually displayed in the thread.
2. Call `verify_kb_experiences_for_scope`.
3. Confirm `verified_for_scope`; never say the Catalog record became globally
   verified.
4. Refresh `list_scope_verified_experiences`.

An acknowledgement such as “OK” is not consent. Require an explicit choice
such as “use these two for this run” or a clear list of IDs.

## Start Hyperloom

When the user asks to start:

1. Read the thread's run parameters and exact scope.
2. Call `list_scope_verified_experiences`. Zero results are allowed only after
   the user explicitly chooses to continue without prior Experience.
3. Invoke the Slack runtime's existing Worker execution mechanism.
4. Follow the Hyperloom workload Skill selected by the user's run request.
5. Apply the installed `hyperloom-fleet-worker-run` overlay in that same Worker
   execution.
6. Inject the same `scope_id` as `HYPERLOOM_FLEET_KB_SCOPE_ID`, along with the
   Worker token from the secret store. Never send the Bot token.

The workload Skill and Hyperloom CLI own all model, workload, launch, budget,
and monitoring behavior. The Worker overlay adds only Fleet branch/SDK,
environment, cold-start, launch flag, and read/write acceptance. This toolbox
owns the conversation and tool calls only.

## Report

Use the event cursor to post:

- each new `kb.experience.cataloged` record as unverified;
- `kb.experiences.verified_for_scope` with reviewer and scope;
- `kb.read.completed` with the actually injected refs;
- write spool/failure warnings.

Advance the event cursor only after Slack accepts the message. Never claim an
Experience caused an improvement merely because it was injected.

## Failure rules

- HTTP 401: verify endpoint role, Bot token, and Fleet ID; do not request the
  token from the user in chat.
- Catalog pagination failure: stop review; never present a partial page set as
  “all Experiences.”
- Scope mismatch: stop before verification or launch.
- No verified records: runtime read must remain empty.
- Worker cold-start failure: report it and do not launch a long run.
