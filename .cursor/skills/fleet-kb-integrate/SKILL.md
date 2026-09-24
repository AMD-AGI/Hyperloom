---
name: fleet-kb-integrate
description: Integrates Fleet KB reads into a Hyperloom-style optimization workflow at the real pre-proposal decision boundary. Use when adapting another workflow to call Fleet KB, map runtime state into read context, inject evidence, or preserve read-to-write exposure traces.
---

# Integrate Fleet KB

Use this checklist when wiring Fleet KB into an optimization workflow.

## 1. Find the real decision boundary

Locate the last point before the decision-making LLM proposes the next measured
change. Read Fleet KB there, not during startup and not after a proposal exists.

The caller supplies:

- `decision`: one stable sentence describing the decision being made.
- `context`: facts already available from the running workflow.

Do not ask the user to author a query, signal list, weights, filters, or search
strategy. The Fleet KB Planner derives those from `decision + context`.

## 2. Map runtime state

Build this pre-proposal context from actual workflow state:

```json
{
  "identity": {},
  "workload": {},
  "objective": {},
  "benchmark_baseline": {},
  "current_best": {},
  "observations": {},
  "recent_results": [],
  "already_tried": []
}
```

Omit unavailable facts instead of guessing them. Sanitize values to JSON. Do
not include secrets, full prompts, a proposed change, or a user-authored query.

Keep these meanings distinct:

- `benchmark_baseline` is the immutable original Recipe measurement used for
  gain accounting.
- `current_best` is the best measured candidate so far.
- Historical KB evidence may seed proposals or `current_best`; it never
  replaces `benchmark_baseline`.

In this repository, use
`hyperloom.inference_optimizer.fleet_kb.FleetKBIntegration` as the reference
adapter.

## 3. Inject only completed evidence

Call Fleet KB before the proposal LLM. If the read completes, insert its
rendered prompt block into that same LLM request. Keep the read fail-open: a
network, Planner, or Executor failure must not stop optimization.

Require one Slack-created `HYPERLOOM_FLEET_KB_SCOPE_ID`. Runtime read searches
only Experiences a human selected for that scope. Do not fall back to the full
Fleet Catalog when the scope is absent or empty.

Never let the workflow reinterpret raw hits or rebuild ranking. The Executor's
rendered result is the evidence contract.

## 4. Preserve exposure

When evidence is actually placed in the decision prompt, carry its `read_id`
and rendered Experience references onto:

1. the proposal,
2. the SBD or implementation artifact,
3. the newly measured Experience.

Do not stamp refs from stale or unconsumed reads. This trace records exposure;
it does not claim the evidence caused the decision.

## 5. Preserve write semantics

Publish the measured Experience through the Fleet client. Reuse the immutable
Experience ID for retries, and configure a durable worker spool so transient
network failures do not lose writes. A new record is cataloged as
`unverified`; publishing must not make it automatically readable.

## 6. Verify the integration

Test:

- context is captured from realistic runtime state;
- no `candidate_change`, `question`, weights, or search strategy is supplied;
- retries inside one decision reuse its read;
- a changed context or later decision reads the current run-scoped selected view;
- an empty or different Run scope cannot see unselected Catalog records;
- completed evidence appears before proposal generation;
- failures continue without evidence;
- only consumed refs reach the measured Experience;
- benchmark gain still uses the original Recipe measurement;
- failed writes spool and retry idempotently.

For deployment environment variables and the central service contract, read
`docs/reference/fleet-kb-demo.md`.
