# E2E Smoke Scenarios

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

This document defines the end-to-end smoke scenarios that should be used to decide whether cleanup preserved behavior.

The smoke scenarios are not meant to prove every helper still behaves the same. They are meant to prove that the system can still perform its real job.

## Smoke 1: No-Kernel Optimization Session

Purpose:

- Verify the optimizer can run without kernel optimization.
- Protect CLI, baseline, explore, sweep, report, and session breakdown behavior.

Expected flow:

1. Start an optimization session with kernel disabled.
2. Session initializes and writes manifest/state artifacts.
3. Baseline or a mocked baseline path establishes initial measurement.
4. EXPLORE can run at least one candidate path or a mocked equivalent.
5. SWEEP/CLOSE can execute or be deterministically entered.
6. Report and `session_breakdown.json` are written.

Protected contracts:

- CLI launch.
- Session layout.
- Coordinator loop.
- Action executor dispatch.
- Breakdown export.

## Smoke 2: Kernel-Enabled Request Path

Purpose:

- Verify the kernel-owned request path still works.
- Protect request routing, kernel handler selection, shape/path safety gates, and kernel result promotion.

Expected flow:

1. Start a kernel-enabled session or targeted kernel-flow harness.
2. Orchestration request is routed as a kernel request, not ordinary delegate.
3. Known request kinds resolve to handlers.
4. Missing or invalid shape/path evidence is rejected or degraded safely.
5. A mocked successful kernel result can be recorded and surfaced to state/reporting.

Protected contracts:

- Kernel-owned action transport.
- Kernel request handler registry.
- Kernel safety gates.
- Kernel artifact path conventions.

## Smoke 3: Framework PR Phase

Purpose:

- Verify FRAMEWORK_PR remains a Coordinator-owned phase.
- Protect discovery, critic gate, apply/bench/revert semantics.

Expected flow:

1. Coordinator enters FRAMEWORK_PR when enabled.
2. Candidate discovery can return an empty, rejected, or approved batch.
3. Approved candidate is materialized as a framework PR task.
4. Apply-only or mocked benchmark result can produce KEEP/REVERT.
5. Phase exits to EXPLORE without wedging.

Protected contracts:

- Framework-agent discovery bridge.
- Critic pre-gate.
- Framework patch application boundary.
- Phase transition out of FRAMEWORK_PR.

## Smoke 4: Specialist And Dynamic Action Handoff

Purpose:

- Verify LLM sub-agent proposal paths still funnel through the same safe integration channel.

Expected flow:

1. EXPLORE phase can dispatch a specialist or dynamic action.
2. Specialist/dynamic output is captured as a proposal set or synthetic equivalent.
3. Proposal is Critic-gated.
4. Approved patch proposal flows through `integrate_patch`.
5. Result is recorded as KEEP, REVERT, or failure without bypassing the integration gate.

Protected contracts:

- Specialist task shape.
- Dynamic action lifecycle.
- Critic gate.
- `integrate_patch` as the single source patch integration path.

## Smoke 5: Resume Or Explicit Resume Rejection

Purpose:

- Verify interrupted sessions are either supported or rejected explicitly according to the compatibility decision.

Expected flow if resume is kept:

1. Existing session artifacts are loaded.
2. Event replay reconstructs pending state.
3. Deferred proposals or abandoned dynamic actions are reconciled.
4. Run can continue or close cleanly.

Expected flow if resume is dropped for old sessions:

1. Unsupported session schema is detected.
2. Runtime fails with a clear operator-facing reason.
3. No partial state corruption occurs.

Protected contracts:

- Session persistence.
- State migration or explicit rejection.
- Operator-visible failure semantics.

## Smoke 6: Session Breakdown Dump

Purpose:

- Verify dashboard-facing output remains consumable.

Expected flow:

1. Build breakdown from a session directory with manifest/state.
2. Missing optional data does not fabricate values.
3. Headline fields are present when source data exists.
4. Warnings are recorded for unavailable sections.
5. Output is valid JSON and matches current schema expectations.

Protected contracts:

- `session_breakdown.json` schema.
- Missing-data semantics.
- Dashboard headline fields.

## Smoke 7: Robustness And Critic Envelope Validation

Purpose:

- Verify subprocess agent bridges still emit valid Coordinator intents.

Expected flow:

1. Robustness tick request can produce an intent envelope or heartbeat.
2. Critic review bridge can produce a verdict envelope or safe unavailable verdict.
3. Invalid envelopes are rejected.
4. Role-specific restrictions remain enforced.

Protected contracts:

- Intent vocabulary.
- Payload shape.
- Role permission boundaries.
- Subprocess JSON bridge.

## Cleanup Acceptance Rule

A cleanup step can change internal unit behavior if these smoke scenarios still pass or have an explicit accepted change.

If a cleanup breaks one of these smoke scenarios, the change is not a pure cleanup anymore. It must be treated as a behavior change and reviewed separately.
