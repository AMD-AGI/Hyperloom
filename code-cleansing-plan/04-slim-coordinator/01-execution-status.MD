# Step 04 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`04-slim-coordinator.MD` asks for two ordered phases:

1. De-duplicate obvious repeated logic inside `coordinator.py`.
2. Split responsibility domains into separate modules.

The plan is explicit: de-dup first, split later. Splitting a 12k-line file is high risk and should only happen after duplication is removed and each move is single-purpose.

This step covers the first low-risk de-duplication batch.

## Implemented In This Step (De-dup Batch 1)

### Removed duplicate backend error streak initialization

`Coordinator.__init__` initialized the backend error streak tracking twice with identical blocks (`_backend_error_streak`, `_backend_error_alarm_armed`, and `_backend_error_streak_threshold`). The second block simply overwrote the first.

Action:

- Removed the duplicate block. Behavior is unchanged because both blocks were identical.

### Extracted shared resume rebuild logic

`tick()` and `run()` both contained the same resume-rebuild sequence:

- `replay_for_resume()`,
- drain `_proposals_awaiting_roofline`,
- `_resume_abandon_dynamic_actions()`.

Action:

- Added `Coordinator._replay_resume_if_needed()` and called it from both `tick()` and `run()`.
- Behavior is unchanged: the guard condition, call order, and side effects match the previous inline blocks.

### Extracted shared FRAMEWORK_PR pump wrapper

`tick()` and `run()` both wrapped `_pump_framework_pr_phase()` in the same
best-effort `try/except` block with only the caller name differing.

Action:

- Added `Coordinator._pump_framework_pr_phase_safely(caller=...)`.
- Replaced both inline wrappers with calls to the shared helper.
- Behavior is unchanged: the pump still runs in the same point in each
  loop and still logs and suppresses defensive failures.

### Centralized roofline pending gate checks

The pending auto-analysis gate appeared separately in propose,
materialize, and delegate paths.

Action:

- Added `Coordinator._roofline_denial_for_action(action_name)`.
- Added `Coordinator._defer_approved_proposal_for_roofline(...)` for the
  materialize-only deferred queue / resume observation logic.
- Replaced the three inline roofline-gate checks with the shared helper.

Behavior is unchanged:

- propose/delegate still record `PolicyDenied`,
- already-approved proposals are still deferred instead of dropped,
- resume can still rebuild the deferred queue from the same observation.

## Verification

- Lint: no errors on `coordinator.py`.
- Tests: `test_coordinator_runtime.py`, `test_resume.py`, `test_resume_deferred_proposals.py`, `test_dynamic_action_resume.py` all pass (112 passed).
- Follow-up FRAMEWORK_PR pump verification: `test_coordinator_runtime.py`,
  `test_framework_pr_pump_integration.py`, `test_framework_pr_executor.py`
  pass (98 passed).
- Follow-up roofline gate verification: coordinator, deferred-proposal,
  roofline, and prompt-visibility tests pass (244 passed, 1 skipped).

## Not Done Yet (Future Step 04 Batches)

The plan lists more de-dup and the larger responsibility split. These are intentionally deferred to keep each change single-purpose and low-risk:

De-dup still pending:

- time budget prompt block duplication in `_compose_prompt`.
- `extra_server_args` compat read entry consolidation.
- shared terminal task-result promotion hook.

Responsibility split still pending (each is a separate future commit):

- phase hooks module,
- framework PR flow module,
- specialist flow module,
- dynamic action flow module,
- kernel flow module,
- fact write flow module.

## Risk Control Applied

- Only behavior-preserving de-dup in this batch.
- No file split yet.
- Public loop call order preserved.
- Resume path verified by the resume test suite.
