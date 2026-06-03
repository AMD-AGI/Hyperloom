# Step 01 Implementation Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`01-establish-baseline-and-contracts.MD` defines a preparation step, not a deletion step.

Its intent is to establish the boundaries that make later aggressive deletion safe:

- Preserve end-to-end optimizer behavior.
- Protect external contracts.
- Identify which internal details are not contracts.
- Decide which compatibility paths are worth keeping.
- Build a minimum test baseline.
- Produce an initial dead-code candidate list.

This step is successful when future deletion can answer: "Does this path protect a real live contract, or is it only historical/internal complexity?"

## Already Implemented In The Codebase

### Live action catalogue is implemented

The codebase already has a live action catalogue and drift guards.

Evidence:

- `inference_optimizer/actions/_meta/*.yaml` contains the action metadata.
- `inference_optimizer/tests/test_action_catalogue.py` defines the expected live action set.
- `inference_optimizer/orchestrator/action_registry.py` loads and validates action metadata.
- `inference_optimizer/session_paths.py` derives run workspace actions from action metadata.
- `inference_optimizer/cli.py` registers executor implementations for live runtime tasks.

Conclusion:

The action contract exists. Step 01 does not need to invent it; it needs to document it as a cleanup boundary.

### Phase pipeline is implemented

The phase model is implemented as a closed runtime contract.

Evidence:

- `inference_optimizer/orchestrator/phase_state.py` defines the six-phase chain.
- `PHASE_ALLOWED_ACTIONS` defines which actions are allowed in each phase.
- `PolicyGate` uses phase compatibility to reject invalid action proposals.

Conclusion:

The phase contract exists and should be treated as a protected runtime contract during cleanup.

### Intent envelope is implemented

Agent communication is already centralized around a structured intent contract.

Evidence:

- `inference_optimizer/orchestrator/intent_parser.py` defines `IntentType`, required payload fields, and envelope validation.
- `robustness-agent/src/robustness_agent/role/envelope.py` mirrors this contract for the robustness subprocess runtime.
- `critic-agent` is bridged through a subprocess prepare/commit envelope flow.

Conclusion:

Intent vocabulary and payload shape are live external contracts, not implementation details.

### Session artifact paths are implemented

Session file and directory layout has centralized helpers.

Evidence:

- `inference_optimizer/paths.py` resolves workspace and session roots.
- `inference_optimizer/session_paths.py` defines paths inside the session directory.
- `inference_optimizer/storage/schema.py` defines the SQLite state database schema.
- `inference_optimizer/manifest.py` writes `manifest.json`.
- `inference_optimizer/orchestrator/shared_state.py` owns `state.json`.

Conclusion:

Session layout is a live artifact contract. Later cleanup may simplify helpers, but it must not accidentally break the externally consumed layout.

### Breakdown contract is implemented

The external dashboard/reporting contract exists.

Evidence:

- `inference_optimizer/breakdown/schema.py` defines `session_breakdown.json`.
- `inference_optimizer/breakdown/exporter.py` builds and writes the breakdown.
- `docs/INTEGRATION_SESSION_BREAKDOWN.md` documents downstream consumption, though it appears to lag current schema details.

Conclusion:

The breakdown contract is live and protected. The documentation needs alignment, but producer fields should not be removed without downstream confirmation.

### Test baseline exists, but is not yet tiered for cleanup

There are many tests covering contracts, executor behavior, compatibility, and implementation details.

Evidence:

- `pyproject.toml` sets testpaths across `inference_optimizer`, `robustness-agent`, and `critic-agent`.
- `inference_optimizer/tests` contains a large number of focused regression tests.
- Some tests are clearly contract tests, while many others lock implementation details or historical behavior.

Conclusion:

The raw test surface exists. The missing piece is a cleanup-oriented test tiering plan.

## Not Yet Implemented As Step 01 Artifacts

### Live contracts were not previously summarized in one cleanup artifact

The code contains the contracts, but cleanup work needs a single document that tells future deletion work what is protected.

Status:

- Implemented by this Step 01 execution in `02-live-contracts.MD`.

### Compatibility decisions are not yet product-final

Several compatibility paths exist, but whether to keep them long term is not yet decided.

Examples:

- Old session resume.
- Old session breakdown readers.
- `extra_sglang_args` read alias.
- Legacy flat session layout.
- Robustness legacy mode.
- Old Cortex v1 KB paths.

Status:

- Captured as decision items in `03-compatibility-decisions.MD`.
- These should be decided before destructive deletion.

### E2E smoke scenarios were not previously defined as cleanup gates

The code has tests and CLI entry points, but there was no concise cleanup smoke matrix.

Status:

- Defined in `04-e2e-smoke-scenarios.MD`.

### Test tiers were not previously separated by cleanup value

The current test suite mixes contract tests, behavior tests, compatibility tests, and implementation-detail tests.

Status:

- Initial tiering is defined in `05-test-baseline-and-tiering.MD`.

### Dead-code candidates were not previously tracked with Step 01 evidence

Some candidates are obvious from the codebase, but they need to be listed with caution notes before Step 02.

Status:

- Initial list is defined in `06-dead-code-initial-candidates.MD`.

## Immediate Next Step

With Step 01 artifacts in place, the next cleanup step can begin with low-risk deletion candidates only:

- backup files,
- stale coverage omit entries,
- stale documentation claims,
- retired feature references,
- unimplemented feature scaffold descriptions.

Do not start deleting compatibility readers or state migrations until the compatibility decisions are resolved.
