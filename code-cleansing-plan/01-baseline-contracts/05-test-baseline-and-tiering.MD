# Test Baseline And Tiering

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

This document defines the cleanup-oriented test baseline.

It does not say every current test must be kept. It says which kinds of tests should protect cleanup, and which kinds can be deleted or merged when they only lock old implementation details.

## Existing Test Surface

The repository already has broad coverage:

- `inference_optimizer/tests`
- `robustness-agent/tests`
- `critic-agent/runtime/tests`
- kernel-agent tool tests

The current test surface mixes:

- live runtime contract tests,
- executor behavior tests,
- subprocess bridge tests,
- historical compatibility tests,
- implementation-detail regression tests,
- prompt wording tests,
- retired feature tests.

The cleanup needs tiering before tests are deleted.

## Tier 0: Must-Keep Contract Tests

Purpose:

- Catch drift in live runtime contracts.
- Run quickly after small cleanup edits.

Conceptual coverage:

- live action catalogue,
- phase order and phase allowlist,
- PolicyGate role/action restrictions,
- intent envelope validation,
- session path layout,
- payload alias guard, if alias is retained.

Current representative tests:

- `test_action_catalogue.py`
- `test_phase_state_machine.py`
- `test_agent_roles_and_policy.py`
- `test_protocol_layer.py`
- `test_session_layout.py`
- `test_no_legacy_writer_sites.py`, while `extra_sglang_args` compatibility remains.

Cleanup stance:

- Keep, but simplify comments and remove historical implementation assertions.

## Tier 1: Runtime Flow Tests

Purpose:

- Verify the Coordinator and action executors still perform the end-to-end flow with fakes or small fixtures.

Conceptual coverage:

- Coordinator tick/run behavior.
- Proposal to Critic to task materialization.
- Explore KEEP/REVERT and stack rebench behavior.
- Framework PR apply/revert path.
- Specialist/dynamic handoff into `integrate_patch`.
- Kernel request handler safety and result recording.
- Close/report/session breakdown path.

Current representative tests:

- `test_coordinator_runtime.py`
- `test_explore_executor.py`
- `test_framework_pr_executor.py`
- `test_framework_pr_pump_integration.py`
- `test_integrate_patch_executor.py`
- `test_kernel_request_handlers_units.py`
- `test_kernel_shape_validation.py`
- `test_dynamic_action_e2e.py`
- `test_close_phase_sequencer.py`
- `test_breakdown_exporter_units.py`

Cleanup stance:

- Keep behavior intent.
- Merge small duplicate tests where possible.
- Avoid locking private helper names.

## Tier 2: Integration Boundary Tests

Purpose:

- Protect subprocess and external-system boundaries.

Conceptual coverage:

- critic-agent bridge,
- robustness-agent bridge,
- framework-agent discovery client,
- kernel-agent handler request/response shape,
- multi-node topology gates if still supported.

Current representative tests:

- `test_robustness_agent_e2e.py`
- robustness-agent role/envelope tests.
- critic runtime tests under `critic-agent/runtime/tests`.
- `test_framework_agent_client.py`
- `test_framework_pr_critic_gate.py`
- `test_multi_node_scripts.py`

Cleanup stance:

- Keep bridge contract tests.
- Delete tests for unimplemented future bridges.
- Avoid testing subprocess temp layout unless it is a contract.

## Tier 3: Compatibility Tests

Purpose:

- Protect compatibility only when compatibility is intentionally retained.

Conceptual coverage:

- old state migration,
- old session breakdown readers,
- legacy payload aliases,
- old session layout,
- retained retired-action rejection.

Current representative tests:

- `test_shared_state_evolution.py`
- `test_payload_aliases.py`
- `test_back_compat_legacy_field_name.py`
- `test_breakdown_smoke.py` legacy sections.
- `test_resume.py`

Cleanup stance:

- Keep only for retained compatibility.
- Delete after compatibility decision says the old path is no longer supported.
- Do not let compatibility tests block canonical state simplification.

## Tier 4: Delete Or Merge Candidates

Tests in this category may still be useful temporarily, but they should not be considered cleanup blockers.

Typical patterns:

- private helper behavior already covered by a higher flow test,
- old prompt wording exact-match tests,
- retired feature positive tests,
- multiple tests around the same tiny parameter normalization helper,
- tests that exist only because a historical PR introduced an intermediate implementation.

Cleanup stance:

- Merge into a behavior test if the behavior matters.
- Delete if the behavior is not externally visible.
- Replace exact prompt text assertions with structural assertions.

## Minimal Cleanup Test Set

For small Step 01 and Step 02 edits, the recommended conceptual minimum is:

- action catalogue,
- phase state,
- PolicyGate and intent parser,
- session path layout,
- payload alias guard, if alias remains,
- breakdown exporter smoke,
- coordinator runtime smoke.

For larger cleanup steps, add:

- explore flow,
- kernel request handler flow,
- framework PR flow,
- specialist/dynamic flow,
- close/report flow.

## What Is Not Required

The cleanup does not require preserving:

- exact helper names,
- exact fixture dict shape,
- exact prompt paragraphs,
- exact comment wording,
- retired action positive behavior,
- historical test file layout.

If a test only protects one of these, it should be deleted or rewritten.

## Output For Later Steps

Later cleanup steps should use this tiering to decide:

- Which tests must remain before deletion.
- Which tests to merge.
- Which tests to delete.
- Which new smoke tests to add after old unit tests are removed.
