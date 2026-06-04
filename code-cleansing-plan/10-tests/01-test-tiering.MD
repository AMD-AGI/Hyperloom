# Test Tiering And Minimal Must-Run Set

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

`inference_optimizer/tests` has ~179 test files. Before deleting or merging tests, classify them so cleanup keeps contract guards and only sheds implementation-detail / retired tests. Per `01-baseline-contracts/03-compatibility-decisions.MD`, compatibility tests (old-session resume, breakdown readers, `extra_sglang_args` alias) stay this round, so they are NOT deletion candidates yet.

This is a cleanup guide. Runtime source of truth stays the tests themselves.

## Tier 0: Contract Tests (must-run on every change)

Fast, few, stable, no heavy subprocess. These guard the runtime contracts touched by every cleanup step.

- `test_action_catalogue.py` — live action set + source-of-truth drift guards.
- `test_phase_state_machine.py` — phase order + phase allowlist.
- `test_agent_roles_and_policy.py` — role permissions / PolicyGate.
- `test_protocol_layer.py` — intent envelope.
- `test_session_layout.py` — session path / artifact layout.
- `test_no_legacy_writer_sites.py` — legacy `extra_sglang_args` writer guard.

Verified: 201 tests pass in ~8s.

## Tier 1: Runtime Flow Tests (run after each cleanup step in the touched area)

Use fakes; assert behavior flow, not private helpers.

- coordinator: `test_coordinator_runtime.py`, `test_resume.py`, `test_resume_deferred_proposals.py`
- explore: `test_explore_executor.py`, `test_explore_grid_limits.py`, `test_grid_runner.py`
- kernel: `test_kernel_request_handlers_units.py`, `test_kernel_shape_validation.py`, `test_kernel_integrate_and_report.py`, `test_profile_and_kernel_handlers.py`
- framework PR: `test_framework_pr_executor.py`, `test_framework_pr_pump_integration.py`, `test_framework_pr_critic_gate.py`
- specialist / dynamic: `test_specialist_integration.py`, `test_specialist_lifecycle.py`, `test_dynamic_action_e2e.py`, `test_integrate_patch_executor.py`
- sweep / close: `test_sweep_phase_auto.py`, `test_conc_sweep.py`, `test_close_phase_sequencer.py`
- breakdown: `test_breakdown_smoke.py`, `test_breakdown_exporter_units.py`, `test_session_breakdown_executor_units.py`

## Tier 2: Package Integration Tests (run after large steps)

- whole `inference_optimizer/tests`,
- `robustness-agent/tests`,
- critic-agent runtime tests (if env allows),
- kernel-agent tool tests.

## Tier 3: Real Smoke (final acceptance)

- no-kernel optimize,
- kernel-enabled smoke,
- resume,
- session_breakdown dump.

## Classification Buckets (for future delete/merge)

- contract: Tier 0 above + envelope/policy/catalogue drift guards. Keep.
- behavior/flow: Tier 1. Keep; merge only same-executor micro-tests.
- compatibility: `test_shared_state_evolution.py`, `test_payload_aliases.py`, `test_back_compat_legacy_field_name.py`, `test_extra_sglang_args_merge.py`, breakdown legacy sections. KEEP this round (resume + breakdown readers retained).
- implementation-detail: micro-tests that only assert private helper names / internal dict ordering / exact prompt sentences. Merge or delete later.
- retired-feature positive tests: none found — retired actions (`setup`/`classify`/`backends`/`params`/`validate_stack`/`select_kernels`) have no positive execution path; only rejection/migration tests exist (`test_retired_cli_flags_hard_fail.py`, `test_drop_scoreboard.py`) and those are contract guards to keep.

## Minimal Must-Run Set (copy-paste)

```bash
python -m pytest \
  inference_optimizer/tests/test_action_catalogue.py \
  inference_optimizer/tests/test_phase_state_machine.py \
  inference_optimizer/tests/test_agent_roles_and_policy.py \
  inference_optimizer/tests/test_protocol_layer.py \
  inference_optimizer/tests/test_session_layout.py \
  inference_optimizer/tests/test_no_legacy_writer_sites.py -q
```

## Delete / Merge Plan (deferred, decision-gated)

Do NOT execute until the noted gate clears:

- Merge same-executor micro-helper tests into the flow test for that executor (low risk; defer to a focused churny commit).
- Replace exact-prompt-text assertions with structural assertions (coordinate with Step 09 prompt work).
- Delete compatibility tests only AFTER the resume / breakdown-reader / alias decisions say the old path is dropped.
- No retired-feature positive tests to delete (none exist).

## Risk Control

- Keep every Tier 0 contract test.
- Keep at least one contract test per external protocol.
- Resolve compatibility decisions before deleting compat tests.
- Name future test files by behavior domain to keep failures locatable.
