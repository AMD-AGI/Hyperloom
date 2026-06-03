# Live Contracts For Cleanup

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

This document defines the live contracts that later cleanup work must protect.

If a code path protects one of these contracts, it should not be deleted until an equivalent simpler path exists. If a code path does not protect one of these contracts, it is eligible for deletion, merging, or rewriting.

## 1. CLI Contract

Protected behavior:

- The package exposes the `inference_optimizer` console script.
- `inference_optimizer optimize` is the primary runtime entry.
- The optimizer can launch a fresh session and resume a session.
- CLI-provided workload settings and key environment variables flow into the runtime.
- Production preflight failures are surfaced clearly enough for an operator or agent to act.

Contract owner:

- `inference_optimizer/cli.py`
- `inference_optimizer/README.md`
- `inference_optimizer/SKILL.md`

Cleanup guidance:

- CLI internals may be reorganized.
- Helper functions and argument parsing internals may change.
- The public launch path and operator-visible behavior must remain stable unless explicitly changed.

## 2. Session Artifact Contract

Protected artifacts:

- `manifest.json`
- `state.json`
- `storage/coordinator.db`
- `runs/<action>/<task_id>/`
- `kernel-agent/runs/<session_id>/`
- `kernel-agent-workspace/<kernel_id>/`
- `reports/`
- `session_breakdown.json`

Contract owners:

- `inference_optimizer/paths.py`
- `inference_optimizer/session_paths.py`
- `inference_optimizer/manifest.py`
- `inference_optimizer/orchestrator/shared_state.py`
- `inference_optimizer/storage/schema.py`

Cleanup guidance:

- Path helper internals may be simplified.
- New sessions should write canonical paths only.
- Historical path readers should be kept only if old-session support is intentionally retained.

## 3. Action Catalogue Contract

Protected live actions:

- `target_analysis`
- `baseline`
- `replay_warm_recipe`
- `roofline`
- `profile`
- `explore`
- `integrate_patch`
- `framework_pr`
- `sweep`
- `conc_sweep`
- `report`
- `session_breakdown`
- `specialist`
- `assess_remaining_gaps`
- `dynamic_action`
- `kernel_opt`
- `integrate`
- `deep_kernel_analysis`
- `operator_tuning`
- `vendor_kernel_config`
- `gemm_tuning`
- `recover`

Contract owners:

- `inference_optimizer/actions/_meta/*.yaml`
- `inference_optimizer/orchestrator/action_registry.py`
- `inference_optimizer/tests/test_action_catalogue.py`
- `inference_optimizer/cli.py`
- `inference_optimizer/session_paths.py`

Cleanup guidance:

- Retired action names should not remain in live prompt or executor surfaces.
- Action metadata, executor wiring, phase allowlist, and run workspace rules must stay aligned.
- Internal action implementation can change as long as live action behavior remains.

## 4. Phase Contract

Protected phase order:

```text
PRELUDE -> FRAMEWORK_PR -> EXPLORE -> KERNEL -> SWEEP -> CLOSE
```

Protected semantics:

- `PRELUDE`: target analysis, baseline, initial profile/roofline.
- `FRAMEWORK_PR`: Coordinator-owned upstream PR exploration, if enabled.
- `EXPLORE`: config search, specialists, dynamic actions, patch integration.
- `KERNEL`: kernel-owned request flow.
- `SWEEP`: final sweep and concurrency comparison.
- `CLOSE`: report and session breakdown.

Contract owners:

- `inference_optimizer/orchestrator/phase_state.py`
- `inference_optimizer/orchestrator/policy.py`
- `inference_optimizer/orchestrator/coordinator.py`

Cleanup guidance:

- Phase transition internals may be simplified.
- Phase names, monotonic order, and phase action compatibility are runtime contracts.
- Coordinator-internal auto-enqueue behavior should be preserved unless explicitly redesigned.

## 5. Intent Envelope Contract

Protected vocabulary:

- `send_message`
- `delegate`
- `propose_action`
- `update_state`
- `update_persona`
- `ask_question`
- `answer`
- `alert`
- `request`
- `response`
- `review_verdict`
- `kill_task`
- `force_dispatch`
- `prune_branch`
- `escalate_strategy_change`
- `specialist_done`

Contract owners:

- `inference_optimizer/orchestrator/intent_parser.py`
- `inference_optimizer/orchestrator/policy.py`
- `robustness-agent/src/robustness_agent/role/envelope.py`
- `critic-agent` runtime bridge

Cleanup guidance:

- Intent parsing and validation may be reorganized.
- Vocabulary and payload shape must not drift across subprocess boundaries.
- Mirror packages should keep contract tests if the mirror remains.

## 6. Agent Role Contract

Protected roles:

- Orchestration: proposes/delegates/request kernel work.
- Kernel: responder path for kernel-owned work.
- Critic: reviews proposals.
- Robustness: monitors health and emits recovery/scheduling intents.

Contract owners:

- `inference_optimizer/orchestrator/agent_role.py`
- `inference_optimizer/orchestrator/policy.py`
- `inference_optimizer/orchestrator/backends/*`

Cleanup guidance:

- Backend implementation details may change.
- Role authority boundaries should remain explicit.
- Robustness-only and Critic-only permissions must stay protected.

## 7. Kernel Bridge Contract

Protected request kinds:

- `trace_analyze`
- `run_gemm_tuning`
- `run_optimization`
- `integrate`
- `apply_patch`

Contract owners:

- `inference_optimizer/orchestrator/kernel_request_handlers.py`
- `kernel-agent/tools/*`
- `inference_optimizer/orchestrator/policy.py`

Cleanup guidance:

- Handler internals may be simplified.
- Shape/path/safety gates must remain.
- Removed kernel request kinds should not remain as live prompt instructions.

## 8. Framework PR Contract

Protected behavior:

- Framework PR discovery can run as a Coordinator-owned phase.
- Candidate diffs can be Critic-gated.
- Approved candidates can be applied, benchmarked, kept, or reverted.

Contract owners:

- `inference_optimizer/orchestrator/framework_agent_client.py`
- `inference_optimizer/orchestrator/action_executors/framework_pr.py`
- `framework-agent` CLI

Cleanup guidance:

- Only the actually wired framework-agent bridge is protected.
- Unimplemented future framework-agent role descriptions are not live contracts.

## 9. Critic And Robustness Bridge Contract

Protected behavior:

- Critic subprocess bridge can prepare and commit review results.
- Robustness subprocess bridge can tick and emit a valid intent envelope.
- Invalid envelopes are rejected before mutating runtime state.

Contract owners:

- `inference_optimizer/orchestrator/backends/critic_agent.py`
- `inference_optimizer/orchestrator/backends/robustness_agent.py`
- `critic-agent`
- `robustness-agent`

Cleanup guidance:

- Transport workdir details may change if the JSON contract stays stable.
- Contract tests are more important than private subprocess helper tests.

## 10. Breakdown Contract

Protected behavior:

- A completed or interrupted session can produce `session_breakdown.json`.
- Missing data is represented as missing, empty, or null, not fabricated.
- Downstream consumers can read headline fields: session, workload, baseline, final result, phase/action timeline, capability summary, kernel lifecycle, sweep, warnings.

Contract owners:

- `inference_optimizer/breakdown/schema.py`
- `inference_optimizer/breakdown/exporter.py`
- `docs/INTEGRATION_SESSION_BREAKDOWN.md`

Cleanup guidance:

- Schema documentation should be aligned with producer code.
- Archived compatibility readers should be separated from live producer logic.
- Additive optional fields are acceptable; breaking field removal needs downstream decision.

## Non-Contracts

The following are not protected unless separately tied to a live contract:

- Private helper names.
- Historical PR/GAP/Nxx comments.
- Exact unit test fixture structure.
- Retired action positive execution paths.
- Old demo scripts.
- Backup files.
- Internal dict ordering not consumed outside the module.
- Long prompt wording, except required protocol instructions.
