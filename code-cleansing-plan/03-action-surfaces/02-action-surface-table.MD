# Action Surface Table

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

This table summarizes the current action runtime surface after the first Step 03 convergence pass.

It is a cleanup guide, not a second source of truth. Runtime source remains in code:

- metadata: `actions/_meta/*.yaml`,
- phase permission: `phase_state.PHASE_ALLOWED_ACTIONS`,
- ownership/transport surface: `action_surfaces.py`,
- executor wiring: `cli._register_executors`.

## Kernel-Owned Request Actions

These actions are live, but Orchestration should use the Kernel request path rather than ordinary delegate/propose execution:

- `kernel_opt`
- `integrate`
- `deep_kernel_analysis`
- `operator_tuning`
- `vendor_kernel_config`
- `gemm_tuning`

Runtime source:

- `action_surfaces.KERNEL_OWNED_ACTIONS`

Protected behavior:

- PolicyGate rejects the wrong transport.
- Prompt builder renders kernel-owned action guidance.
- CLI no-op task registration stays aligned with the shared set.

## Coordinator-Internal Actions

These actions are live but should be auto-managed by the Coordinator, not proposed directly by the LLM:

- `roofline`
- `profile`
- `replay_warm_recipe`
- `framework_pr`

Runtime source:

- `action_surfaces.INTERNAL_ONLY_ACTION_NAMES`
- `action_surfaces.FRAMEWORK_PR_INTERNAL_ACTION_NAMES`

Protected behavior:

- PolicyGate denies direct LLM proposals.
- Coordinator enqueues these actions from phase hooks or warm-start logic.

## Grid-Injectable Actions

These actions accept LLM-supplied grid candidates in prompt guidance:

- `explore`
- `sweep`

Runtime source:

- `action_surfaces.GRID_INJECTABLE_ACTIONS`

Cleanup note:

- Future work should confirm whether `sweep` should remain LLM grid-injectable or be fully Coordinator-owned in SWEEP.

## LLM-Proposable / Delegate Surfaces

These remain conceptually LLM-facing, subject to phase, policy, and sequence gates:

- `baseline`
- `explore`
- `specialist`
- `integrate_patch`
- `assess_remaining_gaps`
- `dynamic_action`
- `sweep`
- `report`
- `recover`, but only for allowed robustness ownership paths.

Runtime source:

- `action_surfaces.FULL_ENABLED_ACTIONS`
- `action_surfaces.NO_KERNEL_ENABLED_ACTIONS`

Cleanup note:

- These sets are still explicit because they encode launch-mode prompt
  visibility. Drift guards now ensure they only contain live metadata
  actions and do not expose kernel-owned actions in no-kernel mode.

## Retired Runtime Action Names

These should not appear as live enabled actions, live executor registrations, or prompt catalogue entries:

- `setup`
- `classify`
- `backends`
- `params`
- `validate_stack`
- `select_kernels`

Allowed appearances:

- migration reader,
- archived breakdown reader,
- explicit rejection test,
- short upgrade note.

## Follow-Up Checks

Future Step 03 work should verify:

- Every live metadata action is either executor-backed, kernel-owned, or Coordinator-internal.
- Every phase-allowed action is present in metadata or explicitly synthetic.
- Prompt enabled actions do not include retired action names.
- Retired action names are absent from new writers.
