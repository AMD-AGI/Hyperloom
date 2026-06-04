# Step 03 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`03-converge-action-runtime-surfaces.MD` asks for a clearer runtime action model:

- live actions should be explicit,
- LLM-proposable actions should not be confused with Coordinator-internal actions,
- Kernel-owned actions should have one transport classification,
- retired action names should not remain in live prompt or executor surfaces,
- tests should guard drift rather than act as an independent source of truth.

## Already Implemented Before This Step

The codebase already had several important action-surface safeguards:

- `ActionRegistry` loads live action metadata from `actions/_meta/*.yaml`.
- `test_action_catalogue.py` guards the live action set.
- `phase_state.PHASE_ALLOWED_ACTIONS` gates actions by runtime phase.
- `PolicyGate` rejects invalid role/action/phase combinations.
- `session_paths._runs_actions()` derives run workspace ownership from action metadata.
- CLI executor registration is checked against run workspace ownership.

## Implemented In This Step

### Added shared action surface constants

Added `inference_optimizer/orchestrator/action_surfaces.py`.

This module now owns action-name sets that were previously duplicated across layers:

- `KERNEL_OWNED_ACTIONS`
- `INTERNAL_ONLY_ACTION_NAMES`
- `FRAMEWORK_PR_INTERNAL_ACTION_NAMES`
- `COORDINATOR_INTERNAL_ACTIONS`
- `GRID_INJECTABLE_ACTIONS`

Why:

- PolicyGate, prompt rendering, and CLI wiring should not carry divergent copies of the same ownership/transport classifications.

### Updated PolicyGate to use shared surfaces

`policy.py` now imports kernel-owned and Coordinator-internal action sets from `action_surfaces.py`.

Runtime behavior should remain unchanged; this is a source-of-truth consolidation.

### Updated prompt builder to use shared surfaces

`prompt_builder.py` now imports `KERNEL_OWNED_ACTIONS` and `GRID_INJECTABLE_ACTIONS` from `action_surfaces.py`.

This removes a duplicate kernel-owned action set from prompt rendering.

### Updated CLI no-op kernel-owned registration

`cli.py` now derives `_NOOP_KINDS_KERNEL_ONLY` from the shared `KERNEL_OWNED_ACTIONS`.

This keeps the compatibility no-op registration aligned with the kernel-owned request surface.

### Added drift guard

`test_action_catalogue.py` now verifies:

- PolicyGate uses the shared kernel-owned set.
- Prompt builder uses the shared kernel-owned set.
- CLI no-op registration matches the shared kernel-owned set.
- Prompt builder uses the shared grid-injectable set.
- PolicyGate uses the shared Coordinator-internal set.

### Moved prompt enabled action sets into action surfaces

Moved the full and no-kernel prompt-visible action sets into
`action_surfaces.py`:

- `FULL_ENABLED_ACTIONS`
- `NO_KERNEL_ENABLED_ACTIONS`

`prompt_builder.py` now imports and re-exports those shared sets instead
of owning separate lists.

Why:

- Prompt visibility is part of the action runtime surface.
- Keeping it next to kernel-owned, internal-only, and grid-injectable
  classifications prevents the prompt catalogue from becoming a hidden
  source of truth.

Drift guard added:

- prompt-visible actions must have metadata,
- prompt-visible sets must not include retired action names,
- full mode includes kernel-owned actions,
- no-kernel mode excludes kernel-owned actions.

### Removed stale action-surface comments

Updated comments that described `specialist` as yaml-less and removed stale references to retired `setup`, `classify`, `backends`, `params`, and `validate_stack` as current action semantics.

## Still Not Fully Implemented

### Full action classification is still split

Live action metadata remains in action yaml files, while runtime phase permission remains in `phase_state.py`.

This is acceptable for now:

- metadata describes static action shape,
- phase allowlist describes runtime lifecycle permission.

Future cleanup can add a generated or checked action surface report if needed.

### Prompt enabled action sets remain explicit

`FULL_ENABLED_ACTIONS` and `NO_KERNEL_ENABLED_ACTIONS` now live in
`action_surfaces.py`, but they are still explicit lists.

Reason:

- They encode launch-mode prompt visibility, not only action existence.
- Deriving them automatically requires deeper policy decisions about
  Coordinator-internal actions and no-kernel mode.

Future cleanup:

- Use the shared action surfaces and phase metadata to derive or further
  constrain these lists safely.

### Phase allowlist remains hand-authored

`PHASE_ALLOWED_ACTIONS` remains explicit.

Reason not changed in this step:

- It is a state-machine contract, not just metadata.
- Automatically deriving it from yaml would risk changing runtime behavior.

Future cleanup:

- Add drift tests between phase allowlist, action metadata, and ownership surfaces.

## Next Safe Follow-Ups

- Document LLM-proposable vs Coordinator-internal action surfaces in a concise runtime table.
- Remove retired action wording from prompts and tests where it is not an explicit migration/rejection check.
- Reduce historical comments around action families.
- Decide whether action `family` is still a runtime-relevant concept or display-only metadata.
