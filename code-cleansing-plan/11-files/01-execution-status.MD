# Step 11 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical
complexity of the codebase. File moves should express real ownership
boundaries and must not hide behavior changes.

## Design Intent Understood

`11-reorganize-files-and-imports.MD` asks to reorganize files only after
dead code, compatibility paths, comments, and tests have been cleaned
enough that moves are reviewable. Each migration should cover one
responsibility domain and avoid long-lived shims.

## Implemented In This Step

### Created a protocol package for action surfaces

Added:

- `inference_optimizer/protocol/__init__.py`
- `inference_optimizer/protocol/action_surfaces.py`

Moved the pure action-surface constants out of
`inference_optimizer/orchestrator/action_surfaces.py`:

- `KERNEL_OWNED_ACTIONS`
- `INTERNAL_ONLY_ACTION_NAMES`
- `FRAMEWORK_PR_INTERNAL_ACTION_NAMES`
- `COORDINATOR_INTERNAL_ACTIONS`
- `PHASE_ALLOWLIST_BYPASS_ACTIONS`
- `GRID_INJECTABLE_ACTIONS`
- `FULL_ENABLED_ACTIONS`
- `NO_KERNEL_ENABLED_ACTIONS`

Updated imports in:

- `inference_optimizer/cli.py`
- `inference_optimizer/orchestrator/policy.py`
- `inference_optimizer/orchestrator/system_prompts/prompt_builder.py`
- `inference_optimizer/tests/test_action_catalogue.py`

No old-path shim was kept because this is an internal module and all
in-repo references were updated.

## Verification

- Import search: no `inference_optimizer.orchestrator.action_surfaces`
  references remain.
- Tests: action catalogue, prompt builder, prompt visibility, and CLI
  preflight tests pass (84 passed).

## Next Safe Follow-Ups

- Move more pure protocol modules only when their import graph is small
  and acyclic.
- Defer large `Coordinator` responsibility moves until the remaining
  coordinator flow split is complete.
