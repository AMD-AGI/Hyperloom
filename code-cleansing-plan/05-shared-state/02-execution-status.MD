# Step 05 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`05-clean-shared-state-and-compatibility.MD` asks to:

- classify `SharedState` fields,
- keep new-session writes canonical,
- centralize compatibility migration,
- record deletion conditions for retained compat,
- reduce prompt/mutation surface,
- eventually remove dead fields.

## Compatibility Gate

Per `01-baseline-contracts/03-compatibility-decisions.MD`:

- Old session resume: KEEP this round.
- `extra_sglang_args` read alias: KEEP read-only.

So this step does not delete migration code or compat fields. Deleting them now would break old-session resume, which is still supported this round.

## Implemented In This Step

### Field classification artifact

Added `code-cleansing-plan/05-shared-state/01-field-classification.MD`:

- core runtime fields,
- action ledger fields,
- contract/export fields,
- knowledge/warm-start fields,
- compat fields (read-only this round),
- centralized migration/drop sites with deletion conditions,
- dead-field candidate note,
- canonical new-session schema intent.

### Deletion-condition comments on centralized compat sites

Added concise deletion-condition comments (no behavior change) to:

- `_PHASE4_LEGACY_KEY_RENAMES` / `_migrate_legacy_extra_sglang_args_keys` (the `extra_sglang_args` rename walker).
- `from_dict._legacy_drop_fields` (retired scoreboard + `last_select_kernels` drop list).

This satisfies the plan's "every retained compatibility path needs a deletion condition" rule.

### Removed a test-only legacy writer helper

Removed `SharedState.push_backend_winners_round()`.

Why:

- No live runtime code called it.
- It wrote the legacy `backend_winners_history` field even though
  `explore_search` is the canonical merged ledger.
- Existing prompt rendering still supports `backend_winners_history` as a
  read-only old-session field.

Test adjustment:

- `test_prompt_renders_gain_values.py` now constructs legacy
  `backend_winners_history` directly when testing old-session rendering,
  instead of relying on a production writer helper.

### Centralized legacy plateau proxy writes

`params_no_promote_streak` is retained this round because the legacy M2
plateau proxy still supports old-session resume when `explore_search`
signals are empty.

Action:

- Added `SharedState.reset_explore_plateau_proxy()`.
- Extended `SharedState.note_explore_outcome(promoted=...)` so explore
  KEEP/no-KEEP outcomes update the retained proxy in one place.
- Removed direct Coordinator increment/reset writes from explore result
  promotion; steward continuation now resets via the helper.

## Verification

- Lint: no errors on `shared_state.py`.
- Tests: `test_shared_state_*`, `test_resume`, `test_payload_aliases`, `test_no_legacy_writer_sites` pass.
- Follow-up compat cleanup verification: `test_prompt_renders_gain_values.py`,
  `test_shared_state_evolution.py`, `test_no_legacy_writer_sites.py` pass
  (64 passed).
- Follow-up plateau proxy verification: phase-state, assess-remaining-gaps,
  gaps, and shared-state evolution tests pass (152 passed).

## Not Done Yet

Deferred to keep this batch low-risk and resume-safe:

- removing any field (needs a read/write/export map and a stop-new-writes-first pass),
- removing migration code (blocked by the keep-resume decision this round),
- prompt surface trimming (separate change; touches `to_prompt_summary` rendering and prompt tests),
- mutation surface trimming (touches PolicyGate CORE_STATE_FIELDS coupling).

## Next Safe Follow-Ups

- Build the field read/write/export map.
- Stop any remaining new writes to compat fields, then remove old reads.
- Trim prompt rendering of historical/debug fields once a prompt-contract test exists.
