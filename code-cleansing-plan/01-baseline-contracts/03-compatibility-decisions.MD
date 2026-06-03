# Compatibility Decisions

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

This document lists compatibility paths that must be explicitly kept or dropped before later cleanup steps delete code.

Default stance:

- Keep compatibility only when it protects a real production, dashboard, operator, or historical-session need.
- Drop compatibility when it only protects old unit tests, old implementation details, or stale docs.
- Prefer read-only compatibility over new-write compatibility.

## Decision Matrix

| Area | Current Status | Default Recommendation | Decision Needed |
|---|---|---|---|
| Old session resume | Implemented through state loading, phase inference, event replay, and migration paths | Keep for one cleanup round, then revisit | Yes |
| Old session breakdown readers | Implemented in breakdown collectors with legacy aliases and old artifact layouts | Keep until downstream dashboard/offline needs are confirmed | Yes |
| `extra_sglang_args` read alias | Implemented through compat helpers and guarded by tests | Keep read-only for now; block new writers | Yes |
| Legacy flat session layout | Supported by path resolution and docs | Keep only if operators still use it | Yes |
| Robustness legacy mode | Present in robustness package docs/runtime surfaces | Drop if production uses reactor mode only | Yes |
| Old Cortex v1 graph KB | Residual methods/constants remain around KnowledgePlane and cortex naming | Drop from live runtime if RecipeKB is canonical | Yes |
| Retired action recognition | Some retired names remain in tests, docs, breakdown aliases, or rejection paths | Keep rejection/migration only; delete positive paths | Mostly decided |
| Old kernel-agent artifact layout | Breakdown collectors still scan legacy paths | Keep only for archived session reports | Yes |
| Historical examples | Example scripts exist under `inference_optimizer/examples` | Drop unless used as active smoke docs | Yes |
| Multi-node `.bak` artifacts | Backup files exist | Drop | No, safe candidate |
| Stale coverage omit entries | Omit list includes paths that appear deleted or retired | Drop stale entries after confirmation | No, safe candidate |

## Compatibility Rules

### Rule 1: No new writes to legacy shapes

If a compatibility path remains, it should generally be read-only. New sessions should write canonical fields, canonical action names, and canonical artifact paths.

### Rule 2: Compatibility lives at the boundary

Compatibility should be centralized in one of these places:

- state load migration,
- payload alias helper,
- archived breakdown reader,
- explicit rejection path,
- operator-facing upgrade docs.

Compatibility should not spread across normal executor, prompt, or promotion logic.

### Rule 3: Every retained compatibility path needs a deletion condition

Retained compatibility should answer:

- What old data or old behavior does it protect?
- Who still needs it?
- When can it be removed?
- Is it read-only?

### Rule 4: Tests should match the compatibility decision

If compatibility is kept:

- Keep a small contract test.
- Do not keep broad implementation-detail tests.

If compatibility is dropped:

- Delete positive tests.
- Optionally keep one test that verifies the retired path is rejected, if rejection is user-visible behavior.

## Area Notes

### Old session resume

Why it may matter:

- Long optimization sessions can be interrupted.
- Existing production sessions may need resume during the cleanup window.

Cleanup recommendation:

- Keep during the first cleanup pass.
- Stop adding new historical branches.
- Later decide a minimum supported session schema.

### Old session breakdown readers

Why it may matter:

- Dashboards and offline analysis may still inspect historical runs.
- `session_breakdown.json` is an external contract.

Cleanup recommendation:

- Keep initially.
- Separate archived readers from live producer logic.
- Delete only after downstream consumers confirm old sessions are no longer required.

### `extra_sglang_args` alias

Why it may matter:

- Old payloads and sub-agent envelopes may still contain the old field.

Cleanup recommendation:

- Keep read-only helper temporarily.
- Do not allow new writer sites.
- Use the existing static guard as the enforcement mechanism until the alias is fully removed.

### Old Cortex v1 KB

Why it may matter:

- Some comments and helper surfaces still reference Cortex naming.

Cleanup recommendation:

- If RecipeKB is now canonical, move old graph-KB code out of live runtime.
- Preserve only import/migration utilities if still needed for data conversion.

### Retired actions

Known retired action concepts include:

- `setup`
- `classify`
- `backends`
- `params`
- `validate_stack`
- `select_kernels`

Cleanup recommendation:

- No positive execution path.
- No prompt instruction encouraging them.
- Keep only migration, archived reporting, or explicit rejection behavior if necessary.

## Decisions To Ask Maintainers

Before Step 02 deletes compatibility-heavy code, ask:

1. Do we still need to resume sessions created before the current phase/action model?
2. Do dashboards still need to read old `session_breakdown.json` shapes?
3. Can `extra_sglang_args` be removed entirely after this cleanup branch?
4. Is robustness legacy mode still used by any operator?
5. Is old Cortex v1 graph KB still reachable in production?
6. Are `inference_optimizer/examples` considered user-facing docs or disposable historical demos?

If no answer is available, keep the compatibility boundary documented but do not let it spread further.
