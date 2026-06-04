# Step 10 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`10-restructure-tests-around-e2e-behavior.MD` asks to classify tests, keep contract guards, drop retired/implementation-detail tests, merge duplicates, and define a minimal must-run set. The plan explicitly says "classify first, then delete".

## Findings

- ~179 test files under `inference_optimizer/tests`.
- A Tier-0 contract subset (catalogue / phase / policy / protocol / session layout / legacy-writer guard) runs 201 tests in ~8s and is green.
- No retired-feature positive tests exist: retired actions have no positive execution path; only rejection/migration guards remain, which are contract tests to keep.
- Compatibility tests (resume, payload alias, breakdown legacy) are gated KEEP this round per the Step 01 compatibility decisions.

## Implemented In This Step

### Test tiering + minimal must-run set artifact

Added `code-cleansing-plan/10-tests/01-test-tiering.MD`:

- Tier 0 contract set (verified green),
- Tier 1 runtime flow tests by domain,
- Tier 2 package integration,
- Tier 3 real smoke,
- classification buckets,
- copy-paste minimal must-run command,
- deferred (decision-gated) delete/merge plan.

## Why No Test Deletion This Step

- Retired-feature positive tests: none to delete.
- Compatibility tests: gated KEEP this round.
- Merging same-executor micro-tests and replacing exact-prompt assertions are high-churn and better done as focused, separately reviewable commits; doing them now would mix classification with large diffs and risk hiding protocol drift.

So this step delivers the classification + minimal set (required outputs) and defers the churny deletions to gated follow-ups, consistent with the safe, single-purpose commit cadence used in earlier steps.

## Verification

- Tier-0 minimal set: 201 passed in ~8s.
- Lint: no errors (docs only).

## Next Safe Follow-Ups

- After compatibility decisions land: delete the now-unsupported compat tests with their code.
- Merge same-executor micro-helper tests into the executor flow test (one executor per commit).
- Replace exact-prompt-text assertions with structural assertions alongside Step 09 prompt compression.
