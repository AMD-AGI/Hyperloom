# Step 06 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`06-simplify-knowledge-and-priors.MD` asks to:

- keep only the knowledge systems the live loop actually uses,
- stop the old KB protocol from polluting the main flow,
- keep research priors advisory-only,
- ensure KB write failure never aborts the run,
- isolate or remove old Cortex v1 graph KB residuals.

## Findings

- RecipeKB is the canonical cross-session KB: CLI builds it and passes it to the Coordinator; warm start reads it; KEEP/REVERT/CLOSE write it.
- KnowledgePlane's live role is the PR feed for specialist prompts.
- The KnowledgePlane `cortex_*` graph methods are inert in production: CLI always constructs `KnowledgePlane.from_clients(cortex_kb=None)`, so `cortex_enabled` is False and the specialist KB warmup short-circuits to an empty subgraph.
- The `cortex_*` methods are still reachable (Coordinator warmup, guarded by `cortex_enabled`) and still exercised by tests that inject a fake v1 client.
- `cortex_kb_constants.py` is imported by live `recipe_kb/` code, not only by the v1 surface.

## Compatibility Gate

Per `01-baseline-contracts/03-compatibility-decisions.MD`, "drop old Cortex v1 graph KB" is marked Decision Needed: Yes.

Therefore this step does NOT delete the cortex_* methods, `cortex_kb_constants`, or their tests. Deleting now would also need to touch live recipe_kb imports and active tests.

## Implemented In This Step

### Knowledge responsibility map

Added `code-cleansing-plan/06-knowledge/01-knowledge-responsibility-map.MD`:

- subsystem roles (RecipeKB, warm start, KnowledgePlane, PR feed, research scout, journal),
- canonical KB read/write flow,
- old KB path deletion candidates (decision pending),
- specialist knowledge input bundle target.

### Strengthened the old-KB deletion-condition note

`knowledge_plane.py` header now states explicitly that the cortex_* surface is inert in production and records the deletion condition: remove once the "drop old Cortex v1 graph KB" decision lands; canonical knowledge is RecipeKB.

## Verification

- Lint: no errors on `knowledge_plane.py`.
- Tests: `test_knowledge_plane`, `test_kb_pr_assembly`, `test_pr_monitor`, `test_research_scout`, `test_recipe_kb_dispatcher`, `test_local_recipe_store`, `test_cortex_t0_anchor` pass.

## Not Done Yet

Deferred pending the Cortex v1 decision and to stay low-risk:

- removing the cortex_* methods + v1-only constants,
- removing the fake-v1-client tests,
- consolidating the specialist knowledge input bundle assembly,
- trimming the prompt KB block wording.

## Next Safe Follow-Ups

- Get the product decision on dropping old Cortex v1 graph KB.
- If dropped: remove cortex_* methods, v1-only constants, and their tests; keep RecipeBK + PR feed + research scout + journal.
- Consolidate specialist knowledge assembly into one bundle builder without changing the rendered prompt.
