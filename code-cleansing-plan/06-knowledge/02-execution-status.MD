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

The first Step 06 pass deferred old Cortex v1 graph deletion because the
compatibility decision was still open. The high-risk cleanup pass now
treats that decision as resolved for runtime code: the old KnowledgePlane
v1 graph surface is removed, while live RecipeKB and PR feed paths remain.

## Implemented In This Step

### Knowledge responsibility map

Added `code-cleansing-plan/06-knowledge/01-knowledge-responsibility-map.MD`:

- subsystem roles (RecipeKB, warm start, KnowledgePlane, PR feed, research scout, journal),
- canonical KB read/write flow,
- old KB path deletion candidates (decision pending),
- specialist knowledge input bundle target.

### Strengthened the old-KB deletion-condition note

`knowledge_plane.py` header now states explicitly that the cortex_* surface is inert in production and records the deletion condition: remove once the "drop old Cortex v1 graph KB" decision lands; canonical knowledge is RecipeKB.

### Removed old Cortex v1 graph surface from KnowledgePlane

Removed from `knowledge_plane.py`:

- `select_kb_for_domain`
- `select_kb_for_domains`
- `_DOMAIN_FALLBACK_STRATEGIES`
- `_domain_fallback_query`
- `cortex_propose_point`
- `mint_pr_node`
- `pr_node_canonical_id`

Updated:

- `Coordinator._warm_specialist_params` no longer attempts to fill
  `kb_subgraph` from KnowledgePlane.
- Specialist prompt wording no longer describes Section 4 as a Cortex KB
  subgraph; it is now optional advisory KB context.
- `actions/specialist.md` now describes RecipeKB warm-start facts,
  research hints, PR feed, and source roots rather than a Cortex KB
  subgraph.
- `test_kb_pr_assembly.py` was deleted because it only tested the removed
  fake v1 graph path.
- `test_pr_monitor.py` no longer tests Cortex mint/propose helpers.

Kept:

- RecipeKB warm start / writeback.
- PR feed warmup.
- Research scout hints.
- Optimization journal.
- `cortex_kb_constants.py`, because live `recipe_kb/` code still imports
  shared constants from it.

## Verification

- Lint: no errors on `knowledge_plane.py`.
- Original first-pass tests: `test_knowledge_plane`, `test_kb_pr_assembly`,
  `test_pr_monitor`, `test_research_scout`, `test_recipe_kb_dispatcher`,
  `test_local_recipe_store`, `test_cortex_t0_anchor` pass.
- Follow-up old-Cortex deletion verification: `test_knowledge_plane`,
  `test_pr_monitor`, `test_research_scout`, `test_recipe_kb_dispatcher`,
  `test_local_recipe_store`, `test_cortex_t0_anchor` pass (136 passed).
- Full knowledge + specialist prompt verification: knowledge-plane,
  PR-monitor, research-scout, RecipeKB, cortex T0, specialist lifecycle,
  specialist prompt, and policy-gate tests pass (277 passed).

## Not Done Yet

Still deferred:

- consolidating the specialist knowledge input bundle assembly,
- deeper renaming of historical `cortex_kb` variables that now hold
  `RecipeKB` objects.

## Next Safe Follow-Ups

- Consolidate specialist knowledge assembly into one bundle builder without changing the rendered prompt.
