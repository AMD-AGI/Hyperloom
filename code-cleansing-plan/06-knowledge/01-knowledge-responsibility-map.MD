# Knowledge System Responsibility Map

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

Map the current knowledge/priors subsystems so later cleanup can simplify each one without breaking warm start, specialist prompts, or reporting.

This is a cleanup guide, not a second source of truth. Runtime source remains the code.

## Subsystem Roles

### RecipeKB (canonical cross-session KB)

- Source: `inference_optimizer/recipe_kb/` + `recipe_snapshot_constants.py`.
- Wiring: CLI `_bootstrap_cortex_kb(...)` builds the RecipeKB client; passed to `Coordinator(cortex_kb=...)`.
- Reads: warm start (T0 anchor in `cortex_t0.py`).
- Writes: Coordinator `_kb_amend_recipe` / `_fact_write_hook` on terminal task results (KEEP/REVERT) and CLOSE summary.
- Status: canonical. Keep.

### Warm start / warm replay

- Source: `cortex_t0.py` (anchor), Coordinator warm replay hooks, `SharedState.warm_start_*` / `warm_replay_*`.
- Role: seed the run with a prior best config; optionally replay-validate it.
- Status: live end-to-end. Keep, but keep auto-replay simple and degrade gracefully on low confidence.

### KnowledgePlane (read facade)

- Source: `inference_optimizer/orchestrator/knowledge_plane.py`.
- Live role: PR feed warmup + domain repo config for specialist prompts.
- Old Cortex v1 graph traversal / write methods were removed from this
  facade. `from_clients(cortex_kb=...)` remains as an ignored compatibility
  parameter for older callers.

### PR feed

- Source: `pr_monitor.py` + KnowledgePlane PR methods.
- Role: surface recent upstream PRs into specialist prompts.
- Status: live, optional (disabled under `--degraded-pr`). Keep.

### Research scout hints

- Source: `research_hints.py` (+ `competitor_target` helpers), Coordinator scout enqueue/harvest.
- Role: advisory, source-backed priors written to `research_hints.{md,json}` and surfaced to prompts.
- Constraints (enforced): every hint must carry a `source`; sourceless hints are dropped. Read-only; never benchmarks, KEEP/REVERTs, or mutates the optimization stack.
- Status: keep as advisory-only.

### Optimization journal

- Source: `optimization_journal.py`, Coordinator fact-write hook.
- Role: per-session KEEP/REVERT facts; safe source for KB writes.
- Status: keep, decoupled from runtime gating; KB write failure is a soft degrade.

## Canonical KB Read/Write Flow

Read:

- warm start: T0 anchor reads RecipeKB for the workload 5-tuple.
- specialist prompt: KnowledgePlane PR feed, warm-start RecipeKB facts,
  research hints, and source-root context.

Write (best-effort, never aborts the run):

- terminal task result -> optimization journal.
- KEEP / REVERT -> RecipeKB amend.
- CLOSE -> recipe + session summary.
- warm replay result -> recipe.

No persistent knowledge should be created at proposal time.

## Old KB Path Deletion

Removed in the high-risk cleanup pass:

- `knowledge_plane.py` v1 graph traversal methods:
  `select_kb_for_domain`, `select_kb_for_domains`, and fallback strategies.
- `knowledge_plane.py` v1 graph write helpers:
  `cortex_propose_point`, `mint_pr_node`, `pr_node_canonical_id`.
- Fake v1 client tests for the removed surface.

Still retained:

- the live RecipeKB path,
- PR feed,
- research scout,
- optimization journal,
- warm start / warm replay.
- `cortex_kb_constants.py` remains because live `recipe_kb/` code imports
  shared constants from it.

## Specialist Knowledge Input Bundle (target)

A specialist's knowledge context should be assembled from one clear input set:

- gap,
- domain tags,
- relevant recipe facts,
- relevant PRs,
- source roots,
- research hints.

Today this is split across Coordinator `_warm_specialist_params`, KnowledgePlane, and `specialist_prompt_builder._section_kb_subgraph`. Future cleanup should consolidate the assembly without changing the rendered prompt contract.
