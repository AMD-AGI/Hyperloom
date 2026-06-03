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
- Dead/compat: the `cortex_*` methods (`select_kb_for_domain(s)`, `cortex_propose_point`, fallback strategies) + `cortex_kb_constants` import.
  - In production `cli` always passes `cortex_kb=None`, so `cortex_enabled` is False and these paths return empty.
  - Still referenced by Coordinator specialist warmup (guarded by `cortex_enabled`) and by tests that inject a fake v1 client.

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
- specialist prompt: KnowledgePlane PR feed (+ inert cortex subgraph today).

Write (best-effort, never aborts the run):

- terminal task result -> optimization journal.
- KEEP / REVERT -> RecipeKB amend.
- CLOSE -> recipe + session summary.
- warm replay result -> recipe.

No persistent knowledge should be created at proposal time.

## Old KB Path Deletion Candidates (decision pending)

Per `01-baseline-contracts/03-compatibility-decisions.MD`, "drop old Cortex v1 graph KB" needs a product decision. Candidates, once the decision lands:

- `knowledge_plane.py` cortex_* methods: `select_kb_for_domain`, `select_kb_for_domains`, `cortex_propose_point`, `cortex_commit_point` (if present), `_DOMAIN_FALLBACK_STRATEGIES`, and the `cortex_enabled` gate.
- `cortex_kb_constants.py` entries used only by the v1 graph surface (it is also imported by `recipe_kb/` live code, so only the v1-only constants can be removed).
- tests that inject a fake v1 client: `test_kb_pr_assembly.py`, the cortex branches in `test_pr_monitor.py` / `test_knowledge_plane.py`.

Not deletable yet:

- the live RecipeKB path,
- PR feed,
- research scout,
- optimization journal,
- warm start / warm replay.

## Specialist Knowledge Input Bundle (target)

A specialist's knowledge context should be assembled from one clear input set:

- gap,
- domain tags,
- relevant recipe facts,
- relevant PRs,
- source roots,
- research hints.

Today this is split across Coordinator `_warm_specialist_params`, KnowledgePlane, and `specialist_prompt_builder._section_kb_subgraph`. Future cleanup should consolidate the assembly without changing the rendered prompt contract.
