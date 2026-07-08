# Session Breakdown v1.1 — Design Rationale

This document explains **why** we extended `session_breakdown.json`, how responsibilities
are split across the pipeline, and what “complete” export means in practice.

Schema target: `hyperloom.session_breakdown.v1.1` (additive over v1).

## Problem

v1 breakdown is the contract for dashboards, stats-service, and offline analysis, but
collectors intentionally **summarize** search and kernel activity:

| Area | On disk / in `state.json` | v1 export gap |
|------|---------------------------|---------------|
| params/backends rounds | full `tested`, `rejected`, `last_round` | ledger capped; no per-variant reject reason in a dedicated view |
| promotion rationale | `*_attempts[].extras` (gain, best name) | no structured `promotion_rule` |
| GEAK/OOB | `optimization_attempts.jsonl`, result JSON | paths only; proposal/verification not inline |
| profiling | traces, TraceLens status JSON, kernel CSV | trace paths in `telemetry`; parsed kernels not exported |
| baseline | server.log per attempt | final `baseline.invocation` only |

Downstream consumers (KB ingest, post-mortems, arbor-compare) need **per-round
decision detail** without re-walking `state.json` + `runs/` by hand.

## Design principles

### 1. Additive compatibility

- Bump `schema_version` once to `hyperloom.session_breakdown.v1.1`.
- Add top-level sections; do not rename or remove v1 keys.
- v1 consumers ignore unknown keys; v1.1 consumers read new sections as canonical detail.

### 2. Data already exists — export, don’t re-run

Most v1.1 fields are **collector-only**: read `state.json`, search ledgers, and artifact
trees under `session_dir`. We avoid re-executing benchmarks or calling LLMs in the JSON path.

Exception: **Phase 2** adds structured promotion fields at write time in the Coordinator so
`decision_journal[].round_decision` can carry machine-readable rules (`single_shot`,
`cross_round_consistent`, `accuracy_blocked`, `below_threshold`).

### 3. Deterministic JSON, narrative markdown

- `build()` / `write_breakdown_json()` — numbers and paths only; failures → `warnings[]`.
- `render_session_report()` — markdown report; LLM optional; deterministic blocks preserved.

### 4. Size control via `detail_level`

| Mode | Behavior |
|------|----------|
| `standard` | `decision_journal`: all promoted/rejected + top 30 tested by \|gain\|; report caps rounds |
| `verbose` | all variants; TraceLens CLI log tail (40 lines max) in kernel_profiling report |

CLI: `dump_session_breakdown --detail-level standard|verbose`.

## Architecture

```
state.json + runs/ + kernel-agent/
        │
        ▼
  collectors.py          coordinator.py (Phase 2 only)
  - collect_decision_journal     audit_extras on promote:
  - collect_kernel_profiling       promotion_rule, rule_detail,
  - enrich baseline / GEAK/OOB     keep_threshold_pct, …
        │
        ▼
  exporter.build(detail_level)
        │
        ├── session_breakdown.json
        └── reporters/ → session report markdown
```

### Responsibility matrix

| Field / behavior | Primary writer | Reader |
|------------------|----------------|--------|
| `explore_search.tested`, `rejected`, `last_round` | Coordinator / grid runner | `collect_decision_journal` |
| `*_attempts[].extras` (gain, best variant) | Coordinator audit | `_round_decision_from_attempt` |
| `promotion_rule`, `variants_tested_count`, … | Coordinator Phase 2 | same |
| variant `invocation`, `benchmark_report_path` | disk under `runs/{params,backends}/` | collector workspace walk |
| `kernel_profiling.outputs.top_kernels` | profile report / CSV / TraceLens JSON | `collect_kernel_profiling` |
| `geak_invocations` proposal/verification | kernel-agent results | existing invocation collector |

**Missing data is not always an orchestrator bug.** Gaps fall into:

1. **Capability not run** (e.g. no GEAK → empty invocations).
2. **Pre-Phase-2 session** (no `promotion_rule` in `extras`).
3. **Ledger not populated** (e.g. `gain_pct` absent in `tested` entries).
4. **Archive incomplete** (variant workspace / server.log missing on wekafs).
5. **Collector gap** (e.g. duplicate `params-last` row, `baseline_ref_tput` not resolved).

## New sections

### `decision_journal[]`

One entry per params/backends **round**.

Sources: `backend_winners_history`, `{phase}_search.last_round` + `tested` + `rejected`,
`{phase}_attempts[-1]`.

Each variant: name, fingerprint, args/envs, throughput, gain, outcome
(`tested` | `round_winner` | `promoted` | `rejected`), reject_reason, optional invocation.

Each round: `round_decision` from audit attempt (`outcome`, `best_variant_name`,
`gain_vs_cb_pct`, Phase 2 promotion fields).

### `kernel_profiling[]`

One entry per profile task or TraceLens status run.

Inline: launch args, artifact paths, parsed `top_kernels`, `analysis_summary`.
Never inline `.trace.json.gz` blobs.

## Implementation phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Schema + collectors + exporter; no Coordinator changes | done |
| 2 | Coordinator `audit_extras` promotion fields | done |
| 3 | Report renderers + `--detail-level` CLI | done |
| 4 | wekafs replay on real sessions | done (2/3 sessions; see below) |

## Validation criteria

### Unit tests

Fixtures assert schema shape, collector output, Coordinator wiring, report registration.
Pass = expected keys and counts on **synthetic** session trees.

### wekafs replay (integration)

Pass = for a archived session under
`/wekafs/users/<hash>/.claw/workspaces/<claw_sid>/current/hyperloom/`:

- `dump_session_breakdown` exits 0, writes v1.1 JSON
- `decision_journal` variant **names** match `state.*_search` ledgers
- markdown report includes Decision Journal and Kernel Profiling sections

Pass ≠ every optional field populated. Real sessions may lack promotion rules (old code),
validated gain (no validate_stack), kernel top-k (empty profile summary), or per-variant
invocation (missing server.log in archive).

Replayed sessions (2026-05-20):

| Claw session | Result |
|--------------|--------|
| `7efd182f-…` DSV3.1 ablation | pass — 3 variants in journal |
| `1c6e15d5-…` qwen3-235b | pass — 5 rounds, params/backends coverage |
| `f3200ba8-…` llama validated | skip — empty claw workspace |

## Known follow-ups

- Deduplicate `params-last` journal row when it overlaps `backend_winners_history`.
- Resolve `baseline_ref_tput` from round metadata consistently.
- Improve variant workspace / server.log resolution on wekafs archive paths.
- Re-replay with a **post-Phase-2** session to verify `promotion_rule` end-to-end.

## File touch summary

| File | Role |
|------|------|
| `breakdown/schema.py` | v1.1 TypedDicts, `SCHEMA_VERSION` |
| `breakdown/collectors.py` | `collect_decision_journal`, `collect_kernel_profiling`, enrichments |
| `breakdown/exporter.py` | wire collectors, `detail_level` |
| `orchestrator/coordinator.py` | Phase 2 `audit_extras` |
| `breakdown/reporters/_renderers/decision_journal.py` | markdown section |
| `breakdown/reporters/_renderers/kernel_profiling.py` | markdown section |
| `breakdown/reporters/compose.py` | section groups |
| `scripts/dump_session_breakdown.py` | `--detail-level` |
