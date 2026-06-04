# Breakdown And Reporting Contract

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

`session_breakdown.json` is the external contract between `inference_optimizer` (producer) and downstream consumers (claw-stats-service, results service, dashboards, notebooks). This artifact records the live contract layering so later cleanup keeps the dashboard contract intact while still simplifying internals.

Runtime source of truth:

- schema: `inference_optimizer/breakdown/schema.py` (`SCHEMA_VERSION = "hyperloom.session_breakdown.v2"`).
- producer: `inference_optimizer/breakdown/exporter.py`.
- collectors: `inference_optimizer/breakdown/collectors.py`.
- consumer doc: `docs/INTEGRATION_SESSION_BREAKDOWN.md`.

## Versioning

- Current producer version: `hyperloom.session_breakdown.v2`.
- `v2` is additive over `v1`: it only adds sections, so a `v1` reader can consume a `v2` file by ignoring unknown keys.
- Additive optional fields do not bump the version. Renamed/removed/semantically-changed fields require a major bump and a one-release parallel-write window.
- `exporter_version` tracks the exporter implementation independently of the wire schema.

## Core Output Contract (must keep)

- session metadata
- workload
- baseline
- final result
- phase / action timeline
- optimization stack
- capability summary
- kernel lifecycle
- sweep / concurrency result
- critic / robustness summary
- warnings
- source artifact references

These drive dashboards and the success judgment.

## Optional Diagnostic Sections (additive)

- specialist runs (+ optional transcripts)
- verbose kernel attempt details
- raw telemetry paths
- PR feed details
- internal policy denial history
- `kernel_optimization_summary`, `conc_sweep_summary`

Missing-data rule: represent as `null` / `[]` / `{}`, never fabricate.

## Archived Compatibility Sections

Per `01-baseline-contracts/03-compatibility-decisions.MD`, old session breakdown readers are kept until downstream needs are confirmed. These only serve archived sessions:

- legacy `backends` / `params` / `validate_stack` provenance aliases (folded into `legacy_*` / `explore` rows)
- legacy kernel-agent nested artifact layout scan
- old provenance strings

Deletion condition: remove the archived readers and aliases once downstream dashboard / offline analysis confirms old sessions are no longer needed.

Removed in the high-risk cleanup pass:

- old T2/T3 graph placeholders:
  `pending_edges`, `edges_promoted`, `edges_negated`,
  `points_created`, `points_by_kind`, and `commit_summary`.
- These fields belonged to the retired hypothesize/verify graph protocol
  and were either always empty or derived only from old `propose_point`
  audit rows.

## Producer / Report Split

- `session_breakdown.json`: machine-readable, stable fields, minimal narrative; dashboard source of truth.
- Markdown report: human-readable summary; may reference breakdown data but must not become a second source of truth.

## Implemented In This Step

- Aligned `docs/INTEGRATION_SESSION_BREAKDOWN.md` to the current `v2` schema (versioning section, top-level shape example, worked example), preserving the v1->v2 additive-compatibility note.
- Aligned `inference_optimizer/breakdown/SKILL.md` schema line to `v2`.

Initial pass was doc-to-code alignment. Follow-up cleanup also removed
retired old-KB placeholder fields from the producer/schema.

## Not Done Yet (decision-gated or higher-risk)

- Splitting `collectors.py` by section (large file; defer to a focused refactor).
- Removing archived readers / legacy aliases (blocked by the downstream-needs decision).
- Trimming markdown-report duplication of breakdown data.

## Downstream Contract Test Notes

Keep:

- live breakdown smoke (schema version + headline fields),
- missing-data-not-fabricated,
- final throughput / gain / action path,
- kernel lifecycle summary,
- specialist/action attribution if dashboards use it.

Trim later (only after archived-reader decision):

- exhaustive legacy-alias detail tests,
- collector fallback-order micro-tests,
- markdown exact-sentence assertions.
