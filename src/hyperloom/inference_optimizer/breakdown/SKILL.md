---
name: session_breakdown
description: |
  Build a single self-contained `session_breakdown.json` capturing every
  fact a dashboard needs about one hyperloom optimization session. Use
  when the user mentions session-breakdown, kernel attribution,
  a stats/reporting service, capability summary, or wants to surface hyperloom
  data to an external consumer (a results/stats service, notebook, dashboard).
globs:
  - "**/breakdown/**"
  - "**/session_breakdown*"
  - "**/dump_session_breakdown*"
---

# Session Breakdown Skill

## What it produces

A single JSON file: **`<session_dir>/session_breakdown.json`**.

- Schema:   `hyperloom.session_breakdown.v6.0` (hardcoded; see `SCHEMA_VERSION` in `breakdown/schema.py`).
- Producer: `src/hyperloom/inference_optimizer/breakdown/exporter.py`
- Filename: `BREAKDOWN_FILENAME` (= `session_breakdown.json`)

Beyond the envelope (`schema_version` / `exported_at_utc` / `exporter_version`), the JSON
carries the sections below. This is the subset most consumers use, not the full set —
`breakdown/exporter.py` (the `breakdown = {...}` literal) and `breakdown/schema.py` are
authoritative.

| Section              | What it carries                                                                                          |
|----------------------|----------------------------------------------------------------------------------------------------------|
| `session`            | Internal `session_id`, Claw `claw_session_id`, sandbox user, start/end ts, stop_reason, host, code SHA.  |
| `workload`           | Framework, model, GPU, TP, CONC/ISL/OSL/precision, objective.                                            |
| `baseline`           | Baseline throughput / accuracy / latency, config path, benchmark report path, failure streak.            |
| `final`              | `current_best` throughput, validated cumulative gain, action path, extra args/envs.                      |
| `phase_timeline`     | Chronological list of every action attempt + kernel_opt + integrate event.                               |
| `capability_summary` | One row per live capability: geak / forge / explore / sweep / specialist, plus legacy rows kept for archived sessions. |
| `geak`               | GEAK route diagnostics, normalized result, accepted artifacts, and recovery evidence when the route ran outside the native kernel-agent layout. |
| `optimizations`      | Canonical adopted-optimization API, projected from author-time recorder streams. **Read `available` first**: `false` means the records are missing, not that nothing was adopted. `attempts[]` holds every attempt, `entries[]` the adopted ledger, `validation` the reconciliation. |
| `kernel_lifecycle`   | 5 stages: `detected` / `recommended` / `optimized` / `adopted` / `rejected`.                             |
| `collective`         | Collective-lane campaigns: `only_mode` / `attempts[]` / `last`. Adoption is decided by `integration_decision` (E2E gate), not `decision` (microbenchmark). |
| `param_search`       | Compatibility alias for the merged explore ledger (tested / accepted / rejected / top_by_gain / winner_history). |
| `sweep`              | Grid size, best_overall, pareto_front, every variant's benchmark numbers.                                |
| `critic_robustness`  | Per-iter critic verdicts + robustness signals.                                                           |
| `telemetry`          | Paths to `benchmark_report.json` / `torch_trace` / `system_profile` / server logs + aggregated GPU monitor. `telemetry.orchestration_context` carries the compaction-loop health: `seed_prompts`, `delta_prompts`, `compactions`, `degenerate_compactions`, `tick_count`, `compactions_per_tick`, `delta_ratio`, `context_tokens_at_compaction`. See `docs/reference/session-breakdown.md §telemetry.orchestration_context`. |
| `metadata`           | Additive V6 schema/version, session, launch configuration, Langfuse, and warning metadata.                |
| `outcome`            | Additive V6 terminal status, reached stage, stop reason, and final measured result.                       |
| `timeline`           | Additive V6 ordered stage events; startup source events live under `reports/sbd_v6/timeline/`.            |
| `close`              | Additive V6 close-stage payload; currently empty until the close-stage collector is implemented.          |
| `warnings`           | Best-effort caveats (missing files, partial sections, reconstructed fields).                             |
| `source_files`       | Mapping from logical section to relative path under `session_dir`.                                       |

## Who reads it

- **A downstream stats/reporting service** — primary consumer. Replaces the
  MAE-synthesized `raw_report` / `fact_sheet`. Recommended read order:
  prefer `session_breakdown.json` if present, fall back to legacy MAE
  output otherwise.
- **`hyperloom-results-service`** — downstream automation may POST this
  JSON when `HYPERLOOM_RESULTS_SERVICE_URL` is set.
- **Offline / notebook analysis** — single file, easy to load, no DB
  needed.

## When to refresh (LLM Orchestrator decision tree)

The Coordinator's `cli.py` finally block writes `session_breakdown.json`
unconditionally at end-of-session — that's the safety net. But you
should ALSO refresh it eagerly when downstream dashboards may be
watching this session live:

1. **Always** at end-of-session (handled by `cli.py finally` — you do not need to dispatch this).
2. After every KEEP'd explore, specialist, framework, or kernel result
   whose benchmark changed the final stack.
3. After a successful sweep or conc_sweep when a live dashboard is
   observing this session.
4. **Never** mid-action — collectors expect a coherent state snapshot.

Dispatch action `session_breakdown` — a single 1-minute action, no inputs
required.

## How to invoke

### LLM-driven (Coordinator action)

```yaml
# Issue an action intent like any other action
{ "action": "session_breakdown", "params": {} }
```

### Code-driven (Python import)

```python
from hyperloom.inference_optimizer.breakdown import build, write_breakdown_json

# Just compute the dict (no side effects)
breakdown = build("/workspace/hyperloom")

# Compute + atomically write to <sd>/session_breakdown.json
out_path = write_breakdown_json("/workspace/hyperloom")
```

### CLI / offline (`hyperloom.inference_optimizer.tools.dump_session_breakdown`)

```bash
# Live session in this sandbox
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown

# Historical session on a shared filesystem
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
    --session-dir /shared/hyperloom-sessions/<user>/<sid>

# Override output path (e.g. write to a staging area)
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
    --session-dir <SD> --output /tmp/breakdown.json
```

### Bulk historical (operator)

```bash
for d in /shared/hyperloom-sessions/*/*; do
    [ -d "$d" ] || continue
    python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
        --session-dir "$d" > /dev/null
done
```

## Field reference (what comes from where)

The collector for each section reads only from the listed sources. All
collectors are pure functions; failure in one section never poisons
another (each becomes a `warnings[]` entry instead). Like the table above,
this reference is partial — `breakdown/exporter.py` is authoritative.

| Section              | Reads from                                                                                                            |
|----------------------|----------------------------------------------------------------------------------------------------------------------|
| `session`            | `manifest.json` + `state.{session_id, stop_reason, stop_ts, max_minutes, tick, start_ts, resumed_ts}`                |
| `workload`           | `manifest.{framework, model_*, gpu_type, tp, workload, objective}` + `state.{model_class, framework, gpu_type}`      |
| `baseline`           | `state.{baseline_tput, baseline_accuracy, last_baseline.workspace, baseline_attempts}` + `<workspace>/benchmark_*/benchmark_report.json` |
| `final`              | `state.{current_best, cumulative_gain_validated, cumulative_gain_validated_*, optimization_stack}`                  |
| `phase_timeline`     | `state.{<action>_attempts, kernel_opt_attempts.history, kernel_integrate_attempts.attempts}` sorted by `ts`           |
| `capability_summary` | Reduces invocations + per-action attempts + search ledgers into 8 rows: geak / forge / explore / sweep / specialist plus the backends / params / validate_stack compatibility rows |
| `optimizations`      | The recorder's own streams only — `operations` / `adoptions` / `measurements` / `artifacts`, as the producers wrote them. Never rebuilt from `state.json`; when the records are absent the section reports `available: false` instead. |
| `kernel_lifecycle`   | `runs/profile/*/benchmark_*/benchmark_report.json` (detected) + `state.last_trace_analyze` (recommended) + invocations folded (optimized) + `state.{kernel_integrate_attempts, rejected_kernel_*}` (adopted/rejected) |
| `collective`         | `state.{collective_only_mode, collective_attempts, last_collective}`                                                  |
| `param_search`       | `state.{explore_search, discovered_flags}` (`synergy_attempted` now comes from `explore_search`; `winner_history` / `backend_winners_history` are emitted empty); `params` / `backends` ledgers are historical aliases only |
| `critic_robustness`  | `critic-workdir/<NNN>/{request,judge_bundle,emit,review}.json` + `robustness-workdir/<NNN>/{signal,action}.json`   |
| `telemetry`          | All `runs/**/benchmark_*/benchmark_report.json` + `torch_trace/` + `system_profile/` + `server*.log`                  |

## What is NOT in scope

- **Real-time event streaming** — use `claw_session_events` for that.
- **LLM-based attribution** — every collector is deterministic /
  rule-based. Attribution is `delta_pct` math, not natural language.
- **Schema migration** — consumers MUST check `schema_version` and gate
  features on it, comparing the **major** version (`vN`) rather than the
  exact string: the producer emits both `…v2` and `…v3.0` today (see
  Versioning policy) and they are wire-compatible.
- **Cross-session aggregation** — one file per session. Use a Jupyter
  notebook or downstream analytics job for fleet views.

## Failure modes

| Symptom                                            | Cause                                                                                            | Mitigation                                                                                |
|----------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `warnings: ["state.json missing"]`                 | Session was created (manifest written) but `Coordinator.save()` never ran                        | Sections fall back to manifest-only data.                                                  |
| `warnings: ["manifest.json missing"]`              | Session was created without the standard cli.py path (rare)                                      | `session.session_id` falls back to `state.session_id`.                                     |
| `optimizations.available = false`                   | No operations were recorded, or the projection failed                                            | Read `unavailable_reason`. The empty arrays mean unknown, not none; a `warnings` entry says whether `state.json` knew of adopted work the recorder missed. |
| `optimizations.validation.unclaimed_integration_count > 0` | A change recorded as integrated has no adoption crediting it                              | The adoption write was lost. `unattributed_gain_pct` is overstated by whatever those steps earned. |
| `optimizations.entries[].gain_method = local_gain_projected` | The step recorded no finishing throughput                                              | Its gain is projected from the executor's own percentage, not measured against the chain. |
| `kernel_lifecycle.detected = []`                    | `profile` action never ran or its `benchmark_report.json` had no `kernel_summary`                | Re-run profile, or fall back to `recommended` from `state.last_trace_analyze`.            |
| Large `warnings[]`                                   | Multiple JSON parse failures on `optimization_attempts.jsonl`                                    | Inspect `kernel-agent-workspace/.../logs/` for the corresponding kernel-agent CLI logs.    |

## Versioning policy

- `schema_version` (in `schema.py`) carries the **major** contract
  version; it is bumped ONLY on breaking changes (renamed/removed
  fields, changed semantics).
- New exports carry `hyperloom.session_breakdown.v6.0`. V6 is a breaking
  cutover for the timeline: the actions record their own events as they run,
  so an event's `start_time` is when the work began rather than when its
  artefacts were written, and the KERNEL and BASELINE projections are gone.
  Consumers that sorted around the old collapsed windows need to be rechecked.
- V5 was the preceding cutover, for optimization results: `optimizations` is
  reshaped, and the `optimization_stack`, `attribution`, `geak_invocations`,
  `forge_invocations`, and `gemm_tuning` projections are gone. Consumers
  MUST match on the `vN` major prefix, never on exact-string equality, and
  archived V2/V3/V4/V5 documents need a migration before a V6 reader sees them.
- `optimizations` carries its own `schema_version` (currently `5`),
  independent of the envelope's.
- Adding optional fields is **never** a breaking change.
- `exporter_version` tracks the exporter implementation independently;
  consumers can ignore it.

## Testing

Each collector has a unit test under `src/hyperloom/inference_optimizer/tests/`
that runs it against a fixture session_dir tree. An end-to-end test calls
`build(...)` on a fully-populated fixture and JSON-schema-validates the
result. Run:

```bash
pytest src/hyperloom/inference_optimizer/tests/ -k breakdown -v
```
