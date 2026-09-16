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

| Section             | What it carries                                                                                          |
|---------------------|----------------------------------------------------------------------------------------------------------|
| `schema_version`    | The wire contract. Gate features on the **major** version, not the exact string.                         |
| `exported_at_utc`   | When this export was built.                                                                              |
| `exporter_version`  | Which exporter built it.                                                                                 |
| `metadata`          | Session identity, launch configuration, component versions, Langfuse receipt, and `warnings` -- how the export itself went, reported once and only here. |
| `outcome`           | Terminal status, stage reached, stop reason, the `baseline` and `final` measured results, and the `validation` that reconciles the stack's parts against its total. |
| `timeline`          | The run itself: one event per stage, oldest first, each carrying its span, status, and an `ext` block of what that kind of stage records. Startup source events live under `reports/sbd_v6/timeline/`. |
| `close`             | What the session settled at close: the steps the sequencer ran, the artifacts it published, the robustness findings it collected. |
| `critic`            | The critic agent's own run, iteration by iteration: what it was asked about, how its rulings fell (`verdict_counts`), and the four artifacts each pass left behind. Per-proposal verdicts stay with the proposals, on the timeline. |
| `robustness`        | What the robustness agent raised, turn by turn. Read `outcome` before counting `intents`: a turn that produced no envelope and a turn with nothing to raise are different findings. |

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

| Section     | Read from                                                                                                       |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| `metadata`  | `manifest.json` + `state.json`, overlaid by the recorder's own `session` / `task_config` / `versions` fragments  |
| `outcome`   | The recorder's `close` and stack fragments, plus `state.{current_best, cumulative_gain_validated, optimization_stack}` for the sessions that predate them |
| `timeline`  | The event fragments in the spool, closed and assembled per event; orphans left open by a killed phase are closed on first build |
| `close`     | The CLOSE sequencer's own `close` / `close_step` fragments                                                        |
| `critic`    | `critic-workdir/<NNN>/{request,judge_bundle,emit,review}.json`, recorded as each pass returns                     |
| `robustness`| The robustness agent's per-turn envelopes, recorded as each turn returns                                          |

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
- Inside v6, `enablement` gained the round ledger: `rounds[]` and its
  counters are added to the block, and the three state-sourced fields they
  replace (`stall_streak`, `inflight_task_id`, `dispatch_tick`) are no longer
  emitted. The block is runtime observability that is already `{}` on a
  session that ran no enablement, so it carries no field a consumer can gate
  a version on; the disposition of each replaced field is in
  `docs/reference/session-breakdown.md`.
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
