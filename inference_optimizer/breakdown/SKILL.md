---
name: session_breakdown
description: |
  Build a single self-contained `session_breakdown.json` capturing every
  fact a dashboard needs about one hyperloom optimization session. Use
  when the user mentions session-breakdown, kernel attribution,
  stats-service, capability summary, or wants to surface hyperloom data
  to an external consumer (claw-stats-service, results service, notebook).
globs:
  - "**/breakdown/**"
  - "**/session_breakdown*"
  - "**/dump_session_breakdown*"
---

# Session Breakdown Skill

## What it produces

A single JSON file: **`<session_dir>/session_breakdown.json`**.

- Schema:   `hyperloom.session_breakdown.v2` (additive over v1; see `breakdown/schema.py`)
- Producer: `inference_optimizer/breakdown/exporter.py`
- Filename: `BREAKDOWN_FILENAME` (= `session_breakdown.json`)

The JSON has 14 top-level sections plus envelope:

| Section              | What it carries                                                                                          |
|----------------------|----------------------------------------------------------------------------------------------------------|
| `session`            | Internal `session_id`, Claw `claw_session_id`, sandbox user, start/end ts, stop_reason, host, code SHA.  |
| `workload`           | Framework, model, GPU, TP, CONC/ISL/OSL/precision, objective.                                            |
| `baseline`           | Baseline throughput / accuracy / latency, config path, benchmark report path, failure streak.            |
| `final`              | `current_best` throughput, validated cumulative gain, action path, extra args/envs.                      |
| `phase_timeline`     | Chronological list of every action attempt + kernel_opt + integrate event.                               |
| `capability_summary` | One row per family: geak / oob / backends / params / sweep / validate_stack with status + attempts + keeps. |
| `geak_invocations`   | Per-attempt detail: prompt path, optimized files, verification, decision, micro_speedup.                 |
| `oob_invocations`    | Same schema as `geak_invocations`, with `backend ∈ {claude, codex}`.                                     |
| `kernel_lifecycle`   | 5 stages: `detected` / `recommended` / `optimized` / `adopted` / `rejected`.                             |
| `param_search`       | Backends + params search ledgers (tested / accepted / rejected / top_by_gain / winner_history).          |
| `sweep`              | Grid size, best_overall, pareto_front, every variant's benchmark numbers.                                |
| `critic_robustness`  | Per-iter critic verdicts + robustness signals.                                                           |
| `telemetry`          | Paths to `benchmark_report.json` / `torch_trace` / `system_profile` / server logs + aggregated GPU monitor. |
| `attribution`        | Per-stack-entry gain ledger + family breakdown (geak / oob / backends / params / sweep / validated).     |
| `warnings`           | Best-effort caveats (missing files, partial sections, reconstructed fields).                             |
| `source_files`       | Mapping from logical section to relative path under `session_dir`.                                       |

## Who reads it

- **`claw-stats-service`** — primary consumer. Replaces the
  MAE-synthesized `raw_report` / `fact_sheet`. Read order in
  stats-service should be: prefer `session_breakdown.json` if present,
  fall back to legacy MAE output otherwise.
- **`hyperloom-results-service`** — `ci/publish_artifacts.py` POSTs this
  JSON when `HYPERLOOM_RESULTS_SERVICE_URL` is set.
- **Offline / notebook analysis** — single file, easy to load, no DB
  needed.

## When to refresh (LLM Orchestrator decision tree)

The Coordinator's `cli.py` finally block writes `session_breakdown.json`
unconditionally at end-of-session — that's the safety net. But you
should ALSO refresh it eagerly when downstream dashboards may be
watching this session live:

1. **Always** at end-of-session (handled by `cli.py finally` — you do not need to dispatch this).
2. After every successful `validate_stack` (cumulative gain just changed).
3. After every KEEP'd kernel/backends/params variant when a live
   dashboard is observing this session.
4. **Never** mid-action — collectors expect a coherent state snapshot.

Dispatch action `session_breakdown` (yaml meta lives at
`actions/_meta/session_breakdown.yaml`) — it's a single 1-minute action,
no inputs required.

## How to invoke

### LLM-driven (Coordinator action)

```yaml
# Issue an action intent like any other action
{ "action": "session_breakdown", "params": {} }
```

### Code-driven (Python import)

```python
from inference_optimizer.breakdown import build, write_breakdown_json

# Just compute the dict (no side effects)
breakdown = build("/workspace/hyperloom")

# Compute + atomically write to <sd>/session_breakdown.json
out_path = write_breakdown_json("/workspace/hyperloom")
```

### CLI / offline (`scripts/dump_session_breakdown.py`)

```bash
# Live session in this sandbox
python -m inference_optimizer.scripts.dump_session_breakdown

# Historical session on WekaFS
python -m inference_optimizer.scripts.dump_session_breakdown \
    --session-dir /wekafs/users/zgong/inference_optimizer-sessions/<sid>

# Override output path (e.g. write to a staging area)
python -m inference_optimizer.scripts.dump_session_breakdown \
    --session-dir <SD> --output /tmp/breakdown.json
```

### Bulk historical (operator)

```bash
for d in /wekafs/users/*/inference_optimizer-sessions/*; do
    [ -d "$d" ] || continue
    python -m inference_optimizer.scripts.dump_session_breakdown \
        --session-dir "$d" > /dev/null
done
```

## Field reference (what comes from where)

The collector for each section reads only from the listed sources. All
collectors are pure functions; failure in one section never poisons
another (each becomes a `warnings[]` entry instead).

| Section              | Reads from                                                                                                            |
|----------------------|----------------------------------------------------------------------------------------------------------------------|
| `session`            | `manifest.json` + `state.{session_id, stop_reason, max_minutes, tick, start_ts}`                                     |
| `workload`           | `manifest.{framework, model_*, gpu_type, tp, workload, objective}` + `state.{model_class, framework, gpu_type}`      |
| `baseline`           | `state.{baseline_tput, baseline_accuracy, last_baseline.workspace, baseline_attempts}` + `<workspace>/benchmark_*/benchmark_report.json` |
| `final`              | `state.{current_best, cumulative_gain, cumulative_gain_validated_*, optimization_stack}`                            |
| `phase_timeline`     | `state.{<action>_attempts, kernel_opt_attempts.history, kernel_integrate_attempts.attempts}` sorted by `ts`           |
| `capability_summary` | Reduces invocations + per-action attempts + search ledgers into 6 rows                                              |
| `geak_invocations`   | `kernel-agent/runs/<sid>/{optimization_attempts.jsonl, prompts/, optimized/, results/, verification/}` filtered by `backend == "geak"` (also scans legacy `kernel-agent-workspace/.../kernel-agent/runs/...` for historical sessions). Per-attempt files under `optimized/` are discovered by `glob("<attempt_id>*")`, so both the historical `<attempt_id>_optimized.<suffix>` name and the post-2026-05 `<attempt_id>_stdout.log` name are picked up transparently — see `kernel-agent/SKILL.md` § *Per-attempt stdout file naming*. |
| `oob_invocations`    | Same as GEAK, filtered by `backend ∈ {claude, codex}`                                                                |
| `kernel_lifecycle`   | `runs/profile/*/benchmark_*/benchmark_report.json` (detected) + `state.last_trace_analyze` (recommended) + invocations folded (optimized) + `state.{kernel_integrate_attempts, rejected_kernel_*}` (adopted/rejected) |
| `param_search`       | `state.{explore_search, params_winner_history, synergy_attempted, discovered_flags, backend_winners_history}` |
| `sweep`              | `state.last_sweep` + `runs/sweep/<task>/variant_*/benchmark_*/benchmark_report.json`                                |
| `critic_robustness`  | `critic-workdir/<NNN>/{request,judge_bundle,emit,review}.json` + `robustness-workdir/<NNN>/{signal,action}.json`   |
| `telemetry`          | All `runs/**/benchmark_*/benchmark_report.json` + `torch_trace/` + `system_profile/` + `server*.log`                  |
| `attribution`        | `state.gain_per_stack_entry` (preferred) OR best-effort reconstruction from `state.optimization_stack`              |

## What is NOT in scope

- **Real-time event streaming** — use `claw_session_events` for that.
- **LLM-based attribution** — every collector is deterministic /
  rule-based. Attribution is `delta_pct` math, not natural language.
- **Schema migration** — consumers MUST check `schema_version` and gate
  features on it.
- **Cross-session aggregation** — one file per session. Use
  `ci/build_summary.py` (or a Jupyter notebook) for fleet views.

## Failure modes

| Symptom                                            | Cause                                                                                            | Mitigation                                                                                |
|----------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `warnings: ["state.json missing"]`                 | Session was created (manifest written) but `Coordinator.save()` never ran                        | Sections fall back to manifest-only data.                                                  |
| `warnings: ["manifest.json missing"]`              | Session was created without the standard cli.py path (rare)                                      | `session.session_id` falls back to `state.session_id`.                                     |
| `attribution.notes ≠ []`                            | `Coordinator` did not write `state.gain_per_stack_entry`                                         | Attribution is reconstructed from `optimization_stack`; consumer should treat as approximate. |
| Empty `geak_invocations` & `oob_invocations`        | Kernel-agent never ran, or wrote to a non-standard workspace                                     | Verify `$SD/kernel-agent/runs/` exists (or, for pre-migration sessions, `$SD/kernel-agent-workspace/kernel-agent/runs/`). |
| `kernel_lifecycle.detected = []`                    | `profile` action never ran or its `benchmark_report.json` had no `kernel_summary`                | Re-run profile, or fall back to `recommended` from `state.last_trace_analyze`.            |
| Large `warnings[]`                                   | Multiple JSON parse failures on `optimization_attempts.jsonl`                                    | Inspect `kernel-agent-workspace/.../logs/` for the corresponding kernel-agent CLI logs.    |

## Versioning policy

- `schema_version` (in `schema.py`) is bumped ONLY on breaking
  changes (renamed/removed fields, changed semantics).
- Adding optional fields is **never** a breaking change.
- `exporter_version` tracks the exporter implementation independently;
  consumers can ignore it.

## Testing

Each collector has a unit test under `tests/test_breakdown_*.py` that
runs it against a fixture session_dir tree. An end-to-end test calls
`build(...)` on a fully-populated fixture and JSON-schema-validates the
result. Run:

```bash
pytest tests/test_breakdown_*.py -v
```
