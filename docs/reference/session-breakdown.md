---
myst:
    html_meta:
        "description": "Reference for the session_breakdown.json contract in Hyperloom. Covers schema versioning, top-level shape, all sections, a worked example, and stability guarantees."
        "keywords": "Hyperloom, session_breakdown.json, schema, API contract, LLM inference, AMD GPU, ROCm, session data, downstream integration, versioning, telemetry, observability"
---
# `session_breakdown.json` integration in Hyperloom

```{note}
This page is for **integrators and downstream consumers** — teams building
dashboards, reporting pipelines, or services (such as `claw-stats-service`)
that read Hyperloom session output programmatically. If you just ran an
optimization and want to check your results, read the three headline fields
described in [Run a Hyperloom optimization](../how-to/optimize.md#output-and-artifacts)
first.
```

`session_breakdown.json` is the single external contract between
the `inference_optimizer` runtime (producer) and any downstream
consumer (`claw-stats-service`, results service, notebooks, custom
dashboards). One file per session, written to
`$SESSION_DIR/session_breakdown.json` at session end (and on
operator demand using [`dump_session_breakdown.py`](operator-scripts.md)).

The authoritative source of truth for the wire shape is
[`src/hyperloom/inference_optimizer/breakdown/schema.py`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/breakdown/schema.py).
This page describes the contract from a consumer's perspective.

---

## Versioning

The top-level `schema_version` field is a stable string. The producer
emits one of two version strings depending on the aggregation path:

```json
"schema_version": "hyperloom.session_breakdown.v2"     // legacy collector-only fallback
"schema_version": "hyperloom.session_breakdown.v3.0"   // when author-time recorder fragments are present
```

When the session has author-time recorder fragments (the "new way"
write-side spool), the exporter aggregates from them and stamps
`v3.0`; sessions without fragments (historical runs, recorder-disabled
runs) fall back to the legacy collectors and keep the `v2` stamp. Both
strings can therefore appear in production today.

`v3.0` is the same additive wire shape as `v2` — the recorder path
captures facts the collectors can miss (pre-dispatch / infra failures,
pruned robustness signals) but adds no breaking field changes. A reader
written for `v2` parses a `v3.0` file unchanged by ignoring unknown keys.
`v2` is itself additive over `v1`: it only adds sections (for example,
`specialist_runs`, `action_timeline`, `kernel_optimization_summary`,
`conc_sweep_summary`).

Compatibility rules:

* **Do not gate on string equality**. A consumer that treats
  `schema_version` as a contract MUST accept the `v2` and `v3.0`
  family (for example, parse the `vN[.M]` prefix and compare the major
  component, or allowlist both strings). Pinning to the exact `v2`
  string will reject `v3.0` files even though they are wire-compatible.
* **New optional fields** might appear at any time without bumping
  the major version. Consumers must tolerate unknown keys.
* **Renamed, removed, or semantically changed** fields require a major
  bump (for example, `v3` → `v4`). The runtime will continue to write the previous
  version's file in parallel for at least one release after the bump.
* **Missing data** is always represented as `null`, `[]`, or `{}` —
  never as a default / fabricated value. Consumers MUST treat
  missing data as "not available".
* All values are JSON-serializable (no dataclasses, enums, or
  Python-specific types in the wire shape).

The `exporter_version` field carries the exporter implementation version
(currently `"session-breakdown-1.0.0"`), independent of the Hyperloom package
version, for incident triage and per-version filtering.

---

## Top-level shape

The following JSON structure shows all top-level fields in `session_breakdown.json`.

```text
{
  "schema_version": "hyperloom.session_breakdown.v2",
  "exported_at_utc": "2026-05-17T12:34:56.789Z",
  "exporter_version": "session-breakdown-1.0.0",

  "session":            { /* §3  SessionMeta */ },
  "workload":           { /* §4  Workload */ },
  "baseline":           { /* §5  Baseline */ },
  "final":              { /* §6  Final state — SaFE contract core */ },
  "phase_timeline":     [ /* §7  PhaseEvent[] */ ],
  "capability_summary": { /* §8  Capability cards */ },
  "geak_invocations":   [ /* §9  Invocation[] */ ],
  "forge_invocations":  [ /* §10 Invocation[] */ ],
  "kernel_lifecycle":   { /* §11 4+1-stage kernel lifecycle */ },
  "param_search":       { /* §12 ParamSearch */ },
  "sweep":              { /* §13 Sweep */ },
  "critic_robustness":  { /* §14 Critic iterations + Robustness signals */ },
  "telemetry":          { /* §15 Telemetry artefact paths */ },
  "attribution":        { /* §16 Gain attribution per stack entry */ },

  "warnings":           [ /* string[] — non-fatal collector warnings */ ],
  "source_files":       { /* §17 SourceFiles — raw artefact paths */ },

  /* Optional sections — present when the run produced the relevant data.
     Consumers MUST tolerate their absence (total=False TypedDict). */
  "model_info":                  { /* model architecture summary */ },
  "phase_segments":              [ /* per-phase segment records */ ],
  "explore_search":              { /* EXPLORE dedup ledger */ },
  "perfskills":                  { /* perf-skill telemetry */ },
  "kb_provenance":               { /* KB read/write provenance */ },
  "specialist_runs":             [ /* specialist sub-agent runs */ ],
  "optimization_stack":          { /* accepted KEEP stack */ },
  "gemm_tuning":                 { /* FP8 GEMM tuning results */ },
  "kernel_roofline":             { /* kernel roofline snapshot */ },
  "kernel_optimization_summary": { /* kernel-opt rollup */ },
  "conc_sweep_summary":          { /* post-run concurrency sweep */ },
  "roofline":                    { /* roofline analysis */ },
  "roofline_progress":           [ /* roofline watermark crossings */ ],
  "decision_trace":              { /* KEEP/REVERT decisions + token rollup */ },
  "token_usage":                 { /* LLM token spend rollup (see below) */ },
  "langfuse":                    { /* Langfuse push receipt */ },
  "kernel_journey":              { /* kernel lifecycle journey */ },
  "versions":                    { /* component/version stamps */ }
}
```

The `session` (SessionMeta) section also carries `user_data_path` and a
`recovery` sub-object in addition to the fields documented in §3.

All sections use the `total=False` TypedDict convention — every field
is optional. Consumers should expect partial documents when a session
ended early (`baseline_failed`, `time_exhausted` before kernel-opt
started, …).

---

## `session` — `SessionMeta`

The `session` section contains the following metadata fields.

| Field              | Type    | Description                                                                                  |
|--------------------|---------|----------------------------------------------------------------------------------------------|
| `session_id`       | string  | Hyperloom-internal session id (from `manifest.session_id`).                                  |
| `claw_session_id`  | string \| null | Hosted SaFE / Claw id; populated from env `CLAW_SESSION_ID`.                          |
| `sandbox_user_id`  | string \| null | Hosted SaFE user id; populated from env `SANDBOX_USER_ID`.                            |
| `created_at_utc`   | string  | ISO-8601 UTC.                                                                                |
| `ended_at_utc`     | string  | ISO-8601 UTC.                                                                                |
| `stop_reason`      | string  | One of `target_reached`, `time_exhausted`, `global_converged`, `max_ticks`, `baseline_failed`, ... |
| `max_minutes`      | int     | Configured time budget.                                                                       |
| `elapsed_minutes`  | float   | Actual wall-clock.                                                                            |
| `host`             | string  | Hostname of the Coordinator pod.                                                              |
| `code_revision`    | string  | Hyperloom git SHA.                                                                            |
| `pid`              | int     | Coordinator PID.                                                                              |
| `session_dir`      | string  | Concrete session directory, typically `$USER_DATA_PATH/<model_basename>/<timestamp>/`.       |
| `tick_count`       | int     | Number of Coordinator ticks.                                                                  |
| `image`            | string \| null | Container image fully-qualified, if configured.                                       |

---

## `workload` — `Workload`

The workload the session optimised: model, framework, GPU type, shape,
precision, and the optimization objective (gain %, target throughput,
baseline-relative, or time-only). See `schema.py::Workload` for the
full field list. Consumers should treat the `objective.kind` enum as
the canonical optimisation goal.

---

## `baseline` — `Baseline`

The starting point Hyperloom measured before any modifications.
Includes throughput, accuracy, optional time to first token (TTFT) and end-to-end latency (E2EL), the materialised
benchmark config path, attempt history (in case the baseline required
retries), and the `BenchmarkInvocation` record needed to **replay**
the exact baseline benchmark.

`baseline.invocation.framework_args_source` is one of:

* `log_non_default_args` — Most authoritative (parsed from the
  vllm/sglang server's own arg echo).
* `log_args_line` — `Args: Namespace(...)` header.
* `log_python_cmd` — Literal `python …` launch line scraped from logs.
* `yaml_cmd` — `cmd:` / `command:` / `launch:` field in the
  materialised config YAML.
* `yaml_benchmark` — Synthesised from Magpie's `benchmark.*` YAML
  fields.
* `unknown` — None of the above; a warning is appended to
  top-level `warnings`.

`extra_envs` is allowlist-filtered to keep secrets out of the
breakdown. Do not assume it contains every env var the session ran with.

---

## `final` — `Final` (SaFE contract core)

The end-state Hyperloom validated against the SaFE (Safe and Fast Execution) contract. The two most important fields for
downstream consumers:

| Field                              | Meaning                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------|
| `throughput_tok_s_per_gpu`         | Validated end-of-session throughput. The headline number.                                 |
| `cumulative_gain_pct_validated`    | Validated cumulative gain vs `baseline.throughput_tok_s_per_gpu`. The headline %.         |
| `action_path`                      | Ordered list of `action:variant` labels that made the final stack — the recipe.            |
| `extra_server_args`                | The exact extra args needed to reproduce the final config.                                 |
| `extra_envs`                       | The exact env overrides needed to reproduce the final config (allowlisted, no secrets).    |
| `invocation`                       | Same shape as `baseline.invocation`; lets a consumer replay the final benchmark.          |
| `closing_phase_entered`            | True iff Coordinator entered the closing phase cleanly (vs SIGTERM exit).                  |

> Consumer best practice: index on
> `(session.session_id, final.throughput_tok_s_per_gpu,
> final.cumulative_gain_pct_validated, workload.model_name,
> workload.gpu_type)`. Everything else is detail.

---

## `phase_timeline` — `PhaseEvent[]`

Chronologically ordered events, one per Coordinator action completion.
Each entry has `action`, `task_id`, `status`, `decision`,
`key_metric`, optional `kernel_id` (for kernel-owned actions),
optional `workspace`, and an `extras` dict for action-specific payload.

Useful for rendering session-progress timelines and "what changed at
T+90 min" charts.

---

## `capability_summary` — `CapabilitySummary`

One card per live capability (`geak`, `forge`, `explore`, `sweep`,
`specialist`) with: `status`, `attempts`, `keeps`, `tested`,
`best_gain_pct`, `reason`. Legacy `backends`, `params`, and
`validate_stack` rows can appear when archived sessions are rebuilt.
Drives the per-session UI cards in Primus-Claw.

---

## `geak_invocations` / `forge_invocations` — `Invocation[]`

Same `Invocation` shape across both lists; `backend` distinguishes
(`geak` / `forge`). One entry per attempt-on-a-kernel. The `decision` enum is
`KEEP` / `PARTIAL` / `REVERT` / `FAILED`.

---

## `kernel_lifecycle` — `KernelLifecycle`

The 4+1-stage kernel pipeline:

* `detected` — TraceLens-identified hot kernels.
* `recommended` — Critic-filtered candidates with backend
  recommendations.
* `optimized` — Kernels with at least one completed backend attempt
  and `best_micro_speedup`.
* `adopted` — Kernels promoted into the final stack (end-to-end validated).
* `rejected` — Kernels considered then dropped, with `reason`.

The same `kernel_id` appears in multiple lists as it progresses.

---

## `param_search`

The canonical field is `explore_search` (the native merged ledger), with
`ParamSearchEntry` records for every tested variant: `status` ∈ `accepted` /
`rejected` / `tested`, the `extra_server_args` / `extra_envs` it injected, the
`output_throughput` it measured, and the resulting `gain_pct`. The
`param_search` ledger is a v1-reader compatibility alias for the same data;
`params` and `backends` are older compatibility aliases emitted for archived
sessions and old readers. The section also includes
`synergy_attempted`, `discovered_flags`, and `backend_winners_history`.

---

## `sweep`

Final concurrency / input sequence length (ISL) / output sequence length (OSL) sweep. Always includes `all_variants`
(a `SweepPoint[]`) and `best_overall`. `best_for_each_conc` and
`pareto_front` are populated when the sweep grid is large enough.

---

## `critic_robustness`

Decision-review trail: every Critic iteration (verdict + paths to
request / judge_bundle / emit / review JSONs), plus every Robustness
signal (`crash` / `stall` / `disk_full` / `cluster_fault` / …).

---

## `telemetry`

Paths only (no copied content): `baseline_report_path`,
`profile_report_paths[]`, `torch_trace_paths[]`,
`system_profile_paths[]`, `server_log_paths[]`, and a
`gpu_monitor_aggregate` summary.

Paths are session-dir relative when the producer can express them
that way; absolute otherwise. Consumers that need to pull raw
artifacts (for example, for a replay) should resolve relative paths against
`session.session_dir`.

---

## `attribution`

Gain attribution per stack entry: a list of `StackGainEntry`
(per-validation incremental contribution) plus a `SourceBreakdown`
that splits the validated total across geak / forge / explore / sweep,
with legacy alias buckets populated only when the source session
contains archived action names.

`method` is one of `validated`, `single_source`, `reconstructed`,
`missing` — consumers should display reconstruction caveats from the
`notes[]` field.

---

## `source_files` — `SourceFiles`

Pointers to the raw artifacts the breakdown was built from
(manifest, state, baseline_report, profile_reports[], …). Use this
when you need to drop into the raw session artifacts for deeper
investigation than the breakdown summarises.

---

## Worked example

The following example shows a complete `session_breakdown.json` for a finished GLM-5 session.

```text
{
  "schema_version": "hyperloom.session_breakdown.v2",
  "exported_at_utc": "2026-05-17T14:02:15.001Z",
  "exporter_version": "session-breakdown-1.0.0",

  "session": {
    "session_id": "sess-20260517-1130",
    "claw_session_id": "claw-abc123",
    "sandbox_user_id": "user-42",
    "created_at_utc": "2026-05-17T11:30:00Z",
    "ended_at_utc": "2026-05-17T13:58:42Z",
    "stop_reason": "target_reached",
    "max_minutes": 240,
    "elapsed_minutes": 148.7,
    "host": "claw-sandbox-7",
    "code_revision": "a1b2c3d",
    "pid": 12345,
    "session_dir": "/workspace/hyperloom/GLM-5-FP8/20260517T113000Z",
    "tick_count": 89,
    "image": "lmsysorg/sglang:v0.5.11-rocm720-mi30x-profilerfix"
  },

  "workload": {
    "framework_name": "sglang",
    "framework_version": "0.5.11",
    "model_name": "GLM-5-FP8",
    "model_path": "/wekafs/models/GLM-5-FP8",
    "model_class": "moe_mla_nsa",
    "gpu_type": "mi355x",
    "tp": 4,
    "conc": 64,
    "isl": 1024,
    "osl": 1024,
    "max_model_len": 8192,
    "precision": "fp8",
    "objective": { "kind": "tput", "value": 500.0 }
  },

  "baseline": {
    "throughput_tok_s_per_gpu": 344.8,
    "accuracy": 0.812,
    "ttft_mean_ms": 142.3,
    "e2el_mean_ms": 2210.5,
    "ttft_e2el_source": "state_workspace",
    "config_path": "runs/baseline/baseline_config.with_envs.yaml",
    "benchmark_report_path": "runs/baseline/report.json",
    "attempts_history": [{
      "ts": "2026-05-17T11:32:10Z",
      "task_id": "t-baseline-1",
      "status": "succeeded",
      "decision": "promoted",
      "key_metric": 344.8,
      "workspace": "runs/baseline",
      "error_class": null
    }],
    "failure_streak": 0,
    "invocation": {
      "framework_args": "python -m sglang.launch_server --model /wekafs/models/GLM-5-FP8 --tp 4",
      "framework_args_source": "log_non_default_args",
      "extra_envs": { "GPU_TYPE": "mi355x", "TP": "4", "ISL": "1024", "OSL": "1024" },
      "config_path": "runs/baseline/baseline_config.with_envs.yaml",
      "server_log_path": "runs/baseline/server.log"
    }
  },

  "final": {
    "throughput_tok_s_per_gpu": 509.4,
    "cumulative_gain_pct_validated": 47.7,
    "cumulative_gain_pct_per_round_sum": 51.2,
    "validated_at_stack_len": 4,
    "validated_ts": "2026-05-17T13:48:01Z",
    "stack_changed_after_validation": false,
    "extra_server_args": "--nsa-decode-backend aiter --enable-mixed-chunk --enable-aiter-allreduce-fusion",
    "extra_envs": {},
    "action_path": [
      "explore:nsa_decode_aiter",
      "explore:mixed_chunk",
      "explore:aiter_allreduce_fusion",
      "kernel_opt:moe_router_gemm_n256_k6144"
    ],
    "ttft_mean_ms": 118.7,
    "e2el_mean_ms": 1604.1,
    "ttft_e2el_source": "current_best",
    "invocation": {
      "framework_args": "python -m sglang.launch_server --model ... --nsa-decode-backend aiter --enable-mixed-chunk --enable-aiter-allreduce-fusion",
      "framework_args_source": "log_non_default_args",
      "extra_envs": { "GPU_TYPE": "mi355x", "TP": "4" },
      "config_path": "runs/explore/final_config.with_envs.yaml",
      "server_log_path": "runs/explore/server.log"
    },
    "closing_phase_entered": true,
    "closing_started_unix": 1747487201.0,
    "closing_report_task_id": "t-close-final"
  },

  "warnings": [],
  "source_files": {
    "manifest": "manifest.json",
    "state": "state.json",
    "baseline_report": "runs/baseline/report.json",
    "profile_reports": ["runs/profile/report.json"],
    "sweep_reports": ["runs/sweep/grid.json"],
    "kernel_attempts": ["kernel-agent/runs/sess-20260517-1130/optimization_attempts.jsonl"],
    "critic_workdir": "critic-workdir",
    "robustness_workdir": "agents/robustness"
  }
}
```

(The remaining sections are elided here for brevity but follow the same
TypedDict shapes.)

---

## Producing the file

* **Live, in-session**: The Coordinator emits the `session_breakdown`
  action and the `cli.py` finally-block as a safety net.
* **Offline and historical**: See
  [Hyperloom operator scripts](operator-scripts.md):
  ```bash
  python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
      --session-dir /path/to/session \
      [--output /tmp/breakdown.json]
  ```

All three paths share the same builder
(`hyperloom.inference_optimizer.breakdown.build`), so the output is identical
regardless of producer.

---

## Stability guarantee

The Hyperloom team commits to the following compatibility guarantees.

1. Never removing or renaming a documented field within a
   major `schema_version`. Such changes require a major bump (the next
   being `v4`) and a one-release deprecation window with both files
   written in parallel. Note `v3.0` is not such a break — it shares
   `v2`'s wire shape and only marks the recorder-aggregation path.
2. Never fabricating values for fields the runtime did not
   actually measure. Missing → null / `[]` / `{}`.
3. Adding new optional fields freely. Consumers must tolerate
   unknown keys.

Consumers may rely on these guarantees for production indexing and
alerting.

## See also

Use the following resources for related reference information.

* [Hyperloom operator scripts](operator-scripts.md) — How to produce a breakdown from a finished session directory.
* [Hyperloom self-hosting and operations guide](operations.md) — Retention recommendations.
* [`src/hyperloom/inference_optimizer/breakdown/schema.py`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/breakdown/schema.py) — TypedDict source of truth.
