# Integration: `session_breakdown.json`

`session_breakdown.json` is the **single external contract** between
the `inference_optimizer` runtime (producer) and any downstream
consumer (`claw-stats-service`, results service, notebooks, custom
dashboards). One file per session, written to
`$USER_DATA_PATH/session_breakdown.json` at session end (and on
operator demand via [`scripts/dump_session_breakdown.py`](OPERATOR_SCRIPTS.md)).

The authoritative source of truth for the wire shape is
[`inference_optimizer/breakdown/schema.py`](../inference_optimizer/breakdown/schema.py).
This page describes the contract from a consumer's perspective.

---

## 1. Versioning

The top-level `schema_version` field is a stable string:

```json
"schema_version": "hyperloom.session_breakdown.v1"
```

Compatibility rules:

* **New optional fields** may appear at any time **without** bumping
  `schema_version`. Consumers must tolerate unknown keys.
* **Renamed, removed, or semantically changed** fields require a major
  bump (`v1` → `v2`). The runtime will continue to write the previous
  version's file in parallel for at least one release after the bump.
* **Missing data** is always represented as `null`, `[]`, or `{}` —
  **never** as a default / fabricated value. Consumers MUST treat
  missing data as "not available".
* All values are JSON-serialisable (no dataclasses, enums, or
  Python-specific types in the wire shape).

The `exporter_version` field carries the producing Hyperloom version
(e.g. `"0.6.0"`) for incident triage and per-version filtering.

---

## 2. Top-level shape

```jsonc
{
  "schema_version": "hyperloom.session_breakdown.v1",
  "exported_at_utc": "2026-05-17T12:34:56.789Z",
  "exporter_version": "0.6.0",

  "session":            { /* §3  SessionMeta */ },
  "workload":           { /* §4  Workload */ },
  "baseline":           { /* §5  Baseline */ },
  "final":              { /* §6  Final state — SaFE contract core */ },
  "phase_timeline":     [ /* §7  PhaseEvent[] */ ],
  "capability_summary": { /* §8  Capability cards */ },
  "geak_invocations":   [ /* §9  Invocation[] */ ],
  "oob_invocations":    [ /* §10 Invocation[] */ ],
  "kernel_lifecycle":   { /* §11 4+1-stage kernel lifecycle */ },
  "param_search":       { /* §12 ParamSearch */ },
  "sweep":              { /* §13 Sweep */ },
  "critic_robustness":  { /* §14 Critic iterations + Robustness signals */ },
  "telemetry":          { /* §15 Telemetry artefact paths */ },
  "attribution":        { /* §16 Gain attribution per stack entry */ },

  "warnings":           [ /* string[] — non-fatal collector warnings */ ],
  "source_files":       { /* §17 SourceFiles — raw artefact paths */ }
}
```

All sections use the `total=False` TypedDict convention — every field
is optional. Consumers should expect partial documents when a session
ended early (`baseline_failed`, `time_exhausted` before kernel-opt
started, …).

---

## 3. `session` — `SessionMeta`

| Field              | Type    | Description                                                                                  |
|--------------------|---------|----------------------------------------------------------------------------------------------|
| `session_id`       | string  | Hyperloom-internal session id (from `manifest.session_id`).                                  |
| `claw_session_id`  | string \| null | Hosted SaFE / Claw id; populated from env `CLAW_SESSION_ID`.                          |
| `sandbox_user_id`  | string \| null | Hosted SaFE user id; populated from env `SANDBOX_USER_ID`.                            |
| `created_at_utc`   | string  | ISO-8601 UTC.                                                                                |
| `ended_at_utc`     | string  | ISO-8601 UTC.                                                                                |
| `stop_reason`      | string  | One of `target_reached`, `time_exhausted`, `no_more_leverage`, `max_ticks`, `baseline_failed`, ... |
| `max_minutes`      | int     | Configured time budget.                                                                       |
| `elapsed_minutes`  | float   | Actual wall-clock.                                                                            |
| `host`             | string  | Hostname of the Coordinator pod.                                                              |
| `code_revision`    | string  | Hyperloom git SHA.                                                                            |
| `pid`              | int     | Coordinator PID.                                                                              |
| `session_dir`      | string  | `$USER_DATA_PATH` for this session.                                                          |
| `tick_count`       | int     | Number of Coordinator ticks.                                                                  |
| `image`            | string \| null | Container image fully-qualified, if configured.                                       |

---

## 4. `workload` — `Workload`

The workload the session optimised: model, framework, GPU type, shape,
precision, and the optimization objective (gain %, target throughput,
baseline-relative, or time-only). See `schema.py::Workload` for the
full field list. Consumers should treat the `objective.kind` enum as
the canonical optimisation goal.

---

## 5. `baseline` — `Baseline`

The starting point Hyperloom measured before any modifications.
Includes throughput, accuracy, optional TTFT / E2EL, the materialised
benchmark config path, attempt history (in case the baseline required
retries), and the `BenchmarkInvocation` record needed to **replay**
the exact baseline benchmark.

`baseline.invocation.framework_args_source` is one of:

* `log_non_default_args` — most authoritative (parsed from the
  vllm/sglang server's own arg echo).
* `log_args_line` — `Args: Namespace(...)` header.
* `log_python_cmd` — literal `python …` launch line scraped from logs.
* `yaml_cmd` — `cmd:` / `command:` / `launch:` field in the
  materialised config YAML.
* `yaml_benchmark` — synthesised from Magpie's `benchmark.*` YAML
  fields.
* `unknown` — none of the above; a warning is appended to
  top-level `warnings`.

`extra_envs` is **allowlist-filtered** to keep secrets out of the
breakdown. Do not assume it contains every env var the session ran with.

---

## 6. `final` — `Final` (SaFE contract core)

The end-state Hyperloom validated. The two most important fields for
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

> **Consumer best practice:** index on
> `(session.session_id, final.throughput_tok_s_per_gpu,
> final.cumulative_gain_pct_validated, workload.model_name,
> workload.gpu_type)`. Everything else is detail.

---

## 7. `phase_timeline` — `PhaseEvent[]`

Chronologically ordered events, one per Coordinator action completion.
Each entry has `action`, `task_id`, `status`, `decision`,
`key_metric`, optional `kernel_id` (for kernel-owned actions),
optional `workspace`, and an `extras` dict for action-specific payload.

Useful for rendering session-progress timelines and "what changed at
T+90 min" charts.

---

## 8. `capability_summary` — `CapabilitySummary`

One card per capability (`geak`, `oob`, `backends`, `params`, `sweep`,
`validate_stack`) with: `status`, `attempts`, `keeps`, `tested`,
`best_gain_pct`, `reason`. Drives the per-session UI cards in
PrimusClaw.

---

## 9–10. `geak_invocations` / `oob_invocations` — `Invocation[]`

Same shape; `backend` distinguishes (`geak` / `claude` / `codex` /
`cursor`). One entry per attempt-on-a-kernel. The
`decision` enum is `KEEP` / `PARTIAL` / `REVERT` / `FAILED`.

---

## 11. `kernel_lifecycle` — `KernelLifecycle`

The 4+1-stage kernel pipeline:

* `detected` — TraceLens-identified hot kernels.
* `recommended` — Critic-filtered candidates with backend
  recommendations.
* `optimized` — kernels with at least one completed backend attempt
  and `best_micro_speedup`.
* `adopted` — kernels promoted into the final stack (E2E-validated).
* `rejected` — kernels considered then dropped, with `reason`.

The same `kernel_id` appears in multiple lists as it progresses.

---

## 12. `param_search`

Two ledgers (`params`, `backends`) of `ParamSearchEntry` records:
every tested variant with `status` ∈ `accepted` / `rejected` /
`tested`, the `extra_server_args` / `extra_envs` it injected, the
`output_throughput` it measured, and the resulting `gain_pct`. Also
includes `synergy_attempted`, `discovered_flags`, and
`backend_winners_history`.

---

## 13. `sweep`

Final concurrency / ISL / OSL sweep. Always includes `all_variants`
(a `SweepPoint[]`) and `best_overall`. `best_for_each_conc` and
`pareto_front` are populated when the sweep grid is large enough.

---

## 14. `critic_robustness`

Decision-review trail: every Critic iteration (verdict + paths to
request / judge_bundle / emit / review JSONs), plus every Robustness
signal (`crash` / `stall` / `disk_full` / `cluster_fault` / …).

---

## 15. `telemetry`

Paths only (no copied content): `baseline_report_path`,
`profile_report_paths[]`, `torch_trace_paths[]`,
`system_profile_paths[]`, `server_log_paths[]`, and a
`gpu_monitor_aggregate` summary.

Paths are **session-dir relative** when the producer can express them
that way; absolute otherwise. Consumers that need to pull raw
artefacts (e.g. for a replay) should resolve relative paths against
`session.session_dir`.

---

## 16. `attribution`

Gain attribution per stack entry: a list of `StackGainEntry`
(per-validation incremental contribution) plus a `SourceBreakdown`
that splits the validated total across geak / oob / backends / params
/ sweep.

`method` is one of `validated`, `single_source`, `reconstructed`,
`missing` — consumers should display reconstruction caveats from the
`notes[]` field.

---

## 17. `source_files` — `SourceFiles`

Pointers to the raw artefacts the breakdown was built from
(manifest, state, baseline_report, profile_reports[], …). Use this
when you need to drop into the raw session artefacts for deeper
investigation than the breakdown summarises.

---

## 18. Worked example

```jsonc
{
  "schema_version": "hyperloom.session_breakdown.v1",
  "exported_at_utc": "2026-05-17T14:02:15.001Z",
  "exporter_version": "0.6.0",

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
    "session_dir": "/workspace/hyperloom",
    "tick_count": 89,
    "image": "lmsysorg/sglang:v0.5.11-rocm720-mi30x"
  },

  "workload": {
    "framework": "sglang",
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
      "backends:nsa_decode_aiter",
      "backends:mixed_chunk",
      "backends:aiter_allreduce_fusion",
      "kernel_opt:moe_router_gemm_n256_k6144"
    ],
    "ttft_mean_ms": 118.7,
    "e2el_mean_ms": 1604.1,
    "ttft_e2el_source": "current_best",
    "invocation": {
      "framework_args": "python -m sglang.launch_server --model ... --nsa-decode-backend aiter --enable-mixed-chunk --enable-aiter-allreduce-fusion",
      "framework_args_source": "log_non_default_args",
      "extra_envs": { "GPU_TYPE": "mi355x", "TP": "4" },
      "config_path": "runs/validate_stack/final_config.with_envs.yaml",
      "server_log_path": "runs/validate_stack/server.log"
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
    "kernel_attempts": ["agents/kernel/runs/sess-20260517-1130/optimization_attempts.jsonl"],
    "critic_workdir": "critic-workdir",
    "robustness_workdir": "agents/robustness"
  }
}
```

(Sections §7–§16 are elided here for brevity but follow the same
TypedDict shapes.)

---

## 19. Producing the file

* **Live, in-session:** the Coordinator emits the `session_breakdown`
  action and the `cli.py` finally-block as a safety net.
* **Offline / historical:** see
  [`OPERATOR_SCRIPTS.md`](OPERATOR_SCRIPTS.md):
  ```bash
  python -m inference_optimizer.scripts.dump_session_breakdown \
      --session-dir /path/to/session \
      [--output /tmp/breakdown.json]
  ```

All three paths share the same builder
(`inference_optimizer.breakdown.build`), so the output is identical
regardless of producer.

---

## 20. Stability guarantee

The Hyperloom team commits to:

1. Never **removing** or **renaming** a documented field within a
   major `schema_version`. Such changes require a `v2` bump and a
   one-release deprecation window with both files written in parallel.
2. Never **fabricating** values for fields the runtime did not
   actually measure. Missing → null / `[]` / `{}`.
3. Adding new **optional** fields freely. Consumers must tolerate
   unknown keys.

Consumers may rely on these guarantees for production indexing and
alerting.

## See also

* [`OPERATOR_SCRIPTS.md`](OPERATOR_SCRIPTS.md) — how to produce a
  breakdown from a finished session directory.
* [`OPERATIONS.md`](OPERATIONS.md) §3 — retention recommendations.
* [`../inference_optimizer/breakdown/schema.py`](../inference_optimizer/breakdown/schema.py)
  — TypedDict source of truth.
