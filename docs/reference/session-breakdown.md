---
myst:
    html_meta:
        "description": "Reference for the session_breakdown.json contract in Hyperloom. Covers schema versioning, top-level shape, all sections, a worked example, and stability guarantees."
        "keywords": "Hyperloom, session_breakdown.json, schema, API contract, LLM inference, AMD GPU, ROCm, session data, downstream integration, versioning, telemetry, observability"
---
# `session_breakdown.json` integration in Hyperloom

```{note}
This page is for **integrators and downstream consumers** — teams building
dashboards, reporting pipelines, or services
that read Hyperloom session output programmatically. If you just ran an
optimization and want to check your results, read the three headline fields
described in [Run a Hyperloom optimization](../how-to/optimize.md#output-and-artifacts)
first.
```

`session_breakdown.json` is the single external contract between
the `inference_optimizer` runtime (producer) and any downstream
consumer (results service, notebooks, custom
dashboards). One file per session, written to
`$SESSION_DIR/session_breakdown.json` at session end (and on
operator demand using [`dump_session_breakdown.py`](operator-scripts.md)).

The authoritative source of truth for the wire shape is
[`src/hyperloom/inference_optimizer/breakdown/schema.py`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/breakdown/schema.py).
This page describes the contract from a consumer's perspective.

---

## Versioning

The top-level `schema_version` field is a stable string. New exports use the
recorded-timeline wire shape:

```json
"schema_version": "hyperloom.session_breakdown.v6.1"
```

6.1 changes three things inside the 6 major, an exception to the compatibility
rules below: it renames one key, removes keys no producer populated or that
only carried a constant, and reports two unmeasured figures as null. `throughput_tok_s_per_gpu` in
`outcome.baseline`, `outcome.final` and the `baseline` event's `measurement` is
now `output_throughput`, and the old key is no longer written. The value is
unchanged: the whole-server output throughput, not a per-GPU figure. Archived
6.0 documents and documents exported by 1.1.x carry the old key, so readers must
accept both keys or migrate. Within 6.1, `perf.output_tput_per_gpu` also carries
its value on the `baseline` event and on stack validation rows.

6.1 also removes keys that no producer ever populated. Each carries the
disposition `removed in 6.1, never populated`:

- `framework_agent` `policy.force_exit_budget_pct`, `policy.variant_timeout_sec`
  and `policy.overtime_kill_ratio`
- `kernel_agent` `trace_analyze_runs[].trace_validate_ref`
- `critic_review.iteration` on framework proposals
- `kernel_agent` `geak.product.cfg_hash` and `geak.product.final_overlay_digest`
- `kernel_agent` `entry.budget_remaining_sec`
- `roofline` event `ext.actions[].kernel_roofline.kernels[].rocprof_roofline`

The `event_id` and `ordinal` row-scope declarations are removed from the schema
types: they were already stripped before export and never appeared on the wire.
`conc_sweep` `points[].budget_remaining_sec` is unchanged.

Two keys that only ever carried a constant are removed with the disposition
`removed in 6.1, constant`: `roofline` event
`ext.actions[].kernel_roofline.kernels[].roofline_measured` (always `false`;
`roofline_source` states whether a row's bound is analytical or a placeholder)
and `baseline` event `ext.actions[].convergence` (always `insufficient_rounds`).

Two figures are null instead of a fabricated default when they were not
measured or not provided. `outcome.validation.attributed_gain_pct` and
`unattributed_gain_pct` are null when no adoption could be measured, rather than
`0.0`, and `empty` on `framework_agent` runs is null when the producer did not
report it, rather than `false`. `kernel_agent` `entry.snapshot_staleness` is the
string `"absent"` or `"fresh"`.

Re-exporting a 6.0 session directory reads its old `baseline`
event, so `outcome.baseline.output_throughput` is null while `outcome.final` and
the gain keep their values, and `metadata.workflow.contract_digest` keeps the
digest recorded at session creation rather than the one pinned by the packaged
workflow-evaluation schema. The default output path overwrites the archived
`<session_dir>/session_breakdown.json`; pass `--output` to keep it.

V6 was the preceding cutover, for the timeline: each action records its own event
while it runs, so an event's `start_time` is when the work began rather than
when its artefacts were written, and the KERNEL and BASELINE projections are no
longer emitted. Consumers that ordered events around the old collapsed windows
see a different ordering.

The round ledger arrives inside v6, as an added section of `enablement` rather
than a new version: the round lifecycle is read from the durable round ledger
in `storage/coordinator.db` — the only record that outlives the process that
took a round — and is reported through `rounds[]` and its counters. The three
state-sourced fields that ledger replaces are no longer emitted; each one's
disposition is under
[`enablement`](#enablement--admission-round-lifecycle-builds--attempt-runtimes).
That block reports on the runtime a session built rather than on its results,
and is emitted as `{}` on every session that ran no enablement, so it has never
carried a field a consumer could gate on.

V5 was the preceding cutover, for optimization results: consumers read only
`optimizations`; the old `optimization_stack`, attribution, GEAK invocation,
Forge invocation, and GEMM-tuning result projections are no longer emitted.
Archived V2/V3/V4/V5 documents require a downstream migration before V6 readers
consume them.

Compatibility rules:

* **Parse the version, do not gate on string equality**. Read the
  `vN[.M]` prefix and compare the major component so a future minor
  revision of V6 is still accepted.
* **New optional fields** might appear at any time without bumping
  the major version. Consumers must tolerate unknown keys.
* **Renamed, removed, or semantically changed** fields of a result section
  require a major bump; `enablement` is outside that rule, as the stability
  guarantee below records, and so are the 6.1 changes declared above: the
  rename of `throughput_tok_s_per_gpu` to `output_throughput`, the removal of
  keys no producer populated, and the null reporting of unmeasured
  `attributed_gain_pct`, `unattributed_gain_pct` and run `empty`. Only one
  version is written per session; there is no parallel write of the previous
  version's file.
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
  "schema_version": "hyperloom.session_breakdown.v6.1",
  "exported_at_utc": "2026-05-17T12:34:56.789Z",
  "exporter_version": "session-breakdown-1.0.0",

  "metadata":           { /* §3  Session identity, launch config, versions, Langfuse, warnings */ },
  "outcome":            { /* terminal result — SaFE contract core */ },
  "timeline":           [ /* the run itself: one event per stage, in order */ ],
  "close":              { /* what the session settled at close */ },
  "critic":             { /* the critic agent's own run, iteration by iteration */ }
}
```

Those eight keys are the whole of the export. Every key is always present; a
section the run produced nothing for is `{}` or `[]` rather than absent.

Two things a reader of an older export will look for and not find. The flat
per-topic sections (`baseline`, `final`, `phase_timeline`,
`capability_summary`, `kernel_lifecycle`, `param_search`, `geak`,
`telemetry`, `optimizations`, `source_files` and the optional tail) are gone:
each was a projection of the run rather than a fact of it, and they now come
out of `timeline`, whose events carry the same facts attached to the stage
that produced them. The old combined `critic_robustness` key is absent. Critic
activity lives under `critic`.

How the export itself went is reported once, on `metadata.warnings`. An
earlier shape also carried a top-level `warnings`, taken partway through the
export; it was a strict subset and so disagreed with `metadata.warnings`
about the same export.

The V6 surface is identified by
`metadata.versions.schema_version = "hyperloom.session_breakdown.v6.1"`. Startup
source events are stored in execution order under `reports/sbd_v6/timeline/`;
writer failures are reported through `metadata.warnings` rather than being
indistinguishable from a stage that never ran.

All sections use the `total=False` TypedDict convention — every field
is optional. Consumers should expect partial documents when a session
ended early (`baseline_failed`, `time_exhausted` before kernel-opt
started, …).

---

## `metadata` — `V6Metadata`

Task identity, recorded as each fact is decided rather than re-derived at
export. Five blocks: `session`, `task_config`, `grading`, `versions` and
`langfuse`, plus the export's own `exported_at_utc` and `warnings`.

`metadata.session` — identity and lifecycle:

| Field              | Type    | Description                                                                                  |
|--------------------|---------|----------------------------------------------------------------------------------------------|
| `session_id`       | string  | Hyperloom-internal session id (from `manifest.session_id`).                                  |
| `claw_session_id`  | string \| null | Hosted SaFE / Claw id; populated from env `CLAW_SESSION_ID`.                          |
| `sandbox_user_id`  | string \| null | Hosted SaFE user id; populated from env `SANDBOX_USER_ID`.                            |
| `created_at_utc`   | string  | ISO-8601 UTC.                                                                                |
| `start_ts`         | string  | The anchor `--max-hours` is counted from; a resume may re-anchor it.                         |
| `ended_at_utc`     | string  | ISO-8601 UTC; empty while the session is still running.                                      |
| `max_minutes`      | int     | Configured time budget.                                                                       |
| `elapsed_minutes`  | float   | Actual wall-clock, measured from `start_ts` to the recorded end (or to now).                 |
| `host`             | string  | Hostname of the Coordinator pod.                                                              |
| `code_revision`    | string  | Hyperloom git SHA.                                                                            |
| `pid`              | int     | Coordinator PID.                                                                              |
| `session_dir`      | string  | Concrete session directory, typically `$USER_DATA_PATH/<model_basename>/<timestamp>/`.       |
| `user_data_path`   | string  | The operator-chosen workspace base.                                                           |
| `tick_count`       | int     | Number of Coordinator ticks.                                                                  |
| `image`            | string \| null | Container image fully-qualified, if configured.                                       |
| `image_id`         | string \| null | The image reference without its registry path.                                        |
| `recovery`         | object  | Crash / interruption / resume history: `recovered`, `crash_count`, `crash_timestamps`, `degraded_mode`, `resume_pending_revalidation` (true while the optimization stack holds entries no validation has covered yet), `last_tick_exception`. |

Why the run ended is an outcome rather than an identity, and lives on
`outcome.stop_reason`.

`metadata.task_config` — the workload the session optimised: model, framework,
GPU type, shape, precision, launch overrides, and the optimization objective
(gain %, target throughput, baseline-relative, or time-only). Consumers should
treat the `objective.kind` enum as the canonical optimisation goal. Its
`architecture` sub-object is the structural model summary parsed from the
model's own `config.json`, and is empty on non-transformers models. Its
`launch_env` keeps every operator-supplied variable name, but a credential's
value (an API key, secret, password, passphrase, private or access key, auth,
signature or header variable, or a name with a `TOKEN` segment) is written as
`[REDACTED]`. Tuning knobs such as `MAX_NUM_BATCHED_TOKENS` or `*_PER_TOKEN`
keep their values, with any recognizable credential inside a value masked.
Replaying a session takes credentials from its own environment, never from the
breakdown.

`metadata.grading` — the grading mode this session was started in:
`benchmark_mode` (`agentx` or `synthetic`), `objective`, and the `tput_guard`
that rides along with the interactivity objective (`enabled`, `noise_pct`).

An AgentX replay is ranked on the median interactivity percentile
(`e2e_norm_intvty_p50`) with the slow tail (`e2e_norm_intvty_p90`) and output
throughput each held as a guard; a synthetic run
is ranked on output throughput alone. Every throughput field elsewhere in this
document is the output axis by construction, so without this block a consumer
cannot tell the two kinds of session apart — and on the canonical corpus the
two axes differ by roughly two orders of magnitude.

`grading.objective` is a mode marker written when the session starts: an
interactivity session always carries `e2e_norm_intvty_p90`, which is not the
axis promotions are decided on. The deciding axis is
`outcome.validation.graded_on` (and `outcome.final.graded_on`), read off the
promotion itself (`e2e_norm_intvty_p50` on an AgentX replay). A session that
adopted nothing has no `graded_on`; one that adopted but never validated
reports the adoption axis.

`tput_guard.noise_pct` is null on a session that predates the band being
recorded.

`metadata.versions` — the schema version, the Hyperloom revision, the framework
and its version, and a `tools` map carrying `{tool, root_dir, commit, version}`
per external tool.

`metadata.langfuse` — the live-Langfuse entrypoint: `enabled`,
`disabled_reason`, `trace_id`, `session_id`, `trace_url` and push `counts`. The
local trace jsonl is always written regardless.

---

## `outcome.baseline` — `Baseline`

The starting point Hyperloom measured before any modifications.
Includes throughput, accuracy, optional time to first token (TTFT) and end-to-end latency (E2EL), the materialised
benchmark config path, attempt history (in case the baseline required
retries), and the `BenchmarkInvocation` record needed to **replay**
the exact baseline benchmark.

Disposition: `baseline.throughput_tok_s_per_gpu` (6.0) → `baseline.output_throughput` (6.1).

`baseline.invocation.framework_args_source` is one of:

* `log_non_default_args`: Most authoritative (parsed from the
  vllm/sglang server's own arg echo).
* `log_args_line`: `Args: Namespace(...)` header.
* `log_python_cmd`: Literal `python …` launch line scraped from logs.
* `yaml_cmd`: `cmd:` / `command:` / `launch:` field in the
  materialised config YAML.
* `yaml_benchmark`: Synthesised from Magpie's `benchmark.*` YAML
  fields.
* `unknown`: None of the above; a warning is appended to
  `metadata.warnings`.

`extra_envs` is allowlist-filtered to keep secrets out of the
breakdown. Do not assume it contains every env var the session ran with.

`baseline.perf` and `final.perf` carry the AgentX axes the measurement
reported, each an explicit `null` where nothing measured it. Absent would
be indistinguishable from an axis the framework failed to report, and zero
reads as "measured, and it was zero", so the AgentX-only axes stay `null`
in a synthetic run, while `output_tput_per_gpu` carries a value whenever the
tensor-parallel degree is known. The set is
`common/perf_metric.py:GRADED_AXIS_KEYS`, which both publishing projections
read, and it is grouped as:

* the objective and its two guards — `e2e_norm_intvty_p50`,
  `e2e_norm_intvty_p90`, `output_tput_per_gpu`;
* the comparability inputs a candidate/anchor pair is refused on —
  `duration_seconds`, `request_error_rate`. A pair is graded only when both
  replayed a window of the same length and the candidate's error rate exceeds
  its anchor's by no more than `ERROR_RATE_SLACK_PCT` percentage points, so a
  verdict published without them could not be re-derived from the record;
* the latency detail — `ttft_p50_ms`, `ttft_p90_ms`, `tpot_p50_ms`,
  `tpot_p90_ms`;
* reported for continuity and part of no verdict — `total_throughput`,
  `input_throughput`.

`baseline.submission_valid` is upstream's own verdict on whether the round was
a submittable measurement at all. Tri-state: `null` means the framework never
answered, which is not the same fact as it answering no, and a reader weighing
any axis above needs to know the round it came from was admissible. The reasons
behind a `false` travel with it as `baseline.submission_invalid_reasons`, and
on the timeline as `submission_invalid_reasons` on the baseline round's
`measurement`.

`final.graded_on` names the axis `final.gain_pct` is on, and always agrees with
`outcome.validation.graded_on`: they are the same figure read twice.

---

## `outcome.final` — `Final` (SaFE contract core)

The end-state Hyperloom validated against the SaFE (Safe and Fast Execution) contract. The two most important fields for
downstream consumers:

| Field                              | Meaning                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------|
| `output_throughput`                | Validated end-of-session output throughput. The headline number. See the scope note below. |
| `cumulative_gain_pct_validated`    | Validated cumulative gain vs `baseline.output_throughput`. The headline %.                   |
| `action_path`                      | Ordered list of `action:variant` labels that made the final stack — the recipe.            |
| `extra_server_args`                | The exact extra args needed to reproduce the final config.                                 |
| `extra_envs`                       | The exact env overrides needed to reproduce the final config (allowlisted, no secrets).    |
| `invocation`                       | Same shape as `baseline.invocation`; lets a consumer replay the final benchmark.          |
| `closing_phase_entered`            | True iff Coordinator entered the closing phase cleanly (vs SIGTERM exit).                  |

> **Scope of `output_throughput`: whole-server total in `throughput_unit`, not per-GPU.**
> Do not divide it by a GPU count.
> `cumulative_gain_pct_validated` is a ratio of two such numbers and is unaffected.
> Disposition: `throughput_tok_s_per_gpu` (6.0) → `output_throughput` (6.1).

> Consumer best practice: index on
> `(session.session_id, final.output_throughput,
> final.cumulative_gain_pct_validated, workload.model_name,
> workload.gpu_type)`. Everything else is detail.

---

## `outcome`, `timeline` and `close`

`outcome` is the terminal result: `status`, `stop_reason`, `stage_reached`,
the `baseline` and `final` blocks documented above, and the `validation`
block that reconciles the optimization stack's parts against its total. That
reconciliation is single-axis and `validation.graded_on` names the axis: an
attributed figure on one axis against an unattributed figure on another makes
the gap meaningless. Every adoption in that sum is on the named axis by
construction — a comparison that could not supply the configured axis pair
failed instead of being graded on another one.

`outcome.variants` counts what the framework phase benched: `bench_runs`
configuration variants and source patches, of which `kept` were adopted into
the stack, `reverted` were measured and not adopted, and `failed` reached the
bench without a measurement; the `distinct` changes among them; and
`keep_rate_pct`, kept per bench run. Each attempt row says whether it reached
the bench (`benched`), so a patch that never applied is not counted, and a
change benched again on a later stack is one distinct change and two bench
runs. Kernel candidates are not variants: the kernel lane reports its own
funnel.

`timeline` is the run itself — one event per stage, oldest first. An event
carries its `type`, its identity (`event_id`, `phase`, `macro_cycle`), its
span, its `status`, and an `ext` block holding what that kind of stage
records. This is where the facts the older flat sections projected now live,
attached to the stage that produced them.

`close` is what the session settled at close: the `steps` the close sequencer
ran and the `artifacts` it published.

`V6Outcome`, `V6TimelineEvent` and `V6Close` in
`src/hyperloom/inference_optimizer/breakdown/schema.py` are the authority on
the fields of each; they are typed and versioned with the export.

---

## `critic`

The critic agent's own run, iteration by iteration. The per-proposal verdicts
are not here — those stay with the proposals they judge — so this key answers
a different question: how often the agent was asked, about what, and how its
rulings fell each time.

| Field | Type | Meaning |
|---|---|---|
| `iterations` | list | One row per review pass, in the order the agent ran them |

Each iteration carries `iter`, `ts`, `phase`, `macro_cycle`, the `topic` it
spoke about and the `summary` it wrote, plus the four artifacts it left
behind (`request_path`, `judge_bundle_path`, `emit_path`, `review_path`).

An iteration rules on every proposal in front of it, so it has no single
verdict. `verdict_counts` is the distribution of its rulings and `verdict`
reads the same thing as one line (`2 approve, 1 reject`). A pass that only
spoke — a heartbeat, a request for context — rules on nothing and leaves both
empty; such a pass is still reported, because a session where the critic was
asked forty times and ruled on nothing reads very differently from one where
it was never asked.

`framework_reviews` holds the framework-phase rulings of that pass, each with
both the authored `verdict` and the `effective_verdict` the loop held it to.
Both are kept: a reject the loop downgraded to advice still ran, and
reporting either alone misreads the round.

---

## Worked example

The following example shows a complete `session_breakdown.json` for a finished GLM-5 session.

```text
{
  "schema_version": "hyperloom.session_breakdown.v6.1",
  "exported_at_utc": "2026-05-17T14:02:15.001Z",
  "exporter_version": "session-breakdown-1.0.0",

  "metadata": {
    "exported_at_utc": "2026-05-17T14:02:15.001Z",
    "versions": {
      "schema_version": "hyperloom.session_breakdown.v6.1",
      "hyperloom": "a1b2c3d",
      "framework": "sglang",
      "framework_version": "0.5.21",
      "tools": {
        "geak": { "tool": "geak", "root_dir": "/opt/geak", "commit": "9f8e7d6", "version": "0.4.2" }
      }
    },
    "session": {
      "session_id": "sess-20260517-1130",
      "claw_session_id": "claw-abc123",
      "sandbox_user_id": "user-42",
      "created_at_utc": "2026-05-17T11:30:00Z",
      "start_ts": "2026-05-17T11:30:00Z",
      "ended_at_utc": "2026-05-17T13:58:42Z",
      "max_minutes": 240,
      "elapsed_minutes": 148.7,
      "host": "claw-sandbox-7",
      "code_revision": "a1b2c3d",
      "pid": 12345,
      "session_dir": "/workspace/hyperloom/GLM-5-FP8/20260517T113000Z",
      "user_data_path": "/workspace",
      "tick_count": 89,
      "image": "lmsysorg/sglang-rocm:v0.5.21-rocm10-mi30x-20261008",
      "image_id": "sglang-rocm:v0.5.21-rocm10-mi30x-20261008",
      "recovery": {
        "recovered": false,
        "crash_count": 0,
        "crash_timestamps": [],
        "degraded_mode": false,
        "resume_pending_revalidation": false,
        "last_tick_exception": null
      }
    },
    "task_config": {
      "framework_name": "sglang",
      "framework_version": "0.5.21",
      "model_name": "GLM-5-FP8",
      "model_path": "/models/GLM-5-FP8",
      "gpu_type": "mi355x",
      "tp": 4,
      "conc": 64,
      "isl": 1024,
      "osl": 1024,
      "max_model_len": 8192,
      "precision": "fp8",
      "objective": { "kind": "tput", "value": 150.0 },
      "launch_env": {},
      "launch_server_args": "",
      "architecture": { "model_class": "moe_mla_nsa", "model_type": "glm5", "is_moe": true }
    },
    "grading": {
      "benchmark_mode": "synthetic",
      "objective": "output_throughput",
      "tput_guard": { "enabled": false, "noise_pct": 5.0 }
    },
    "langfuse": { "enabled": false, "disabled_reason": "no_credentials", "trace_url": null, "counts": {} },
    "warnings": []
  },

  "outcome": {
    "status": "succeeded",
    "stop_reason": "target_reached",
    "stage_reached": "CLOSE",

    "baseline": {
        "output_throughput": 100.0,
      "accuracy": 0.812,
      "ttft_mean_ms": 0.0,
      "e2el_mean_ms": 0.0,
      "perf": {
        "e2e_norm_intvty_p50": null,
        "e2e_norm_intvty_p90": null,
        "output_tput_per_gpu": 25.0,
        "duration_seconds": null,
        "request_error_rate": null,
        "ttft_p50_ms": null,
        "ttft_p90_ms": null,
        "tpot_p50_ms": null,
        "tpot_p90_ms": null,
        "total_throughput": null,
        "input_throughput": null
      },
      "submission_valid": null,
      "submission_invalid_reasons": [],
      "ttft_e2el_source": "state_workspace",
      "config_path": "runs/baseline/baseline_config.with_envs.yaml",
      "benchmark_report_path": "runs/baseline/report.json",
      "attempts_history": [{
        "ts": "2026-05-17T11:32:10Z",
        "task_id": "t-baseline-1",
        "status": "succeeded",
        "decision": "promoted",
        "key_metric": 100.0,
        "workspace": "runs/baseline",
        "error_class": null
      }],
      "failure_streak": 0,
      "invocation": {
        "framework_args": "python -m sglang.launch_server --model /models/GLM-5-FP8 --tp 4",
        "framework_args_source": "log_non_default_args",
        "extra_envs": { "GPU_TYPE": "mi355x", "TP": "4", "ISL": "1024", "OSL": "1024" },
        "config_path": "runs/baseline/baseline_config.with_envs.yaml",
        "server_log_path": "runs/baseline/server.log"
      }
    },

    "final": {
      "output_throughput": 150.0,
      "graded_on": "output_throughput",
      "perf": {
        "e2e_norm_intvty_p50": null,
        "e2e_norm_intvty_p90": null,
        "output_tput_per_gpu": 37.5,
        "duration_seconds": null,
        "request_error_rate": null,
        "ttft_p50_ms": null,
        "ttft_p90_ms": null,
        "tpot_p50_ms": null,
        "tpot_p90_ms": null,
        "total_throughput": null,
        "input_throughput": null
      },
      "cumulative_gain_pct_validated": 50.0,
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
      "ttft_mean_ms": 0.0,
      "e2el_mean_ms": 0.0,
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

    "variants": { "bench_runs": 12, "distinct": 10, "kept": 3, "reverted": 8, "failed": 1, "keep_rate_pct": 25.0 }
  },

  "timeline": [
    {
      "event_id": "framework_agent:2:phase",
      "type": "phase",
      "phase": "FRAMEWORK_AGENT",
      "macro_cycle": 2,
      "start_time": "2026-05-17T12:10:00Z",
      "end_time": "2026-05-17T12:41:33Z",
      "status": "succeeded",
      "ext": { "actions": [], "proposals": [] }
    }
  ],

  "close": {
    "status": "succeeded",
    "close_sequence_done": true,
    "start_time": "2026-05-17T13:48:10Z",
    "end_time": "2026-05-17T13:58:42Z",
    "steps": [],
    "artifacts": {}
  },

  "critic": {
    "iterations": [
      {
        "iteration_id": "critic-iteration:7:9f8e7d6c",
        "iter": 7,
        "ts": "2026-05-17T12:38:02Z",
        "phase": "FRAMEWORK_AGENT",
        "macro_cycle": 2,
        "topic": "backends:nsa_decode_aiter",
        "verdict": "2 approve, 1 reject",
        "verdict_counts": { "approve": 2, "reject": 1 },
        "summary": "the decode backend pays for itself; the allreduce change needs a measurement first",
        "request_path": "critic-workdir/000007/request.json",
        "judge_bundle_path": "critic-workdir/000007/judge_bundle.json",
        "emit_path": "critic-workdir/000007/emit.json",
        "review_path": "critic-workdir/000007/review.json",
        "framework_reviews": [
          {
            "proposal_msg_id": "msg-4f21",
            "arm": "config",
            "verdict": "reject",
            "effective_verdict": "advise",
            "reasoning": "no measurement backs the projected gain"
          }
        ]
      }
    ]
  }
}
```

(Rows are elided here for brevity but follow the same TypedDict shapes.)

---

## Producing the file

* **Live, in-session**: The Coordinator emits the `session_breakdown`
  action and the `cli.py` finally-block as a safety net.
* **Offline and historical**: See
  [Hyperloom operator scripts](operator-scripts.md):
  ```bash
  python -m hyperloom session breakdown \
      --session-dir /path/to/session \
      [--output /tmp/breakdown.json]
  ```

All three paths share the same builder
(`hyperloom.inference_optimizer.breakdown.build`), so the output is identical
regardless of producer.

---

## Stability guarantee

The Hyperloom team commits to the following compatibility guarantees.

1. Never removing or renaming a documented field of a result section within
   a major `schema_version`. Such changes require a major bump, as the `v5.0`
   optimization cutover did, and every removed or renamed field is given a
   disposition in the section it left. The 6.1 changes (the
   rename of `throughput_tok_s_per_gpu` to `output_throughput`, the removal of
   keys no producer populated, and null reporting of unmeasured
   `attributed_gain_pct`, `unattributed_gain_pct` and run `empty`) are a
   declared exception to this rule, recorded under Versioning. `enablement` is outside this guarantee:
   it reports on the runtime a session built rather than on its results, is
   emitted as `{}` whenever no enablement ran, and its fields move with the
   runtime they describe — every field the round ledger replaced is still given
   a disposition in that section.
2. Never fabricating values for fields the runtime did not
   actually measure. Missing → null / `[]` / `{}`.
3. Adding new optional fields freely. Consumers must tolerate
   unknown keys.

Consumers can rely on these guarantees for production indexing and
alerting.

## Related topics

Use the following resources for related reference information.

* [Hyperloom operator scripts](operator-scripts.md): How to produce a breakdown from a finished session directory.
* [Hyperloom self-hosting and operations guide](operations.md): Retention recommendations.
* [`src/hyperloom/inference_optimizer/breakdown/schema.py`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/breakdown/schema.py): TypedDict source of truth.
