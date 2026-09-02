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
unified optimization wire shape:

```json
"schema_version": "hyperloom.session_breakdown.v5.0"
```

V5 is a breaking cutover for optimization results: consumers read only
`optimizations`; the old `optimization_stack`, attribution, GEAK invocation,
Forge invocation, and GEMM-tuning result projections are no longer emitted.
Archived V2/V3/V4 documents require a downstream migration before V5 readers
consume them.

Compatibility rules:

* **Parse the version, do not gate on string equality**. Read the
  `vN[.M]` prefix and compare the major component so a future minor
  revision of V5 is still accepted.
* **New optional fields** might appear at any time without bumping
  the major version. Consumers must tolerate unknown keys.
* **Renamed, removed, or semantically changed** fields require a major
  bump. Only one version is written per session; there is no parallel
  write of the previous version's file.
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
  "schema_version": "hyperloom.session_breakdown.v5.0",
  "exported_at_utc": "2026-05-17T12:34:56.789Z",
  "exporter_version": "session-breakdown-1.0.0",

  "session":            { /* §3  SessionMeta */ },
  "workload":           { /* §4  Workload */ },
  "baseline":           { /* §5  Baseline */ },
  "final":              { /* §6  Final state — SaFE contract core */ },
  "phase_timeline":     [ /* §7  PhaseEvent[] */ ],
  "capability_summary": { /* §8  Capability cards */ },
  "geak":               { /* GEAK route diagnostics; {} when GEAK never ran */ },
  "kernel_lifecycle":   { /* §11 4+1-stage kernel lifecycle */ },
  "param_search":       { /* §12 ParamSearch */ },
  "critic_robustness":  { /* §14 Critic iterations + Robustness signals */ },
  "telemetry":          { /* §15 Telemetry artefact paths */ },
  "optimizations":      { /* canonical adopted-optimization API */ },
  "metadata":           { /* additive V6 metadata and launch configuration */ },
  "outcome":            { /* additive V6 terminal result */ },
  "timeline":           [ /* additive V6 ordered stage events */ ],
  "close":              { /* additive V6 close-stage result */ },

  "warnings":           [ /* string[] — non-fatal collector warnings */ ],
  "source_files":       { /* §17 SourceFiles — raw artefact paths */ },

  /* Optional sections — present when the run produced the relevant data.
     Consumers MUST tolerate their absence (total=False TypedDict). */
  "model_info":                  { /* model architecture summary */ },
  "phase_segments":              [ /* per-phase segment records */ ],
  "explore_search":              { /* config-arm dedup ledger */ },
  "perfskills":                  { /* perf-skill telemetry */ },
  "specialist_runs":             [ /* specialist sub-agent runs */ ],
  "kernel_roofline":             { /* kernel roofline snapshot */ },
  "kernel_optimization_summary": { /* kernel-opt rollup */ },
  "conc_sweep_summary":          { /* post-run concurrency sweep */ },
  "roofline":                    { /* roofline analysis */ },
  "roofline_progress":           [ /* roofline watermark crossings */ ],
  "decision_trace":              { /* KEEP/REVERT decisions + token rollup */ },
  "token_usage":                 { /* LLM token spend rollup (see below) */ },
  "langfuse":                    { /* Langfuse push receipt */ },
  "kernel_journey":              { /* kernel lifecycle journey */ },
  "collective":                  { /* §11a collective-lane campaigns */ },
  "versions":                    { /* component/version stamps */ },
  "enablement":                  { /* enablement / targeted-build subsystem summary */ }
}
```

The `session` (SessionMeta) section also carries `user_data_path` and a
`recovery` sub-object in addition to the fields documented in §3.

The additive V6 surface is identified by
`metadata.versions.schema_version = "hyperloom.session_breakdown.v6.0"` while
the existing top-level V5 contract remains unchanged. Startup source events are
stored in execution order under `reports/sbd_v6/timeline/`; writer failures are
reported through `metadata.warnings` rather than being indistinguishable from a
stage that never ran.

All sections use the `total=False` TypedDict convention — every field
is optional. Consumers should expect partial documents when a session
ended early (`baseline_failed`, `time_exhausted` before kernel-opt
started, …).

---

## `optimizations` — canonical adopted optimizations

`optimizations` is the only section downstream dashboards need to read for
formally adopted optimization results. It normalizes Warm Replay, Explore,
Framework Agent, and Kernel Agent KEEPs without exposing internal action names
such as `integrate_patch`.

The section is projected from what the producers recorded while they worked —
the operation, adoption, measurement, and artifact streams — and from nothing
else. It is never rebuilt from `state.json`. That is what makes
`available` meaningful: a session whose records never landed reports as
unavailable instead of as a session that optimized nothing.

```text
optimizations
├── schema_version              5
├── source_of_truth             "recorder"
├── available                   bool — always present, on both paths
├── unavailable_reason          string — present only when available=false
├── attempts[]                  every attempt, adopted or not
├── entries[]                   the adopted ledger, in adoption order
├── backend_attempts[]
├── summary_by_agent
├── summary_by_source
├── summary_by_kind
├── validation
└── gemm_tuning_runs[]
```

### `available` — records missing vs nothing adopted

**Consumers must read `available` before reading anything else in this
section.** It is present on both paths: `true` on a normal export, `false`
when the recorder projection could not be built, alongside an
`unavailable_reason` (`"no operations were recorded for this session"` or
`"the recorder projection failed"`). When it is `false`, every array in the
section is empty and `validation.method` is `"unavailable"` — those empty
arrays mean *unknown*, not *none*.

An unavailable section is also cross-checked against `state.json`: if the run
state carries an optimization stack the recorder never captured, the export
says so in `warnings` rather than quietly emitting an empty section.

### `attempts[]` — every attempt, adopted or not

New in V5. One row per recorded unit of optimization work, whichever way it
was decided. `entries[]` covers only what was adopted; `attempts[]` is where a
REVERT, a failure, or a KEEP that nothing credited remains visible. Adopted
entries join back to their attempt through `entries[].adopted_attempt_id`.

| Group | Fields |
|---|---|
| Identity | `attempt_id`, `adoption_id`, `producer`, `kind`, `name`, `subject` (`{type,name}`), `kernel_id`, `backend`, `phase`, `macro_cycle` |
| Timing | `started_at`, `ended_at`, `duration_sec` |
| Ownership | `agent`, `agent_method` |
| Verdict | `status`, `decision`, `decision_source`, `decision_reason`, `adopted`, `integrated`, `validation_basis`, `attribution_eligible`, `keep_threshold_pct`, `keep_threshold_source` |
| Numbers | `local_gain_pct`, `local_gain_source`, `throughput_before`, `throughput_before_source`, `throughput_after`, `throughput_after_source`, `alias_conflicts` |
| Evidence | `gates[]`, `backend_attempts[]`, `measurements[]`, `measurement_source`, `measurement_occurrences`, `artifacts[]` |

`kind` is one of `kernel_optimization`, `kernel_collective`, `gemm_tuning`,
`integrate_patch`, `framework_agent`, `explore`, or `replay_warm_recipe`.

Several fields exist to say where a contested value came from, because the
value alone cannot:

* `agent_method` — `recorded` when the producer stamped the owner,
  `derived` when the read side had to infer one.
* `decision_source` — a verdict an executor stated is a different claim from
  a status inferred from the operation around it.
* `keep_threshold_source` — one of `gate.inputs`, `gate.evidence`,
  `decision.evidence`, or `outputs`. A bar recorded on the gate is the one
  that gate ruled against; one recorded on the outputs is the executor's
  configuration, which need not be what applied.
* `throughput_before_source` / `throughput_after_source` — `adoption` means
  the number was frozen when the decision was made; `measurement.<name>`
  means it was read back afterwards and could since have moved.
* `alias_conflicts` — roles that more than one recorded measurement name laid
  claim to with readings that disagree. The first name won; this records that
  the choice was not free.
* `local_gain_source` — deliberately never named `gain_pct`. `local_gain_pct`
  is what the executor measured against its own starting point, which is not
  the session baseline once anything has been adopted. **These must not be
  summed across attempts**; `entries[].gain_pct` is the summable figure.

### `entries[]` — the adopted ledger

One row per adopted optimization, in adoption order, carrying only what the
chained arithmetic needs. Descriptive evidence (artifacts, kernel id, the
starting throughput, gates, measurements) lives on the attempt and is reached
through `adopted_attempt_id`.

| Field | Description |
|---|---|
| `id` | `<session_id>:optimization:<stack_index>` |
| `stack_index` | Position in the adopted ledger. |
| `adopted_attempt_id` | Join key into `attempts[]`. |
| `adoption_id` | The adoption record that credited this step. |
| `source` | `warm_replay`, `explore`, `framework_agent`, `kernel_agent`, or `unattributed`. |
| `source_method` | `recorded` or `derived`, as on the attempt. |
| `optimization_kind` | The attempt's `kind`. |
| `name` | Operation name, typically the kernel or variant. |
| `backend` | Producing engine, e.g. `geak` or `forge`. |
| `gain_pct` | Gain against the **session baseline**. The only figure that can be summed. |
| `gain_method` | How `gain_pct` was arrived at; see below. |
| `chain_continuous` | `false` when this step recorded no finishing throughput, so the drift across it could not be measured. |
| `local_gain_pct` | The executor's own figure, kept beside `gain_pct` so the two are visibly different numbers. Not summable. |
| `cumulative_gain_pct` | Running total including unattributed drift. |
| `throughput_after` | Finishing throughput, when recorded. |
| `validated` | Always `true`; only adopted steps become entries. |
| `ts` | The attempt's `ended_at`. |

`gain_method` is one of:

* `baseline_chain` — measured against the previous step's finishing
  throughput. The trustworthy case.
* `local_gain_projected` — the finishing throughput was never recorded, so
  the step's own percentage was projected onto the chain.
* `recorded_adoption` — taken from the adoption record directly.
* `missing` — no gain figure could be established.

A `kernel_agent` entry's `optimization_kind` records which lane produced it:
`gemm_tuning`, `kernel_collective`, or `kernel_optimization` for a generic
source-level rewrite. `kernel_collective` comes from the collective lane,
which records its promotion as an operation of that kind with the integrate
that settled it; it attributes to `kernel_agent` like any other kernel work.

Only adopted entries contribute to `summary_by_source`, `summary_by_agent`,
and `summary_by_kind`. The first answers which agent produced the gain, the
second adds a per-kind split under each agent, and the third groups the same
gains by kind alone. These are alternate views of one set of gains and **must
not be added together**.

### `validation` — reconciliation, not arithmetic

The headline figure and the ledger's own sum are reported side by side so the
two can be seen to disagree. When the run recorded an end-to-end validated
gain, `validated_total_gain_pct` is that measurement and
`ledger_total_gain_pct` is what the adopted steps add up to; when it did not,
they are the same number by construction and nothing here can be checked.

| Field | Description |
|---|---|
| `method` | `recorded_session_validation`, `ledger_sum`, or `unavailable`. |
| `validation_basis` / `validation_source` | How and by which promote path the session figure was measured. |
| `validated_at_stack_len` | Ledger length the session figure was measured at. |
| `validated_total_gain_pct` | What the session was measured to have gained. |
| `ledger_total_gain_pct` | What the adopted steps sum to on their own. |
| `reconciliation_gap_pct` | The difference between the two. `null` when there is no measured figure to compare against. |
| `attributed_total_gain_pct` | Gain claimed by adopted steps. |
| `unattributed_gain_pct` | Gain sitting between adopted steps, credited to nobody rather than to whoever follows. |
| `attribution_gap_pct` | Session figure minus attributed. |
| `attempt_count` / `keep_count` | Attempts recorded, and how many were adopted. |
| `non_attributable_keep_count` | Adoptions explicitly marked not attributable. |
| `unmeasured_keep_count` | Adopted with nothing measured behind them. |
| `projected_keep_count` | Adopted steps whose gain came from `local_gain_projected`. |
| `stale_evidence_count` | Adopted steps whose evidence trail no longer resolves. |
| `unclaimed_integration_count` | Operations recording an integrated change with no adoption crediting it. Any number here means `unattributed_gain_pct` is overstated by whatever those steps earned. |
| `unscored_keep_count` | Adopted on a KEEP verdict alone, with no accuracy gate having ruled. |
| `notes` | Free-text provenance of the projection. |

### Remaining arrays

`backend_attempts` retains adopted and non-adopted GEAK/Forge attempts,
including KEEP, PARTIAL, REVERT, and FAILED outcomes. `sequence` is ordered
within each kernel. Adopted kernel entries link back through
`adopted_attempt_id`. When multiple KEEP attempts match the same entry and the
producer did not identify the adopted one, the link stays `null` and a warning
is emitted rather than guessing.

`gemm_tuning_runs` retains the complete tuning run records; the corresponding
adopted gain remains represented exactly once by a `gemm_tuning` entry.

The historical `optimization_stack`, attribution, GEAK invocation, Forge
invocation, and GEMM-tuning result projections are not emitted in the V5 wire
shape. Their required downstream evidence is instead normalized into the
canonical fields above.

### Migrating from the V4 shape

`entries[]` no longer carries `action`, `variant_name`, `fingerprint`,
`scope`, `source_phase`, `task_id`, `provenance`, `configuration`,
`execution_mode`, `accepted_heads`, `candidate_flags`, or
`extra_server_args_is_invariant`. Three of the removed fields moved rather
than disappeared and are reachable through `adopted_attempt_id`:

| Was on `entries[]` | Now |
|---|---|
| `artifacts` | `attempts[].artifacts` |
| `kernel_id` | `attempts[].kernel_id` |
| `throughput_before` | `attempts[].throughput_before` |

`validation.source_breakdown`, `validation.phase_breakdown`, and
`validation.domain_attribution` are gone. Per-agent totals are now
`summary_by_agent`; the gain belonging to no adopted step is
`validation.unattributed_gain_pct` rather than a bucket inside a breakdown.

The Recipe KB touchpoints are recorded as ordered `timeline` events rather
than a standalone `kb_provenance` section (removed in the V5→V6 migration):
`warm_start` (which identity was requested and what the KB returned),
`warm_replay` (whether replaying a prior recipe reproduced its gain), and
`kb_write_back` (whether this session's own recipe reached the KB Store).

When Warm Replay uses a donor recipe, `timeline[type=warm_replay].ext.donor`
preserves the available `canonical_id`, `model`, `session_id`, `gain_pct`, and
`breakdown_link`. Fields absent from the source recipe remain absent rather
than being inferred.

`timeline[type=warm_replay].ext.accuracy` records what the replay was judged
on, and whether it passed. A replayed recipe is evidence from another session
on another machine, so reproducing its throughput says nothing about whether it
still computes correctly here.

| Field | Type | Description |
|---|---|---|
| `eval_ran` | bool | Whether an eval produced output for this replay. Separates a model that answered nothing (`eval_ran` true, `replay` `0.0`) from a replay nothing checked (`eval_ran` false, `replay` `null`). |
| `replay` | float \| null | Score measured on the replayed config. `null` when no score could be read — not a score of zero. |
| `baseline` | float \| null | Reference the replay was compared against. `null` when the session recorded none, in which case the replay is judged against an absolute floor instead of a relative drop. |
| `passed` | bool \| null | Whether the replay cleared the accuracy gate. `null` when no verdict could be reached (no eval ran). |

A replay whose accuracy could not be measured is still promoted — a failed
measurement is not evidence the config broke the model — so `eval_ran` is what
tells an unjudged promotion apart from a judged one.

The `optimization_stack` entry a warm replay pushes carries the same score as
`accuracy`, so the promotion and the evidence behind it are readable from one
place. `null` there means the lane recorded no verdict.

Sessions started with `--no-eval` run no eval at all, warm replay included, so
these fields record the absence rather than a score.

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

* `log_non_default_args`: Most authoritative (parsed from the
  vllm/sglang server's own arg echo).
* `log_args_line`: `Args: Namespace(...)` header.
* `log_python_cmd`: Literal `python …` launch line scraped from logs.
* `yaml_cmd`: `cmd:` / `command:` / `launch:` field in the
  materialised config YAML.
* `yaml_benchmark`: Synthesised from Magpie's `benchmark.*` YAML
  fields.
* `unknown`: None of the above; a warning is appended to
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

One card per live capability (`geak`, `forge`, `explore`,
`specialist`) with: `status`, `attempts`, `keeps`, `micro_only_keeps`,
`pending_integrate`, `reverts`, `e2e_gain_pct`, `tested`, `best_gain_pct`,
`reason`. Legacy `backends`, `params`, and `validate_stack` rows can appear
when archived sessions are rebuilt. Drives the per-session UI cards in
Primus-Claw.

For the kernel lanes (`geak`, `forge`) these counts are not interchangeable:

- `keeps` — **distinct kernels adopted at integrate**, i.e. end-to-end
  verified. A kernel re-tried across runs counts once.
- `micro_only_keeps` — kernels that cleared the micro benchmark but never
  reached integrate. Not adoptions: a faster kernel in isolation does not
  imply a faster service.
- `pending_integrate` — kernels whose integrate verdict is `NEEDS_REVIEW` or
  not yet recorded. Undecided, not successful.
- `reverts` — kernels integrate rejected (end-to-end regression).
- `attempts` — **invocation rows**, not distinct kernels: how many tries the
  lane made. Deliberately a different unit from `keeps`.

GEAK e2e runs do not always create the native
`kernel-agent/runs/*/optimization_attempts.jsonl` layout. In that case the
summary falls back to the normalized top-level `geak` section: a real route is
reported as at least one attempt, and a promoted `geak_e2e` stack entry is
reported as kept instead of `not_attempted`.

The `specialist` row uses `keeps` / `attempts` differently: see
`CapabilitySummary` in `schema.py`.

---

## `kernel_lifecycle` — `KernelLifecycle`

The 4+1-stage kernel pipeline:

* `detected`: TraceLens-identified hot kernels.
* `recommended`: Critic-filtered candidates with backend
  recommendations.
* `optimized`: Kernels with at least one completed backend attempt
  and `best_micro_speedup`.
* `adopted`: Kernels promoted into the final stack (end-to-end validated).
* `rejected`: Kernels considered then dropped, with `reason`.

The same `kernel_id` appears in multiple lists as it progresses.

---

## `collective` — `Collective`

Multi-rank communication campaigns run at KERNEL entry, mirroring the
`collective_only_mode`, `collective_attempts` and `last_collective` SharedState
fields. Absent (`{}`) when the lane never ran.

* `only_mode`: mirrors `HYPERLOOM_COLLECTIVE_ONLY`, so a reader can tell a
  collective-only session from one where the lane merely happened to run.
* `attempts`: one `CollectiveAttempt` per logical campaign, deduplicated by
  `collective_attempt_id` so a resumed or salvaged run is not double-counted.
* `last`: the most recent campaign record, which additionally carries the
  measurement evidence the ledger rows omit — `bandwidth` (per case: `bytes`,
  `algbw_gbps`, `busbw_gbps`) and `artifact_files`.

This section is deliberately separate from `optimizations`. Adoption is decided
by `integration_decision` (the E2E gate), not by `decision` (the
microbenchmark), so a campaign that wins its micro run and then loses the gate
never reaches `optimizations` — and without this section would leave no trace
in the breakdown at all. Read `integration_gain_pct` against
`integration_base_tput` / `integration_new_tput` for the throughput delta that
actually decided the outcome; `kernel_speedup` is microbenchmark-only.

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

Terminal Recipe publication is reported alongside the artifact paths:

| Field | Type | Meaning |
|---|---|---|
| `recipe_finalize` | dict | Secret-free outcome from the latest finalize attempt, including its source, attempt number, timestamp, and write/skip/error details |
| `recipe_finalize_status` | string | Durable lifecycle state: `pending`, `written`, `skipped`, `disabled`, or `failed` |
| `recipe_finalize_attempts` | int | Number of idempotent finalize attempts across CLOSE and graceful-teardown fallback paths |

`failed` is retryable during the same process lifetime. Terminal statuses
(`written`, `skipped`, and `disabled`) suppress duplicate publication.

### `telemetry.orchestration_context`

Health of the orchestration conversation's compaction loop (`OrchestrationContext`).
All fields are `total=False`; sessions predating this field report zeroes.

| Field | Type | Meaning |
|---|---|---|
| `seed_prompts` | int | Full-state SEED pushes to the orchestration backend |
| `delta_prompts` | int | Thin DELTA pushes (verbose state omitted) |
| `compactions` | int | `orchestration_checkpoint` events recorded |
| `degenerate_compactions` | int | Compactions skipped on an unusable summary |
| `tick_count` | int | Ticks executed; denominator for the rates below |
| `compactions_per_tick` | float | `compactions / tick_count`; near 1.0 means every tick re-seeds the conversation |
| `delta_ratio` | float | `delta_prompts / (seed + delta)`; near 0 means the DELTA path is not being reached |
| `context_tokens_at_compaction` | dict[str, int] | `min` / `median` / `max` token water level at each compaction; a `min` above the soft budget means the token trigger cannot be un-tripped by compacting |

---

## `enablement` — admission, round lifecycle, builds & attempt runtimes

`EnablementBreakdown`. The enablement subsystem's observability section: which
lane was admitted, what each authoring round did, the patches and stack actions
it landed, the attempt runtimes it provisioned, and the targeted builds (AITER /
sgl-kernel / vLLM-source) it attempted.

Emitted when the lane did something, or when it was explicitly turned off — the
opt-out is what explains a run that failed to establish a baseline without
anything trying to repair it. Since `all` is the default, an armed lane that was
never needed stays hidden.

Admission and round lifecycle are reported independently of the artifacts: a
boot-origin round repaired by a plain source patch provisions no runtime and
builds nothing, and would otherwise leave no trace at all.

Admission and lifecycle (always present when the block is emitted):

| Field                       | Type   | Description                                                                              |
|-----------------------------|--------|--------------------------------------------------------------------------------------------|
| `mode`                      | string | Admitted lane from `--enablement`: `off` / `launch` / `eval` / `all`.                        |
| `engaged`                   | bool   | A round was dispatched, attempted, or landed a patch. `false` with a non-`off` mode means the lane was armed but never needed. |
| `origin`                    | string | Trigger origin: `boot` (cannot launch) or `eval` (accuracy).                                 |
| `attempts`                  | int    | Authoring rounds dispatched this session.                                                    |
| `dispatched`                | bool   | An authoring round is in flight.                                                             |
| `succeeded`                 | bool   | A round was KEPT. Eval-origin additionally requires the revalidation baseline to promote at or above the floor. |
| `pending`                   | bool   | A trigger is captured but unconsumed.                                                        |
| `validation_pending`        | bool   | An eval-origin KEEP awaits baseline revalidation.                                            |
| `stall_streak`              | int    | Consecutive no-progress rounds toward `enablement_stalled`.                                  |

Round detail (present when set):

| Field                       | Type                             | Description                                                        |
|-----------------------------|----------------------------------|------------------------------------------------------------------------|
| `inflight_task_id`          | string                           | Specialist task id of the in-flight round.                             |
| `last_specialist_task_id`   | string                           | Specialist task id of the most recent round.                           |
| `dispatch_tick`             | int                              | Coordinator tick the in-flight round was dispatched on.                |
| `revalidation_task_id`      | string                           | TaskRegistry id of the tracked revalidation task.                      |
| `revalidation_generation`   | int                              | Revalidation window counter (idempotency).                             |
| `launch_log_excerpt`        | string                           | Tail (2000 chars) of the boot failure text that triggered the round.    |
| `kept_patches`              | string[]                         | Session-relative paths of patches landed by enablement.                |
| `kept_stack_action`         | `EnablementStackActionSummary`   | The stack action behind the KEPT attempt runtime.                      |
| `candidate_refs`            | string[]                         | Bridging candidate refs considered for rotation.                       |
| `setup_commands`            | string[]                         | Setup commands the specialist requested.                               |
| `localization_manifest`     | string[]                         | Files the localization pass identified.                                |
| `build_novelty`             | string[]                         | Novelty keys of the targeted builds requested.                         |
| `human_review_count`        | int                              | Logs parked for human review.                                          |
| `accepted_config_path`      | string                           | Effective config from the KEPT candidate bench.                        |

Eval-origin trigger (present when `origin` is `eval`):

| Field                       | Type   | Description                                                                              |
|-----------------------------|--------|--------------------------------------------------------------------------------------------|
| `trigger_kind`              | string | `eval_runtime_failure` / `accuracy_below_floor` / `accuracy_unavailable`.                    |
| `observed_accuracy`         | float  | Baseline accuracy observed at the trigger.                                                   |
| `accuracy_floor`            | float  | Effective floor for the trigger and the KEEP gate.                                           |
| `observed_task`             | string | Eval task name observed at the trigger.                                                      |
| `observed_metric`           | string | Eval metric observed at the trigger.                                                         |
| `eval_contract_fingerprint` | string | Fingerprint of the captured eval contract.                                                   |
| `probe_config_path`         | string | Materialized config re-run to reproduce the contract.                                        |
| `trigger_evidence_excerpt`  | string | Tail (2000 chars) of the captured eval-failure evidence.                                     |

Stack actions, runtimes, and builds:

| Field                 | Type                          | Description                                                                                     |
|-----------------------|-------------------------------|-------------------------------------------------------------------------------------------------|
| `stack_actions`       | `EnablementStackActionSummary[]` | Candidate stack actions considered this session (see below).                                 |
| `active_runtime`      | `EnablementAttemptRuntime`    | The currently-promoted attempt runtime, or `{}` when none.                                      |
| `attempt_runtimes`    | `EnablementAttemptRuntime[]`  | Retained attempt-runtime records (capped).                                                      |
| `failure_kind`        | string                        | Last classified enablement failure kind (present only when set).                               |
| `build_attempts`      | `TargetedBuildAttemptSummary[]` | Targeted-build attempt history, newest last (see below).                                      |
| `last_build_failure`  | object                        | `{failure_class, failure_summary}` from the most recent failed build (framework-channel input). |
| `build_attempt_count` | int                           | Total number of targeted-build rows attempted.                                                  |

### `stack_actions[]` — `EnablementStackActionSummary`

One attempt-runtime stack action considered or applied.

| Field                | Type   | Description                                                             |
|----------------------|--------|-------------------------------------------------------------------------|
| `kind`               | string | Stack-action kind (for example, `runtime_candidate`).                   |
| `framework`          | string | Target framework.                                                       |
| `capability`         | string | Missing capability being repaired.                                      |
| `acquisition_method` | string | `wheel` / `editable_ref` / … .                                          |
| `repo_url`           | string | Origin git URL (source acquisition), or `""`.                           |
| `ref`                | string | Pinned ref (source acquisition), or `""`.                               |
| `index_url`          | string | Pip index (wheel acquisition), or `""`.                                 |
| `reason`             | string | Human-readable justification.                                           |

### `active_runtime` / `attempt_runtimes[]` — `EnablementAttemptRuntime`

One provisioned attempt runtime (promoted or discarded). `active_runtime`
is the single promoted runtime; `attempt_runtimes[]` is the retained
history, each flagged with `promoted`.

| Field                | Type               | Description                                                                    |
|----------------------|--------------------|--------------------------------------------------------------------------------|
| `venv_root`          | string             | Attempt venv root (`$SESSION_DIR/enablement/stacks/…`).                        |
| `bin_path`           | string             | Attempt bin dir prepended to the materialized-YAML `PATH`.                     |
| `python_path`        | string             | Attempt interpreter.                                                           |
| `installed_versions` | object (str → str) | Package → version installed into the attempt venv.                             |
| `promoted`           | bool               | `true` when this runtime was kept (survives rearm).                            |

### `build_attempts[]` — `TargetedBuildAttemptSummary`

One targeted-build attempt (AITER / sgl-kernel / vLLM-source).

| Field                | Type               | Description                                                                    |
|----------------------|--------------------|--------------------------------------------------------------------------------|
| `component`          | string             | `aiter` / `sgl_kernel` / `vllm_source` / `framework_ext`.                      |
| `ref`                | string             | Git ref / tag used for the build.                                              |
| `gpu_arch`           | string             | Explicit target arch (`gfx942` / `gfx950` / …).                                |
| `max_jobs`           | int                | Parallelism cap passed to the compile.                                         |
| `ok`                 | bool               | Whether the build probe and install succeeded.                                 |
| `failure_class`      | string             | One of the `FAILURE_CLASSES` values, or `"ok"`.                                |
| `failure_summary`    | string             | Human-readable reason (agent decision input).                                  |
| `installed_versions` | object (str → str) | torch/ref/sha/arch recorded after a successful build (see below).              |
| `build_probes`       | string[]           | Post-build probe descriptors, e.g. `"import aiter: ok"` (up to 8).            |
| `build_log_path`     | string             | Path to the compile log inside the attempt dir.                                |
| `attempt_root`       | string             | Attempt directory anchoring the build.                                         |

`installed_versions` is a free-form string → string provenance map copied
verbatim from the build manifest. Keys include torch and commit-SHA stamps,
`arch`, and the component ref keys `aiter_ref` / `vllm_ref` / `sgl_kernel_ref`
(the first present ref is also surfaced as the top-level `ref` field). When a
discovered PR ref drove the build, it additionally carries a `source_pr_url`
key pointing at the source PR. Because the map is free-form, `source_pr_url`
is not a declared TypedDict key — consumers should read it opportunistically.

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
  "schema_version": "hyperloom.session_breakdown.v5.0",
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
    "image": "lmsysorg/sglang-rocm:v0.5.18-rocm724-mi30x-20260825"
  },

  "workload": {
    "framework_name": "sglang",
    "framework_version": "0.5.18",
    "model_name": "GLM-5-FP8",
    "model_path": "/models/GLM-5-FP8",
    "model_class": "moe_mla_nsa",
    "gpu_type": "mi355x",
    "tp": 4,
    "conc": 64,
    "isl": 1024,
    "osl": 1024,
    "max_model_len": 8192,
    "precision": "fp8",
    "objective": { "kind": "tput", "value": 150.0 }
  },

  "baseline": {
    "throughput_tok_s_per_gpu": 100.0,
    "accuracy": 0.812,
    "ttft_mean_ms": 0.0,
    "e2el_mean_ms": 0.0,
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
    "throughput_tok_s_per_gpu": 150.0,
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

  "warnings": [],
  "source_files": {
    "manifest": "manifest.json",
    "state": "state.json",
    "baseline_report": "runs/baseline/report.json",
    "profile_reports": ["runs/profile/report.json"],
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
   major `schema_version`. Such changes require a major bump, as the
   `v5.0` optimization cutover did.
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
