# Session breakdown: where each metric comes from

A field in `session_breakdown.json` is only as trustworthy as the moment it was
captured. This page inventories every V6 block by **provenance** and names the
fields whose value can depend on *when* or *where* the export happens rather
than on what the session did.

It exists because that dependence is invisible in the output. A field derived
at export time from an environment variable, a file a later phase overwrites,
or an aggregate recomputed by a second implementation looks exactly like a
field recorded when the fact occurred. The difference only shows up as a number
nobody can reproduce.

## The four sources, best to worst

| Source | Written when | Survives an offline re-export |
|---|---|---|
| **Durable event stream** (`session/sbd_v6.py`, `write_timeline_event`) | the fact happens | yes |
| **Recorder fragments** (`breakdown/recorder/`) | the fact happens | yes |
| **`state.json`** (`SharedState`) | during the run | yes |
| **Export-time re-derivation** (inside a collector) | at export | **no** |

The first three record. The fourth reconstructs, and reconstruction is where
drift lives.

`install` and `model_gate` are the reference implementation of the first row:
both are written by `write_timeline_event` when they happen, and `model_gate`
even updates its own event in place as it progresses. Neither can be wrong
about a session that finished last week.

## What counts as drift risk

Rated HIGH when the value can differ between two exports of the *same*
finished session:

- reads `os.environ` — CLOSE frequently exports from a subprocess that did not
  inherit the run's environment
- re-parses a file a later phase can overwrite, rotate, or drain
- recomputes an aggregate the orchestrator already computed, so two
  implementations can disagree
- probes for a file's existence, so the answer depends on export ordering

## Inventory

### `metadata`

| Field | Source | Risk |
|---|---|---|
| `grading.*` | `state.grading`, recorded at seed | NONE |
| `session.user_data_path` | manifest → state → **`os.environ`** | HIGH |
| `session.ended_at_utc` | state, falls back to export wall clock | MEDIUM |
| `session.elapsed_minutes` | derived; uses `now()` while still running | MEDIUM |
| `session.session_dir` | the *resolved* export path | MEDIUM |
| `langfuse.{enabled,trace_url}` | receipt → in-process emitter → **env** | HIGH |
| `versions.tools` | recorder | NONE |
| everything else under `session` / `task_config` | state.json / manifest | LOW |

`metadata.langfuse` is the worst of these: in a subprocess CLOSE there is no
in-process emitter and often no receipt yet, so it silently drops to the
environment view. The `patch_breakdown_langfuse` splice that repairs this
rewrites only the V5 top-level `langfuse`, not `metadata.langfuse`.

### `outcome`

| Field | Source | Risk |
|---|---|---|
| `baseline.{total_throughput,input_throughput,intvty_p90,tpot_p90_ms}` | `state.baseline_perf` | LOW |
| `baseline.{ttft_mean_ms,e2el_mean_ms}` | re-parses `benchmark_report.json`, else newest-by-mtime walk of `runs/baseline/` | MEDIUM-HIGH |
| `final.*` | `state.current_best` | LOW |
| `final.graded_on` / `validation.graded_on` | same helper as `metadata.grading` | NONE |
| `validation.{attributed,unattributed,gap,notes}` | recorder | LOW |
| `validation.attribution.by_source.*` | **re-buckets and re-sums** the recorder summary at export | MEDIUM |
| `stage_reached` | derived from a ~15-branch probe ladder over state | MEDIUM |

`validation.attribution` is a second summation over entries the recorder
already totalled. The ledger itself warns when its own sum disagrees with the
run-promoted total, which is evidence the two-implementation risk is not
theoretical.

### `timeline`

| Stage | Source | Risk |
|---|---|---|
| `install`, `model_gate` | durable events | NONE |
| `warm_start`, `warm_replay`, `kb_write_back` core | state.json | LOW |
| `kb_write_back.ext.queue.*` | live line counts of the KB ndjson queues | HIGH |
| `baseline.ext.{ttft,e2el}` | as `outcome.baseline` above | MEDIUM-HIGH |
| `sweep.ext.sweep.all_variants[]` | disk scan of grid point directories | MEDIUM |
| `conc_sweep.ext.*` | mirrors `reports/conc_sweep_summary.json`, else walks `runs/` | MEDIUM |
| `framework_agent.*` core | state + recorder operations | LOW |
| `framework_agent.ext.critic_reviews[]` | re-reads `critic-workdir/`, which the backend prunes | MEDIUM-HIGH |
| `framework_agent.*.plateau.*` | recomputed when evidence did not carry it | MEDIUM |
| `kernel.geak_runs[]` | `state.geak_result`, else reconstructs from the `geak/` tree | MEDIUM-HIGH |
| `kernel.fusion_runs[]` | `state.last_fusion` — last-write-wins, so a second fusion erases the first | MEDIUM |
| `kernel.attempts[]`, `gemm_tuning_runs[]` | recorder | LOW |

`kb_write_back.ext.queue` is the clearest ordering bug of the set: the counts
are taken while the breakdown is written at CLOSE step 2, and the `ndjson_drain`
step that empties those queues runs *after*. The timeline is never re-patched,
so the shipped depths are a mid-CLOSE snapshot that no longer describes
anything by the time the session ends.

### `close`

| Field | Source | Risk |
|---|---|---|
| `status`, `steps[]`, `close_sequence_done` | state; deliberately two-pass, repatched after the sequencer finishes | LOW |
| `robustness.*` | state + recorder | LOW |
| `artifacts.{final_json_path,final_md_path}` | existence probe at export | MEDIUM |
| `artifacts.artifact_package_path` | state | LOW |

## Ranked backlog

Ordered by drift risk times how load-bearing the value is.

| # | Field | Where it should be recorded instead |
|---|---|---|
| 1 | ~~`metadata.grading` env fallback~~ | **done** — recorded at seed; the collector no longer reads the environment |
| 2 | `metadata.langfuse.{enabled,trace_url}` | when the emitter is created in PRELUDE |
| 3 | `kb_write_back.ext.queue.*` | at the `ndjson_drain` close step |
| 4 | `metadata.session.user_data_path` | at session create, beside `session_dir` |
| 5 | `outcome.baseline.{ttft,e2el}` | when the baseline benchmark completes |
| 6 | `kernel.geak_runs[]` | when GEAK's final validation lands |
| 7 | `conc_sweep.ext.*` | when the paired comparison settles |
| 8 | `sweep.ext.sweep.all_variants[]` | per grid point, at measurement time |
| 9 | `outcome.validation.attribution` | consume the recorder summary instead of re-bucketing |
| 10 | `framework_agent.ext.critic_reviews[]` | rely on the durable `critic_iterations` fragments; treat the workdir scan as legacy-only |

## How to record a new fact

The recorder already exposes general entry points — `record_item` and
`record_singleton` on `Recorder`, surfaced to producers through
`breakdown/recorder/instrument.py` (`record_operation`, `record_measurement`,
`record_phase_event`, …). Each write lands as one atomic JSON fragment
(`tmp` + `os.replace`) under an envelope of `section` / `kind` / `seq` / `ts` /
`producer` / `payload`, so concurrent producers do not collide. Adding a fact
means declaring its section shape and calling one of those from the place that
knows the fact.

`write_timeline_event` is usable mid-run as well — nothing binds it to the
pre-session window, and `model_gate` already updates a live event through
`read_timeline_event_for_update`. Extending it to another stage needs that
stage added to `_EVENT_TYPES`, a call where the stage settles, and a collector
that prefers the durable event over its projection.

## The constraint this relaxes

`breakdown/collectors/v6_stages.py` states that the collectors are pure
projections that "add no writer call sites anywhere in `orchestrator/`". That
rule bought a clean layering, and it should keep holding for anything that is
genuinely derivable from what the run already recorded. What it should not do
is force a value that is only knowable at the moment it happens to be
reconstructed afterwards from whatever files and environment survive. The
backlog above is the list of places where that trade is currently being made
the wrong way round.
