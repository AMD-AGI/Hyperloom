---
myst:
    html_meta:
        "description": "AgentX performance metric variables for Hyperloom: 2-D interactivity grading, noise band, early-stop mode, and the MLPerf agentic backend for MLCommons inference-endpoint replays."
        "keywords": "Hyperloom, environment variables, AgentX, HYPERLOOM_PERF_METRIC, HYPERLOOM_AGENTX_EARLY_STOP, HYPERLOOM_AGENTIC_BACKEND, mlperf, interactivity, intvty_v1"
---
# Hyperloom AgentX performance metric

Grading defaults to output throughput alone. On an agentic replay that is the
wrong objective: the canonical corpus averages ~114k prompt tokens against ~810
output tokens per request, so output-only grading optimises about 1% of the
token budget, and a variant can lift decode tok/s while degrading user-perceived
latency with no visible cost.

`HYPERLOOM_AGENTX=1` (workload YAML generated from Hyperloom flags), or an explicit
AgentX YAML passed with `--benchmark-config`, adopts the 2-D grading shape InferenceX uses. Upstream
sweeps a concurrency ladder, keeps TTFT / ITL / TPOT percentiles separately, and
publishes a Pareto frontier with E2E normalised interactivity as the x-axis
and token throughput per chip as the y-axis. It never collapses the axes into one
weighted number, and it has no fixed interactivity target — interactivity is
a frontier coordinate, not a constraint, so a point that trades interactivity for
throughput moves along the frontier rather than violating a rule.

Hyperloom's local KEEP rule (fixed concurrency, no ladder) approximates the
per-concurrency arm selection maintainers apply before submitting:

- **KEEP** — E2E normalised interactivity P50 (median) gain ≥ 2%, and the P90
  slow tail and total token throughput each no worse than the noise band, and the
  pair is comparable: benchmark windows within 5% of each other and no more failed
  requests than the anchor.
- **REVERT** — anything short of all of that.

The median bar is the P50 objective's fixed `bar_pct` in
`hyperloom.common.agentx_objective` (2%, the same bar and guards
`hyperloom-multibranch` applies), not `keep_threshold_pct`: it is a property of
the objective rather than of how far into the session a round lands, so the
decaying session threshold does not apply to it. Output throughput is recorded
but guards nothing: it follows the replayed window's output-length mix, not the
server.

Hyperloom reads `e2e_norm_intvty_p90` from the accepted `current_best` for both
grading and advisory comparison. This is aiperf's summary P10 of the
per-request rate `OSL / E2EL_s`, representing the slow tail for a
`LARGER_IS_BETTER` metric. Comparison does not recompute request-level metrics.
The external reference uses `1 / P90(E2EL_s / OSL)`; finite-sample linear
interpolation means the estimators need not be numerically identical.
The comparison is advisory and does not change KEEP/REVERT.

TTFT is included in E2EL, unlike per-user `1/ITL`; on a ~114k-prompt replay TTFT
is most of what a user waits for so grading on `1/ITL` would miss it.

Default-on for AgentX runs, explicit opt-in via `HYPERLOOM_PERF_METRIC=intvty_v1`
otherwise. Any AgentX signal turns it on: `HYPERLOOM_AGENTX=1`, the bound AgentX YAML, or
`benchmark_mode=agentx` stamped at seed — so a round in a subprocess that never
inherited the env var still grades on the agentic axis. Serving frameworks only;
scriptable frameworks (xDiT, custom) keep output-throughput grading.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_PERF_METRIC`        | `intvty_v1` for AgentX, else output tput | `intvty_v1` grades E2E normalised interactivity P50 (median) as the primary objective, with the P90 slow tail and total token throughput each held as a guard. Reported in the final summary as `grading mode`. |
| `HYPERLOOM_PERF_NOISE_PCT`     | `5.0`                         | Noise band in percent applied to the guards. A candidate whose slow tail or total token throughput sits within this band of the anchor is not considered worse on that axis; the median has its own fixed bar and is not subject to the band. The default is the top of the 1–5% run-to-run noise upstream records for this workload. An unparseable value raises `EnvValueError` naming the variable, rather than grading against a band nobody chose. |
| `HYPERLOOM_AGENTX_EARLY_STOP`  | `enforce`                     | AgentX checkpoint after 1800 s of profiling: the round's median interactivity over the requests finished by then, against the reference's own first 1800 s. The ruling is written to `agentx_checkpoint.json` in the round's slot. `enforce` stops an explore decision round that trails by more than 2% and records it as REVERT; `shadow` only writes the ruling; `off` skips the checkpoint. Any other value raises `EnvValueError`. A reference measured without a checkpoint median is never compared. `hyperloom-multibranch run` reads it once at session start with `shadow` as its default, keeps the mode in `MB-manifest.json`, and judges only explore tests under `p50_interactivity`; see [Run a multi-branch session](../how-to/run-multibranch.md#environment). |
| `INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC` | `7200` | Initial server-boot budget written by the persistent-server lifecycle configuration helper. Actual benchmark launches synchronize this field to `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC`; this variable is not an additional benchmark deadline or a way to extend one. Non-benchmark lifecycle callers that do not perform that synchronization retain their own boot budget. |

## MLPerf agentic backend

`HYPERLOOM_AGENTIC_BACKEND=mlperf` (with `HYPERLOOM_AGENTX=1`) swaps the
AgentX client for the MLCommons `inference-endpoint` harness
(`utility/run_agentic.sh`), replaying `agentic_combined_v6` trajectories instead
of aiperf Weka traces. Each round runs on inference_suite: it serves the
session's frozen base launch (golden layer included) and runs the harness from
`MLPERF_ENDPOINTS_DIR` as the round's load, then maps the harness's
`result_summary.json` and `scores.json` into the run directory's
`client/inferencex_result.json`. A session that froze no base launch is refused
at preflight. The backend is recorded at seed; resuming a session under the
other backend is refused.

The harness publishes no per-request OSL/E2EL series, so MLPerf sessions grade
on output throughput (`HYPERLOOM_PERF_METRIC` is not honoured here) and gate
every KEEP on the harness's inline accuracy (`scores.json` `score`) against the
baseline's smoke score, within the usual accuracy tolerance. A round with
unscored turns, or over the 10% error-rate ceiling, fails that gate. A baseline
without a fully scored inline accuracy stops the session
(`baseline_accuracy_failed`), so the gate is never left without a reference.

Search runs the 150-trajectory smoke set. The canonical 613-trajectory
submission is not run by Hyperloom: run `run_agentic.sh full` and upstream
`check_compliance.py` on the published recipe.

The AgentX switch pins `PORT=30000`, `MLPERF_AGENTIC_MODEL=kimi-k3` and
`--served-model-name kimi-k3`, and settles `MLPERF_AGENTIC_FLOW`,
`AGENTIC_NUM_TRAJECTORIES` and `AGENTIC_CONCURRENCY` (from the round's `CONC`)
into the recipe. The round's server serves under `kimi-k3`, the name the
harness requests, on the round's `PORT`, which the harness is given.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_AGENTIC_BACKEND` | — | `mlperf` selects the separate MLPerf client together with `HYPERLOOM_AGENTX=1`. Otherwise `HYPERLOOM_AGENTX=1` selects native AgentX; an explicit AgentX YAML also selects it. |
| `MLPERF_ENDPOINTS_DIR` | `/opt/mlperf-endpoints` | Checkout of `mlcommons/endpoints` holding `utility/run_agentic.sh`. Checked at preflight. |
| `AGENTIC_DATASET_PATH` | — (required) | Path to `agentic_combined_v6.jsonl`. Checked at preflight; its stem is recorded as the corpus. |
| `MLPERF_TOKENIZER_DIR` | — (required) | Kimi-K3 fast-tokenizer directory the harness counts tokens with. Checked at preflight. |
| `MLPERF_AGENTIC_FLOW` | `smoke_test` | `run_agentic.sh` flow: `smoke_test` (150 trajectories) or `full` (613). |
| `AGENTIC_NUM_TRAJECTORIES` | Derived from the flow | Overrides the trajectory count. Sizes the benchmark timeout. |
| `MLPERF_AGENTIC_HARDWARE` | `mi355x` | Hardware key passed to `run_agentic.sh`. |

Preflight also refuses a non-SGLang framework and a model path that is not
Kimi. With `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` unset, the benchmark cap
defaults to `trajectories × 26 s × 2 + 1800 s` (26 s per trajectory measured on
8×MI355X at concurrency 16, doubled for a cold first round, plus boot), and never
below the stock 7800 s: about 9600 s for smoke and 33,700 s for `full`.

With `HYPERLOOM_AGENTX=1`, Hyperloom generates and archives the AgentX workload YAML
from its ordinary workload flags. `--docker-image` supplies an explicit serving
image identity, required for custom models.

Canonical replay warms up 11 requests per lane plus one probe before it opens its
fixed profiling window, so its round grows with the concurrency being measured
while the window does not. That shape is fixed by `CANON_WARMUP_PER_LANE` and
`CANON_PROFILING_WINDOW_SEC` in `hyperloom.common.agentx_workload`; Hyperloom
composes only `CONC`, and the numbers below have to be re-measured when either moves. With
`INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` unset and a `CONC` pinned, the cap
defaults to `(11 × CONC + 1) × 20 s + 3640 s + 300 s + 1800 s` (20 s per warmup
request, sized to the slowest rung of one 8×MI355X sweep -- 18.8 s at `CONC` 20 --
with about 6% over it, plus the `metric_window_s` window, metrics finalisation and
boot), and never below the stock 7800 s: about 10,200 s at `CONC` 20. A
concurrency sweep sizes this per rung from the environment the rung's own config
is launched with -- whether the `CONC` in it was pinned by the rung, inherited
from the base config or cleared by the rung -- so the rung that measures more
lanes is granted more time rather than the ladder sharing one grant. The
override itself stays the operator's: it is read from the process environment
that preflight validated, never from a rung's config. Sizing reads a rung's
workload pins as its launch would, so a rung whose config spells one the reader
refuses (`HYPERLOOM_AGENTX=maybe`) is recorded as failed with error class
`benchmark_cap_error` and the sweep carries on with the next rung.

Native AgentX profiling shares TraceLens patch and kernel-shape-tool preparation with
fixed-sequence profiling. Configure capture steps and delay through
`benchmark.profiler.torch_profiler` (`num_steps`, `start_seconds`); a missing setting
becomes one capture of 256 steps, starting 300 seconds after measurement. A profile round
captures one window, so a YAML whose `num_profiles` is not 1 is refused when the session
starts. A profile round adds the profiler layer and the shape-tool
environment to the round's launch, preserving the server's runtime paths, and opens its
capture window once the replay reaches measurement. InferenceSuite owns the profiler
flags and the capture and flush; each window remains a separate TraceLens input.
Profile reports are diagnostic only and cannot supply KEEP evidence.
See [Optimize an AgentX workload](../how-to/optimize.md#optimize-an-agentx-workload).
