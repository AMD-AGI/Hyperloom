---
myst:
    html_meta:
        "description": "Reference for InferenceSuite, the package that resolves a session's base launch and runs every Hyperloom benchmark round: recipe sources, framework checks, launch layers, server lifecycle, staging, the run directory and report.json."
        "keywords": "Hyperloom, InferenceSuite, inference_suite, base launch, recipe, InferenceX, SGLang cookbook, vLLM recipes, ATOM, AgentX, run_round, report.json, staging, lm-eval, GSM8K, AMD GPU, ROCm"
---
# InferenceSuite reference

InferenceSuite (`src/inference_suite/`) resolves a session's base launch and runs
every benchmark round Hyperloom measures. It builds only on `hyperloom_common` and
third-party libraries. It records what a round measured; thresholds, KEEP decisions
and validity stay in Hyperloom.

## Base launch

A fresh session resolves its base launch once, at launch, and freezes it as
`<session>/base_launch.json` (schema `hyperloom-base-launch/1`). A resume reads
that file and resolves nothing. Sources are tried in this order, and the first that
holds a launch for the session's model, board, workload, concurrency, TP and EP
gives the base:

1. the operator's `--reference-script`: a shell launch script, or a
   `base_launch.json` frozen by an earlier session;
2. InferenceX's AMD single-node master entries;
3. the framework's cookbook: the SGLang cookbook, the vLLM recipes site, or
   ATOM's catalog (fixed workloads only);
4. for a fixed workload only, the default table shipped in
   `src/inference_suite/recipe/defaults/table.yaml`.

An AgentX session that no recipe and no `--reference-script` serves is refused.
When a source cannot be read, a fixed session runs the default table and an AgentX
session is refused. Repositories are read at the head of their `main` and recorded
with that commit; the vLLM recipes site has no commit and is recorded as the
response it was read from.

A launch script is read, never run. It must launch the session's framework and fix
every flag in its own text: a flag whose value names a shell variable, or a
`# hyperloom-launch-controls:` line, is refused.

A source may hold several launches for one point. The base takes the one with the
largest TP the GPUs fit, preferring an arm that does not offload KV cache, and
records the others as alternatives.

The base's digest is its identity in every cache and comparison key. It covers the
framework, entrypoint, model ID, topology, argv, environment, golden layer and
client. It leaves out the source, the image, the model's local path and the
runner-owned flags and environment. Argv keeps its order, because of two aliases of
one flag the engine applies the last.

## Framework checks

Before a round boots a server, InferenceSuite asks the serving interpreter about
the installed framework:

- **Version.** The installed framework must be at least the version of the
  recipe's image. Only the release segment is compared, so `0.5.23.dev4` and
  `0.5.23+rocm700` both count as `0.5.23`. An image tag that names no version
  (a nightly, `latest`, `dev-*`) stands for the framework's latest release when the
  base was resolved. An installed version that cannot be read is refused
  (`framework_version_unknown`), an older one too (`framework_version_too_old`).
- **Flags.** The composed launch is parsed by the installed framework's own argument
  parser, after every patch on the server's path has taken effect. A flag the parser
  does not know is refused, never dropped (`flag_unknown`). A candidate's flag fails
  the candidate; a flag from the base or the operator stops the session.

A round against a server it did not start (`external`) runs neither check.

## Launch layers

A round's launch is composed in a fixed order, and `launch.json` records the layer
that set each flag and environment variable and what it overrode:

1. the frozen base;
2. the operator's layer: removals, then additions;
3. the candidate's layer: removals, then additions;
4. the golden layer, when the round applies it;
5. the guard: the model path, address and GPU mask of the round;
6. the runner's layer: the environment and floors a host needs;
7. the profiler layer of a profile round.

Neither the operator's nor the candidate's layer may set what the guard or the
runner owns, nor, under a golden layer, what golden owns
(`golden_owned_touched`); such a layer is refused, not overridden.

The golden layer carries a base's golden acceptance length (speculative decoding
draft and simulation settings) as its own record. An AgentX session runs a GSM8K
accuracy round, on its own server and without the golden layer, at the baseline and
at each candidate that passes the KEEP threshold. `hyperloom optimize
--no-agentx-accuracy` turns it off for the session.

## Server lifecycle

A round holds its server in one of four modes:

| Mode | Behavior |
|---|---|
| `own` | Starts the server, logs to the run directory and tears it down when the round ends. |
| `boot` | Starts the server with its live log and `server.pid` in a separate `pid_dir`, and leaves it running for later rounds. |
| `reuse` | Runs against the server `pid_dir` records, once it has shown it owns that server (`reuse_not_owned` otherwise). |
| `external` | Runs the client only, against a server something else starts and stops. Multi-node rounds and rounds given a `BENCHMARK_BASE_URL` use it. |

A server is ready when `/health` answers 200, its process is alive and its log
shows a ready marker. Every child runs in a session of its own with a run mark in
its environment; reaping reaches the child's tree and every process carrying that
mark. Nothing is matched by port or command line. `stop` reaps the server a
`pid_dir` records and keeps `server.pid` while a process outlives SIGKILL.

## Staging and pins

A round fetches nothing. The InferenceX client, at the commit
`src/inference_suite/pins.py` pins, and the AgentX harness, in its own virtual
environment, are staged ahead of time into a tool cache with one
`staging_manifest.json`. A round refuses with `staging_missing` when the manifest
lists a file that is absent, changed or staged at another commit, or when an eval
round finds no `lm_eval` installed in the Python that runs it.

`install.sh` stages the cache at `INFERENCE_SUITE_TOOL_CACHE`, which defaults to
`inference-suite-tools` under the dependency cache. To stage or check by hand:

```bash
python -m inference_suite.staging stage --source <InferenceX checkout> --tool-cache <dir>
python -m inference_suite.staging check --tool-cache <dir>
```

## Command line

`python -m inference_suite` has three commands:

| Command | Behavior |
|---|---|
| `run --config <config.yaml> --run-dir <dir>` | Runs one round into an absent or empty directory and prints the `report.json` path. SIGTERM and SIGINT cancel the round, which still reaps what it started and writes its report. |
| `stop --pid-dir <dir>` | Reaps the server the directory's `server.pid` records and prints `{"reaped": [...], "survivors": [...]}`. |
| `resolve --query <query.json> --gpus <n> --out <base_launch.json> [--user-file <file>]` | Reads the sources once and writes the base they give, printing its digest. |

Exit codes: `0` when the round completed (or nothing survived `stop`, or a base was
written); `1` when the round did not complete, or a process outlived `stop`; `2`
when the config, query or pidfile breaks its contract, the run directory is not
empty, or no source holds a base, in which case each source's reason is printed to
stderr.

## Run directory

One round writes one run directory and leaves it unchanged once the round ends.

| Path | Content |
|---|---|
| `config.yaml` | The round's config (`inference-suite-config/1`), stored verbatim. |
| `launch.json` | The exact launch (`inference-suite-launch/1`), with the base it was composed on. |
| `report.json` | What the round measured, or the step it failed at (`inference-suite-report/1`). Written last. |
| `server.log` | The server's log. A `boot` or `reuse` round copies the bytes logged during the round and records their range. |
| `server_ready_at` | `<ready unix time> <boot seconds>`; `0` boot seconds for a reused server. |
| `server.pid` | An `own` round's pidfile while its server runs; `boot` and `reuse` keep theirs in `pid_dir`. |
| `phase_marks.json` | The instants the round passed from one phase to the next. |
| `kv_metrics.json`, `agentx_timeline.jsonl` | KV-cache samples and the AgentX request timeline. |
| `gpu_samples.csv`, `gpu_power.json` | GPU samples over the measured window, and the power summary derived from them. |
| `client/` | `stdout.log`, `stderr.log`, `agentx_client.log`, `inferencex_result.json` and `aiperf_artifacts/` (`profile_export.jsonl`, `logs/aiperf.log`). |
| `eval/` | `stdout.log`, `stderr.log`, lm-eval's `results_<time>.json`, `hyperloom_eval_probe.json` and `definition.json`. |
| `profile/` | `torch_trace/`, `capture_traces/`, `graph_capture_profile/` and `capture.json`. |

Hyperloom writes `gpu_metrics.json` from `report.json` into its own round
directory, never into the run directory.

## report.json

`run_status` is one of:

| Status | Meaning |
|---|---|
| `completed` | Every step of the round ran. |
| `partial` | The load measured, and a later step (eval or profile) failed. |
| `failed` | The round failed before the load measured. |
| `stopped` | The round was cancelled. |

A round that did not complete names one `failed_step`:

| Stage | `failed_step` |
|---|---|
| Before start | `config_invalid`, `base_digest_mismatch`, `framework_version_too_old`, `framework_version_unknown`, `reuse_not_owned`, `eval_context_too_short`, `staging_missing`, `source_files_changed`, `flag_unknown`, `golden_owned_touched`, `port_in_use` |
| Server | `server_exited_before_ready`, `server_silent_before_ready`, `server_boot_timeout`, `server_stalled`, `server_died` |
| Client | `client_failed`, `client_output_missing` |
| Eval | `eval_failed` |
| Any | `round_timeout`, `deadline_exhausted`, `cancelled`, `disk_full`, `teardown_incomplete` |

Every metric key is fixed: a metric nobody measured is `null`, never `0`. An eval
records its definition digest, and two scores compare only under one definition.
