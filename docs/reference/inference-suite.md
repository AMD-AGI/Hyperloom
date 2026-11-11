---
myst:
    html_meta:
        "description": "Reference for InferenceSuite, the package that resolves a session's base launch and runs one benchmark round: recipe sources, framework checks, launch layers, server lifecycle, staging, the run directory and report.json."
        "keywords": "Hyperloom, InferenceSuite, inference_suite, base launch, recipe, InferenceX, SGLang cookbook, vLLM recipes, ATOM, AgentX, run_round, report.json, staging, lm-eval, GSM8K, AMD GPU, ROCm"
---
# InferenceSuite reference

InferenceSuite (`src/inference_suite/`) resolves a session's base launch and runs
one benchmark round. It runs standalone as `python -m inference_suite` with the
commands `resolve`, `run`, `stop`, `stage` and `publish`, and `install.sh
<tool-cache-dir>` stages the tools a round runs. It builds only on
`hyperloom_common` and third-party libraries. It records what a round measured;
thresholds, KEEP decisions and validity belong to its caller.

## Base launch

A fresh session resolves its base launch once, at launch, and freezes it as
`<session>/base_launch.json` (schema `hyperloom-base-launch/1`). A resume reads
that file and resolves nothing. Sources are tried in this order, and the first that
holds a launch for the session's model, board, workload, concurrency, TP and EP
gives the base:

1. the operator's own launch file (`resolve --user-file`): a shell launch script, or a
   `base_launch.json` frozen by an earlier session; a script's credential exports
   (`HF_TOKEN`, `*_API_KEY`, ...) are recorded as their sha256 only, and a round
   takes each value from its host, which must hold the value the script set;
2. InferenceX's AMD single-node master entries;
3. the framework's cookbook: the SGLang cookbook, the vLLM recipes site, or
   ATOM's catalog (fixed workloads only);
4. for a fixed workload only, the default table shipped in
   `src/inference_suite/recipe/defaults/table.yaml`.

An AgentX session that no recipe and no operator launch file serves is refused.
When a source cannot be read, a fixed session runs the default table, unless
InferenceX is pinned (below), and an AgentX session is refused. A pinned fixed
session no recipe serves is refused too, naming why each source missed. Repositories are read at the head of their `main`. The base's
`upstream` list records the revision of every source read, whether or not it gave
the launch: the commit of InferenceX, the SGLang cookbook and ATOM's catalog, the
commit of the Hugging Face model whose card was read, and, since the vLLM recipes
site publishes no commit and its renderings carry no ETag, the URL and body sha256
of each response it gave. `upstream` is not part of the digest, and a
`base_launch.json` written without it reads with an empty list.

To match a run made at an older InferenceX, pin it with `resolve --inferencex-commit
<sha>`: a full sha, or a prefix of 7 or more hex digits of a commit on one of
InferenceX's branches or at the head of one of its refs, a pull request's
included. The base records the full sha it read. Only InferenceX is pinned; the
cookbooks, the vLLM recipes site and ATOM's catalog are still read at `main`. A
resume keeps its frozen base whatever the flag says. The commit is refused, with
exit 2, when InferenceX has no such commit or the prefix names more than one, when
any source cannot be read while it is pinned, InferenceX and its own code
included (the message names the source and keeps git's error; a missing commit
and an unreachable remote are told apart; no default table is taken), and
when the commit lacks a function the resolver runs from InferenceX's own code
(for example, a commit without `acceptance_length`, `generation_role` and
`verifier` in `infx.srt_slurm.synthetic_acceptance`); the message names what it
lacks. To run on such a commit, resolve the base where its code was run and hand
the frozen `base_launch.json` to `resolve --user-file`.

A launch script is read, never run. It must start one server of the session's
framework, on a line that is not a comment, and fix every flag in its own text: a
second server, a flag whose value names a shell variable, or a
`# hyperloom-launch-controls:` line, is refused. Options are read with one dash or
two (`-tp 8`, `--tp 8`).

A launch script, frozen base or default row that needs more GPUs than `--gpus` is
refused, naming both counts.
A frozen base resolved for another model, workload or framework, or for a board
of another architecture or HBM capacity than the session's, is refused, naming both
values.

An InferenceX entry of the session's model that its config cannot place (a runner
on no board label, a missing key) is that entry's miss; entries of other models are
not read.

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
  `0.5.23+rocm700` both count as `0.5.23`. An installed version that cannot be
  read is refused (`framework_version_unknown`), an older one too
  (`framework_version_too_old`). An image tag that names no version (a nightly,
  `latest`, `dev-*`) is not held: the base records the framework's latest release
  when it was resolved as an assumption (`version_basis: assumed_latest`), and
  `launch.json` records the installed version beside it with `version_check:
  assumed_not_held`.
- **Flags.** The composed launch is parsed by the installed framework's own argument
  parser, after every patch on the server's path has taken effect. A flag the parser
  does not know is refused, never dropped (`flag_unknown`). A candidate's flag fails
  the candidate; a flag from the base or the operator stops the session.

A round against a server it did not start (`external`) runs neither check.

The server runs in the same interpreter the checks ask: `<framework_python> -m
sglang.launch_server`, `-m vllm.entrypoints.cli.main serve` or `-m
atom.entrypoints.openai_server`, never the base's entrypoint (`python3 -m ...`,
`vllm serve`) through `PATH`. `launch.json` records that command as `entrypoint`
and at the head of `argv`; the base and its digest keep the entrypoint the source
spelled. A base whose entrypoint is not its framework's is refused
(`config_invalid`); name another install with the server's `framework_python`
instead.

## Launch layers

A round's launch is composed in a fixed order, and `launch.json` records the layer
that set each flag and environment variable and what it overrode:

1. the frozen base;
2. the operator's layer: removals, then additions;
3. the candidate's layer: removals, then additions;
4. the golden layer, when the round applies it;
5. the guard: the model path, address and GPU mask of the round;
6. the runner's layer: the environment and floors a host needs, and the
   framework's metrics switch (`--enable-metrics` for SGLang; vLLM needs none)
   when the round samples KV telemetry;
7. the profiler layer of a profile round.

Neither the operator's nor the candidate's layer may set what the guard or the
runner owns, nor, under a golden layer, what golden owns
(`golden_owned_touched`); such a layer is refused, not overridden.

The golden layer carries a base's golden acceptance length (speculative decoding
draft and simulation settings) as its own record. An AgentX session runs a GSM8K
accuracy round, on its own server and without the golden layer, at the baseline and
at each candidate that passes the KEEP threshold.

An AgentX round replays the trace corpus for a fixed profiling duration, as
InferenceX does: the point's `DURATION` in seconds, 3600 when it sets none, passed
as `--benchmark-duration`. The harness's `agentx` scenario holds the replay to 95%
latency coverage of that duration, so no request count bounds it. Below 900 s, the
scenario's minimum, the round adds `--unsafe-override`; use 600 s or 900 s for
quick tests.

## Server lifecycle

A round holds its server in one of four modes:

| Mode | Behavior |
|---|---|
| `own` | Starts the server, logs to the run directory and tears it down when the round ends. |
| `boot` | Starts the server with its live log and `server.pid` in a separate `pid_dir`, and leaves it running for later rounds. |
| `reuse` | Runs against the server `pid_dir` records, once it has shown it owns that server (`reuse_not_owned` otherwise). |
| `external` | Runs the client only, against a server something else starts and stops. Multi-node rounds and rounds given a `BENCHMARK_BASE_URL` use it. |

Before `own` or `boot` starts a server, the round refuses a port something already
listens on (`port_in_use`) and GPUs that `amd-smi` does not list or that another
process holds (`gpus_unavailable`, naming each GPU, its amd-smi card and its holder; a
round's GPU numbers are `HIP_VISIBLE_DEVICES` numbers, matched to amd-smi cards through
their KFD node). A process holds a card when `amd-smi process` reports more than 1 GiB
of VRAM for it there, or reports none at all: every ROCm process opens all cards and
keeps up to a few MiB of page tables and runtime context on each, which does not count.

A server is ready when `/health` answers 200, its process is alive and its log
shows a ready marker. Every child runs in a session of its own with a run mark in
its environment; reaping reaches the child's tree and every process carrying that
mark. Nothing is matched by port or command line. `stop` reaps the server a
`pid_dir` records and keeps `server.pid` while a process outlives SIGKILL.

## Staging and pins

A round fetches nothing. The InferenceX client, at the commit
`src/inference_suite/pins.py` pins, and the AgentX harness, installed with
`--no-deps` from `src/inference_suite/agentx-client.lock` into the relocatable
CPython 3.12 build `pins.py` pins, are staged ahead of time into a tool cache with
one `staging_manifest.json`. InferenceX's files are read from the pinned commit's tree (`git archive`),
never from a checkout's working tree: a `--source` checkout's local edits, untracked files and later commits are
not staged, and a checkout that lacks the pinned commit is refused. Staging patches the harness's aiperf so a reply it found no content in exports the
server's output token count as `error_osl`; a harness the patch does not apply to fails the staging, and a cache
staged before the patch is staged again. Two hosts compare stagings by their normalised
manifest: each installed package's name, version and wheel sha256, the
interpreter's version and build sha256, the InferenceX pin and lm-eval, with no
path. A session's frozen base launch pins its client at that same InferenceX
commit; a measurement sheet names it, and the result manifest a runner signs
repeats it. `python -m inference_suite publish --stagings <dir>` publishes a staging
shared by several hosts as `<dir>/<staging key>` by rename. A round refuses with `staging_missing` when the manifest
lists a file that is absent, changed or staged at another commit, or when an eval
round finds no `lm_eval` installed in the Python that runs it.

`install.sh <tool-cache-dir>` stages the cache at `<tool-cache-dir>` and installs the pinned
lm-eval into the Python rounds run under (`PYTHON`, default `python3`). lm-eval and its dependency tree come from the hash lock
the sealed measurement runner installs, `requirements/runner.lock` (`runner-cpXY.lock` for a
Python that has its own), which `inference_suite` ships as package data: pip installs only
distributions that lock pins, each matching a hash it lists, and an install missing the lock
refuses with `staging_missing`. `check` fails with `staging_missing`
when the cache does not verify or that Python lacks the pinned lm-eval; `stage`
without `--source` fetches nothing and installs lm-eval beside a cache that
verifies. To stage or check by hand:

```bash
python -m inference_suite.staging stage --source <InferenceX repository> --tool-cache <dir>
python -m inference_suite.staging stage --tool-cache <dir>
python -m inference_suite.staging check --tool-cache <dir>
```

## Command line

`python -m inference_suite` has five commands:

| Command | Behavior |
|---|---|
| `run --config <config.yaml> --run-dir <dir>` | Runs one round into an absent or empty directory and prints the `report.json` path. A relative path, on the command line or in the config, is read below the directory `run` starts in; a relative `model_path` only when it names something there, otherwise it is a hub id. SIGTERM and SIGINT cancel the round, which still reaps what it started and writes its report. |
| `stop --pid-dir <dir>` | Reaps the server the directory's `server.pid` records and prints `{"reaped": [...], "survivors": [...]}`. |
| `resolve --query <query.json> --gpus <n> --out <base_launch.json> [--user-file <file>] [--inferencex-commit <sha>]` | Reads the sources once and writes the base they give, printing its digest. With `--inferencex-commit`, InferenceX is read at that commit instead of the head of its `main`. |
| `stage --tool-cache <dir> [--source <InferenceX repository>]` | Stages the pinned tools into the tool cache unless it holds them intact, fetching InferenceX at its pin without `--source`, and prints the manifest's sha256. |
| `publish --stagings <dir> [--source <InferenceX repository>]` | Publishes the staging of the pins in force as `<dir>/<staging key>`, staging it there when absent, and prints its path. |

Exit codes: `0` when the round completed (or nothing survived `stop`, a base was
written, or a staging is in place); `1` when the round did not complete, or a
process outlived `stop`; `2` when the config, query or pidfile breaks its contract,
the run directory is not empty, no source holds a base, in which case each
source's reason is printed to stderr, `--inferencex-commit` is refused, a source
answers in a form resolve cannot read (an HTML page where JSON was asked), or a staging cannot be built or verified,
printed as `staging_missing: <reason>`.

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
| `power.json` | Every power reading the node gave over the whole round; see [Power](#power). |
| `client/` | `stdout.log`, `stderr.log`, `agentx_client.log`, `inferencex_result.json`, `aiperf_artifacts/` (`profile_export.jsonl`, `logs/aiperf.log`) and, on an MLPerf round, `mlperf_artifacts/` (the harness's `result_summary.json` and `scores.json`). |
| `eval/` | `stdout.log`, `stderr.log`, lm-eval's `results_<time>.json`, `hyperloom_eval_probe.json` and `definition.json`. |
| `profile/` | `torch_trace/`, `capture_traces/`, `graph_capture_profile/` and `capture.json`. |

A host writes `gpu_metrics.json` from `report.json` into its own round
directory, never into the run directory.

## Power

A round with GPU telemetry on writes `power.json` (`schema_version` 2): one file
holding every power reading the node gives, from the start of the round to its
end. The config's `telemetry.gpu.interval_sec` sets the sysfs interval: below
0.5 s it is raised to 0.5 s, and above 1.5 s the config is refused, since samples
more than 3 s apart invalidate the window's energy.

| Source | Read | Recorded |
|---|---|---|
| sysfs, each second | every card's `hwmon` power, junction and memory temperature, sclk, mclk, board voltage; `gpu_busy_percent`, `mem_busy_percent`, VRAM and GTT use | `samples.fast` |
| `amd-smi metric -E`, at each phase mark and both ends | each card's energy accumulator (J) | `samples.energy` |
| `amd-smi metric`, every 10 s and at each phase mark | every field, flattened to dotted key paths, with units | `samples.full` |
| NPM `board/node_power`, `cur_node_power_limit`, each second | the node's power and its enforced limit | `samples.node` |
| RAPL powercap zones, else `amd_energy` hwmon, each second | CPU energy counters (µJ) | `samples.cpu` |
| `amd-smi static` and sysfs, at both ends | caps (`power1_cap*`), performance level, `pp_dpm_*` tables, the NPM board files | `static.start`, `static.end` |

Power is read from sysfs because a card can read `N/A` in `amd-smi` while its
`hwmon` reads. A source the node does not have is `null`, with its reason under
`recorder.sources`; a read that fails is counted under `errors` and never fails
the round. The file is rewritten every 60 s while the round runs (`final:
false`), so a killed round leaves its samples.

`phases` summarizes each phase between the marks the round passed: per card the
power distribution, energy, temperatures, clocks, busy percentages, peak VRAM
share and the share of firmware ticks each throttle accumulator counted; the
node's and the CPU's power where read. Each phase begins at the instant its mark
carries, not when the recorder heard it.

`energy` is the measured window's, computed as InferenceX does. A fixed round's
window is the client's own timed run (`benchmark_start_time_unix` to
`benchmark_end_time_unix` in its result), so its tokenizer load, prompt
building and result writing fall outside it; an AgentX round's is aiperf's
profiling phase. Each configured
card's power is interpolated at the window's boundaries and integrated by
trapezoid, giving `total_gpu_energy_j`, `avg/p75/p90_power_w` per card and
`avg/p75/p90_total_gpu_power_w` (time-weighted quantiles of the summed curve),
and `joules_per_input_token`, `joules_per_output_token`,
`joules_per_total_token` and `joules_per_successful_query` from the load's
counts. `power_valid` is `1` when the energy can be trusted; otherwise every
figure is `null` and `reasons` says why:

| Reason | Meaning |
|---|---|
| `invalid_benchmark_window` | The round never entered its measured phase. |
| `benchmark_duration_mismatch` | The client's stamps and the duration it timed differ by more than 0.5 s or 1%. |
| `benchmark_window_not_bracketed`, `insufficient_power_samples` | A card has no sample on one side of the window. |
| `sampling_gap_exceeded` | A card's samples across the window, its boundaries included, are more than 3 s apart. |
| `invalid_power_sample` | A card's power did not read inside the window. |
| `expected_gpu_count_mismatch` | A configured card holds no model or was never read, or the round's GPUs could not be matched to amd-smi cards. Cards outside the configured ones are sampled but never counted, so another server on them does not invalidate the round. |
| `external_server` | The load ran on a server this host does not run; nothing local is its energy. |
| `aborted` | The round stopped on a failed step. |

A load without a count (`invalid_input_token_count` and its two siblings) nulls
only the figure it feeds: a scriptable round counts no tokens, an MLPerf round
no input tokens. The cards' accumulator delta is compared with the integral over
the same span (`accumulator_check`); beyond 5% it is the warning
`accumulator_mismatch`, not an invalid energy. `node_energy_j` and
`cpu_energy_j` cover the same window.

`report.json` carries the summary in its `power` section: the energy figures,
`per_gpu_avg_power_w`, `accumulator_relative_error`,
`power_throttle_fraction_max`, `power_valid`, `reasons` and `warnings`.
`gpu_monitor` is derived from the same samples. `power.json`, the `power` section
and `gpu_monitor` number cards by amd-smi index, not by the round's
`HIP_VISIBLE_DEVICES` numbers.

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
| Before start | `config_invalid`, `base_digest_mismatch`, `framework_version_too_old`, `framework_version_unknown`, `reuse_not_owned`, `eval_context_too_short`, `staging_missing`, `source_files_changed`, `flag_unknown`, `golden_owned_touched`, `port_in_use`, `gpus_unavailable` |
| Server | `server_exited_before_ready`, `server_silent_before_ready`, `server_boot_timeout`, `server_stalled`, `server_died` |
| Client | `client_failed`, `client_output_missing` |
| Eval | `eval_failed` |
| Any | `round_timeout`, `deadline_exhausted`, `cancelled`, `teardown_incomplete` |

Every metric key is fixed: a metric nobody measured is `null`, never `0`. An eval
records its definition digest, and two scores compare only under one definition.

