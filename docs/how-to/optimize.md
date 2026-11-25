---
myst:
    html_meta:
        "description": "Step-by-step guide to running a Hyperloom optimization. Covers launching from Claude Code or Codex, monitoring, resuming, and reading output artifacts."
        "keywords": "Hyperloom, optimization, how-to, LLM inference, AMD GPU, ROCm, Claude Code, Codex, GEAK, TraceLens, session, throughput"
---
# Run a Hyperloom optimization

This topic assumes you have already completed installation. If you haven't, follow the [Hyperloom installation instructions](../install/install.md) then return here to launch your first run.

## Launch from Claude Code or Codex

Open the Hyperloom workspace in Claude Code or Codex, then paste the following
prompt into the agent chat, filling in your workload details:

```{note}
The prompt includes `install.sh`. This is intentional: the agent runs in its own
shell process, which does not inherit the environment you sourced during
installation. The agent must re-source the env files and re-run `install.sh` in
its own context before launching the optimizer. Because `install.sh` is
idempotent, the second run is fast and safe.
```

```{note}
Paths in the prompts on this page follow the recommended `pip install --target .`
layout. In a source checkout, replace the `hyperloom/` prefix with
`src/hyperloom/`.
```

```text
@hyperloom/inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: sglang
- GPU: MI300X
- TP: 1
- CONC: 64
- ISL: 1024
- OSL: 1024
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Before launch, run exactly:
export REPO_ROOT="$(pwd -P)"
export USER_DATA_PATH='/path/to/hyperloom-run'
bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"
# The optimizer preflight loads the generated runtime environment in process.

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Read persisted state on requested status checks; report completion or failure. Do not start a watchdog or automatic resume loop.
```

| Field | Meaning | How to choose |
|-------|---------|---------------|
| `TP` | Tensor-parallel size — number of GPUs the model is sharded across | Must match the number of GPUs in your server node (for example, `8` for a single 8-GPU MI300X node) |
| `CONC` | Concurrent requests — baseline benchmark concurrency (`--conc`, default `64`) | Set to your target concurrency. An AgentX session runs one fixed concurrency. |
| `ISL` | Input sequence length — tokens in each request's prompt | Match your production workload; `1024` is a common starting point |
| `OSL` | Output sequence length — tokens generated per response | Match your production workload; `1024` is a common starting point |

```{note}
`ISL` / `OSL` describe the synthetic request shape. Under the opt-in AgentX
trace-replay mode (`HYPERLOOM_AGENTX=1` or `--benchmark-config agentx.yaml`) request
lengths come from the recorded trace corpus instead, so these two values do not
affect what is measured — the server's context window comes from the session's
base launch.
```

See [`src/hyperloom/inference_optimizer/SKILL.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)
for the full prompt field reference (every field maps to a CLI flag defined in
`cli/parser.py`, in `cli/constraint_args.py` for the latency and power budgets,
or in `cli/gpu_power_args.py` for the GPU power settings).

## Monitor the run

The agent reports a session ID, log path, and PID, then reads persisted state
on requested status checks. Recurring checks may use the hosting platform's
scheduled invocations; no background supervisor or automatic restart is started.
Logs are useful evidence, but activity alone does not prove useful progress.
Under the hood the optimizer walks the phase chain
`PRELUDE → ENABLEMENT → FRAMEWORK_AGENT → KERNEL_AGENT → CYCLE_END → CLOSE`; see
[Hyperloom optimization loop](../conceptual/optimization-loop.md) for each phase
and [benchmark deadlines](../reference/environment-variables.md#benchmark-deadlines-and-lifecycle)
for the independent benchmark and session limits.

## Resume an interrupted session

Paste this prompt into the agent chat to resume an existing session:

```text
@hyperloom/inference_optimizer/SKILL.md

Resume the existing Hyperloom optimization session.

Requirements:
1. Launch `python -m hyperloom optimize --resume-from "$SESSION_DIR"`; do not start a new session.
2. Do not pass `--model`; read the model and workload from the saved manifest.
3. Resolve `$SESSION_DIR` from the launch-info JSON or the `HYPERLOOM_LAUNCH` line, never from the newest timestamp dir.
4. Before launching, verify `manifest.json` and `state.json` exist.
5. Report the log path, PID, health check, current phase, cumulative gain, and best config.
6. Read persisted state on requested status checks; report completion or failure. Do not start a watchdog or automatic resume loop.
```

## Select plug implementations

A plain `optimize` run needs none of this: it uses the built-in `local` plug profile, writes nothing extra to the
session and prints nothing about plugs. To choose other implementations of a plug socket, pass any of:

| Flag | Effect |
|------|--------|
| `--plug-profile NAME` | Load a built-in profile (`local`, `hosted-gpu`, `hosted-cpu`). Not `--profile`, which stays the abbreviation of `--profile-osl`. |
| `--manifest PATH` | Lay a JSON manifest over the profile. `$HYPERLOOM_MANIFEST` is the same, from the environment. |
| `--plug SOCKET=IMPL` | Pick one implementation for one socket (repeatable). |
| `--plug-config SOCKET.KEY=VALUE` | Set one configuration key (repeatable; a JSON value is read as JSON). |

Later layers win: profile, manifest, `$HYPERLOOM_MANIFEST_OVERRIDES`, then `--plug` and `--plug-config`. See
[Plug selection](../reference/environment-variables.md#plug-selection) for the two variables.

A run that selects anything prints one `Plugs : profile=... fingerprint=...` line and records the outcome in
`$SESSION_DIR/plugs.resolved.json`. `--resume-from` reads that file back before applying flags, so the resumed run keeps
the same implementations; pass the flags again only to change them. An exported `$HYPERLOOM_MANIFEST` or
`$HYPERLOOM_MANIFEST_OVERRIDES` is a later layer than that file, so it still changes the selection on a resume.

The selection is resolved and checked before the session is opened or changed. A refused one (an unknown socket or
implementation, a malformed or missing manifest, an unset `${VAR}` without a default, or a damaged
`plugs.resolved.json`) prints `ERROR: plug configuration refused: ...` and exits 2, and a refused `--resume-from`
leaves `state.json` and the rest of the session exactly as they were, so it can be retried with the flag fixed.
Fix or remove a damaged `plugs.resolved.json` to resume.

## Choose the base launch

Every server a fixed-workload or native AgentX session starts begins from one
base launch: the framework's flags and environment that the optimizer's own
changes sit on top of. A fresh session resolves it once, at launch, from the
first source that serves the model on your GPU board:

1. `--reference-script PATH_OR_URL`, when given.
2. InferenceX `main`, or the commit `--inferencex-commit SHA` pins.
3. The framework's cookbook `main` (SGLang cookbook, vLLM recipes, ATOM).
4. A built-in default table, with a warning (fixed workloads without
   `--inferencex-commit` only).

`--inferencex-commit SHA` resolves from InferenceX at that commit, a full sha
or a prefix of 7 or more hex digits, for example to match a CI run; the full
sha is recorded in `base_launch.json`. Only InferenceX is pinned: the
cookbooks are still read at `main`. The launch exits 2 when InferenceX has no
such commit, when the commit lacks a function the resolver runs from
InferenceX's own code (naming what it lacks), when any source cannot be read
(naming the source and its error, git's included), or when no recipe serves
the session (naming why each source missed): a pinned session never runs on
the default table. To use such a commit, resolve the
base where its code was run and pass that `base_launch.json` as
`--reference-script`. A resume keeps its frozen base.

Without `--inferencex-commit`, when a source cannot be read, InferenceX
included, a fixed session runs on the default table and the warning names that
source; an AgentX session exits 2. An AgentX session that no source serves
exits 2 and names why each source missed. When `--tp` or `--ep`
is omitted, the session runs at the
topology the base chose. The base is frozen as `$SESSION_DIR/base_launch.json`
and its warnings print in the startup banner; `--resume-from` reads that file
and never refetches. A session whose state predates `base_launch.json` cannot
be resumed: start a new one.

`--reference-script` takes a shell launch script or a `base_launch.json` from
an earlier session. A script must name the session's framework entrypoint and
fix every flag in its text. These exit 2:

- a flag whose value is a shell variable, such as `--tp $TP`;
- a `# hyperloom-launch-controls:` line;
- a file that cannot be read or launches another framework.
- a launch that needs more GPUs than the session has.

`--host`, `--port`, `--served-model-name` and `--result-*` are dropped and
recorded under `stripped`, because the runner sets them every round.
`$SESSION_DIR/current_setting.sh` re-feeds as a launch file unless it carries
launch controls (an overlay, unset envs, removed args or replaced args).

---

## Output and artifacts

When the loop exits, Hyperloom writes the final report, reproducible session
artifacts, and `session_breakdown.json` to your session directory. The three
fields to read first are:

| Field | What it tells you |
|-------|-------------------|
| `final.output_throughput` | Validated end-of-session serving throughput — the headline number for SGLang / vLLM |
| `final.cumulative_gain_pct_validated` | Validated gain over baseline |
| `final.action_path` | Ordered list of changes that make up the final optimized stack |

For the full schema — useful if you are building a dashboard, reporting
pipeline, or downstream integration on top of this file — see
[`session_breakdown.json` integration in Hyperloom](../reference/session-breakdown.md).

## Sweep concurrency after the session

A concurrency sweep runs once the session closes, apart from its time budget
(on by default; `--no-enable-conc-sweep` turns it off); `hyperloom conc-sweep SESSION_DIR` runs the same sweep by
hand on a finished session. It runs only when the session kept a validated
gain, and measures the baseline and the optimized launch at every concurrency
the session's InferenceX recipe publishes for its workload (its TP and EP, and
its ISL/OSL or agentic scenario), with the session's own `--conc` among them.
`--conc-sweep-concs C1,C2,...`, on `optimize` or on `conc-sweep`, sweeps those
concurrencies instead; a base from `--reference-script` has no recipe points and
needs them.

Each point launches as the recipe launches its own concurrency: the base the
session froze holds the recipe's launch at every concurrency of its arm
(`point.source_launches` in `base_launch.json`), and a point runs that launch's
args and environment in place of the session's own, with the optimization
stacked on top (what it sets wins; what it removes stays removed). A concurrency
the recipe does not publish runs the launch of the nearest higher one, else its
highest. An AgentX point boots a server of its own; a fixed-length workload
keeps one server across consecutive points that launch alike. A point runs the
request cap the recipe sets at its concurrency (`--max-num-seqs`,
`--max-running-requests`) even below that concurrency, as InferenceX measures it:
an AgentX concurrency counts agent sessions, not requests in flight.

A multi-branch session runs a concurrency's baseline and optimized point side by
side, within `--max-parallel-tests`; `--conc-sweep-parallel N` runs up to N of
the sweep's tests at once, 1 one at a time. A single-lineage session runs one
point at a time: arm by arm on held servers, and a concurrency's two arms one
after the other where each point boots its own server, so a deadline leaves
pairs measured.

The sweep logs its estimated length before it starts, rewrites
`reports/conc_sweep_summary.json` and `reports/conc_sweep_raw.csv` after every
point, and draws `reports/conc_sweep_curve.png` (interactivity against
throughput per GPU) at the end. The summary's header names the framework,
precision, GPU type and count, TP and EP, the image and the recipe the base came
from, and the host and process writing it with `updated_at`, so a summary still
`in_progress` whose process is gone reads as cut short. Its `status` is
`succeeded`, `partial`, `failed`, `skipped` (with `skip_reason`) or `cancelled`
(SIGINT or SIGTERM; the measured points stay). Given a deadline (a hosted run's
grant), a point that would end past it is not started and reads
`skipped_budget`. A later run measures only the points not yet measured on the
same launch of each arm. Multi-node and MLPerf sessions are skipped, and so are a
session whose server is built from its checkout (`checkout_built_server`) and one
whose retained overlay cannot load (`optimized_overlay_unavailable`).
The session package is rebuilt after the sweep so it carries the report.

The latency columns follow InferenceX's own reports: for an AgentX workload its
client computes `itl` and `tpot` from the same per-request inter-token latency
(`infx/results/agentic/request_metrics.py`), so `tpot_p90_ms` is the ITL p90.

A hosted run sweeps a closed session with a campaign spec of `flow:
"conc_sweep"`: `conc_sweep.session_dir` names the session (an absolute path the
run can read and write; the report is written back into it), `conc_sweep.concs`
its points and `conc_sweep.parallel` the tests at once of a multi-branch
session. The run optimizes nothing and ends `final: true` however the sweep
ends; to finish a `partial` or `cancelled` sweep, start another such run on the
same session, which measures only the points not yet measured. A single-lineage
session is measured on the run's own cards.

## Troubleshooting

If a run fails on first launch, see [Troubleshooting Hyperloom](../reference/troubleshooting.md).

## Optimize an AgentX workload

Use the ordinary Hyperloom workload flags with `HYPERLOOM_AGENTX=1`. Hyperloom
writes them into the session's AgentX workload, `$SESSION_DIR/benchmark.yaml`:

```bash
HYPERLOOM_AGENTX=1 hyperloom optimize \
  --model deepseek-ai/DeepSeek-V4-Pro-0813 \
  --framework sglang --precision fp4 \
  --tp 8 --ep 8 --conc 32 \
  --max-hours 12
```

This path is **single-node** SGLang, vLLM or ATOM: run Hyperloom inside your serving
image. `--docker-image IMAGE` records that runtime image; it does not launch a
container. Custom models require an explicit `--docker-image`.

The server starts from the session's base launch (see
[Choose the base launch](#choose-the-base-launch)). An AgentX session that neither
`--reference-script`, InferenceX nor the framework's cookbook serves is refused,
and so is a serving image whose framework is older than the recipe's image.
`--server-args` and serving `--extra-env` pins form the operator's layer on the
base, and candidate changes sit on top. A flag the installed framework does not
know is refused, never dropped. Request lengths come from the AgentX trace corpus;
concurrency stays fixed for the session.

AgentX rejects `--isl`, `--osl`, `--profile-osl`,
`--max-model-len`, `--quantize` and `--quantize-scheme`.
It also rejects multi-node execution and frameworks other than SGLang, vLLM and ATOM.
An ATOM session takes its base from InferenceX's ATOM agentic entries; ATOM's own catalog serves
fixed workloads only.
Through `--extra-env`, `ISL`, `OSL`, `RANDOM_RANGE_RATIO`, `MAX_MODEL_LEN`,
`EP_SIZE` and keys beginning with `AIPERF_`, `AGENTX_` or `AGENTIC_` are unsupported;
use `--ep` for expert parallelism. Prepare any quantized checkpoint before this run,
and start a separate session for another concurrency.

To define the workload in a file, pass a YAML. Start with
`src/hyperloom/inference_optimizer/assets/configs/agentx_sglang.yaml`:

```bash
hyperloom optimize --benchmark-config /workspace/agentx.yaml --max-hours 12
```

The YAML defines the workload, including positive `envs.TP` and `envs.CONC` for
resource accounting. Conflicting CLI workload flags are rejected. Both entrypoints
use the same execution and result-validation path. For a local checkpoint, pass
its path with `--model`, or name it in `envs.MODEL_PATH` while `model` keeps its
canonical identifier. The model must support the replay's context length and
protocol.

Every round runs through [InferenceSuite](../reference/inference-suite.md), which
starts the server, replays the trace, profiles and cleans up. Hyperloom compares
the rounds' `report.json`: a round that did not complete, or whose measurement is
invalid, cannot supply KEEP evidence, and baseline and candidate must have the
same workload identity. GEAK's search measurements are provisional until the
candidate passes the same workload and Hyperloom's KEEP gate. GEAK searches on
the trace the accepted round replayed: the handoff's `workload_spec` declares
the corpus and entry count from that round's aiperf export, the session's
concurrency and the mean request shape the round measured.

When the serving image's kernels run only some draft shapes, declare the bounds
under `speculative_draft_limits` so a candidate outside them is refused before
its server boots. Each of `draft_length` (`--speculative-num-steps`, vLLM's
`num_speculative_tokens`), `draft_tokens` (`--speculative-num-draft-tokens`) and
`topk` (`--speculative-eagle-topk`) takes a positive `min`, `max` or both:

```yaml
benchmark:
  speculative_draft_limits:
    draft_tokens: {max: 4}
```

A setting is judged as the candidate spells it. One the engine resolves for
itself is not judged: with `--speculative-eagle-topk 1`, SGLang verifies
`num_steps + 1` tokens, so bound `draft_length` too when the cap is on the
verify window. A malformed declaration stops the session before it starts.

The baseline, and each candidate that passes the KEEP threshold, also runs a
GSM8K accuracy round on a server of its own. `--no-agentx-accuracy` turns it off
for the session.

A profile round adds the profiler layer and the SGLang kernel-shape tool to the
round's launch, and opens its capture window once the replay reaches measurement.
A missing capture setting becomes one capture of 256 steps after a 300-second start
delay. A profile round captures one window; a YAML whose `num_profiles` is not 1 is
refused when the session starts.
Profiled measurements never replace the unprofiled performance baseline.

`hyperloom optimize --resume-from SESSION_DIR` reuses the archived `benchmark.yaml`
without requiring the switch or original YAML again. Changing the workload requires
a new session. Do not combine this entrypoint with CLI ISL/OSL,
quantization or benchmark-script overrides. The independent MLPerf agentic backend
keeps its existing entrypoint.
