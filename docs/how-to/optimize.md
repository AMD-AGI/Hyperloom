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
| `CONC` | Concurrent requests — baseline benchmark concurrency (`--conc`, default `64`) | Set to your target concurrency. The SWEEP phase separately measures a ladder around it: `256,128,64,32,16,8,4,2` for a synthetic workload. An AgentX session runs one fixed concurrency. Override the ladder with `--conc-sweep-concs`. |
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
`PRELUDE → ENABLEMENT → FRAMEWORK_AGENT → KERNEL_AGENT → SWEEP → CLOSE`; see
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
4. A built-in default table, with a warning (fixed workloads only).

`--inferencex-commit SHA` resolves from InferenceX at that commit, a full sha
or a prefix of 7 or more hex digits, for example to match a CI run; the full
sha is recorded in `base_launch.json`. Only InferenceX is pinned: the
cookbooks are still read at `main`. The launch exits 2 when InferenceX has no
such commit, when the commit lacks a function the resolver runs from
InferenceX's own code (naming what it lacks), or when any source cannot be read
(naming the source and its error, git's included): a pinned session never runs
on the default table. To use such a commit, resolve the
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

This path is **single-node** SGLang or vLLM: run Hyperloom inside your serving
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

AgentX rejects `--enable-conc-sweep`, `--isl`, `--osl`, `--profile-osl`,
`--max-model-len`, `--quantize` and `--quantize-scheme`.
It also rejects multi-node execution and frameworks other than SGLang/vLLM.
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
candidate passes the same workload and Hyperloom's KEEP gate.

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
a new session. Do not combine this entrypoint with CLI ISL/OSL, concurrency sweeps,
quantization or benchmark-script overrides. The independent MLPerf agentic backend
keeps its existing entrypoint.
