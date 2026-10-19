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
| `CONC` | Concurrent requests — baseline benchmark concurrency (`--conc`, default `64`) | Set to your target concurrency. The SWEEP phase separately measures a ladder around it: `256,128,64,32,16,8,4,2` for a synthetic workload. Native Magpie AgentX uses one fixed YAML concurrency per session. Override the ladder with `--conc-sweep-concs`. |
| `ISL` | Input sequence length — tokens in each request's prompt | Match your production workload; `1024` is a common starting point |
| `OSL` | Output sequence length — tokens generated per response | Match your production workload; `1024` is a common starting point |

```{note}
`ISL` / `OSL` describe the synthetic request shape. Under the opt-in agentic
Magpie AgentX trace-replay mode (`HYPERLOOM_AGENTX=1` or `--benchmark-config agentx.yaml`) request lengths come from the recorded
trace corpus instead, so these two values do not affect what is measured — the
server's context window comes from Magpie's resolved recipe or custom-model
configuration.
```

See [`src/hyperloom/inference_optimizer/SKILL.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)
for the full prompt field reference (every field maps to a CLI flag defined in
`cli/parser.py`, or in `cli/constraint_args.py` for the latency and power budgets).

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
translates them into Magpie's native AgentX YAML and saves it in the session:

```bash
HYPERLOOM_AGENTX=1 hyperloom optimize \
  --model deepseek-ai/DeepSeek-V4-Pro-0813 \
  --framework sglang --precision fp4 \
  --tp 8 --ep 8 --conc 32 \
  --max-hours 12
```

This path is **single-node** SGLang or vLLM. Hyperloom invokes Magpie in local
mode: run Hyperloom inside your serving image. `--docker-image IMAGE` records
that runtime image; it does not launch a container. A registered recipe can
supply its image when the flag is omitted. Custom models require an explicit
`--docker-image`, including local runs.

`--model`, `--framework`, `--precision`, `--tp`, `--ep` and `--conc` use
Hyperloom's existing resolution rules. `--server-args` and serving `--extra-env`
pins become Magpie server launch overrides, with candidate changes applied on
top. ISL/OSL are not injected: request lengths come from the AgentX trace corpus.
Concurrency stays fixed for the session.

Native AgentX rejects `--enable-conc-sweep`, `--isl`, `--osl`, `--profile-osl`,
`--max-model-len`, `--reference-script`, `--quantize` and `--quantize-scheme`.
It also rejects multi-node execution and frameworks other than SGLang/vLLM.
Through `--extra-env`, `ISL`, `OSL`, `RANDOM_RANGE_RATIO`, `MAX_MODEL_LEN`,
`EP_SIZE` and keys beginning with `AIPERF_`, `AGENTX_` or `AGENTIC_` are unsupported;
use `--ep` for expert parallelism. Request lengths and model context come from
Magpie's resolved workload. Prepare any quantized checkpoint before this run,
and start a separate session for another concurrency.

For recipe selectors, profiling settings or other advanced Magpie options,
provide a YAML directly. Start with
`src/hyperloom/inference_optimizer/assets/configs/agentx_sglang.yaml`:

```bash
hyperloom optimize --benchmark-config /workspace/agentx.yaml --max-hours 12
```

The YAML defines the workload, including positive `envs.TP` and `envs.CONC` for
resource accounting. Conflicting CLI workload flags are rejected. Both entrypoints
use the same materialization, execution and result-validation path.

Magpie resolves the pinned InferenceX recipe and owns server startup, replay,
profiling and cleanup. If the model has no registered recipe, Magpie's custom-model
resolver uses the generated or supplied configuration. For a local custom checkpoint,
pass its path with `--model`. In an explicit YAML, `envs.MODEL_PATH` can name the local
checkpoint while `model` retains its canonical identifier. The model must support
the replay's context length and protocol; custom resolution does not make an
arbitrary small-context model a valid canonical AgentX target.

Hyperloom changes serving arguments and environment through
`agentx.launch_overrides`, applies reviewed source/kernel patches, and compares
Magpie's public report. Missing, invalid, non-publishable, or diagnostic reports
cannot supply KEEP evidence. Baseline and candidate must have the same workload
identity. GEAK's search measurements are provisional until the candidate passes
the same canonical Magpie workload and Hyperloom's KEEP gate.

The profile phase enables `profiler.torch_profiler` on a separate copy of the YAML.
The generated YAML uses Magpie's capture defaults. For custom capture steps, count,
interval or annotation options, supply a YAML with those settings and leave
`enabled` false in the benchmark input. Magpie owns capture timing; Hyperloom selects a
primary rank and passes each capture window separately to TraceLens. Profiled
measurements never replace the unprofiled performance baseline.

The session archives its resolved workload as `benchmark.yaml` before writing the
manifest. `hyperloom optimize --resume-from SESSION_DIR` reuses that file without
requiring the switch or original YAML again. Changing the workload requires a new
session. Old AIPerf session history is not migrated; `HYPERLOOM_AGENTX=1` selects
Magpie for new sessions. Do not combine this entrypoint
with CLI ISL/OSL, concurrency sweeps, quantization or benchmark-script overrides.
The independent MLPerf agentic backend keeps its existing entrypoint.
