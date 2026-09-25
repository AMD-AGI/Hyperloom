---
myst:
    html_meta:
        "description": "Step-by-step guide to running a Hyperloom optimization. Covers launching from Claude Code, monitoring, resuming, and reading output artifacts."
        "keywords": "Hyperloom, optimization, how-to, LLM inference, AMD GPU, ROCm, Claude Code, GEAK, TraceLens, session, throughput"
---
# Run a Hyperloom optimization

This topic assumes you have already completed installation. If you haven't, follow the [Hyperloom installation instructions](../install/install.md) then return here to launch your first run.

## Launch from Claude Code

Open the Hyperloom workspace in Claude Code, then paste the following prompt into
the Claude Code Chat, filling in your workload details:

```{note}
The prompt includes `install.sh`. This is intentional: Claude Code runs in its own
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
| `CONC` | Concurrent requests — baseline benchmark concurrency (`--conc`, default `64`) | Set to your target concurrency. Synthetic workloads can measure a SWEEP ladder around it. Native AgentX measures this one recipe point: its concurrency sweep defaults off, and `--enable-conc-sweep` is rejected. |
| `ISL` | Input sequence length — tokens in each request's prompt | Match your production workload; `1024` is a common starting point |
| `OSL` | Output sequence length — tokens generated per response | Match your production workload; `1024` is a common starting point |

```{note}
`ISL` / `OSL` describe the synthetic request shape. In a source config with
`benchmark.agentx: enable`, request lengths come from the recorded trace corpus
instead, so these two values do not affect what is measured — the server's
context window is sized from the model's own configuration rather than from
`ISL+OSL`.
```

See [`src/hyperloom/inference_optimizer/SKILL.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)
for the full prompt field reference (every field maps to a CLI flag defined in
`cli/parser.py`).

## Run an InferenceX AgentX workload

AgentX measurement uses Magpie's native InferenceX integration. Pass a source
Magpie YAML with `--benchmark-config`; its `benchmark.agentx: enable` switch
automatically selects Hyperloom's AgentX session and grading mode. Do not export
`HYPERLOOM_AGENTX` for this path. That environment variable remains only as a
legacy mode switch for callers that do not supply a source YAML. This
integration is pinned to Magpie
[0.3.0 release candidate](https://github.com/AMD-AGI/Magpie/pull/105) commit
`3642ce66ae46ca4dc125340b3d14a3f4640c369b` and InferenceX commit
`3d5581562f643f9bdeb8410cd924e2c70906c966`.

For the pinned GLM-5.2 TP4 recipe, create a source YAML. It carries the public
model identity, framework, launcher, image pin, and fixed concurrency; recipe
internals remain in InferenceX:

```yaml
benchmark:
  framework: sglang
  model: amd/GLM-5.2-MXFP4
  precision: fp4
  runner_type: mi355x
  run_mode: local
  agentx: enable
  benchmark_script: single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh
  docker_image: lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728
  gpu_selection:
    auto: false
  envs:
    CONC: 8
    ROCR_VISIBLE_DEVICES: 0,1,2,3
```

Then run Hyperloom inside the exact image named by the YAML:

```bash
python -m hyperloom.inference_optimizer.cli optimize \
  --benchmark-config ./agentx-glm52.yaml \
  --max-hours 3
```

The explicit three-hour budget is for one canonical baseline. The normal
two-hour default is commonly shorter than model load, warmup, drain, and the
3600-second measurement together. Size it upward if more rounds are intended.

`--benchmark-config` is for a fresh launch. A resume rejects it. Once a baseline
has been accepted, resume restores its materialized config and runtime pins;
before that point it restores the session's snapshotted source YAML and pins.
The installer defaults must still resolve to the tested Magpie and InferenceX
revisions above; changing either pin is a coordinated compatibility change.

The source and resolved fields have different owners:

| Owner | Fields |
|---|---|
| Source YAML / operator | canonical `model`, `framework`, `precision`, `runner_type`, `run_mode: local`, `benchmark_script`, effective `docker_image`, fixed `envs.CONC`, and optional `agentx.recipe` / `agentx.selector` controls |
| InferenceX recipe | logical TP/PP/PCP/EP, `MODEL_PREFIX`, KV-offload settings, CPU DRAM allocation, corpus, duration, and AIPerf protocol |
| Hyperloom | physical GPU reservation `TP×PP×PCP`, the zero-based ROCR mask, `gpu_selection.auto=false`, session pins, and result validation |

`benchmark.model` is the exact model id in InferenceX's `amd-master.yaml`;
the optional CLI `--model` is the local mounted checkpoint. Magpie resolves the
recipe's TP/PP/PCP/EP, model prefix, KV-offload settings, CPU DRAM allocation,
and AIPerf protocol. Before acquiring GPUs, Hyperloom resolves the recipe with
the same benchmark interpreter that will execute Magpie. It runs the result in
the already selected serving container (`run_mode: local`) and does not start a
nested Docker container. `benchmark.docker_image` overrides and pins the
effective recipe image, and Magpie includes that value in the recipe
fingerprint. An existing `HYPERLOOM_IMAGE` is an optional consistency assertion
and must match it exactly. Neither value starts a container or proves the image
of the process already running.

The example omits `benchmark.inferencex_path`. Preflight reuses a writable
checkout at the tested commit or clones one into the dependency cache. An
optional source path only nominates a preferred checkout: a missing or
wrong-revision path falls back to the pinned clone, while an explicit checkout
at the right revision that is not writable fails preflight. The pinned Magpie
local runner interpolates this path into an unquoted `bash -c` command, so the
resolved absolute checkout path must contain only shell-safe token characters;
whitespace or shell metacharacters fail closed before launch.

If the checkpoint should come directly from Hugging Face, omit CLI `--model`;
Hyperloom uses `benchmark.model` from the source YAML and removes
`MODEL_PATH` from the native InferenceX subprocess so it uses its normal
Hugging Face cache. For a local checkpoint, pass an absolute path or an
explicit relative path such as `./models/GLM-5.2`.

For native AgentX, omitted `--tp` and `--ep` are filled from the selected
recipe. An explicit `--tp` is an exact outer resource assertion and must equal
the resolved `TP×PP×PCP` physical GPU count; an explicit `--ep` must equal the
resolved EP. The example therefore omits both and resolves to TP4/EP4 on
`4×1×1` physical GPUs. Topology-changing AgentX sweeps are rejected. The pinned
launchers overwrite logical `HIP_VISIBLE_DEVICES` with physical ROCR values,
so Hyperloom derives `gpu_selection.auto=false` and the zero-based mask
`ROCR_VISIBLE_DEVICES=0,...,N-1` when omitted. Explicit source values are
assertions; a nonzero or reordered mask is rejected.

The effective concurrency comes from CLI `--conc` or `benchmark.envs.CONC` and
must exist in the selected recipe. Native AgentX concurrency sweeps default
off; explicitly passing `--enable-conc-sweep` fails preflight because the
pinned launcher has no distinct optimized arm to compare. `CONC=8` uniquely
selects the example's TP4/EP4 DRAM+HiCache arm; `envs.TP` is not a selector. If
a concurrency belongs to multiple arms, use the YAML-native object form:

```yaml
agentx:
  enabled: true
  selector:
    tp: 4
    kv_offloading: dram
    kv_offload_backend: hicache
```

Add `agentx.recipe` only when multiple recipe names match. The legacy
`AGENTX_RECIPE` and JSON `AGENTX_SELECTOR` environment variables provide the
same escape hatches for callers without a source object. Magpie fails closed
rather than guessing. Hyperloom writes the fixed measurement concurrency to
`benchmark.envs.CONC` and removes a stale `benchmark.agentx.concurrency` from
the input YAML. The resolved recipe owns
the corpus; native `AGENTX_DATASET` and `WEKA_LOADER_OVERRIDE` overrides are
rejected. Canonical mode runs for 3600 seconds and configures a 393-trace
dataset-entry cap. That value is a loader ceiling, not a guarantee that 393
traces, sessions, or requests survive availability and context-length filters.
For a 1200-second diagnostic directly in Magpie, set
`benchmark.agentx.mode: fast` in an enabled AgentX configuration, or pass
`--agentx --agentx-mode fast` to its `benchmark` command. Fast results are
non-publishable, so a full Hyperloom `optimize` rejects them as its baseline,
including when selected through Hyperloom's `AGENTX_MODE=fast` override.

This Hyperloom integration supports Magpie AgentX v1 in local, single-node
SGLang or vLLM mode. It bypasses Hyperloom's outer Ray actor automatically;
leave `INFERENCE_OPTIMIZER_RAY_EXEC` unset or set it to `0`, because explicitly
setting it to `1` is rejected. Multi-node/disaggregated execution,
`server_lifecycle`, and Atom are also unsupported.

The pinned InferenceX launchers own their complete server argv and expose no
optimizer-argument hook. Hyperloom does not modify a launcher. Server-argument,
server-environment, removal, and reference-launch candidate overrides therefore
fail closed instead of measuring an unchanged server. This release provides
native AgentX measurement only; server configuration optimization and a
baseline-versus-optimized concurrency sweep require an upstream launcher hook.

Native Magpie AgentX v1 does not collect PyTorch traces. Measurement rounds use
the native launcher above, including its radix/prefix-cache behavior. The
launcher also forces the model's native context, so `ISL`, `OSL`, and
`MAX_MODEL_LEN` do not reshape native replay. If PRELUDE schedules roofline or
profile analysis, Hyperloom uses `aiperf_client.sh` with a generic server. That
compatibility trace is diagnostic only; it is not recipe-identical native
AgentX. The diagnostic path preserves the installed framework source and skips
TraceLens/CK source patches, so annotation coverage can be lower without
changing subsequent native measurements. AIPerf's
`profile`/`profiled` fields describe workload statistics, not a PyTorch profiler
trace. The native `KERNEL_AGENT`/GEAK phase is not dispatched: it records
`status=skipped` and `error_class=unsupported_upstream_launcher_hook`.

Accepted native results require `benchmark_valid=true`, `publishable=true`, an
`agentic-coding` scenario, matching strict recipe/launch/raw fingerprints, and
a trusted fingerprint-bound GPU topology. `publishable` attests Magpie's
canonical protocol; Hyperloom separately binds the selected recipe to the exact
audited launcher bytes and pinned checkout. It still cannot cryptographically
prove the actual outer image. Its execution identity covers the resolved
`BenchmarkConfig` plus the effective, scrubbed launcher environment for an
audited set of server/framework/runtime controls; credentials, cache routing,
output paths, and unrelated login-shell variables remain outside that hash.

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

Paste this prompt into the Claude Code chat to resume an existing session:

```text
@hyperloom/inference_optimizer/SKILL.md

Resume the existing Hyperloom optimization session.

Requirements:
1. Launch `python -m hyperloom.inference_optimizer.cli optimize --resume-from "$SESSION_DIR"`; do not start a new session.
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
| `final.throughput_tok_s_per_gpu` | Validated end-of-session serving throughput — the headline number for SGLang / vLLM |
| `final.cumulative_gain_pct_validated` | Validated gain over baseline |
| `final.action_path` | Ordered list of changes that make up the final optimized stack |

For the full schema — useful if you are building a dashboard, reporting
pipeline, or downstream integration on top of this file — see
[`session_breakdown.json` integration in Hyperloom](../reference/session-breakdown.md).

## Troubleshooting

If a run fails on first launch, see [Troubleshooting Hyperloom](../reference/troubleshooting.md).
