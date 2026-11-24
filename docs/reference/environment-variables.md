---
myst:
    html_meta:
        "description": "Reference for the environment variables read by the Hyperloom runtime, grouped by purpose: credentials, paths, workload parameters, backend selection, and observability. Opt-in benchmark modes carry additional variables documented alongside them."
        "keywords": "Hyperloom, environment variables, configuration, OPENAI_API_KEY, USER_DATA_PATH, ROCm, AMD GPU, LLM inference, kernel optimization, LLM gateway, Langfuse, session"
---
# Environment variables

User-configurable environment variables for Hyperloom, grouped by purpose.
Runtime parameters such as framework, tensor parallelism, prompt lengths, and
phase toggles are configured with CLI flags; internal subprocess handoff envs
are intentionally not listed as user configuration.

Variables marked Required must be set (using shell or `$REPO_ROOT/.env`)
or the CLI will exit fast at startup. Variables marked Optional have
sensible defaults; the default is shown in the Default column.

Precedence rule (applies everywhere): shell-exported env wins over `.env`.
See [Hyperloom authentication and credentials](authentication.md).

Each group of variables has its own reference page:

| Page | Description |
|------|-------------|
| [Credentials](environment-variables/env-vars-credentials.md) | API keys and gateway tokens (required) |
| [Paths](environment-variables/env-vars-paths.md) | User data, runtime, ledger, and package destination paths |
| [Workload configuration](environment-variables/env-vars-workload.md) | Model, framework, GPU topology, and benchmark mode flags |
| [Accuracy gates](environment-variables/env-vars-accuracy-gates.md) | Pass/fail tolerances and eval generation bounds |
| [Experience KB](environment-variables/env-vars-experience-kb.md) | Knowledge-base storage, retrieval, and publishing settings |
| [Multi-branch RSI archive](environment-variables/env-vars-rsi-archive.md) | RSI snapshot archive location and retention |
| [Multi-branch prompt-cache TTL](environment-variables/env-vars-prompt-cache-ttl.md) | Per-branch prompt-cache time-to-live settings |
| [PrimaTune predictor](environment-variables/env-vars-primatune.md) | PrimaTune cost-model and predictor configuration |
| [Kernel-opt backend selection](environment-variables/env-vars-kernel-backend.md) | GEAK, Forge, and kernel agent backend switches |
| [Fusion lane](environment-variables/env-vars-fusion-lane.md) | Fusion kernel candidate lane controls |
| [KERNEL lane behaviour](environment-variables/env-vars-kernel-lane.md) | Tick cadence, idle policy, and lane scheduling |
| [Collective candidate extraction](environment-variables/env-vars-collective-extraction.md) | Collective-op kernel extraction settings |
| [Kernel source resolution](environment-variables/env-vars-kernel-source-resolution.md) | Source-tree discovery and patch application |
| [Single-node Ray execution](environment-variables/env-vars-ray-execution.md) | Ray cluster address, resources, and runtime settings |
| [Codex (OpenAI) agent sandbox](environment-variables/env-vars-codex-sandbox.md) | OpenAI Codex agent execution environment |
| [Single-node Ray GPU scheduling](environment-variables/env-vars-ray-gpu-scheduling.md) | Per-task GPU allocation for Ray workers |
| [Compute partitioning](environment-variables/env-vars-compute-partitioning.md) | Partition mode, hand-off variables, and sweep script |
| [Multi-node / prefill-decode](environment-variables/env-vars-multinode-pd.md) | Multi-node topology and cluster hand-off variables |
| [Quantization prelude](environment-variables/env-vars-quantization.md) | Quantization format and calibration settings |
| [Enablement admission](environment-variables/env-vars-enablement.md) | Phase and feature enablement gates |
| [Benchmark deadlines](environment-variables/env-vars-benchmark-deadlines.md) | Timeout and deadline controls for benchmark rounds |
| [EXPLORE pre-filters](environment-variables/env-vars-explore-prefilters.md) | Measurement store, screen, projection, and InferaSim anchors |
| [Reactor budgets](environment-variables/env-vars-reactor-budgets.md) | Token and time budgets for reactor phases |
| [TraceLens analysis route](environment-variables/env-vars-tracelens-route.md) | TraceLens profile routing and backend selection |
| [TraceLens analysis budgets](environment-variables/env-vars-tracelens-budgets.md) | Token and time budgets for TraceLens analysis |
| [Framework / source-tree discovery](environment-variables/env-vars-framework-discovery.md) | Framework detection and source-tree search settings |
| [Targeted builds (Rung 5)](environment-variables/env-vars-targeted-builds.md) | Rung 5 targeted-build controls and overrides |
| [Security compatibility switches](environment-variables/env-vars-security.md) | Agent isolation, bubblewrap, and security policy switches |
| [Critic / knowledge base](environment-variables/env-vars-critic-kb.md) | Critic agent and knowledge-base configuration |
| [Session / observability hand-off](environment-variables/env-vars-session-handoff.md) | Langfuse tracing, gateway attribution, and session IDs |
| [AgentX performance metric](environment-variables/env-vars-agentx-performance.md) | 2-D interactivity grading and MLPerf agentic backend |
| [Phase tuning](environment-variables/env-vars-phase-tuning.md) | Macro-cycle duration, idle thresholds, and variant timeouts |
| [Plug selection](environment-variables/env-vars-plug-selection.md) | Manifest path and overrides for plug socket implementations |
| [Hosted run inputs](environment-variables/env-vars-hosted-inputs.md) | Campaign request file, run ID, state directory, and sandbox variables |
| [Measurement card guard](environment-variables/env-vars-measurement-guard.md) | GPU card hold policy during measurement |
| [Merge smoke](environment-variables/env-vars-merge-smoke.md) | Pre-sandbox Python import smoke test for patched files |
| [Variables intentionally not exposed](environment-variables/env-vars-not-exposed.md) | Internal-only variables that should not be set by hand |

## More info

Use these resources for related configuration and reference information:

* [Hyperloom authentication and credentials](authentication.md): Credential precedence and direct upstream gateway wiring.
* [Troubleshooting Hyperloom](troubleshooting.md): Symptom → variable reverse-lookup for common failures.
