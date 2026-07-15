---
myst:
    html_meta:
        "description": "Learn about TraceLens, Hyperloom's trace analysis library. Covers hierarchical GPU performance breakdowns, roofline modeling, multi-GPU analysis, and trace comparison."
        "keywords": "TraceLens, Hyperloom, GPU performance, trace analysis, roofline modeling, PyTorch profiler, JAX, rocprofv3, AMD GPU, ROCm, benchmarking, LLM inference, vLLM, SGLang"
---
# TraceLens

TraceLens is a Python library focused on automating analysis from trace
files. It turns raw PyTorch / JAX / rocprofv3 traces into hierarchical
performance breakdowns, roofline metrics, and multi-GPU communication analysis.

Within Hyperloom, TraceLens is the profiling brain of the workload
understanding stage: it consumes traces collected by [Magpie](magpie.md),
captures bottlenecks, and derives the roofline targets that seed the
optimization search tree.

- **Source**: <https://github.com/AMD-AGI/TraceLens>
- **License**: MIT

## Role in Hyperloom

The orchestration runtime enters TraceLens through two paths:

- The kernel request path:
  `src/hyperloom/orchestrator/kernel/request_handlers.py` dispatches
  `trace_analyze` requests as subprocesses that run
  `src/hyperloom/agents/kernel/tools/tracelens_analysis.py`. That script itself
  imports and calls `run_tracelens_skill` (the skill runner) internally under
  the agent route — the runner is not a separate subprocess dispatched by the
  orchestrator.
- The composite roofline action: `RooflineExecutor`
  (`src/hyperloom/orchestrator/actions/executors/roofline.py`) is an atomic
  `profile` + `trace_analyze` pipeline that first profiles the workload with
  Magpie, then calls `trace_analyze_handler()` directly on the trace.

Before profiling, Hyperloom can patch the active vLLM/SGLang server tree with
TraceLens-specific runtime flags through
`src/hyperloom/orchestrator/actions/executors/_server_patcher.py` and the
workload environment helpers. The generated report feeds the roofline ceilings
and bottleneck list used to score candidate optimizations. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md).

## TraceLens Documentation

For detailed documentation on TraceLens, please visit [ROCm Docs](https://rocm.docs.amd.com/projects/tracelens/en/latest/).
