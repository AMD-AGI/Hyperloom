---
myst:
    html_meta:
        "description": "Learn about TraceLens, Hyperloom's trace analysis library. Covers hierarchical GPU performance breakdowns, roofline modeling, multi-GPU analysis, and trace comparison."
        "keywords": "TraceLens, Hyperloom, GPU performance, trace analysis, roofline modeling, PyTorch profiler, JAX, rocprofv3, AMD GPU, ROCm, benchmarking, LLM inference, vLLM, SGLang"
---
# TraceLens

TraceLens is a Python library focused on automating analysis from trace
files. It turns raw PyTorch, JAX, or rocprofv3 traces into hierarchical
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
  `trace_analyze` requests to
  `src/hyperloom/orchestrator/actions/executors/trace_analyze.py`, which runs
  `python -m hyperloom.orchestrator.trace_analysis.tracelens_analysis` as a subprocess.
  That module itself imports and calls `run_tracelens_skill` (the skill runner)
  internally under the agent route — the runner is not a separate subprocess
  dispatched by the orchestrator.
- The composite roofline action: `RooflineExecutor`
  (`src/hyperloom/orchestrator/actions/executors/roofline.py`) is an atomic
  `profile` + `trace_analyze` pipeline that first profiles the workload with
  Magpie, then calls `trace_analyze_handler()` directly on the trace.

Before profiling, Hyperloom lays TraceLens's runtime patch for vLLM/SGLang in a
copy of the installed package on the host's local disk and puts that copy first
on the profiled server's `PYTHONPATH`, through
`src/hyperloom/orchestrator/actions/executors/_server_patcher.py` and the
workload environment helpers; the installed tree, which every other test and
session imports, is never written. A copy is reused only when the same user laid
it and no one else can write it; with vLLM's image source checkout
(`HYPERLOOM_VLLM_IMAGE_SOURCE=1`) the copy mirrors that checkout. A profile test
whose rebuilt framework the patch cannot be laid in runs without it, and its
notes and points say so (`profile_patch_not_laid`,
`HYPERLOOM_TRACELENS_PATCH_STATUS=unavailable`). The generated report feeds the roofline ceilings
and bottleneck list used to score candidate optimizations. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md).

## TraceLens documentation

For detailed documentation on TraceLens, see [TraceLens on ROCm Docs](https://rocm.docs.amd.com/projects/tracelens/en/latest/).
