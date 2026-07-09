# Overview

Hyperloom is an agentic system that autonomously optimizes LLM inference on AMD
GPUs. It treats optimization as a **search problem**: given a workload, it
explores candidate optimizations — backend swaps, server parameters, GEMM
tuning, kernel rewrites, parallelism configs — one change at a time, always
measuring against the real workload and using prior results plus KB priors to
choose the next move.

Provide your workload and the agent delivers a fully optimized codebase:
profiling against peak hardware potential, identifying bottlenecks, and
iteratively rewriting code to maximize throughput on AMD GPUs.

## The optimization loop

![Hyperloom optimization loop: classify the model, measure a baseline, profile the GPU, and build the action stack, then run the scored DFS loop — pop the highest-scored action, execute and benchmark it, re-score and push new candidates — until scores converge, then sweep and report.](figs/optimization_loop.png)

*Hyperloom classifies the model, measures a baseline, profiles the GPU, and
builds a stack of candidate actions. It then runs a scored depth-first loop —
pop the highest-scored action, execute and benchmark it, then re-score and push
new candidates — until the remaining candidates no longer promise a gain, at
which point it runs a final sweep and writes the report.*

1. **Workload understanding & profiling** — submit your workload; the agent
   profiles it with TraceLens (trace collection via Magpie), capturing
   bottlenecks and roofline targets.
2. **Code optimization loop** — the core of Hyperloom. The agent explores
   candidates one change at a time: **Think → Implement → Benchmark → Decide**.
   In parallel, hot kernels are optimized asynchronously via Kernel-Forge,
   GEAK, and explicitly enabled OOB backends.
3. **Validated delivery** — every change is correctness-gated before
   acceptance. When the loop exits, the runtime writes the final report,
   reproducible session artifacts, and `session_breakdown.json` for downstream
   delivery workflows.

## Components

Hyperloom is composed of multiple tools, each documented on its own page.

| Component | Role |
|-----------|------|
| [IntelliKit](components/intellikit.md) | Low-level GPU profiling primitives |
| [Magpie](components/magpie.md) | Benchmark engine with trace-collection support |
| [TraceLens](components/tracelens.md) | Agentic trace analysis and roofline targets |
| [GEAK](components/geak.md) | GPU kernel generation and optimization (Triton / HIP / FlyDSL) |

## Next steps

- [Installation](installation.md) — set up Hyperloom locally or via the hosted UI.
- [How-to: run your first optimization](how_to_optimize.md) — step-by-step usage.
- [API reference](reference/api-reference.rst) — generated from in-code docstrings.
- [Release notes](release-notes.md) — per-version changes.
