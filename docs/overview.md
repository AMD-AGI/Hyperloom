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

This tree-search orchestration is published as **Arbor**
([arXiv:2606.12563](https://arxiv.org/abs/2606.12563)) — the research name for
Hyperloom's Orchestrator / Specialist / Critic loop.

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
- [API reference](api/index) — generated from in-code docstrings.
- [Release notes](release_notes.md) — per-version changes.
