---
myst:
    html_meta:
        "description": "Learn how Hyperloom autonomously optimizes LLM inference on AMD GPUs using an agentic search loop — profiling, benchmarking, and iteratively rewriting code."
        "keywords": "Hyperloom, LLM inference, AMD GPU, optimization, agentic, ROCm, GEMM tuning, kernel optimization, throughput, TraceLens, Magpie, GEAK, IntelliKit"
---

# What is Hyperloom?

Hyperloom is an agentic system that autonomously optimizes large language model (LLM) inference on AMD
GPUs. It treats optimization as a search problem: given a workload, it
explores candidate optimizations — backend swaps, server parameters, general matrix multiplication (GEMM)
tuning, kernel rewrites, parallelism configs — one change at a time, always
measuring against the real workload and using prior results plus knowledge base (KB) priors to
choose the next move.

Provide your workload, and the agent works toward an optimized configuration:
profiling against peak hardware potential, identifying bottlenecks, and
iteratively rewriting code to maximize throughput on AMD GPUs.

## The optimization loop

- **Workload understanding and profiling** — Submit your workload; the agent
   profiles it with TraceLens (trace collection using Magpie), capturing
   bottlenecks and roofline targets.
- **Code optimization loop** — The core of Hyperloom. The agent explores
   candidates one change at a time: **Think → Implement → Benchmark → Decide**.
   In parallel, hot kernels are optimized asynchronously using Kernel-Forge
   and GEAK.
- **Validated delivery** — Every change is correctness-gated before
   acceptance. When the loop exits, the runtime writes the final report,
   reproducible session artifacts, and `session_breakdown.json` for downstream
   delivery workflows.

## Components

Hyperloom is composed of multiple tools, each documented on its own page:

| Component | Role |
|-----------|------|
| [IntelliKit](components/intellikit.md) | Low-level GPU profiling primitives |
| [Magpie](components/magpie.md) | Benchmark engine with trace-collection support |
| [TraceLens](components/tracelens.md) | Agentic trace analysis and roofline targets |
| KernelForge | Deterministic forge backend |
| [GEAK](components/geak.md) | GPU kernel generation and optimization (Triton / HIP / FlyDSL) |
| [AgentKernelArena](components/agentkernelarena.md) | Optional standardized evaluation arena for agent benchmarking (not part of the default install or optimization loop) |

## Next steps

Use these resources to get started with Hyperloom:

- [Hosted UI quickstart](install/quickstart.md) — Launch through the hosted UI.
- [Bare-metal quickstart](install/setup.md) — Install directly on a ROCm host.
- [Run your first optimization](how-to/optimize.md) — Step-by-step usage.
- [API reference](reference/api-reference.rst) — Generated from in-code docstrings.
- [Release notes](release-notes.md) — Per-version changes.
