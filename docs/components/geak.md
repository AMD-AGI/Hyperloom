---
myst:
    html_meta:
        "description": "Learn about GEAK, Hyperloom's agent-driven GPU kernel optimization framework. Covers Triton, HIP, and FlyDSL kernel rewriting, parallel optimization, and patch validation."
        "keywords": "GEAK, Hyperloom, GPU kernel optimization, Triton, HIP, FlyDSL, AMD GPU, ROCm, kernel rewriting, benchmarking, parallel optimization, LLM inference, agent, Ray"
---
# GEAK

GEAK (Generating Efficient AI-Centric Kernels) is an agent-driven framework for end-to-end GPU kernel optimization in
real codebases. It runs a closed loop of profiling, optimization, and
validation, and produces reviewable patches backed by reproducible benchmarks.
GEAK supports Triton, HIP (and CUDA / Composable Kernel (CK) / HSA Code Object (HSACO)), and FlyDSL
kernels, and extends [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent)
for its agent loop and environment tooling.

Within Hyperloom, GEAK is one of the kernel-rewrite backends: when a hot kernel
is identified, it is optimized asynchronously through GEAK. The kernel agent
dispatches GEAK runs with placement precedence SSH (Dynamo multi-node) > Ray
(when available) > direct CLI, so multiple candidates can be explored in
parallel on the cluster's GPUs.

## Role in Hyperloom

Hyperloom uses GEAK as the **whole-pipeline e2e delegate** when
`KERNEL_OPT_BACKEND_ORDER=geak` (the bare-metal default). In this mode the
orchestrator hands the optimization workload to
`src/hyperloom/agents/kernel/tools/backends/geak_runner.py`, which resolves the
GEAK checkout and launches GEAK's e2e runner (`interface/run_e2e.py`) with the
generated session context.

GEAK is distinct from the per-kernel `forge` backend:

- `geak` runs the whole e2e optimization loop through GEAK.
- `forge` targets individual kernel/GEMM opportunities through KernelForge and
  related forge tools.

This keeps the backend split explicit: use `geak` for whole-pipeline delegation
and `forge` for per-kernel optimization. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md).

## GEAK Documentation

For detailed documentation on GEAK, please visit [ROCm Docs](https://rocm.docs.amd.com/projects/geak/en/latest/).
