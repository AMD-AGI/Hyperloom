---
myst:
    html_meta:
        "description": "Learn about GEAK, Hyperloom's agent-driven GPU kernel optimization framework. Covers Triton, HIP, and FlyDSL kernel rewriting, parallel optimization, and patch validation."
        "keywords": "GEAK, Hyperloom, GPU kernel optimization, Triton, HIP, FlyDSL, AMD GPU, ROCm, kernel rewriting, benchmarking, parallel optimization, LLM inference, agent, Ray"
---
# GEAK

GEAK (Generating Efficient AI-Centric Kernels) is a multi-agent framework for end-to-end GPU kernel
optimization in real codebases. It runs a closed loop of profiling, optimization, and
validation, and produces reviewable patches backed by reproducible benchmarks.
GEAK supports Triton, HIP (and CUDA / Composable Kernel (CK) / HSA Code Object (HSACO)), and FlyDSL
kernels. It is driven by [Claude Code](https://www.anthropic.com/claude-code) and ships two
deterministic JS Workflows — `e2e_workflow` for whole-model serving throughput and `kernel_workflow`
for single kernels — with a deterministic control plane (budget loop, parallel fan-out, verification,
stop conditions) that invokes LLM agents only for judgment.

Within Hyperloom, GEAK is the whole-pipeline end-to-end optimization delegate: when a workload is
handed off, the orchestrator invokes GEAK once at the kernel-agent phase through the stable
`interface/run_e2e.py` contract (a `handoff.json` in, a `result.json` back). Parallel exploration of
candidate kernels then happens inside GEAK's Workflows on the on-box GPUs.

## Role in Hyperloom

Hyperloom uses GEAK as the **whole-pipeline e2e delegate** when
`KERNEL_OPT_BACKEND_ORDER=geak` (the bare-metal default). In this mode the
orchestrator hands the optimization workload to
`src/hyperloom/agents/kernel/tools/backends/geak_runner.py`, which resolves the
GEAK checkout and launches GEAK's e2e runner (`interface/run_e2e.py`) with the
generated session context.

When GEAK owns the phase it runs the whole optimization loop itself — both the
end-to-end serving optimization *and* the per-kernel work underneath it, since
GEAK's `e2e_workflow` recursively drives `kernel_workflow` to author and tune the
individual hot kernels worth fixing. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md).

## GEAK documentation

For detailed documentation on GEAK, see [GEAK on ROCm Docs](https://rocm.docs.amd.com/projects/geak/en/latest/).
