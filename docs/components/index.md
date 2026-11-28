---
myst:
    html_meta:
        "description": "Explore Hyperloom's specialized components: TraceLens, GEAK, and KernelForge for GPU kernel development and optimization."
        "keywords": "Hyperloom, components, TraceLens, GEAK, KernelForge, GPU profiling, benchmarking, kernel optimization, AMD GPU, ROCm"
---
# Hyperloom components

Hyperloom orchestrates several specialized tools. Each has its own
documentation page; the overarching optimization flow that ties them together
is described in [Hyperloom optimization loop](../conceptual/optimization-loop.md).
Benchmark rounds run in-tree, through InferenceSuite; see the
[InferenceSuite reference](../reference/inference-suite.md).

The following table lists each component with its role, source repository, and documentation link.

| Component | Role | Source | Documentation|
|-----------|------|--------|--------------|
| [TraceLens](tracelens.md) | Agentic trace analysis and roofline targets | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) | [TraceLens Docs](https://rocm.docs.amd.com/projects/tracelens/en/latest/) |
| [GEAK](geak.md) | GPU kernel generation and optimization (Triton, HIP, or FlyDSL) | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) | [GEAK Docs](https://rocm.docs.amd.com/projects/geak/en/latest/) |
| [KernelForge](kernelforge/what-is-kernelforge.md) | Autonomous GPU kernel development and optimization; ships inside Hyperloom as the built-in kernel-optimization agent | [AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom) (`src/kernelforge`) | [KernelForge docs](kernelforge/what-is-kernelforge.md) |
