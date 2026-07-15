---
myst:
    html_meta:
        "description": "Explore Hyperloom's specialized components: IntelliKit for GPU profiling, Magpie for benchmarking, TraceLens for trace analysis, GEAK for kernel optimization, and AgentKernelArena for agent evaluation."
        "keywords": "Hyperloom, components, IntelliKit, Magpie, TraceLens, GEAK, AgentKernelArena, GPU profiling, benchmarking, kernel optimization, AMD GPU, ROCm, evaluation"
---
# Hyperloom components

Hyperloom orchestrates several specialized tools. Each has its own
documentation page; the overarching optimization flow that ties them together
is described in [Hyperloom optimization loop](../conceptual/optimization-loop.md).

The following table lists each component with its role, source repository, and documentation link.

| Component | Role | Source | Documentation|
|-----------|------|--------|--------------|
| [IntelliKit](intellikit.md) | Low-level GPU profiling primitives | [AMDResearch/intellikit](https://github.com/AMDResearch/intellikit) | [IntelliKit Docs](https://rocm.docs.amd.com/projects/intellikit/en/latest/) |
| [Magpie](magpie.md) | Benchmark engine with trace-collection support | [AMD-AGI/Magpie](https://github.com/AMD-AGI/Magpie) | [Magpie Docs](https://rocm.docs.amd.com/projects/magpie/en/latest/) |
| [TraceLens](tracelens.md) | Agentic trace analysis and roofline targets | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) | [TraceLens Docs](https://rocm.docs.amd.com/projects/tracelens/en/latest/) |
| KernelForge | Deterministic forge backend | [AMD-AGI/KernelForge](https://github.com/AMD-AGI/KernelForge) |  |
| [GEAK](geak.md) | GPU kernel generation and optimization (Triton / HIP / FlyDSL) | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) | [GEAK Docs](https://rocm.docs.amd.com/projects/geak/en/latest/) |
| [AgentKernelArena](agentkernelarena.md) | Optional standardized evaluation arena for agent benchmarking | [AMD-AGI/AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) | [AgentKernelArena Docs](https://rocm.docs.amd.com/projects/agent-kernel-arena/en/latest/) |
