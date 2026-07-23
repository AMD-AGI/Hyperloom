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

| Component | Role | Source |
|-----------|------|--------|
| [IntelliKit](https://rocm.docs.amd.com/projects/intellikit/en/latest/) | Low-level GPU profiling primitives | [AMDResearch/intellikit](https://github.com/AMDResearch/intellikit) | 
| [Magpie](https://rocm.docs.amd.com/projects/magpie/en/latest/) | Benchmark engine with trace-collection support | [AMD-AGI/Magpie](https://github.com/AMD-AGI/Magpie) |
| [TraceLens](https://rocm.docs.amd.com/projects/tracelens/en/latest/) | Agentic trace analysis and roofline targets | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) | 
| [GEAK](https://rocm.docs.amd.com/projects/geak/en/latest/) | GPU kernel generation and optimization (Triton / HIP / FlyDSL) | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) | 
| [AgentKernelArena](https://rocm.docs.amd.com/projects/agent-kernel-arena/en/latest/) | Optional standardized evaluation arena for agent benchmarking | [AMD-AGI/AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) | 
