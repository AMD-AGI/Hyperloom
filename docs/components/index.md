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

| Component | Role | Source |
|-----------|------|--------|
| [IntelliKit](intellikit.md) | Low-level GPU profiling primitives | [AMDResearch/intellikit](https://github.com/AMDResearch/intellikit) |
| [Magpie](magpie.md) | Benchmark engine with trace-collection support | [AMD-AGI/Magpie](https://github.com/AMD-AGI/Magpie) |
| [TraceLens](tracelens.md) | Agentic trace analysis and roofline targets | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) |
| KernelForge | Deterministic forge backend and OOB source checkout carrier | [AMD-AGI/KernelForge](https://github.com/AMD-AGI/KernelForge) |
| [GEAK](geak.md) | GPU kernel generation and optimization (Triton / HIP / FlyDSL) | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) |
| [AgentKernelArena](agentkernelarena.md) | Optional standardized evaluation arena for agent benchmarking | [AMD-AGI/AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) |
