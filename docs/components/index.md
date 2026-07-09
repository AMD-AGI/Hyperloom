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
| [IntelliKit](https://advanced-micro-devices-demo--131.com.readthedocs.build/projects/intellikit/en/131/index.html) | Low-level GPU profiling primitives | [AMDResearch/intellikit](https://github.com/AMDResearch/intellikit) |
| [Magpie](https://advanced-micro-devices-demo--50.com.readthedocs.build/projects/magpie/en/50/) | Benchmark engine with trace-collection support | [AMD-AGI/Magpie](https://github.com/AMD-AGI/Magpie) |
| [TraceLens](https://advanced-micro-devices-tracelens--772.com.readthedocs.build/en/772/) | Agentic trace analysis and roofline targets | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) |
| KernelForge | Deterministic forge backend and OOB source checkout carrier | [AMD-AGI/KernelForge](https://github.com/AMD-AGI/KernelForge) |
| [GEAK](https://advanced-micro-devices-demo--311.com.readthedocs.build/projects/geak/en/311/) | GPU kernel generation and optimization (Triton / HIP / FlyDSL) | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) |
| [AgentKernelArena](https://advanced-micro-devices-agentkernelarena--55.com.readthedocs.build/en/55) | Optional standardized evaluation arena for agent benchmarking | [AMD-AGI/AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) |
