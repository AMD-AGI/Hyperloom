# Components

Hyperloom orchestrates several specialized tools. Each has its own
documentation page; the overarching optimization flow that ties them together
is described in [How the optimization loop works](../HOW_THE_OPTIMIZATION_LOOP_WORKS.md).

| Component | Role | Source |
|-----------|------|--------|
| [IntelliKit](intellikit.md) | Low-level GPU profiling primitives | [AMDResearch/intellikit](https://github.com/AMDResearch/intellikit) |
| [Magpie](magpie.md) | Benchmark engine with trace-collection support | [AMD-AGI/Magpie](https://github.com/AMD-AGI/Magpie) |
| [TraceLens](tracelens.md) | Agentic trace analysis and roofline targets | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens) |
| [GEAK](geak.md) | GPU kernel generation and optimization (Triton / HIP / FlyDSL) | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) |

```{toctree}
:maxdepth: 1
:hidden:

intellikit
magpie
tracelens
geak
```
