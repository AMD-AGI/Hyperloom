---
myst:
  html_meta:
    "description": "Learn what Hyperloom is: an autonomous agentic system that optimizes LLM inference workloads on AMD GPUs through profiling, kernel optimization, and iterative benchmarking."
    "keywords": "Hyperloom, what is Hyperloom, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, IntelliKit, Magpie, kernel optimization, optimization loop"
---

# What is Hyperloom?

The system operates through a sophisticated multi-stage pipeline. First, an agent profiles your workload,
using tools like IntelliKit for low-level GPU profiling, Magpie for trace collection, and TraceLens
for trace analysis to identify top bottlenecked kernels and create a bridge plan.

Next, Hyperloom employs a self-evolving code optimization engine following an iterative agentic loop (Think
→ Decide → Implement → Benchmark), alongside a Dynamic Specialist Agent and Knowledge Base to intelligently
search the optimization space. GEAK, a multi-agent GPU performance optimizer, optimizes hot kernels in
parallel. Once optimizations are identified and validated, Hyperloom prepares the optimized code and
generates a report with all proposed changes and expected performance improvements. This end-to-end
automation enables developers to achieve significant performance improvements while maintaining code
quality and reducing the manual effort traditionally required for GPU optimization.

Provide your workload, and the agent works toward an optimized configuration: profiling against peak
hardware potential, identifying bottlenecks, and iteratively rewriting code to maximize throughput on
AMD GPUs.

## The optimization loop

The following diagram and steps describe how Hyperloom processes a workload from submission to validated delivery.

```{image} ../images/Hyperloom_architecture.png
:alt: Hyperloom architecture diagram showing the multi-stage optimization pipeline from workload profiling through kernel optimization to validated delivery
```

- **Workload understanding and profiling** — Submit your inference workload; the agent profiles it with
   TraceLens (trace collection using Magpie), capturing bottlenecks and roofline targets.
- **Optimization loop** — The agent explores candidates one change at a time: **Think → Decide →
   Implement → Benchmark**. In parallel, hot kernels are optimized asynchronously using GEAK.
- **Validated delivery** — Every change is correctness-gated before acceptance. When the loop exits, the
   runtime writes the final report, reproducible session artifacts, and `session_breakdown.json` for
   downstream delivery workflows.

