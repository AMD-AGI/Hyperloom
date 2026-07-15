.. meta::
  :description: Learn how Hyperloom autonomously optimizes large language model inference on AMD GPUs — from installation and components to API reference and case studies.
  :keywords: Hyperloom, LLM inference, AMD GPU, optimization, agentic, ROCm, GEMM tuning, kernel optimization, throughput, TraceLens, Magpie, GEAK, IntelliKit, documentation

***********************
Hyperloom documentation
***********************

ROCm™ Hyperloom is an autonomous agentic system designed to optimize end-to-end inference workloads
(targeting both host code and GPU kernels) on AMD GPUs. Using advanced AI agents and profiling tools,
Hyperloom analyzes your workload, identifies performance bottlenecks, implements targeted optimizations,
and validates the performance and correctness of the optimizations without requiring manual intervention.
 
The system operates through a sophisticated multi-stage pipeline. First, an agent profiles your workload,
leveraging tools like IntelliKit for low-level GPU profiling, Magpie for trace collection, and TraceLens
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

![Hyperloom optimization loop](../images/Hyperloom_architecture.png)

- **Workload understanding and profiling** — Submit your inference workload; the agent profiles it with
   TraceLens (trace collection using Magpie), capturing bottlenecks and roofline targets.
- **Optimization loop** — The agent explores candidates one change at a time: **Think → Decide →
   Implement → Benchmark**. In parallel, hot kernels are optimized asynchronously using Kernel-Forge and
   GEAK.
- **Validated delivery** — Every change is correctness-gated before acceptance. When the loop exits, the
   runtime writes the final report, reproducible session artifacts, and `session_breakdown.json` for
   downstream delivery workflows.

The Hyperloom source code is hosted on GitHub at `https://github.com/AMD-AGI/Hyperloom <https://github.com/AMD-AGI/Hyperloom>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Docker quickstart </install/local-mode>`
    * `Setup and examples <../examples/README.md>`_
    * :doc:`Slurm quickstart </install/slurm>`

  .. grid-item-card:: Components

    * :doc:`Components </components/index>`
    * :doc:`Hyperloom optimization loop </conceptual/optimization-loop>`

  .. grid-item-card:: How to

    * :doc:`Run a Hyperloom optimization </how-to/optimize>`
    * :doc:`Quantization with AMD Quark </how-to/quantization-quark>`

  .. grid-item-card:: Conceptual

    * :doc:`Hyperloom optimization loop <conceptual/optimization-loop>`

  .. grid-item-card:: Reference

    * :doc:`API reference </reference/api-reference>`
    * :doc:`Environment variables </reference/environment-variables>`
    * :doc:`Authentication and credentials </reference/authentication>`
    * :doc:`Troubleshooting </reference/troubleshooting>`

To contribute to the documentation, see `Contributing to Hyperloom <https://github.com/AMD-AGI/Hyperloom/blob/main/CONTRIBUTING.md>`_.

Hyperloom is released under the MIT license. For details, see the :doc:`License <license>` page.