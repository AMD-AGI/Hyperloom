.. meta::
  :description: Learn how Hyperloom autonomously optimizes large language model inference on AMD GPUs — from installation and components to API reference and case studies.
  :keywords: Hyperloom, LLM inference, AMD GPU, optimization, agentic, ROCm, GEMM tuning, kernel optimization, throughput, TraceLens, Magpie, GEAK, IntelliKit, documentation

***********************
Hyperloom documentation
***********************

ROCm Hyperloom is an autonomous agentic system designed to optimize end-to-end inference workloads
on AMD GPUs. It uses advanced AI agents and profiling tools to analyze a workload, identify performance
bottlenecks, implement targeted optimizations, and validate the performance and correctness of the
optimizations -- all without requiring manual intervention.
 
The Hyperloom source code is hosted on GitHub at `https://github.com/AMD-AGI/Hyperloom <https://github.com/AMD-AGI/Hyperloom>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install on Docker or bare metal </install/install>`
    * :doc:`Install on a Slurm cluster </install/slurm>`

  .. grid-item-card:: Components

    * :doc:`Components </components/index>`
    * :doc:`Hyperloom optimization loop </conceptual/optimization-loop>`

  .. grid-item-card:: How to

    * :doc:`Run a Hyperloom optimization </how-to/optimize>`
    * :doc:`Optimize your own workload </how-to/optimize-custom-workload>`
    * :doc:`Quantization with AMD Quark </how-to/quantization-quark>`

  .. grid-item-card:: Reference

    * :doc:`API reference </reference/api-reference>`
    * :doc:`Environment variables </reference/environment-variables>`
    * :doc:`Authentication and credentials </reference/authentication>`
    * :doc:`Kernel optimization execution path </reference/kernel-execution-path>`
    * :doc:`Operations and self-hosting </reference/operations>`
    * :doc:`Upgrade guide </reference/upgrade>`
    * :doc:`Session output schema </reference/session-breakdown>`
    * :doc:`Knowledge base integration </reference/integrate-kb>`
    * :doc:`Operator scripts </reference/operator-scripts>`
    * :doc:`Multi-node inference optimization </reference/multi-node>`
    * :doc:`Predictor HTTP contract </reference/primatune-predictor>`
    * :doc:`Troubleshooting </reference/troubleshooting>`

To contribute to the documentation, see `Contributing to Hyperloom <https://github.com/AMD-AGI/Hyperloom/blob/main/CONTRIBUTING.md>`_.

Hyperloom is released under the MIT license. For details, see the :doc:`License <license>` page.
