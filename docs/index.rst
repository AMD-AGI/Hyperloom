.. meta::
  :description: Learn how Hyperloom autonomously optimizes large language model inference on AMD GPUs — from installation and components to API reference and case studies.
  :keywords: Hyperloom, LLM inference, AMD GPU, optimization, agentic, ROCm, GEMM tuning, kernel optimization, throughput, TraceLens, Magpie, GEAK, IntelliKit, documentation

***********************
Hyperloom documentation
***********************

ROCm Hyperloom is an autonomous agentic system designed to optimize end-to-end inference workloads
(targeting both host code and GPU kernels) on AMD GPUs. Using advanced AI agents and profiling tools,
Hyperloom analyzes your workload, identifies performance bottlenecks, implements targeted optimizations,
and validates the performance and correctness of the optimizations without requiring manual intervention.
 
The Hyperloom source code is hosted on GitHub at `https://github.com/AMD-AGI/Hyperloom <https://github.com/AMD-AGI/Hyperloom>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install on Docker or Bare metal </install/install>`
    * :doc:`Install for Slurm </install/slurm>`

  .. grid-item-card:: Components

    * :doc:`Components </components/index>`
    * :doc:`Hyperloom optimization loop </conceptual/optimization-loop>`
    * :doc:`Kernel optimization execution path </conceptual/kernel-execution-path>`

  .. grid-item-card:: How to

    * :doc:`Run a Hyperloom optimization </how-to/optimize>`
    * :doc:`Quantization with AMD Quark </how-to/quantization-quark>`
    * :doc:`Multi-node inference optimization demo </how-to/multi-node/hyperloom-remote-demo>`

  .. grid-item-card:: Reference

    * :doc:`API reference </reference/api-reference>`
    * :doc:`Environment variables </reference/environment-variables>`
    * :doc:`Authentication and credentials </reference/authentication>`
    * :doc:`Operations and self-hosting </reference/operations>`
    * :doc:`Upgrade guide </reference/upgrade>`
    * :doc:`Session output schema </reference/session-breakdown>`
    * :doc:`Knowledge base integration </reference/integrate-kb>`
    * :doc:`Operator scripts </reference/operator-scripts>`
    * :doc:`Troubleshooting </reference/troubleshooting>`

To contribute to the documentation, see `Contributing to Hyperloom <https://github.com/AMD-AGI/Hyperloom/blob/main/CONTRIBUTING.md>`_.

Hyperloom is released under the MIT license. For details, see the :doc:`License <license>` page.
