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

    * `Quickstart <../examples/README.md>`_
    * :doc:`Slurm quickstart </install/slurm>`

  .. grid-item-card:: Overview

    * :doc:`Components </components/index>`
    * :doc:`Hyperloom optimization loop <conceptual/optimization-loop>`

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