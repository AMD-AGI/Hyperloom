.. meta::
  :description: Learn how Hyperloom autonomously optimizes large language model inference on AMD GPUs — from installation and components to API reference and case studies.
  :keywords: Hyperloom, LLM inference, AMD GPU, optimization, agentic, ROCm, GEMM tuning, kernel optimization, throughput, TraceLens, Magpie, GEAK, IntelliKit, documentation

***********************
Hyperloom documentation
***********************

Hyperloom is an agentic system that autonomously optimizes large language model (LLM) inference on AMD
GPUs. It treats optimization as a search problem: given a workload, it
explores candidate optimizations one change at a time, always measuring against
the real workload and using prior results plus knowledge base (KB) priors to choose the next
move.

The Hyperloom source code is hosted on GitHub at `https://github.com/AMD-AGI/Hyperloom <https://github.com/AMD-AGI/Hyperloom>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install Hyperloom </install/hyperloom-installation>`

  .. grid-item-card:: Components

    * :doc:`Components </components/index>`

  .. grid-item-card:: How to

    * :doc:`Run a Hyperloom optimization </how-to/optimize>`

  .. grid-item-card:: Conceptual

    * :doc:`Hyperloom optimization loop <conceptual/optimization-loop>`
  
  .. grid-item-card:: Reference

    * :doc:`API reference </reference/api-reference>`
    * :doc:`Environment variables </reference/environment-variables>`
    * :doc:`Authentication and credentials </reference/authentication>`
    * :doc:`Troubleshooting </reference/troubleshooting>`  

To contribute to the documentation, see `Contributing to Hyperloom <https://github.com/AMD-AGI/Hyperloom/blob/main/CONTRIBUTING.md>`_.

Magpie is released under the MIT license. For details, see the :doc:`License <license>` page.