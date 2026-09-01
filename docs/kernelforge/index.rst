.. meta::
  :description: KernelForge autonomously develops and optimizes high-performance GPU kernels on AMD Instinct hardware using domain-specialized AI agents and hardware-counter-driven measurement.
  :keywords: KernelForge, GPU kernel, AMD Instinct, MI355X, gfx950, optimization, agentic, ROCm, Composable Kernel, Triton, HIP, hipBLASLt, FlyDSL, PMC, documentation

***********
KernelForge
***********

KernelForge is an autonomous, measurement-driven system for developing and
optimizing high-performance GPU kernels on AMD Instinct hardware. It replaces
weeks of manual expert iteration with domain-specialized AI agents for
Composable Kernel, Triton, HIP, hipBLASLt, FlyDSL and AITER that build,
benchmark, and profile every change against real hardware counters — then learn
from each campaign.

KernelForge ships inside Hyperloom as the built-in kernel-optimization agent: installing
Hyperloom installs it, and its standalone CLI stays available as ``kernelforge``.
The source lives under ``src/kernelforge`` in the Hyperloom repository.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Quickstart </kernelforge/install/quickstart>`

  .. grid-item-card:: Overview

    * :doc:`Architecture </kernelforge/conceptual/architecture>`
    * :doc:`Optimization loop </kernelforge/conceptual/optimization-loop>`

  .. grid-item-card:: How to

    * :doc:`Run a campaign </kernelforge/how-to/run-a-campaign>`
    * :doc:`Autonomous overnight loop </kernelforge/how-to/autonomous-loop>`
    * :doc:`Fuse a launch-bound decode path </kernelforge/how-to/kernel-fusion>`
    * :doc:`Debug task preparation </kernelforge/how-to/debug-task-preparation>`
    * :doc:`Add a kernel backend, tool, or knowledge </kernelforge/how-to/extending>`

  .. grid-item-card:: Reference

    * :doc:`CLI reference </kernelforge/reference/cli>`
    * :doc:`Experience store </kernelforge/reference/experience-store>`
    * :doc:`Deployment modes </kernelforge/reference/deployment-modes>`
    * :doc:`API reference </kernelforge/reference/api-reference>`

To contribute to the documentation, see
`Contributing to Hyperloom <https://github.com/AMD-AGI/Hyperloom/blob/main/CONTRIBUTING.md>`_.
