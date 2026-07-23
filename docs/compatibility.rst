.. meta::
   :description: Compatibility matrix for Hyperloom: supported AMD Instinct GPUs, inference frameworks (SGLang, vLLM), container images, and component dependencies.
   :keywords: Hyperloom, compatibility, AMD Instinct, MI300X, MI355X, SGLang, vLLM, ROCm, container images, GPU support

******************************
Hyperloom compatibility matrix
******************************

.. |github-icon| raw:: html

   <i class="fab fa-github"></i>

.. |tracelens-github| raw:: html

   <a href="https://github.com/AMD-AGI/TraceLens"><i class="fab fa-github"></i></a>

.. |geak-github| raw:: html

   <a href="https://github.com/AMD-AGI/GEAK"><i class="fab fa-github"></i></a>

.. |intellikit-github| raw:: html

   <a href="https://github.com/AMDResearch/intellikit"><i class="fab fa-github"></i></a>

.. |agent-kernel-arena-github| raw:: html

   <a href="https://github.com/AMD-AGI/AgentKernelArena"><i class="fab fa-github"></i></a>

.. |magpie-github| raw:: html

   <a href="https://github.com/AMD-AGI/Magpie"><i class="fab fa-github"></i></a>

This topic lists the hardware, inference frameworks, and container images that
Hyperloom is validated against.

.. note::

  ROCm versions or framework builds not listed in this matrix might work, but are not regularly tested.

Hyperloom support matrix
========================

The following table lists the minimum requirements for running Hyperloom.

+---------------------+---------------------------------------+
| Requirement         | Support                               |
+=====================+=======================================+
| AMD Instinct GPU    | MI300X, MI325X, MI355X                |
+---------------------+---------------------------------------+
| Operating System    | Ubuntu 22.04, Ubuntu 24.04            |
+---------------------+---------------------------------------+
| ROCm Version        | 7.2.x                                 |
+---------------------+---------------------------------------+
| Python              | >= 3.10                               |
+---------------------+---------------------------------------+
| Inference Framework | SGLang (>= 0.5.12), vLLM (>= 0.21.0)  |
+---------------------+---------------------------------------+
| Kernel Languages    | HIP, Triton, FlyDSL                   |
+---------------------+---------------------------------------+

Component support matrix
========================

The following table lists the validated Hyperloom version and component combinations.

.. role:: version-start

.. table::
   :widths: 6 27 10 10 14 30 3
   :align: left
   :class: compat-matrix format-big-table

+-------------------+---------------------------+------------------------+----------------------------+---------------+-------------+-----------------------------+
| Hyperloom version | Component                 | GPU                    | ROCm version               | Ubuntu        | Python      | GitHub                      |
+===================+===========================+========================+============================+===============+=============+=============================+
| 1.0.0a1           | `TraceLens 0.1.0`_        | Hardware-agnostic      | No dependency              | OS-independent| >= 3.6      | |tracelens-github|          |
+                   +---------------------------+------------------------+----------------------------+---------------+-------------+-----------------------------+
|                   | `GEAK 4.0.0`_             | MI300X, MI325X, MI355X | 6.4.x, 7.0.x, 7.1.x, 7.2.x | 22.04, 24.04  | 3.8, 3.12   | |geak-github|               |
+                   +---------------------------+------------------------+----------------------------+---------------+-------------+-----------------------------+
|                   | `IntelliKit 0.1.0`_       | MI300X, MI325X, MI355X | 7.2.x                      | 22.04, 24.04  | >= 3.10     | |intellikit-github|         |
+                   +---------------------------+------------------------+----------------------------+---------------+-------------+-----------------------------+
|                   | `AgentKernelArena 0.2.0`_ | MI300X, MI325X, MI355X | 7.2.x                      | 22.04, 24.04  | >= 3.10     | |agent-kernel-arena-github| |
+                   +---------------------------+------------------------+----------------------------+---------------+-------------+-----------------------------+
|                   | `Magpie 0.2.0`_           | MI300X, MI325X, MI355X | 7.0.x, 7.1.x, 7.2.x        | 22.04, 24.04  | >= 3.10     | |magpie-github|             |
+-------------------+---------------------------+------------------------+----------------------------+---------------+-------------+-----------------------------+

.. _TraceLens 0.1.0: https://rocm.docs.amd.com/projects/tracelens/en/latest/
.. _GEAK 4.0.0: https://rocm.docs.amd.com/projects/geak/en/latest/
.. _IntelliKit 0.1.0: https://rocm.docs.amd.com/projects/intellikit/en/latest/
.. _AgentKernelArena 0.2.0: https://rocm.docs.amd.com/projects/agent-kernel-arena/en/latest/
.. _Magpie 0.2.0: https://rocm.docs.amd.com/projects/magpie/en/latest

.. note::

   TraceLens does not have hard requirements for the GPU, ROCm version, or the OS; it has scripts to verify whether a trace is valid/parseable. TraceLens is:

   - OS-independent and runs anywhere Python does.
   - Not limited to MI300X/MI325X/MI355X; it's hardware-agnostic.

   See the `TraceLens documentation <https://rocm.docs.amd.com/projects/tracelens/en/latest/reference/compatibility.html>`_ for more information.

.. note::

   MI325X shares the gfx942/CDNA3 runner family with MI300X. Hyperloom
   keeps the resolved GPU type distinct (``mi325x``), but Magpie benchmark
   rendering reuses the MI300X runner scripts and image family unless a dedicated
   image is supplied.

Inference frameworks
--------------------

The following inference frameworks are supported:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Framework
     - ROCm version
     - Notes
   * - SGLang
     - 7.2.0
     - Default framework
   * - vLLM
     - 7.2.0
     - Do not mix frameworks within one session
   * - Atom
     - 7.2.0
     - Single-node only (multi-node rejected by the IR-8 guard)
   * - xDiT (diffusion)
     - 7.2.0
     - Scriptable diffusion pipeline (no serving server). Internal throughput is tracked in img/s, but the primary session-facing metric is end-to-end latency ``e2el_mean_ms`` (ms).

Container images
----------------

Pick the image that matches your environment. Public Docker Hub refs
(``primussafe/sglang:<tag>``) are used on your own GPU machine. If your
deployment uses a private registry mirror, set the registry prefix
accordingly.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Image
     - GPU
   * - ``primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix``
     - MI300X / MI325X
   * - ``primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix``
     - MI355X
   * - ``primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix``
     - MI300X / MI325X / MI355X

Browse all available SGLang tags at
`hub.docker.com/r/primussafe/sglang/tags <https://hub.docker.com/r/primussafe/sglang/tags>`_.

Bare-metal recommended environment
-----------------------------------

For ``baremetal`` setup, align the host to this combination before running setup.
Hyperloom does not install ROCm or torch itself.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Item
     - Recommended
     - Notes
   * - ROCm
     - 7.2.x
     - Bare-metal patch level differs per framework: the vLLM stack uses ROCm 7.2.2 and the SGLang stack uses ROCm 7.2.0 (see the note below). ``docker`` mode uses ROCm 7.2.0 for both frameworks (the ``-rocm720-`` images).
   * - Python
     - 3.12
     - Required by the vLLM ROCm wheel.
   * - ROCm torch
     - ROCm build matching the host ROCm
     - Preinstalled by the operator; not managed by Hyperloom.
   * - SGLang
     - v0.5.12 (rocm720)
     - Installed in ``shared`` mode (reuses the host torch). Uses the ROCm 7.2.0 AMD wheel index (``SGLANG_ROCM_EXTRA=rocm720``), so the SGLang ROCm layer is 7.2.0.
   * - vLLM
     - v0.21.0 (rocm722), isolated venv
     - Installs ``vllm==0.21.0+rocm722`` from the wheels.vllm.ai pip index. vLLM's ROCm wheel pins its own torch, so it installs into a dedicated venv (``--framework-env isolated``, the default for vLLM) and never touches the host torch.

Bare-metal ROCm patch levels differ per framework. The vLLM version matches the
``v0.21.0`` Docker image, but the pip index only publishes the ``rocm722``
variant (ROCm 7.2.2), so the bare-metal vLLM ROCm layer is 7.2.2 rather than the
image's rocm720. The SGLang stack installs from the ROCm 7.2.0 AMD wheel index,
so its ROCm layer is 7.2.0. In ``docker`` mode both frameworks use ROCm 7.2.0
(the ``-rocm720-`` images). For a fully validated, pre-aligned vLLM stack with
rocm720, prefer ``docker`` mode with
``primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix``.

These are recommended defaults, not hard pins. Framework and ROCm versions are
overridable via env (``SGLANG_REF``, ``SGLANG_ROCM_EXTRA``, ``VLLM_VERSION``,
``VLLM_ROCM_VARIANT``) for hosts that need a different pinned stack.
