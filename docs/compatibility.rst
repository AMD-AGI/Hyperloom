.. meta::
   :description: Compatibility matrix for Hyperloom: supported AMD Instinct GPUs, inference frameworks (SGLang, vLLM), container images, and component dependencies.
   :keywords: Hyperloom, compatibility, AMD Instinct, MI300X, MI355X, SGLang, vLLM, ROCm, container images, GPU support

Hyperloom compatibility matrix
===============================

.. |github-icon| raw:: html

   <i class="fab fa-github"></i>

This topic lists the hardware, inference frameworks, and container images that
Hyperloom is validated against.

.. note::

   This matrix tracks the currently validated combinations. Other ROCm versions
   or framework builds might work but, are not regularly tested.

+-------------------+-----------------------+---------------------+----------------------+----------------------------+--------+-----------------------------------------------+
| Hyperloom Version | Component             | GPU Support         | ROCm Version Support | OS Support                 | Python | GitHub                                        |
+===================+=======================+=====================+======================+============================+========+===============================================+
| 0.9.0             | TraceLens 0.1.0       | N/A                 | N/A                  | N/A                        | >= 3.6 | |github-icon| `AMD-AGI/TraceLens`_            |
+                   +-----------------------+---------------------+----------------------+----------------------------+--------+-----------------------------------------------+
|                   | GEAK 4.0              | MI300, MI325, MI355 | 7.2.x                | Ubuntu 22.04, Ubuntu 24.04 | > 3.8  | |github-icon| `AMD-AGI/GEAK`_                 |
+                   +-----------------------+---------------------+----------------------+----------------------------+--------+-----------------------------------------------+
|                   | IntelliKit 0.1.0      | MI300, MI325, MI355 | 7.2.x                | Ubuntu 22.04, Ubuntu 24.04 | >= 3.10| |github-icon| `AMDResearch/intellikit`_       |
+                   +-----------------------+---------------------+----------------------+----------------------------+--------+-----------------------------------------------+
|                   | AgentKernelArena 0.1.0| MI300, MI325, MI355 | 7.2.x                | Ubuntu 22.04, Ubuntu 24.04 | >= 3.10| |github-icon| `AMD-AGI/AgentKernelArena`_     |
+                   +-----------------------+---------------------+----------------------+----------------------------+--------+-----------------------------------------------+
|                   | Magpie 0.1.0          | MI300, MI325, MI355 | 7.2.x                | Ubuntu 22.04, Ubuntu 24.04 | >= 3.10| |github-icon| `AMD-AGI/Magpie`_               |
+-------------------+-----------------------+---------------------+----------------------+----------------------------+--------+-----------------------------------------------+

.. _AMD-AGI/TraceLens: https://github.com/AMD-AGI/TraceLens
.. _AMD-AGI/GEAK: https://github.com/AMD-AGI/GEAK
.. _AMDResearch/intellikit: https://github.com/AMDResearch/intellikit
.. _AMD-AGI/AgentKernelArena: https://github.com/AMD-AGI/AgentKernelArena
.. _AMD-AGI/Magpie: https://github.com/AMD-AGI/Magpie

.. note::

   TraceLens does not have hard requirements for the GPU, ROCm version, or the OS; it has scripts to verify whether a trace is valid/parseable. TraceLens is:

   - OS-independent and runs anywhere Python does.
   - Not limited to MI300/MI325/MI355; it's hardware-agnostic.

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
   * - SGLang (ROCm)
     - 7.2.0
     - Default framework
   * - vLLM (ROCm)
     - 7.2.0
     - Do not mix frameworks within one session
   * - Atom (ROCm)
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
     - Matches the validated framework stacks above.
   * - Python
     - 3.12
     - Required by the vLLM ROCm wheel.
   * - ROCm torch
     - ROCm build matching the host ROCm
     - Preinstalled by the operator; not managed by Hyperloom.
   * - SGLang
     - v0.5.12
     - Installed in ``shared`` mode (reuses the host torch).
   * - vLLM
     - isolated venv
     - vLLM's ROCm wheel pins its own torch, so it installs into a dedicated venv (``--framework-env isolated``, the default for vLLM) and never touches the host torch.

For a fully validated, pre-aligned vLLM stack, prefer ``docker`` mode with
``primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`` — the bare-metal vLLM
wheel index only publishes rolling versions, so exact ``v0.21.0`` parity is
available through the container image, not pip.

Framework versions are overridable via env (``SGLANG_REF``, ``VLLM_VERSION``,
``VLLM_ROCM_VARIANT``) for hosts that need a different pinned stack.
