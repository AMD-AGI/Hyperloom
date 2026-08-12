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

.. list-table::
   :header-rows: 1
   :widths: 15 30 45 35 20 15 5
   :class: compat-matrix format-big-table

   * - Hyperloom version
     - Component
     - GPU
     - ROCm version
     - Ubuntu
     - Python
     - GitHub
   * - 1.0.0a1
     - `TraceLens 0.1.0`_
     - Hardware-agnostic
     - No dependency
     - OS-independent
     - >= 3.6
     - |tracelens-github|
   * - 1.0.0a2
     - `TraceLens 0.1.0`_
     - Hardware-agnostic
     - No dependency
     - OS-independent
     - >= 3.6
     - |tracelens-github|
   * - 1.0.0a3
     - `TraceLens 0.1.0`_
     - Hardware-agnostic
     - No dependency
     - OS-independent
     - >= 3.6
     - |tracelens-github|
   * - 1.0.0a2
     - `GEAK 4.0.0`_
     - MI300X, MI325X, MI355X, RX 9070 XT, RX 9070, RX 9060 XT, R9000
     - 6.4.x, 7.0.x, 7.1.x, 7.2.x
     - 22.04, 24.04
     - 3.8, 3.12
     - |geak-github|
   * - 1.0.0a2
     - `IntelliKit 0.1.0`_
     - MI300X, MI325X, MI355X
     - 7.2.x
     - 22.04, 24.04
     - >= 3.10
     - |intellikit-github|
   * - 1.0.0a2
     - `AgentKernelArena 0.2.0`_
     - MI300X, MI325X, MI355X
     - 7.2.x
     - 22.04, 24.04
     - >= 3.10
     - |agent-kernel-arena-github|
   * - 1.0.0a2
     - `Magpie 0.2.0`_
     - MI300X, MI325X, MI355X, RX 9070 XT, RX 9070, RX 9060 XT, R9000
     - 7.0.x, 7.1.x, 7.2.x (Linux), 7.3+ (Windows for RDNA4)
     - 22.04, 24.04, Win 11
     - >= 3.10
     - |magpie-github|

.. _TraceLens 0.1.0: https://rocm.docs.amd.com/projects/tracelens/en/docs-0.1.0/
.. _GEAK 4.0.0: https://rocm.docs.amd.com/projects/geak/en/docs-4.0.0/
.. _IntelliKit 0.1.0: https://rocm.docs.amd.com/projects/intellikit/en/docs-0.1.0/
.. _AgentKernelArena 0.2.0: https://rocm.docs.amd.com/projects/agent-kernel-arena/en/docs-0.2.0/
.. _Magpie 0.2.0: https://rocm.docs.amd.com/projects/magpie/en/docs-0.2.0/

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
(``rocm/hyperloom:<tag>``) are used on your own GPU machine. If your
deployment uses a private registry mirror, set the registry prefix
accordingly.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Image
     - GPU
   * - ``rocm/hyperloom:sglang-v0.5.16-rocm7.2.0-mi300x``
     - MI300X / MI325X
   * - ``rocm/hyperloom:sglang-v0.5.16-rocm7.2.0-mi350x``
     - MI355X
   * - ``rocm/hyperloom:vllm-v0.24.0-rocm7.2.0``
     - MI300X / MI325X / MI355X

Browse all available tags at
`hub.docker.com/r/rocm/hyperloom/tags <https://hub.docker.com/r/rocm/hyperloom/tags>`_.

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
     - Bare-metal patch level differs per framework: the vLLM stack uses ROCm 7.2.3 and the SGLang stack uses ROCm 7.2.0 (see the note below). ``docker`` mode uses the ``rocm/hyperloom`` images (ROCm 7.2.x) for both frameworks.
   * - Python
     - 3.12
     - Required by the vLLM ROCm wheel.
   * - ROCm torch
     - ROCm build matching the host ROCm
     - Preinstalled by the operator; not managed by Hyperloom.
   * - SGLang
     - v0.5.16 (rocm720)
     - Installed in ``shared`` mode (reuses the host torch). Uses the ROCm 7.2.0 AMD wheel index (``SGLANG_ROCM_EXTRA=rocm720``), so the SGLang ROCm layer is 7.2.0. Note: ``SGLANG_REF`` (v0.5.16) only pins the version on the source-install branch (non-3.10 Python); on Python 3.10 the AMD wheel index installs ``amd-sglang`` unpinned, which may resolve to a different patch release.
   * - vLLM
     - v0.24.0 (rocm723), isolated venv
     - Installs ``vllm==0.24.0+rocm723`` from the wheels.vllm.ai pip index. vLLM's ROCm wheel pins its own torch, so it installs into a dedicated venv (``--framework-env isolated``, the default for vLLM) and never touches the host torch.

Bare-metal ROCm patch levels differ per framework. The vLLM version matches the
``v0.24.0`` Docker image; the pip index publishes 0.24.0 only as the ``rocm723``
variant (ROCm 7.2.3), so the bare-metal vLLM ROCm layer is 7.2.3. The SGLang
stack installs from the ROCm 7.2.0 AMD wheel index, so its ROCm layer is 7.2.0.
In ``docker`` mode both frameworks use the ``rocm/hyperloom`` images (ROCm
7.2.x). For a fully validated, pre-aligned vLLM stack, prefer ``docker`` mode
with ``rocm/hyperloom:vllm-v0.24.0-rocm7.2.0``.

These are recommended defaults, not hard pins. Framework and ROCm versions are
overridable via env (``SGLANG_REF``, ``SGLANG_ROCM_EXTRA``, ``VLLM_VERSION``,
``VLLM_ROCM_VARIANT``) for hosts that need a different pinned stack.

