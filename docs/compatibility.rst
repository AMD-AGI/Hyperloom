.. meta::
   :description: Compatibility matrix for Hyperloom: supported AMD Instinct GPUs, inference frameworks (SGLang, vLLM, Atom, xDiT), container images, and component dependencies.
   :keywords: Hyperloom, compatibility, AMD Instinct, MI300X, MI325X, MI355X, SGLang, vLLM, Atom, xDiT, ROCm, container images, GPU support

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

+---------------------+--------------------------------------------------------+
| Requirement         | Support                                                |
+=====================+========================================================+
| AMD Instinct™ GPU   | MI300X, MI325X, MI355X                                 |
+---------------------+--------------------------------------------------------+
| Operating System    | Ubuntu 24.04 (recommended); Ubuntu 22.04 (legacy)      |
+---------------------+--------------------------------------------------------+
| ROCm Version        | 7.2.x                                                  |
+---------------------+--------------------------------------------------------+
| Python              | >= 3.10                                                |
+---------------------+--------------------------------------------------------+
| Inference Framework | SGLang (>= 0.5.12), vLLM (>= 0.21.0),                  |
|                     | Atom (>= 0.1.7-rc0), plus ``custom`` for your own      |
|                     | benchmark script                                       |
+---------------------+--------------------------------------------------------+
| Kernel Languages    | HIP, Triton, FlyDSL                                    |
+---------------------+--------------------------------------------------------+

Component support matrix
========================

The following table lists the validated Hyperloom version and component combinations.

.. role:: version-start

.. table::
   :widths: 6 27 10 10 14 30 3
   :align: left
   :class: compat-matrix format-big-table

+-------------------+---------------------------+------------------------+------------------------------------+---------------+-------------+-----------------------------+
| Hyperloom version | Component                 | GPU                    | ROCm version                       | Ubuntu        | Python      | GitHub                      |
+===================+===========================+========================+====================================+===============+=============+=============================+
| 1.1.1             | `TraceLens 1.0.0`_        | Hardware-agnostic      | No dependency                      | OS-independent| >= 3.6      | |tracelens-github|          |
+                   +---------------------------+------------------------+------------------------------------+---------------+-------------+-----------------------------+
|                   | `GEAK 4.0.0`_             | MI300X, MI325X, MI355X | 6.4.x, 7.0.x, 7.1.x, 7.2.x, 10.0.0 | 22.04, 24.04  | 3.8, 3.12   | |geak-github|               |
+                   +---------------------------+------------------------+------------------------------------+---------------+-------------+-----------------------------+
|                   | `IntelliKit 0.1.1`_       | MI300X, MI325X, MI355X | 7.2.x, 10.0.0                      | 22.04, 24.04  | >= 3.10     | |intellikit-github|         |
+                   +---------------------------+------------------------+------------------------------------+---------------+-------------+-----------------------------+
|                   | `AgentKernelArena 0.2.0`_ | MI300X, MI325X, MI355X | 7.2.x, 10.0.0                      | 22.04, 24.04  | >= 3.10     | |agent-kernel-arena-github| |
+                   +---------------------------+------------------------+------------------------------------+---------------+-------------+-----------------------------+
|                   | `Magpie 0.2.0`_           | MI300X, MI325X, MI355X | 7.0.x, 7.1.x, 7.2.x, 10.0.0        | 22.04, 24.04  | >= 3.10     | |magpie-github|             |
+-------------------+---------------------------+------------------------+------------------------------------+---------------+-------------+-----------------------------+

.. _TraceLens 1.0.0: https://rocm.docs.amd.com/projects/tracelens/en/docs-1.0.0/
.. _GEAK 4.0.0: https://rocm.docs.amd.com/projects/geak/en/docs-4.0.0/
.. _IntelliKit 0.1.1: https://rocm.docs.amd.com/projects/intellikit/en/docs-0.1.1/
.. _AgentKernelArena 0.2.0: https://rocm.docs.amd.com/projects/agent-kernel-arena/en/docs-0.2.0/
.. _Magpie 0.2.0: https://rocm.docs.amd.com/projects/magpie/en/docs-0.2.0/

.. note::

   TraceLens does not have hard requirements for the GPU, ROCm version, or the OS; it has scripts to verify whether a trace is valid/parseable. TraceLens is:

   - OS-independent and runs anywhere Python does.
   - Not limited to MI300X/MI325X/MI355X; it's hardware-agnostic.

   See the `TraceLens documentation <https://rocm.docs.amd.com/projects/tracelens/en/latest/reference/compatibility.html>`_ for more information.

.. note::

   MI325X shares the gfx942/CDNA3 runner family with MI300X. Hyperloom keeps the
   resolved GPU types distinct, but Magpie benchmark rendering reuses the MI300X
   runner scripts and image family unless a dedicated image is supplied.

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
     - 7.2.4
     - Default framework; recommended docker/bare-metal stack uses ``rocm724`` (see below)
   * - vLLM
     - 7.2.3
     - Do not mix frameworks within one session
   * - Atom
     - 7.2.4
     - AMD out-of-tree engine, launched as ``python3 -m atom.entrypoints.openai_server``. Container image only: ``install_baremetal.sh`` verifies ``atom`` but cannot install it, because ``--install-framework`` accepts only ``none``, ``sglang`` and ``vllm``. The kernel phase defaults to the KernelForge backend here (``KERNEL_OPT_BACKEND_ORDER`` is set to ``forge`` when you leave it unset). On a quantized non-vLLM backend GEAK must resolve a live rewrite seam rather than guess one, which forge does not require.
   * - ``custom``
     - Host-defined
     - Escape hatch for your own benchmark script; Hyperloom does not manage the server lifecycle. Requires ``HYPERLOOM_BENCHMARK_BACKEND=bypass`` plus ``--framework-path`` (or ``FRAMEWORK_REPO_PATH``) and ``--benchmark-scripts-dir`` (or ``HYPERLOOM_BYPASS_SCRIPTS_DIR``); the CLI exits with status 2 when any of the three is missing.

Container images
----------------

Pick the image that matches your environment. Public Docker Hub refs are used
on your own GPU machine: the official upstream ``lmsysorg/sglang-rocm:<tag>``
for SGLang, ``vllm/vllm-openai-rocm:<tag>`` for vLLM and
``rocm/atom-dev:<tag>`` for Atom. If your deployment uses a private registry
mirror, set the registry prefix accordingly.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Image
     - GPU
   * - ``lmsysorg/sglang-rocm:v0.5.20-rocm724-mi30x-20260919``
     - MI300X / MI325X
   * - ``lmsysorg/sglang-rocm:v0.5.20-rocm724-mi35x-20260920``
     - MI355X
   * - ``vllm/vllm-openai-rocm:v0.29.0``
     - MI300X / MI325X / MI355X
   * - ``rocm/atom-dev:v0.1.7-rc0``
     - MI355X (verified); MI300X / MI325X untested

The vLLM image entrypoint is ``vllm serve``, so override it (for example
``--entrypoint tail``) when starting a long-running Hyperloom container.

``rocm/atom-dev`` also publishes a ``latest`` tag, which tracks the newest
nightly build and moves. Pin the versioned tag so a session stays reproducible.

Browse all available tags at
`hub.docker.com/r/lmsysorg/sglang-rocm/tags <https://hub.docker.com/r/lmsysorg/sglang-rocm/tags>`_,
`hub.docker.com/r/vllm/vllm-openai-rocm/tags <https://hub.docker.com/r/vllm/vllm-openai-rocm/tags>`_
and
`hub.docker.com/r/rocm/atom-dev/tags <https://hub.docker.com/r/rocm/atom-dev/tags>`_.

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
   * - Operating System
     - Ubuntu 24.04
     - Recommended bare-metal baseline. vLLM 0.28.0+ ROCm wheels require glibc >= 2.39, so Ubuntu 22.04 hosts must downgrade vLLM (for example ``VLLM_VERSION=0.27.1``) or use ``docker`` mode instead.
   * - ROCm
     - 7.2.x
     - The patch level differs per framework and is the same in both setup modes: the vLLM stack uses ROCm 7.2.3 and the SGLang stack uses ROCm 7.2.4 (see the note below).
   * - Python
     - 3.12
     - Required by the vLLM ROCm wheel.
   * - ROCm torch
     - ROCm build matching the host ROCm
     - Preinstalled by the operator; not managed by Hyperloom.
   * - SGLang
     - 0.5.20 (rocm724), pinned to commit ``d158602ff1d2``
     - Installed in ``shared`` mode (reuses the host torch). The wheel target is derived from the ROCm build of the installed torch rather than defaulted, so a ROCm 7.2.x stack resolves ``SGLANG_ROCM_EXTRA=rocm724`` and the SGLang ROCm layer is 7.2.4. ``SGLANG_REF`` defaults to the ``v0.5.20`` release commit, aligned with the ``lmsysorg/sglang-rocm:v0.5.20-rocm724-*`` images. Kernel-shape profiling for SGLang >= 0.5.18 uses TraceLens ``kernel_shape_tool`` (``PYTHONPATH`` + ``TRACELENS_SHAPE_DISCOVERY``) rather than git-applying SGLang roofline patches. Note: ``SGLANG_REF`` only pins the version on the source-install branch, which is taken for any Python other than 3.10 and for a ROCm stack no published wheel targets; on Python 3.10 with a derived target the AMD wheel index installs ``amd-sglang`` unpinned, which might resolve to a different patch release.
   * - vLLM
     - v0.29.0 (rocm723), isolated venv
     - Installs ``vllm==0.29.0+rocm723`` from the wheels.vllm.ai pip index on Ubuntu 24.04+. vLLM's ROCm wheel pins its own torch, so it installs into a dedicated venv (``--framework-env isolated``, the default for vLLM) and never touches the host torch.

Bare-metal ROCm patch levels differ per framework, and each one matches its
container image. The vLLM stack installs the ``rocm723`` variant (ROCm
7.2.3), matching ``vllm/vllm-openai-rocm:v0.29.0``; the SGLang stack
installs from the ROCm 7.2.4 AMD wheel index, matching the two
``lmsysorg/sglang-rocm:v0.5.20-rocm724-*`` images. ``docker`` mode is still
the preferred route for a pre-validated stack, since the images also pin the
surrounding torch, Triton, and AITER builds.

These are recommended defaults, not hard pins. Framework and ROCm versions are
overridable via env (``SGLANG_REF``, ``SGLANG_ROCM_EXTRA``, ``VLLM_VERSION``,
``VLLM_ROCM_VARIANT``) for hosts that need a different pinned stack.

The table above is the validated combination, and ROCm 7.2.x under a single
``/opt/rocm`` prefix is the layout to prefer. A host where ROCm arrives as
TheRock's pip wheels instead, split across the ``_rocm_sdk_*`` namespace
packages, is handled rather than validated: the bare-metal installer probes
those packages for library resolution and, before a framework source build,
supplies the devel headers and toolchain root from them, so setup does not fail
on that layout. Only ROCm 7.0.x and 7.2.x have a published ``amd-sglang`` wheel;
any other stack falls back to a source install. See :doc:`/install/install`.
