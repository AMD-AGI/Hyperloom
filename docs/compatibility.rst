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
| ROCm Version        | 7.2.x, 10.0                                            |
+---------------------+--------------------------------------------------------+
| Python              | >= 3.10                                                |
+---------------------+--------------------------------------------------------+
| Inference Framework | SGLang (>= 0.5.12), vLLM (>= 0.21.0),                  |
|                     | ATOM (preinstalled; see below), plus ``custom``        |
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
| 1.1.2             | `TraceLens 1.0.0`_        | Hardware-agnostic      | No dependency                      | OS-independent| >= 3.6      | |tracelens-github|          |
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

   The base benchmark path remains compatible with Magpie 0.2.0. The tested
   native AgentX pair is `Magpie 0.3.0 release candidate
   <https://github.com/AMD-AGI/Magpie/pull/105>`_ commit
   ``3642ce66ae46ca4dc125340b3d14a3f4640c369b`` and InferenceX commit
   ``3d5581562f643f9bdeb8410cd924e2c70906c966``. Pass the Magpie YAML with
   ``--benchmark-config``; its ``benchmark.agentx: enable`` source switch
   automatically selects Hyperloom's persisted AgentX session and grading mode.
   ``HYPERLOOM_AGENTX`` remains an optional legacy switch and is not required
   for that path. ``benchmark.docker_image`` overrides/pins the effective
   recipe image and is included in its fingerprint; an existing
   ``HYPERLOOM_IMAGE`` is only a strict consistency assertion. Neither value
   starts or attests the outer container.
   ``install.sh`` upgrades an importable pre-AgentX Magpie
   build instead of treating import success as sufficient.

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
     - 10.0 (docker) / 7.2.4 (bare-metal wheel)
     - Default framework. Validated ``docker`` images use ROCm 10.0 (``rocm10`` tags below). On a ROCm 7.2.x bare-metal host the installer still derives ``SGLANG_ROCM_EXTRA=rocm724`` from ``torch.version.hip`` (see below).
   * - vLLM
     - 10.0 (docker, bare-metal source) / 7.2.3 (bare-metal wheel)
     - Validated ``docker`` image is the ROCm 10.0 build below. On bare metal the
       installer routes by the detected stack: a ROCm 7.2.x host installs the
       ``rocm723`` wheel, a ROCm 10 host (``torch.version.hip`` 7.15) builds vLLM
       from source, and any other combination is rejected (see below). Do not mix
       frameworks within one session
   * - ATOM
     - 7.2.4 (recorded image stack)
     - AMD out-of-tree engine, launched as ``python3 -m atom.entrypoints.openai_server``. Supports Docker or direct execution in a preinstalled ATOM/ROCm torch environment. The recorded image-based stack used ``rocm/atom-dev:v0.1.7-rc0`` on MI355X, not a universal ATOM pip-version minimum. Other builds require local validation. The CLI defaults an unset/empty ``KERNEL_OPT_BACKEND_ORDER`` to ``forge`` on ATOM and preserves explicit values; GEAK's live rewrite-seam resolution is unproven here.
   * - ``custom``
     - Host-defined
     - Escape hatch for your own benchmark script; Hyperloom does not manage the server lifecycle. Requires ``HYPERLOOM_BENCHMARK_BACKEND=bypass`` plus ``--framework-path`` (or ``FRAMEWORK_REPO_PATH``) and ``--benchmark-scripts-dir`` (or ``HYPERLOOM_BYPASS_SCRIPTS_DIR``); the CLI exits with status 2 when any of the three is missing.

Container images
----------------

Pick the image that matches your environment. Public Docker Hub refs are used
on your own GPU machine: the official upstream ``lmsysorg/sglang-rocm:<tag>``
for SGLang, ``rocm/vllm:<tag>`` for vLLM and
``rocm/atom-dev:<tag>`` for Atom. If your deployment uses a private registry
mirror, set the registry prefix accordingly.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Image
     - GPU
   * - ``lmsysorg/sglang-rocm:v0.5.20-rocm10-mi30x-20260920``
     - MI300X / MI325X
   * - ``lmsysorg/sglang-rocm:v0.5.20-rocm10-mi35x-20260920``
     - MI355X
   * - ``rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0``
     - MI300X / MI325X / MI355X
   * - ``rocm/atom-dev:v0.1.7-rc0``
     - MI355X (verified); MI300X / MI325X untested

The images default to ``/bin/bash``, which exits at once in a detached
container, so override the entrypoint (for example ``--entrypoint tail``) when
starting a long-running Hyperloom container.

The ``rocm/vllm`` image serves an installed wheel, so ``install.sh`` turns its
``/app/vllm`` checkout into the patchable source tree: it pins the tree to vLLM
commit ``f46a9dfe2c5f`` (``VLLM_IMAGE_SOURCE_COMMIT``) under a synthetic baseline
commit and puts it ahead of site-packages on the server's ``PYTHONPATH``. This
activates only when the installed wheel's version names that commit; any other
vLLM image is left on its wheel.

``rocm/atom-dev`` also publishes a ``latest`` tag, which tracks the newest
nightly build and moves. Pin the versioned tag so a session stays reproducible.

Browse all available tags at
`hub.docker.com/r/lmsysorg/sglang-rocm/tags <https://hub.docker.com/r/lmsysorg/sglang-rocm/tags>`_,
`hub.docker.com/r/rocm/vllm/tags <https://hub.docker.com/r/rocm/vllm/tags>`_
and
`hub.docker.com/r/rocm/atom-dev/tags <https://hub.docker.com/r/rocm/atom-dev/tags>`_.

Bare-metal recommended environment
-----------------------------------

For ``baremetal`` setup, align the host to this combination before running setup.
Hyperloom does not install ROCm or torch itself.

For ATOM, ``baremetal`` means running directly in the development machine's
selected Python environment, including when the development platform itself is
a container; it does not start an additional Docker container. ATOM must already
be installed. Verify a real ``import atom`` and a non-empty ``torch.version.hip``
with that Python, keep its executable first on ``PATH`` for Magpie's ``python3``
launch, and use ``PYTHON`` with ``INFERENCE_OPTIMIZER_FORCE_PYTHON=1`` if pinning
the interpreter. Keep any activated venv consistent; ``/opt/venv`` is not required.

Run ``python -m hyperloom.inference_optimizer.setup --check-only --
--install-framework none --frameworks atom --require-frameworks`` with the
selected interpreter first. Only after approval, repeat without ``--check-only``:
``none`` skips framework installation but can still write configuration and
apply ROCm hotfixes. Preserve the selected ``USER_DATA_PATH``; an existing setup
need not be repeated. Hyperloom does not install ATOM and does not assert a
minimum ATOM package version derived from a Docker tag. Import checks establish
local prerequisites, not end-to-end validation of every stack.

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
     - 7.2.x or 10.0 (bare metal) / 10.0 (docker)
     - Patch levels differ by framework and setup mode. On ROCm 7.2.x, vLLM installs the ROCm 7.2.3 wheel (``rocm723``); on ROCm 10.0, vLLM is built from source. On a ROCm 7.2.x bare-metal host, SGLang resolves ``SGLANG_ROCM_EXTRA=rocm724`` (ROCm 7.2.4 layer). The validated ``docker`` stacks use ROCm 10.0 user space (``rocm10`` images above), not the 7.2.x bare-metal wheel paths.
   * - Python
     - 3.12
     - Required by the vLLM ROCm wheel. The vLLM source build accepts Python >= 3.10, < 3.15.
   * - ROCm torch
     - ROCm build matching the host ROCm
     - Preinstalled by the operator; not managed by Hyperloom.
   * - SGLang
     - 0.5.20 (rocm10), pinned to commit ``94602c9c2b7c``
     - Recommended ``docker`` stack uses the ``lmsysorg/sglang-rocm:v0.5.20-rocm10-*`` images above (ROCm 10.0 user space). ``SGLANG_REF`` is the ``v0.5.20`` release commit (peeled from the tag object). The pin sits at 0.5.20 because through 0.5.18 the HIP extra pinned ``compressed-tensors==0.15.0``, which caps torch below 2.11 and therefore cannot resolve at all against a ROCm 10 stack; 0.5.19 moved that dependency into ``runtime_common`` unpinned, leaving the installer's ROCm torch constraint as the version pip solves for. Bare-metal installs on ROCm 10 take the source-install path; on ROCm 7.2.x hosts, ``SGLANG_ROCM_EXTRA=rocm724`` still selects the AMD wheel index. Kernel-shape profiling for SGLang >= 0.5.18 uses TraceLens ``kernel_shape_tool`` rather than git-applying SGLang roofline patches.
   * - vLLM
     - v0.29.0 (rocm723) wheel on ROCm 7.2.x; v0.29.0 source build, pinned to commit ``98dff2a81d74``, on ROCm 10; isolated venv
     - On ROCm 7.2.x, installs ``vllm==0.29.0+rocm723`` from the wheels.vllm.ai pip index on Ubuntu 24.04+. vLLM's ROCm wheel pins its own torch, so it installs into a dedicated venv (``--framework-env isolated``, the default for vLLM) and never touches the host torch. On ROCm 10 (``torch.version.hip`` 7.15) no wheel is published, so the installer checks out ``VLLM_SOURCE_REF`` (the ``v0.29.0`` release commit) into ``VLLM_ROOT`` (default ``/opt/hyperloom/vllm``) and builds it into ``VLLM_VENV_ROOT`` as a system-site-packages venv over the host ROCm torch. The source route requires ``--framework-env isolated``, ``git``, ``gcc``/``g++`` >= 11.3, ``cmake`` >= 3.26.1, ``ninja``, ``hipcc`` and the ROCm devel headers, installs AITER into the same venv when it is not already importable, and writes ``VLLM_ROCM_USE_AITER=1`` to ``.env`` when ``aiter`` imports. ``VLLM_INSTALL_METHOD`` (``auto``, ``wheel``, ``source``) can only confirm the detected route; a ROCm stack other than 7.2.x or 10 is rejected.

Bare-metal ROCm patch levels differ per framework. On ROCm 7.2.x the vLLM stack
installs the ``rocm723`` variant (ROCm 7.2.3); on ROCm 10 it builds from source
against the host ROCm, matching the ROCm 10 user space of the ``rocm/vllm`` image
the ``docker`` route uses; the SGLang stack
installs from the ROCm 7.2.4 AMD wheel index when a host stays on ROCm 7.2.x;
the recommended SGLang ``docker`` stack uses the two
``lmsysorg/sglang-rocm:v0.5.20-rocm10-*`` images (ROCm 10.0). ``docker`` mode is still
the preferred route for a pre-validated stack, since the images also pin the
surrounding torch, Triton, and AITER builds.

These are recommended defaults, not hard pins. Framework and ROCm versions are
overridable via env (``SGLANG_REF``, ``SGLANG_ROCM_EXTRA``, ``VLLM_VERSION``,
``VLLM_ROCM_VARIANT``, ``VLLM_INSTALL_METHOD``, ``VLLM_REPO``, ``VLLM_SOURCE_REF``,
``VLLM_ROOT``) for hosts that need a different pinned stack.

The table above is the validated combination. ROCm 7.2.x is validated under a
single ``/opt/rocm`` prefix. ROCm 10.0 arrives as TheRock's pip wheels, split
across the ``_rocm_sdk_*`` namespace packages, which is the layout the ``rocm10``
images are built from and the one the ROCm 10 routes are validated on: the
bare-metal installer probes those packages for library resolution and, before a
framework source build, supplies the devel headers and toolchain root from them.
Only ROCm 7.0.x and 7.2.x have a published ``amd-sglang`` wheel; any other stack
falls back to a source install. See :doc:`/install/install`.
