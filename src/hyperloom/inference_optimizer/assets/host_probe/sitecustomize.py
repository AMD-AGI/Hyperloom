# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-import shim of the profile run's processes.

It drops ``ROCPROFILER_REGISTER_LIBRARY``, imports the ``sitecustomize`` it shadows, then
installs the host-side evidence probe when the run arms it. The ROCm runtime sets that
variable in any process that initializes the GPU; a child spawned afterwards (vLLM's
EngineCore) inherits it, and that child's torch profiler then records no ``kernel`` or
``cuda_runtime`` events. It must be gone before torch loads the HIP runtime.
"""

from __future__ import annotations

import os

os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)


def _chain_preexisting_sitecustomize() -> None:
    """Import the ``sitecustomize`` this module shadows, if there is one."""
    import importlib.util
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        try:
            resolved = os.path.abspath(entry or ".")
        except (OSError, ValueError):
            continue
        if resolved == here:
            continue
        candidate = os.path.join(resolved, "sitecustomize.py")
        if not os.path.isfile(candidate):
            continue
        spec = importlib.util.spec_from_file_location("_hl_prior_sitecustomize", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


# A failing prior hook still reaches ``site``, which reports it; the probe installs either way.
try:
    _chain_preexisting_sitecustomize()
finally:
    import hl_host_probe

    hl_host_probe.install_from_env()
