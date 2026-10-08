# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-import shim that keeps GPU activity in a spawned engine's torch-profiler trace.

The ROCm runtime sets ``ROCPROFILER_REGISTER_LIBRARY`` in any process that initializes
the GPU. A child spawned afterwards (vLLM's EngineCore) inherits it, and that child's
torch profiler then records no ``kernel`` or ``cuda_runtime`` events. The variable must
be gone before torch loads the HIP runtime, so it is dropped at interpreter start-up.
"""

from __future__ import annotations

import os

os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)


def _chain_following_sitecustomize() -> None:
    """Import the next ``sitecustomize`` after this one on ``sys.path``, if there is one.

    Only entries after this directory are searched, and a second run in the same
    process chains nothing, so a hook that chains back to this one cannot loop.
    """
    import importlib.util
    import sys

    if getattr(sys, "_hl_profile_trace_env_chained", False):
        return
    sys._hl_profile_trace_env_chained = True
    here = os.path.dirname(os.path.abspath(__file__))
    seen_here = False
    for entry in sys.path:
        try:
            resolved = os.path.abspath(entry or ".")
        except (OSError, ValueError):
            continue
        if resolved == here:
            seen_here = True
            continue
        if not seen_here:
            continue
        candidate = os.path.join(resolved, "sitecustomize.py")
        if not os.path.isfile(candidate):
            continue
        try:
            spec = importlib.util.spec_from_file_location("_hl_following_sitecustomize", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001 - the following hook's failure is not ours
            pass
        return


try:
    _chain_following_sitecustomize()
except Exception:  # noqa: BLE001 - never block interpreter start-up
    pass
