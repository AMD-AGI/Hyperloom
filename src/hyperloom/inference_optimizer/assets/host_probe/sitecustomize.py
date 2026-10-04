# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-import shim that installs the host-side evidence probe."""

from __future__ import annotations


def _chain_preexisting_sitecustomize() -> None:
    """Import the ``sitecustomize`` this module shadows, if there is one."""
    import importlib.util
    import os
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
        try:
            spec = importlib.util.spec_from_file_location("_hl_prior_sitecustomize", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001 - the prior hook's failure is not ours
            pass
        return


try:
    _chain_preexisting_sitecustomize()
except Exception:  # noqa: BLE001
    pass

try:
    import hl_host_probe

    hl_host_probe.install_from_env()
except Exception:  # noqa: BLE001 - never block interpreter start-up
    pass
