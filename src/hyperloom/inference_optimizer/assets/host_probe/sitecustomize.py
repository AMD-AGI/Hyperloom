# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-import shim that installs the host-side evidence probe.

CPython imports a module named ``sitecustomize`` during start-up if one is
importable, which is how the probe gets into a benchmark process launched by an
arbitrary framework's own entrypoint (``torchrun``, a shell wrapper, whatever)
without editing that entrypoint. The orchestrator prepends this directory to
``PYTHONPATH``; nothing else is required of the target framework.

Prepending this directory shadows any ``sitecustomize`` the image already
ships, so this module executes the next one on ``sys.path`` under a private
name before installing the probe. The probe itself installs only when
``HYPERLOOM_HOST_PROBE`` is truthy, so leaving the prefix in place across a
session is inert.

Every failure path is swallowed: a start-up shim must never be the reason a
benchmark process fails to boot.
"""

from __future__ import annotations


def _chain_preexisting_sitecustomize() -> None:
    """Import the ``sitecustomize`` this module shadows, if there is one.

    Searches ``sys.path`` entries after this file's own directory so the
    environment's own start-up hook still runs.
    """
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
