# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-import shim that installs the host-side evidence probe.

CPython imports a module named ``sitecustomize`` during start-up if one is
importable, which is how the probe gets into a benchmark process launched by an
arbitrary framework's own entrypoint (``torchrun``, a shell wrapper, whatever)
without editing that entrypoint. The orchestrator prepends this directory to
``PYTHONPATH``; nothing else is required of the target framework.

Two properties matter here:

* **Chaining.** Prepending this directory shadows any ``sitecustomize`` the
  image already ships, which could silently drop that environment's own
  start-up configuration. So this module finds the next ``sitecustomize`` on the
  path *after* our directory and imports it first, under its real name, before
  installing the probe.
* **Inertness.** The probe installs only when ``HYPERLOOM_HOST_PROBE`` is
  truthy. A benchmark that merely has this directory on ``PYTHONPATH`` behaves
  exactly as if it did not, which is what makes it safe to leave the prefix in
  place across an entire session.

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
