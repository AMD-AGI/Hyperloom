# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Workspace-root resolver for the standalone kernel-agent tools.

The tools in this directory (``kernel_optimization.py``,
``tracelens_analysis.py``, ``parallel_e2e_runner.py``) run as standalone
scripts with ``sys.path`` injection of the tools dir and therefore
CANNOT ``import inference_optimizer.paths``. This module mirrors the
``workspace_root()`` semantics of that package using only the standard
library so every kernel-agent artefact lands under ``$USER_DATA_PATH``
just like the orchestrator's.

Contract (kept in lock-step with
``inference_optimizer/paths.py::workspace_root``):

* ``$USER_DATA_PATH`` set/non-empty  -> return it verbatim; NEVER the
  ``/workspace/hyperloom`` default.
* ``$USER_DATA_PATH`` unset/empty     -> emit ONE loud ``logging.warning``
  (process-wide, guarded) and return ``DEFAULT_WORKSPACE_ROOT`` so a
  misconfigured launcher is visible instead of silently writing to the
  pod-local default.
"""

from __future__ import annotations

import logging
import os

# Single source of the fallback literal for the kernel-agent tools. Kept
# byte-for-byte identical to
# ``inference_optimizer/paths.py::DEFAULT_SESSION_DIR`` so a bare-image run
# without ``$USER_DATA_PATH`` lands in the same place the orchestrator does.
DEFAULT_WORKSPACE_ROOT = "/workspace/hyperloom"
ENV_USER_DATA_PATH = "USER_DATA_PATH"

log = logging.getLogger(__name__)

# One-shot guard: workspace_root() is called from argparse defaults and
# hot output-path helpers, so we only want the misconfiguration warning to
# fire once per process.
_WARNED_NO_USER_DATA = False


def workspace_root() -> str:
    """Return ``$USER_DATA_PATH`` if set, else ``DEFAULT_WORKSPACE_ROOT``.

    Mirrors ``inference_optimizer.paths.workspace_root()`` for the
    standalone kernel-agent tools. Emits exactly one ``logging.warning``
    when ``$USER_DATA_PATH`` is unset/empty so an operator immediately
    sees that artefacts are going to the pod-local default rather than
    their chosen workspace root.
    """
    global _WARNED_NO_USER_DATA
    user_data = os.environ.get(ENV_USER_DATA_PATH)
    if user_data:
        return user_data
    if not _WARNED_NO_USER_DATA:
        _WARNED_NO_USER_DATA = True
        log.warning(
            "%s is not set; falling back to %s. kernel-agent tool outputs "
            "will be written there, NOT to an operator-chosen location. "
            "Export %s to the intended workspace root before launching to "
            "avoid silently writing to the pod-local default.",
            ENV_USER_DATA_PATH,
            DEFAULT_WORKSPACE_ROOT,
            ENV_USER_DATA_PATH,
        )
    return DEFAULT_WORKSPACE_ROOT


__all__ = ["DEFAULT_WORKSPACE_ROOT", "ENV_USER_DATA_PATH", "workspace_root"]
