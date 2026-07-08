# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Workspace-root resolver for the standalone kernel-agent tools.

Stdlib-only mirror of ``hyperloom.inference_optimizer.session.paths.workspace_root``.

Kept as an independent, stdlib-only mirror (not a ``hyperloom.common.paths``
re-export) because ``tools/`` scripts must run standalone on remote nodes
without a ``hyperloom`` import: they are invoked as bare
``python3 <root>/tools/<tool>.py --args`` subprocesses (see
``HYPERLOOM_KERNEL_AGENT_ROOT`` in
``hyperloom.orchestrator.kernel.request_handlers``), imported via the bare
module name ``from _paths import workspace_root`` (not a package-relative
``from ._paths import``), and some of their code paths execute inside Ray
workers (``tools/backends/``) that do not inherit the driver's ``sys.path``.
This mirrors the same, deliberate exception already made for
``_payload_aliases.py`` (see tree-reform-lessons.MD §13) — do not "finish"
this extraction by importing ``hyperloom.common`` or
``hyperloom.inference_optimizer.session.paths`` here.
"""

from __future__ import annotations

import logging
import os

# Kept byte-for-byte identical to src/hyperloom/inference_optimizer/paths.py::DEFAULT_SESSION_DIR.
DEFAULT_WORKSPACE_ROOT = "/workspace/hyperloom"
ENV_USER_DATA_PATH = "USER_DATA_PATH"

log = logging.getLogger(__name__)

# One-shot guard so the misconfiguration warning fires once per process.
_WARNED_NO_USER_DATA = False


def workspace_root() -> str:
    """Resolve the workspace root for kernel-agent tool outputs.

    Returns ``$USER_DATA_PATH`` when set; otherwise falls back to
    ``DEFAULT_WORKSPACE_ROOT`` and logs a one-time misconfiguration warning.

    Returns:
        The resolved workspace root path.
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
