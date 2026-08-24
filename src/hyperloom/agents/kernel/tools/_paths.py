# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workspace-root resolver for the standalone kernel-agent tools.

Stdlib-only mirror of ``hyperloom.inference_optimizer.session.paths.workspace_root``,
kept independent so ``tools/`` scripts run standalone on remote nodes without a
``hyperloom`` import. Do not import ``hyperloom.common`` or
``hyperloom.inference_optimizer.session.paths`` here.
"""

from __future__ import annotations

import logging
import os

DEFAULT_WORKSPACE_ROOT = "/workspace/hyperloom"
POD_LOCAL_WORKSPACE = "/workspace"
ENV_USER_DATA_PATH = "USER_DATA_PATH"

log = logging.getLogger(__name__)

# Fires the misconfiguration warning once per process.
_WARNED_NO_USER_DATA = False


def default_workspace_root() -> str:
    """Workspace to use when ``$USER_DATA_PATH`` is unset.

    Mirrors ``session.paths.default_workspace_root``, which this module cannot
    import: container images ship a writable ``/workspace``, a bare-metal
    non-root host has neither it nor permission to create it.
    """
    probe = POD_LOCAL_WORKSPACE
    while not os.path.exists(probe) and probe != os.path.dirname(probe):
        probe = os.path.dirname(probe)
    if os.access(probe, os.W_OK):
        return DEFAULT_WORKSPACE_ROOT
    return os.path.join(os.getcwd(), "session")


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
            "avoid silently writing to the default.",
            ENV_USER_DATA_PATH,
            default_workspace_root(),
            ENV_USER_DATA_PATH,
        )
    return default_workspace_root()


__all__ = ["DEFAULT_WORKSPACE_ROOT", "ENV_USER_DATA_PATH", "default_workspace_root", "workspace_root"]
