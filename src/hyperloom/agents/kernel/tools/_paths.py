# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
ENV_USER_DATA_PATH = "USER_DATA_PATH"

log = logging.getLogger(__name__)

# Fires the misconfiguration warning once per process.
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
