# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The node name this machine reports, one rule for every backend."""

from __future__ import annotations

import os
import socket

__all__ = ["node_name"]


def node_name() -> str:
    """The node this process runs on.

    Returns:
        str: ``NODE_NAME`` from the environment when set and non-empty (the
        scheduler's node name), otherwise the machine's hostname. Isolation lists,
        same-node checks and dispersion priors compare this value, so every
        backend must use this function.
    """
    return os.environ.get("NODE_NAME") or socket.gethostname()
