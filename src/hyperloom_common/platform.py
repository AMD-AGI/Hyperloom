# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One place that names the host OS, so platform routing stays consistent.

Linux keeps the native paths (Magpie serving scripts, the Ray execution
backend). Windows 11 takes the tailored alternatives (the pure-Python bypass
benchmark backend, the local-subprocess executor) because the native paths
need POSIX process groups, ``fork``/``setsid`` shells and ``.sh`` scripts.
An explicit env choice always wins over the OS default; this helper only
reports the host, it never reads configuration.
"""

from __future__ import annotations

import os


def is_windows() -> bool:
    """Whether this host is native Windows (Win32), as opposed to POSIX."""
    return os.name == "nt"
