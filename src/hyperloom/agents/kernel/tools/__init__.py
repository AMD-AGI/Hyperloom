# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-agent tools (TraceLens analysis, kernel optimization, patch apply).

These modules are primarily consumed as standalone CLI scripts invoked via
``HYPERLOOM_KERNEL_AGENT_ROOT`` (``python3 <root>/tools/<tool>.py --args``), and
rely on ``sys.path``-based sibling imports rather than package-relative imports
so they keep working both as a package member and as a directly-executed script.
"""

from __future__ import annotations
