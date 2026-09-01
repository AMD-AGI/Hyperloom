# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestrator's import path into the framework agent's shared tables.

Re-exports :func:`repo_url_for_framework` so the orchestrator has one place to
reach it, and owns the discovery retry budget the source arm declines on.
"""

from __future__ import annotations

from hyperloom.agents.framework.repo_map import repo_url_for_framework

# Consecutive empty discovery rounds tolerated before the source arm declines.
DISCOVER_FAILURE_RETRY_LIMIT: int = 3

__all__ = [
    "DISCOVER_FAILURE_RETRY_LIMIT",
    "repo_url_for_framework",
]
