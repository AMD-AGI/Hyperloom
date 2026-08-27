# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Constants and lookups the source arm shares with the framework agent.

The Coordinator no longer shells out to ``fa phase-*``: candidate discovery is
a specialist, and its verdicts arrive in the deliverable rather than from a
per-candidate audit call.
"""

from __future__ import annotations

from hyperloom.agents.framework.repo_map import repo_url_for_framework

# Consecutive empty discovery rounds tolerated before the source arm declines.
DISCOVER_FAILURE_RETRY_LIMIT: int = 3

__all__ = [
    "DISCOVER_FAILURE_RETRY_LIMIT",
    "repo_url_for_framework",
]
