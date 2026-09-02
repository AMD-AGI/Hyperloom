# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR Monitor client stub.

Retains only the ``enabled`` flag and connection metadata used by
KnowledgePlane to gate ``mcp__pr_monitor__*`` tools in the specialist
whitelist. All REST-level PR fetching has been removed; specialists query
the PR Monitor directly via MCP.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PRMonitorClient:
    """Minimal PR Monitor stub.

    Retains only the ``enabled`` flag so KnowledgePlane can gate the
    ``mcp__pr_monitor__*`` specialist tool group and the ``--degraded-pr``
    CLI flag can disable it. All REST fetching is removed; specialists query
    via MCP directly.
    """

    enabled: bool = True

    @classmethod
    def from_args(cls, *, enabled: bool = True) -> "PRMonitorClient":
        """Build a client from the enablement flag.

        Args:
            enabled (bool): Whether the MCP tool group is enabled.

        Returns:
            PRMonitorClient: The configured client instance.
        """
        return cls(enabled=bool(enabled))


__all__ = [
    "PRMonitorClient",
]
