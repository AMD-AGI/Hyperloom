# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR Monitor client stub.

Retains only the ``enabled`` flag and connection metadata used by
KnowledgePlane to gate ``mcp__pr_monitor__*`` tools in the specialist
whitelist. All REST-level PR fetching has been removed; specialists query
the PR Monitor directly via MCP.
"""

from __future__ import annotations

from dataclasses import dataclass


# MCP URL passed to specialist LLM backend. Empty means PR Monitor MCP is not
# advertised unless the operator explicitly configures --pr-monitor-mcp-url.
DEFAULT_PR_MONITOR_MCP_URL: str = ""


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
    def from_args(
        cls,
        *,
        url: str | None = None,
        enabled: bool = True,
        timeout_sec: float | None = None,
    ) -> "PRMonitorClient":
        """Build a client, resolving config from args then env vars.

        Args:
            url (str | None): Accepted for CLI compatibility; REST access was removed.
            enabled (bool): Whether the MCP tool group is enabled.
            timeout_sec (float | None): Ignored (kept for call-site compat).

        Returns:
            PRMonitorClient: The configured client instance.
        """
        del url, timeout_sec
        return cls(enabled=bool(enabled))


__all__ = [
    "DEFAULT_PR_MONITOR_MCP_URL",
    "PRMonitorClient",
]
