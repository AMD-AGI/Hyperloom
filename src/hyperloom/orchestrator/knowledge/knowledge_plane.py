# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KnowledgePlane facade for advisory knowledge inputs.

Coordinator-side glue exposing the PR Monitor MCP URL for specialist tool
whitelisting. Stateless. This facade only gates whether the PR Monitor MCP
server is advertised in the specialist tool whitelist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .pr_monitor import (
    DEFAULT_PR_MONITOR_MCP_URL,
    PRMonitorClient,
)


log = logging.getLogger(__name__)


@dataclass
class KnowledgePlane:
    """Single facade for the knowledge sources.

    Lives for one Coordinator run. Tracks whether the PR Monitor MCP is wired
    so the specialist runner can gate the corresponding tool group.
    """

    pr_monitor: PRMonitorClient | None = None
    pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL

    @classmethod
    def from_clients(
        cls,
        *,
        pr_monitor: PRMonitorClient | None = None,
        pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL,
    ) -> "KnowledgePlane":
        """Construct a plane from injected clients and config.

        Args:
            pr_monitor (PRMonitorClient | None): Optional PR Monitor client
                (used only to check ``enabled`` for tool whitelisting).
            pr_monitor_mcp_url (str): MCP URL advertised to specialists.

        Returns:
            KnowledgePlane: The constructed facade.
        """
        return cls(
            pr_monitor=pr_monitor,
            pr_monitor_mcp_url=(pr_monitor_mcp_url or DEFAULT_PR_MONITOR_MCP_URL).strip(),
        )

    def reset_round_caches(self) -> None:
        """No-op at EXPLORE round boundaries."""

    @property
    def pr_monitor_enabled(self) -> bool:
        """Whether the PR Monitor client is wired and enabled.

        Returns:
            bool: ``True`` when a PR Monitor client is present and enabled.
        """
        return self.pr_monitor is not None and self.pr_monitor.enabled

    def specialist_mcp_url(self) -> str:
        """MCP URL to advertise in the specialist tool whitelist.

        Returns ``""`` when PR Monitor is disabled so the runner can elide
        the ``mcp__pr_monitor__*`` tool block.

        Returns:
            The PR Monitor MCP URL, or ``""`` when PR Monitor is disabled.
        """
        if not self.pr_monitor_enabled:
            return ""
        return self.pr_monitor_mcp_url


__all__ = [
    "KnowledgePlane",
]
