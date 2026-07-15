# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KnowledgePlane facade for advisory knowledge inputs.

Coordinator-side glue exposing the PR Monitor MCP URL for specialist tool
whitelisting and the Cortex KB MCP for optional graph-query tools. Stateless.
This facade only gates whether those MCP servers are advertised in the
specialist tool whitelist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .pr_monitor import (
    DEFAULT_PR_MONITOR_MCP_URL,
    PRMonitorClient,
)


log = logging.getLogger(__name__)


@dataclass
class KnowledgePlane:
    """Single facade for the knowledge sources.

    Lives for one Coordinator run. Tracks whether the PR Monitor MCP and
    Cortex KB MCP are wired so the specialist runner can gate the
    corresponding tool groups.
    """

    pr_monitor: PRMonitorClient | None = None
    pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL
    # Read-only KB-graph (gbrain) MCP advertised as the ``cortex_kb`` server;
    # empty strips the mcp__cortex_kb__* tools from the specialist whitelist.
    cortex_kb_mcp_url: str = ""
    cortex_kb_mcp_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_clients(
        cls,
        *,
        pr_monitor: PRMonitorClient | None = None,
        pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL,
        cortex_kb_mcp_url: str = "",
        cortex_kb_mcp_headers: dict[str, str] | None = None,
    ) -> "KnowledgePlane":
        """Construct a plane from injected clients and config.

        Args:
            pr_monitor (PRMonitorClient | None): Optional PR Monitor client
                (used only to check ``enabled`` for tool whitelisting).
            pr_monitor_mcp_url (str): MCP URL advertised to specialists.
            cortex_kb_mcp_url (str): Read-only KB-graph MCP URL advertised to
                specialists as the ``cortex_kb`` server; empty disables it.
            cortex_kb_mcp_headers (dict[str, str] | None): Optional auth headers
                (e.g. ``Authorization: Bearer …``) for the KB-graph MCP server.

        Returns:
            KnowledgePlane: The constructed facade.
        """
        return cls(
            pr_monitor=pr_monitor,
            pr_monitor_mcp_url=(pr_monitor_mcp_url or DEFAULT_PR_MONITOR_MCP_URL).strip(),
            cortex_kb_mcp_url=(cortex_kb_mcp_url or "").strip(),
            cortex_kb_mcp_headers=dict(cortex_kb_mcp_headers or {}),
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

    @property
    def cortex_enabled(self) -> bool:
        """Whether the read-only ``cortex_kb`` KB-graph MCP is wired.

        Returns:
            bool: ``True`` when a ``cortex_kb_mcp_url`` is configured.
        """
        return bool((self.cortex_kb_mcp_url or "").strip())

    def cortex_specialist_mcp_url(self) -> str:
        """KB-graph MCP URL to advertise as the specialist ``cortex_kb`` server.

        Returns:
            The configured URL, or ``""`` when the KB-graph MCP is disabled.
        """
        return (self.cortex_kb_mcp_url or "").strip()

    def cortex_specialist_mcp_headers(self) -> dict[str, str]:
        """Auth headers for the specialist ``cortex_kb`` MCP server.

        Returns:
            A copy of the configured header map (``{}`` when none / disabled).
        """
        if not (self.cortex_kb_mcp_url or "").strip():
            return {}
        return dict(self.cortex_kb_mcp_headers or {})

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
