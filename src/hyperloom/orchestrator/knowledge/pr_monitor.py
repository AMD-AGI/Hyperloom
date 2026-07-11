# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR Monitor client stub.

Retains only the ``enabled`` flag and connection metadata used by
KnowledgePlane to gate ``mcp__pr_monitor__*`` tools in the specialist
whitelist. All REST-level PR fetching has been removed; specialists query
the PR Monitor directly via MCP.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass


log = logging.getLogger(__name__)


# Default in-cluster service URL; operator overrides via ``--pr-monitor-url``.
DEFAULT_PR_MONITOR_URL: str = "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/v1"
# MCP URL passed to specialist LLM backend; trailing slash mandatory.
DEFAULT_PR_MONITOR_MCP_URL: str = "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/mcp/"

DEFAULT_PR_MONITOR_TIMEOUT_SEC: float = 5.0


class PRMonitorError(RuntimeError):
    """Raised for unrecoverable PR Monitor interactions."""


@dataclass
class PRMonitorClient:
    """Minimal PR Monitor stub.

    Retains only the ``enabled`` flag so KnowledgePlane can gate the
    ``mcp__pr_monitor__*`` specialist tool group and the ``--degraded-pr``
    CLI flag can disable it. All REST fetching is removed; specialists query
    via MCP directly.
    """

    base_url: str = DEFAULT_PR_MONITOR_URL
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
            url (str | None): Explicit base URL; falls back to
                ``PR_MONITOR_URL`` / ``PRIMUS_CORTEX_PR_URL`` env vars
                then :data:`DEFAULT_PR_MONITOR_URL`.
            enabled (bool): Whether the MCP tool group is enabled.
            timeout_sec (float | None): Ignored (kept for call-site compat).

        Returns:
            PRMonitorClient: The configured client instance.
        """
        del timeout_sec  # no longer used
        resolved_url = (
            url
            or os.environ.get("PR_MONITOR_URL", "").strip()
            or os.environ.get("PRIMUS_CORTEX_PR_URL", "").strip()
            or DEFAULT_PR_MONITOR_URL
        ).rstrip("/")
        return cls(
            base_url=resolved_url,
            enabled=bool(enabled),
        )


__all__ = [
    "DEFAULT_PR_MONITOR_MCP_URL",
    "DEFAULT_PR_MONITOR_TIMEOUT_SEC",
    "DEFAULT_PR_MONITOR_URL",
    "PRMonitorClient",
    "PRMonitorError",
]
