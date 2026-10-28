# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 KnowledgePlane facade for Recipe KB and PR Monitor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import KnowledgeConfig


log = logging.getLogger(__name__)


@dataclass
class KnowledgePlane:
    """Single facade for the knowledge sources."""

    pr_monitor_enabled: bool = False
    pr_monitor_mcp_url: str = ""
    recipe_kb: Any = None
    config: KnowledgeConfig | None = None
    kb_disabled: bool = False

    @classmethod
    def from_clients(
        cls,
        *,
        pr_monitor_enabled: bool = False,
        pr_monitor_mcp_url: str = "",
        recipe_kb: Any = None,
        config: KnowledgeConfig | None = None,
        kb_disabled: bool = False,
    ) -> "KnowledgePlane":
        """Construct a plane from injected clients and config."""
        return cls(
            pr_monitor_enabled=pr_monitor_enabled,
            pr_monitor_mcp_url=(pr_monitor_mcp_url or "").strip(),
            recipe_kb=recipe_kb,
            config=config or KnowledgeConfig.from_env(),
            kb_disabled=kb_disabled,
        )

    def specialist_mcp_url(self) -> str:
        """MCP URL to advertise in the specialist tool whitelist."""
        if not self.pr_monitor_enabled:
            return ""
        return self.pr_monitor_mcp_url


__all__ = [
    "KnowledgeConfig",
    "KnowledgePlane",
]
