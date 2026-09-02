# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 KnowledgePlane facade for Recipe KB and KernelForge control-plane."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from .config import KnowledgeConfig, KnowledgeStoreMode
from .kernel_experience_bridge import KernelExperienceBridge

from .pr_monitor import PRMonitorClient


log = logging.getLogger(__name__)


@dataclass
class KnowledgePlane:
    """Single facade for the knowledge sources.

    Lives for one Coordinator run. Tracks whether the PR Monitor MCP is wired
    so the specialist runner can gate the corresponding tool group.
    """

    pr_monitor: PRMonitorClient | None = None
    pr_monitor_mcp_url: str = ""
    recipe_kb: Any = None
    config: KnowledgeConfig | None = None
    kernel_experience: KernelExperienceBridge | None = None
    kb_disabled: bool = False

    @classmethod
    def from_clients(
        cls,
        *,
        pr_monitor: PRMonitorClient | None = None,
        pr_monitor_mcp_url: str = "",
        recipe_kb: Any = None,
        config: KnowledgeConfig | None = None,
        kb_disabled: bool = False,
    ) -> "KnowledgePlane":
        """Construct a plane from injected clients and config.

        Args:
            pr_monitor (PRMonitorClient | None): Optional PR Monitor client
                (used only to check ``enabled`` for tool whitelisting).
            pr_monitor_mcp_url (str): MCP URL advertised to specialists.

        Returns:
            KnowledgePlane: The constructed facade.
        """
        resolved = config or KnowledgeConfig.from_env()
        resolved = replace(
            resolved,
            pr_monitor_enabled=bool(pr_monitor is not None and pr_monitor.enabled),
        )
        return cls(
            pr_monitor=pr_monitor,
            pr_monitor_mcp_url=(pr_monitor_mcp_url or "").strip(),
            recipe_kb=recipe_kb,
            config=resolved,
            kernel_experience=(None if kb_disabled else KernelExperienceBridge(resolved)),
            kb_disabled=kb_disabled,
        )

    @property
    def status(self) -> dict[str, Any]:
        """Return secret-free Recipe and KernelForge status."""

        config = self.config or KnowledgeConfig.from_env()
        return {
            "recipe": {
                **config.public_dict(),
                "enabled": (
                    not self.kb_disabled and (self.recipe_kb is not None or config.mode is KnowledgeStoreMode.REMOTE)
                ),
                "read_enabled": (not self.kb_disabled and self.recipe_kb is not None),
                "disabled_reason": ("degraded_kb" if self.kb_disabled else ""),
            },
            "kernel_experience": (
                self.kernel_experience.status.to_dict()
                if self.kernel_experience is not None
                else {"status": "unconfigured"}
            ),
        }

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
    "KnowledgeConfig",
    "KnowledgePlane",
]
