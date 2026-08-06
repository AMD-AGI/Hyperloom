# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 KnowledgePlane facade for Recipe KB and KernelForge control-plane."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import KnowledgeConfig, KnowledgeReadResult, KnowledgeStoreMode, KnowledgeWriteResult
from .kernel_experience_bridge import KernelExperienceBridge

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
    recipe_kb: Any = None
    config: KnowledgeConfig | None = None
    kernel_experience: KernelExperienceBridge | None = None

    @classmethod
    def from_clients(
        cls,
        *,
        pr_monitor: PRMonitorClient | None = None,
        pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL,
        recipe_kb: Any = None,
        config: KnowledgeConfig | None = None,
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
        return cls(
            pr_monitor=pr_monitor,
            pr_monitor_mcp_url=(pr_monitor_mcp_url or DEFAULT_PR_MONITOR_MCP_URL).strip(),
            recipe_kb=recipe_kb,
            config=resolved,
            kernel_experience=KernelExperienceBridge(resolved),
        )

    @property
    def status(self) -> dict[str, Any]:
        """Return secret-free Recipe, graph, and KernelForge status."""

        config = self.config or KnowledgeConfig.from_env()
        graph_status = {
            "mode": config.mode.value,
            "backend": "local-filesystem" if config.mode is KnowledgeStoreMode.LOCAL else "gbrain",
            "root": (
                str(Path(config.local_root) / "hyperloom" / "kg")
                if config.mode is KnowledgeStoreMode.LOCAL
                else ""
            ),
            "remote_configured": config.mode is KnowledgeStoreMode.REMOTE,
        }
        return {
            "recipe": {**config.public_dict(), "enabled": self.recipe_kb is not None},
            "kg": graph_status,
            "kernel_experience": (
                self.kernel_experience.status.to_dict()
                if self.kernel_experience is not None
                else {"status": "unconfigured"}
            ),
        }

    def read_recipe(self, *, canonical_id: str, **kwargs: Any) -> KnowledgeReadResult[dict[str, Any]]:
        """Typed Recipe read while retaining RecipeKB's dict API."""

        config = self.config or KnowledgeConfig.from_env()
        provenance = {"component": "recipe_kb", "canonical_id": canonical_id}
        try:
            value = (
                self.recipe_kb.get_recipe(canonical_id=canonical_id, **kwargs)
                if self.recipe_kb is not None
                else None
            )
            return KnowledgeReadResult(
                value=value,
                mode=config.mode,
                backend=config.backend,
                hit=value is not None,
                provenance=provenance,
                error="" if self.recipe_kb is not None else "recipe knowledge disabled",
            )
        except Exception as exc:  # noqa: BLE001 - typed API reports failure
            return KnowledgeReadResult(
                value=None,
                mode=config.mode,
                backend=config.backend,
                hit=False,
                provenance=provenance,
                error=f"{type(exc).__name__}: {exc}",
            )

    def write_recipe(self, **kwargs: Any) -> KnowledgeWriteResult[dict[str, Any]]:
        """Typed Recipe write with observable failure."""

        config = self.config or KnowledgeConfig.from_env()
        provenance = {
            "component": "recipe_kb",
            "canonical_id": str(kwargs.get("canonical_id") or ""),
        }
        try:
            if self.recipe_kb is None:
                raise RuntimeError("recipe knowledge disabled")
            value = self.recipe_kb.put_recipe(**kwargs)
            return KnowledgeWriteResult(
                value=value,
                mode=config.mode,
                backend=config.backend,
                success=True,
                provenance=provenance,
            )
        except Exception as exc:  # noqa: BLE001 - failure is returned, not hidden
            return KnowledgeWriteResult(
                value=None,
                mode=config.mode,
                backend=config.backend,
                success=False,
                provenance=provenance,
                error=f"{type(exc).__name__}: {exc}",
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
    "KnowledgeConfig",
    "KnowledgePlane",
]
