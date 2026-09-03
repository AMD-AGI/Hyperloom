# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared Phase 1 KnowledgePlane configuration contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from hyperloom.common.pr_monitor_urls import (
    PR_MONITOR_ENABLED_ENV,
    kb_store_url as resolve_kb_service_url,
    pr_monitor_enabled as resolve_pr_monitor_enabled,
)


class KnowledgeStoreMode(str, Enum):
    """The selected knowledge backend."""

    LOCAL = "local"
    REMOTE = "remote"


def _expanded(value: str) -> str:
    """Expand a user-home prefix without changing relative-path semantics."""

    return str(Path(value).expanduser())


def _default_local_root(env: Mapping[str, str]) -> str:
    compatibility_root = str(env.get("HYPERLOOM_LOCAL_KB_ROOT") or "").strip()
    if compatibility_root:
        return _expanded(compatibility_root)

    user_data_path = str(env.get("USER_DATA_PATH") or "")
    if user_data_path:
        return str(Path(user_data_path).expanduser() / "knowledge")
    return str(Path("~/.cache/hyperloom/knowledge").expanduser())


@dataclass(frozen=True)
class KnowledgeConfig:
    """Validated shared configuration consumed by Hyperloom and KernelForge."""

    mode: KnowledgeStoreMode
    local_root: str
    kb_store_url: str = ""
    kb_store_token: str = ""
    pr_monitor_enabled: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "KnowledgeConfig":
        """Resolve and strictly validate the shared environment contract."""

        source = os.environ if env is None else env
        raw_mode = str(source.get("KNOWLEDGE_STORE_MODE") or "local").strip()
        try:
            mode = KnowledgeStoreMode(raw_mode)
        except ValueError as exc:
            raise ValueError(f"invalid KNOWLEDGE_STORE_MODE={raw_mode!r}; expected 'local' or 'remote'") from exc

        explicit_root = source.get("KNOWLEDGE_LOCAL_ROOT")
        local_root = (
            _expanded(str(explicit_root).strip()) if explicit_root not in (None, "") else _default_local_root(source)
        )
        kb_store_url = str(source.get("KB_STORE_URL") or "").strip()
        kb_store_token = str(source.get("KB_STORE_TOKEN") or "").strip()
        if mode is KnowledgeStoreMode.REMOTE:
            missing = [
                name
                for name, value in (
                    ("KB_STORE_URL", kb_store_url),
                    ("KB_STORE_TOKEN", kb_store_token),
                )
                if not value
            ]
            if missing:
                raise ValueError("KNOWLEDGE_STORE_MODE=remote requires " + " and ".join(missing))
        return cls(
            mode=mode,
            local_root=local_root,
            # The URL also hosts PR Monitor and is therefore useful in local
            # Recipe mode; only the Recipe bearer token is mode-specific.
            kb_store_url=(kb_store_url if mode is KnowledgeStoreMode.REMOTE else resolve_kb_service_url(env=source)),
            kb_store_token=kb_store_token if mode is KnowledgeStoreMode.REMOTE else "",
            pr_monitor_enabled=resolve_pr_monitor_enabled(source),
        )

    @property
    def backend(self) -> str:
        """Stable audit backend label."""

        return "kb-store" if self.mode is KnowledgeStoreMode.REMOTE else "local-json"

    def apply_to_child_env(self, env: MutableMapping[str, str]) -> None:
        """Apply the exact shared contract to a KernelForge child environment."""

        env["KNOWLEDGE_STORE_MODE"] = self.mode.value
        env["KNOWLEDGE_LOCAL_ROOT"] = self.local_root
        env[PR_MONITOR_ENABLED_ENV] = "1" if self.pr_monitor_enabled else "0"
        if self.mode is KnowledgeStoreMode.REMOTE:
            env["KB_STORE_URL"] = self.kb_store_url
            env["KB_STORE_TOKEN"] = self.kb_store_token
        else:
            if self.kb_store_url and self.pr_monitor_enabled:
                env["KB_STORE_URL"] = self.kb_store_url
            else:
                env.pop("KB_STORE_URL", None)
            env.pop("KB_STORE_TOKEN", None)
        # GBrain credentials are used by the Framework PR client but must
        # never cross into the KernelForge child.
        env.pop("GBRAIN_BASE_URL", None)
        env.pop("GBRAIN_TOKEN", None)
        env["KERNELFORGE_GBRAIN_ENABLED"] = "false"
        # Section drafts are owned by the parent inference Recipe publisher.
        env.pop("KB_DRAFT_DIR", None)
        env.pop("KB_WARM_START_DIR", None)

    def public_dict(self) -> dict[str, Any]:
        """Return secret-free configuration suitable for status/audit output."""

        return {
            "mode": self.mode.value,
            "backend": self.backend,
            "local_root": self.local_root,
            "remote_configured": self.mode is KnowledgeStoreMode.REMOTE,
        }


__all__ = [
    "KnowledgeConfig",
    "KnowledgeStoreMode",
]
