# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration, protocol and factory for experience storage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class KnowledgeStoreMode(str, Enum):
    """Supported durable experience-store transports."""

    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class KnowledgeConfig:
    """Strict configuration for the KernelForge experience store."""

    mode: KnowledgeStoreMode
    local_root: Path
    kb_store_url: str = ""
    kb_store_token: str = ""

    @property
    def rewrite_root(self) -> Path:
        """Filesystem root holding local Rewrite records."""
        return self.local_root / "kernelforge" / "rewrite"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        mode: str | KnowledgeStoreMode | None = None,
        local_root: str | os.PathLike[str] | None = None,
        kb_store_url: str | None = None,
        kb_store_token: str | None = None,
    ) -> "KnowledgeConfig":
        """Parse the cross-repository environment contract with strict validation."""
        env = os.environ if environ is None else environ
        raw_mode = mode.value if isinstance(mode, KnowledgeStoreMode) else mode
        if raw_mode is None:
            raw_mode = env.get("KNOWLEDGE_STORE_MODE", KnowledgeStoreMode.LOCAL.value)
        normalized_mode = str(raw_mode).strip()
        try:
            parsed_mode = KnowledgeStoreMode(normalized_mode)
        except ValueError as exc:
            supported = ", ".join(item.value for item in KnowledgeStoreMode)
            raise ValueError(f"KNOWLEDGE_STORE_MODE must be one of: {supported}; got {raw_mode!r}") from exc

        raw_root = local_root
        if raw_root is None:
            configured_root = env.get("KNOWLEDGE_LOCAL_ROOT")
            if configured_root is not None:
                if not configured_root.strip():
                    raise ValueError("KNOWLEDGE_LOCAL_ROOT must not be empty")
                raw_root = configured_root
            else:
                user_data_path = env.get("USER_DATA_PATH", "").strip()
                raw_root = (
                    Path(user_data_path) / "knowledge" if user_data_path else Path("~/.cache/hyperloom/knowledge")
                )
        if not str(raw_root).strip():
            raise ValueError("KNOWLEDGE_LOCAL_ROOT must not be empty")
        root = Path(raw_root).expanduser()

        store_url = (env.get("KB_STORE_URL", "") if kb_store_url is None else str(kb_store_url)).strip()
        store_token = (env.get("KB_STORE_TOKEN", "") if kb_store_token is None else str(kb_store_token)).strip()
        if parsed_mode is KnowledgeStoreMode.REMOTE:
            missing = [
                name for name, value in (("KB_STORE_URL", store_url), ("KB_STORE_TOKEN", store_token)) if not value
            ]
            if missing:
                raise ValueError("KNOWLEDGE_STORE_MODE=remote requires " + " and ".join(missing))
        else:
            # Ambient credentials must never activate the network in local mode.
            store_url = ""
            store_token = ""

        return cls(
            mode=parsed_mode,
            local_root=root,
            kb_store_url=store_url,
            kb_store_token=store_token,
        )


def knowledge_config_from_runtime(config: Any) -> KnowledgeConfig:
    """Extract a validated ``KnowledgeConfig`` from the runtime config."""
    if isinstance(config, KnowledgeConfig):
        return config
    parsed = getattr(config, "knowledge_config", None)
    if isinstance(parsed, KnowledgeConfig):
        return parsed
    return KnowledgeConfig.from_env()
