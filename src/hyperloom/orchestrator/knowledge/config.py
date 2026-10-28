# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared Phase 1 KnowledgePlane configuration contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping


class KnowledgeStoreMode(str, Enum):
    """The selected knowledge backend."""

    LOCAL = "local"
    REMOTE = "remote"


def _expanded(value: str) -> str:
    """Expand a user-home prefix without changing relative-path semantics."""

    return str(Path(value).expanduser())


def _default_local_root(env: Mapping[str, str]) -> str:
    user_data_path = str(env.get("USER_DATA_PATH") or "")
    if user_data_path:
        return str(Path(user_data_path).expanduser() / "knowledge")
    return str(Path("~/.cache/hyperloom/knowledge").expanduser())


@dataclass(frozen=True)
class KnowledgeConfig:
    """Validated shared configuration consumed by Hyperloom and KernelForge."""

    mode: KnowledgeStoreMode
    local_root: str

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
        if mode is KnowledgeStoreMode.REMOTE:
            missing = [name for name in ("KB_STORE_URL", "KB_STORE_TOKEN") if not str(source.get(name) or "").strip()]
            if missing:
                raise ValueError("KNOWLEDGE_STORE_MODE=remote requires " + " and ".join(missing))
        return cls(mode=mode, local_root=local_root)


__all__ = [
    "KnowledgeConfig",
    "KnowledgeStoreMode",
]
