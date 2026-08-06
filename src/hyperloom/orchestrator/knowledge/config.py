# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared Phase 1 KnowledgePlane configuration contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Mapping, MutableMapping, TypeVar


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
    gbrain_base_url: str = ""
    gbrain_token: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "KnowledgeConfig":
        """Resolve and strictly validate the shared environment contract."""

        source = os.environ if env is None else env
        raw_mode = str(source.get("KNOWLEDGE_STORE_MODE") or "local").strip()
        try:
            mode = KnowledgeStoreMode(raw_mode)
        except ValueError as exc:
            raise ValueError(
                f"invalid KNOWLEDGE_STORE_MODE={raw_mode!r}; expected 'local' or 'remote'"
            ) from exc

        explicit_root = source.get("KNOWLEDGE_LOCAL_ROOT")
        local_root = (
            _expanded(str(explicit_root).strip())
            if explicit_root not in (None, "")
            else _default_local_root(source)
        )
        base_url = str(source.get("GBRAIN_BASE_URL") or "").strip()
        token = str(source.get("GBRAIN_TOKEN") or "").strip()
        if mode is KnowledgeStoreMode.REMOTE:
            missing = [
                name
                for name, value in (("GBRAIN_BASE_URL", base_url), ("GBRAIN_TOKEN", token))
                if not value
            ]
            if missing:
                raise ValueError(
                    "KNOWLEDGE_STORE_MODE=remote requires " + " and ".join(missing)
                )
        return cls(
            mode=mode,
            local_root=local_root,
            gbrain_base_url=base_url if mode is KnowledgeStoreMode.REMOTE else "",
            gbrain_token=token if mode is KnowledgeStoreMode.REMOTE else "",
        )

    @property
    def backend(self) -> str:
        """Stable audit backend label."""

        return "gbrain" if self.mode is KnowledgeStoreMode.REMOTE else "local-json"

    def apply_to_child_env(self, env: MutableMapping[str, str]) -> None:
        """Apply the exact shared contract to a KernelForge child environment."""

        env["KNOWLEDGE_STORE_MODE"] = self.mode.value
        env["KNOWLEDGE_LOCAL_ROOT"] = self.local_root
        # This is derived state. Never honor an operator-provided value.
        env["KERNELFORGE_GBRAIN_ENABLED"] = (
            "true" if self.mode is KnowledgeStoreMode.REMOTE else "false"
        )
        if self.mode is KnowledgeStoreMode.REMOTE:
            env["GBRAIN_BASE_URL"] = self.gbrain_base_url
            env["GBRAIN_TOKEN"] = self.gbrain_token
        else:
            env.pop("GBRAIN_BASE_URL", None)
            env.pop("GBRAIN_TOKEN", None)

    def public_dict(self) -> dict[str, Any]:
        """Return secret-free configuration suitable for status/audit output."""

        return {
            "mode": self.mode.value,
            "backend": self.backend,
            "local_root": self.local_root,
            "remote_configured": self.mode is KnowledgeStoreMode.REMOTE,
        }


T = TypeVar("T")


@dataclass(frozen=True)
class KnowledgeReadResult(Generic[T]):
    """Typed read result with backend and provenance."""

    value: T | None
    mode: KnowledgeStoreMode
    backend: str
    hit: bool
    provenance: Mapping[str, Any]
    error: str = ""


@dataclass(frozen=True)
class KnowledgeWriteResult(Generic[T]):
    """Typed write result with an observable success/failure outcome."""

    value: T | None
    mode: KnowledgeStoreMode
    backend: str
    success: bool
    provenance: Mapping[str, Any]
    error: str = ""


__all__ = [
    "KnowledgeConfig",
    "KnowledgeReadResult",
    "KnowledgeStoreMode",
    "KnowledgeWriteResult",
]
