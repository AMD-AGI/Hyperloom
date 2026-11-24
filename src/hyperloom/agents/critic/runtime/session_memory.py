# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-session memory store for the Critic agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from hyperloom_common.io import JsonFormat, WriteOptions
from hyperloom_common.io import atomic_write_json as _common_atomic_write_json
from hyperloom_common.jsonio import read_json_strict as _common_read_json_strict
from hyperloom_common.timeutil import now_iso

from .errors import SessionMemoryError


DEFAULT_SESSION_MEMORY_DIR = "/var/lib/critic-session-memory"


# ---------------------------------------------------------------------------
class SessionMemory:
    """File-backed session memory."""

    def __init__(self, root: str | Path | None = None):
        """Initialise the store rooted at ``root``."""
        if root is None:
            root = os.environ.get("CRITIC_SESSION_MEMORY_DIR", DEFAULT_SESSION_MEMORY_DIR)
        self.root = Path(root)

    # Path helpers
    def session_dir(self, session_id: str) -> Path:
        """Return the directory for ``session_id`` under the store root."""
        if not session_id or not isinstance(session_id, str):
            raise SessionMemoryError(f"invalid session_id: {session_id!r}")
        # Disallow path traversal.
        if "/" in session_id or ".." in session_id:
            raise SessionMemoryError(f"session_id must not contain slashes: {session_id!r}")
        return self.root / session_id

    def _ensure_session_dir(self, session_id: str) -> Path:
        """Create the session directory if needed and return it."""
        d = self.session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _reviewed_path(self, session_id: str) -> Path:
        """Return the path to the session's ``reviewed_msg_ids.json``."""
        return self.session_dir(session_id) / "reviewed_msg_ids.json"

    # Already-reviewed proposals
    def mark_reviewed(
        self,
        session_id: str,
        msg_id: str,
        verdict: str,
    ) -> None:
        """Record that a proposal message was reviewed with a verdict."""
        if not msg_id or not verdict:
            raise SessionMemoryError("msg_id and verdict are required")
        self._ensure_session_dir(session_id)
        path = self._reviewed_path(session_id)
        data = _read_object(path, default={})
        data[msg_id] = {
            "verdict": verdict,
            "ts": now_iso(timespec="microseconds"),
        }
        _common_atomic_write_json(
            path, data, fmt=JsonFormat(sort_keys=False, ensure_ascii=False), options=WriteOptions()
        )

    def filter_unreviewed(
        self,
        session_id: str,
        msg_ids: Iterable[str],
    ) -> list[str]:
        """Return the subset of ``msg_ids`` not yet reviewed this session."""
        data = _read_object(self._reviewed_path(session_id), default={})
        return [m for m in msg_ids if m not in data]


# Shared JSON helper
def _read_object(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    """Read a JSON object from ``path``."""
    if not path.exists():
        return default
    try:
        return _common_read_json_strict(path, require_dict=True, empty_value=default)
    except (OSError, ValueError) as exc:
        raise SessionMemoryError(f"corrupt session memory file at {path}: {exc}") from exc


__all__ = [
    "DEFAULT_SESSION_MEMORY_DIR",
    "SessionMemory",
]
