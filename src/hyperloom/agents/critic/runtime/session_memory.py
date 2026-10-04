# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-session memory store for the Critic agent."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.io import append_jsonl as _common_append_jsonl
from hyperloom.common.io import atomic_write_json as _common_atomic_write_json
from hyperloom.common.jsonio import read_json as _common_read_json
from hyperloom.common.timeutil import now_iso

from .errors import SessionMemoryError


DEFAULT_SESSION_MEMORY_DIR = "/var/lib/critic-session-memory"
DEFAULT_PRIOR_CACHE_TTL_SECONDS = 3600

# Keys whose stored value can be filled in for the next request.
_MERGEABLE_CONTEXT_KEYS: tuple[str, ...] = (
    "model",
    "framework",
    "model_family",
    "workload",
    "precision",
    "scale",
    "objective",
    "baseline_tput",
    "baseline_label",
    "current_best",
    "session_label",
)

_MISSING_VALUES: frozenset[str] = frozenset({"", "unknown", "null", "none"})


def _is_missing(value: Any) -> bool:
    """Return ``True`` if value should be treated as absent."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in _MISSING_VALUES:
        return True
    return False


# ---------------------------------------------------------------------------
@dataclass
class MergeResult:
    """Result of merging an incoming context against stored memory."""

    merged: dict[str, Any] = field(default_factory=dict)
    explicit_keys: list[str] = field(default_factory=list)
    from_memory_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of this merge result."""
        return {
            "merged": dict(self.merged),
            "explicit_keys": list(self.explicit_keys),
            "from_memory_keys": list(self.from_memory_keys),
            "missing_keys": list(self.missing_keys),
        }


# ---------------------------------------------------------------------------
class SessionMemory:
    """File-backed session memory."""

    def __init__(self, root: str | Path | None = None):
        """Initialise the store rooted at ``root``."""
        if root is None:
            root = os.environ.get("CRITIC_SESSION_MEMORY_DIR", DEFAULT_SESSION_MEMORY_DIR)
        self.root = Path(root)
        self.prior_cache_ttl = float(
            os.environ.get(
                "CRITIC_PRIOR_CACHE_TTL_SECONDS",
                str(DEFAULT_PRIOR_CACHE_TTL_SECONDS),
            )
        )

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

    def _context_path(self, session_id: str) -> Path:
        """Return the path to the session's ``context.json``."""
        return self.session_dir(session_id) / "context.json"

    def _decisions_path(self, session_id: str) -> Path:
        """Return the path to the session's ``decisions.jsonl``."""
        return self.session_dir(session_id) / "decisions.jsonl"

    def _priors_cache_path(self, session_id: str) -> Path:
        """Return the path to the session's ``kb_priors_cache.json``."""
        return self.session_dir(session_id) / "kb_priors_cache.json"

    def _reviewed_path(self, session_id: str) -> Path:
        """Return the path to the session's ``reviewed_msg_ids.json``."""
        return self.session_dir(session_id) / "reviewed_msg_ids.json"

    # Context
    def load_context(self, session_id: str) -> dict[str, Any]:
        """Load the stored context for a session."""
        path = self._context_path(session_id)
        return _read_object(path, default={})

    def save_context(self, session_id: str, context: dict[str, Any]) -> None:
        """Persist ``context`` as the session's full context."""
        if not isinstance(context, dict):
            raise SessionMemoryError(f"context must be a dict, got {type(context).__name__}")
        self._ensure_session_dir(session_id)
        _common_atomic_write_json(
            self._context_path(session_id),
            context,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            make_parents=False,
        )

    def merge_context(
        self,
        session_id: str,
        incoming: dict[str, Any],
        *,
        persist: bool = True,
    ) -> MergeResult:
        """Merge ``incoming`` against stored context with explicit-wins semantics."""
        if not isinstance(incoming, dict):
            raise SessionMemoryError(f"incoming context must be a dict, got {type(incoming).__name__}")
        stored = self.load_context(session_id)
        merged: dict[str, Any] = dict(stored)
        explicit: list[str] = []
        from_memory: list[str] = []
        for key, value in incoming.items():
            if _is_missing(value):
                continue
            merged[key] = value
            explicit.append(key)
        for key in _MERGEABLE_CONTEXT_KEYS:
            if key in explicit:
                continue
            if key in stored and not _is_missing(stored.get(key)):
                merged[key] = stored[key]
                from_memory.append(key)

        missing: list[str] = []
        for key in _MERGEABLE_CONTEXT_KEYS:
            if _is_missing(merged.get(key)):
                missing.append(key)

        result = MergeResult(
            merged=merged,
            explicit_keys=explicit,
            from_memory_keys=from_memory,
            missing_keys=missing,
        )
        if persist:
            self.save_context(session_id, merged)
        return result

    # Decisions
    def append_decision(self, session_id: str, decision_review: dict[str, Any]) -> None:
        """Append a decision review record to the session's decisions log."""
        if not isinstance(decision_review, dict):
            raise SessionMemoryError("decision_review must be a dict")
        self._ensure_session_dir(session_id)
        record = {
            "ts": now_iso(timespec="microseconds"),
            "decision_review": decision_review,
        }
        _common_append_jsonl(self._decisions_path(session_id), record, ensure_ascii=False)

    # KB priors cache (per-scope+topic)
    def get_cached_priors(
        self,
        session_id: str,
        cache_key: str,
        *,
        now: float | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return cached KB priors for a key if present and not expired."""
        cache = _read_object(self._priors_cache_path(session_id), default={})
        entry = cache.get(cache_key)
        if not isinstance(entry, dict):
            return None
        ts = entry.get("ts")
        priors = entry.get("priors")
        if not isinstance(priors, list) or not isinstance(ts, (int, float)):
            return None
        if (now or time.time()) - float(ts) > self.prior_cache_ttl:
            return None
        return priors

    def put_cached_priors(
        self,
        session_id: str,
        cache_key: str,
        priors: list[dict[str, Any]],
    ) -> None:
        """Store KB priors under ``cache_key`` with the current timestamp."""
        if not isinstance(priors, list):
            raise SessionMemoryError("priors must be a list")
        self._ensure_session_dir(session_id)
        path = self._priors_cache_path(session_id)
        cache = _read_object(path, default={})
        cache[cache_key] = {"ts": time.time(), "priors": list(priors)}
        _common_atomic_write_json(path, cache, ensure_ascii=False, indent=2, sort_keys=False, make_parents=False)

    # Already-reviewed proposals
    def mark_reviewed(
        self,
        session_id: str,
        msg_id: str,
        verdict: str,
        *,
        decision_id: str | None = None,
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
            "decision_id": decision_id,
        }
        _common_atomic_write_json(path, data, ensure_ascii=False, indent=2, sort_keys=False, make_parents=False)

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
        return _common_read_json(path, require_dict=True, strict=True, empty_value=default)
    except (OSError, ValueError) as exc:
        raise SessionMemoryError(f"corrupt session memory file at {path}: {exc}") from exc


__all__ = [
    "DEFAULT_PRIOR_CACHE_TTL_SECONDS",
    "DEFAULT_SESSION_MEMORY_DIR",
    "MergeResult",
    "SessionMemory",
]
