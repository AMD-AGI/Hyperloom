# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-session memory store for the Critic agent.

Hosts only send full context on the first call, so Critic merges later
incremental turns against stored context, recalls already-reviewed
proposals, and caches KB priors. Constraints (handoff doc §5): local
JSON/JSONL only; per-session and not meant to outlive the session
(long-term knowledge goes to remote KB); append-only for decisions/events,
rewritten for context/priors/reviewed ids; context merge is explicit-wins
with ``""`` / ``"unknown"`` treated as missing.

Layout under ``CRITIC_SESSION_MEMORY_DIR``::

    <root>/<session_id>/
      context.json
      decisions.jsonl
      events.jsonl
      kb_priors_cache.json
      reviewed_msg_ids.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
        return {
            "merged": dict(self.merged),
            "explicit_keys": list(self.explicit_keys),
            "from_memory_keys": list(self.from_memory_keys),
            "missing_keys": list(self.missing_keys),
        }


# ---------------------------------------------------------------------------
class SessionMemory:
    """File-backed session memory.

    Concurrency: the Critic agent is single-process per A2A session today.
    We therefore do not implement file locking — the worst we'd do under
    concurrent writers is overwrite the small JSON files. If multi-writer
    becomes a concern later, swap the underlying store, not the API.
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = os.environ.get(
                "CRITIC_SESSION_MEMORY_DIR", DEFAULT_SESSION_MEMORY_DIR
            )
        self.root = Path(root)
        self.prior_cache_ttl = float(
            os.environ.get(
                "CRITIC_PRIOR_CACHE_TTL_SECONDS",
                str(DEFAULT_PRIOR_CACHE_TTL_SECONDS),
            )
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def session_dir(self, session_id: str) -> Path:
        if not session_id or not isinstance(session_id, str):
            raise SessionMemoryError(f"invalid session_id: {session_id!r}")
        # Disallow path traversal — session_id is meant to be a short
        # opaque token, not a path fragment.
        if "/" in session_id or ".." in session_id:
            raise SessionMemoryError(f"session_id must not contain slashes: {session_id!r}")
        return self.root / session_id

    def _ensure_session_dir(self, session_id: str) -> Path:
        d = self.session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _context_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "context.json"

    def _decisions_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "decisions.jsonl"

    def _events_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "events.jsonl"

    def _priors_cache_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "kb_priors_cache.json"

    def _reviewed_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "reviewed_msg_ids.json"

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def load_context(self, session_id: str) -> dict[str, Any]:
        path = self._context_path(session_id)
        if not path.exists():
            return {}
        return _read_json(path, default={})

    def save_context(self, session_id: str, context: dict[str, Any]) -> None:
        if not isinstance(context, dict):
            raise SessionMemoryError(
                f"context must be a dict, got {type(context).__name__}"
            )
        self._ensure_session_dir(session_id)
        _write_json_atomic(self._context_path(session_id), context)

    def merge_context(
        self,
        session_id: str,
        incoming: dict[str, Any],
        *,
        persist: bool = True,
    ) -> MergeResult:
        """Merge ``incoming`` against stored context with explicit-wins semantics.

        ``persist`` defaults to ``True`` so callers don't forget to write
        back; pass ``persist=False`` for read-only merges (e.g. dry-run).
        """
        if not isinstance(incoming, dict):
            raise SessionMemoryError(
                f"incoming context must be a dict, got {type(incoming).__name__}"
            )
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

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------
    def append_decision(self, session_id: str, decision_review: dict[str, Any]) -> None:
        if not isinstance(decision_review, dict):
            raise SessionMemoryError(
                "decision_review must be a dict"
            )
        self._ensure_session_dir(session_id)
        record = {
            "ts": _now_iso(),
            "decision_review": decision_review,
        }
        _append_jsonl(self._decisions_path(session_id), record)

    def list_decisions(self, session_id: str) -> list[dict[str, Any]]:
        path = self._decisions_path(session_id)
        if not path.exists():
            return []
        return list(_read_jsonl(path))

    # ------------------------------------------------------------------
    # Events (free-form audit trail)
    # ------------------------------------------------------------------
    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise SessionMemoryError("event must be a dict")
        self._ensure_session_dir(session_id)
        _append_jsonl(self._events_path(session_id), {"ts": _now_iso(), **event})

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self._events_path(session_id)
        if not path.exists():
            return []
        return list(_read_jsonl(path))

    # ------------------------------------------------------------------
    # KB priors cache (per-scope+topic)
    # ------------------------------------------------------------------
    def get_cached_priors(
        self,
        session_id: str,
        cache_key: str,
        *,
        now: float | None = None,
    ) -> list[dict[str, Any]] | None:
        cache = _read_json(self._priors_cache_path(session_id), default={})
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
        if not isinstance(priors, list):
            raise SessionMemoryError("priors must be a list")
        self._ensure_session_dir(session_id)
        path = self._priors_cache_path(session_id)
        cache = _read_json(path, default={})
        cache[cache_key] = {"ts": time.time(), "priors": list(priors)}
        _write_json_atomic(path, cache)

    # ------------------------------------------------------------------
    # Already-reviewed proposals
    # ------------------------------------------------------------------
    def is_msg_already_reviewed(self, session_id: str, msg_id: str) -> bool:
        data = _read_json(self._reviewed_path(session_id), default={})
        if not isinstance(data, dict):
            return False
        return msg_id in data

    def reviewed_verdict_for(self, session_id: str, msg_id: str) -> str | None:
        data = _read_json(self._reviewed_path(session_id), default={})
        if not isinstance(data, dict):
            return None
        entry = data.get(msg_id)
        if isinstance(entry, dict):
            verdict = entry.get("verdict")
            if isinstance(verdict, str):
                return verdict
        return None

    def mark_reviewed(
        self,
        session_id: str,
        msg_id: str,
        verdict: str,
        *,
        decision_id: str | None = None,
    ) -> None:
        if not msg_id or not verdict:
            raise SessionMemoryError("msg_id and verdict are required")
        self._ensure_session_dir(session_id)
        path = self._reviewed_path(session_id)
        data = _read_json(path, default={})
        if not isinstance(data, dict):
            data = {}
        data[msg_id] = {
            "verdict": verdict,
            "ts": _now_iso(),
            "decision_id": decision_id,
        }
        _write_json_atomic(path, data)

    def filter_unreviewed(
        self,
        session_id: str,
        msg_ids: Iterable[str],
    ) -> list[str]:
        data = _read_json(self._reviewed_path(session_id), default={})
        if not isinstance(data, dict):
            data = {}
        return [m for m in msg_ids if m not in data]


# ---------------------------------------------------------------------------
# Tiny JSON helpers — kept private so we don't grow them into a real ORM.
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8") or "null")
    except json.JSONDecodeError as exc:
        raise SessionMemoryError(f"corrupt json at {path}: {exc}") from exc


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionMemoryError(
                    f"corrupt jsonl line at {path}: {exc}"
                ) from exc
            if isinstance(obj, dict):
                yield obj


__all__ = [
    "DEFAULT_PRIOR_CACHE_TTL_SECONDS",
    "DEFAULT_SESSION_MEMORY_DIR",
    "MergeResult",
    "SessionMemory",
]
