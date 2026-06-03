"""Per-session memory store for the Critic agent.

The Critic must keep state inside a single session because the Coordinator
(and other A2A hosts) only send the full context on the *first* call —
subsequent turns may carry incremental messages plus a decision. Critic
needs to merge those turns against the previously known context, recall
already-reviewed proposals, and reuse cached KB priors instead of hammering
the KB on every reactor tick.

Design constraints (handoff doc §5):

* MVP storage is local JSON / JSONL — no database dependency.
* Stateful per-session, stateless across sessions: nothing in this file
  is intended to outlive the session (long-term knowledge goes to remote
  KB).
* Files are append-only where it makes sense (decisions, events) and
  small-and-rewritten where consolidation is cheaper (context, priors
  cache, reviewed msg ids).
* Merging an incoming context dict with the stored one is **explicit-wins**:
  values present in the request override stored values, but stored values
  fill in for keys the request omitted. Both ``""`` and ``"unknown"`` are
  treated as missing.

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
    """Return ``True`` if value should be treated as absent.

    A value counts as missing when it is ``None`` or a string that, once
    trimmed and lower-cased, is one of the placeholder tokens in
    ``_MISSING_VALUES`` (``""``, ``"unknown"``, ``"null"``, ``"none"``).

    Args:
        value (Any): The candidate value to test.

    Returns:
        bool: ``True`` if the value should be treated as absent, else ``False``.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in _MISSING_VALUES:
        return True
    return False


# ---------------------------------------------------------------------------
@dataclass
class MergeResult:
    """Result of merging an incoming context against stored memory.

    Attributes:
        merged (dict[str, Any]): The merged context after explicit-wins
            resolution against stored memory.
        explicit_keys (list[str]): Keys supplied (and non-missing) in the
            incoming request.
        from_memory_keys (list[str]): Mergeable keys filled in from stored
            memory because the request omitted them.
        missing_keys (list[str]): Mergeable keys still absent after the merge.
    """

    merged: dict[str, Any] = field(default_factory=dict)
    explicit_keys: list[str] = field(default_factory=list)
    from_memory_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of this merge result.

        Returns:
            dict[str, Any]: A dict with ``merged``, ``explicit_keys``,
            ``from_memory_keys`` and ``missing_keys`` entries, each copied so
            callers cannot mutate the underlying result.
        """
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
        """Initialise the store rooted at ``root``.

        Args:
            root (str | Path | None): Directory under which per-session
                folders live. When ``None``, ``CRITIC_SESSION_MEMORY_DIR`` is
                used, falling back to ``DEFAULT_SESSION_MEMORY_DIR``.
        """
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
        """Return the directory for ``session_id`` under the store root.

        The id is treated as an opaque token; slashes and ``..`` are rejected
        to prevent path traversal outside the store root.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            Path: The per-session directory (not created by this call).

        Raises:
            SessionMemoryError: If ``session_id`` is empty, not a string, or
                contains a slash or ``..``.
        """
        if not session_id or not isinstance(session_id, str):
            raise SessionMemoryError(f"invalid session_id: {session_id!r}")
        # Disallow path traversal — session_id is meant to be a short
        # opaque token, not a path fragment.
        if "/" in session_id or ".." in session_id:
            raise SessionMemoryError(f"session_id must not contain slashes: {session_id!r}")
        return self.root / session_id

    def _ensure_session_dir(self, session_id: str) -> Path:
        """Create the session directory if needed and return it.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            Path: The session directory, created (with parents) if absent.
        """
        d = self.session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _context_path(self, session_id: str) -> Path:
        """Return the path to the session's ``context.json``.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            Path: Path to the merged-context file for the session.
        """
        return self.session_dir(session_id) / "context.json"

    def _decisions_path(self, session_id: str) -> Path:
        """Return the path to the session's ``decisions.jsonl``.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            Path: Path to the append-only decisions log for the session.
        """
        return self.session_dir(session_id) / "decisions.jsonl"

    def _events_path(self, session_id: str) -> Path:
        """Return the path to the session's ``events.jsonl``.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            Path: Path to the append-only audit-trail log for the session.
        """
        return self.session_dir(session_id) / "events.jsonl"

    def _priors_cache_path(self, session_id: str) -> Path:
        """Return the path to the session's ``kb_priors_cache.json``.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            Path: Path to the cached KB priors file for the session.
        """
        return self.session_dir(session_id) / "kb_priors_cache.json"

    def _reviewed_path(self, session_id: str) -> Path:
        """Return the path to the session's ``reviewed_msg_ids.json``.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            Path: Path to the reviewed-message-id index for the session.
        """
        return self.session_dir(session_id) / "reviewed_msg_ids.json"

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def load_context(self, session_id: str) -> dict[str, Any]:
        """Load the stored context for a session.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            dict[str, Any]: The stored context, or an empty dict if none has
            been persisted yet.
        """
        path = self._context_path(session_id)
        if not path.exists():
            return {}
        return _read_json(path, default={})

    def save_context(self, session_id: str, context: dict[str, Any]) -> None:
        """Persist ``context`` as the session's full context.

        Args:
            session_id (str): The opaque session identifier.
            context (dict[str, Any]): The context dict to write atomically.

        Raises:
            SessionMemoryError: If ``context`` is not a dict.
        """
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

        Non-missing values from ``incoming`` override stored values; stored
        values for ``_MERGEABLE_CONTEXT_KEYS`` fill in keys the request
        omitted. ``persist`` defaults to ``True`` so callers don't forget to
        write back; pass ``persist=False`` for read-only merges (e.g. dry-run).

        Args:
            session_id (str): The opaque session identifier.
            incoming (dict[str, Any]): The incoming context to merge in.
            persist (bool): When ``True``, the merged context is written back
                to disk before returning. Defaults to ``True``.

        Returns:
            MergeResult: The merged context plus the explicit, from-memory and
            still-missing key lists.

        Raises:
            SessionMemoryError: If ``incoming`` is not a dict.
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
        """Append a decision review record to the session's decisions log.

        The record is timestamped and written as one JSONL line.

        Args:
            session_id (str): The opaque session identifier.
            decision_review (dict[str, Any]): The decision review payload to
                persist.

        Raises:
            SessionMemoryError: If ``decision_review`` is not a dict.
        """
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
        """Return all decision records logged for a session.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            list[dict[str, Any]]: The decision records in append order, or an
            empty list if none exist.
        """
        path = self._decisions_path(session_id)
        if not path.exists():
            return []
        return list(_read_jsonl(path))

    # ------------------------------------------------------------------
    # Events (free-form audit trail)
    # ------------------------------------------------------------------
    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Append a free-form audit event to the session's events log.

        The event is timestamped (``ts``) and written as one JSONL line.

        Args:
            session_id (str): The opaque session identifier.
            event (dict[str, Any]): The event payload to persist.

        Raises:
            SessionMemoryError: If ``event`` is not a dict.
        """
        if not isinstance(event, dict):
            raise SessionMemoryError("event must be a dict")
        self._ensure_session_dir(session_id)
        _append_jsonl(self._events_path(session_id), {"ts": _now_iso(), **event})

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        """Return all audit events logged for a session.

        Args:
            session_id (str): The opaque session identifier.

        Returns:
            list[dict[str, Any]]: The event records in append order, or an
            empty list if none exist.
        """
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
        """Return cached KB priors for a key if present and not expired.

        Args:
            session_id (str): The opaque session identifier.
            cache_key (str): The scope/topic cache key.
            now (float | None): Current Unix time, injectable for testing.
                Defaults to ``time.time()`` when ``None``.

        Returns:
            list[dict[str, Any]] | None: The cached priors, or ``None`` if the
            entry is absent, malformed, or older than ``prior_cache_ttl``.
        """
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
        """Store KB priors under ``cache_key`` with the current timestamp.

        Args:
            session_id (str): The opaque session identifier.
            cache_key (str): The scope/topic cache key.
            priors (list[dict[str, Any]]): The priors to cache.

        Raises:
            SessionMemoryError: If ``priors`` is not a list.
        """
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
        """Return whether a proposal message has already been reviewed.

        Args:
            session_id (str): The opaque session identifier.
            msg_id (str): The proposal message id to check.

        Returns:
            bool: ``True`` if a verdict was recorded for ``msg_id``.
        """
        data = _read_json(self._reviewed_path(session_id), default={})
        if not isinstance(data, dict):
            return False
        return msg_id in data

    def reviewed_verdict_for(self, session_id: str, msg_id: str) -> str | None:
        """Return the recorded verdict for a reviewed message, if any.

        Args:
            session_id (str): The opaque session identifier.
            msg_id (str): The proposal message id to look up.

        Returns:
            str | None: The stored verdict string, or ``None`` if the message
            was not reviewed or no verdict was recorded.
        """
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
        """Record that a proposal message was reviewed with a verdict.

        Args:
            session_id (str): The opaque session identifier.
            msg_id (str): The proposal message id being marked.
            verdict (str): The verdict assigned to the proposal.
            decision_id (str | None): Optional id of the owning decision.

        Raises:
            SessionMemoryError: If ``msg_id`` or ``verdict`` is empty.
        """
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
        """Return the subset of ``msg_ids`` not yet reviewed this session.

        Args:
            session_id (str): The opaque session identifier.
            msg_ids (Iterable[str]): Candidate proposal message ids.

        Returns:
            list[str]: The message ids that have no recorded verdict yet,
            preserving input order.
        """
        data = _read_json(self._reviewed_path(session_id), default={})
        if not isinstance(data, dict):
            data = {}
        return [m for m in msg_ids if m not in data]


# ---------------------------------------------------------------------------
# Tiny JSON helpers — kept private so we don't grow them into a real ORM.
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """Return the current UTC time as a microsecond ISO-8601 string.

    Returns:
        str: The current UTC timestamp, e.g. ``2026-06-02T18:00:00.000000+00:00``.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _read_json(path: Path, *, default: Any) -> Any:
    """Read and decode a JSON file, returning ``default`` if absent.

    Args:
        path (Path): The file to read.
        default (Any): Value returned when the file does not exist.

    Returns:
        Any: The decoded JSON value, or ``default`` if the file is missing.

    Raises:
        SessionMemoryError: If the file exists but contains invalid JSON.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8") or "null")
    except json.JSONDecodeError as exc:
        raise SessionMemoryError(f"corrupt json at {path}: {exc}") from exc


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write ``data`` as indented JSON atomically via a temp file + rename.

    Args:
        path (Path): The destination file.
        data (Any): A JSON-serialisable value to write.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record as a JSON line to ``path``.

    Args:
        path (Path): The JSONL file to append to.
        record (dict[str, Any]): The record to serialise on its own line.
    """
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield dict records from a JSONL file, skipping blank lines.

    Args:
        path (Path): The JSONL file to read.

    Yields:
        dict[str, Any]: Each decoded JSON object line (non-dict lines are
        skipped).

    Raises:
        SessionMemoryError: If a non-blank line contains invalid JSON.
    """
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
