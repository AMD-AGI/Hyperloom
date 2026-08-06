# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Direct GBrain Recipe store used by remote KnowledgePlane mode."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import now_iso

from .gbrain_ingest import recipe_to_page
from .gbrain_remote_client import (
    GbrainRemoteError,
    GbrainRemoteRecipeClient,
    _GbrainMcp,
    _slug_for_canonical,
)
from .local_store import (
    _collection_counts,
    _normalise_lessons,
    _normalise_prs,
    _normalise_sessions,
    _normalise_str_dicts,
)
from .schema import Recipe

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Linux production path
    _fcntl = None  # type: ignore[assignment]


class GbrainRecipeLockError(RuntimeError):
    """Remote Recipe locking is unavailable or unsafe to use."""


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.Lock] = {}


def _thread_lock_for(path: Path) -> threading.Lock:
    """Return one process-local lock for a lock-file path."""

    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


def _lock_error() -> GbrainRecipeLockError:
    return GbrainRecipeLockError(
        "remote Recipe locking failed; KNOWLEDGE_LOCAL_ROOT must be a shared read-write POSIX-lock-capable filesystem"
    )


@contextmanager
def _canonical_lock(lock_root: Path, canonical_id: str):
    """Serialize a canonical Recipe across threads and processes."""

    if _fcntl is None:
        raise GbrainRecipeLockError("remote Recipe mode requires POSIX fcntl file locking, which is unavailable")
    digest = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()
    lock_path = lock_root / f"{digest}.lock"
    thread_lock = _thread_lock_for(lock_path)
    with thread_lock:
        fd: int | None = None
        locked = False
        try:
            lock_root.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(lock_path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("lock target is not a regular file")
            _fcntl.flock(fd, _fcntl.LOCK_EX)
            locked = True
        except Exception as exc:
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)
            raise _lock_error() from exc
        try:
            yield
        finally:
            if locked and fd is not None:
                with suppress(Exception):
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)


def _json_key(value: Any) -> str:
    """Return a stable JSON identity for collection deduplication."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _merge_json_collection(latest: list[Any], incoming: list[Any]) -> list[Any]:
    """Dedupe by deterministic JSON while retaining latest-first order."""

    merged: list[Any] = []
    seen: set[str] = set()
    for value in [*latest, *incoming]:
        key = _json_key(value)
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def _merge_sessions(latest: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace same-id sessions in place and append newly observed ids."""

    merged = [dict(session) for session in latest]
    indexes = {str(session.get("session_id") or ""): index for index, session in enumerate(merged)}
    for session in incoming:
        value = dict(session)
        session_id = str(value.get("session_id") or "")
        if session_id in indexes:
            merged[indexes[session_id]] = value
        else:
            indexes[session_id] = len(merged)
            merged.append(value)
    return merged


def _merge_nonempty_mapping(latest: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Overlay only non-empty incoming mapping values."""

    merged = dict(latest)
    for key, value in incoming.items():
        if value:
            merged[key] = value
    return merged


@dataclass
class GbrainRecipeStore:
    """Single-source direct remote store with shared per-canonical locking."""

    client: GbrainRemoteRecipeClient
    mcp: _GbrainMcp
    lock_root: Path | str | None = None
    backend_name: str = "gbrain"

    def __post_init__(self) -> None:
        if self.lock_root is None or not str(self.lock_root).strip():
            raise ValueError("remote GBrain Recipe store requires lock_root derived from KNOWLEDGE_LOCAL_ROOT")
        self.lock_root = Path(self.lock_root)
        if _fcntl is None:
            raise GbrainRecipeLockError("remote Recipe mode requires POSIX fcntl file locking, which is unavailable")

    @classmethod
    def from_credentials(
        cls,
        *,
        base_url: str,
        token: str,
        lock_root: Path | str,
        timeout_sec: float | None = None,
    ) -> "GbrainRecipeStore":
        client = GbrainRemoteRecipeClient(
            base_url=base_url,
            token=token,
            enabled=True,
            timeout_sec=timeout_sec,
        )
        if client._mcp is None:
            raise ValueError("GBrain Recipe store requires valid remote credentials")
        return cls(client=client, mcp=client._mcp, lock_root=lock_root)

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        row = self.client.get_recipe(canonical_id=canonical_id, version=version)
        return self._arbor(row)

    def get_recipe_exact(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Read only the stable canonical slug without fallback search."""

        row = self.client.get_recipe_exact(canonical_id=canonical_id, version=version)
        return self._arbor(row)

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [self._arbor(row) for row in self.client.search(**kwargs) if row]

    def put_recipe(
        self,
        *,
        canonical_id: str,
        model: str = "",
        hardware: str = "",
        framework_name: str = "",
        framework_version: str = "",
        precision: str = "",
        best_config: dict[str, str] | None = None,
        best_throughput: float = 0.0,
        what_worked: list[Any] | None = None,
        what_failed: list[Any] | None = None,
        remaining_gaps: list[Any] | None = None,
        prs_tested: list[Any] | None = None,
        pitfalls: list[Any] | None = None,
        lessons: list[Any] | None = None,
        last_profiled: str = "",
        stack_fingerprint: dict[str, str] | None = None,
        sessions: list[Any] | None = None,
        authority: str = "EXPERIENTIAL",
        confidence: float = 0.85,
        evidence_refs: list[Any] | None = None,
        provenance: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Lock, reread, merge, and replace the canonical remote page."""

        if not canonical_id:
            raise ValueError("put_recipe requires a non-empty canonical_id")
        assert isinstance(self.lock_root, Path)
        with _canonical_lock(self.lock_root, canonical_id):
            # This exact remote read must happen after both locks are held.
            remote_row = self.client._get_page_recipe(_slug_for_canonical(canonical_id))
            prior = self._arbor(remote_row) or {}
            if str(prior.get("canonical_id") or "") != canonical_id:
                prior = {}
            prior_recipe = Recipe.from_dict(prior) if prior else None
            latest = prior_recipe.to_dict() if prior_recipe is not None else {}
            latest_extras = dict(prior_recipe.extras) if prior_recipe is not None else {}
            now = now_iso(timespec="microseconds")
            version = int(latest.get("version") or 0) + 1

            latest_config = dict(latest.get("best_config") or {})
            incoming_config = dict(best_config or {})
            latest_throughput = float(latest.get("best_throughput") or 0.0)
            incoming_throughput = float(best_throughput)
            champion = "latest_preserved"
            if not latest:
                merged_config = incoming_config
                merged_throughput = incoming_throughput
                champion = "incoming_new"
            elif not latest_config and incoming_config:
                merged_config = incoming_config
                merged_throughput = incoming_throughput
                champion = "incoming_filled_empty"
            elif latest_config and incoming_config and incoming_throughput > latest_throughput:
                merged_config = incoming_config
                merged_throughput = incoming_throughput
                champion = "incoming_higher_throughput"
            else:
                merged_config = latest_config
                merged_throughput = latest_throughput

            latest_worked = _normalise_str_dicts(latest.get("what_worked"), ("description", "measured_impact"))
            latest_failed = _normalise_str_dicts(latest.get("what_failed"), ("description", "reason"))
            latest_gaps = _normalise_str_dicts(latest.get("remaining_gaps"), ("description", "metrics"))
            latest_pitfalls = _normalise_str_dicts(latest.get("pitfalls"), ("description", "severity"))
            payload: dict[str, Any] = {
                "canonical_id": canonical_id,
                "version": version,
                "created_at": str(latest.get("created_at") or now),
                "updated_at": now,
                "model": model or str(latest.get("model") or ""),
                "hardware": hardware or str(latest.get("hardware") or ""),
                "framework_name": framework_name or str(latest.get("framework_name") or ""),
                "framework_version": framework_version or str(latest.get("framework_version") or ""),
                "precision": precision or str(latest.get("precision") or ""),
                "best_config": merged_config,
                "best_throughput": merged_throughput,
                "what_worked": _merge_json_collection(
                    latest_worked,
                    _normalise_str_dicts(what_worked, ("description", "measured_impact")),
                ),
                "what_failed": _merge_json_collection(
                    latest_failed,
                    _normalise_str_dicts(what_failed, ("description", "reason")),
                ),
                "remaining_gaps": _merge_json_collection(
                    latest_gaps,
                    _normalise_str_dicts(remaining_gaps, ("description", "metrics")),
                ),
                "prs_tested": _merge_json_collection(
                    _normalise_prs(latest.get("prs_tested")),
                    _normalise_prs(prs_tested),
                ),
                "pitfalls": _merge_json_collection(
                    latest_pitfalls,
                    _normalise_str_dicts(pitfalls, ("description", "severity")),
                ),
                "lessons": _merge_json_collection(
                    _normalise_lessons(latest.get("lessons")),
                    _normalise_lessons(lessons),
                ),
                "last_profiled": last_profiled or str(latest.get("last_profiled") or ""),
                "stack_fingerprint": _merge_nonempty_mapping(
                    dict(latest.get("stack_fingerprint") or {}),
                    dict(stack_fingerprint or {}),
                ),
                "sessions": _merge_sessions(
                    _normalise_sessions(latest.get("sessions")),
                    _normalise_sessions(sessions),
                ),
                "authority": authority,
                "confidence": float(confidence),
                "evidence_refs": _merge_json_collection(
                    list(latest.get("evidence_refs") or []),
                    list(evidence_refs or []),
                ),
                "provenance": dict(provenance or {}),
            }
            merged_extras = _merge_nonempty_mapping(latest_extras, dict(extras or {}))
            for key, value in merged_extras.items():
                payload.setdefault(key, value)
            written = Recipe.from_dict(payload).to_dict()
            page = recipe_to_page(written)
            if page is None:
                raise GbrainRemoteError(
                    f"could not serialize remote recipe {canonical_id}",
                    category="validation",
                )
            slug, content = page
            # Deliberately direct: lock metadata is the only local filesystem state.
            self.mcp.call("put_page", {"slug": slug, "content": content})
            self.client._scan_cache = None
            return {
                "canonical_id": canonical_id,
                "version": version,
                "created": not bool(latest),
                "prior_counts": _collection_counts(latest),
                "counts": _collection_counts(written),
                "write_safety": {
                    "lock": "thread+posix-flock",
                    "latest_read": "inside_lock",
                    "merge": "latest_then_incoming",
                    "champion": champion,
                },
            }

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _arbor(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        # Runtime import avoids a package-init cycle.
        from .dispatcher import _v2_to_arbor

        return _v2_to_arbor(row)


__all__ = ["GbrainRecipeLockError", "GbrainRecipeStore"]
