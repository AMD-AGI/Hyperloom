# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""In-memory KBClient for tests + dry-run mode (contract §7.3 faithful mock).

Honours: ``(scope, kind, slug)`` UNIQUE with upsert idempotency, partial
merge (G-1) and importance protection (G-2); ``contradicts`` auto-mirroring
(G-8); ``scope_filter`` containment with ``trim().lowercase()`` (G-3);
``metadata_filter`` nested + array-contains (G-7). :meth:`simulate_failure`
injects faults for dead-letter tests. ``updated_at`` uses an injectable
``time_fn`` so tests can pin time.
"""

from __future__ import annotations

import copy
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import KBValidationError


# ---------------------------------------------------------------------------
def _normalise_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalise_scope(scope: dict[str, Any]) -> dict[str, str]:
    return {k: _normalise_value(v) for k, v in scope.items()}


def _scope_contains(row_scope: dict[str, str], scope_filter: dict[str, Any]) -> bool:
    """Return True if ``row_scope`` matches every key/value in the filter."""
    for k, v in scope_filter.items():
        wanted = _normalise_value(v)
        if row_scope.get(k, "") != wanted:
            return False
    return True


def _matches_metadata(metadata: dict[str, Any], filter_obj: dict[str, Any]) -> bool:
    """Recursive nested + array-contains matcher (G-7)."""
    for key, expected in filter_obj.items():
        if isinstance(expected, dict):
            sub = metadata.get(key)
            if not isinstance(sub, dict):
                return False
            if not _matches_metadata(sub, expected):
                return False
            continue
        if isinstance(expected, list):
            haystack = metadata.get(key)
            if isinstance(haystack, list):
                # array contains: every item in expected must be in haystack
                for item in expected:
                    if item not in haystack:
                        return False
                continue
            return False
        if metadata.get(key) != expected:
            return False
    return True


# ---------------------------------------------------------------------------
@dataclass
class _Row:
    id: str
    scope: dict[str, str]
    kind: str
    slug: str
    importance: float
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    deleted: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": dict(self.scope),
            "kind": self.kind,
            "slug": self.slug,
            "importance": self.importance,
            "summary": self.summary,
            "metadata": copy.deepcopy(self.metadata),
            "edges": {k: list(v) for k, v in self.edges.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
        }


_REQUIRED_UPSERT_FIELDS: tuple[str, ...] = ("scope", "kind", "slug", "importance")


# ---------------------------------------------------------------------------
class InMemoryKBClient:
    """Faithful in-memory mock of the KB service surface."""

    def __init__(self, *, time_fn: Callable[[], float] = time.time):
        self._rows: dict[str, _Row] = {}
        self._index: dict[tuple[str, str, str], str] = {}
        self._time_fn = time_fn
        self._failure_queue: deque[dict[str, Any]] = deque()

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------
    def simulate_failure(
        self,
        *,
        endpoint: str,
        times: int,
        error: dict[str, Any],
    ) -> None:
        """Schedule ``times`` consecutive failures for ``endpoint``.

        ``error`` is propagated as a :class:`KBValidationError` so callers
        can simulate 4xx-class failures; for 5xx-class behaviour (retry
        loops) compose with :class:`runtime.kb_client.HTTPKBClient` in an
        integration test instead.
        """
        for _ in range(times):
            self._failure_queue.append({"endpoint": endpoint, "error": error})

    def _maybe_fail(self, endpoint: str) -> None:
        if not self._failure_queue:
            return
        head = self._failure_queue[0]
        if head["endpoint"] == endpoint:
            self._failure_queue.popleft()
            err = head["error"]
            raise KBValidationError(f"{endpoint}: simulated failure {err!r}")

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------
    def list(
        self,
        *,
        scope_filter: dict[str, Any],
        kind: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        sort_by: str = "updated_at_desc",
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        self._maybe_fail("list")
        if not isinstance(scope_filter, dict):
            raise KBValidationError("list: scope_filter must be an object")
        out: list[_Row] = []
        for row in self._rows.values():
            if row.deleted and not include_deleted:
                continue
            if kind is not None and row.kind != kind:
                continue
            if not _scope_contains(row.scope, scope_filter):
                continue
            if metadata_filter and not _matches_metadata(row.metadata, metadata_filter):
                continue
            out.append(row)
        out.sort(key=lambda r: r.updated_at, reverse=(sort_by == "updated_at_desc"))
        if limit:
            out = out[:limit]
        return {"entries": [r.to_dict() for r in out], "count": len(out)}

    # ------------------------------------------------------------------
    # upsert
    # ------------------------------------------------------------------
    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._maybe_fail("upsert")
        for f in _REQUIRED_UPSERT_FIELDS:
            if f not in payload:
                raise KBValidationError(f"upsert: missing field {f!r}")
        scope = _normalise_scope(payload["scope"])
        kind = payload["kind"]
        slug = payload["slug"]
        importance_in = float(payload["importance"])
        summary_in = payload.get("summary", "")
        metadata_in = payload.get("metadata") or {}
        edges_in = payload.get("edges") or {}

        warnings: list[str] = []
        # Scope value normalisation (G-3): warn when an incoming value differs
        # from its normalised form.
        for k, v in payload["scope"].items():
            if str(v).strip().lower() != _normalise_value(v) or _normalise_value(v) != _normalise_value(payload["scope"][k]):
                if str(v) != _normalise_value(v):
                    warnings.append("scope_value_normalized")
                    break

        key = (json_scope_key(scope), kind, slug)
        existing_id = self._index.get(key)
        now = self._time_fn()
        if existing_id is None:
            row = _Row(
                id=f"kb_{uuid.uuid4().hex[:10]}",
                scope=scope,
                kind=kind,
                slug=slug,
                importance=importance_in,
                summary=summary_in,
                metadata=copy.deepcopy(metadata_in),
                edges={k: list(v) for k, v in edges_in.items()},
                created_at=now,
                updated_at=now,
            )
            self._rows[row.id] = row
            self._index[key] = row.id
            return {"row": row.to_dict(), "created": True, "warnings": warnings}

        row = self._rows[existing_id]
        # Partial field merge (G-1).
        if "summary" in payload:
            row.summary = summary_in
        if metadata_in:
            row.metadata = _deep_merge(row.metadata, metadata_in)
        for kind_edge, ids in edges_in.items():
            existing = row.edges.setdefault(kind_edge, [])
            for eid in ids:
                if eid not in existing:
                    existing.append(eid)
        # Importance protection (G-2).
        if importance_in < row.importance:
            warnings.append("importance_protected")
        else:
            row.importance = importance_in
        row.updated_at = now
        return {"row": row.to_dict(), "created": False, "warnings": warnings}

    # ------------------------------------------------------------------
    # batch_insert
    # ------------------------------------------------------------------
    def batch_insert(
        self,
        items: list[dict[str, Any]],
        *,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        self._maybe_fail("batch_insert")
        if on_conflict not in ("upsert", "error"):
            raise KBValidationError(
                f"batch_insert: on_conflict must be upsert|error, got {on_conflict!r}"
            )
        results: list[dict[str, Any]] = []
        for item in items:
            if on_conflict == "upsert":
                results.append(self.upsert(item))
            else:
                key = (
                    json_scope_key(_normalise_scope(item["scope"])),
                    item["kind"],
                    item["slug"],
                )
                if key in self._index:
                    raise KBValidationError(
                        f"batch_insert: conflict on {key!r}; use on_conflict=upsert"
                    )
                results.append(self.upsert(item))
        return {"results": results, "count": len(results)}

    # ------------------------------------------------------------------
    # add_edges
    # ------------------------------------------------------------------
    def add_edges(self, edges: list[dict[str, Any]]) -> dict[str, Any]:
        self._maybe_fail("edges/add")
        added: list[dict[str, Any]] = []
        mirrored_to: list[dict[str, Any]] = []
        mirror_skipped: list[dict[str, Any]] = []
        for edge in edges:
            kind = edge.get("kind")
            src = edge.get("from_id")
            dst = edge.get("to_id")
            if not (kind and src and dst):
                raise KBValidationError(
                    f"edges/add: missing kind/from_id/to_id: {edge!r}"
                )
            row_src = self._rows.get(src)
            if row_src is None:
                # Source missing → standard 404 in real service.
                mirror_skipped.append({"edge": edge, "reason": "src_missing"})
                continue
            bucket = row_src.edges.setdefault(kind, [])
            if dst not in bucket:
                bucket.append(dst)
            added.append({"from_id": src, "to_id": dst, "kind": kind})
            # Auto-mirror for ``contradicts``.
            if kind == "contradicts":
                row_dst = self._rows.get(dst)
                if row_dst is None:
                    mirror_skipped.append({"edge": edge, "reason": "dst_missing"})
                    continue
                back = row_dst.edges.setdefault(kind, [])
                if src not in back:
                    back.append(src)
                mirrored_to.append({"from_id": dst, "to_id": src, "kind": kind})
        return {"added": added, "mirrored_to": mirrored_to, "mirror_skipped": mirror_skipped}

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    def all_rows(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._rows.values()]

    def reset(self) -> None:
        self._rows.clear()
        self._index.clear()
        self._failure_queue.clear()


# ---------------------------------------------------------------------------
def json_scope_key(scope: dict[str, str]) -> str:
    """Stable scope key used for `(scope, kind, slug)` UNIQUE."""
    return "|".join(f"{k}={scope[k]}" for k in sorted(scope.keys()))


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


__all__ = ["InMemoryKBClient", "json_scope_key"]
