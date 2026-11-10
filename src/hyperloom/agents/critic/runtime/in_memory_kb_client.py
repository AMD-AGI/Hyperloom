# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-memory KBClient for tests + dry-run mode."""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import KBValidationError
from .kb_client import KBListQuery
from .scope_builder import normalise_scope_value, scope_key


# ---------------------------------------------------------------------------
def _normalise_scope(scope: dict[str, Any]) -> dict[str, str]:
    """Normalise every value in a scope dict via :func:`normalise_scope_value`."""
    return {k: normalise_scope_value(v) for k, v in scope.items()}


def _scope_contains(row_scope: dict[str, str], scope_filter: dict[str, Any]) -> bool:
    """Return True if ``row_scope`` matches every key/value in the filter."""
    for k, v in scope_filter.items():
        wanted = normalise_scope_value(v)
        if row_scope.get(k, "") != wanted:
            return False
    return True


# ---------------------------------------------------------------------------
@dataclass
class _Row:
    """A single stored KB row in the in-memory client."""

    id: str
    scope: dict[str, str]
    kind: str
    slug: str
    importance: float
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied, JSON-serialisable view of the row."""
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
        }


_REQUIRED_UPSERT_FIELDS: tuple[str, ...] = ("scope", "kind", "slug", "importance")


# ---------------------------------------------------------------------------
class InMemoryKBClient:
    """Faithful in-memory mock of the KB service surface."""

    def __init__(self, *, time_fn: Callable[[], float] = time.time):
        """Initialise an empty in-memory KB."""
        self._rows: dict[str, _Row] = {}
        self._index: dict[tuple[str, str, str], str] = {}
        self._time_fn = time_fn

    # list
    def list(self, query: KBListQuery) -> dict[str, Any]:
        """List rows matching a scope filter, newest first."""
        if not isinstance(query.scope_filter, dict):
            raise KBValidationError("list: scope_filter must be an object")
        out: list[_Row] = []
        for row in self._rows.values():
            if query.kind is not None and row.kind != query.kind:
                continue
            if not _scope_contains(row.scope, query.scope_filter):
                continue
            out.append(row)
        out.sort(key=lambda r: r.updated_at, reverse=True)
        if query.limit:
            out = out[: query.limit]
        return {"entries": [r.to_dict() for r in out], "count": len(out)}

    # upsert
    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert or merge a row keyed by ``(scope, kind, slug)``."""
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
        key = (scope_key(scope), kind, slug)
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
        # Partial field merge.
        if "summary" in payload:
            row.summary = summary_in
        if metadata_in:
            row.metadata = _deep_merge(row.metadata, metadata_in)
        for kind_edge, ids in edges_in.items():
            existing = row.edges.setdefault(kind_edge, [])
            for eid in ids:
                if eid not in existing:
                    existing.append(eid)
        # Importance protection.
        if importance_in < row.importance:
            warnings.append("importance_protected")
        else:
            row.importance = importance_in
        row.updated_at = now
        return {"row": row.to_dict(), "created": False, "warnings": warnings}


# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``incoming`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


__all__ = ["InMemoryKBClient"]
