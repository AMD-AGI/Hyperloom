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
    """Normalise a scope value to a trimmed, lower-cased string.

    Args:
        value (Any): The raw scope value (any type, or ``None``).

    Returns:
        str: ``""`` for ``None``, otherwise ``str(value).strip().lower()``.
    """
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalise_scope(scope: dict[str, Any]) -> dict[str, str]:
    """Normalise every value in a scope dict via :func:`_normalise_value`.

    Args:
        scope (dict[str, Any]): The raw scope mapping.

    Returns:
        dict[str, str]: A new mapping with the same keys and normalised values.
    """
    return {k: _normalise_value(v) for k, v in scope.items()}


def _scope_contains(row_scope: dict[str, str], scope_filter: dict[str, Any]) -> bool:
    """Return True if ``row_scope`` matches every key/value in the filter.

    Args:
        row_scope (dict[str, str]): The stored row's normalised scope.
        scope_filter (dict[str, Any]): Wanted key/value pairs (normalised before
            comparison).

    Returns:
        bool: ``True`` if every filter key matches the row's value.
    """
    for k, v in scope_filter.items():
        wanted = _normalise_value(v)
        if row_scope.get(k, "") != wanted:
            return False
    return True


def _matches_metadata(metadata: dict[str, Any], filter_obj: dict[str, Any]) -> bool:
    """Recursive nested + array-contains matcher (G-7).

    Nested dict filters recurse into nested metadata; list filters require
    every expected item to be present in the metadata's list (array-contains);
    scalar filters require equality.

    Args:
        metadata (dict[str, Any]): The row's metadata.
        filter_obj (dict[str, Any]): The (possibly nested) match filter.

    Returns:
        bool: ``True`` if the metadata satisfies every filter clause.
    """
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
    """A single stored KB row in the in-memory client.

    Attributes:
        id (str): Synthetic row id (``kb_<hex>``).
        scope (dict[str, str]): Normalised scope key/value pairs.
        kind (str): The row kind (e.g. ``technique``, ``pitfall``).
        slug (str): Stable slug, unique within ``(scope, kind)``.
        importance (float): Importance score (monotonic under upsert).
        summary (str): Human-readable summary text.
        metadata (dict[str, Any]): Arbitrary nested metadata.
        edges (dict[str, list[str]]): Outgoing edges keyed by edge kind.
        deleted (bool): Soft-delete flag.
        created_at (float): Unix creation time.
        updated_at (float): Unix last-update time.
    """

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
        """Return a deep-copied, JSON-serialisable view of the row.

        Returns:
            dict[str, Any]: All row fields with ``metadata`` deep-copied and
            ``scope``/``edges`` copied so callers cannot mutate stored state.
        """
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
        """Initialise an empty in-memory KB.

        Args:
            time_fn (Callable[[], float]): Clock used for ``created_at`` /
                ``updated_at`` stamps; injectable so tests can pin time.
                Defaults to :func:`time.time`.
        """
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

        Args:
            endpoint (str): The endpoint name to fail (e.g. ``upsert``).
            times (int): Number of consecutive failures to schedule.
            error (dict[str, Any]): Error payload echoed into the raised
                :class:`KBValidationError`.
        """
        for _ in range(times):
            self._failure_queue.append({"endpoint": endpoint, "error": error})

    def _maybe_fail(self, endpoint: str) -> None:
        """Raise a scheduled failure if the queue head targets ``endpoint``.

        Args:
            endpoint (str): The endpoint about to be exercised.

        Raises:
            KBValidationError: If a failure was scheduled for this endpoint;
                the matching entry is consumed from the queue.
        """
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
        """List rows matching a scope filter (and optional metadata filter).

        Args:
            scope_filter (dict[str, Any]): Scope key/values rows must contain.
            kind (str | None): If set, only rows of this kind are returned.
            metadata_filter (dict[str, Any] | None): Optional nested metadata
                match (see :func:`_matches_metadata`).
            limit (int): Maximum number of rows to return (``0`` for no cap).
            sort_by (str): ``updated_at_desc`` (default) sorts newest first;
                any other value sorts ascending.
            include_deleted (bool): When ``True``, soft-deleted rows are kept.

        Returns:
            dict[str, Any]: ``{"entries": [...], "count": n}`` with each entry a
            serialised row dict.

        Raises:
            KBValidationError: If ``scope_filter`` is not a dict, or a failure
                was scheduled for the ``list`` endpoint.
        """
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
        """Insert or merge a row keyed by ``(scope, kind, slug)``.

        New keys insert a fresh row; existing keys merge summary/metadata/edges
        (partial merge, G-1) and only raise importance, never lower it
        (importance protection, G-2). Scope normalisation that alters a value
        surfaces a ``scope_value_normalized`` warning (G-3).

        Args:
            payload (dict[str, Any]): Must contain ``scope``, ``kind``,
                ``slug`` and ``importance``; ``summary``, ``metadata`` and
                ``edges`` are optional.

        Returns:
            dict[str, Any]: ``{"row": <row dict>, "created": bool,
            "warnings": [...]}``.

        Raises:
            KBValidationError: If a required field is missing or a failure was
                scheduled for the ``upsert`` endpoint.
        """
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
        """Insert many rows, optionally erroring on key conflicts.

        Args:
            items (list[dict[str, Any]]): Upsert payloads to insert.
            on_conflict (str): ``upsert`` (default) merges conflicts; ``error``
                raises when a ``(scope, kind, slug)`` key already exists.

        Returns:
            dict[str, Any]: ``{"results": [...], "count": n}`` with one upsert
            result per item.

        Raises:
            KBValidationError: If ``on_conflict`` is invalid, a conflict occurs
                under ``on_conflict="error"``, or a failure was scheduled for
                the ``batch_insert`` endpoint.
        """
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
        """Add directed edges, auto-mirroring ``contradicts`` edges (G-8).

        Edges whose source row is missing are skipped; ``contradicts`` edges
        are mirrored back onto the destination row, with a skip recorded when
        the destination is missing.

        Args:
            edges (list[dict[str, Any]]): Each edge needs ``kind``, ``from_id``
                and ``to_id``.

        Returns:
            dict[str, Any]: ``{"added": [...], "mirrored_to": [...],
            "mirror_skipped": [...]}``.

        Raises:
            KBValidationError: If an edge is missing a required field or a
                failure was scheduled for the ``edges/add`` endpoint.
        """
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
        """Return serialised copies of every stored row (test helper).

        Returns:
            list[dict[str, Any]]: All rows, including soft-deleted ones.
        """
        return [r.to_dict() for r in self._rows.values()]

    def reset(self) -> None:
        """Clear all rows, the unique index and any scheduled failures."""
        self._rows.clear()
        self._index.clear()
        self._failure_queue.clear()


# ---------------------------------------------------------------------------
def json_scope_key(scope: dict[str, str]) -> str:
    """Stable scope key used for `(scope, kind, slug)` UNIQUE.

    Args:
        scope (dict[str, str]): The normalised scope mapping.

    Returns:
        str: A deterministic ``k=v|k=v`` string with keys sorted.
    """
    return "|".join(f"{k}={scope[k]}" for k in sorted(scope.keys()))


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``incoming`` into a deep copy of ``base``.

    Nested dicts are merged key-by-key; any other value type in ``incoming``
    replaces the corresponding value in ``base``.

    Args:
        base (dict[str, Any]): The starting mapping (not mutated).
        incoming (dict[str, Any]): Values to overlay onto ``base``.

    Returns:
        dict[str, Any]: A new merged dict.
    """
    out = copy.deepcopy(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


__all__ = ["InMemoryKBClient", "json_scope_key"]
