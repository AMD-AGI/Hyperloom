"""Fan-out read remote that aggregates several recipe-KB backends.

Slots into :class:`recipe_kb.RecipeKB`'s single ``remote`` field like any
other read client, but fans every read across an ordered list of sub-remotes
(e.g. ``[gbrain, cortex]``) and merges the results so callers see ONE merged
arbor-shaped corpus.

Design (matches the dispatcher contract):

* Writes stay local-only — this client is read-side just like the others;
  ``put_recipe`` / ``append_attempt`` are absent.
* Every sub-remote's row is run through :func:`dispatcher._v2_to_arbor`, which
  is idempotent: gbrain rows (already arbor-shaped) pass through untouched and
  cortex rows (nested v2 envelope) are projected. So the composite operates on
  a uniform arbor shape and advertises ``returns_arbor_shape = True`` (the
  dispatcher then re-projects nothing).
* Merge policy (config: field-level enrichment with adaptive precedence):
    - dedup by the 5-tuple ``canonical_id``;
    - within a dedup group the highest-precedence row is the BASE
      (precedence = authority rank → confidence → field richness →
      best_throughput);
    - list fields (lessons / pitfalls / what_worked / …) are dedup-UNIONed
      across the group so coverage is maximised;
    - empty scalar fields on the base are back-filled from lower-precedence
      rows (so a base missing ``best_config`` can still inherit one);
    - a ``_sources`` marker records which backends contributed.
* Search results are the per-cid-merged rows ranked by the same precedence and
  truncated to ``limit``.

A sub-remote that is disabled or raises is skipped (best-effort), so a flaky
backend degrades coverage without failing the read. Session/attempt reads are
local-only at the dispatcher layer, so the composite delegates those to the
first source purely for direct-use completeness.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from .canonical_id import InvalidCanonicalIdError
from .dispatcher import _labels_from_canonical_id, _v2_to_arbor
from .remote_client import RemoteRecipeClientError

log = logging.getLogger(__name__)

# Authority ladder (higher = more trustworthy). Unknown authorities sort last.
_AUTHORITY_RANK = {
    "EXPERIENTIAL": 3,
    "VALIDATED": 2,
    "HYPOTHESIZED": 1,
}

# List-valued arbor fields that get dedup-unioned across a dedup group.
_LIST_UNION_FIELDS = (
    "what_worked", "what_failed", "remaining_gaps", "prs_tested",
    "pitfalls", "lessons", "evidence_refs", "sessions",
)
# Scalar/dict arbor fields back-filled onto the base when the base value is empty.
_BACKFILL_FIELDS = (
    "best_config", "best_throughput", "framework_version", "last_profiled",
    "stack_fingerprint", "provenance", "created_at", "updated_at",
)

# Per-source fetch floor so a low caller ``limit`` (e.g. get_recipe's limit=1)
# still pulls enough rows from each backend for a meaningful merge.
_FETCH_FLOOR = 25


def _richness(row: dict[str, Any]) -> int:
    """Count populated 'rich' fields — a tie-breaker that favours the row
    carrying actual warm-start payload over a sparse stub."""
    score = 0
    if row.get("best_config"):
        score += 1
    if row.get("what_worked"):
        score += 1
    if row.get("lessons"):
        score += 1
    if row.get("stack_fingerprint"):
        score += 1
    if row.get("prs_tested"):
        score += 1
    return score


def _precedence_key(row: dict[str, Any]) -> tuple[int, float, int, float]:
    """Higher tuple = preferred base. authority → confidence → richness → tput."""
    authority = str(row.get("authority") or "")
    try:
        confidence = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        throughput = float(row.get("best_throughput") or 0.0)
    except (TypeError, ValueError):
        throughput = 0.0
    return (_AUTHORITY_RANK.get(authority, 0), confidence, _richness(row), throughput)


def _is_empty(value: Any) -> bool:
    """Return whether a value counts as empty for back-fill purposes.

    Args:
        value: Value to test.

    Returns:
        ``True`` for ``None``, empty containers/strings, and numeric zero.
    """
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple)) and len(value) == 0:
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    return False


def _dedup_preserve(items: Iterable[Any]) -> list[Any]:
    """Order-preserving dedup of a small sequence."""
    out: list[Any] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out


def _union_lists(
    rows: list[dict[str, Any]],
    field: str,
) -> tuple[list[Any], list[Any]]:
    """Order-preserving dedup-union of one list field across rows.

    Returns ``(merged_items, contributor_sources)`` where ``contributor_sources``
    is the ordered, de-duplicated list of ``_source`` tags that contributed at
    least one element to the union (for field-level provenance).

    """
    out: list[Any] = []
    seen: set[str] = set()
    contributors: list[Any] = []
    for row in rows:
        src = row.get("_source")
        for item in (row.get(field) or []):
            try:
                key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                key = repr(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
                if src is not None and src not in contributors:
                    contributors.append(src)
    return out, contributors


def _merge_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Field-level enrichment merge of all rows sharing one canonical_id.

    Also emits ``_field_sources``: per actionable field, which backend(s) the
    surviving value came from — scalar fields map to the single winning/
    back-filling source, list fields to the ordered set of contributors. This
    is what lets the KB-eval layer attribute an outcome to gbrain vs cortex.
    """
    ordered = sorted(rows, key=_precedence_key, reverse=True)
    base = dict(ordered[0])
    base_src = base.get("_source")
    others = ordered[1:]
    field_sources: dict[str, Any] = {}

    # List fields: dedup-union across the whole group; provenance = contributors.
    for field in _LIST_UNION_FIELDS:
        merged, contributors = _union_lists(ordered, field)
        base[field] = merged
        if merged:
            field_sources[field] = contributors or ([base_src] if base_src else [])

    # Scalar/dict fields: keep base when populated, else back-fill from the
    # highest-precedence non-empty donor; provenance = whoever supplied it.
    for field in _BACKFILL_FIELDS:
        if not _is_empty(base.get(field)):
            if base_src is not None:
                field_sources[field] = base_src
            continue
        for other in others:
            if not _is_empty(other.get(field)):
                base[field] = other[field]
                if other.get("_source") is not None:
                    field_sources[field] = other["_source"]
                break

    base["_sources"] = _dedup_preserve(
        [r.get("_source") for r in ordered if r.get("_source")]
    )
    base["_field_sources"] = field_sources
    base.pop("_source", None)
    return base


class CompositeRemoteRecipeClient:
    """Aggregating read remote over an ordered list of sub-remotes."""

    # The composite emits already-final arbor rows; the dispatcher must not
    # re-project them.
    returns_arbor_shape = True

    def __init__(self, sources: Iterable[Any], *, names: list[str] | None = None) -> None:
        """Initialize the composite over an ordered list of sub-remotes.

        Args:
            sources: Sub-remote clients, in precedence order; ``None`` entries
                are dropped.
            names: Optional display names parallel to ``sources``; defaults to
                each source's ``_name`` or class name.
        """
        self._sources = [s for s in sources if s is not None]
        self._names = names or [
            getattr(s, "_name", None) or type(s).__name__ for s in self._sources
        ]

    @property
    def enabled(self) -> bool:
        """Enabled iff at least one sub-remote is enabled."""
        return any(getattr(s, "enabled", False) for s in self._sources)

    def _active(self) -> list[tuple[str, Any]]:
        """Return ``(name, source)`` pairs for the enabled sub-remotes.

        Returns:
            The enabled sources paired with their display names, in precedence
            order.
        """
        return [
            (self._names[i], s)
            for i, s in enumerate(self._sources)
            if getattr(s, "enabled", False)
        ]

    def _fan_out_search(self, *, name: str, source: Any, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Run one source's search, tag + normalize each row. Best-effort."""
        try:
            rows = source.search(**kwargs)
        except RemoteRecipeClientError as exc:
            log.warning("composite: source %s search failed (%s); skipping", name, exc)
            return []
        except Exception as exc:  # noqa: BLE001 — never let one backend break the read
            log.warning("composite: source %s search raised %r; skipping", name, exc)
            return []
        out: list[dict[str, Any]] = []
        for row in rows or []:
            arbor = _v2_to_arbor(row)
            if not isinstance(arbor, dict) or not arbor:
                continue
            arbor["_source"] = name
            out.append(arbor)
        return out

    def _merged_search(self, *, per_source_limit: int, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Fan out a search to all active sources and merge by canonical id.

        Args:
            per_source_limit: Row limit applied to each sub-remote.
            kwargs: Search keyword arguments forwarded to each source.

        Returns:
            Field-merged rows, one per canonical id, sorted by precedence.
        """
        sub_kwargs = dict(kwargs)
        sub_kwargs["limit"] = per_source_limit
        grouped: dict[str, list[dict[str, Any]]] = {}
        source_candidates: dict[str, int] = {}
        for name, source in self._active():
            rows = self._fan_out_search(name=name, source=source, kwargs=sub_kwargs)
            source_candidates[name] = len(rows)
            for row in rows:
                cid = row.get("canonical_id") or ""
                grouped.setdefault(cid, []).append(row)
        merged = [_merge_group(rows) for rows in grouped.values()]
        # Stamp per-source candidate counts on every merged row so the
        # downstream audit/trace can attribute coverage to each path
        # (e.g. gbrain vs cortex) without re-querying the backends.
        for row in merged:
            row["_source_candidates"] = dict(source_candidates)
        merged.sort(key=_precedence_key, reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Read surface consumed by RecipeKB
    # ------------------------------------------------------------------
    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search all active sources and return merged, ranked rows.

        Args:
            label_match: Exact label filters (the 5-tuple identity labels).
            metric_filters: ``{metric: {min, max}}`` filters.
            updated_since: Lower bound on ``updated_at``.
            order_by: Ordering directive forwarded to sub-remotes.
            limit: Maximum number of merged rows to return.
            prefer: Workload-similarity hints (accepted for parity; rerank is
                applied by the dispatcher).

        Returns:
            Up to ``limit`` merged recipe rows.
        """
        # Main's dispatcher passes workload-similarity hints through this
        # parameter. Sub-remotes accept it for interface parity while the
        # dispatcher performs the actual client-side rerank.
        kwargs = {
            "label_match": label_match,
            "metric_filters": metric_filters,
            "updated_since": updated_since,
            "order_by": order_by,
            "limit": limit,
            "prefer": prefer,
        }
        per_source = max(int(limit), _FETCH_FLOOR)
        return self._merged_search(per_source_limit=per_source, kwargs=kwargs)[: int(limit)]

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Exact-cid read: search every source by the 5-tuple labels, merge top."""
        try:
            labels = _labels_from_canonical_id(canonical_id)
        except InvalidCanonicalIdError as exc:
            log.warning("composite get_recipe: %s", exc)
            return None
        merged = self._merged_search(
            per_source_limit=_FETCH_FLOOR,
            kwargs={"label_match": labels, "limit": 1},
        )
        for row in merged:
            if row.get("canonical_id") == canonical_id:
                return row
        return merged[0] if merged else None

    # ------------------------------------------------------------------
    # Completeness for direct use (dispatcher reads these LOCAL-only, so these
    # are only hit by direct callers). Delegate to the first capable source.
    # ------------------------------------------------------------------
    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent merged recipes across all sources.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            Up to ``limit`` merged recipe rows ordered most-recent first.
        """
        return self._merged_search(
            per_source_limit=max(int(limit), _FETCH_FLOOR),
            kwargs={"label_match": None, "limit": limit, "metric_filters": None,
                    "updated_since": None, "order_by": "updated_at DESC"},
        )[: int(limit)]

    def _first_active(self) -> Any | None:
        """Return the first enabled sub-remote, or ``None`` if none are active."""
        for _name, source in self._active():
            return source
        return None

    def list_attempts(self, *, canonical_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return attempt rows for a recipe from the first active source.

        Args:
            canonical_id: Canonical recipe id.
            limit: Maximum number of attempts.

        Returns:
            Attempt rows from the first active source, or ``[]`` when none are
            active.
        """
        src = self._first_active()
        return src.list_attempts(canonical_id=canonical_id, limit=limit) if src else []

    def list_session_attempts(self, *, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """Return attempt rows for a session from the first active source.

        Args:
            session_id: Session identifier.
            limit: Maximum number of attempts.

        Returns:
            Session attempt rows, or ``[]`` when no source is active.
        """
        src = self._first_active()
        return src.list_session_attempts(session_id=session_id, limit=limit) if src else []

    def session_summary(self, *, session_id: str) -> dict[str, Any] | None:
        """Return a session summary from the first active source.

        Args:
            session_id: Session identifier.

        Returns:
            The session summary, or ``None`` when no source is active.
        """
        src = self._first_active()
        return src.session_summary(session_id=session_id) if src else None

    def health(self) -> bool:
        """Healthy iff any active source is healthy."""
        for _name, source in self._active():
            try:
                if source.health():
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def close(self) -> None:
        """Close all sub-remotes, ignoring individual close failures."""
        for source in self._sources:
            try:
                source.close()
            except Exception:  # noqa: BLE001 — best-effort
                pass


__all__ = ["CompositeRemoteRecipeClient"]
