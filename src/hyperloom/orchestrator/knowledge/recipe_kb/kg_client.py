# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Knowledge-Graph query client over the gbrain page store.

The kb-mirror drivers embed a ``## Facts`` fence in every consumable page
body (recipe / framework_patch / kernel / gemm_tune / arch_family). Each
fence line is a typed triple::

    - {Subject} {PREDICATE} {Object} (prop1: val1, prop2: val2, ...)

This module turns that document substrate into a small graph query
surface (``query_facts`` / ``graph_traverse`` / ``find_conflicts``) plus
incremental writes (``emit_fact`` / ``retract_fact``).

Two execution modes:

* **Simulation** — client-side. ``query_facts`` drives gbrain ``search`` +
  client-side fence parsing/filtering; ``graph_traverse`` runs a client BFS
  over ``query_facts``; writes are read-modify-write over ``get_page`` /
  ``put_page``.
* **Native (``use_native_kg``)** — maps the triple API onto gbrain's
  first-class *link graph*. A triple ``(subject, predicate, object)`` becomes
  an edge ``add_link(from=subject, to=object, link_type=predicate,
  context=json(properties))``; reads use ``get_links`` / ``get_backlinks`` /
  ``traverse_graph``. Fact properties are encoded in the edge ``context`` JSON.
  ``add_link`` requires both endpoints to exist, so writes materialize missing
  nodes; ``remove_link`` is pair-coarse, so ``retract_fact`` does a
  read-modify-write that preserves the other link types on the same pair.
  gbrain reports some write failures as an in-band ``{"error": ...}`` payload,
  so the write paths inspect the decoded result via :func:`_rpc_failed`.

Every read method has a ``*_safe`` companion that swallows transport failures
and returns an empty result, so the KG layer can never block or break a
warm-start that would otherwise succeed from the local store.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .gbrain_remote_client import GbrainRemoteError, _GbrainMcp

log = logging.getLogger(__name__)

# Cap the number of fact lines parsed per page (mirrors the mirror-side
# write cap) so a malformed/oversized page never blows up a query tick.
_MAX_FACTS_PER_PAGE = 50
# Hard ceiling on BFS breadth to bound client-side traversal cost.
_MAX_TRAVERSE_NODES = 200
# Default TTL for the per-client search cache (seconds). One warm-start
# enhancement issues many repeated searches; caching avoids re-hitting the
# foreground HTTP budget for the same query terms.
_SEARCH_CACHE_TTL_SEC = 30.0

# Parse one fence line: "- SUBJECT PREDICATE OBJECT (props)". Subject and
# object are slug tokens (no spaces); predicate is an uppercase relation.
_FACT_LINE_RE = re.compile(
    r"^-\s+(?P<subject>\S+)\s+(?P<predicate>[A-Z][A-Z0-9_]*)\s+(?P<object>\S+)\s*(?:\((?P<props>.*)\))?\s*$"
)


def _entity(value: Any) -> str:
    """Normalize an entity token the same way the mirror writer does.

    Args:
        value: Raw entity name.

    Returns:
        Lowercased slug (spaces/slashes to underscores), or ``""``.
    """
    s = str(value or "").strip().replace(" ", "_").replace("/", "_")
    return s.lower()


def _as_set(value: Any) -> set[str]:
    """Coerce a scalar / pipe-string / iterable of entities to a norm set.

    A predicate or entity filter may arrive as a single string, a
    pipe-delimited alternation (``"REVERTED_ON|DEGRADES"``), or a list.

    Args:
        value: The filter value (``None`` yields an empty set).

    Returns:
        The set of normalized tokens (empty means "match any").
    """
    if value is None:
        return set()
    items: Iterable[Any]
    if isinstance(value, str):
        items = value.split("|")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return {_entity(item) for item in items if str(item or "").strip()}


def _parse_props(raw: str | None) -> dict[str, str]:
    """Parse the ``(key: val, key: val)`` property blob of a fact line.

    Args:
        raw: The inner text between the parentheses (may be ``None``/empty).

    Returns:
        A mapping of property name to value (both stripped strings).
    """
    props: dict[str, str] = {}
    if not raw:
        return props
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, val = part.partition(":")
        if sep:
            props[key.strip()] = val.strip()
        else:
            props[part] = ""
    return props


@dataclass
class Fact:
    """A typed triple parsed from a page ``## Facts`` fence."""

    subject: str
    predicate: str
    object: str
    properties: dict[str, str] = field(default_factory=dict)
    source_slug: str = ""

    @property
    def gain(self) -> float:
        """Return the signed gain percentage, or ``0.0`` when absent."""
        return _pct(self.properties.get("gain"))

    @property
    def confidence(self) -> float:
        """Return the fact confidence in ``[0,1]`` (defaults to ``0.8``)."""
        try:
            return float(self.properties.get("confidence", 0.8))
        except (TypeError, ValueError):
            return 0.8


@dataclass
class GraphNode:
    """A node reached during :meth:`KGClient.graph_traverse`."""

    entity: str
    depth: int
    via: Fact


def _pct(value: Any) -> float:
    """Parse a percentage-like string (``"+35.2%"``) into a float.

    Args:
        value: Raw value (``"+35.2%"`` / ``"-4.2"`` / number / ``None``).

    Returns:
        The numeric value, or ``0.0`` when it cannot be parsed.
    """
    if value is None:
        return 0.0
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return 0.0


def parse_facts_fence(body: str, *, source_slug: str = "") -> list[Fact]:
    """Extract all triples from a page body's ``## Facts`` fence.

    Reads the bullet lines under the first ``## Facts`` header up to the
    next ``## `` header (or end of body). Malformed lines are skipped.

    Args:
        body: The full page body markdown.
        source_slug: Slug recorded on each parsed fact for provenance.

    Returns:
        The parsed facts (capped at :data:`_MAX_FACTS_PER_PAGE`).
    """
    if not body or "## Facts" not in body:
        return []
    facts: list[Fact] = []
    in_fence = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            # Enter the Facts section; any other ## header ends it.
            in_fence = stripped[3:].strip().lower().startswith("facts")
            continue
        if not in_fence:
            continue
        m = _FACT_LINE_RE.match(stripped)
        if not m:
            continue
        facts.append(
            Fact(
                subject=_entity(m.group("subject")),
                predicate=m.group("predicate").strip().upper(),
                object=_entity(m.group("object")),
                properties=_parse_props(m.group("props")),
                source_slug=source_slug,
            )
        )
        if len(facts) >= _MAX_FACTS_PER_PAGE:
            break
    return facts


def format_fact_line(subject: str, predicate: str, obj: str, properties: dict[str, Any] | None) -> str:
    """Render a triple into the canonical ``## Facts`` fence line.

    Args:
        subject: Subject entity.
        predicate: Relation (forced uppercase).
        obj: Object entity.
        properties: Optional property mapping.

    Returns:
        A single fence line, e.g. ``"- a IMPROVES b (gain: +5%)"``.
    """
    props = properties or {}
    inner = ", ".join(f"{k}: {v}" for k, v in props.items())
    return f"- {_entity(subject)} {str(predicate).strip().upper()} {_entity(obj)} ({inner})"


def _link_type(predicate: Any) -> str:
    """Normalize a predicate into a gbrain ``link_type`` token.

    gbrain link types are lowercase relation tokens (``invested_in``). The
    mirror writer and this reader must agree on the same normalization so
    edges round-trip.

    Args:
        predicate: Raw predicate / relation name.

    Returns:
        The lowercased link-type token, or ``""``.
    """
    return str(predicate or "").strip().lower()


def _props_to_context(properties: dict[str, Any] | None) -> str:
    """Serialize fact properties into an edge ``context`` JSON string.

    gbrain edges carry no structured properties — only a free-text
    ``context``. We encode the property map as compact JSON so the reader
    can faithfully reconstruct it. Empty-valued properties are dropped to
    match the mirror writer (``kg_links.props_to_context``).

    Args:
        properties: The fact property map (``None`` yields ``"{}"``).

    Returns:
        A compact JSON object string.
    """
    if not properties:
        return "{}"
    return json.dumps(
        {str(k): str(v) for k, v in properties.items() if str(v or "").strip()},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _context_to_props(context: Any) -> dict[str, str]:
    """Parse an edge ``context`` back into a property map.

    Args:
        context: The edge context (JSON string, dict, or ``None``).

    Returns:
        A ``str -> str`` property map (empty on missing/invalid context).
    """
    if not context:
        return {}
    if isinstance(context, dict):
        return {str(k): str(v) for k, v in context.items()}
    try:
        data = json.loads(context)
    except (ValueError, TypeError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _rpc_failed(result: Any) -> bool:
    """Return ``True`` when a gbrain tool result reports an in-band error.

    gbrain returns some failures (e.g. ``add_link`` on a missing page) as a
    normal content payload whose decoded JSON carries an ``error`` key, not
    as a JSON-RPC / ``isError`` envelope. The transport only raises for the
    latter, so write callers must inspect the decoded result themselves or a
    failed write silently parses as success.

    Args:
        result: The decoded tool result.

    Returns:
        ``True`` when the result is an error payload.
    """
    return isinstance(result, dict) and bool(result.get("error"))


def _edge_to_fact(edge: dict[str, Any]) -> Fact:
    """Build a :class:`Fact` from a gbrain link-graph edge row.

    A link-graph edge ``{from_slug, to_slug, link_type, context}`` maps to
    the triple ``(subject=from, predicate=link_type, object=to)`` with the
    decoded ``context`` as properties.

    Args:
        edge: A ``get_links`` / ``traverse_graph`` edge row.

    Returns:
        The corresponding :class:`Fact`.
    """
    return Fact(
        subject=_entity(edge.get("from_slug")),
        predicate=str(edge.get("link_type") or "").strip().upper(),
        object=_entity(edge.get("to_slug")),
        properties=_context_to_props(edge.get("context")),
        source_slug=str(edge.get("from_slug") or ""),
    )


def _conditions_match(fact: Fact, conditions: dict[str, Any] | None) -> bool:
    """Return ``True`` when every requested condition matches the fact.

    Matching is lenient (normalized substring either way) because the
    simulation layer compares free-form property text (``hw``/``fw``/
    ``condition``/``precision``).

    Args:
        fact: The candidate fact.
        conditions: Property constraints (``None``/empty matches all).

    Returns:
        ``True`` when all constraints are satisfied.
    """
    if not conditions:
        return True
    for key, want in conditions.items():
        have = fact.properties.get(key)
        if have is None:
            return False
        nh, nw = _entity(have), _entity(want)
        if nw and nw not in nh and nh not in nw:
            return False
    return True


class KGClient:
    """Knowledge-graph query surface over the gbrain page store.

    Simulates graph queries client-side via ``search`` + fence parsing, or
    delegates to native gbrain KG tools when ``use_native_kg`` is set.
    """

    def __init__(self, mcp: Any, *, use_native_kg: bool = False, search_limit: int = 100) -> None:
        """Initialize the client.

        Args:
            mcp: A duck-typed MCP client exposing ``call(tool, arguments)``.
            use_native_kg: Delegate to native gbrain KG tools when ``True``.
            search_limit: Default page fan-out per ``query_facts`` search.
        """
        self._mcp = mcp
        self._native = bool(use_native_kg)
        self._search_limit = max(1, int(search_limit))
        # query_str -> (monotonic_ts, [(slug, body), ...]); bounded by TTL.
        self._search_cache: dict[str, tuple[float, list[tuple[str, str]]]] = {}
        # Slugs confirmed to exist as pages, memoized by the native write path.
        self._known_nodes: set[str] = set()

    def is_available(self) -> bool:
        """Probe whether the backing store is reachable.

        Returns:
            ``True`` when a tiny ``list_pages`` probe succeeds.
        """
        if self._mcp is None:
            return False
        try:
            self._mcp.call("list_pages", {"limit": 1})
            return True
        except (GbrainRemoteError, OSError, TimeoutError) as exc:
            log.info("kg backend unavailable: %s", exc)
            return False

    def _cache_ttl(self) -> float:
        """Return the search-cache TTL in seconds (env-overridable)."""
        raw = os.environ.get("GBRAIN_KG_CACHE_TTL_SEC", "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        return _SEARCH_CACHE_TTL_SEC

    def _search_pages(self, query: str, limit: int) -> list[tuple[str, str]]:
        """Search pages and return ``(slug, body)`` pairs (TTL-cached).

        Tolerates list / ``{results|pages|hits: [...]}`` envelopes and fetches
        the body via ``get_page`` when the hit omits it. Results are memoized
        per query string for :data:`_SEARCH_CACHE_TTL_SEC`.

        Args:
            query: Free-text search string.
            limit: Maximum hits to fetch.

        Returns:
            A list of ``(slug, body)`` tuples.
        """
        if self._mcp is None:
            return []
        ttl = self._cache_ttl()
        now = time.monotonic()
        cache_key = f"{query}\x00{int(limit)}"
        if ttl > 0.0:
            cached = self._search_cache.get(cache_key)
            if cached is not None and now - cached[0] <= ttl:
                return list(cached[1])
        raw = self._mcp.call("search", {"query": query, "limit": int(limit)})
        hits: Sequence[Any]
        if isinstance(raw, dict):
            hits = raw.get("results") or raw.get("pages") or raw.get("hits") or []
        elif isinstance(raw, list):
            hits = raw
        else:
            hits = []
        out: list[tuple[str, str]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            slug = str(hit.get("slug") or hit.get("id") or "")
            body = hit.get("body")
            if not body and slug:
                body = self._page_body(slug)
            if slug and body:
                out.append((slug, str(body)))
        if ttl > 0.0:
            self._search_cache[cache_key] = (now, list(out))
        return out

    def _page_body(self, slug: str) -> str:
        """Fetch a page and return its body markdown (best-effort).

        Args:
            slug: The page slug.

        Returns:
            The page body, or ``""`` when unavailable.
        """
        if self._mcp is None:
            return ""
        try:
            page = self._mcp.call("get_page", {"slug": slug})
        except (GbrainRemoteError, OSError, TimeoutError):
            return ""
        return _page_content(page)

    @staticmethod
    def _as_edges(raw: Any) -> list[dict[str, Any]]:
        """Coerce a link-graph tool result into a list of edge rows.

        gbrain returns a bare list of edges; tolerate ``{links|paths|
        edges|nodes: [...]}`` envelopes too.

        Args:
            raw: The raw tool result.

        Returns:
            The edge rows (non-dict entries dropped).
        """
        rows: Any
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            rows = raw.get("links") or raw.get("paths") or raw.get("edges") or raw.get("nodes") or []
        else:
            rows = []
        return [r for r in rows if isinstance(r, dict)]

    def _get_links(self, slug: str) -> list[dict[str, Any]]:
        """Return outgoing edges for ``slug`` (best-effort)."""
        if self._mcp is None or not slug:
            return []
        return self._as_edges(self._mcp.call("get_links", {"slug": _entity(slug)}))

    def _get_backlinks(self, slug: str) -> list[dict[str, Any]]:
        """Return incoming edges for ``slug`` (best-effort)."""
        if self._mcp is None or not slug:
            return []
        return self._as_edges(self._mcp.call("get_backlinks", {"slug": _entity(slug)}))

    def _node_exists(self, slug: str) -> bool:
        """Return ``True`` when a page for ``slug`` exists."""
        try:
            page = self._mcp.call("get_page", {"slug": slug})
        except (GbrainRemoteError, OSError, TimeoutError):
            return False
        if isinstance(page, str):
            return bool(page.strip())
        if isinstance(page, dict):
            if page.get("error"):
                return False
            return bool(page.get("slug") or page.get("body") or page.get("content"))
        return False

    def _ensure_node(self, slug: str) -> bool:
        """Ensure a page exists for ``slug`` so ``add_link`` can target it.

        ``add_link`` fails when either endpoint is missing. Missing nodes are
        materialized as a minimal typed stub; confirmed slugs are memoized to
        avoid repeated ``get_page`` probes within a process. The ``put_page``
        result is inspected (gbrain reports failures in-band), so a failed
        creation is not memoized as present.

        Args:
            slug: The entity slug to materialize.

        Returns:
            ``True`` when the node exists (pre-existing or created).
        """
        if not slug:
            return False
        if slug in self._known_nodes:
            return True
        if self._node_exists(slug):
            self._known_nodes.add(slug)
            return True
        try:
            res = self._mcp.call(
                "put_page",
                {
                    "slug": slug,
                    "title": slug,
                    "content": f"# {slug}\n\nKnowledge-graph entity node.\n",
                },
            )
        except (GbrainRemoteError, OSError, TimeoutError) as exc:
            log.warning("kg _ensure_node failed for %s: %s", slug, exc)
            return False
        if _rpc_failed(res):
            log.warning("kg _ensure_node error for %s: %s", slug, res.get("error"))
            return False
        self._known_nodes.add(slug)
        return True

    def query_facts(
        self,
        *,
        subject: Any = None,
        predicate: Any = None,
        object: Any = None,  # noqa: A002 - matches KG API vocabulary
        conditions: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        """Return facts matching the given subject/predicate/object filters.

        Any filter may be ``None`` (match any), a single value, a list, or
        a pipe-delimited alternation for the predicate.

        Args:
            subject: Subject entity filter.
            predicate: Relation filter.
            object: Object entity filter.
            conditions: Property constraints applied to each candidate.
            limit: Maximum facts to return.

        Returns:
            The matching facts (deduplicated, capped at ``limit``).
        """
        if self._native:
            return self._native_query_facts(subject, predicate, object, conditions, limit)

        subj_set = _as_set(subject)
        pred_set = _as_set(predicate)
        obj_set = _as_set(object)

        query_terms: list[str] = []
        for raw in (subject, predicate, object):
            if isinstance(raw, str) and raw and "|" not in raw:
                query_terms.append(raw)
            elif isinstance(raw, (list, tuple, set)) and raw:
                query_terms.append(str(next(iter(raw))))
        query_str = " ".join(query_terms) or "Facts"

        pages = self._search_pages(query_str, self._search_limit)
        seen: set[tuple[str, str, str]] = set()
        out: list[Fact] = []
        for slug, body in pages:
            for fact in parse_facts_fence(body, source_slug=slug):
                if subj_set and fact.subject not in subj_set:
                    continue
                if pred_set and _entity(fact.predicate) not in pred_set:
                    continue
                if obj_set and fact.object not in obj_set:
                    continue
                if not _conditions_match(fact, conditions):
                    continue
                key = (fact.subject, fact.predicate, fact.object)
                if key in seen:
                    continue
                seen.add(key)
                out.append(fact)
                if len(out) >= int(limit):
                    return out
        return out

    def _native_query_facts(
        self, subject: Any, predicate: Any, object: Any, conditions: Any, limit: int  # noqa: A002
    ) -> list[Fact]:
        """Run ``query_facts`` over gbrain's native link graph.

        Anchors on the concrete side of the triple: ``get_links`` when a
        subject is given, otherwise ``get_backlinks`` on the object. Edges are
        mapped to facts and filtered locally. Predicate-only queries cannot be
        served (no global edge scan) and return ``[]``.

        Args:
            subject: Subject filter (slug or list of slugs).
            predicate: Predicate filter.
            object: Object filter (slug or list of slugs).
            conditions: Property constraints.
            limit: Maximum facts.

        Returns:
            The matching facts (deduplicated, capped at ``limit``).
        """
        subj_set = _as_set(subject)
        pred_set = _as_set(predicate)
        obj_set = _as_set(object)

        if subj_set:
            edges = [e for s in sorted(subj_set) for e in self._get_links(s)]
        elif obj_set:
            edges = [e for o in sorted(obj_set) for e in self._get_backlinks(o)]
        else:
            return []

        seen: set[tuple[str, str, str]] = set()
        out: list[Fact] = []
        for edge in edges:
            fact = _edge_to_fact(edge)
            if subj_set and fact.subject not in subj_set:
                continue
            if pred_set and _entity(fact.predicate) not in pred_set:
                continue
            if obj_set and fact.object not in obj_set:
                continue
            if not _conditions_match(fact, conditions):
                continue
            key = (fact.subject, fact.predicate, fact.object)
            if key in seen:
                continue
            seen.add(key)
            out.append(fact)
            if len(out) >= int(limit):
                break
        return out

    def graph_traverse(
        self,
        *,
        start_entity: str,
        predicate_filter: Any = None,
        max_hops: int = 2,
        direction: str = "outbound",
    ) -> list[GraphNode]:
        """Breadth-first traversal from ``start_entity`` over the fact graph.

        Args:
            start_entity: The seed entity.
            predicate_filter: Relations to follow (``None`` = any).
            max_hops: Maximum traversal depth (hard-capped at 3).
            direction: ``outbound`` / ``inbound`` / ``both``.

        Returns:
            The reached nodes (each carrying the fact it was reached by).
        """
        if self._native:
            return self._native_graph_traverse(start_entity, predicate_filter, max_hops, direction)

        max_hops = min(int(max_hops), 3)
        start = _entity(start_entity)
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(start, 0)]
        results: list[GraphNode] = []
        while queue and len(visited) < _MAX_TRAVERSE_NODES:
            entity, depth = queue.pop(0)
            if depth >= max_hops or entity in visited:
                continue
            visited.add(entity)
            facts: list[Fact] = []
            if direction in ("outbound", "both"):
                facts += self.query_facts(subject=entity, predicate=predicate_filter, limit=_MAX_FACTS_PER_PAGE)
            if direction in ("inbound", "both"):
                facts += self.query_facts(object=entity, predicate=predicate_filter, limit=_MAX_FACTS_PER_PAGE)
            for fact in facts:
                nxt = fact.object if fact.subject == entity else fact.subject
                if not nxt or nxt == entity:
                    continue
                results.append(GraphNode(entity=nxt, depth=depth + 1, via=fact))
                if nxt not in visited:
                    queue.append((nxt, depth + 1))
        return results

    def _native_graph_traverse(
        self, start_entity: str, predicate_filter: Any, max_hops: int, direction: str
    ) -> list[GraphNode]:
        """Run ``graph_traverse`` over gbrain's native link graph.

        Maps to ``traverse_graph(slug, depth, direction)``. The ``link_type``
        filter is omitted so multi-relation paths are followed, then edges are
        filtered locally by ``predicate_filter``. Each surviving edge yields a
        :class:`GraphNode` for its far endpoint. Output is capped at
        :data:`_MAX_TRAVERSE_NODES`.

        Args:
            start_entity: Seed entity.
            predicate_filter: Relations to keep (``None`` keeps all).
            max_hops: Traversal depth (hard-capped at 3).
            direction: ``outbound`` / ``inbound`` / ``both``.

        Returns:
            The reached nodes.
        """
        dir_map = {"outbound": "out", "inbound": "in", "both": "both"}
        native_dir = dir_map.get(direction, "out")
        depth = min(int(max_hops), 3)
        start = _entity(start_entity)
        pred_set = _as_set(predicate_filter)

        edges = self._as_edges(
            self._mcp.call("traverse_graph", {"slug": start, "depth": depth, "direction": native_dir})
        )

        out: list[GraphNode] = []
        seen: set[tuple[str, str, int]] = set()
        for edge in edges:
            fact = _edge_to_fact(edge)
            if pred_set and _entity(fact.predicate) not in pred_set:
                continue
            if native_dir == "in":
                nxt = fact.subject
            elif native_dir == "both":
                nxt = fact.object if fact.subject == start else fact.subject
            else:
                nxt = fact.object
            if not nxt or nxt == start:
                continue
            hop = int(edge.get("depth", 1))
            key = (nxt, fact.predicate, hop)
            if key in seen:
                continue
            seen.add(key)
            out.append(GraphNode(entity=nxt, depth=hop, via=fact))
            if len(out) >= _MAX_TRAVERSE_NODES:
                break
        return out

    def find_conflicts(
        self, *, knobs: Sequence[str], hardware: str = "", framework: str = ""
    ) -> list[dict[str, Any]]:
        """Detect ``CONFLICTS_WITH`` relations among a set of knobs.

        Args:
            knobs: Candidate knob/patch entities to check.
            hardware: Optional hardware condition filter.
            framework: Optional framework condition filter.

        Returns:
            One ``{knob, conflicts_with, severity}`` dict per detected
            conflict where both endpoints are in ``knobs``.
        """
        knob_set = {_entity(k) for k in knobs if str(k or "").strip()}
        if len(knob_set) < 2:
            return []
        # CONFLICTS_WITH relations are structural (no hw/fw props), so
        # hardware/framework are accepted for API parity but not applied.
        del hardware, framework
        out: list[dict[str, Any]] = []
        reported: set[frozenset[str]] = set()
        for knob in knob_set:
            facts = self.query_facts(
                subject=knob,
                predicate="CONFLICTS_WITH",
                limit=_MAX_FACTS_PER_PAGE,
            )
            for fact in facts:
                if fact.object not in knob_set:
                    continue
                pair = frozenset({knob, fact.object})
                if pair in reported:
                    continue
                reported.add(pair)
                out.append(
                    {
                        "knob": fact.object,
                        "conflicts_with": knob,
                        "severity": fact.properties.get("severity", "hard"),
                    }
                )
        return out

    # Writes: read-modify-write over get_page/put_page.
    def emit_fact(
        self,
        *,
        page_slug: str,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Append a fact to a page's ``## Facts`` fence (idempotent).

        Reads the page, inserts the formatted line under the existing
        ``## Facts`` header (or creates the fence), and writes it back. A
        duplicate ``(subject, predicate, object)`` line is a no-op.

        Args:
            page_slug: Target page slug.
            subject: Subject entity.
            predicate: Relation.
            object: Object entity.
            properties: Optional fact properties.

        Returns:
            ``True`` when the page was written, ``False`` on no-op.
        """
        if self._native:
            return self._native_emit_fact(subject, predicate, object, properties)
        content = self._page_content_raw(page_slug)
        if content is None:
            return False
        line = format_fact_line(subject, predicate, object, properties)
        existing = {(f.subject, f.predicate, f.object) for f in parse_facts_fence(content)}
        if (_entity(subject), str(predicate).strip().upper(), _entity(object)) in existing:
            return False
        if "## Facts" in content:
            new_content = content.replace("## Facts\n", f"## Facts\n{line}\n", 1)
        else:
            sep = "" if content.endswith("\n") else "\n"
            new_content = f"{content}{sep}\n## Facts\n{line}\n"
        self._mcp.call("put_page", {"slug": page_slug, "content": new_content})
        self._search_cache.clear()
        return True

    def retract_fact(
        self, *, page_slug: str, subject: str, predicate: str, object: str  # noqa: A002
    ) -> bool:
        """Remove a matching fact line from a page's ``## Facts`` fence.

        Args:
            page_slug: Target page slug.
            subject: Subject entity.
            predicate: Relation.
            object: Object entity.

        Returns:
            ``True`` when a line was removed and the page rewritten.
        """
        if self._native:
            return self._native_retract_fact(subject, predicate, object)
        content = self._page_content_raw(page_slug)
        if content is None or "## Facts" not in content:
            return False
        want = (_entity(subject), str(predicate).strip().upper(), _entity(object))
        kept: list[str] = []
        removed = False
        for line in content.splitlines():
            m = _FACT_LINE_RE.match(line.strip())
            if m and (
                _entity(m.group("subject")),
                m.group("predicate").strip().upper(),
                _entity(m.group("object")),
            ) == want:
                removed = True
                continue
            kept.append(line)
        if not removed:
            return False
        new_content = "\n".join(kept)
        if content.endswith("\n"):
            new_content += "\n"
        self._mcp.call("put_page", {"slug": page_slug, "content": new_content})
        self._search_cache.clear()
        return True

    def _native_emit_fact(
        self, subject: str, predicate: str, object: str, properties: dict[str, Any] | None  # noqa: A002
    ) -> bool:
        """Emit a fact as a native link-graph edge.

        Materializes both endpoint nodes (``add_link`` requires them) then
        creates the edge. ``add_link`` is idempotent on ``(from, to,
        link_type)``, so re-emitting upserts the context. The ``add_link``
        result is inspected so an in-band error reports ``False``.

        Args:
            subject: Subject entity.
            predicate: Relation.
            object: Object entity.
            properties: Optional fact properties (encoded into ``context``).

        Returns:
            ``True`` when the edge was written, ``False`` on invalid input
            or a backend error.
        """
        s = _entity(subject)
        o = _entity(object)
        if not s or not o:
            return False
        if not self._ensure_node(s) or not self._ensure_node(o):
            return False
        res = self._mcp.call(
            "add_link",
            {
                "from": s,
                "to": o,
                "link_type": _link_type(predicate),
                "context": _props_to_context(properties),
            },
        )
        if _rpc_failed(res):
            log.warning("kg add_link %s->%s error: %s", s, o, res.get("error"))
            return False
        return True

    def _native_retract_fact(self, subject: str, predicate: str, object: str) -> bool:  # noqa: A002
        """Retract a single typed edge from the native link graph.

        ``remove_link`` deletes *every* edge between the pair, so this reads
        the outgoing edges, removes the pair, and re-adds the survivors that
        had a different ``link_type`` — preserving the other relations on the
        same node pair.

        Args:
            subject: Subject entity.
            predicate: Relation to retract.
            object: Object entity.

        Returns:
            ``True`` when a matching edge was removed, ``False`` otherwise.
        """
        s = _entity(subject)
        o = _entity(object)
        target = _link_type(predicate)
        if not s or not o:
            return False
        pair_edges = [e for e in self._get_links(s) if _entity(e.get("to_slug")) == o]
        if not any(_link_type(e.get("link_type")) == target for e in pair_edges):
            return False
        survivors = [e for e in pair_edges if _link_type(e.get("link_type")) != target]
        res = self._mcp.call("remove_link", {"from": s, "to": o})
        if _rpc_failed(res):
            log.warning("kg remove_link %s->%s error: %s", s, o, res.get("error"))
            return False
        for edge in survivors:
            self._mcp.call(
                "add_link",
                {
                    "from": s,
                    "to": o,
                    "link_type": _link_type(edge.get("link_type")),
                    "context": edge.get("context") or "{}",
                },
            )
        return True

    def _page_content_raw(self, slug: str) -> str | None:
        """Fetch the full raw markdown of a page for read-modify-write.

        Args:
            slug: The page slug.

        Returns:
            The page content, or ``None`` when unavailable.
        """
        if self._mcp is None:
            return None
        try:
            page = self._mcp.call("get_page", {"slug": slug})
        except (GbrainRemoteError, OSError, TimeoutError) as exc:
            log.warning("kg get_page failed for %s: %s", slug, exc)
            return None
        content = _page_content(page, full=True)
        return content or None

    def query_facts_safe(self, **kwargs: Any) -> list[Fact]:
        """Call :meth:`query_facts`, returning ``[]`` on any backend error."""
        try:
            return self.query_facts(**kwargs)
        except (GbrainRemoteError, OSError, TimeoutError, ValueError) as exc:
            log.warning("kg query_facts degraded (returning []): %s", exc)
            return []

    def graph_traverse_safe(self, **kwargs: Any) -> list[GraphNode]:
        """Call :meth:`graph_traverse`, returning ``[]`` on any backend error."""
        try:
            return self.graph_traverse(**kwargs)
        except (GbrainRemoteError, OSError, TimeoutError, ValueError) as exc:
            log.warning("kg graph_traverse degraded (returning []): %s", exc)
            return []

    def find_conflicts_safe(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Call :meth:`find_conflicts`, returning ``[]`` on any backend error."""
        try:
            return self.find_conflicts(**kwargs)
        except (GbrainRemoteError, OSError, TimeoutError, ValueError) as exc:
            log.warning("kg find_conflicts degraded (returning []): %s", exc)
            return []

    def emit_fact_safe(self, **kwargs: Any) -> bool:
        """Call :meth:`emit_fact`, returning ``False`` on any backend error."""
        try:
            return self.emit_fact(**kwargs)
        except (GbrainRemoteError, OSError, TimeoutError, ValueError) as exc:
            log.warning("kg emit_fact degraded (skipped): %s", exc)
            return False

def _page_content(page: Any, *, full: bool = False) -> str:
    """Extract a page's body (or full raw markdown) from a get_page result.

    gbrain may return the page as a raw markdown string, as ``{content}``
    /``{raw}``/``{markdown}``, or as a structured ``{frontmatter, body}``.
    This normalizes those shapes.

    Args:
        page: The raw ``get_page`` result.
        full: When ``True`` return the full frontmatter+body markdown;
            otherwise return only the body (for fact parsing).

    Returns:
        The requested content, or ``""`` when nothing usable is present.
    """
    if isinstance(page, str):
        return page
    if not isinstance(page, dict):
        return ""
    for key in ("content", "raw", "markdown"):
        val = page.get(key)
        if isinstance(val, str) and val.strip():
            return val if full else _strip_frontmatter(val)
    body = page.get("body")
    body = str(body) if body is not None else ""
    if not full:
        return body
    # Full markdown reconstruction: prefer a verbatim frontmatter string,
    # otherwise re-serialize the parsed dict so a read-modify-write never drops
    # ``type``/``tags``/``attrs``.
    fm_raw = page.get("frontmatter_raw")
    if isinstance(fm_raw, str) and fm_raw.strip():
        return f"{fm_raw}\n\n{body}" if body else fm_raw
    fm = page.get("frontmatter")
    if isinstance(fm, dict) and fm:
        fm_block = _serialize_frontmatter(fm)
        return f"{fm_block}\n\n{body}" if body else fm_block
    return body


def _serialize_frontmatter(fm: dict[str, Any]) -> str:
    """Re-serialize a parsed frontmatter dict into a ``---`` YAML block.

    Scalars as ``key: value``, lists as ``key: [a, b]``, nested maps as compact
    JSON. Not byte-identical to the original, but preserves every semantic field.

    Args:
        fm: The parsed frontmatter mapping.

    Returns:
        A ``---``-delimited frontmatter block.
    """
    lines = ["---"]
    for key, val in fm.items():
        if isinstance(val, dict):
            lines.append(f"{key}: {json.dumps(val, ensure_ascii=False, default=str)}")
        elif isinstance(val, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(x) for x in val)}]")
        else:
            lines.append(f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines)


def _strip_frontmatter(content: str) -> str:
    """Return the body portion below a leading ``---`` frontmatter block.

    Args:
        content: Full page markdown.

    Returns:
        The body (content unchanged when no frontmatter delimiter found).
    """
    if not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    return parts[2].lstrip("\n") if len(parts) == 3 else content


def generate_variants_graph_guided(
    kg: KGClient,
    *,
    architectures: Sequence[str],
    hardware: str = "",
    framework: str = "",
    tried: Iterable[str] = (),
    blocked: Iterable[str] = (),
    in_stack: Iterable[str] = (),
    max_variants: int = 8,
) -> list[dict[str, Any]]:
    """Propose optimization knobs from the KG causal graph (not blind search).

    Queries ``IMPROVES*`` facts for the current architecture+hw+fw, drops
    anything already tried / blocked / in the active stack, resolves
    ``REQUIRES`` dependencies, and rejects candidates that ``CONFLICTS_WITH`` an
    in-stack knob. Returns the survivors ordered by expected gain.

    Args:
        kg: The KG client used for fact lookups (degradation-safe).
        architectures: Current model architecture list (the fact object).
        hardware: Hardware condition filter.
        framework: Framework condition filter.
        tried: Knob fingerprints already attempted.
        blocked: Blocked knob entities.
        in_stack: Knobs already present in the optimization stack.
        max_variants: Maximum number of candidates to return.

    Returns:
        A list of ``{knob, expected_gain, confidence, evidence_count,
        source}`` dicts, gain-descending.
    """
    archs = [a for a in (architectures or []) if str(a or "").strip()]
    if not archs:
        return []
    tried_set = {_entity(t) for t in tried}
    blocked_set = {_entity(b) for b in blocked}
    stack_set = {_entity(s) for s in in_stack}
    conditions: dict[str, Any] = {}
    if hardware:
        conditions["hw"] = hardware
    if framework:
        conditions["fw"] = framework

    candidates = kg.query_facts_safe(
        object=list(archs),
        predicate=["IMPROVES", "IMPROVES_TTFT", "IMPROVES_DECODE"],
        conditions=conditions or None,
        limit=_MAX_FACTS_PER_PAGE,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in sorted(candidates, key=lambda f: -f.gain):
        knob = fact.subject
        if not knob or knob in seen or knob in tried_set or knob in blocked_set or knob in stack_set:
            continue
        # REQUIRES: every dependency must already be in the stack.
        deps = kg.query_facts_safe(subject=knob, predicate=["REQUIRES"], limit=_MAX_FACTS_PER_PAGE)
        if any(d.object and d.object not in stack_set for d in deps):
            continue
        # CONFLICTS_WITH: reject if conflicting with an in-stack knob.
        conflicts = kg.query_facts_safe(subject=knob, predicate=["CONFLICTS_WITH"], limit=_MAX_FACTS_PER_PAGE)
        if any(c.object in stack_set for c in conflicts):
            continue
        seen.add(knob)
        out.append({
            "knob": knob,
            "expected_gain": fact.gain,
            "confidence": fact.confidence,
            "evidence_count": int(_pct(fact.properties.get("tested_sessions")) or 1),
            "source": "kg_causal",
        })
        if len(out) >= int(max_variants):
            break
    return out


def _knob_object_nodes(architectures: Sequence[str], precision: str) -> list[str]:
    """Build the ``"{arch}+{precision}"`` object nodes for knob queries.

    Precision is folded into the object node so per-precision knob evidence
    stays distinct; falls back to the bare arch when precision is unknown.

    Args:
        architectures: Current model architecture list.
        precision: The run's baseline precision (e.g. ``"bf16"``/``"fp8"``).

    Returns:
        The list of object node slugs to anchor knob queries on.
    """
    archs = [_entity(a) for a in (architectures or []) if str(a or "").strip()]
    prec = str(precision or "").strip().lower()
    if not prec:
        return archs
    return [f"{a}+{prec}" for a in archs]


def generate_knob_candidates_graph_guided(
    kg: KGClient,
    *,
    architectures: Sequence[str],
    precision: str = "",
    hardware: str = "",
    framework: str = "",
    tried: Iterable[str] = (),
    in_stack: Iterable[str] = (),
    max_variants: int = 8,
) -> list[dict[str, Any]]:
    """Propose runnable config knobs from journal-derived KG knob edges.

    Distinct from :func:`generate_variants_graph_guided`: this reads the
    ``KNOB_IMPROVES`` / ``KNOB_REVERTED_ON`` predicates whose subjects are
    ``variant_fingerprint`` hashes and whose context carries the runnable
    ``args``/``envs``. Candidates with a ``KNOB_REVERTED_ON`` edge for the same
    arch+precision, or already in ``tried`` / ``in_stack``, are dropped.

    Args:
        kg: The KG client (degradation-safe).
        architectures: Current model architecture list.
        precision: Baseline precision condition (folded into the object node).
        hardware: Hardware condition filter.
        framework: Framework condition filter.
        tried: Knob fingerprints already attempted.
        in_stack: Knob fingerprints already in the optimization stack.
        max_variants: Maximum candidates to return.

    Returns:
        A list of ``{knob, args, envs, name, expected_gain, confidence,
        evidence_count, source}`` dicts, gain-descending. ``knob`` is the
        fingerprint; ``args``/``envs`` reconstruct a runnable variant.
    """
    objs = _knob_object_nodes(architectures, precision)
    if not objs:
        return []
    tried_set = {_entity(t) for t in tried}
    stack_set = {_entity(s) for s in in_stack}
    conditions: dict[str, Any] = {}
    if hardware:
        conditions["hw"] = hardware
    if framework:
        conditions["fw"] = framework

    blocked_set = {
        f.subject
        for f in kg.query_facts_safe(
            object=objs, predicate=["KNOB_REVERTED_ON"], limit=_MAX_FACTS_PER_PAGE
        )
    }
    candidates = kg.query_facts_safe(
        object=objs,
        predicate=["KNOB_IMPROVES"],
        conditions=conditions or None,
        limit=_MAX_FACTS_PER_PAGE,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in sorted(candidates, key=lambda f: -f.gain):
        knob = fact.subject
        if not knob or knob in seen or knob in tried_set or knob in blocked_set or knob in stack_set:
            continue
        envs_raw = fact.properties.get("envs")
        envs: dict[str, str] = {}
        if envs_raw:
            try:
                parsed = json.loads(envs_raw)
                if isinstance(parsed, dict):
                    envs = {str(k): str(v) for k, v in parsed.items()}
            except (TypeError, ValueError):
                envs = {}
        keep_n = _pct(fact.properties.get("keep_n")) or 1
        seen.add(knob)
        out.append({
            "knob": knob,
            "args": str(fact.properties.get("args") or ""),
            "envs": envs,
            "name": str(fact.properties.get("name") or ""),
            "expected_gain": fact.gain,
            "confidence": fact.confidence,
            "evidence_count": int(keep_n),
            "source": "kg_knob",
        })
        if len(out) >= int(max_variants):
            break
    return out


def generate_warmstart_donor_graph_guided(
    kg: KGClient,
    *,
    architectures: Sequence[str],
    precision: str = "",
    hardware: str = "",
    framework: str = "",
    model_type: str = "",
) -> dict[str, Any] | None:
    """Synthesize a cross-model warm-start config donor from KG knob edges.

    Reads the cross-model ``KNOB_IMPROVES`` evidence aggregated under the
    ``{arch}+{precision}`` object node and adopts the single highest-evidence,
    positive-gain, non-reverted knob as the donor config (no multi-knob
    composition, so the replayed config was validated as a unit).

    Reuses :func:`generate_knob_candidates_graph_guided` (``max_variants=1``)
    for filtering/ranking, then wraps the top candidate in a recipe-shaped row.

    Args:
        kg: The KG client (degradation-safe).
        architectures: Current model architecture list.
        precision: Baseline precision (folded into the object node).
        hardware: Hardware condition filter.
        framework: Framework condition filter.
        model_type: Target model type (informational, stamped on the donor).

    Returns:
        A recipe-shaped donor row (``best_config`` / ``validated_gain_pct`` /
        ``confidence`` / ``provenance``), or ``None`` when KG carries no usable
        cross-model knob (caller falls back to the recipe-KB sibling search).
    """
    cands = generate_knob_candidates_graph_guided(
        kg,
        architectures=architectures,
        precision=precision,
        hardware=hardware,
        framework=framework,
        max_variants=1,
    )
    if not cands:
        return None
    top = cands[0]
    if float(top.get("expected_gain") or 0.0) <= 0.0:
        return None
    args = str(top.get("args") or "").strip()
    envs = top.get("envs") if isinstance(top.get("envs"), dict) else {}
    if not args and not envs:
        return None
    archs = [str(a) for a in (architectures or []) if str(a or "").strip()]
    prec = str(precision or "").strip().lower()
    obj = "+".join(_entity(a) for a in archs)
    if prec:
        obj = f"{obj}+{prec}"
    return {
        "canonical_id": f"kg-synth:{obj}:{top.get('knob') or ''}",
        "best_config": {"extra_server_args": args, "extra_envs": envs},
        "validated_gain_pct": float(top.get("expected_gain") or 0.0),
        "confidence": float(top.get("confidence") or 0.0),
        "architectures": archs,
        "model_type": str(model_type or ""),
        "provenance": {
            "source": "kg_native_warmstart",
            "knob": top.get("knob") or "",
            "name": top.get("name") or "",
            "evidence_count": int(top.get("evidence_count") or 0),
        },
    }


def build_kg_client_from_env() -> KGClient | None:
    """Construct a :class:`KGClient` from ``GBRAIN_*`` env vars.

    Reuses the same endpoint/token as the gbrain recipe remote. Returns
    ``None`` when the env is not configured so callers degrade silently.

    Returns:
        A configured client, or ``None`` when ``GBRAIN_BASE_URL`` /
        ``GBRAIN_TOKEN`` are unset.
    """
    base_url = (os.environ.get("GBRAIN_BASE_URL", "") or "").strip()
    token = (os.environ.get("GBRAIN_TOKEN", "") or "").strip()
    if not base_url or not token:
        return None
    timeout_env = os.environ.get("GBRAIN_HTTP_TIMEOUT_SEC")
    timeout_sec = 2.0
    if timeout_env:
        try:
            timeout_sec = float(timeout_env)
        except ValueError:
            timeout_sec = 2.0
    use_native = (os.environ.get("GBRAIN_KG_NATIVE", "") or "").strip().lower() in ("1", "true", "yes")
    mcp = _GbrainMcp(base_url, token, timeout_sec)
    return KGClient(mcp, use_native_kg=use_native)


_CACHED_CLIENT: KGClient | None = None
_CLIENT_RESOLVED = False


def get_kg_client() -> KGClient | None:
    """Return a process-wide :class:`KGClient` (lazily built from env).

    The result (including ``None`` when unconfigured) is cached after the
    first call. Use :func:`reset_kg_client` in tests to clear it.

    Returns:
        The cached client, or ``None`` when KG is not configured.
    """
    global _CACHED_CLIENT, _CLIENT_RESOLVED
    if not _CLIENT_RESOLVED:
        _CACHED_CLIENT = build_kg_client_from_env()
        _CLIENT_RESOLVED = True
    return _CACHED_CLIENT


def reset_kg_client() -> None:
    """Clear the cached client (test helper)."""
    global _CACHED_CLIENT, _CLIENT_RESOLVED
    _CACHED_CLIENT = None
    _CLIENT_RESOLVED = False


__all__ = [
    "Fact",
    "GraphNode",
    "KGClient",
    "parse_facts_fence",
    "format_fact_line",
    "generate_variants_graph_guided",
    "generate_knob_candidates_graph_guided",
    "build_kg_client_from_env",
    "get_kg_client",
    "reset_kg_client",
]
