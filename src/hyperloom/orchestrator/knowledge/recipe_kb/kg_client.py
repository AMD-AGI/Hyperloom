# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-Graph query client over local or GBrain page stores.

All backends expose the same native link-graph API (``get_links`` /
``get_backlinks`` / ``traverse_graph`` / ``add_link``).  A triple
``(subject, predicate, object)`` maps to an edge
``add_link(from=subject, to=object, link_type=predicate,
context=json(properties))``.  Fact properties are encoded in the edge
``context`` JSON.  gbrain reports some write failures as an in-band
``{"error": ...}`` payload, so write paths inspect the decoded result via
:func:`_rpc_failed`.

Every read method has a ``*_safe`` companion that swallows transport failures
and returns an empty result, so the KG layer can never block or break a
warm-start that would otherwise succeed from the local store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import KnowledgeConfig, KnowledgeStoreMode
from .gbrain_mcp import GbrainRemoteError, _GbrainMcp
from .local_graph_store import LocalGraphStore

log = logging.getLogger(__name__)

# Cap the number of fact lines parsed per page (mirrors the mirror-side
# write cap) so a malformed/oversized page never blows up a query tick.
_MAX_FACTS_PER_PAGE = 50
# Hard ceiling on BFS breadth to bound native traversal cost.
_MAX_TRAVERSE_NODES = 200


def _entity(value: Any) -> str:
    """Normalize an entity token the same way the mirror writer does.

    Args:
        value: Raw entity name.

    Returns:
        Lowercased slug (spaces/slashes to underscores), or ``""``.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    slug = re.sub(r"[^a-z0-9._+-]+", "_", raw)
    if not slug or not any(character.isalnum() for character in slug):
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"entity_{digest}"
    if not slug[0].isalnum():
        slug = f"entity_{slug}"
    if len(slug) > 128:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        slug = f"{slug[:115]}-{digest}"
    return slug


def _legacy_entity(value: Any) -> str:
    """Return the pre-Phase-1 entity slug used by existing remote KG nodes."""
    return str(value or "").strip().replace(" ", "_").replace("/", "_").lower()


def _entity_aliases(value: Any) -> tuple[str, ...]:
    """Return current and legacy slugs, preserving order and uniqueness."""
    aliases: list[str] = []
    for candidate in (_entity(value), _legacy_entity(value)):
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return tuple(aliases)


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
    return {
        alias
        for item in items
        if str(item or "").strip()
        for alias in _entity_aliases(item)
    }



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
    can faithfully reconstruct it. Empty-valued properties are dropped;
    :func:`_context_to_props` is the inverse used on the read path.

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
    """Knowledge-graph query surface over a local or GBrain page store."""

    def __init__(self, mcp: Any) -> None:
        """Initialize the client.

        Args:
            mcp: A duck-typed backend adapter exposing
                ``call(tool, arguments)``. The remote implementation uses MCP;
                the local implementation is an in-process filesystem adapter.
        """
        self._mcp = mcp
        # Slugs confirmed to exist as pages, memoized by the write path.
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
        """Return outgoing edges for ``slug``."""
        if self._mcp is None or not slug:
            return []
        return self._as_edges(self._mcp.call("get_links", {"slug": slug}))

    def _get_backlinks(self, slug: str) -> list[dict[str, Any]]:
        """Return incoming edges for ``slug``."""
        if self._mcp is None or not slug:
            return []
        return self._as_edges(self._mcp.call("get_backlinks", {"slug": slug}))

    def _node_exists(self, slug: str) -> bool:
        """Return ``True`` when a page for ``slug`` exists, ``False`` when it
        is confirmed absent.  Raises on transport or I/O failures so callers
        cannot mistake a failed probe for a missing node.
        """
        page = self._mcp.call("get_page", {"slug": slug})
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

        A transport or I/O failure from the existence probe is treated as
        "cannot confirm" — no write is issued and ``False`` is returned so the
        caller's ``_safe`` wrapper can surface the failure faithfully.

        Args:
            slug: The entity slug to materialize.

        Returns:
            ``True`` when the node exists (pre-existing or created).
        """
        if not slug:
            return False
        if slug in self._known_nodes:
            return True
        try:
            exists = self._node_exists(slug)
        except (GbrainRemoteError, OSError, TimeoutError) as exc:
            log.warning("kg _ensure_node probe failed for %s: %s", slug, exc)
            return False
        if exists:
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

        Anchors on the concrete side of the triple: ``get_links`` when a
        subject is given, otherwise ``get_backlinks`` on the object. Edges are
        mapped to facts and filtered locally. Predicate-only queries cannot be
        served (no global edge scan) and return ``[]``.

        Any filter may be ``None`` (match any), a single value, or a list.

        Args:
            subject: Subject entity filter.
            predicate: Relation filter.
            object: Object entity filter.
            conditions: Property constraints applied to each candidate.
            limit: Maximum facts to return.

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
        """Traversal from ``start_entity`` via the native link graph.

        Maps to ``traverse_graph(slug, depth, direction)``.  The ``link_type``
        filter is omitted on the wire so multi-relation paths are followed;
        edges are filtered locally by ``predicate_filter``.  Output is capped
        at :data:`_MAX_TRAVERSE_NODES`.

        Args:
            start_entity: The seed entity.
            predicate_filter: Relations to keep (``None`` keeps all).
            max_hops: Maximum traversal depth (hard-capped at 3).
            direction: ``outbound`` / ``inbound`` / ``both``.

        Returns:
            The reached nodes (each carrying the fact it was reached by).
        """
        dir_map = {"outbound": "out", "inbound": "in", "both": "both"}
        native_dir = dir_map.get(direction, "out")
        depth = min(int(max_hops), 3)
        start = _entity(start_entity)
        pred_set = _as_set(predicate_filter)

        edges = [
            edge
            for alias in _entity_aliases(start_entity)
            for edge in self._as_edges(
                self._mcp.call(
                    "traverse_graph",
                    {"slug": alias, "depth": depth, "direction": native_dir},
                )
            )
        ]

        out: list[GraphNode] = []
        seen: set[tuple[str, str, int]] = set()
        discovered = {start}
        for edge in sorted(edges, key=lambda row: int(row.get("depth", 1))):
            fact = _edge_to_fact(edge)
            if pred_set and _entity(fact.predicate) not in pred_set:
                continue
            if native_dir == "in":
                nxt = fact.subject
            elif native_dir == "both":
                if fact.subject in discovered and fact.object not in discovered:
                    nxt = fact.object
                elif fact.object in discovered and fact.subject not in discovered:
                    nxt = fact.subject
                else:
                    nxt = fact.object if fact.subject == start else fact.subject
            else:
                nxt = fact.object
            if not nxt or nxt == start:
                continue
            hop = int(edge.get("depth", 1))
            discovered.add(nxt)
            key = (nxt, fact.predicate, hop)
            if key in seen:
                continue
            seen.add(key)
            out.append(GraphNode(entity=nxt, depth=hop, via=fact))
            if len(out) >= _MAX_TRAVERSE_NODES:
                break
        return out

    def find_conflicts(self, *, knobs: Sequence[str], hardware: str = "", framework: str = "") -> list[dict[str, Any]]:
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

    def emit_fact(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Emit a fact as a native link-graph edge (idempotent).

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

    def _degrade_safe(self, method: Callable[..., Any], default: Any, **kwargs: Any) -> Any:
        """Call *method*, returning *default* on any known backend error."""
        try:
            return method(**kwargs)
        except (GbrainRemoteError, OSError, TimeoutError, ValueError) as exc:
            log.warning("kg %s degraded (returning default): %s", getattr(method, "__name__", "call"), exc)
            return default

    def query_facts_safe(self, **kwargs: Any) -> list[Fact]:
        """Call :meth:`query_facts`, returning ``[]`` on any backend error."""
        return self._degrade_safe(self.query_facts, [], **kwargs)

    def graph_traverse_safe(self, **kwargs: Any) -> list[GraphNode]:
        """Call :meth:`graph_traverse`, returning ``[]`` on any backend error."""
        return self._degrade_safe(self.graph_traverse, [], **kwargs)

    def find_conflicts_safe(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Call :meth:`find_conflicts`, returning ``[]`` on any backend error."""
        return self._degrade_safe(self.find_conflicts, [], **kwargs)

    def emit_fact_safe(self, **kwargs: Any) -> bool:
        """Call :meth:`emit_fact`, returning ``False`` on any backend error."""
        return self._degrade_safe(self.emit_fact, False, **kwargs)



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
        out.append(
            {
                "knob": knob,
                "expected_gain": fact.gain,
                "confidence": fact.confidence,
                "evidence_count": int(_pct(fact.properties.get("tested_sessions")) or 1),
                "source": "kg_causal",
            }
        )
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
        f.subject for f in kg.query_facts_safe(object=objs, predicate=["KNOB_REVERTED_ON"], limit=_MAX_FACTS_PER_PAGE)
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
        out.append(
            {
                "knob": knob,
                "args": str(fact.properties.get("args") or ""),
                "envs": envs,
                "name": str(fact.properties.get("name") or ""),
                "expected_gain": fact.gain,
                "confidence": fact.confidence,
                "evidence_count": int(keep_n),
                "source": "kg_knob",
            }
        )
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
    """Construct a :class:`KGClient` from the shared ``KnowledgeConfig``.

    Local mode uses an in-process :class:`LocalGraphStore` rooted at
    ``$KNOWLEDGE_LOCAL_ROOT/hyperloom/kg``.  Remote mode uses the optional
    GBrain transport when both ``GBRAIN_BASE_URL`` and ``GBRAIN_TOKEN`` are
    configured.  Ambient GBrain credentials are deliberately ignored in local
    mode.

    Returns:
        A configured client, or ``None`` when GBrain is not configured in
        remote mode.
    """
    config = KnowledgeConfig.from_env()
    if config.mode is KnowledgeStoreMode.LOCAL:
        graph_root = Path(config.local_root) / "hyperloom" / "kg"
        return KGClient(LocalGraphStore(graph_root))

    if not config.gbrain_base_url or not config.gbrain_token:
        log.info(
            "remote Recipe mode has no complete optional GBrain KG "
            "configuration; KG disabled"
        )
        return None

    timeout_env = os.environ.get("GBRAIN_HTTP_TIMEOUT_SEC")
    timeout_sec = 2.0
    if timeout_env:
        try:
            timeout_sec = float(timeout_env)
        except ValueError:
            timeout_sec = 2.0
    mcp = _GbrainMcp(config.gbrain_base_url, config.gbrain_token, timeout_sec)
    return KGClient(mcp)


_CACHED_CLIENT: KGClient | None = None
_CLIENT_RESOLVED = False


def get_kg_client() -> KGClient | None:
    """Return a process-wide :class:`KGClient` (lazily built from env).

    The selected local or remote client is cached after the first call. Use
    :func:`reset_kg_client` in tests after changing knowledge configuration.

    Returns:
        The cached client. ``None`` remains possible only for an injected
        compatibility factory.
    """
    global _CACHED_CLIENT, _CLIENT_RESOLVED
    if not _CLIENT_RESOLVED:
        try:
            _CACHED_CLIENT = build_kg_client_from_env()
        except Exception as exc:  # noqa: BLE001 - KG enhancement is optional
            _CACHED_CLIENT = None
            log.warning("knowledge graph is unavailable for this process: %s", exc)
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
    "generate_variants_graph_guided",
    "generate_knob_candidates_graph_guided",
    "build_kg_client_from_env",
    "get_kg_client",
    "reset_kg_client",
]
