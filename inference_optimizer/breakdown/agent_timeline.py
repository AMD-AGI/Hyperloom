# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Build the unified three-actor ``agent_timeline`` section.

Merges the three decision histories that already live in a
``session_breakdown.json`` onto a single chronological axis:

* **orchestrator** decisions  ← ``decision_trace.decision_trace[]``
* **specialist** proposals    ← ``specialist_runs[]``
* **critic** reviews          ← ``critic_robustness.critic_iterations[]``

Each source record is normalised to a flat :class:`TimelineEvent`
(``ts / actor / kind / title / detail / source``) and the merged list is
sorted by ``(ts, seq)``. The builder is a **pure projection**: it never
touches the network or disk, tolerates missing/partial sections (marking the
result ``degraded`` with a warning), and keeps original fields verbatim under
``detail`` so no information is lost.

The same builder serves both paths:

* **local** — called with the breakdown dict produced in-session;
* **langfuse** — called with the breakdown recovered from a trace's root
  output (see :mod:`inference_optimizer.orchestrator.trace.langfuse_reader`),
  which is how the upload path enriches the SBD without a live session dir.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

#: Wire-shape version of the ``agent_timeline`` section.
AGENT_TIMELINE_SCHEMA = "hyperloom.agent_timeline.v1"

# Far-future sentinel so events without a parseable timestamp sort last
# (stable, after every real event) instead of crashing the sort.
_TS_MAX = datetime(9999, 12, 31, tzinfo=timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    """Best-effort parse of an ISO-8601 timestamp into an aware ``datetime``.

    Accepts the trailing-``Z`` form and explicit offsets; naive timestamps are
    assumed UTC. Returns ``None`` for anything unparseable so the caller can
    fall back to the sequence index.

    Args:
        value: Candidate timestamp (typically a string).

    Returns:
        An aware ``datetime`` in UTC, or ``None`` when not parseable.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    """Return the first non-``None`` value among ``keys`` (schema drift guard).

    Args:
        mapping: Source dict.
        *keys: Candidate keys, tried in order.

    Returns:
        The first present, non-``None`` value, else ``None``.
    """
    for key in keys:
        val = mapping.get(key)
        if val is not None:
            return val
    return None


def _orchestrator_events(breakdown: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    """Project ``decision_trace.decision_trace[]`` into orchestrator events.

    Args:
        breakdown: The full breakdown dict.
        warnings: Mutable list to append non-fatal notes to.

    Returns:
        A list of partial :class:`TimelineEvent` dicts (``seq`` not yet set).
    """
    section = breakdown.get("decision_trace")
    if not isinstance(section, dict):
        warnings.append("decision_trace section missing or not an object")
        return []
    rows = section.get("decision_trace")
    if not isinstance(rows, list):
        warnings.append("decision_trace.decision_trace[] missing or not a list")
        return []

    events: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        dec = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        outcome = dec.get("outcome")
        change = _first(dec, "change", "event", "verdict") or ""
        title = " ".join(p for p in (str(outcome or "").strip(), str(change).strip()) if p)
        events.append(
            {
                "ts": row.get("ts"),
                "actor": "orchestrator",
                "kind": "decision",
                "phase": row.get("phase"),
                "tick": row.get("tick"),
                "title": title or "decision",
                "detail": {
                    "outcome": outcome,
                    "change": change or None,
                    "component": dec.get("component"),
                    "operation_kind": dec.get("operation_kind"),
                    "kind": dec.get("kind"),
                    "gain_pct": dec.get("gain_pct"),
                    "task_id": _first(dec, "task_id", "dyn_id"),
                    "variant_name": dec.get("variant_name"),
                    "provenance": dec.get("provenance"),
                    "tokens": row.get("tokens"),
                },
                "source": {"section": "decision_trace.decision_trace", "index": idx},
            }
        )
    return events


def _specialist_events(breakdown: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    """Project ``specialist_runs[]`` into specialist proposal events.

    Defensive about schema drift: production runs carry per-run fields
    (``domain`` / ``confidence`` / ``new_findings`` / ``ensemble_scores``)
    beyond the documented :class:`SpecialistRound`, so every field is read with
    fallbacks and kept verbatim under ``detail``.

    Args:
        breakdown: The full breakdown dict.
        warnings: Mutable list to append non-fatal notes to.

    Returns:
        A list of partial :class:`TimelineEvent` dicts (``seq`` not yet set).
    """
    rows = breakdown.get("specialist_runs")
    if not isinstance(rows, list):
        warnings.append("specialist_runs[] missing or not a list")
        return []

    events: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        domains = row.get("domains") if isinstance(row.get("domains"), list) else None
        domain = _first(row, "domain") or (domains[0] if domains else None)
        domain_label = domain or (", ".join(str(d) for d in domains) if domains else "specialist")
        confidence = _first(row, "confidence", "confidence_avg")
        title = str(domain_label)
        if confidence is not None:
            title = f"{title} (conf {confidence})"
        events.append(
            {
                "ts": _first(row, "completed_at", "dispatched_at"),
                "actor": "specialist",
                "kind": "proposal",
                "phase": row.get("phase"),
                "tick": row.get("tick"),
                "title": title,
                "detail": {
                    "round_id": row.get("round_id"),
                    "domain": domain,
                    "domains": domains,
                    "confidence": confidence,
                    "empty": row.get("empty"),
                    "gap_canonical_id": row.get("gap_canonical_id"),
                    "ensemble_scores": row.get("ensemble_scores"),
                    "new_findings": row.get("new_findings"),
                    "proposals_total": row.get("proposals_total"),
                    "proposals_kept": row.get("proposals_kept"),
                    "proposals_rejected": row.get("proposals_rejected"),
                    "summary": row.get("summary"),
                },
                "source": {"section": "specialist_runs", "index": idx},
            }
        )
    return events


def _critic_events(breakdown: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    """Project ``critic_robustness.critic_iterations[]`` into critic events.

    Args:
        breakdown: The full breakdown dict.
        warnings: Mutable list to append non-fatal notes to.

    Returns:
        A list of partial :class:`TimelineEvent` dicts (``seq`` not yet set).
    """
    section = breakdown.get("critic_robustness")
    if not isinstance(section, dict):
        warnings.append("critic_robustness section missing or not an object")
        return []
    rows = section.get("critic_iterations")
    if not isinstance(rows, list):
        warnings.append("critic_robustness.critic_iterations[] missing or not a list")
        return []

    events: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        verdict = row.get("verdict")
        topic = row.get("topic")
        title = f"verdict {verdict}" if verdict else "review"
        if topic:
            title = f"{title}: {topic}"
        events.append(
            {
                "ts": row.get("ts"),
                "actor": "critic",
                "kind": "review",
                "phase": row.get("phase"),
                "tick": row.get("tick"),
                "title": title,
                "detail": {
                    "iter": row.get("iter"),
                    "verdict": verdict,
                    "topic": topic,
                    "summary": row.get("summary"),
                    "kb_assess": row.get("kb_assess"),
                    "kb_priors": row.get("kb_priors"),
                },
                "source": {"section": "critic_robustness.critic_iterations", "index": idx},
            }
        )
    return events


def build_agent_timeline(
    breakdown: dict[str, Any],
    *,
    source: str = "local",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Merge the three decision histories into one sorted ``agent_timeline``.

    Pulls orchestrator decisions, specialist proposals, and critic reviews out
    of ``breakdown`` (see module docstring for the exact source sections),
    normalises each to a :class:`TimelineEvent`, and returns them sorted by
    ``(ts, seq)``. Never raises on partial input — missing sections add a
    warning and set ``degraded``.

    Args:
        breakdown: A ``session_breakdown.json`` dict (built locally or
            recovered from a Langfuse trace's root output).
        source: Provenance label stamped onto the result
            (``"local"`` or ``"langfuse"``).
        trace_id: Langfuse trace id to record (and stamp onto every event's
            ``source``) when the breakdown was fetched from a trace.

    Returns:
        An ``agent_timeline`` dict matching
        :class:`inference_optimizer.breakdown.schema.AgentTimeline`.
    """
    warnings: list[str] = []
    raw = (
        _orchestrator_events(breakdown, warnings)
        + _specialist_events(breakdown, warnings)
        + _critic_events(breakdown, warnings)
    )

    # Stable sort: timestamp first (unparseable -> last), original build order
    # as the tiebreaker so same-second events keep a deterministic sequence.
    indexed = list(enumerate(raw))
    indexed.sort(key=lambda pair: (_parse_ts(pair[1].get("ts")) or _TS_MAX, pair[0]))

    events: list[dict[str, Any]] = []
    for seq, (_, ev) in enumerate(indexed):
        ev["seq"] = seq
        if trace_id:
            ev["source"]["trace_id"] = trace_id
        events.append(ev)

    counts = {
        "orchestrator": sum(1 for e in events if e["actor"] == "orchestrator"),
        "specialist": sum(1 for e in events if e["actor"] == "specialist"),
        "critic": sum(1 for e in events if e["actor"] == "critic"),
    }
    degraded = bool(warnings) or not events

    timeline: dict[str, Any] = {
        "schema": AGENT_TIMELINE_SCHEMA,
        "source": source,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": events,
        "counts": counts,
        "degraded": degraded,
        "warnings": warnings,
    }
    if trace_id:
        timeline["trace_id"] = trace_id
    return timeline


def _trace_seed_from_breakdown(breakdown: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract an explicit trace id and a correlation seed from a breakdown.

    Prefers the trace id recorded in the ``langfuse`` push receipt; otherwise
    falls back to the same seed chain the writer hashed into the trace id —
    ``session.claw_session_id`` -> ``session.session_id`` -> the
    ``session.session_dir`` basename (the writer's ``correlation_seed``
    final fallback is the session-dir name).

    Args:
        breakdown: A ``session_breakdown.json`` dict.

    Returns:
        A ``(trace_id, seed)`` tuple; either element may be ``None``.
    """
    lf = breakdown.get("langfuse")
    trace_id = lf.get("trace_id") if isinstance(lf, dict) else None
    session = breakdown.get("session") if isinstance(breakdown.get("session"), dict) else {}
    seed = _first(session, "claw_session_id", "session_id")
    if not seed:
        session_dir = session.get("session_dir")
        if session_dir:
            seed = PurePosixPath(str(session_dir).rstrip("/")).name or None
    return (str(trace_id).strip() if trace_id else None), (str(seed).strip() if seed else None)


def _has_populated_timeline(breakdown: dict[str, Any], key: str) -> bool:
    """Whether ``breakdown[key]`` already holds a timeline with events.

    Guards the upload-time enrich path from clobbering a good locally-built
    timeline with a ``degraded`` empty one when Langfuse recovery fails.

    Args:
        breakdown: The breakdown dict.
        key: Timeline section key (e.g. ``"agent_timeline"``).

    Returns:
        True when a non-empty events list is already present.
    """
    existing = breakdown.get(key)
    return isinstance(existing, dict) and bool(existing.get("events"))


def empty_timeline(*, source: str = "langfuse", reason: str = "") -> dict[str, Any]:
    """Return a well-formed but empty, ``degraded`` ``agent_timeline``.

    Used when the source breakdown could not be recovered from Langfuse, so the
    field is always present and self-describing rather than silently absent.

    Args:
        source: Provenance label to stamp.
        reason: Human-readable warning explaining why it is empty.

    Returns:
        An ``agent_timeline`` dict with no events and ``degraded=True``.
    """
    return {
        "schema": AGENT_TIMELINE_SCHEMA,
        "source": source,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": [],
        "counts": {"orchestrator": 0, "specialist": 0, "critic": 0},
        "degraded": True,
        "warnings": [reason] if reason else ["no events recovered"],
    }


def enrich_breakdown_with_langfuse_timeline(
    breakdown: dict[str, Any],
    *,
    trace_id: str | None = None,
    seed: str | None = None,
    credentials: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch the trace's breakdown from Langfuse, build, and inject the timeline.

    This is the upload-path entry point (the "trigger in the SBD upload
    interface"): it reads the three decision histories back from the session's
    Langfuse trace and writes the merged ``agent_timeline`` onto ``breakdown``
    **in place** (also returned for convenience). Best-effort — on any failure
    it injects a ``degraded`` empty timeline so the field always exists.

    Args:
        breakdown: The ``session_breakdown.json`` dict to enrich in place.
        trace_id: Explicit trace id; defaults to one resolved from
            ``breakdown.langfuse.trace_id`` / ``session.claw_session_id``.
        seed: Explicit correlation seed override.
        credentials: Optional explicit Langfuse credentials override.
        timeout: Per-request timeout in seconds.

    Returns:
        The same ``breakdown`` dict, with ``breakdown["agent_timeline"]`` set.
    """
    # Lazy import: keeps the pure builder importable without the trace package
    # (and avoids any import cycle at module load).
    from ..orchestrator.trace.langfuse_reader import fetch_session_breakdown, resolve_trace_id

    # Fill-if-missing. The exporter already builds a local ``agent_timeline``
    # that is consistent with *this file's own* decision sections, so never
    # overwrite a populated one — a Langfuse re-derive could be stale (the
    # trace snapshot predates a later on-disk refresh) or empty. Only
    # (re)derive from Langfuse when the section is absent or has no events
    # (e.g. an older breakdown produced before local population existed).
    if _has_populated_timeline(breakdown, "agent_timeline"):
        return breakdown

    bd_trace_id, bd_seed = _trace_seed_from_breakdown(breakdown)
    resolved = resolve_trace_id(trace_id=trace_id or bd_trace_id, seed=seed or bd_seed)
    if not resolved:
        breakdown["agent_timeline"] = empty_timeline(reason="no trace_id / correlation seed on breakdown")
        return breakdown

    source_breakdown = fetch_session_breakdown(
        trace_id=resolved, credentials=credentials, timeout=timeout
    )
    if source_breakdown is None:
        breakdown["agent_timeline"] = empty_timeline(reason=f"could not fetch trace {resolved} from langfuse")
        return breakdown

    breakdown["agent_timeline"] = build_agent_timeline(
        source_breakdown, source="langfuse", trace_id=resolved
    )
    return breakdown


__all__ = [
    "AGENT_TIMELINE_SCHEMA",
    "build_agent_timeline",
    "empty_timeline",
    "enrich_breakdown_with_langfuse_timeline",
]
