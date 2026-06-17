# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Build the knowledge-base decision timeline (``kb_timeline``).

Merges the KB **read/use** decisions that a session made onto one
chronological axis:

* ``recipe_read``   — recipe lookups / canonical-id matches
* ``warm_start``    — T0 warm-start recipe selection
* ``warm_replay``   — warm-recipe replay accept / reject / promote
* ``critic_assess`` — critic ``/assess`` substrate reads (per review)
* ``critic_priors`` — critic historical-prior reads (per review)

Write-side KB activity (committing lessons/pitfalls back to the recipe KB,
critic ``commit-review`` upserts) is **intentionally excluded**: it is
session-close bookkeeping rather than a decision, and is not timestamped
anywhere (neither locally nor in Langfuse).

Two input shapes are supported, mirroring the ``agent_timeline`` pattern:

* **breakdown sections** (local build / trace ``output``): recipe reads from
  ``kb_provenance.recipe_snapshot_reads.tail`` and critic reads from
  ``critic_robustness.critic_iterations[].kb_assess`` / ``kb_priors``;
* **Langfuse observations** (spans): the per-event ``kb:recipe_snapshot:*`` /
  ``kb_assess:iter_N`` / ``kb_priors:iter_N`` spans, which carry the *full*
  read history (the breakdown ``tail`` is capped). When ``observations`` is
  given it is preferred for these categories.

``warm_start`` / ``warm_replay`` always come from ``kb_provenance`` (they have
no per-event span).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

#: Wire-shape version of the ``kb_timeline`` section.
KB_TIMELINE_SCHEMA = "hyperloom.kb_timeline.v1"

_TS_MAX = datetime(9999, 12, 31, tzinfo=timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    """Best-effort parse of an ISO-8601 timestamp into an aware ``datetime``.

    Args:
        value: Candidate timestamp (string).

    Returns:
        An aware UTC ``datetime``, or ``None`` when not parseable.
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
    """Return the first non-``None`` value among ``keys``.

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


def _warm_events(kbp: dict[str, Any]) -> list[dict[str, Any]]:
    """Project warm-start + warm-replay decisions from ``kb_provenance``.

    Args:
        kbp: The ``kb_provenance`` section dict.

    Returns:
        Zero to two partial KB timeline events (``seq`` not yet set).
    """
    events: list[dict[str, Any]] = []
    warm_ts = kbp.get("warm_start_ts")

    seen = kbp.get("warm_start_recipe_seen")
    if seen is not None or warm_ts:
        tier = kbp.get("warm_start_recipe_tier")
        title = f"warm-start: recipe tier {tier}" if seen else "warm-start: no recipe matched"
        events.append(
            {
                "ts": warm_ts,
                "category": "warm_start",
                "title": title,
                "detail": {
                    "recipe_seen": seen,
                    "tier": tier,
                    "pitfall_count": kbp.get("warm_start_pitfall_count"),
                    "lesson_count": kbp.get("warm_start_lesson_count"),
                    "warm_history_injected": kbp.get("warm_history_injected"),
                    "cortex_session_id": kbp.get("cortex_session_id"),
                },
                "source": {"section": "kb_provenance"},
            }
        )

    replay = kbp.get("warm_replay") if isinstance(kbp.get("warm_replay"), dict) else {}
    if kbp.get("warm_replay_attempted") or replay:
        status = replay.get("status")
        events.append(
            {
                # warm_replay carries no own ts; anchor near warm-start.
                "ts": replay.get("ts") or warm_ts,
                "category": "warm_replay",
                "title": f"warm-replay: {status or 'attempted'}",
                "detail": {
                    "status": status,
                    "expected_gain_pct": replay.get("expected_gain_pct"),
                    "actual_gain_pct": replay.get("actual_gain_pct"),
                    "warm_recipe_tier": replay.get("warm_recipe_tier"),
                    "warm_recipe_conf": replay.get("warm_recipe_conf"),
                    "replay_task_id": replay.get("replay_task_id"),
                    "reason": replay.get("reason"),
                    "attempted": kbp.get("warm_replay_attempted"),
                },
                "source": {"section": "kb_provenance.warm_replay"},
            }
        )
    return events


def _recipe_read_event(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Project one recipe-snapshot read row into a ``recipe_read`` event.

    Args:
        row: One ``recipe_snapshot_reads.tail[]`` audit row.
        idx: Index of the row (for provenance).

    Returns:
        A partial KB timeline event (``seq`` not yet set).
    """
    method = row.get("method")
    resolution = row.get("resolution")
    hit = row.get("hit")
    request = row.get("request") if isinstance(row.get("request"), dict) else {}
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    return {
        "ts": row.get("ts"),
        "category": "recipe_read",
        "title": f"recipe {method or 'read'}: {resolution or '?'} ({'hit' if hit else 'miss'})",
        "detail": {
            "method": method,
            "remote": row.get("remote"),
            "resolution": resolution,
            "hit": hit,
            "candidates": row.get("candidates"),
            "canonical_id": request.get("canonical_id") if request else result.get("canonical_id"),
            "result": result or None,
        },
        "source": {"section": "kb_provenance.recipe_snapshot_reads.tail", "index": idx},
    }


def _critic_kb_events(iterations: list[Any], warnings: list[str]) -> list[dict[str, Any]]:
    """Project ``critic_iterations[].kb_assess`` / ``kb_priors`` into events.

    Args:
        iterations: The ``critic_robustness.critic_iterations`` list.
        warnings: Mutable list to append non-fatal notes to.

    Returns:
        Partial ``critic_assess`` / ``critic_priors`` events (``seq`` unset).
    """
    events: list[dict[str, Any]] = []
    for idx, it in enumerate(iterations):
        if not isinstance(it, dict):
            continue
        ts = it.get("ts")
        itr = it.get("iter")
        assess = it.get("kb_assess") if isinstance(it.get("kb_assess"), dict) else None
        if assess and assess.get("configured"):
            ref = assess.get("referenced_in_verdict")
            events.append(
                {
                    "ts": ts,
                    "category": "critic_assess",
                    "title": f"kb_assess iter {itr}: {'used' if ref else 'not used'}",
                    "detail": {
                        "iter": itr,
                        "mode": assess.get("mode"),
                        "verdict_count": assess.get("verdict_count"),
                        "referenced_in_verdict": ref,
                        "skipped_reason": assess.get("skipped_reason"),
                    },
                    "source": {"section": "critic_robustness.critic_iterations", "index": idx},
                }
            )
        priors = it.get("kb_priors") if isinstance(it.get("kb_priors"), dict) else None
        if priors and priors.get("configured"):
            ref = priors.get("referenced_in_verdict")
            events.append(
                {
                    "ts": ts,
                    "category": "critic_priors",
                    "title": f"kb_priors iter {itr}: {'used' if ref else 'not used'}",
                    "detail": {
                        "iter": itr,
                        "mode": priors.get("mode"),
                        "prior_count": priors.get("prior_count"),
                        "referenced_in_verdict": ref,
                        "skipped_reason": priors.get("skipped_reason"),
                    },
                    "source": {"section": "critic_robustness.critic_iterations", "index": idx},
                }
            )
    return events


def _obs_ts(obs: dict[str, Any]) -> Any:
    """Read an observation's start timestamp (camelCase or snake_case)."""
    return _first(obs, "startTime", "start_time", "timestamp", "ts")


def _events_from_observations(observations: list[Any], warnings: list[str]) -> list[dict[str, Any]]:
    """Project Langfuse KB spans into recipe_read / critic_assess / critic_priors.

    Recognises the three KB span name patterns emitted by the writer:
    ``kb:recipe_snapshot:{method}`` / ``kb_assess:iter_N`` / ``kb_priors:iter_N``.

    Args:
        observations: List of Langfuse observation dicts for the trace.
        warnings: Mutable list to append non-fatal notes to.

    Returns:
        Partial KB timeline events (``seq`` not yet set).
    """
    events: list[dict[str, Any]] = []
    for idx, obs in enumerate(observations):
        if not isinstance(obs, dict):
            continue
        name = str(obs.get("name") or "")
        meta = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
        oid = obs.get("id")
        ts = _obs_ts(obs)
        if name.startswith("kb:recipe_snapshot:"):
            method = name.split(":", 2)[-1]
            events.append(
                {
                    "ts": ts,
                    "category": "recipe_read",
                    "title": f"recipe {method}: {meta.get('resolution') or '?'} "
                    f"({'hit' if meta.get('hit') else 'miss'})",
                    "detail": {
                        "method": meta.get("method") or method,
                        "remote": meta.get("remote"),
                        "resolution": meta.get("resolution"),
                        "hit": meta.get("hit"),
                    },
                    "source": {"span": name, "observation_id": oid},
                }
            )
        elif name.startswith("kb_assess:iter"):
            ref = meta.get("referenced_in_verdict")
            events.append(
                {
                    "ts": ts,
                    "category": "critic_assess",
                    "title": f"kb_assess {name.split(':', 1)[-1]}: {'used' if ref else 'not used'}",
                    "detail": {
                        "iter": meta.get("iter"),
                        "mode": meta.get("mode"),
                        "verdict_count": meta.get("verdict_count"),
                        "referenced_in_verdict": ref,
                    },
                    "source": {"span": name, "observation_id": oid},
                }
            )
        elif name.startswith("kb_priors:iter"):
            ref = meta.get("referenced_in_verdict")
            events.append(
                {
                    "ts": ts,
                    "category": "critic_priors",
                    "title": f"kb_priors {name.split(':', 1)[-1]}: {'used' if ref else 'not used'}",
                    "detail": {
                        "iter": meta.get("iter"),
                        "prior_count": meta.get("prior_count"),
                        "referenced_in_verdict": ref,
                    },
                    "source": {"span": name, "observation_id": oid},
                }
            )
    return events


_CATEGORIES = ("recipe_read", "warm_start", "warm_replay", "critic_assess", "critic_priors")


def build_kb_timeline(
    breakdown: dict[str, Any],
    *,
    observations: list[Any] | None = None,
    source: str = "local",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Merge KB read/use decisions into one sorted ``kb_timeline``.

    Pulls warm-start / warm-replay from ``breakdown.kb_provenance`` and recipe
    reads + critic assess/priors from either Langfuse ``observations`` (when
    given — full span granularity) or the breakdown sections
    (``recipe_snapshot_reads.tail`` + ``critic_iterations``). Never raises on
    partial input — missing sources add a warning and set ``degraded``.

    Args:
        breakdown: A ``session_breakdown.json`` dict.
        observations: Optional Langfuse observation (span) list; preferred for
            ``recipe_read`` / ``critic_assess`` / ``critic_priors`` when given.
        source: Provenance label (``"local"`` / ``"langfuse"``).
        trace_id: Langfuse trace id to record (and stamp onto each event).

    Returns:
        A ``kb_timeline`` dict matching
        :class:`inference_optimizer.breakdown.schema.KBTimeline`.
    """
    warnings: list[str] = []
    raw: list[dict[str, Any]] = []

    kbp = breakdown.get("kb_provenance")
    if isinstance(kbp, dict):
        raw.extend(_warm_events(kbp))
    else:
        warnings.append("kb_provenance section missing or not an object")
        kbp = {}

    if observations:
        # Span path: recipe reads + critic assess/priors come from spans (full).
        # An empty/None list falls through to the breakdown sections so a
        # transient observation-fetch miss doesn't drop data that's in output.
        raw.extend(_events_from_observations(observations, warnings))
    else:
        # Breakdown path: recipe reads from the (capped) audit tail.
        reads = (kbp.get("recipe_snapshot_reads") or {}).get("tail") if isinstance(kbp, dict) else None
        if isinstance(reads, list):
            raw.extend(_recipe_read_event(r, i) for i, r in enumerate(reads) if isinstance(r, dict))
        else:
            warnings.append("kb_provenance.recipe_snapshot_reads.tail missing")
        critic = breakdown.get("critic_robustness")
        iterations = critic.get("critic_iterations") if isinstance(critic, dict) else None
        if isinstance(iterations, list):
            raw.extend(_critic_kb_events(iterations, warnings))
        else:
            warnings.append("critic_robustness.critic_iterations missing")

    indexed = list(enumerate(raw))
    indexed.sort(key=lambda pair: (_parse_ts(pair[1].get("ts")) or _TS_MAX, pair[0]))

    events: list[dict[str, Any]] = []
    for seq, (_, ev) in enumerate(indexed):
        ev["seq"] = seq
        if trace_id:
            ev["source"]["trace_id"] = trace_id
        events.append(ev)

    counts = {cat: sum(1 for e in events if e["category"] == cat) for cat in _CATEGORIES}
    degraded = bool(warnings) or not events

    timeline: dict[str, Any] = {
        "schema": KB_TIMELINE_SCHEMA,
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


def empty_kb_timeline(*, source: str = "langfuse", reason: str = "") -> dict[str, Any]:
    """Return a well-formed but empty, ``degraded`` ``kb_timeline``.

    Args:
        source: Provenance label to stamp.
        reason: Human-readable warning explaining why it is empty.

    Returns:
        A ``kb_timeline`` dict with no events and ``degraded=True``.
    """
    return {
        "schema": KB_TIMELINE_SCHEMA,
        "source": source,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": [],
        "counts": {cat: 0 for cat in _CATEGORIES},
        "degraded": True,
        "warnings": [reason] if reason else ["no events recovered"],
    }


def enrich_breakdown_with_langfuse_kb_timeline(
    breakdown: dict[str, Any],
    *,
    trace_id: str | None = None,
    seed: str | None = None,
    credentials: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch trace output + observations from Langfuse and inject ``kb_timeline``.

    Recovers the breakdown (for warm-start / warm-replay) and the KB spans (for
    full recipe-read / critic-assess / critic-priors granularity) from the
    session's Langfuse trace, builds the merged timeline, and writes it onto
    ``breakdown`` in place. Best-effort — injects a ``degraded`` empty timeline
    on any failure.

    Args:
        breakdown: The ``session_breakdown.json`` dict to enrich in place.
        trace_id: Explicit trace id (defaults to one resolved from the breakdown).
        seed: Explicit correlation seed override.
        credentials: Optional explicit Langfuse credentials override.
        timeout: Per-request timeout in seconds.

    Returns:
        The same ``breakdown`` dict, with ``breakdown["kb_timeline"]`` set.
    """
    from ..orchestrator.trace.langfuse_reader import (
        fetch_observations,
        fetch_session_breakdown,
        resolve_trace_id,
    )
    from .agent_timeline import _trace_seed_from_breakdown

    bd_trace_id, bd_seed = _trace_seed_from_breakdown(breakdown)
    resolved = resolve_trace_id(trace_id=trace_id or bd_trace_id, seed=seed or bd_seed)
    if not resolved:
        breakdown["kb_timeline"] = empty_kb_timeline(reason="no trace_id / correlation seed on breakdown")
        return breakdown

    source_breakdown = fetch_session_breakdown(trace_id=resolved, credentials=credentials, timeout=timeout)
    if source_breakdown is None:
        breakdown["kb_timeline"] = empty_kb_timeline(reason=f"could not fetch trace {resolved} from langfuse")
        return breakdown

    observations = fetch_observations(resolved, credentials=credentials, timeout=timeout)
    breakdown["kb_timeline"] = build_kb_timeline(
        source_breakdown, observations=observations, source="langfuse", trace_id=resolved
    )
    return breakdown


__all__ = [
    "KB_TIMELINE_SCHEMA",
    "build_kb_timeline",
    "empty_kb_timeline",
    "enrich_breakdown_with_langfuse_kb_timeline",
]
