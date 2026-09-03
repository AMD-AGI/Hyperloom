# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-side of the breakdown recorder.

Assembles the per-producer fragments written by :class:`~.recorder.Recorder`
into a ``{section: value}`` mapping ready to drop into the
``session_breakdown.json`` envelope:

* ``singleton`` sections -> the payload of the latest fragment (by ``ts``).
* plain ``item`` sections -> payloads concatenated into a list, ordered by
  ``seq`` then ``ts``.
* v4 entity streams (``_V4_ENTITY_IDS``) -> partial updates ordered by ``ts``
  then ``seq`` and deep-merged by stable id, so the result carries one entry
  per entity rather than one per fragment.

A compose/normalize pass runs last and reconciles across fragments:
``versions`` collapses to a ``{tool: meta}`` map (last write per tool wins),
``critic_iterations`` / ``robustness_signals`` and the ``kernel_*``
substreams are folded into their composed sections, and competing kernel
route operations are rewritten to ``status="superseded"``. Bad/partial
fragments are skipped and noted in ``warnings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json

_UNREADABLE = object()
#: Streams whose records are entities rather than append-only events: several
#: fragments can describe the same entity and are merged by its id.
_V4_ENTITY_IDS: dict[str, tuple[str, ...]] = {
    "phase_transitions": ("transition_id", "event_id"),
    "subjects": ("subject_id",),
    "operations": ("operation_id",),
    "measurements": ("measurement_id",),
    "adoptions": ("adoption_id",),
    "artifacts": ("artifact_id",),
    "trace_events": ("trace_event_id", "event_id", "span_id"),
}
_NESTED_ENTITY_IDS: tuple[str, ...] = (
    "attempt_id",
    "substep_id",
    "gate_id",
    "decision_id",
    "relation_id",
    "measurement_id",
    "artifact_id",
    "adoption_id",
    "subject_id",
    "operation_id",
)


def parts_dir(session_dir: Path | str) -> Path:
    """Return the breakdown spool directory for ``session_dir``.

    Args:
        session_dir (Path | str): the session root directory.

    Returns:
        Path: the breakdown parts (spool) directory under the session.
    """
    from ...session.session_paths import breakdown_parts_dir  # local: avoid import cycle

    return breakdown_parts_dir(Path(session_dir))


def has_parts(session_dir: Path | str) -> bool:
    """True iff at least one record fragment exists for this session.

    Args:
        session_dir: The session root directory.

    Returns:
        ``True`` when the spool directory holds at least one ``*.json``
        fragment.
    """
    d = parts_dir(session_dir)
    return d.is_dir() and any(d.glob("*.json"))


def _load(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    """Read and parse one fragment file, noting problems into ``warnings``.

    Args:
        path (Path): the fragment file to read.
        warnings (list[str]): a list that parse/validation warnings are
            appended to.

    Returns:
        dict[str, Any] | None: the parsed fragment record, or ``None`` when it
            cannot be read or is not a JSON object.
    """
    rec = read_json(
        path,
        default=_UNREADABLE,
        on_error=lambda exc: warnings.append(f"recorder: failed to read {path.name}: {exc!r}"),
    )
    if rec is _UNREADABLE:
        return None
    if not isinstance(rec, dict):
        warnings.append(f"recorder: {path.name} is not an object")
        return None
    return rec


def assemble_parts(
    session_dir: Path | str,
    *,
    warnings: list[str] | None = None,
    keep_event_rows: bool = False,
) -> dict[str, Any]:
    """Return ``{section: list | dict}`` assembled from the spool directory.

    Empty mapping when no fragments exist (caller falls back to collectors).

    Args:
        session_dir: The session root directory.
        warnings: Optional list to append parse/validation warnings to; a
            fresh list is used when not provided.
        keep_event_rows: Retain the :data:`EVENT_SECTIONS` substreams instead
            of dropping them. Only :func:`event_parts` sets this; the breakdown
            envelope never wants them.

    Returns:
        A ``{section: list | dict}`` mapping assembled from the spool
        directory, or ``{}`` when no fragments exist.
    """
    warns = warnings if warnings is not None else []
    d = parts_dir(session_dir)
    if not d.is_dir():
        return {}

    items: dict[str, list[dict[str, Any]]] = {}
    singletons: dict[str, dict[str, Any]] = {}
    discarded: dict[str, list[str]] = {}

    for path in sorted(d.glob("*.json")):
        rec = _load(path, warns)
        if rec is None:
            continue
        section = rec.get("section")
        if not isinstance(section, str) or not section:
            warns.append(f"recorder: {path.name} missing 'section'")
            continue
        if rec.get("kind") == "singleton":
            prev = singletons.get(section)
            if prev is None or str(rec.get("ts") or "") >= str(prev.get("ts") or ""):
                if prev is not None:
                    discarded.setdefault(section, []).append(str(prev.get("producer") or "?"))
                singletons[section] = rec
            else:
                discarded.setdefault(section, []).append(str(rec.get("producer") or "?"))
        else:
            items.setdefault(section, []).append(rec)

    # A singleton fragment is named for its producer, so a section with more
    # than one is a section two producers both claimed. Only the newest
    # survives, and the other producer's payload does not merge into it -- it
    # is dropped whole. Nothing downstream can see that it existed.
    for section, producers in discarded.items():
        warns.append(
            f"recorder: {section} was written as a singleton by more than one "
            f"producer; only the newest was kept and "
            f"{sorted(set(producers))} were dropped whole"
        )

    out: dict[str, Any] = {}
    for section, recs in items.items():
        if section in _V4_ENTITY_IDS:
            recs.sort(key=_v4_record_sort_key)
            conflicts: list[str] = []
            out[section] = _merge_v4_entities(
                [r.get("payload") for r in recs],
                id_fields=_V4_ENTITY_IDS[section],
                conflicts=conflicts,
            )
            if conflicts:
                # Repeated updates from one producer merge into one fragment
                # before they ever reach here, so two payloads for one id came
                # from two producers. Merging is the point; disagreeing on a
                # field is not, and the later write wins purely on timestamp.
                warns.append(
                    f"recorder: {section} entities were written by more than "
                    "one producer with conflicting values, and the later write "
                    f"won: {sorted(set(conflicts))[:5]}"
                )
        else:
            recs.sort(key=lambda r: (int(r.get("seq") or 0), str(r.get("ts") or "")))
            out[section] = [r.get("payload") for r in recs]
    for section, rec in singletons.items():
        out[section] = rec.get("payload")

    _normalize_kernel_route_operations(out)
    _compose_critic_robustness(out)
    _compose_kernel_journey(out)
    if not keep_event_rows:
        _drop_event_rows(out)
    _compose_versions(out)
    return out


def _normalize_kernel_route_operations(out: dict[str, Any]) -> None:
    """Normalize active Kernel routes from canonical operation fragments only."""
    operations = out.get("operations")
    if not isinstance(operations, list):
        return
    selections = [
        operation
        for operation in operations
        if isinstance(operation, dict)
        and operation.get("kind") == "strategy_selection"
        and operation.get("strategy_group") == "kernel_optimizer"
        and str(operation.get("status") or "").lower() not in {"revoked", "reverted", "superseded", "skipped"}
    ]
    selections_by_cycle: dict[str, list[dict[str, Any]]] = {}
    for selection in selections:
        macro_cycle = selection.get("macro_cycle")
        if macro_cycle is None:
            continue
        selections_by_cycle.setdefault(str(macro_cycle), []).append(selection)

    for cycle_selections in selections_by_cycle.values():
        selection_ids = {
            str(selection.get("operation_id") or "") for selection in cycle_selections if selection.get("operation_id")
        }
        if len(selection_ids) != 1:
            continue
        selection = cycle_selections[-1]
        selection_id = next(iter(selection_ids))
        selected_strategy = str((selection.get("outputs") or {}).get("selected_strategy") or "")
        cycle_routes = [
            operation
            for operation in operations
            if isinstance(operation, dict)
            and operation.get("kind") == "kernel_optimizer_run"
            and str(operation.get("parent_operation_id") or "") == selection_id
        ]
        selected_routes = [
            route
            for route in cycle_routes
            if str(route.get("strategy") or "") == selected_strategy
            and str(route.get("status") or "").lower() not in {"revoked", "reverted", "skipped"}
        ]
        active_route = selected_routes[-1] if selected_routes else None
        active_route_id = str(active_route.get("operation_id") or "") if isinstance(active_route, dict) else ""
        for route in cycle_routes:
            competition = dict(((route.get("extensions") or {}).get("route_competition") or {}))
            if route is active_route:
                competition.update(
                    {
                        "active": True,
                        "selected": True,
                        "normalized_from_operations": True,
                    }
                )
            elif str(route.get("strategy") or "") != selected_strategy or active_route is not None:
                previous_status = str(route.get("status") or "")
                route["status"] = "superseded"
                competition.update(
                    {
                        "active": False,
                        "selected": False,
                        "historical_executed": True,
                        "historical_status": previous_status,
                        "superseded_by": active_route_id or None,
                        "normalized_from_operations": True,
                    }
                )
            else:
                competition.update(
                    {
                        "active": False,
                        "selected": True,
                        "normalized_from_operations": True,
                    }
                )
            extensions = dict(route.get("extensions") or {})
            extensions["route_competition"] = competition
            route["extensions"] = extensions


def _v4_record_sort_key(record: dict[str, Any]) -> tuple[str, int, str]:
    """Order v4 updates by recorder time, then producer-local sequence."""
    return (
        str(record.get("ts") or ""),
        int(record.get("seq") or 0),
        str(record.get("producer") or ""),
    )


def _entity_id(value: Any, id_fields: tuple[str, ...]) -> str:
    """Return the first stable id present in an entity mapping."""
    if not isinstance(value, dict):
        return ""
    return next((str(value.get(name) or "") for name in id_fields if value.get(name)), "")


def _merge_v4_entities(
    payloads: list[Any],
    *,
    id_fields: tuple[str, ...],
    conflicts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Merge time-ordered partial entity updates by stable id.

    Args:
        payloads: Entity payloads in recorder order, oldest first.
        id_fields: The fields any of which carries the entity's stable id.
        conflicts: Optional list that ``<id>.<field>`` is appended to whenever
            a later update replaces a value with a different one, so a caller
            can report what the merge decided on its own.
    """
    merged: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        stable_id = _entity_id(payload, id_fields)
        if not stable_id:
            merged.append(dict(payload))
            continue
        index = index_by_id.get(stable_id)
        if index is None:
            index_by_id[stable_id] = len(merged)
            merged.append(dict(payload))
        else:
            merged[index] = _deep_merge(
                merged[index],
                payload,
                conflicts=conflicts,
                path=stable_id,
            )
    return merged


def _deep_merge(
    current: dict[str, Any],
    update: dict[str, Any],
    *,
    conflicts: list[str] | None = None,
    path: str = "",
    entity_root: bool = True,
) -> dict[str, Any]:
    """Merge partial entity state while preserving nested keyed histories."""
    merged = dict(current)
    for key, value in update.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(
                previous,
                value,
                conflicts=conflicts,
                path=f"{path}.{key}" if path else key,
                entity_root=False,
            )
        elif isinstance(previous, list) and isinstance(value, list):
            merged[key] = _merge_lists(previous, value, conflicts=conflicts, path=path)
        elif key == "started_at" and previous and value and entity_root:
            # Partial updates describe one operation. The first fragment owns
            # its start time; later result/finalization updates must not move
            # the start forward merely because they were written later.
            #
            # Only at the entity root (``path`` is the bare stable id, or empty
            # for a direct call): a substep / gate / extension nested below has
            # its own start time, and quietly keeping the smaller of two would
            # stamp one sub-entity with another's clock and swallow the conflict.
            merged[key] = min(str(previous), str(value))
        else:
            if conflicts is not None and previous is not None and value is not None and previous != value:
                # A field arriving twice with two values. Filling a gap is what
                # partial updates are for and says nothing; replacing an answer
                # with a different one is a disagreement settled by whichever
                # fragment happened to sort last.
                conflicts.append(f"{path}.{key}" if path else key)
            merged[key] = value
    return merged


def _merge_lists(
    current: list[Any],
    update: list[Any],
    *,
    conflicts: list[str] | None = None,
    path: str = "",
) -> list[Any]:
    """Merge list entries with stable nested ids and append other new values."""
    merged = list(current)
    indexes: dict[tuple[str, str], int] = {}
    for index, value in enumerate(merged):
        if not isinstance(value, dict):
            continue
        for field in _NESTED_ENTITY_IDS:
            if value.get(field):
                indexes[(field, str(value[field]))] = index
                break
    for value in update:
        if not isinstance(value, dict):
            if value not in merged:
                merged.append(value)
            continue
        identity = next(
            ((field, str(value[field])) for field in _NESTED_ENTITY_IDS if value.get(field)),
            None,
        )
        index = indexes.get(identity) if identity else None
        if index is None:
            if value not in merged:
                merged.append(dict(value))
                if identity:
                    indexes[identity] = len(merged) - 1
        else:
            merged[index] = _deep_merge(
                merged[index],
                value,
                conflicts=conflicts,
                path=f"{path}.{identity[1]}" if identity and path else path,
                entity_root=False,
            )
    return merged


def _compose_versions(out: dict[str, Any]) -> None:
    """Fold the ``versions`` item substream into a top-level ``{tool: meta}``
    map (last write per tool wins). No-op when nothing was recorded.

    Args:
        out: The assembled section mapping mutated in place.
    """
    rows = out.get("versions")
    if not isinstance(rows, list):
        return
    merged: dict[str, Any] = {}
    for r in rows:
        if isinstance(r, dict):
            tool = str(r.get("tool") or "").lower()
            if tool:
                merged[tool] = r
    out["versions"] = merged


def _compose_critic_robustness(out: dict[str, Any]) -> None:
    """Fold the ``critic_iterations`` / ``robustness_signals`` item substreams
    into the ``critic_robustness`` singleton. Pops the raw substreams so they
    don't leak into the breakdown envelope.

    Args:
        out: The assembled section mapping mutated in place.
    """
    critic_iters = out.pop("critic_iterations", None)
    rob_signals = out.pop("robustness_signals", None)
    if critic_iters is None and rob_signals is None:
        return
    # A directly-recorded singleton takes precedence over substreams.
    if "critic_robustness" in out:
        return
    critic_iters = critic_iters if isinstance(critic_iters, list) else []
    rob_signals = rob_signals if isinstance(rob_signals, list) else []
    out["critic_robustness"] = {
        "critic_iterations": critic_iters,
        "robustness_signals": rob_signals,
        "kb_writes_summary": _kb_writes_summary(critic_iters),
    }


#: The KERNEL substreams, in the order a reader follows them.
KERNEL_EVENT_SECTIONS: tuple[str, ...] = (
    "kernel_event",
    "kernel_lane_run",
    "kernel_rebench_attempt",
    "kernel_trace_analyze",
    "kernel_geak_attempt",
    "kernel_geak_discovery",
    "kernel_geak_acceptance",
)

#: The roofline substreams. They belong to whichever event their rows are
#: tagged with, which is the roofline's own event when it was dispatched and
#: the enclosing phase's event when it was called inline.
ROOFLINE_EVENT_SECTIONS: tuple[str, ...] = (
    "roofline_event",
    "roofline_action",
    "roofline_profile_run",
    "roofline_analysis_run",
)

#: The baseline substreams, in the order a reader follows them: the event, the
#: measurements dispatched into it, each measurement's passes, and each pass's
#: benchmark rounds.
BASELINE_EVENT_SECTIONS: tuple[str, ...] = (
    "baseline_event",
    "baseline_action",
    "baseline_run",
    "baseline_round",
)

#: Every section holding v6 event rows. They are consumed by the timeline
#: rather than by the breakdown envelope, so assembly pops them here to keep
#: them from leaking into the wire shape.
EVENT_SECTIONS: tuple[str, ...] = KERNEL_EVENT_SECTIONS + ROOFLINE_EVENT_SECTIONS + BASELINE_EVENT_SECTIONS


def event_parts(sections: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    """Read back the event rows of the bound session, keyed by section.

    Assembly pops these sections, so a caller that wants them -- the phase
    closing an event, or finalize recovering one that never closed -- reads
    them through here instead.

    Args:
        sections: The sections to read, e.g. :data:`KERNEL_EVENT_SECTIONS`.

    Returns:
        A ``{section: [payload, ...]}`` mapping holding those sections, each
        defaulting to an empty list.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    from ...session.session_binding import bound_session  # local: avoid import cycle

    assembled = assemble_parts(bound_session(), warnings=[], keep_event_rows=True)
    parts: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        rows = assembled.get(section)
        parts[section] = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return parts


def kernel_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the KERNEL substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`KERNEL_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(KERNEL_EVENT_SECTIONS)


def roofline_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the roofline substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`ROOFLINE_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(ROOFLINE_EVENT_SECTIONS)


def baseline_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the baseline substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`BASELINE_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(BASELINE_EVENT_SECTIONS)


def _drop_event_rows(out: dict[str, Any]) -> None:
    """Drop the v6 event substreams from the breakdown envelope.

    These are recorded for the v6 timeline, which assembles them into events
    when the phase or action that produced them ends. They carry no meaning of
    their own in ``session_breakdown.json``, and leaving them in would publish
    ten undocumented sections alongside the events built from them.

    Args:
        out: The assembled section mapping mutated in place.
    """
    for section in EVENT_SECTIONS:
        out.pop(section, None)


def _compose_kernel_journey(out: dict[str, Any]) -> None:
    """Fold the four kernel-lifecycle item substreams into a single
    kernel-major ``kernel_journey`` view (discovery -> dispatch -> backend
    attempts -> e2e), then pop the raw substreams. No-op when no substream was
    recorded.

    Args:
        out: The assembled section mapping mutated in place.
    """
    discovery = out.pop("kernel_discovery", None)
    dispatch = out.pop("kernel_dispatch", None)
    backend = out.pop("kernel_backend_result", None)
    e2e = out.pop("kernel_e2e", None)
    if discovery is None and dispatch is None and backend is None and e2e is None:
        return
    # A directly-recorded singleton takes precedence over substreams.
    if "kernel_journey" in out:
        return

    discovery_runs = [r for r in (discovery or []) if isinstance(r, dict)]
    dispatch_rows = [r for r in (dispatch or []) if isinstance(r, dict)]
    backend_rows = [r for r in (backend or []) if isinstance(r, dict)]
    e2e_rows = [r for r in (e2e or []) if isinstance(r, dict)]

    # Latest discovery snapshot per kernel_id (later runs win).
    discovery_by_kid: dict[str, dict[str, Any]] = {}
    for run in discovery_runs:
        for hk in run.get("hot_kernels") or []:
            if not isinstance(hk, dict):
                continue
            kid = str(hk.get("kernel_id") or "")
            if kid:
                discovery_by_kid[kid] = hk

    dispatch_by_kid = {str(r.get("kernel_id") or ""): r for r in dispatch_rows if str(r.get("kernel_id") or "")}
    e2e_by_kid = {str(r.get("kernel_id") or ""): r for r in e2e_rows if str(r.get("kernel_id") or "")}
    attempts_by_kid: dict[str, list[dict[str, Any]]] = {}
    for r in backend_rows:
        kid = str(r.get("kernel_id") or "")
        if kid:
            attempts_by_kid.setdefault(kid, []).append(r)

    kids: list[str] = []
    for source in (discovery_by_kid, dispatch_by_kid, attempts_by_kid, e2e_by_kid):
        for kid in source:
            if kid and kid not in kids:
                kids.append(kid)

    kernels: list[dict[str, Any]] = []
    for kid in kids:
        disc = discovery_by_kid.get(kid, {})
        disp = dispatch_by_kid.get(kid, {})
        atts = attempts_by_kid.get(kid, [])
        kernel_e2e = e2e_by_kid.get(kid, {})
        kernels.append(
            {
                "kernel_id": kid,
                "name": str(disc.get("name") or ""),
                "gpu_pct": disc.get("gpu_pct"),
                "bound_type": str(disc.get("bound_type") or ""),
                "source_file": disc.get("source_file"),
                "micro_speedup": _best_micro_speedup(atts),
                "discovery": disc,
                "dispatch": disp,
                "backend_attempts": atts,
                "e2e": kernel_e2e,
                "outcome": _kernel_outcome(disp, atts, kernel_e2e),
            }
        )

    def _gpu(k: dict[str, Any]) -> float:
        """Return a kernel's gpu_pct as a float (``-inf`` when absent/unparseable).

        Args:
            k: A kernel record mapping.

        Returns:
            The kernel's ``gpu_pct`` as a float, or ``-inf`` when absent or
            unparseable.
        """
        v = k.get("gpu_pct")
        try:
            return float(v) if v is not None else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    kernels.sort(key=_gpu, reverse=True)
    out["kernel_journey"] = {
        "discovery_runs": discovery_runs,
        "kernels": kernels,
    }


def _best_micro_speedup(attempts: list[dict[str, Any]]) -> float | None:
    """Best (max) micro_speedup across a kernel's attempts, or None.

    Surfaces the kernel-level achieved speedup at the journey-entry top level.

    Args:
        attempts: A kernel's backend attempt rows.

    Returns:
        The maximum ``micro_speedup`` across the attempts, or ``None`` when
        none is present/parseable.
    """
    best: float | None = None
    for att in attempts:
        if not isinstance(att, dict):
            continue
        v = att.get("micro_speedup")
        try:
            f = float(v) if v is not None else None
        except (TypeError, ValueError):
            f = None
        if f is not None and (best is None or f > best):
            best = f
    return best


def _kernel_outcome(
    dispatch: dict[str, Any],
    attempts: list[dict[str, Any]],
    e2e: dict[str, Any],
) -> str:
    """Coarse per-kernel outcome: adopted / reverted / attempted / dispatched /
    skipped / discovered (in lifecycle-descending precedence).

    Args:
        dispatch: The kernel's dispatch row.
        attempts: The kernel's backend attempt rows.
        e2e: The kernel's end-to-end row.

    Returns:
        The coarse per-kernel outcome label.
    """
    if e2e:
        decision = str(e2e.get("decision") or "").upper()
        validation_tier = str(e2e.get("final_validation_tier") or e2e.get("validation_tier") or "").strip().lower()
        final_validated = e2e.get("validated") is True or validation_tier in {
            "final",
            "final_validation",
            "orchestrator_final",
            "same_harness_final",
            "integrate_e2e",
        }
        if final_validated and (e2e.get("integrated") or decision in ("KEEP", "ADOPTED")):
            return "adopted"
        if decision in ("REVERT", "REJECTED"):
            return "reverted"
    if attempts:
        return "attempted"
    if dispatch:
        return "dispatched" if dispatch.get("dispatched") else "skipped"
    return "discovered"


def _kb_writes_summary(critic_iters: list[Any]) -> dict[str, Any]:
    """Count each critic iteration's verdict (mirrors the collector).

    Args:
        critic_iters: Critic iteration entries to tally by verdict.

    Returns:
        A summary ``{"total": int, "by_verdict": {verdict: count}}``.
    """
    by_verdict: dict[str, int] = {}
    total = 0
    for entry in critic_iters:
        if not isinstance(entry, dict):
            continue
        verdict = str(entry.get("verdict") or "").strip().upper()
        if not verdict:
            continue
        total += 1
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    return {"total": total, "by_verdict": by_verdict}


__all__ = [
    "assemble_parts",
    "has_parts",
    "parts_dir",
]
