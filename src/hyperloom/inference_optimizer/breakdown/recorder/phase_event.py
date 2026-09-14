# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``phase`` event: where the run was, and what it dispatched there.

Every other event is scoped by a phase, but the phases themselves were only
derived at export: from ``phase_history`` rows paired two at a time, so the
segment the session ended in had no successor to close it, and by testing each
action's timestamp against ``[entered_ts, exit_ts)``, which charges an action
outliving its phase to whichever phase inherited it. So the phase becomes an
event, opened on entry and closed on exit, and each dispatch records its own
phase at the moment it is dispatched.

:data:`SECTION_ACTION` rows stay thin -- identity, the ordering phase, the
verdict -- because the stage events hold the per-dispatch detail and joining on
``task_id`` reaches it. For ``report``, ``recover``, ``session_breakdown`` and
``target_analysis``, which no stage event covers, the join finds nothing and
these rows are the only record.

Nothing here holds a recorder object, for the same reason the enablement event
does not: the call sites live in different modules on different ticks.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    now_iso_seconds as _now,
    text_or_none as _text_or_none,
)
from .event_ids import event_id
from .event_rows import rows_for_event, sort_rows, wire_rows
from .event_sink import EventSink, make_sink
from .event_timeline import finish_event, open_event

# Every section a phase event assembles from. Named from the leaf module the
# assembler shares, so :func:`_finish` reads its parts without an import cycle.
from .sections import PHASE_EVENT_SECTIONS
from .recorder_warnings import note_failure

log = logging.getLogger(__name__)

EVENT_TYPE = "phase"
EVENT_KIND = "phase"

#: The component segment of a phase event's id; the phase segment carries the
#: phase name, so ``framework_agent:2:phase`` is that phase's time in cycle 2.
EVENT_COMPONENT = "phase"

PRODUCER = "orchestrator"

#: The event-level section: which phase, and the span it ended up covering.
SECTION_EVENT = "phase_event"

#: One row per entry into the phase, keyed by the entering transition's
#: ``phase_history`` position. A re-entry inside one cycle gets a second row,
#: not a second event: the id has no segment that could distinguish them.
SECTION_SEGMENT = "phase_segment"

#: One row per dispatched action, keyed by its task id. Opened at dispatch and
#: settled in place, so an action killed mid-flight reads as dispatched with no
#: verdict, not as never having happened.
SECTION_ACTION = "phase_action"

#: One row per non-transition ``phase_history`` marker, keyed by its position.
SECTION_MARKER = "phase_marker"

#: One row per proposal the phase raised, keyed by the bus message carrying it.
#: A proposal the Critic refused is never dispatched, so it gets no action row,
#: and the ruling would otherwise have no subject to be filed against.
SECTION_PROPOSAL = "phase_proposal"

STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded"
STATUS_INTERRUPTED = "interrupted"

#: Transition reasons and marker evidence are bounded prose, not logs.
MAX_REASON_CHARS = 500


def phase_event_id(phase: str, macro_cycle: int) -> str:
    """Build ``{phase}:{macro_cycle}:phase``. Raises ``ValueError`` if ``phase``
    is not a token or ``macro_cycle`` is negative."""
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _sink(event: str) -> EventSink | None:
    """The sink for ``event``, or ``None`` when no session is bound."""
    try:
        from ...session.session_binding import bound_session_or_none

        if bound_session_or_none() is None:
            return None
        return make_sink(event, producer=PRODUCER)
    except Exception as exc:  # noqa: BLE001 — the run outranks its own record
        note_failure(section="phase_event", error=exc, detail="phase event: cannot resolve a sink")
        return None


def _open(event: str, *, phase: str, macro_cycle: int, start_time: str = "") -> int | None:
    """Put a phase on the timeline, once, however many callers ask.

    :func:`open_event` hands back the sequence an earlier open took, so a
    dispatch can open its phase without knowing whether the transition already
    did. ``None`` means the shell write failed; the caller records anyway."""
    shell: dict[str, Any] = {
        "phase": str(phase or "").strip().upper(),
        "macro_cycle": int(macro_cycle or 0),
    }
    # Onto the fragment as well as the shell: assembly rebuilds ``ext`` from the
    # fragments and never re-reads the shell.
    sink = _sink(event)
    if sink is not None:
        sink.record(SECTION_EVENT, dict(shell))
    return open_event(
        event_type=EVENT_TYPE,
        event=event,
        event_section=SECTION_EVENT,
        producer=PRODUCER,
        kind=EVENT_KIND,
        start_time=start_time or _now(),
        ext=shell,
    )


def record_entry(
    *,
    phase: str,
    macro_cycle: int,
    sequence: int,
    from_phase: str = "",
    reason: str = "",
    evidence: Mapping[str, Any] | None = None,
    entered_at: str = "",
    entered_unix: float | None = None,
) -> None:
    """Open the phase being entered and record how the run got there. Never
    raises. ``sequence`` is the transition's ``phase_history`` position and keys
    the segment row; ``entered_unix`` measures the segment when it closes."""
    try:
        event = phase_event_id(phase, macro_cycle)
        sink = _sink(event)
        if sink is None:
            return
        entered = str(entered_at or "") or _now()
        _open(event, phase=phase, macro_cycle=macro_cycle, start_time=entered)
        sink.record(
            SECTION_SEGMENT,
            {
                "sequence": int(sequence or 0),
                "from_phase": str(from_phase or "").strip().upper(),
                "entered_at": entered,
                "entered_unix": _float_or_none(entered_unix),
                "entered_reason": _clip(reason, MAX_REASON_CHARS),
                "entered_evidence": _as_dict(evidence),
            },
            row_type="segment",
            natural_ids=str(int(sequence or 0)),
        )
    except Exception as exc:  # noqa: BLE001 — a phase change outranks its own record
        note_failure(section="phase_event", error=exc, detail="phase event: entry record failed")


def record_exit(
    *,
    phase: str,
    macro_cycle: int,
    to_phase: str = "",
    reason: str = "",
    evidence: Mapping[str, Any] | None = None,
    exited_at: str = "",
    exited_unix: float | None = None,
) -> None:
    """Close the phase being left on the exit that ended it. Never raises.

    The segment settled is the open one, found by reading the phase's own rows
    back rather than by trusting the caller's macro cycle: the loopback bumps
    ``macro_cycle`` on its way out, so an id computed from it would close an
    event that was never opened. ``macro_cycle`` is only the fallback for when
    no open segment can be found.
    """
    try:
        found = _open_segment(phase)
        if found is None:
            event = phase_event_id(phase, macro_cycle)
            row_sequence: int | None = None
            entered_unix: float | None = None
        else:
            event, row_sequence, entered_unix = found
        sink = _sink(event)
        if sink is None:
            return
        exited = str(exited_at or "") or _now()
        duration = None
        if entered_unix is not None and exited_unix is not None:
            duration = max(0.0, float(exited_unix) - float(entered_unix))
        settle: dict[str, Any] = {
            "to_phase": str(to_phase or "").strip().upper(),
            "exited_at": exited,
            "exited_unix": _float_or_none(exited_unix),
            "exit_reason": _clip(reason, MAX_REASON_CHARS),
            "exit_evidence": _as_dict(evidence),
            "duration_sec": duration,
        }
        if row_sequence is not None:
            sink.record(
                SECTION_SEGMENT,
                dict(settle, sequence=int(row_sequence)),
                row_type="segment",
                natural_ids=str(int(row_sequence)),
            )
        sink.record(SECTION_EVENT, {"end_time": exited})
        _finish(event, end_time=exited)
    except Exception as exc:  # noqa: BLE001
        note_failure(section="phase_event", error=exc, detail="phase event: exit record failed")


def record_marker(
    *,
    phase: str,
    macro_cycle: int,
    sequence: int,
    reason: str = "",
    evidence: Mapping[str, Any] | None = None,
    ts: str = "",
) -> None:
    """Record one non-transition marker against the phase it was raised in.
    Never raises. ``sequence`` is its ``phase_history`` position and keys it."""
    try:
        event = phase_event_id(phase, macro_cycle)
        sink = _sink(event)
        if sink is None:
            return
        _open(event, phase=phase, macro_cycle=macro_cycle)
        sink.record(
            SECTION_MARKER,
            {
                "sequence": int(sequence or 0),
                "reason": _clip(reason, MAX_REASON_CHARS),
                "evidence": _as_dict(evidence),
                "ts": str(ts or "") or _now(),
            },
            row_type="marker",
            natural_ids=str(int(sequence or 0)),
        )
    except Exception as exc:  # noqa: BLE001
        note_failure(section="phase_event", error=exc, detail="phase event: marker record failed")


#: What a specialist round contributes to the action row that dispatched it.
#: ``proposal_set`` is deliberately absent: for the FRAMEWORK arm each proposal
#: owns a row on the framework event, and for the scouts the product is findings.
_ROUND_TEXT_FIELDS = ("domain", "gap_canonical_id", "summary", "reason", "source")
_ROUND_LIST_FIELDS = ("tags", "new_findings", "residual_questions", "notes")


def record_specialist_round(
    *,
    task_id: str,
    phase: str,
    macro_cycle: int,
    round_id: str = "",
    proposals_total: Any = None,
    empty: Any = None,
    confidence: Any = None,
    ensemble_scores: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Merge what a specialist round produced onto the action row that ordered it.

    Merged onto the action row the dispatcher already opened for ``task_id``,
    not added as a second one. ``phase`` and ``macro_cycle`` are the fallback
    for when no dispatch row exists.
    """
    try:
        key = str(task_id or "")
        if not key:
            return
        event = _action_event(key)
        if event is None:
            if not str(phase or ""):
                return
            event = phase_event_id(phase, macro_cycle)
            record_dispatch(action="specialist", task_id=key, phase=phase, macro_cycle=macro_cycle)
        sink = _sink(event)
        if sink is None:
            return
        row: dict[str, Any] = {"task_id": key}
        if str(round_id or "") and str(round_id) != key:
            row["round_id"] = str(round_id)
        for name in _ROUND_TEXT_FIELDS:
            if name in fields:
                row[name] = _clip(str(fields.get(name) or ""), MAX_REASON_CHARS)
        for name in _ROUND_LIST_FIELDS:
            if name in fields:
                row[name] = [str(item) for item in (fields.get(name) or []) if str(item or "")]
        if proposals_total is not None:
            row["proposals_total"] = _int_or_none(proposals_total)
        if empty is not None:
            row["empty"] = bool(empty)
        if confidence is not None:
            row["confidence"] = _float_or_none(confidence)
        if ensemble_scores:
            row["ensemble_scores"] = _as_dict(ensemble_scores)
        sink.record(SECTION_ACTION, row, row_type="action", natural_ids=key)
    except Exception as exc:  # noqa: BLE001 — a round outranks its own record
        note_failure(section="phase_event", error=exc, detail="phase event: specialist round record failed")


def record_dispatch(
    *,
    action: str,
    task_id: str,
    phase: str,
    macro_cycle: int,
    tick: int = 0,
    dispatched_at: str = "",
    dispatched_unix: float | None = None,
) -> None:
    """Record an action against the phase that ordered it, at dispatch. Never raises.

    Written here rather than at settle because an action can outlive the phase
    that ordered it: the phase in scope when the result lands would charge a
    plateau-straddling baseline to whichever phase inherited it.
    """
    try:
        if not str(task_id or ""):
            return
        event = phase_event_id(phase, macro_cycle)
        sink = _sink(event)
        if sink is None:
            return
        _open(event, phase=phase, macro_cycle=macro_cycle)
        sink.record(
            SECTION_ACTION,
            {
                "action": str(action or ""),
                "task_id": str(task_id),
                "phase": str(phase or "").strip().upper(),
                "macro_cycle": int(macro_cycle or 0),
                "tick": int(tick or 0),
                "dispatched_at": str(dispatched_at or "") or _now(),
                "dispatched_unix": _float_or_none(dispatched_unix),
            },
            row_type="action",
            natural_ids=str(task_id),
        )
    except Exception as exc:  # noqa: BLE001 — an action outranks its own record
        note_failure(section="phase_event", error=exc, detail="phase event: dispatch record failed")


def record_proposal(
    *,
    proposal_msg_id: str,
    action: str,
    phase: str,
    macro_cycle: int,
    from_agent: str = "",
    tick: int = 0,
    predicted_gain_pct: Any = None,
    candidate_id: Any = None,
    variant_name: Any = None,
    proposed_at: str = "",
) -> None:
    """Record a proposal against the phase that raised it. Never raises.

    Written when the proposal is minted, not when it is acted on: one refused by
    the Critic or left pending at the phase exit has no dispatch and so no other
    row, and the ruling needs a subject to be filed against.
    """
    try:
        if not str(proposal_msg_id or ""):
            return
        event = phase_event_id(phase, macro_cycle)
        sink = _sink(event)
        if sink is None:
            return
        _open(event, phase=phase, macro_cycle=macro_cycle)
        sink.record(
            SECTION_PROPOSAL,
            {
                "proposal_msg_id": str(proposal_msg_id),
                "action": str(action or ""),
                "from_agent": str(from_agent or ""),
                "phase": str(phase or "").strip().upper(),
                "macro_cycle": int(macro_cycle or 0),
                "tick": int(tick or 0),
                "predicted_gain_pct": _float_or_none(predicted_gain_pct),
                "candidate_id": _text_or_none(candidate_id),
                "variant_name": _text_or_none(variant_name),
                "proposed_at": str(proposed_at or "") or _now(),
            },
            row_type="proposal",
            natural_ids=str(proposal_msg_id),
        )
    except Exception as exc:  # noqa: BLE001 — a proposal outranks its own record
        note_failure(section="phase_event", error=exc, detail="phase event: proposal record failed")


def record_proposal_review(
    *,
    proposal_msg_id: str,
    verdict: str,
    effective_verdict: str = "",
    source: str = "",
    reasoning: Any = None,
    confidence: Any = None,
    failure_reason_code: Any = None,
    required_evidence: Any = None,
    risks: Any = None,
    advice_text: Any = None,
    alternative_action: Any = None,
    variants: Any = None,
    reviewed_at: str = "",
) -> None:
    """File the Critic's ruling on the proposal it ruled on. Never raises.

    The row is located by reading back which phase event holds the proposal, not
    from the phase in scope: the Critic runs on its own tick and routinely rules
    after the raising phase has exited. A ruling for a proposal with no row is
    dropped rather than minting one. ``effective_verdict`` is what was
    committed, which the envelope validator can change.
    """
    try:
        if not str(proposal_msg_id or ""):
            return
        event = _proposal_event(str(proposal_msg_id))
        if event is None:
            log.debug("phase event: no proposal row for %s to file a ruling on", proposal_msg_id)
            return
        sink = _sink(event)
        if sink is None:
            return
        authored = str(verdict or "")
        effective = str(effective_verdict or "") or authored
        sink.record(
            SECTION_PROPOSAL,
            {
                "proposal_msg_id": str(proposal_msg_id),
                "critic_review": {
                    "verdict": authored,
                    "effective_verdict": effective,
                    # Its own field because the two verdicts alone say that they
                    # differ, not that the envelope validator imposed it.
                    "held_to_rule": effective != authored,
                    "source": str(source or ""),
                    "reasoning": _clip(reasoning, MAX_REASON_CHARS),
                    "confidence": _float_or_none(confidence),
                    "failure_reason_code": _text_or_none(failure_reason_code),
                    "required_evidence": [str(item) for item in _as_list(required_evidence)],
                    "risks": [dict(risk) for risk in _as_list(risks) if isinstance(risk, Mapping)],
                    "advice_text": _text_or_none(advice_text),
                    "alternative_action": _text_or_none(alternative_action),
                    "variants": [dict(row) for row in _as_list(variants) if isinstance(row, Mapping)],
                    "reviewed_at": str(reviewed_at or "") or _now(),
                },
            },
            row_type="proposal",
            natural_ids=str(proposal_msg_id),
        )
    except Exception as exc:  # noqa: BLE001 — a ruling outranks its own record
        note_failure(section="phase_event", error=exc, detail="phase event: proposal review record failed")


def record_proposal_outcome(
    *,
    proposal_msg_id: str,
    materialized: bool = False,
    denied: bool = False,
    reauthored: bool = False,
    task_id: Any = None,
    patch_verdict_key: Any = None,
    settled_at: str = "",
) -> None:
    """Record what the loop did with a proposal. Never raises.

    ``task_id`` joins the proposal to the dispatch row it became, beside it on
    this same event. Located and dropped the same way as the ruling."""
    try:
        if not str(proposal_msg_id or ""):
            return
        event = _proposal_event(str(proposal_msg_id))
        if event is None:
            return
        sink = _sink(event)
        if sink is None:
            return
        sink.record(
            SECTION_PROPOSAL,
            {
                "proposal_msg_id": str(proposal_msg_id),
                "outcome": {
                    "materialized": bool(materialized),
                    "denied": bool(denied),
                    "reauthored": bool(reauthored),
                    "task_id": _text_or_none(task_id),
                    "patch_verdict_key": _text_or_none(patch_verdict_key),
                    "settled_at": str(settled_at or "") or _now(),
                },
            },
            row_type="proposal",
            natural_ids=str(proposal_msg_id),
        )
    except Exception as exc:  # noqa: BLE001
        note_failure(section="phase_event", error=exc, detail="phase event: proposal outcome record failed")


def record_settle(
    *,
    task_id: str,
    status: str = "",
    decision: str = "",
    error_class: Any = None,
    workspace: Any = None,
    settled_at: str = "",
    settled_unix: float | None = None,
    phase: str = "",
    macro_cycle: int = 0,
    action: str = "",
) -> None:
    """Settle a dispatched action's row with the verdict it got. Never raises.

    The row settled is the one the dispatch opened, found by reading the spool
    back: the dispatching phase is not in scope at the settle, and a resumed
    process holds no memory of it. The other arguments are only the fallback.
    """
    try:
        if not str(task_id or ""):
            return
        event = _action_event(str(task_id))
        if event is None:
            # No dispatch row: a task settled by a path that never went through
            # the runner, or an unreadable spool. Better placed than dropped.
            if not str(phase or ""):
                return
            event = phase_event_id(phase, macro_cycle)
            record_dispatch(
                action=action,
                task_id=str(task_id),
                phase=phase,
                macro_cycle=macro_cycle,
            )
        sink = _sink(event)
        if sink is None:
            return
        settled = str(settled_at or "") or _now()
        row: dict[str, Any] = {
            "task_id": str(task_id),
            "status": str(status or ""),
            "decision": str(decision or ""),
            "error_class": _text_or_none(error_class),
            "workspace": _text_or_none(workspace),
            "settled_at": settled,
            "settled_unix": _float_or_none(settled_unix),
        }
        sink.record(SECTION_ACTION, row, row_type="action", natural_ids=str(task_id))
    except Exception as exc:  # noqa: BLE001
        note_failure(section="phase_event", error=exc, detail="phase event: settle record failed")


def _finish(event: str, *, end_time: str) -> None:
    """Close ``event`` on the rows recorded against it so far. The sequence is
    re-derived through :func:`_open`, which hands back the one the first open
    took: a phase exited twice in a cycle would otherwise publish twice."""
    from .assembler import event_parts
    from .event_ids import parse_event_id

    parsed = parse_event_id(event)
    sequence = _open(event, phase=parsed.phase, macro_cycle=parsed.macro_cycle)
    parts = event_parts(PHASE_EVENT_SECTIONS, event=event)
    ext, status = assemble_phase_ext(parts, event=event)
    finish_event(
        event_type=EVENT_TYPE,
        event=event,
        sequence=sequence,
        status=status or STATUS_SUCCEEDED,
        ext=ext,
        kind=EVENT_KIND,
        start_time=str(ext.get("entered_at") or ""),
        end_time=end_time,
    )


def _rows(section: str, event: str) -> list[dict[str, Any]]:
    """Read one section's rows for one event back out of the spool."""
    try:
        from .assembler import event_parts

        return rows_for_event(event_parts((section,)).get(section) or [], event)
    except Exception as exc:  # noqa: BLE001 — a read-back failure is not a write failure
        note_failure(section=section, error=exc, detail=f"reading back the rows of event {event}")
        return []


def _open_segment(phase: str) -> tuple[str, int, float | None] | None:
    """The phase's most recent unclosed entry: its event id, sequence, and
    entry epoch. ``None`` at a first transition or on an unreadable spool."""
    try:
        from .assembler import event_parts

        wanted = str(phase or "").strip().upper()
        rows = event_parts((SECTION_SEGMENT,)).get(SECTION_SEGMENT) or []
        best: tuple[str, int, float | None] | None = None
        best_sequence = -1
        for row in rows:
            if not isinstance(row, Mapping) or row.get("exited_at"):
                continue
            event = str(row.get("event_id") or "")
            if not event or str(event.split(":")[0]).upper() != wanted:
                continue
            sequence = _int_or_none(row.get("sequence")) or 0
            if sequence >= best_sequence:
                best_sequence = sequence
                best = (event, sequence, _float_or_none(row.get("entered_unix")))
        return best
    except Exception as exc:  # noqa: BLE001
        note_failure(section="phase_event", error=exc, detail="phase event: cannot resolve the open segment")
        return None


def _action_event(task_id: str) -> str | None:
    """The phase event holding ``task_id``'s dispatch row, if one was recorded."""
    try:
        from .assembler import event_parts

        for row in event_parts((SECTION_ACTION,)).get(SECTION_ACTION) or []:
            if isinstance(row, Mapping) and str(row.get("task_id") or "") == str(task_id):
                event = str(row.get("event_id") or "")
                if event:
                    return event
        return None
    except Exception as exc:  # noqa: BLE001
        note_failure(section="phase_event", error=exc, detail="phase event: cannot resolve the action's event")
        return None


def _proposal_event(proposal_msg_id: str) -> str | None:
    """The phase event holding ``proposal_msg_id``'s row, if it was recorded."""
    try:
        from .assembler import event_parts

        for row in event_parts((SECTION_PROPOSAL,)).get(SECTION_PROPOSAL) or []:
            if isinstance(row, Mapping) and str(row.get("proposal_msg_id") or "") == str(proposal_msg_id):
                event = str(row.get("event_id") or "")
                if event:
                    return event
        return None
    except Exception as exc:  # noqa: BLE001
        note_failure(section="phase_event", error=exc, detail="phase event: cannot resolve the proposal's event")
        return None


def assemble_phase_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble a phase event's ``ext``, and the status it reports, out of the
    sections read back from the spool."""
    header = _header(rows_for_event(parts.get(SECTION_EVENT) or [], event))
    segments = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_SEGMENT) or [], event), keys=("sequence",)),
        drop=("event_id",),
    )
    actions = [
        # Measured here because the settle cannot see the dispatch that opened
        # the row.
        dict(row, duration_sec=_span(row.get("dispatched_unix"), row.get("settled_unix")))
        for row in wire_rows(
            sort_rows(
                rows_for_event(parts.get(SECTION_ACTION) or [], event),
                keys=("dispatched_at", "task_id"),
            ),
            drop=("event_id",),
        )
    ]
    markers = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_MARKER) or [], event), keys=("sequence", "ts")),
        drop=("event_id",),
    )
    proposals = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_PROPOSAL) or [], event),
            keys=("proposed_at", "proposal_msg_id"),
        ),
        drop=("event_id",),
    )
    # Summed over the entries, not measured first to last: a phase re-entered
    # inside one cycle did not own the time the run spent elsewhere in between.
    measured = [row.get("duration_sec") for row in segments if isinstance(row.get("duration_sec"), (int, float))]
    ext: dict[str, Any] = {
        "phase": str(header.get("phase") or ""),
        "macro_cycle": int(header.get("macro_cycle") or 0),
        "entered_at": str((segments[0] or {}).get("entered_at") or "") if segments else "",
        "exited_at": str((segments[-1] or {}).get("exited_at") or "") if segments else "",
        "exit_reason": str((segments[-1] or {}).get("exit_reason") or "") if segments else "",
        "entries": len(segments),
        "duration_sec": round(sum(float(d) for d in measured), 6) if measured else None,
        # An entry with no exit is the segment the run was in when it stopped.
        "open": any(not row.get("exited_at") for row in segments),
        "segments": segments,
        "actions": {
            "count": len(actions),
            # The gap to ``count`` is the actions still in or killed in flight.
            "settled": sum(1 for row in actions if row.get("status")),
            "kinds": sorted({str(row.get("action") or "") for row in actions if row.get("action")}),
            "rows": actions,
        },
        "markers": {"count": len(markers), "rows": markers},
        "proposals": {
            "count": len(proposals),
            # The gap to ``count`` is the ones the Critic never reached, which
            # is not the same as the ones it refused.
            "reviewed": sum(1 for row in proposals if row.get("critic_review")),
            "materialized": sum(1 for row in proposals if _as_dict(row.get("outcome")).get("materialized")),
            "rows": proposals,
        },
    }
    return ext, _status_for(segments, actions)


def _status_for(
    segments: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
) -> str:
    """The status a phase event reports: a phase the run left cleanly succeeded
    whatever the actions inside it decided, one still open at assembly is
    ``interrupted``, and one whose every dispatch failed is degraded."""
    if not segments or any(not row.get("exited_at") for row in segments):
        return STATUS_INTERRUPTED
    settled = [row for row in actions if row.get("status")]
    if settled and all(str(row.get("status") or "") == STATUS_FAILED for row in settled):
        return STATUS_DEGRADED
    return STATUS_SUCCEEDED


def _span(start: Any, end: Any) -> float | None:
    """Elapsed seconds between two Unix epochs, or ``None`` when either is
    missing -- an open span, not a zero-length one."""
    lo, hi = _float_or_none(start), _float_or_none(end)
    if lo is None or hi is None:
        return None
    return round(max(0.0, hi - lo), 6)


def _header(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold the event-level fragments into one header."""
    header: dict[str, Any] = {}
    for row in rows:
        if isinstance(row, Mapping):
            header.update({k: v for k, v in row.items() if k != "event_id"})
    return header


__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "MAX_REASON_CHARS",
    "PHASE_EVENT_SECTIONS",
    "PRODUCER",
    "SECTION_ACTION",
    "SECTION_EVENT",
    "SECTION_MARKER",
    "SECTION_SEGMENT",
    "STATUS_DEGRADED",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_SUCCEEDED",
    "assemble_phase_ext",
    "phase_event_id",
    "record_dispatch",
    "record_entry",
    "record_exit",
    "record_marker",
    "record_specialist_round",
    "record_settle",
]
