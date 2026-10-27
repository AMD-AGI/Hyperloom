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

from hyperloom.common.coerce import to_float as _float_or_none, to_int as _int_or_none, to_mapping as _as_dict
from hyperloom.plugs import progress as _progress

from .event_fields import now_iso_seconds as _now
from .event_ids import event_id
from .event_rows import fold_header, rows_for_event, sort_rows, wire_rows
from .event_sink import bound_sink
from .event_timeline import EventHeader, finish_event, open_event
from .phase_records import (
    MAX_REASON_CHARS,
    ActionDispatch,
    ActionSettlement,
    PhaseDenial,
    PhaseEntry,
    PhaseExit,
    PhaseMarker,
    PhaseProposal,
    ProposalOutcome,
    ProposalReview,
    SpecialistRound,
)

# Every section a phase event assembles from. Named from the leaf module the
# assembler shares, so :func:`_finish` reads its parts without an import cycle.
from .sections import PHASE_EVENT_SECTIONS

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

#: One row per PolicyGate denial, including intents that never became proposals.
SECTION_DENIAL = "phase_denial"

STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded"
STATUS_INTERRUPTED = "interrupted"


def is_phase_transition_row(row: Any) -> bool:
    """True when ``row`` records an actual phase change, not an in-phase marker."""
    if not isinstance(row, dict):
        return False
    to_phase = str(row.get("to_phase") or "").strip().upper()
    from_phase = str(row.get("from_phase") or "").strip().upper()
    return bool(to_phase) and to_phase != from_phase


def phase_event_id(phase: str, macro_cycle: int) -> str:
    """Build ``{phase}:{macro_cycle}:phase``. Raises ``ValueError`` if ``phase``
    is not a token or ``macro_cycle`` is negative."""
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


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
    sink = bound_sink(event, producer=PRODUCER)
    if sink is not None:
        sink.record(SECTION_EVENT, dict(shell))
    return open_event(
        EventHeader(EVENT_TYPE, event, EVENT_KIND, start_time or _now()),
        event_section=SECTION_EVENT,
        producer=PRODUCER,
        ext=shell,
    )


def record_entry(entry: PhaseEntry) -> None:
    """Open the phase being entered and record how the run got there. Never raises."""
    event = phase_event_id(entry.at.phase, entry.at.macro_cycle)
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    entered = str(entry.entered_at or "") or _now()
    _open(event, phase=entry.at.phase, macro_cycle=entry.at.macro_cycle, start_time=entered)
    sink.record(
        SECTION_SEGMENT,
        entry.row(entered),
        row_type="segment",
        natural_ids=str(int(entry.sequence or 0)),
    )


def record_exit(exit_: PhaseExit) -> None:
    """Close the phase being left on the exit that ended it. Never raises.

    The segment settled is the open one, found by reading the phase's own rows
    back rather than by trusting the caller's macro cycle: the loopback bumps
    ``macro_cycle`` on its way out, so an id computed from it would close an
    event that was never opened. ``macro_cycle`` is only the fallback for when
    no open segment can be found.
    """
    found = _open_segment(exit_.at.phase)
    if found is None:
        event = phase_event_id(exit_.at.phase, exit_.at.macro_cycle)
        row_sequence: int | None = None
        entered_unix: float | None = None
    else:
        event, row_sequence, entered_unix = found
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    exited = str(exit_.exited_at or "") or _now()
    duration = None
    if entered_unix is not None and exit_.exited_unix is not None:
        duration = max(0.0, float(exit_.exited_unix) - float(entered_unix))
    settle = exit_.settle(exited, duration)
    if row_sequence is not None:
        sink.record(
            SECTION_SEGMENT,
            dict(settle, sequence=int(row_sequence)),
            row_type="segment",
            natural_ids=str(int(row_sequence)),
        )
    sink.record(SECTION_EVENT, {"end_time": exited})
    _finish(event, end_time=exited)


def record_marker(marker: PhaseMarker) -> None:
    """Record one non-transition marker against the phase it was raised in.

    Never raises. ``marker.sequence`` is its ``phase_history`` position and keys it."""
    event = phase_event_id(marker.at.phase, marker.at.macro_cycle)
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    _open(event, phase=marker.at.phase, macro_cycle=marker.at.macro_cycle)
    sink.record(
        SECTION_MARKER,
        marker.row(),
        row_type="marker",
        natural_ids=str(int(marker.sequence or 0)),
    )


def record_specialist_round(round_: SpecialistRound) -> None:
    """Merge what a specialist round produced onto the action row that ordered it.

    Merged onto the action row the dispatcher already opened for ``task_id``,
    not added as a second one. Missing dispatch evidence is left missing rather
    than reconstructed after the fact with guessed provenance.
    """
    key = str(round_.task_id or "")
    if not key:
        return
    event = _action_event(key)
    if event is None:
        return
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    sink.record(SECTION_ACTION, round_.row(), row_type="action", natural_ids=key)


def record_dispatch(dispatch: ActionDispatch) -> bool:
    """Record an action against its authoring phase and report durability."""
    if not str(dispatch.task_id or ""):
        return False
    event = phase_event_id(dispatch.at.phase, dispatch.at.macro_cycle)
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return False
    opened = _open(event, phase=dispatch.at.phase, macro_cycle=dispatch.at.macro_cycle)
    recorded = sink.record(
        SECTION_ACTION,
        dispatch.row(),
        row_type="action",
        natural_ids=str(dispatch.task_id),
    )
    if opened is not None and recorded is not None and dispatch.action == "profile":
        from ...session.session_binding import bound_session
        from .outcome_stage import record_stage_reached

        record_stage_reached(bound_session(), "profile")
    return opened is not None and recorded is not None


def record_denial(denial: PhaseDenial) -> None:
    """Record one PolicyGate denial at the point the decision is made."""
    event = phase_event_id(denial.at.phase, denial.at.macro_cycle)
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    _open(event, phase=denial.at.phase, macro_cycle=denial.at.macro_cycle)
    sink.append(SECTION_DENIAL, denial.row())


def record_proposal(proposal: PhaseProposal) -> None:
    """Record a proposal against the phase that raised it. Never raises.

    Written when the proposal is minted, not when it is acted on: one refused by
    the Critic or left pending at the phase exit has no dispatch and so no other
    row, and the ruling needs a subject to be filed against.
    """
    if not str(proposal.proposal_msg_id or ""):
        return
    event = phase_event_id(proposal.at.phase, proposal.at.macro_cycle)
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    _open(event, phase=proposal.at.phase, macro_cycle=proposal.at.macro_cycle)
    sink.record(
        SECTION_PROPOSAL,
        proposal.row(),
        row_type="proposal",
        natural_ids=str(proposal.proposal_msg_id),
    )


def record_proposal_review(review: ProposalReview) -> None:
    """File the Critic's ruling on the proposal it ruled on. Never raises.

    The row is located by reading back which phase event holds the proposal, not
    from the phase in scope: the Critic runs on its own tick and routinely rules
    after the raising phase has exited. A ruling for a proposal with no row is
    dropped rather than minting one. ``effective_verdict`` is what was
    committed, which the envelope validator can change.
    """
    if not str(review.proposal_msg_id or ""):
        return
    event = _proposal_event(str(review.proposal_msg_id))
    if event is None:
        log.debug("phase event: no proposal row for %s to file a ruling on", review.proposal_msg_id)
        return
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    sink.record(SECTION_PROPOSAL, review.row(), row_type="proposal", natural_ids=str(review.proposal_msg_id))


def record_proposal_outcome(outcome: ProposalOutcome) -> None:
    """Record what the loop did with a proposal. Never raises.

    ``task_id`` joins the proposal to the dispatch row it became, beside it on
    this same event. Located and dropped the same way as the ruling."""
    if not str(outcome.proposal_msg_id or ""):
        return
    event = _proposal_event(str(outcome.proposal_msg_id))
    if event is None:
        return
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    sink.record(SECTION_PROPOSAL, outcome.row(), row_type="proposal", natural_ids=str(outcome.proposal_msg_id))


def record_settle(settlement: ActionSettlement) -> None:
    """Settle a dispatched action's row with the verdict it got. Never raises.

    The row settled is the one the dispatch opened, found by reading the spool
    back: the dispatching phase is not in scope at the settle. Missing dispatch
    evidence is left missing rather than reconstructed with guessed provenance.
    """
    if not str(settlement.task_id or ""):
        return
    row = _action_row(str(settlement.task_id))
    _progress.candidate_settled(
        str(settlement.task_id), str((row or {}).get("action") or ""), str(settlement.decision or "")
    )
    event = str(row["event_id"]) if row else None
    if event is None:
        return
    sink = bound_sink(event, producer=PRODUCER)
    if sink is None:
        return
    sink.record(SECTION_ACTION, settlement.row(), row_type="action", natural_ids=str(settlement.task_id))


def _finish(event: str, *, end_time: str) -> None:
    """Close ``event`` on the rows recorded against it so far. The sequence is
    re-derived through :func:`_open`, which hands back the one the first open
    took: a phase exited twice in a cycle would otherwise publish twice.

    A spool that cannot be read, or an event id the spool handed back that will
    not parse, is noted and dropped: ``record_exit`` must not raise into the
    phase transition that asked for this close, and finalize recovers an event
    left open as interrupted.
    """
    from .assembler import event_parts
    from .event_ids import parse_event_id
    from .recorder_warnings import RECORDING_ERRORS, note_failure

    try:
        parsed = parse_event_id(event)
        sequence = _open(event, phase=parsed.phase, macro_cycle=parsed.macro_cycle)
        parts = event_parts(PHASE_EVENT_SECTIONS, event=event)
        ext, status = assemble_phase_ext(parts, event=event)
        finish_event(
            EventHeader(EVENT_TYPE, event, EVENT_KIND, str(ext.get("entered_at") or "")),
            sequence=sequence,
            status=status or STATUS_SUCCEEDED,
            ext=ext,
            end_time=end_time,
        )
    except RECORDING_ERRORS as exc:
        note_failure(section=SECTION_EVENT, error=exc, detail=f"closing phase event {event}")


def _open_segment(phase: str) -> tuple[str, int, float | None] | None:
    """The phase's most recent unclosed entry: its event id, sequence, and
    entry epoch. ``None`` at a first transition or on an unreadable spool.

    A segment whose ``event_id`` will not parse is skipped: handing it to
    :func:`~.event_sink.bound_sink` or :func:`_finish` would raise the same
    ``ValueError`` the sink already swallows on write, and the exit must not
    surface it.
    """
    from .assembler import recorded_section
    from .event_ids import parse_event_id

    wanted = str(phase or "").strip().upper()
    best: tuple[str, int, float | None] | None = None
    best_sequence = -1
    for row in recorded_section(SECTION_SEGMENT):
        if row.get("exited_at"):
            continue
        event = str(row.get("event_id") or "")
        if not event or str(event.split(":")[0]).upper() != wanted:
            continue
        try:
            parse_event_id(event)
        except ValueError:
            continue
        sequence = _int_or_none(row.get("sequence")) or 0
        if sequence >= best_sequence:
            best_sequence = sequence
            best = (event, sequence, _float_or_none(row.get("entered_unix")))
    return best


def _row_holding(section: str, field: str, value: str) -> dict[str, Any] | None:
    """The ``section`` row, with an event, that has ``field == value``, if any."""
    from .assembler import recorded_section

    for row in recorded_section(section):
        if str(row.get(field) or "") == str(value) and str(row.get("event_id") or ""):
            return row
    return None


def _event_holding(section: str, field: str, value: str) -> str | None:
    """The phase event whose ``section`` row has ``field == value``, if any."""
    row = _row_holding(section, field, value)
    return str(row["event_id"]) if row else None


def _action_row(task_id: str) -> dict[str, Any] | None:
    """The dispatch row of ``task_id``, if one was recorded."""
    return _row_holding(SECTION_ACTION, "task_id", task_id)


def _action_event(task_id: str) -> str | None:
    """The phase event holding ``task_id``'s dispatch row, if one was recorded."""
    row = _action_row(task_id)
    return str(row["event_id"]) if row else None


def _proposal_event(proposal_msg_id: str) -> str | None:
    """The phase event holding ``proposal_msg_id``'s row, if it was recorded."""
    return _event_holding(SECTION_PROPOSAL, "proposal_msg_id", proposal_msg_id)


def assemble_phase_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble a phase event's ``ext``, and the status it reports, out of the
    sections read back from the spool."""
    header = fold_header(rows_for_event(parts.get(SECTION_EVENT) or [], event))
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
    denials = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_DENIAL) or [], event),
            keys=("denied_at", "proposal_msg_id", "action"),
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
        "denials": {"count": len(denials), "rows": denials},
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


__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "MAX_REASON_CHARS",
    "PHASE_EVENT_SECTIONS",
    "PRODUCER",
    "SECTION_ACTION",
    "SECTION_DENIAL",
    "SECTION_EVENT",
    "SECTION_MARKER",
    "SECTION_SEGMENT",
    "STATUS_DEGRADED",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_SUCCEEDED",
    "assemble_phase_ext",
    "is_phase_transition_row",
    "phase_event_id",
    "record_denial",
    "record_dispatch",
    "record_entry",
    "record_exit",
    "record_marker",
    "record_specialist_round",
    "record_settle",
]
