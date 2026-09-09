# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``phase`` event: where the run was, and what it dispatched there.

Every other event on the timeline is scoped by a phase -- the event id's first
segment *is* a phase name -- and until now the timeline held no record of the
phases themselves. A reader could see a baseline event tagged ``framework_agent``
and had no way to learn when that phase was entered, why the run left it, or how
long it had. Those facts were published as two derived top-level keys instead:

**``phase_segments`` was recomputed at export from ``phase_history``.** Rows were
paired off two at a time to synthesize each segment's exit and duration
(``collectors/timeline.py:753-769``), which is a reconstruction of a timespan
that the transition itself knew exactly. It also cannot describe the segment the
session ended in, because that one has no successor row to be paired with.

**``phase_timeline`` attributed actions to phases by guessing.** The audit writer
is handed the phase and the macro cycle (``shared_state.py:2918``) and drops both
from the payload it records (``instrument.py:849-859``), forwarding them only to
a v4 mirror. Export therefore had to attribute each action by testing its
timestamp against ``[entered_ts, exit_ts)`` (``collectors/timeline.py:826-855``).
A guess is the wrong answer whenever an action outlives the phase that ordered
it -- a long baseline settling after a plateau exit gets charged to the phase
that inherited it, not the one that asked for it -- and the fact needed to get it
right was in scope at the writer and thrown away.

So the phase becomes an event, opened on entry and closed on exit, and each
dispatch records its own phase at the moment it is dispatched.

What this event does *not* do is duplicate the per-dispatch detail the stage
events already record. The original plan for this section was a flat action
stream carrying every attempt's status, decision and key metric; that row is a
strict subset of what ``baseline.ext.actions[]``, ``roofline.ext.actions[]``,
``framework_agent.ext.attempts[]`` and ``enablement.ext.attempts.rows[]`` hold,
and poorer -- none of them can express a baseline's discarded cold-warmup rounds
or a framework attempt's arm and provenance. Recording it anyway would put one
semantic in two places, which is the drift this whole layer exists to remove.

:data:`SECTION_ACTION` rows are therefore deliberately thin: identity, the phase
that ordered the dispatch, and the verdict. The detail is reached by joining on
``task_id``, which every stage event's per-dispatch rows already carry -- so the
link needs no ``event_ref`` field, and more to the point no author-time table
mapping action kinds to components. Such a table is exactly how
``_AUDIT_ACTIONS`` came to cover four kinds and stay there while the catalogue
grew to fifteen.

For the actions no stage event covers -- ``report``, ``recover``,
``session_breakdown``, ``target_analysis``, which were invisible on the timeline
entirely -- these rows are the only record, and the join simply finds nothing.

Coverage does not come from enumerating the action catalogue, which is how
``_AUDIT_ACTIONS`` came to cover four of fifteen kinds and stay there. It comes
from the three places the code already funnels through:
:meth:`~hyperloom.orchestrator.loop.dispatcher.DispatcherCollaborator.run_task_registered`
(whose docstring calls itself "the only way an action runs", and which holds the
sole call to ``sub.run_task`` in the tree), the settle branch in
``_reap_dispatched_task`` that both the promote and the unpromotable paths pass
through, and :func:`~hyperloom.orchestrator.phases.machine_state.record_phase_transition`.

Nothing here holds a recorder object, for the same reason the enablement event
does not: the three call sites live in different modules on different ticks, and
threading one object between them would make the record depend on the call graph
that reached it. Each function below resolves its own sink and opens its own
event idempotently.
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

log = logging.getLogger(__name__)

EVENT_TYPE = "phase"
EVENT_KIND = "phase"

#: The component segment of a phase event's id. The phase segment carries the
#: phase name itself, so ``framework_agent:2:phase`` is "the run's time in
#: FRAMEWORK_AGENT during macro cycle 2".
EVENT_COMPONENT = "phase"

PRODUCER = "orchestrator"

#: The event-level section: which phase, and the span it ended up covering.
SECTION_EVENT = "phase_event"

#: One row per entry into the phase, keyed by the ``phase_history`` position the
#: entering transition took. A phase re-entered inside one macro cycle does not
#: get a second event -- the id has no segment that could distinguish them, and
#: inventing one would break the rule that every segment be recomputable from
#: persisted state -- so each entry is a row and the event covers all of them.
SECTION_SEGMENT = "phase_segment"

#: One row per dispatched action, keyed by its task id. Opened at dispatch and
#: settled in place, so an action killed mid-flight reads as dispatched with no
#: verdict rather than as never having happened.
SECTION_ACTION = "phase_action"

#: One row per non-transition ``phase_history`` marker, keyed by its position.
SECTION_MARKER = "phase_marker"

#: One row per proposal the phase raised, keyed by the bus message that carries
#: it. The proposal is recorded here because this is the one event that exists
#: for every phase: an action becomes a ``phase_action`` row only once it is
#: dispatched, and a proposal that the Critic rejected is never dispatched at
#: all -- so without this row the refusal had nothing to be recorded against.
#:
#: This is also the only place a Critic ruling can be filed with the thing it
#: ruled on. ``framework_agent`` carries ``proposals[].critic_review``, but its
#: rows exist for two creation paths inside one phase; a ruling on a KERNEL
#: ``kernel_opt``, on a PRELUDE ``baseline``, or on an action no framework arm
#: maps to had no subject row anywhere, and was dropped without trace.
SECTION_PROPOSAL = "phase_proposal"

STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded"
STATUS_INTERRUPTED = "interrupted"

#: Transition reasons and marker evidence are bounded prose, not logs.
MAX_REASON_CHARS = 500


def phase_event_id(phase: str, macro_cycle: int) -> str:
    """Build the event id for one phase's time in one macro cycle.

    Returns:
        str: ``{phase}:{macro_cycle}:phase``.

    Raises:
        ValueError: If ``phase`` is not a token or ``macro_cycle`` is negative.
    """
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _sink(event: str) -> EventSink | None:
    """The sink rows for ``event`` are written through, or ``None``.

    Returns:
        EventSink | None: The bound session's sink, or ``None`` when no session
        is bound -- a unit test driving the phase machine directly, or a resume
        before the session scope is entered. Recording is best-effort either way.
    """
    try:
        from ...session.session_binding import bound_session_or_none

        if bound_session_or_none() is None:
            return None
        return make_sink(event, producer=PRODUCER)
    except Exception:  # noqa: BLE001 — the run outranks its own record
        log.debug("phase event: cannot resolve a sink", exc_info=True)
        return None


def _open(event: str, *, phase: str, macro_cycle: int, start_time: str = "") -> int | None:
    """Put a phase on the timeline, once, however many callers ask.

    :func:`open_event` hands back the sequence an earlier open took instead of
    writing a second shell, which is what lets a dispatch open the phase it
    belongs to without knowing whether the transition already did.

    Args:
        event (str): The phase event id.
        phase (str): The phase name, as recorded.
        macro_cycle (int): The macro cycle.
        start_time (str): When the phase was entered; defaults to now.

    Returns:
        int | None: The storage sequence to close with, or ``None`` when the
        shell write failed. A caller that gets ``None`` still records: the
        fragments land, and finalize recovers the event from them.
    """
    shell: dict[str, Any] = {
        "phase": str(phase or "").strip().upper(),
        "macro_cycle": int(macro_cycle or 0),
    }
    # Onto the fragment as well as the shell. Assembly rebuilds ``ext`` from
    # the fragments and never re-reads the shell, so a field that rode only on
    # the shell is dropped the moment anything closes the event.
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
    """Open the phase being entered and record how the run got there. Never raises.

    Args:
        phase (str): The phase being entered.
        macro_cycle (int): The macro cycle it is entered in.
        sequence (int): The entering transition's position in ``phase_history``,
            which keys the segment row.
        from_phase (str): The phase being left, empty at the run's first entry.
        reason (str): The transition reason, from ``PHASE_EXIT_REASONS``.
        evidence (Mapping[str, Any] | None): The transition's structured evidence.
        entered_at (str): The transition's ISO timestamp; defaults to now.
        entered_unix (float | None): The matching Unix epoch, used to measure
            the segment when it closes.
    """
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
    except Exception:  # noqa: BLE001 — a phase change outranks its own record
        log.debug("phase event: entry record failed", exc_info=True)


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
    back rather than by trusting the caller's macro cycle. The loopback bumps
    ``macro_cycle`` on its way out of a phase, so the cycle in scope at the
    transition can already be the *next* one -- computing the outgoing event id
    from it would close an event that was never opened and leave the real one
    hanging.

    Args:
        phase (str): The phase being left.
        macro_cycle (int): The cycle in scope at the transition, used only as
            the fallback when no open segment can be found.
        to_phase (str): The phase being entered.
        reason (str): The exit reason.
        evidence (Mapping[str, Any] | None): The transition's evidence.
        exited_at (str): The transition's ISO timestamp; defaults to now.
        exited_unix (float | None): The matching Unix epoch, which measures the
            segment against its own recorded entry.
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
    except Exception:  # noqa: BLE001
        log.debug("phase event: exit record failed", exc_info=True)


def record_marker(
    *,
    phase: str,
    macro_cycle: int,
    sequence: int,
    reason: str = "",
    evidence: Mapping[str, Any] | None = None,
    ts: str = "",
) -> None:
    """Record one non-transition marker against the phase it was raised in. Never raises.

    Args:
        phase (str): The phase the marker belongs to.
        macro_cycle (int): The macro cycle it was raised in.
        sequence (int): The marker's position in ``phase_history``, which keys it.
        reason (str): The marker's reason.
        evidence (Mapping[str, Any] | None): Its structured payload.
        ts (str): Its ISO timestamp; defaults to now.
    """
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
    except Exception:  # noqa: BLE001
        log.debug("phase event: marker record failed", exc_info=True)


#: What a specialist round contributes to the action row that dispatched it.
#: The dispatch and settle already say when it ran and how it ended; these say
#: what it came back with. ``proposal_set`` is deliberately absent -- for the
#: FRAMEWORK arm each proposal owns a row on the framework event, and for the
#: phase-independent scouts the product is the findings, not a variant list.
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

    The dispatcher already opened an action row for this task, keyed by
    ``task_id``, on the phase that ordered it. This merges onto that row rather
    than adding a second one, so the round's product sits with the dispatch it
    came from instead of in a flat ledger that had to be re-attributed to a
    phase at export.

    Args:
        task_id (str): The specialist task's id, which keys the row.
        phase (str): The phase in scope, used only when no dispatch row exists.
        macro_cycle (int): The macro cycle, same fallback.
        round_id (str): The round's own id, when it differs from the task id.
        proposals_total (Any): How many proposals came back.
        empty (Any): Whether the round produced nothing.
        confidence (Any): The round's self-reported confidence.
        ensemble_scores (Mapping[str, Any] | None): Advisory multi-model
            scoring of the proposals, when it ran.
        **fields: Any of ``domain``, ``gap_canonical_id``, ``summary``,
            ``reason``, ``source``, ``tags``, ``new_findings``,
            ``residual_questions``, ``notes``.
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
    except Exception:  # noqa: BLE001 — a round outranks its own record
        log.debug("phase event: specialist round record failed", exc_info=True)


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

    Written here rather than at settle because the phase that ordered a dispatch
    is the phase that owns it, and an action can outlive the phase that ordered
    it. Recording the phase in scope when the result lands would charge a
    plateau-exit-straddling baseline to whichever phase inherited it, which is
    what the export-time timestamp-window attribution did.

    Args:
        action (str): The action kind.
        task_id (str): The task id, which keys the row.
        phase (str): The dispatching phase.
        macro_cycle (int): The dispatching macro cycle.
        tick (int): The coordinator tick, for ordering within a phase.
        dispatched_at (str): The dispatch's ISO timestamp; defaults to now.
        dispatched_unix (float | None): The matching Unix epoch, which measures
            the action when it settles.
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
    except Exception:  # noqa: BLE001 — an action outranks its own record
        log.debug("phase event: dispatch record failed", exc_info=True)


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

    Written when the proposal is minted, not when it is acted on, because most
    proposals are never acted on: one refused by the Critic, or left pending
    when the phase exits, has no dispatch and therefore no other row. The
    proposal is the unit the Critic rules on, so it has to exist before the
    ruling can be filed against it.

    Args:
        proposal_msg_id (str): The bus message carrying it, which keys the row.
        action (str): The action proposed, in the proposer's own words --
            including the ones no framework arm maps to.
        phase (str): The phase in scope when it was raised.
        macro_cycle (int): The macro cycle it was raised in.
        from_agent (str): The role that raised it.
        tick (int): The coordinator tick, for ordering within a phase.
        predicted_gain_pct (Any): The gain the proposer claimed, when it did.
        candidate_id (Any): The upstream candidate, when the proposal names one.
        variant_name (Any): The variant, when the proposal names one.
        proposed_at (str): The ISO timestamp; defaults to now.
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
    except Exception:  # noqa: BLE001 — a proposal outranks its own record
        log.debug("phase event: proposal record failed", exc_info=True)


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

    The row is located by reading back which phase event holds the proposal,
    rather than by using the phase in scope: the Critic runs on its own tick
    and a proposal raised in one phase is routinely ruled on after the phase
    has exited. Charging the ruling to whichever phase happened to be current
    would file it against a phase that never saw the proposal.

    A ruling for a proposal with no row is dropped rather than minting one. A
    ruling cannot bring into existence the thing it claims to be about, and a
    minted row would carry a verdict with no proposal behind it.

    Args:
        proposal_msg_id (str): The proposal ruled on.
        verdict (str): What the Critic authored.
        effective_verdict (str): What was committed, which the envelope
            validator can change; defaults to the authored one.
        source (str): ``critic`` or ``critic_unavailable``.
        reasoning (Any): The Critic's own account.
        confidence (Any): How sure it was.
        failure_reason_code (Any): The code behind a refusal.
        required_evidence (Any): What it asked to see first.
        risks (Any): The risks it named.
        advice_text (Any): The advisory attached to the ruling.
        alternative_action (Any): What it proposed instead.
        variants (Any): Per-variant rulings, when the proposal is a grid.
        reviewed_at (str): The ISO timestamp; defaults to now.
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
                    # Recorded as its own field because the two verdicts alone
                    # say that they differ without saying that the difference
                    # was imposed by the envelope validator.
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
    except Exception:  # noqa: BLE001 — a ruling outranks its own record
        log.debug("phase event: proposal review record failed", exc_info=True)


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

    ``task_id`` is what joins the proposal to the action it became: the
    dispatch row beside it on this same event. Without it the two halves of one
    decision -- what was asked for, and what was run -- sit on the same event
    with nothing connecting them.

    Located the same way as the ruling, and dropped the same way when the
    proposal has no row.

    Args:
        proposal_msg_id (str): The proposal acted on.
        materialized (bool): Whether it went ahead.
        denied (bool): Whether the framework refused it.
        reauthored (bool): Whether it was sent back to be authored again.
        task_id (Any): The task it materialized into, when it did.
        patch_verdict_key (Any): The patch the ruling was recorded against.
        settled_at (str): The ISO timestamp; defaults to now.
    """
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
    except Exception:  # noqa: BLE001
        log.debug("phase event: proposal outcome record failed", exc_info=True)


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

    The row settled is the one the dispatch opened, located by reading back
    which phase event holds ``task_id``. The dispatching phase is not in scope
    at the settle -- that is the whole point of recording it at dispatch -- and
    a resumed process has no memory of the dispatch either, so the lookup goes
    through the spool rather than through anything held.

    Args:
        task_id (str): The settled task's id.
        status (str): The status it settled on.
        decision (str): The promotion verdict.
        error_class (Any): The failure class, when it failed.
        workspace (Any): The workspace it ran in.
        settled_at (str): The settle's ISO timestamp; defaults to now.
        settled_unix (float | None): The matching Unix epoch.
        phase (str): The phase in scope now, used only to place the row when no
            dispatch row can be found.
        macro_cycle (int): The macro cycle in scope now, same fallback.
        action (str): The action kind, same fallback.
    """
    try:
        if not str(task_id or ""):
            return
        event = _action_event(str(task_id))
        if event is None:
            # No dispatch row: a task settled by a path that never went through
            # the runner, or a spool that could not be read. Place it in the
            # phase in scope and say so, rather than dropping the verdict.
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
    except Exception:  # noqa: BLE001
        log.debug("phase event: settle record failed", exc_info=True)


def _finish(event: str, *, end_time: str) -> None:
    """Close ``event`` on the rows recorded against it so far.

    The sequence is re-derived through :func:`_open`, which hands back the one
    the first open took. A phase is closed once per exit and can be exited
    several times in a cycle, so closing on a fresh sequence would publish one
    phase as two timeline entries -- the re-entry's close would not overwrite
    the first entry's.
    """
    from .assembler import event_parts
    from .event_ids import parse_event_id

    parsed = parse_event_id(event)
    sequence = _open(event, phase=parsed.phase, macro_cycle=parsed.macro_cycle)
    parts = event_parts(PHASE_EVENT_SECTIONS)
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
    """Read one section's rows for one event back out of the spool.

    Returns:
        list[dict[str, Any]]: The matching rows, empty when nothing is readable.
    """
    try:
        from .assembler import event_parts

        return rows_for_event(event_parts((section,)).get(section) or [], event)
    except Exception:  # noqa: BLE001 — a read-back failure is not a write failure
        log.debug("phase event: cannot read back %s", section, exc_info=True)
        return []


def _open_segment(phase: str) -> tuple[str, int, float | None] | None:
    """Find the phase's most recent entry that has not been closed.

    Args:
        phase (str): The phase whose open segment is wanted.

    Returns:
        tuple[str, int, float | None] | None: The event id holding it, the
        segment's sequence, and the Unix epoch it was entered at. ``None`` when
        the phase has no open segment, which is the normal reading for a first
        transition or an unreadable spool.
    """
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
    except Exception:  # noqa: BLE001
        log.debug("phase event: cannot resolve the open segment", exc_info=True)
        return None


def _action_event(task_id: str) -> str | None:
    """Find which phase event holds ``task_id``'s dispatch row.

    Returns:
        str | None: The owning event id, or ``None`` when no dispatch was
        recorded for it.
    """
    try:
        from .assembler import event_parts

        for row in event_parts((SECTION_ACTION,)).get(SECTION_ACTION) or []:
            if isinstance(row, Mapping) and str(row.get("task_id") or "") == str(task_id):
                event = str(row.get("event_id") or "")
                if event:
                    return event
        return None
    except Exception:  # noqa: BLE001
        log.debug("phase event: cannot resolve the action's event", exc_info=True)
        return None


def _proposal_event(proposal_msg_id: str) -> str | None:
    """Find which phase event holds ``proposal_msg_id``'s row.

    Returns:
        str | None: The owning event id, or ``None`` when the proposal was
        never recorded.
    """
    try:
        from .assembler import event_parts

        for row in event_parts((SECTION_PROPOSAL,)).get(SECTION_PROPOSAL) or []:
            if isinstance(row, Mapping) and str(row.get("proposal_msg_id") or "") == str(proposal_msg_id):
                event = str(row.get("event_id") or "")
                if event:
                    return event
        return None
    except Exception:  # noqa: BLE001
        log.debug("phase event: cannot resolve the proposal's event", exc_info=True)
        return None


#: Every section a phase event assembles from. Declared here as well as in the
#: assembler so :func:`_finish` can read its own parts without importing the
#: assembler's tuple, which would close an import cycle.
PHASE_EVENT_SECTIONS: tuple[str, ...] = (
    SECTION_EVENT,
    SECTION_SEGMENT,
    SECTION_ACTION,
    SECTION_MARKER,
    SECTION_PROPOSAL,
)


def assemble_phase_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble a phase event's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The phase sections as read
            back from the spool, section name to row list.
        event (str): The event id to assemble.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload and the status the phase
            reports.
    """
    header = _header(rows_for_event(parts.get(SECTION_EVENT) or [], event))
    segments = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_SEGMENT) or [], event), keys=("sequence",)),
        drop=("event_id",),
    )
    actions = [
        # Measured from the row's own two endpoints rather than recorded at the
        # settle, which cannot see the dispatch that opened the row.
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
    # Summed over the entries, not measured from the first to the last: a phase
    # re-entered inside one cycle did not own the time the run spent elsewhere
    # in between, and charging it that time is how a budget guard comes to
    # believe a phase overran.
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
        # ``phase_segments`` could not represent this at all: it paired rows two
        # at a time, so the final segment had no successor to be closed by and
        # was published with an empty exit and no duration.
        "open": any(not row.get("exited_at") for row in segments),
        "segments": segments,
        "actions": {
            "count": len(actions),
            # Dispatches that got a verdict. The gap between this and ``count``
            # is the actions still in flight or killed mid-flight, which the
            # flat projection could not express: it only ever held settled rows,
            # so a cancelled dispatch read as one that never happened.
            "settled": sum(1 for row in actions if row.get("status")),
            "kinds": sorted({str(row.get("action") or "") for row in actions if row.get("action")}),
            "rows": actions,
        },
        "markers": {"count": len(markers), "rows": markers},
        "proposals": {
            "count": len(proposals),
            # Proposals the Critic ruled on. The gap between this and ``count``
            # is the ones it never reached -- left pending when the phase
            # exited, or filtered as already reviewed -- which is a different
            # thing from having been refused.
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
    """The status a phase event reports.

    A phase the run left cleanly succeeded, whatever the actions inside it
    decided -- their own events carry those verdicts, and a phase is not failed
    by having dispatched a failing action. A phase still open at assembly is
    ``interrupted``: nothing ruled on it. A phase whose every dispatch failed
    left without doing what it was entered for, which is degraded.
    """
    if not segments or any(not row.get("exited_at") for row in segments):
        return STATUS_INTERRUPTED
    settled = [row for row in actions if row.get("status")]
    if settled and all(str(row.get("status") or "") == STATUS_FAILED for row in settled):
        return STATUS_DEGRADED
    return STATUS_SUCCEEDED


def _span(start: Any, end: Any) -> float | None:
    """Measure a span from its two recorded endpoints.

    Args:
        start (Any): The Unix epoch the span opened at.
        end (Any): The Unix epoch it closed at.

    Returns:
        float | None: The elapsed seconds, or ``None`` when either endpoint is
        missing -- an open span, not a zero-length one.
    """
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
