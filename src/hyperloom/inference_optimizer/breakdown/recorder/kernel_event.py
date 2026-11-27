# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``kernel`` event: one row per fact, assembled at the close.

A KERNEL entry produces facts over minutes to hours -- lane runs, rebench
verdicts, GEAK's whole delegated campaign -- and the reason they are written as
one fragment per row rather than accumulated in memory is not write volume. It
is that the row is the unit that gets updated: a lane run is recorded when it
starts and again when its rebench settles, and a fragment keyed by that run
merges the two without anyone reading the first write back. Held in memory, the
same two-stage arrival needs a mutable object that only the writing process
has, so a resumed process either loses the first half or re-derives it.

Recording and assembly are therefore separate halves of this module, and only
assembly ever sees a whole event. :class:`KernelEventRecorder` writes rows and
knows nothing about arrays; :func:`assemble_kernel_ext` reads the rows back and
decides every wire position, ordering and count. The split is what lets
finalize rebuild the event of a session that was killed mid-phase from exactly
the same rows, through exactly the same code, rather than through a second
projection that agrees with this one only until one of them is edited.

The wire shape assembly produces has three layers, and which layer a fact
belongs in is decided by what kind of fact it is rather than by who wrote it.
``attempts`` holds every candidate either route produced, normalized to one
shape. ``rebench``, ``integrate`` and ``measurements`` hold the evidence that
ruled on them. ``forge`` and ``geak`` hold only what is peculiar to one route
-- forge's trace analysis and re-profile, GEAK's handoff, delegation and
product -- and ``outcome`` states how the whole visit ran, in one vocabulary
both routes reach.

Settlement stays in assembly because it is a join: an attempt is kept because
an instrument measured it, and neither row exists when the other is written.
The instruments differ by route -- a GEAK candidate is re-benched end to end, a
forge candidate timed by the lane that produced it -- so each settled attempt
records which one ruled on it. The event's verdict and status are read off
those settled attempts, at the close, once every row is on disk.

One instrument is deliberately not waited for. The integrate gate runs after
the visit exits and records into the event of the cycle it settles in, so its
verdict on a forge patch may never reach the event that produced the patch.
The visit therefore concludes on what it can measure itself, and a reader
tracking a patch end to end follows ``delivered`` into the ``integrate`` rows
of a later event rather than expecting this one to be amended.

Every ``*_at`` argument in this module is an ISO timestamp.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom.inference_optimizer.session.geak_rebench_verdict import (
    REBENCH_NO_MATERIAL,
    REBENCH_NO_PROMOTE,
    REBENCH_VALIDATED,
)
from hyperloom_common.coerce import (
    text_or_none as _text,
    to_float as _float_or_none,
    to_int as _int_or_none,
    to_mapping as _as_dict,
    to_list as _as_list,
)
from .assembler import event_parts
from .construct import event_header
from .event_ids import event_id
from .event_rows import group_rows, rows_for_event, sort_rows, wire_rows
from .event_timeline import EventHeader
from .kernel_gain import lane_gain_axis, lane_gain_pct, stamp_gate_gain, stamp_rebench_gain
from .kernel_records import (
    LEDGER_GEAK,
    SOURCE_FUSION,
    SOURCE_GEAK_AUTHORED_KERNEL,
    SOURCE_GEMM_TUNING,
    SOURCE_KERNEL_REWRITE,
)
from .baseline_event import assemble_baseline_actions
from .roofline_event import assemble_roofline_action
from .sections import EVENT_SECTIONS

EVENT_TYPE = "kernel"
EVENT_KIND = "kernel_agent"

#: The phase and component segments of the kernel event id. The phase names the
#: coordinator phase the entry belongs to; the component names what inside that
#: phase the event is about, which leaves room for a second event type in the
#: same phase without either of them having to be renamed.
EVENT_PHASE = "kernel_agent"
EVENT_COMPONENT = "kernel"

#: Rows the coordinator observed itself.
PRODUCER = "orchestrator"

#: Rows replayed out of GEAK's own JSON. Tagged apart from the rows above so a
#: fact the orchestrator watched happen stays distinguishable from one it read
#: back off disk after the fact -- they carry different evidence even when they
#: land in adjacent wire positions.
PRODUCER_GEAK = "geak_replay"

SECTION_EVENT = "kernel_event"
SECTION_LANE_RUN = "kernel_lane_run"
SECTION_REBENCH = "kernel_rebench_attempt"
SECTION_TRACE_ANALYZE = "kernel_trace_analyze"
SECTION_GEAK_ATTEMPT = "kernel_geak_attempt"
SECTION_GEAK_DISCOVERY = "kernel_geak_discovery"
SECTION_GEAK_ACCEPTANCE = "kernel_geak_acceptance"
#: One row per kernel the trace attributed during this visit, with the profiling
#: fields the hot-kernel summary deliberately drops. Kept apart from the lane
#: rows because discovery is a snapshot of what the analysis found, not an
#: optimization attempt.
SECTION_DISCOVERED = "kernel_discovered"

#: One row per end-to-end integrate gate verdict, keyed by the integration id.
#: The gate is the orchestrator's, not a lane's: a lane produces a candidate
#: and rules on its own micro-benchmark, and whether the patch survives an
#: end-to-end measurement is decided afterwards, by a step outside the phase
#: that produced it. The verdict is its own row rather than a field on the
#: lane's because one patch can be gated several times -- an integration fault
#: is retried on its own budget -- and because the settling can land in a
#: later cycle than the rewrite it rules on.
SECTION_INTEGRATE = "kernel_integrate"

ROW_LANE_RUN = "lane_run"
ROW_REBENCH = "rebench"
ROW_TRACE_ANALYZE = "trace_analyze"

#: What asked for an analysis. The bus request is its own trigger because it
#: is the one path with no roofline event of its own to account for it.
TRACE_ANALYZE_TRIGGER_BUS_REQUEST = "bus_request"

ROW_GEAK_ATTEMPT = "geak_attempt"
ROW_GEAK_DISCOVERY = "geak_discovery"
ROW_GEAK_ACCEPTANCE = "geak_acceptance"
ROW_DISCOVERED = "discovered"
ROW_INTEGRATE = "integrate"

ROUTE_GEAK = "geak"
ROUTE_FORGE = "forge"

SOURCE_GEAK_ENV_SELECTION = "geak_env_selection"

#: What became of one candidate. ``failed`` is distinct from ``rejected``: the
#: first never produced a candidate to judge, the second produced one that was
#: judged and lost. Collapsing them loses the only signal that separates a
#: broken lane from a lane that is working and finding nothing.
OUTCOME_ADOPTED = "adopted"
OUTCOME_NEEDS_REVIEW = "needs_review"
OUTCOME_REJECTED = "rejected"
OUTCOME_FAILED = "failed"

#: Which evidence settled a candidate. The two routes are judged by different
#: gates -- GEAK by its rebench, forge by the integrate gate -- and a reader
#: who cannot see which one ruled cannot tell an unsettled candidate from one
#: whose evidence simply lives elsewhere.
SETTLED_BY_REBENCH = "rebench"
SETTLED_BY_INTEGRATE = "integrate"
SETTLED_BY_LANE = "lane"

#: How one KERNEL entry ran. The subject is the visit, not the candidates it
#: left for a later gate: whether the agent ran, and whether it kept anything
#: this stage measured a gain on. Both routes reach the same three, and each
#: names evidence the visit itself holds -- a verdict that waited on a gate
#: running after the visit closed could never be stated at all, which is what
#: an "unconcluded" value would have been hiding.
VERDICT_IMPROVED = "improved"
VERDICT_NO_IMPROVEMENT = "no_improvement"
VERDICT_FAILED = "failed"

#: The two statuses a closed visit reports. Separate from the verdict above:
#: the verdict says what the visit found, the status whether it got to the end
#: to find it. A visit can deliver a measured win and then raise, and calling
#: that "succeeded" because of the win, or "no_improvement" because of the
#: raise, each hides one half of what happened.
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

#: A lane run that reports one of these produced no candidate to judge. Owned
#: here because this is where the distinction is spent: the producers that
#: stamp a run's ``micro_decision`` ask the same question, and answering it
#: from a second list let the two drift.
LANE_FAULTED_STATUSES = frozenset({"failed", "error", "failure", "faulted", "timeout", "aborted", "crashed"})

_ACCEPTANCE_AUTHORED = "authored"
_ACCEPTANCE_ENV = "env"

__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_PHASE",
    "EVENT_TYPE",
    "LANE_FAULTED_STATUSES",
    "LEDGER_GEAK",
    "OUTCOME_ADOPTED",
    "OUTCOME_FAILED",
    "OUTCOME_NEEDS_REVIEW",
    "OUTCOME_REJECTED",
    "PRODUCER",
    "PRODUCER_GEAK",
    "ROUTE_FORGE",
    "ROUTE_GEAK",
    "SECTION_EVENT",
    "SECTION_GEAK_ACCEPTANCE",
    "SECTION_GEAK_ATTEMPT",
    "SECTION_GEAK_DISCOVERY",
    "SECTION_INTEGRATE",
    "SECTION_LANE_RUN",
    "SECTION_REBENCH",
    "SECTION_TRACE_ANALYZE",
    "SETTLED_BY_INTEGRATE",
    "SETTLED_BY_LANE",
    "SETTLED_BY_REBENCH",
    "SOURCE_FUSION",
    "SOURCE_GEAK_AUTHORED_KERNEL",
    "SOURCE_GEAK_ENV_SELECTION",
    "SOURCE_GEMM_TUNING",
    "SOURCE_KERNEL_REWRITE",
    "TRACE_ANALYZE_TRIGGER_BUS_REQUEST",
    "VERDICT_FAILED",
    "VERDICT_IMPROVED",
    "VERDICT_NO_IMPROVEMENT",
    "assemble_kernel_ext",
    "kernel_event_id",
]


def kernel_event_id(macro_cycle: Any) -> str:
    """Build ``kernel_agent:{macro_cycle}:kernel``. Raises ``ValueError`` if
    ``macro_cycle`` is not a non-negative integer."""
    return event_id(EVENT_PHASE, macro_cycle, EVENT_COMPONENT)


def _republish_closed_event(event: str) -> None:
    """Re-assemble a closed event so a fragment written after it is published."""
    from .construct import republish_closed_event

    republish_closed_event(
        EventHeader(EVENT_TYPE, event, EVENT_KIND),
        section=SECTION_EVENT,
        load_parts=lambda: event_parts(EVENT_SECTIONS, event=event),
        assemble=assemble_kernel_ext,
        end_time=lambda _parts, header: _text(header.get("end_time")),
    )


def _integrate_e2e(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Project one kernel's integrate rows into the standing e2e sub-result.

    A patch can be gated more than once, so the sub-result reports the verdict
    that stands: the last one to settle, with that row's own gain (``None``
    when it measured none). Earlier measurements stay on the integrate rows.

    Returns:
        dict[str, Any] | None: The sub-result, or ``None`` when the kernel was
            never gated -- which is not the same as being gated and rejected.
    """
    if not rows:
        return None
    ordered = sort_rows(rows, keys=("settled_at", "attempt_count"))
    last = ordered[-1]
    gain = last.get("gain_pct")
    return {
        "integrated": str(last.get("decision") or "").upper() == "KEEP",
        "e2e_gain_pct": gain if isinstance(gain, (int, float)) else None,
        "validated": last.get("accuracy_pass") if isinstance(last.get("accuracy_pass"), bool) else None,
        "decision": _text(last.get("decision")),
        "patch_path": _text(last.get("patch_path")),
        "target_file": _text(last.get("target_file")),
        "graded_objective": _text(last.get("graded_objective")) or None,
    }


#: The keys a normalized attempt always carries. Both routes fill the same
#: shape so a reader replaying the visit does not have to know which producer
#: wrote a row before it can read one.
_ATTEMPT_TEMPLATE: dict[str, Any] = {
    "attempt_id": "",
    "route": "",
    "source_kind": "",
    "kernel_id": "",
    "name": "",
    "status": "",
    "dispatched": True,
    "skip_reason": "",
    "started_at": "",
    "ended_at": "",
    "duration_sec": None,
    "backend": "",
    "backends_tried": [],
    "speedup": None,
    # The gain this stage measured on the candidate, as a percentage. The two
    # routes measure on different instruments -- a forge lane times the kernel
    # itself, a GEAK candidate is re-benched end to end -- and state it in
    # different units, so it is normalized here rather than at every reader.
    "gain_pct": None,
    # The instrument behind ``gain_pct``; ``None`` whenever it is.
    "gain_axis": None,
    "compile_status": "",
    "correctness": None,
    "artifact_path": "",
    "error_class": "",
    "failure_reason": "",
    "micro_decision": "",
    "accepted": False,
    "rebench_ref": "",
    "integrate_ref": "",
    "e2e": None,
    "outcome": "",
    "settled_by": "",
    "unsettled_reason": "",
    "detail": {},
}

#: Fields a lane row carries that the normalized attempt already states under
#: another name. Everything else on the row is lane-specific and lands in
#: ``detail``.
_LANE_PROMOTED = frozenset(
    {
        "event_id",
        "source_kind",
        "run_id",
        "status",
        "started_at",
        "ended_at",
        "duration_sec",
        "micro_decision",
        "integrate_ref",
        "error_class",
        "failure_reason",
        "kernel_id",
        "kernel_name",
        "backends_tried",
        "adopted_backend",
        "speedup",
        "gain_pct",
        "e2e",
    }
)

#: The same, for a GEAK kernel row.
_GEAK_PROMOTED = frozenset(
    {
        "event_id",
        "ordinal",
        "kernel_id",
        "name",
        "dispatched",
        "skip_reason",
        "started_at",
        "ended_at",
        "duration_sec",
        "backends",
        "e2e",
    }
)

#: Micro decisions that say the lane declined its own candidate.
_LANE_DECLINED = frozenset({"REVERT", "SKIPPED", "FAILED", "DROP", "REJECT"})


def _attempt(**fields: Any) -> dict[str, Any]:
    """One normalized attempt, defaulted from the shared template."""
    row: dict[str, Any] = dict(_ATTEMPT_TEMPLATE)
    row["backends_tried"] = []
    row["detail"] = {}
    row.update(fields)
    return row


def _lane_attempt(row: Mapping[str, Any], integrate_by_id: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Normalize one forge lane row.

    A lane states its own verdict in ``micro_decision``; anything it did not
    keep it declined, and that is what ``accepted`` records. Fusion says the
    same thing a second way, through ``applied``. A lane that states no e2e
    result of its own takes the one its integrate gate ruled, and only an
    authored integration reference can link a lane to that gate.
    """
    decision = str(row.get("micro_decision") or "").upper()
    integrate_ref = _text(row.get("integrate_ref"))
    gate_rows = integrate_by_id.get(integrate_ref, []) if integrate_ref else []
    gain = lane_gain_pct(row)
    return _attempt(
        attempt_id=_text(row.get("run_id")),
        route=ROUTE_FORGE,
        source_kind=_text(row.get("source_kind")),
        kernel_id=_text(row.get("kernel_id")),
        name=_text(row.get("kernel_name")) or _text(row.get("target_module")) or _text(row.get("tuner")),
        status=_text(row.get("status")),
        started_at=_text(row.get("started_at")),
        ended_at=_text(row.get("ended_at")),
        duration_sec=_float_or_none(row.get("duration_sec")),
        backend=_text(row.get("adopted_backend")),
        backends_tried=[str(item) for item in _as_list(row.get("backends_tried"))],
        speedup=_float_or_none(row.get("speedup")),
        gain_pct=gain,
        gain_axis=lane_gain_axis(row, gain),
        error_class=_text(row.get("error_class")),
        failure_reason=_text(row.get("failure_reason")),
        micro_decision=decision,
        accepted=decision == "KEEP" or bool(row.get("applied")),
        integrate_ref=integrate_ref,
        e2e=_as_dict(row.get("e2e")) or _integrate_e2e(gate_rows),
        detail={key: value for key, value in row.items() if key not in _LANE_PROMOTED},
    )


def _geak_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one GEAK kernel row.

    The backend block is where GEAK states how the attempt itself went; the
    e2e block is what it then claimed about the whole run. They are read into
    the same fields the forge lanes fill, so the two routes can be replayed
    against one set of names.
    """
    backend = _as_dict(row.get("backend_result"))
    e2e = _as_dict(row.get("e2e"))
    detail = {key: value for key, value in row.items() if key not in _GEAK_PROMOTED}
    detail.pop("backend_result", None)
    detail.pop("micro_speedup", None)
    detail["baseline_us"] = _float_or_none(backend.get("baseline_us"))
    detail["candidate_us"] = _float_or_none(backend.get("candidate_us"))
    return _attempt(
        attempt_id=_text(row.get("kernel_id")),
        route=ROUTE_GEAK,
        source_kind=SOURCE_GEAK_AUTHORED_KERNEL,
        kernel_id=_text(row.get("kernel_id")),
        name=_text(row.get("name")) or _text(row.get("kernel_id")),
        status=_text(backend.get("status")),
        dispatched=bool(row.get("dispatched", True)),
        skip_reason=_text(row.get("skip_reason")),
        started_at=_text(row.get("started_at")),
        ended_at=_text(row.get("ended_at")),
        duration_sec=_float_or_none(row.get("duration_sec")),
        backend=_text(backend.get("backend")),
        backends_tried=[str(item) for item in _as_list(row.get("backends"))],
        speedup=_float_or_none(row.get("micro_speedup")),
        compile_status=_text(backend.get("compile_status")),
        correctness=backend.get("correctness") if isinstance(backend.get("correctness"), bool) else None,
        artifact_path=_text(backend.get("artifact_path")),
        error_class=_text(backend.get("error_class")),
        failure_reason=_text(e2e.get("rejection_reason")),
        micro_decision=str(e2e.get("decision") or "").upper(),
        rebench_ref="",
        e2e=e2e or None,
        detail=detail,
    )


def _acceptance_attempt(entry: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    """Normalize one GEAK acceptance that no kernel attempt accounts for.

    An env selection is never a kernel attempt, and GEAK can also name a
    kernel its journey did not report. Both are candidates the rebench will
    rule on, so they need an attempt row of their own.
    """
    return _attempt(
        attempt_id=_acceptance_identity(entry, ordinal),
        route=ROUTE_GEAK,
        source_kind=_text(entry.get("source_kind")),
        kernel_id=_text(entry.get("kernel_id")),
        name=_text(entry.get("short_name")) or _text(entry.get("selection")) or _text(entry.get("cand_tag")),
        status="",
        accepted=True,
        detail={
            key: value
            for key, value in entry.items()
            if key not in {"event_id", "ordinal", "source_kind", "acceptance_kind", "kernel_id", "short_name"}
        },
    )


def _acceptance_identity(row: Mapping[str, Any], ordinal: int) -> str:
    """Name one acceptance stably enough to key its fragment.

    Returns:
        str: The kernel or selection the acceptance names, falling back to its
            reported position when GEAK named it nothing at all.
    """
    for field in ("kernel_id", "short_name", "cand_tag", "selection"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return f"ordinal{int(ordinal)}"


def _merge_acceptances(
    attempts: list[dict[str, Any]],
    acceptances: Mapping[str, list[dict[str, Any]]],
    *,
    geak_ref: str,
) -> list[dict[str, Any]]:
    """Mark the attempts GEAK accepted, and give a row to those it has none for.

    An acceptance is not a separate try at the same kernel -- it is GEAK saying
    which of the attempts above it is handing over. Folding it onto the attempt
    keeps one row per try, so a campaign that tried twenty kernels and accepted
    two reads as eighteen rejections rather than two candidates and no history.

    Returns:
        list[dict[str, Any]]: Attempt rows for the acceptances nothing matched
            -- env selections, which are never kernel attempts, and kernels
            GEAK accepted without reporting a try for them.
    """
    by_identity: dict[str, dict[str, Any]] = {}
    for row in attempts:
        if row["route"] != ROUTE_GEAK:
            continue
        for key in (row["kernel_id"], row["name"]):
            if key:
                by_identity.setdefault(str(key), row)

    unmatched: list[dict[str, Any]] = []
    for kind in (_ACCEPTANCE_AUTHORED, _ACCEPTANCE_ENV):
        for ordinal, entry in enumerate(acceptances.get(kind, [])):
            target = _authored_target(entry, by_identity) if kind == _ACCEPTANCE_AUTHORED else None
            if target is None:
                row = _acceptance_attempt(entry, ordinal=ordinal)
                row["rebench_ref"] = geak_ref
                unmatched.append(row)
                continue
            target["accepted"] = True
            target["rebench_ref"] = geak_ref
            target["detail"]["acceptance"] = {
                key: value
                for key, value in entry.items()
                if key not in {"event_id", "ordinal", "acceptance_kind", "source_kind", "kernel_id", "short_name"}
            }
    return unmatched


def _authored_target(entry: Mapping[str, Any], by_identity: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return the attempt row an authored acceptance names, matched by its first known identity."""
    keys = (str(entry.get(field) or "") for field in ("kernel_id", "short_name", "cand_tag"))
    return next((by_identity[key] for key in keys if key in by_identity), None)


def _settle(
    attempts: list[dict[str, Any]],
    verdicts: Mapping[str, Mapping[str, Any]],
    *,
    geak_conflicted: bool,
) -> None:
    """Settle every attempt against whichever evidence ruled on it.

    The two routes are judged on different instruments and neither is a
    fallback for the other: a GEAK candidate is re-benched end to end, a forge
    candidate is kept or declined by the lane that timed it and may later be
    re-ruled by the integrate gate. All three are read here so one vocabulary
    covers both routes, and ``settled_by`` names the instrument rather than
    leaving a reader to infer it from the route.

    The integrate gate is the one piece of evidence that can arrive after this
    visit has closed -- it records into the event of the cycle it settles in.
    So a forge keep is settled against the lane when no gate has ruled yet:
    waiting for one would leave the visit's own conclusion permanently
    unstated, since the later verdict is never written back to this event.
    """
    for row in attempts:
        if str(row.get("status") or "").lower() in LANE_FAULTED_STATUSES:
            row["outcome"] = OUTCOME_FAILED
            row["settled_by"] = SETTLED_BY_LANE
            continue
        if not row.get("dispatched", True):
            row["outcome"] = OUTCOME_REJECTED
            row["settled_by"] = SETTLED_BY_LANE
            continue
        if not row.get("accepted"):
            # The producer declined its own candidate, so no gate ever saw it.
            row["outcome"] = OUTCOME_REJECTED
            row["settled_by"] = SETTLED_BY_LANE
            continue
        ref = str(row.get("rebench_ref") or "")
        rebench = _as_dict(verdicts.get(ref)) if ref else {}
        decision = str(rebench.get("decision") or "")
        if decision:
            row["settled_by"] = SETTLED_BY_REBENCH
            # The rebench is the instrument that measured this candidate, so
            # its delta is the gain, not whatever GEAK claimed for itself.
            stamp_rebench_gain(row, rebench.get("delta_pct"))
            if decision == REBENCH_VALIDATED:
                row["outcome"] = OUTCOME_ADOPTED
            elif decision in {REBENCH_NO_MATERIAL, REBENCH_NO_PROMOTE}:
                row["outcome"] = OUTCOME_REJECTED
            else:
                row["outcome"] = OUTCOME_NEEDS_REVIEW
                row["unsettled_reason"] = "rebench_inconclusive"
            continue
        e2e = _as_dict(row.get("e2e"))
        gate = str(e2e.get("decision") or "").upper()
        if gate:
            row["settled_by"] = SETTLED_BY_INTEGRATE
            # The gate measured the patch end to end, which outranks the lane's
            # own timing of the kernel in isolation.
            stamp_gate_gain(row, e2e.get("e2e_gain_pct"), e2e.get("graded_objective"))
            row["outcome"] = OUTCOME_ADOPTED if gate == "KEEP" else OUTCOME_REJECTED
            continue
        if str(row.get("micro_decision") or "").upper() in _LANE_DECLINED:
            row["outcome"] = OUTCOME_REJECTED
            row["settled_by"] = SETTLED_BY_LANE
            continue
        if row.get("route") == ROUTE_FORGE:
            row["outcome"] = OUTCOME_ADOPTED
            row["settled_by"] = SETTLED_BY_LANE
            continue
        row["outcome"] = OUTCOME_NEEDS_REVIEW
        row["unsettled_reason"] = "rebench_conflict" if geak_conflicted else "no_gate"


def _delivered(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the visit hands to the stack: one row per kept candidate.

    ``gain_pct`` is what this stage measured, the same number the verdict is
    read off. The integrate gate's own end-to-end figure is a later and
    separate measurement, and stays on the attempt's ``e2e`` block rather than
    being averaged into this one.
    """
    return [
        {
            "route": _text(row.get("route")),
            "source_kind": _text(row.get("source_kind")),
            "ref": _text(row.get("attempt_id")),
            "kernel_id": _text(row.get("kernel_id")),
            "gain_pct": _float_or_none(row.get("gain_pct")),
            "gain_axis": row.get("gain_axis"),
            "settled_by": _text(row.get("settled_by")),
        }
        for row in attempts
        if row.get("outcome") == OUTCOME_ADOPTED
    ]


def _gain_pct(after: float | None, anchor: float | None) -> float | None:
    """The gain from ``anchor`` to ``after``, as a percentage.

    Returns:
        float | None: The gain, or ``None`` when either end is missing or the
            anchor is not positive -- a percentage of nothing states nothing.
    """
    if after is None or anchor is None or anchor <= 0:
        return None
    return round((after - anchor) / anchor * 100.0, 4)


def _throughput(outcome_block: Mapping[str, Any]) -> dict[str, Any]:
    """Every anchor the visit is read against, with the gain against each.

    The gains are derived here rather than stated by the phase, and from the
    anchors sitting beside them, so a reader never has to infer a denominator
    and a further comparison is one more pair computed from what is already
    recorded.
    """
    before = _float_or_none(outcome_block.get("tput_before"))
    after = _float_or_none(outcome_block.get("tput_after"))
    session_baseline = _float_or_none(outcome_block.get("session_baseline_tput"))
    return {
        "before": before,
        "after": after,
        "session_baseline": session_baseline,
        "gain_pct": _gain_pct(after, before),
        "session_gain_pct": _gain_pct(after, session_baseline),
    }


def _derive_verdict(attempts: list[dict[str, Any]], *, failure: Mapping[str, Any]) -> tuple[str, str]:
    """State how the visit ran, and why when it improved nothing.

    The phase is not asked for this. A visit improved something because an
    instrument measured a gain on a candidate the visit kept, and the rows
    above already carry both halves; a phase that reported its own verdict
    could contradict them.

    A kept candidate with no measured gain is still a delivery -- it is in
    ``delivered`` -- but it is not an improvement, because nothing measured
    one. That is the line this verdict draws.

    A measured win outranks a recorded fault. A fault does not end the visit
    -- a raising tick is filed against the session's crash count and the loop
    carries on -- so a visit can blow up somewhere and still hand a measured
    candidate to the stack. Calling that failed would bury the delivery; the
    fault stays readable in ``error_class`` and ``failed_stage``, which is
    what says the win was not come by cleanly.

    Returns:
        tuple[str, str]: The verdict, and the reason -- empty unless the verdict
            needs one to be actionable.
    """
    kept = [row for row in attempts if row.get("outcome") == OUTCOME_ADOPTED]
    if any((_float_or_none(row.get("gain_pct")) or 0.0) > 0.0 for row in kept):
        return VERDICT_IMPROVED, ""
    if failure:
        return VERDICT_FAILED, _text(failure.get("message")) or _text(failure.get("error_class"))
    if attempts and all(row.get("outcome") == OUTCOME_FAILED for row in attempts):
        return VERDICT_FAILED, "every attempt failed before it could be judged"
    if not attempts:
        return VERDICT_NO_IMPROVEMENT, "the visit produced no candidate"
    if not kept:
        return VERDICT_NO_IMPROVEMENT, "no candidate was kept"
    return VERDICT_NO_IMPROVEMENT, "nothing measured a gain on the candidates kept"


#: The event status each verdict closes on, for a visit that returned. A visit
#: that kept nothing still ran to a conclusion, so the distinction the status
#: draws is whether the visit finished, not whether it found a win. A visit
#: that did not return is not in here: its close names the stage it could not
#: get past, and that outranks anything the rows add up to.
_STATUS_BY_VERDICT = {
    VERDICT_IMPROVED: STATUS_SUCCEEDED,
    VERDICT_NO_IMPROVEMENT: STATUS_SUCCEEDED,
    VERDICT_FAILED: STATUS_FAILED,
}


def _geak_rebench(rebench_rows: list[dict[str, Any]], stated: Mapping[str, Any]) -> dict[str, Any]:
    """How GEAK's re-measurement campaign was bounded, and the verdict its acceptances settle against.

    GEAK may rebench the same candidate up to its per-cycle ceiling, so unlike a
    forge lane it can end the entry holding several settled verdicts. Taking the
    newest would let a KEEP after a REVERT read as an adoption; two settled
    verdicts that disagree is a fact worth seeing, so the candidate stays
    pending and neither verdict is honoured: ``settled_against`` is empty and
    ``conflicting_decisions`` names them.
    """
    ledger = group_rows(rebench_rows, "ledger").get(LEDGER_GEAK, [])
    settled = [row for row in ledger if str(row.get("decision") or "")]
    decisions = sorted({str(row.get("decision")) for row in settled})
    rebench = {
        "required": bool(ledger) or bool(stated),
        "attempts_used": len(ledger),
        "settled_against": str(settled[-1].get("attempt_id") or "") if len(decisions) == 1 else "",
        "final_status": _text(stated.get("final_status")),
        "final_error_class": _text(stated.get("final_error_class")),
        "final_error": _text(stated.get("final_error")),
    }
    if len(decisions) > 1:
        rebench["conflicting_decisions"] = decisions
    return rebench


def _forge(
    header: Mapping[str, Any],
    rows: Mapping[str, list[dict[str, Any]]],
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> dict[str, Any] | None:
    """What only forge did on this visit, or ``None`` when forge did not run.

    The re-profile dispatched the roofline executor inline, so its rows are in
    this event under the task id the re-profile recorded.
    """
    reprofile = _as_dict(header.get("forge_reprofile")) or None
    if reprofile:
        reprofile = {
            **reprofile,
            "run": assemble_roofline_action(parts, event=event, task_id=str(reprofile.get("task_id") or "")),
        }
    trace_runs = wire_rows(rows[SECTION_TRACE_ANALYZE])
    discovered = wire_rows(rows[SECTION_DISCOVERED], drop=("event_id", "rank"))
    if not (reprofile or trace_runs or rows[SECTION_LANE_RUN] or discovered):
        return None
    return {
        "engaged": True,
        "reprofile": reprofile,
        "trace_analyze_runs": trace_runs,
        "discovered_kernels": discovered,
        "recommended_kernels": [row for row in discovered if row.get("selected")],
    }


def _geak(
    header: Mapping[str, Any],
    rows: Mapping[str, list[dict[str, Any]]],
    acceptances: Mapping[str, list[dict[str, Any]]],
    rebench: Mapping[str, Any],
) -> dict[str, Any] | None:
    """What only GEAK did on this visit, in causal order, or ``None`` when GEAK did not run."""
    # ``acceptance_kind`` and ``source_kind`` are the fields that chose which of the two arrays a row landed in, so on
    # the wire the array itself says it.
    acceptance_drop = ("event_id", "ordinal", "acceptance_kind", "source_kind")
    authored = wire_rows(acceptances.get(_ACCEPTANCE_AUTHORED, []), drop=acceptance_drop)
    env_selections = wire_rows(acceptances.get(_ACCEPTANCE_ENV, []), drop=acceptance_drop)
    handoff = _as_dict(header.get("geak_handoff")) or None
    delegation = _as_dict(header.get("geak_delegation")) or None
    claim = _as_dict(header.get("geak_claim")) or None
    product = _as_dict(header.get("geak_product")) or None
    if not (handoff or delegation or claim or product or rebench["required"] or rows[SECTION_GEAK_ATTEMPT] or authored):
        return None
    if claim is not None or authored or env_selections:
        claim = {
            **(claim or {}),
            "authored_kernels": authored,
            "env_selections": env_selections,
            "kernels_optimized": len(authored),
            "accepted_heads_count": sum(1 for row in authored + env_selections if row.get("lane") == "headQueue"),
        }
    return {
        "engaged": True,
        "handoff": handoff,
        "delegation": delegation,
        "discovery_runs": wire_rows(rows[SECTION_GEAK_DISCOVERY]),
        "claim": claim,
        "product": product,
        "rebench": rebench,
    }


def _outcome(header: Mapping[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """How the visit ran: the verdict its settled attempts support, beside what the phase stated at exit."""
    stated = _as_dict(header.get("outcome"))
    failure = _as_dict(header.get("failure"))
    verdict, reason = _derive_verdict(attempts, failure=failure)
    return {
        "route": _text(_as_dict(header.get("entry")).get("route")) or "",
        "verdict": verdict,
        "reason": reason,
        "error_class": _text(failure.get("error_class")),
        "failed_stage": _text(failure.get("stage")),
        "exit_reason": _text(stated.get("exit_reason")),
        "throughput": _throughput(stated),
        "cumulative_gain_validated_out": _float_or_none(stated.get("cumulative_gain_validated_out")),
        "stack_depth_out": _int_or_none(stated.get("stack_depth_out")),
        "delivered": _delivered(attempts),
        "stack_delta": _as_dict(stated.get("stack_delta")) or {"added": [], "removed": []},
    }


#: The fields one event's rows are ordered by, per section.
_ROW_ORDER: dict[str, tuple[str, ...]] = {
    SECTION_TRACE_ANALYZE: ("ts", "run_id"),
    SECTION_LANE_RUN: ("started_at", "run_id"),
    SECTION_REBENCH: ("attempt_id",),
    SECTION_GEAK_DISCOVERY: ("ordinal", "source"),
    SECTION_GEAK_ATTEMPT: ("ordinal", "kernel_id"),
    SECTION_GEAK_ACCEPTANCE: ("ordinal", "kernel_id", "selection"),
    SECTION_DISCOVERED: ("snapshot_id", "rank"),
    SECTION_INTEGRATE: ("settled_at", "integration_id"),
}


def assemble_kernel_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one kernel event's ``ext`` out of its recorded rows."""
    header = event_header(parts, SECTION_EVENT, event)
    rows = {
        section: sort_rows(rows_for_event(parts[section], event), keys=keys) for section, keys in _ROW_ORDER.items()
    }

    acceptances = group_rows(rows[SECTION_GEAK_ACCEPTANCE], "acceptance_kind")
    rebench = _geak_rebench(rows[SECTION_REBENCH], _as_dict(header.get("geak_rebench")))
    integrate_by_id = group_rows(rows[SECTION_INTEGRATE], "integration_id")
    attempts = [_lane_attempt(row, integrate_by_id) for row in rows[SECTION_LANE_RUN]]
    attempts.extend(_geak_attempt(row) for row in wire_rows(rows[SECTION_GEAK_ATTEMPT]))
    attempts.extend(_merge_acceptances(attempts, acceptances, geak_ref=rebench["settled_against"]))
    # A row whose producer stated no start time sorts after the ones that did,
    # rather than ahead of them where an empty string would put it.
    attempts.sort(
        key=lambda row: (not row.get("started_at"), str(row.get("started_at") or ""), str(row.get("attempt_id") or ""))
    )
    verdicts = {str(row.get("attempt_id")): row for row in rows[SECTION_REBENCH] if row.get("attempt_id")}
    _settle(attempts, verdicts, geak_conflicted="conflicting_decisions" in rebench)

    forge = _forge(header, rows, parts, event=event)
    geak = _geak(header, rows, acceptances, rebench)
    outcome = _outcome(header, attempts)
    ext: dict[str, Any] = {
        "macro_cycle": _int_or_none(header.get("macro_cycle")) or 0,
        "in_flight_stage": _text(header.get("in_flight_stage")),
        "entry": _as_dict(header.get("entry")),
        "failure": _as_dict(header.get("failure")) or None,
        # Every candidate either route produced, in one shape. The two routes
        # run different machinery and state their results under different
        # names; normalizing here is what lets one reader replay the visit
        # without knowing which producer wrote a given row.
        "attempts": attempts,
        # The re-measurements that settled them. Only GEAK runs one today, but
        # the attempts above point at these rows by id, so they sit beside the
        # attempts rather than inside the route that happened to dispatch them.
        "rebench": wire_rows(rows[SECTION_REBENCH]),
        # Sibling of the two routes rather than nested in either: the gate is
        # the orchestrator's and rules on whatever the visit produced, so a
        # patch from GEAK and one from forge go through the same one.
        "integrate": wire_rows(rows[SECTION_INTEGRATE], drop=("event_id",)),
        # The end-to-end measurements the visit ran as sub-steps -- candidate
        # A/B, vLLM shape capture. Each went through the
        # baseline executor inline, so its rows are in this event, and its task
        # id names what it measured. Without them the gate rows above state a
        # verdict whose measurement is nowhere in the breakdown.
        "measurements": assemble_baseline_actions(parts, event=event),
        "geak": geak,
        "forge": forge,
        "outcome": outcome,
    }
    duration = _float_or_none(header.get("duration_sec"))
    if duration is not None:
        ext["duration_sec"] = duration

    # The verdict describes a visit that ended. Assembly also runs against an
    # event still in flight -- finalize rebuilds one from its rows -- and that
    # event has reached no verdict yet, whatever its rows so far add up to.
    # Leaving the derived one in place had such an event claiming it improved
    # something, contradicting the status beside it; the fault block stays,
    # since a fault recorded before the kill is exactly what a reader wants.
    if not _text(header.get("end_time")):
        outcome["verdict"] = ""
        outcome["reason"] = ""
        return ext, "running"
    # A visit that could not get past its stage said so when it closed, and
    # that outranks the verdict: the rows can hold a measured win the visit
    # delivered before it raised, and reporting the event as succeeded on the
    # strength of that win is the crashed phase reading as a clean one. The
    # verdict keeps the win, because an instrument did measure it.
    return ext, _text(header.get("closed_status")) or _STATUS_BY_VERDICT[outcome["verdict"]]
