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

Two derivations stay in assembly for the same reason. Settlement -- which
candidate was adopted -- is a join between a lane row and the rebench row that
re-measured it, and neither exists yet when the other is written. The event
status is read off what the rebenches measured, so it cannot be known before
the last one lands. Both are computed once, from the rows, at the close.

Every ``*_at`` argument in this module is an ISO timestamp.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any

from .assembler import EVENT_SECTIONS, event_parts
from .event_fields import (
    analysis_detail as _analysis_detail,
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    failure_row as _failure_row,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    now_iso_seconds as _now_iso,
    text_or_none as _text,
)
from .event_ids import event_id
from .event_rows import group_rows, rows_for_event, sort_rows, wire_rows
from .event_sink import make_sink
from .event_timeline import finish_event, open_event
from .roofline_event import assemble_roofline_action

log = logging.getLogger(__name__)

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

_ACTIVE: ContextVar["KernelEventRecorder | None"] = ContextVar("kernel_event_active", default=None)

ROUTE_GEAK = "geak"
ROUTE_FORGE = "forge"

# The five candidate producers a KERNEL entry can adopt from.
SOURCE_KERNEL_REWRITE = "kernel_rewrite"
SOURCE_FUSION = "fusion"
SOURCE_GEMM_TUNING = "gemm_tuning"
SOURCE_GEAK_AUTHORED_KERNEL = "geak_authored_kernel"
SOURCE_GEAK_ENV_SELECTION = "geak_env_selection"

_SOURCE_KINDS = (
    SOURCE_KERNEL_REWRITE,
    SOURCE_FUSION,
    SOURCE_GEMM_TUNING,
    SOURCE_GEAK_AUTHORED_KERNEL,
    SOURCE_GEAK_ENV_SELECTION,
)

#: Which wire array each forge source's rows assemble into. Recording tags the
#: row with its ``source_kind`` and assembly reads the array name off this map,
#: so a lane is added in one place rather than at every call site that writes
#: into it.
LANE_BY_SOURCE = {
    SOURCE_KERNEL_REWRITE: "kernel_rewrites",
    SOURCE_FUSION: "fusion_runs",
    SOURCE_GEMM_TUNING: "gemm_tuning_runs",
}

#: The two rebench ledgers. They are separate wire arrays because they answer
#: different questions -- forge re-measures its own candidates one at a time,
#: GEAK re-measures a whole delegated config against a per-cycle attempt
#: ceiling -- so the row records which ledger asked for it rather than leaving
#: assembly to infer it from the source kind.
LEDGER_FORGE = "forge"
LEDGER_GEAK = "geak"

OUTCOME_ADOPTED = "adopted"
OUTCOME_NEEDS_REVIEW = "needs_review"
OUTCOME_REJECTED = "rejected"
OUTCOME_IN_FLIGHT = "in_flight"
OUTCOME_UNATTEMPTED = "unattempted"

# A rebench either validated the candidate, found its win immaterial, measured it truthfully without beating
# current_best, or could not conclude because the config it was asked to reproduce did not engage.
REBENCH_VALIDATED = "validated"
REBENCH_NO_MATERIAL = "no_material"
REBENCH_NO_PROMOTE = "no_promote"
REBENCH_FALLBACK = "fallback"

# A rebench that reports one of these measured nothing, so it concluded nothing about the candidate it was dispatched
# for.
_REBENCH_FAULTED_STATUSES = frozenset({"failed", "error", "failure", "faulted", "timeout", "aborted"})

_ACCEPTANCE_AUTHORED = "authored"
_ACCEPTANCE_ENV = "env"

__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_PHASE",
    "EVENT_TYPE",
    "LANE_BY_SOURCE",
    "LEDGER_FORGE",
    "LEDGER_GEAK",
    "OUTCOME_ADOPTED",
    "OUTCOME_IN_FLIGHT",
    "OUTCOME_NEEDS_REVIEW",
    "OUTCOME_REJECTED",
    "OUTCOME_UNATTEMPTED",
    "PRODUCER",
    "PRODUCER_GEAK",
    "REBENCH_FALLBACK",
    "REBENCH_NO_MATERIAL",
    "REBENCH_NO_PROMOTE",
    "REBENCH_VALIDATED",
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
    "SOURCE_FUSION",
    "SOURCE_GEAK_AUTHORED_KERNEL",
    "SOURCE_GEAK_ENV_SELECTION",
    "SOURCE_GEMM_TUNING",
    "SOURCE_KERNEL_REWRITE",
    "TRACE_ANALYZE_TRIGGER_BUS_REQUEST",
    "KernelEventRecorder",
    "active_kernel_recorder",
    "assemble_kernel_ext",
    "kernel_event_id",
    "make_kernel_recorder",
    "record_integrate_verdict",
    "record_trace_analyze_request",
]


def active_kernel_recorder() -> "KernelEventRecorder | None":
    """Return the KERNEL visit recorder currently open in this context, if any."""
    return _ACTIVE.get()


def kernel_event_id(macro_cycle: Any) -> str:
    """Build ``kernel_agent:{macro_cycle}:kernel``. Raises ``ValueError`` if
    ``macro_cycle`` is not a non-negative integer."""
    return event_id(EVENT_PHASE, macro_cycle, EVENT_COMPONENT)


def record_integrate_verdict(
    *,
    macro_cycle: Any,
    integration_id: str,
    kernel_id: str,
    decision: str = "",
    status: str = "",
    attempt_count: Any = None,
    fault_count: Any = None,
    gain_pct: Any = None,
    accuracy_pass: Any = None,
    validation_tier: str = "",
    patch_path: str = "",
    target_file: str = "",
    error_class: str = "",
    rejected_reason: str = "",
    retryable: bool = False,
    settled_at: str = "",
    extra_server_args: str = "",
    basis: str = "",
    alignment_status: str = "",
    gain_attributed: Any = None,
) -> None:
    """Record the end-to-end integrate gate's verdict on one patch.

    The gate runs outside the phase that produced the patch: the KERNEL visit
    hands a KEEP to a queue and exits, and the queue is drained by a later
    step which measures the patch end to end and rules on it. So the verdict
    is not the visit's to state, and it does not exist yet when the visit
    closes -- which is why this is a module function taking a cycle rather
    than a method on the visit's recorder. A closed event still accepts row
    fragments; nothing is assembled until the export reads the whole spool.

    The row is attached to the event of the cycle the gate settled in, which
    is the cycle that ran the gate rather than necessarily the one that
    produced the patch: a patch whose integration faulted is retried on its
    own budget and can settle a cycle or more later. ``kernel_id`` is on the
    row so the two can be joined either way, and the row is only written when
    that event exists, so a verdict never mints an event of its own.

    Args:
        decision: The gate's verdict (``KEEP`` / ``REVERT``).
        fault_count: How many attempts never measured the patch fairly.
        rejected_reason: Why the patch was rejected outright, when it was -- an
            exhausted budget rather than a verdict on the code.
        basis: Throughput basis the gain was measured on (``hot`` / ``cold``);
            a gain is meaningless without the baseline behind it.
        gain_attributed: Whether the measured gain is this one kernel's. A
            rebench carrying several kernels measured all of them together, so
            the gain is real but unattributable.
    """
    if not str(integration_id or ""):
        return
    try:
        sink = make_sink(kernel_event_id(macro_cycle), producer=PRODUCER)
        if not sink.has_row(SECTION_EVENT):
            log.debug(
                "kernel timeline: no event %s to hold the integrate verdict for %s",
                sink.event_id,
                integration_id,
            )
            return
        sink.record(
            SECTION_INTEGRATE,
            {
                "integration_id": str(integration_id),
                "kernel_id": str(kernel_id or ""),
                "decision": _text(decision),
                "status": _text(status),
                "attempt_count": _int_or_none(attempt_count),
                "fault_count": _int_or_none(fault_count),
                "gain_pct": _float_or_none(gain_pct),
                "accuracy_pass": accuracy_pass if isinstance(accuracy_pass, bool) else None,
                "validation_tier": _text(validation_tier),
                "patch_path": _text(patch_path),
                "target_file": _text(target_file),
                "error_class": _text(error_class),
                "rejected_reason": _text(rejected_reason),
                "retryable": bool(retryable),
                "settled_at": _text(settled_at),
                "settled_in_macro_cycle": _int_or_none(macro_cycle),
                "extra_server_args": _text(extra_server_args),
                "basis": _text(basis),
                "alignment_status": _text(alignment_status),
                "gain_attributed": gain_attributed if isinstance(gain_attributed, bool) else None,
            },
            row_type=ROW_INTEGRATE,
            natural_ids=str(integration_id),
        )
        _republish_closed_event(sink.event_id)
    except Exception:  # noqa: BLE001 — observability cannot change KERNEL behavior
        log.warning(
            "kernel timeline: could not record the integrate verdict for %s",
            integration_id,
            exc_info=True,
        )


def _write_discovered_kernels(
    sink: Any,
    snapshot: Mapping[str, Any] | None,
    *,
    provenance: str,
) -> None:
    """Write the profiling-rich kernel table from one analysis cache.

    Args:
        sink (Any): The sink to write the rows to.
        snapshot (Mapping[str, Any] | None): A ``last_trace_analyze``-shaped
            cache, or any dict carrying ``hot_kernels_top15``.
        provenance (str): Why this snapshot was taken.
    """
    produced = _as_dict(snapshot)
    rows = _as_list(produced.get("hot_kernels_top15"))
    if not rows:
        rows = _as_list(produced.get("kernel_roofline_top15"))
    if not rows:
        return
    snapshot_id = _int_or_none(produced.get("roofline_snapshot_id"))
    reusable = {str(item) for item in _as_list(produced.get("reusable_native_kernel_ids"))}
    for rank, entry in enumerate(rows):
        if not isinstance(entry, Mapping):
            continue
        row = _discovered_kernel_row(
            entry,
            rank=rank,
            snapshot_id=snapshot_id,
            provenance=provenance,
            reusable_ids=reusable,
        )
        if row is None:
            continue
        sink.record(
            SECTION_DISCOVERED,
            row,
            row_type=ROW_DISCOVERED,
            natural_ids=(str(snapshot_id or 0), str(row.get("kernel_id") or f"rank:{rank}")),
        )


def _write_trace_analyze_run(
    sink: Any,
    *,
    run_id: str,
    trigger: str,
    status: str,
    result: Any,
    requested_by: str = "",
    request_msg_id: str = "",
    trace_input: str = "",
    top_k: Any = None,
    snapshot: dict[str, Any] | None = None,
    cache_hit: bool = False,
) -> None:
    """Write one analysis row and the kernel table it produced.

    ``status`` is ``ok`` or ``failed``.
    """
    produced = _as_dict(snapshot)
    sink.record(
        SECTION_TRACE_ANALYZE,
        {
            "run_id": str(run_id or ""),
            "trigger": _text(trigger),
            "requested_by": _text(requested_by),
            "request_msg_id": _text(request_msg_id),
            "ts": _now_iso(),
            "status": str(status or ""),
            "cache_hit": bool(cache_hit),
            "trace_input": _text(trace_input),
            "top_k": _int_or_none(top_k),
            "roofline_snapshot_id": _int_or_none(produced.get("roofline_snapshot_id")),
            "roofline_baseline_gain_at_snapshot": _float_or_none(produced.get("roofline_baseline_gain_at_snapshot")),
            "steady_state_trace": _text(produced.get("steady_state_trace")),
            "analysis_md_path": _text(produced.get("analysis_md_path")),
            "reusable_native_kernel_ids": [str(item) for item in _as_list(produced.get("reusable_native_kernel_ids"))],
            "trace_validate_ref": None,
            **_analysis_detail(result),
        },
        row_type=ROW_TRACE_ANALYZE,
        natural_ids=str(run_id or ""),
    )
    if produced:
        _write_discovered_kernels(sink, produced, provenance="trace_analyze_run")


def record_trace_analyze_request(
    *,
    macro_cycle: Any,
    run_id: str,
    status: str,
    result: Any,
    requested_by: str = "",
    request_msg_id: str = "",
    trace_input: str = "",
    top_k: Any = None,
    snapshot: dict[str, Any] | None = None,
    cache_hit: bool = False,
) -> None:
    """Record an analysis an agent requested through the bus, not a phase.

    A ``trace_analyze`` dispatched this way advances the roofline snapshot
    counter and replaces the analysis cache, but it opens no roofline event of
    its own -- the counter simply incremented with nothing on the timeline to
    explain it, and the kernel table the analysis produced reached the report
    only by way of the state projection. The row is attached to the visit that
    was running when the request landed, which is the only event that can
    account for it, and is written only when that event exists so a bus
    request never mints one.

    ``status`` is ``ok`` or ``failed``.
    """
    if not str(run_id or ""):
        return
    try:
        sink = make_sink(kernel_event_id(macro_cycle), producer=PRODUCER)
        if not sink.has_row(SECTION_EVENT):
            log.debug(
                "kernel timeline: no event %s to hold the trace_analyze request %s",
                sink.event_id,
                run_id,
            )
            return
        _write_trace_analyze_run(
            sink,
            run_id=run_id,
            trigger=TRACE_ANALYZE_TRIGGER_BUS_REQUEST,
            status=status,
            result=result,
            requested_by=requested_by,
            request_msg_id=request_msg_id,
            trace_input=trace_input,
            top_k=top_k,
            snapshot=snapshot,
            cache_hit=cache_hit,
        )
        _republish_closed_event(sink.event_id)
    except Exception:  # noqa: BLE001 — observability cannot change KERNEL behavior
        log.warning(
            "kernel timeline: could not record the trace_analyze request %s",
            run_id,
            exc_info=True,
        )


def _republish_closed_event(event: str) -> None:
    """Re-assemble a closed event so a fragment written after it is published.

    The export reads the durable timeline rather than re-assembling it, so a
    closed event's published ``ext`` is whatever the close assembled. A row
    that lands afterwards is in the spool but not in the event, and would stay
    that way. Re-assembling and updating the same storage sequence is what
    puts it there -- the same write the close makes, made again.

    An event still running is left alone: its own close will assemble the row
    along with everything else, and publishing a half-finished event here
    would show it closed.
    """
    from ...session.sbd_v6 import timeline_sequence

    parts = event_parts(EVENT_SECTIONS)
    rows = rows_for_event(parts.get(SECTION_EVENT) or [], event)
    header = rows[0] if rows else {}
    end_time = _text(header.get("end_time"))
    if not end_time:
        return
    ext, derived = assemble_kernel_ext(parts, event=event)
    finish_event(
        event_type=EVENT_TYPE,
        event=event,
        sequence=timeline_sequence(header),
        status=_text(header.get("closed_status")) or derived,
        ext=ext,
        kind=EVENT_KIND,
        start_time=_text(header.get("start_time")) or "",
        end_time=end_time,
    )


def _integrate_e2e(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Project one kernel's integrate rows into the standing e2e sub-result.

    A patch can be gated more than once, so the sub-result reports the verdict
    that stands: the last one to settle. The best gain is taken across all of
    them, because a fault that measured nothing does not un-measure what an
    earlier attempt did.

    Returns:
        dict[str, Any] | None: The sub-result, or ``None`` when the kernel was
            never gated -- which is not the same as being gated and rejected.
    """
    if not rows:
        return None
    ordered = sort_rows(rows, keys=("settled_at", "attempt_count"))
    last = ordered[-1]
    gains = [row.get("gain_pct") for row in ordered if isinstance(row.get("gain_pct"), (int, float))]
    return {
        "integrated": str(last.get("decision") or "").upper() == "KEEP",
        "e2e_gain_pct": max(gains) if gains else None,
        "validated": last.get("accuracy_pass") if isinstance(last.get("accuracy_pass"), bool) else None,
        "decision": _text(last.get("decision")),
        "patch_path": _text(last.get("patch_path")),
        "target_file": _text(last.get("target_file")),
    }


def _fold_integrate_into_by_source(
    by_source: dict[str, dict[str, Any]],
    *,
    lanes: Mapping[str, list[dict[str, Any]]],
    acceptances: Mapping[str, list[dict[str, Any]]],
    integrate_by_kernel: Mapping[str, list[dict[str, Any]]],
) -> None:
    """Settle each source's counters against the integrate gate.

    ``adopted`` counts what the rebench validated, which is a different
    question from what the E2E gate then kept: a candidate can clear the micro
    benchmark and never be gated at all. Reporting only the first left a reader
    unable to tell an adoption from a measurement that merely looked good, so
    the gate's own verdict is folded in here as its own counters.

    A kernel gated more than once counts once, under the verdict that stands.

    """
    for counters in by_source.values():
        counters.update({"keeps": 0, "reverts": 0, "micro_only_keeps": 0, "e2e_gain_pct": None})

    def _fold(source: str, kernel_id: str, outcome: str) -> None:
        counters = by_source.get(source)
        if counters is None:
            return
        verdict = _integrate_e2e(list(integrate_by_kernel.get(kernel_id, []))) if kernel_id else None
        if verdict is None:
            # Never gated. Only an adoption is a micro-only keep; a candidate
            # the rebench rejected was not kept at any level.
            if outcome == OUTCOME_ADOPTED:
                counters["micro_only_keeps"] += 1
            return
        decision = str(verdict.get("decision") or "").upper()
        if decision == "KEEP":
            counters["keeps"] += 1
        elif decision:
            counters["reverts"] += 1
        gain = verdict.get("e2e_gain_pct")
        if isinstance(gain, (int, float)):
            best = counters.get("e2e_gain_pct")
            if best is None or gain > best:
                counters["e2e_gain_pct"] = float(gain)

    for lane in lanes.values():
        for row in lane:
            _fold(
                str(row.get("source_kind") or ""),
                str(row.get("kernel_id") or ""),
                str(row.get("outcome") or ""),
            )
    for kind in (_ACCEPTANCE_AUTHORED, _ACCEPTANCE_ENV):
        for entry in acceptances.get(kind, []):
            _fold(
                str(entry.get("source_kind") or ""),
                str(entry.get("kernel_id") or entry.get("short_name") or ""),
                str(entry.get("outcome") or ""),
            )


def _discovered_kernel_row(
    entry: Mapping[str, Any],
    *,
    rank: int,
    snapshot_id: int | None,
    provenance: str,
    reusable_ids: set[str],
) -> dict[str, Any] | None:
    """Normalize one hot-kernel row into the discovered-kernel view.

    Returns the normalized row, or ``None`` without identity.
    """
    kernel_id = _text(entry.get("kernel_id"))
    name = _text(entry.get("name"))
    if not kernel_id and not name:
        return None
    duration = entry.get("duration_us")
    if duration is None:
        duration = entry.get("gpu_time_us")
    call_count = entry.get("call_count")
    if call_count is None:
        call_count = entry.get("count")
    bandwidth = entry.get("bandwidth_utilization_pct")
    if bandwidth is None:
        bandwidth = entry.get("bandwidth_util_pct")
    compute = entry.get("compute_utilization_pct")
    if compute is None:
        compute = entry.get("compute_util_pct")
    intensity = entry.get("arithmetic_intensity")
    if intensity is None:
        intensity = entry.get("flops_per_byte")
    kid = str(kernel_id or "")
    reusable = bool(entry.get("reusable_native_kernel")) or (kid and kid in reusable_ids)
    recommended_backends = [str(item) for item in _as_list(entry.get("recommended_backends"))]
    recommended_actions = [str(item) for item in _as_list(entry.get("recommended_actions"))]
    return {
        "kernel_id": kid,
        "name": _clip(name or ""),
        "rank": int(rank),
        "snapshot_id": snapshot_id,
        "provenance": str(provenance or ""),
        "gpu_pct": _float_or_none(entry.get("gpu_pct")),
        "duration_us": _float_or_none(duration),
        "call_count": _int_or_none(call_count),
        "kernel_category": str(entry.get("kernel_category") or ""),
        "bottleneck": _text(entry.get("bottleneck")),
        "bound_type": _text(entry.get("bound_type")),
        "arithmetic_intensity": _float_or_none(intensity),
        "flops_per_byte": _float_or_none(entry.get("flops_per_byte")),
        "efficiency_percent": _float_or_none(entry.get("efficiency_percent")),
        "bandwidth_util_pct": _float_or_none(bandwidth),
        "compute_util_pct": _float_or_none(compute),
        "source_file": _text(entry.get("source_file")),
        "optimization_notes": _clip(entry.get("optimization_notes") or entry.get("suggestion") or ""),
        "recommended_backends": recommended_backends,
        "recommended_actions": recommended_actions,
        "reusable_native_kernel": reusable,
        "selected": reusable or bool(recommended_backends) or bool(recommended_actions),
    }


def _empty_by_source() -> dict[str, dict[str, int]]:
    """Zeroed per-source counters for every producer the phase can adopt from."""
    return {kind: {"attempted": 0, "adopted": 0, "needs_review": 0, "rejected": 0} for kind in _SOURCE_KINDS}


def _lane_row(
    *,
    source_kind: str,
    run_id: str,
    status: str,
    started_at: str | None,
    ended_at: str | None,
    duration_sec: float | None,
    micro_decision: str | None,
    rebench_ref: str | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    """Build the fields every candidate lane row carries.

    ``outcome`` is left at its unsettled value rather than accepted from the
    caller: a row is adopted or rejected by the rebench that re-measured it, and
    that verdict is a different row which may not have been written yet. It is
    resolved at assembly, once both halves are on disk.

    """
    return {
        "source_kind": str(source_kind),
        "run_id": str(run_id or ""),
        "status": str(status or ""),
        "started_at": _text(started_at),
        "ended_at": _text(ended_at),
        "duration_sec": _float_or_none(duration_sec),
        "micro_decision": _text(micro_decision),
        "rebench_ref": _text(rebench_ref),
        "outcome": OUTCOME_IN_FLIGHT if str(status or "") == "in_flight" else OUTCOME_NEEDS_REVIEW,
        "failure_reason": _text(failure_reason),
    }


def _rebench_row(
    *,
    attempt_id: str,
    source_kind: str,
    ledger: str,
    source_ref: str | None = None,
    idempotency_key: str | None = None,
    task_id: str | None = None,
    dispatched_at: str | None = None,
    settled_at: str | None = None,
    base_tput: float | None = None,
    measured_tput: float | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
    status: str | None = None,
    engagement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one rebench attempt row.

    ``engagement`` is the part the orchestrator already computed but never
    persisted: the GEAK verdict path compares the config fingerprint and the
    overlay digest to decide ``validated`` versus ``fallback``, and dropped both
    booleans on the floor once the decision was made.

    """
    verified = _as_dict(engagement)
    base = _float_or_none(base_tput)
    measured = _float_or_none(measured_tput)
    delta = None
    if base is not None and measured is not None and base > 0:
        delta = round((measured - base) / base * 100.0, 4)
    return {
        "attempt_id": str(attempt_id or ""),
        "source_kind": str(source_kind),
        "ledger": str(ledger),
        "source_ref": _text(source_ref),
        "idempotency_key": _text(idempotency_key),
        "task_id": _text(task_id),
        "dispatched_at": _text(dispatched_at),
        "settled_at": _text(settled_at),
        "base_tput": base,
        "measured_tput": measured,
        "delta_pct": delta,
        "decision": _text(decision),
        "decision_reason": _text(decision_reason),
        "status": _text(status),
        "engagement": {
            "config_matched": verified.get("config_matched"),
            "overlay_loaded": verified.get("overlay_loaded"),
            "expected_cfg_hash": _text(verified.get("expected_cfg_hash")),
            "observed_cfg_hash": _text(verified.get("observed_cfg_hash")),
            "expected_overlay_digest": _text(verified.get("expected_overlay_digest")),
            "observed_overlay_digest": _text(verified.get("observed_overlay_digest")),
        },
    }


def _acceptance_rows(specs: Any) -> list[dict[str, Any]]:
    """Split GEAK's acceptances into authored kernels and env selections."""
    rows: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(_as_list(specs)):
        row = _as_dict(spec)
        if not row:
            continue
        lane = str(row.get("lane") or "")
        delta = _float_or_none(row.get("e2e_delta_pct"))
        op_kind = _text(row.get("op_kind"))
        if str(row.get("kind") or "").strip().lower() == "env":
            rows.append(
                {
                    "acceptance_kind": _ACCEPTANCE_ENV,
                    "source_kind": SOURCE_GEAK_ENV_SELECTION,
                    "ordinal": ordinal,
                    "selection": str(row.get("short_name") or row.get("kernel_id") or row.get("cand_tag") or ""),
                    "op_kind": op_kind,
                    "lane": lane,
                    "e2e_delta_pct": delta,
                }
            )
            continue
        symbol = _text(row.get("short_name")) or _text(row.get("kernel_id"))
        rows.append(
            {
                "acceptance_kind": _ACCEPTANCE_AUTHORED,
                "source_kind": SOURCE_GEAK_AUTHORED_KERNEL,
                "ordinal": ordinal,
                "short_name": _text(row.get("short_name")),
                "kernel_id": _text(row.get("kernel_id")),
                "cand_tag": _text(row.get("cand_tag")),
                "name_source": "symbol" if symbol else "cand_tag",
                "op_kind": op_kind,
                "lane": lane,
                "e2e_delta_pct": delta,
                "alias_collapsed": bool(row.get("alias_collapsed")),
                # The names this acceptance was also written under. Collapsing
                # a twin without keeping its name makes the surviving row
                # unfindable by the name a reader has in hand.
                "aliases": sorted({str(item) for item in _as_list(row.get("aliases")) if str(item)}),
            }
        )
    return rows


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


class KernelEventRecorder:
    """Records the facts of one KERNEL entry, one fragment per row."""

    def __init__(
        self,
        *,
        macro_cycle: int = 0,
        route: str = "",
        route_reason: str = "",
        resumed: bool = False,
        code_revision: str = "",
    ):
        """Bind a recorder to the event of one KERNEL entry."""
        self._event_id = kernel_event_id(macro_cycle)
        self._sink = make_sink(self._event_id, producer=PRODUCER)
        self._geak_sink = make_sink(self._event_id, producer=PRODUCER_GEAK)
        self._t0 = time.monotonic()
        self._start_time = _now_iso()
        self._sequence: int | None = None
        self._closed = False
        self._active_token: Token | None = None
        self._route = str(route or "")
        self._stage = "entry"
        self._sink.record(
            SECTION_EVENT,
            {
                "macro_cycle": int(macro_cycle or 0),
                "route": self._route,
                "in_flight_stage": self._stage,
                "entry": {
                    "route": self._route,
                    "route_reason": str(route_reason or ""),
                    "resumed": bool(resumed),
                    "code_revision": _text(code_revision),
                },
            },
        )

    @property
    def event_id(self) -> str:
        """str: The event id every row this recorder writes is tagged with."""
        return self._event_id

    # ---- lifecycle -------------------------------------------------------

    def begin(
        self,
        *,
        stack_depth_in: Any = None,
        budget_remaining_sec: Any = None,
        tput_before: Any = None,
        session_baseline_tput: Any = None,
        snapshot: dict[str, Any] | None = None,
        snapshot_staleness: str = "",
    ) -> None:
        """Record the entry measurements and put the event on the timeline."""
        inherited = _as_dict(snapshot)
        self._sink.record(
            SECTION_EVENT,
            {
                "entry": {
                    "stack_depth_in": _int_or_none(stack_depth_in),
                    "budget_remaining_sec": _float_or_none(budget_remaining_sec),
                    "roofline_snapshot_id": _int_or_none(inherited.get("roofline_snapshot_id")),
                    "roofline_snapshot_ts": _text(inherited.get("ts")),
                    "roofline_baseline_gain_at_snapshot": _float_or_none(
                        inherited.get("roofline_baseline_gain_at_snapshot")
                    ),
                    "snapshot_staleness": _text(snapshot_staleness),
                },
                "outcome": {
                    "tput_before": _float_or_none(tput_before),
                    "session_baseline_tput": _float_or_none(session_baseline_tput),
                },
            },
        )
        self._sequence = open_event(
            event_type=EVENT_TYPE,
            event=self._event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=self._start_time,
            ext={"route": self._route, "in_flight_stage": self._stage},
        )
        self._active_token = _ACTIVE.set(self)

    def _clear_active(self) -> None:
        """Drop this recorder from the attribution window."""
        token = self._active_token
        if token is not None:
            _ACTIVE.reset(token)
            self._active_token = None

    def record_discovered_kernels(
        self,
        snapshot: Mapping[str, Any] | None,
        *,
        provenance: str = "trace_analyze",
    ) -> None:
        """Record the profiling-rich kernel table this visit is targeting.

        The hot-kernel summary on a roofline event is intentionally thin; the
        KERNEL visit needs the per-kernel profiling fields that decide which
        targets are worth rewriting. This is recorded as its own table rather
        than folded into ``trace_analyze_runs`` so a reader does not have to
        join against a capped top-15 summary missing ``kernel_id``.

        Args:
            snapshot (Mapping[str, Any] | None): A ``last_trace_analyze``-shaped
                cache, or any dict carrying ``hot_kernels_top15``.
            provenance (str): Why this snapshot was taken.
        """
        _write_discovered_kernels(self._sink, snapshot, provenance=provenance)

    def enter_stage(self, stage: str) -> None:
        """Name the stage now in flight so a kill leaves it identifiable.

        This touches the fragment only. The timeline entry keeps the status it
        opened with until the phase ends, and a session killed before then is
        closed out of its fragments by finalize -- which reads this field.
        """
        self._stage = str(stage or "")
        self._sink.record(SECTION_EVENT, {"in_flight_stage": _text(stage)})

    # ---- forge -----------------------------------------------------------

    def record_reprofile(
        self,
        *,
        ran: bool,
        task_kind: str = "",
        trigger: str = "",
        skipped_reason: str = "",
        idempotency_reason: str = "",
        snapshot_landed: bool = False,
        snapshot_id_before: Any = None,
        snapshot_id_after: Any = None,
        task_id: str = "",
    ) -> None:
        """Record the entry re-profile that decides whether analysis is stale."""
        self._sink.record(
            SECTION_EVENT,
            {
                "forge_reprofile": {
                    "ran": bool(ran),
                    "task_id": _text(task_id),
                    "task_kind": _text(task_kind),
                    "trigger": _text(trigger),
                    "skipped_reason": _text(skipped_reason),
                    "idempotency_reason": _text(idempotency_reason),
                    "snapshot_landed": bool(snapshot_landed),
                    "snapshot_id_before": _int_or_none(snapshot_id_before),
                    "snapshot_id_after": _int_or_none(snapshot_id_after),
                }
            },
        )

    def record_trace_analyze_run(
        self,
        *,
        run_id: str,
        trigger: str,
        status: str,
        result: Any,
        requested_by: str = "",
        request_msg_id: str = "",
        trace_input: str = "",
        top_k: Any = None,
        snapshot: dict[str, Any] | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Record an analysis the phase requested for itself.

        This section is normally empty. The entry re-profile dispatches a
        ``roofline`` task by default, which analyses the trace it just captured,
        so the phase's own request is skipped as cached. A non-empty section
        therefore marks the case where the analysis behind a rewrite has no
        roofline event of its own -- previously that request bumped the snapshot
        counter and replaced the cache with nothing on the timeline to explain
        the increment.

        ``reusable_native_kernel_ids`` is recorded because it is the only legal
        source of a ``kernel_id``: the hot-kernel ranking includes vendor
        binaries that dispatch rejects as ``non_reusable_kernel``, so without
        the admitted set there is no way to check afterwards whether the kernel
        the phase went on to rewrite was ever a legitimate target.

        Args:
            run_id: Entry-stable identifier for this analysis.
            trigger: ``pre_run_optimization`` or ``llm_explicit``.
            status: ``ok`` or ``failed``.
            snapshot: The ``last_trace_analyze`` cache the run produced.
        """
        _write_trace_analyze_run(
            self._sink,
            run_id=run_id,
            trigger=trigger,
            status=status,
            result=result,
            requested_by=requested_by,
            request_msg_id=request_msg_id,
            trace_input=trace_input,
            top_k=top_k,
            snapshot=snapshot,
            cache_hit=cache_hit,
        )

    def _record_lane_run(self, row: Mapping[str, Any]) -> None:
        """Write one lane row, keyed by the run it describes."""
        self._sink.record(
            SECTION_LANE_RUN,
            row,
            row_type=ROW_LANE_RUN,
            natural_ids=(str(row.get("source_kind") or ""), str(row.get("run_id") or "")),
        )

    def record_kernel_rewrite(
        self,
        *,
        run_id: str,
        kernel_id: str,
        status: str,
        kernel_name: str = "",
        dispatched: bool = True,
        backends_tried: Any = None,
        adopted_backend: str = "",
        skip_reason: str = "",
        task_group: str = "",
        speedup: Any = None,
        baseline_us: Any = None,
        candidate_us: Any = None,
        compile_status: str = "",
        correctness: Any = None,
        artifact_path: str = "",
        micro_decision: str = "",
        rebench_ref: str = "",
        trace_analyze_ref: str = "",
        started_at: str = "",
        ended_at: str = "",
        duration_sec: Any = None,
        failure_reason: str = "",
    ) -> None:
        """Record one forge source-level kernel rewrite.

        ``adopted_backend`` and ``run_id`` are stated rather than derived. The
        projection had to guess the backend from a speedup plus an artifact path
        and to synthesize an identifier from ``kernel_id:backend:sequence``
        whenever the real attempt id had been lost.
        """
        self._record_lane_run(
            {
                **_lane_row(
                    source_kind=SOURCE_KERNEL_REWRITE,
                    run_id=run_id,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_sec=duration_sec,
                    micro_decision=micro_decision,
                    rebench_ref=rebench_ref,
                    failure_reason=failure_reason,
                ),
                "kernel_id": str(kernel_id or ""),
                "kernel_name": _text(kernel_name),
                "dispatched": bool(dispatched),
                "backends_tried": [str(item) for item in _as_list(backends_tried)],
                "adopted_backend": _text(adopted_backend),
                "skip_reason": _text(skip_reason),
                "task_group": _text(task_group),
                "speedup": _float_or_none(speedup),
                "baseline_us": _float_or_none(baseline_us),
                "candidate_us": _float_or_none(candidate_us),
                "compile_status": _text(compile_status),
                "correctness": correctness if isinstance(correctness, bool) else None,
                "artifact_path": _text(artifact_path),
                "trace_analyze_ref": _text(trace_analyze_ref),
            }
        )

    def record_fusion_run(
        self,
        *,
        run_id: str,
        status: str,
        pattern: str = "",
        target_module: str = "",
        applied: bool = False,
        gain_pct: Any = None,
        patch_path: str = "",
        micro_decision: str = "",
        rebench_ref: str = "",
        started_at: str = "",
        ended_at: str = "",
        duration_sec: Any = None,
        failure_reason: str = "",
    ) -> None:
        """Record one forge-fusion run."""
        self._record_lane_run(
            {
                **_lane_row(
                    source_kind=SOURCE_FUSION,
                    run_id=run_id,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_sec=duration_sec,
                    micro_decision=micro_decision,
                    rebench_ref=rebench_ref,
                    failure_reason=failure_reason,
                ),
                "pattern": _text(pattern),
                "target_module": _text(target_module),
                "applied": bool(applied),
                "gain_pct": _float_or_none(gain_pct),
                "patch_path": _text(patch_path),
            }
        )

    def record_gemm_tuning_run(
        self,
        *,
        run_id: str,
        status: str,
        shapes_total: Any = None,
        shapes_tuned: Any = None,
        config_path: str = "",
        gain_pct: Any = None,
        tuner: str = "",
        micro_decision: str = "",
        rebench_ref: str = "",
        started_at: str = "",
        ended_at: str = "",
        duration_sec: Any = None,
        failure_reason: str = "",
    ) -> None:
        """Record one GEMM shape-table tuning run."""
        self._record_lane_run(
            {
                **_lane_row(
                    source_kind=SOURCE_GEMM_TUNING,
                    run_id=run_id,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_sec=duration_sec,
                    micro_decision=micro_decision,
                    rebench_ref=rebench_ref,
                    failure_reason=failure_reason,
                ),
                "shapes_total": _int_or_none(shapes_total),
                "shapes_tuned": _int_or_none(shapes_tuned),
                "config_path": _text(config_path),
                "gain_pct": _float_or_none(gain_pct),
                "tuner": _text(tuner),
            }
        )

    def record_rebench_attempt(self, **fields: Any) -> None:
        """Record one forge rebench attempt."""
        fields.setdefault("ledger", LEDGER_FORGE)
        row = _rebench_row(**fields)
        self._sink.record(
            SECTION_REBENCH,
            row,
            row_type=ROW_REBENCH,
            natural_ids=(LEDGER_FORGE, row["attempt_id"]),
        )

    # ---- geak ------------------------------------------------------------

    def record_geak_handoff(self, handoff: dict[str, Any] | None) -> None:
        """Record the conditions GEAK was asked to work under.

        The handoff's ``accepted_flags`` is the orchestrator's current best --
        GEAK's *starting* point -- while the ``accepted_flags`` GEAK later
        reports is what it *produced*. The two are recorded under distinct names
        because a single ``config`` block holding both under one key would be
        read backwards, and their difference is the configuration surface this
        delegation actually moved.
        """
        payload = _as_dict(handoff)
        envs = payload.get("accepted_env")
        self._sink.record(
            SECTION_EVENT,
            {
                "geak_handoff": {
                    "schema_version": _int_or_none(payload.get("schema_version")),
                    "model_path": _text(payload.get("model_path")),
                    "framework": _text(payload.get("framework")),
                    "gpu_type": _text(payload.get("gpu_type")),
                    "tp": _int_or_none(payload.get("tp")),
                    "workload": _as_dict(payload.get("workload")),
                    "baseline_flags": _text(payload.get("accepted_flags")),
                    "baseline_envs": _text(envs) if isinstance(envs, str) else _as_dict(envs),
                    "baseline_env_spec_present": bool(payload.get("baseline_env_spec")),
                    "launch_recipe": _text(payload.get("launch_recipe")),
                    "raw_baseline_tput": _float_or_none(payload.get("raw_baseline_tput")),
                    "orchestrator_best_tput_same_config": _float_or_none(
                        payload.get("orchestrator_best_tput_same_config")
                    ),
                    "max_model_len": _int_or_none(payload.get("max_model_len")),
                    "mem_fraction": _float_or_none(payload.get("mem_fraction")),
                    "bench_client": _text(payload.get("bench_client")),
                    "e2e_metric": _text(payload.get("e2e_metric")),
                    "bench_protocol_present": bool(payload.get("bench_protocol")),
                    "gpu_ids": _text(payload.get("gpu_ids")),
                    # The cards GEAK was told to use. A baseline that reads
                    # ``no_gain`` because its servers landed on a foreign
                    # tenant's card is otherwise indistinguishable from a real
                    # result (issue #1312). Both are stated even when the
                    # handoff omits them: ``schema_version`` in this same row
                    # separates a pre-v3 handoff from a genuinely unpinned run,
                    # so an empty value here is not ambiguous.
                    "gpu_ids_space": _text(payload.get("gpu_ids_space")),
                    "gpu_pin": _as_dict(payload.get("gpu_pin")),
                    "exp_root": _text(payload.get("exp_root")),
                    "eval_dir": _text(payload.get("eval_dir")),
                }
            },
        )

    def record_geak_delegation(
        self,
        *,
        runner_status: str,
        started_at: str = "",
        ended_at: str = "",
        duration_sec: Any = None,
        error_class: str = "",
        error: str = "",
        returncode: Any = None,
        runner_timeout_sec: Any = None,
        kill_timeout_sec: Any = None,
        exp_root: str = "",
        eval_dir: str = "",
        report_path: str = "",
        versions: dict[str, Any] | None = None,
        recovered_from_disk: bool = False,
        stages_reached: Any = None,
    ) -> None:
        """Record how the delegated runner itself ended."""
        self._sink.record(
            SECTION_EVENT,
            {
                "geak_delegation": {
                    "runner_status": str(runner_status or ""),
                    "started_at": _text(started_at),
                    "ended_at": _text(ended_at),
                    "duration_sec": _float_or_none(duration_sec),
                    "error_class": _text(error_class),
                    "error": _clip(error) or None,
                    "returncode": _int_or_none(returncode),
                    "runner_timeout_sec": _int_or_none(runner_timeout_sec),
                    "kill_timeout_sec": _int_or_none(kill_timeout_sec),
                    "exp_root": _text(exp_root),
                    "eval_dir": _text(eval_dir),
                    "report_path": _text(report_path),
                    "versions": _as_dict(versions),
                    "recovered_from_disk": bool(recovered_from_disk),
                    "stages_reached": [str(item) for item in _as_list(stages_reached)],
                }
            },
        )

    def record_geak_attempts(self, journey: dict[str, Any] | None) -> None:
        """Replay what GEAK tried, from the journey it emits."""
        parsed = _as_dict(journey)
        for ordinal, run in enumerate(_as_list(parsed.get("discovery_runs"))):
            row = _as_dict(run)
            if not row:
                continue
            source = _text(row.get("source"))
            self._geak_sink.record(
                SECTION_GEAK_DISCOVERY,
                {
                    "ordinal": ordinal,
                    "source": source,
                    "status": _text(row.get("status")),
                    "hot_kernel_count": len(_as_list(row.get("hot_kernels"))),
                    "scan": _as_dict(row.get("scan")),
                },
                row_type=ROW_GEAK_DISCOVERY,
                natural_ids=source or f"ordinal{ordinal}",
            )
        for ordinal, kernel in enumerate(_as_list(parsed.get("kernels"))):
            row = _as_dict(kernel)
            kernel_id = _text(row.get("kernel_id"))
            if not kernel_id:
                continue
            dispatch = _as_dict(row.get("dispatch"))
            backend = _as_dict(row.get("backend_result"))
            e2e = _as_dict(row.get("e2e"))
            verification = _as_dict(backend.get("verification"))
            self._geak_sink.record(
                SECTION_GEAK_ATTEMPT,
                {
                    "ordinal": ordinal,
                    "kernel_id": kernel_id,
                    "name": _text(row.get("name")) or kernel_id,
                    # The journey states the op kind in whichever block resolved
                    # it, so all three are read rather than only the kernel's.
                    "op_kind": _text(row.get("op_kind")) or _text(dispatch.get("op_kind")) or _text(e2e.get("op_kind")),
                    # Share of GPU time the kernel held in the profile that
                    # nominated it -- what makes an attempt worth its cost.
                    "gpu_pct": _float_or_none(row.get("gpu_pct")),
                    # The isolated speedup, which the journey states on the
                    # kernel or leaves to the backend's verification block.
                    "micro_speedup": _float_or_none(row.get("micro_speedup"))
                    if row.get("micro_speedup") is not None
                    else _float_or_none(verification.get("micro_speedup")),
                    "dispatched": bool(dispatch.get("dispatched", True)),
                    "backends": [str(item) for item in _as_list(dispatch.get("backends"))],
                    "skip_reason": _text(dispatch.get("skip_reason")),
                    "task_group": _text(dispatch.get("task_group")),
                    "backend_result": {
                        "backend": _text(backend.get("backend")),
                        "status": _text(backend.get("status")),
                        "speedup": _float_or_none(backend.get("speedup")),
                        "baseline_us": _float_or_none(backend.get("baseline_us")),
                        "candidate_us": _float_or_none(backend.get("candidate_us")),
                        "compile_status": _text(backend.get("compile_status")),
                        "correctness": backend.get("correctness")
                        if isinstance(backend.get("correctness"), bool)
                        else None,
                        "artifact_path": _text(backend.get("artifact_path")),
                        "error_class": _text(backend.get("error_class")),
                    }
                    if backend
                    else None,
                    "e2e": {
                        "integrated": bool(e2e.get("integrated")),
                        "e2e_gain_pct": _float_or_none(e2e.get("e2e_gain_pct")),
                        "validated": e2e.get("validated") if isinstance(e2e.get("validated"), bool) else None,
                        "decision": _text(e2e.get("decision")),
                        "patch_path": _text(e2e.get("patch_path")),
                        "target_file": _text(e2e.get("target_file")),
                    }
                    if e2e
                    else None,
                },
                row_type=ROW_GEAK_ATTEMPT,
                natural_ids=kernel_id,
            )

    def record_geak_claim(self, pending: dict[str, Any] | None, *, specs: Any = None) -> None:
        """Record what GEAK reported about itself, before any re-measurement."""
        slot = _as_dict(pending)
        self._sink.record(
            SECTION_EVENT,
            {
                "geak_claim": {
                    "verified": False,
                    "self_reported_tput": _float_or_none(slot.get("self_reported_tput")),
                    "self_reported_speedup": _float_or_none(slot.get("self_reported_speedup")),
                    "self_reported_gain_pct": _float_or_none(slot.get("self_reported_gain_pct")),
                    "self_reported_basis": _text(slot.get("self_reported_basis")),
                    "geak_status": _text(slot.get("geak_status")),
                    "baseline_alignment_status": _text(slot.get("baseline_alignment_status")),
                    "validated_regimes": _as_list(slot.get("validated_regimes")),
                }
            },
        )
        for row in _acceptance_rows(specs):
            self._geak_sink.record(
                SECTION_GEAK_ACCEPTANCE,
                row,
                row_type=ROW_GEAK_ACCEPTANCE,
                natural_ids=(
                    str(row["acceptance_kind"]),
                    _acceptance_identity(row, int(row["ordinal"])),
                ),
            )

    def record_geak_measurement(self, result: dict[str, Any] | None) -> None:
        """Record the latency and parity GEAK's own harness measured.

        These merge into the ``claim`` block rather than standing on their own,
        because they are the same kind of fact as the throughput beside them:
        GEAK's account of its own run, taken before the orchestrator re-measured
        anything. They are recorded from the runner's result rather than from
        the candidate slot because a run can measure a latency and still not
        produce a candidate -- ``no_gain`` with nothing accepted is exactly that
        -- and reading them off the slot would lose every such run.

        Args:
            result (dict[str, Any] | None): The runner's parsed ``result.json``.
        """
        row = _as_dict(result)
        if not row:
            return
        self._sink.record(
            SECTION_EVENT,
            {
                "geak_claim": {
                    "metric_basis": _text(row.get("metric_basis")),
                    "bench_client": _text(row.get("bench_client")),
                    "ttft_mean_ms": _float_or_none(row.get("ttft_ms")),
                    "tpot_mean_ms": _float_or_none(row.get("tpot_ms")),
                    "output_parity": row.get("output_parity"),
                }
            },
        )

    def record_geak_product(
        self,
        *,
        accepted_flags: Any = None,
        accepted_envs: dict[str, Any] | None = None,
        accepted_config: dict[str, Any] | None = None,
        cfg_hash: str = "",
        final_overlay: str = "",
        final_overlay_digest: str = "",
        final_launch_script: str = "",
        bench_script: str = "",
        final_patch: str = "",
    ) -> None:
        """Record the reproducible configuration GEAK handed back."""
        flags = accepted_flags
        self._sink.record(
            SECTION_EVENT,
            {
                "geak_product": {
                    "accepted_flags": [str(item) for item in _as_list(flags)]
                    if not isinstance(flags, str)
                    else _text(flags),
                    "accepted_envs": _as_dict(accepted_envs),
                    "accepted_config": _as_dict(accepted_config),
                    "cfg_hash": _text(cfg_hash),
                    "final_overlay": _text(final_overlay),
                    "final_overlay_digest": _text(final_overlay_digest),
                    "final_launch_script": _text(final_launch_script),
                    "bench_script": _text(bench_script),
                    "final_patch": _text(final_patch),
                }
            },
        )

    def record_geak_rebench_attempt(self, *, max_attempts: Any = None, **fields: Any) -> None:
        """Record one GEAK rebench attempt."""
        fields.setdefault("source_kind", SOURCE_GEAK_AUTHORED_KERNEL)
        fields["ledger"] = LEDGER_GEAK
        row = _rebench_row(**fields)
        self._sink.record(
            SECTION_REBENCH,
            row,
            row_type=ROW_REBENCH,
            natural_ids=(LEDGER_GEAK, row["attempt_id"]),
        )
        if max_attempts is not None:
            self._sink.record(SECTION_EVENT, {"geak_rebench": {"max_attempts": _int_or_none(max_attempts)}})

    def record_geak_rebench_conclusion(
        self,
        *,
        final_status: str = "",
        final_error_class: str = "",
        final_error: str = "",
    ) -> None:
        """Record the terminal revalidation state of the GEAK candidate."""
        self._sink.record(
            SECTION_EVENT,
            {
                "geak_rebench": {
                    "final_status": _text(final_status),
                    "final_error_class": _text(final_error_class),
                    "final_error": _clip(final_error) or None,
                }
            },
        )

    # ---- conclusion ------------------------------------------------------

    def finish(
        self,
        *,
        verdict: str = "",
        status: str = "",
        exit_reason: str = "",
        tput_after: Any = None,
        cumulative_gain_validated_out: Any = None,
        stack_depth_out: Any = None,
        stack_added: Any = None,
        stack_removed: Any = None,
    ) -> None:
        """Record the exit facts, assemble the event and close it."""
        if self._closed:
            return
        self._closed = True
        end_time = _now_iso()
        self._sink.record(
            SECTION_EVENT,
            {
                "in_flight_stage": None,
                "end_time": end_time,
                "duration_sec": round(time.monotonic() - self._t0, 3),
                "closed_status": _text(status),
                "outcome": {
                    "verdict": _text(verdict),
                    "exit_reason": _text(exit_reason),
                    "tput_after": _float_or_none(tput_after),
                    "cumulative_gain_validated_out": _float_or_none(cumulative_gain_validated_out),
                    "stack_depth_out": _int_or_none(stack_depth_out),
                    "stack_delta": {
                        "added": [_as_dict(row) for row in _as_list(stack_added)],
                        "removed": [_as_dict(row) for row in _as_list(stack_removed)],
                    },
                },
            },
        )
        # Both families of sections: an inline roofline recorded its rows into this event, and assembly needs them to
        # fill the re-profile block.
        ext, derived = assemble_kernel_ext(event_parts(EVENT_SECTIONS), event=self._event_id)
        finish_event(
            event_type=EVENT_TYPE,
            event=self._event_id,
            sequence=self._sequence,
            status=str(status) or derived,
            ext=ext,
            kind=EVENT_KIND,
            start_time=self._start_time,
            end_time=end_time,
        )
        self._clear_active()

    def finish_failed(self, *, stage: str, error_class: str = "", message: Any = "") -> None:
        """Close the event as failed, naming the stage that failed."""
        self._sink.record(
            SECTION_EVENT,
            {
                "failure": _failure_row(
                    phase=stage,
                    error_class=error_class or f"{stage}_failed",
                    message=message,
                )
            },
        )
        self.finish(verdict="failed", status="failed", exit_reason=str(stage or ""))

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an event whose phase raised instead of returning."""
        if self._closed:
            return
        self.finish_failed(
            stage=self._stage or "kernel",
            error_class=type(exc).__name__,
            message=f"kernel phase raised: {exc!r}",
        )


def _settle(
    lanes: Mapping[str, list[dict[str, Any]]],
    verdicts: Mapping[str, Mapping[str, Any]],
    acceptances: Mapping[str, list[dict[str, Any]]],
    *,
    geak_ref: str,
    geak_conflicted: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Resolve every candidate row against the rebench that measured it."""
    by_source = _empty_by_source()
    adopted: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def _resolve(row: dict[str, Any], *, ref: str, gain: Any, unsettled_why: str = "") -> None:
        source = str(row.get("source_kind") or "")
        counters = by_source.setdefault(source, {"attempted": 0, "adopted": 0, "needs_review": 0, "rejected": 0})
        counters["attempted"] += 1
        if str(row.get("status") or "") in {"failed", "timeout"}:
            row["outcome"] = OUTCOME_REJECTED
            counters["rejected"] += 1
            return
        decision = str(_as_dict(verdicts.get(ref)).get("decision") or "") if ref else ""
        if decision == REBENCH_VALIDATED:
            row["outcome"] = OUTCOME_ADOPTED
            counters["adopted"] += 1
            adopted.append(
                {
                    "source_kind": source,
                    "ref": row.get("run_id") or row.get("kernel_id") or "",
                    "gain_pct": _float_or_none(gain),
                    "rebench_ref": ref,
                }
            )
            return
        if decision in {REBENCH_NO_MATERIAL, REBENCH_NO_PROMOTE}:
            row["outcome"] = OUTCOME_REJECTED
            counters["rejected"] += 1
            return
        row["outcome"] = OUTCOME_NEEDS_REVIEW
        counters["needs_review"] += 1
        pending.append(
            {
                "source_kind": source,
                "ref": row.get("run_id") or row.get("kernel_id") or "",
                "why": unsettled_why or ("rebench_inconclusive" if ref else "no_rebench"),
            }
        )

    for lane in LANE_BY_SOURCE.values():
        for row in lanes.get(lane, []):
            _resolve(
                row,
                ref=str(row.get("rebench_ref") or ""),
                gain=row.get("gain_pct")
                if row.get("gain_pct") is not None
                else _as_dict(row.get("e2e")).get("e2e_gain_pct"),
            )

    for kind in (_ACCEPTANCE_AUTHORED, _ACCEPTANCE_ENV):
        for entry in acceptances.get(kind, []):
            synthetic = {
                "source_kind": entry.get("source_kind"),
                "run_id": entry.get("short_name") or entry.get("kernel_id") or entry.get("selection") or "",
                "status": "success",
                "rebench_ref": geak_ref,
            }
            _resolve(
                synthetic,
                ref=geak_ref,
                gain=entry.get("e2e_delta_pct"),
                unsettled_why="rebench_conflict" if geak_conflicted else "",
            )
            entry["outcome"] = synthetic["outcome"]

    return adopted, pending, by_source


def _derive_status(attempts: list[dict[str, Any]], *, candidate_count: int) -> str:
    """Read the event status off what the rebenches actually measured.

    A rebench that concluded against its candidate still concluded, so it makes
    the entry ``succeeded``; the distinction that matters is between measuring
    and not measuring. A candidate that was built and never validated leaves the
    entry ``degraded`` -- the work happened but nothing settled it -- while an
    entry that produced no candidate at all and measured nothing is ``skipped``
    rather than degraded, because there was nothing there to degrade.
    """
    if any(_float_or_none(row.get("measured_tput")) is not None for row in attempts):
        return "succeeded"
    if attempts and all(str(row.get("status") or "").lower() in _REBENCH_FAULTED_STATUSES for row in attempts):
        return "failed"
    return "degraded" if candidate_count else "skipped"


def _geak_settlement(attempts: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Pick the GEAK verdict the acceptances may be settled against.

    GEAK may rebench the same candidate up to its per-cycle ceiling, so unlike a
    forge lane it can end the entry holding several settled verdicts. Taking the
    newest would let a KEEP after a REVERT read as an adoption; two settled
    verdicts that disagree is a fact worth seeing, so the candidate stays
    pending and neither verdict is honoured.

    Returns:
        tuple[str, list[str]]: The attempt id to settle against -- ``""`` when
            none may be -- and the conflicting decisions, when they conflicted.
    """
    settled = [row for row in attempts if str(row.get("decision") or "")]
    decisions = {str(row.get("decision")) for row in settled}
    if len(decisions) == 1:
        return str(settled[-1].get("attempt_id") or ""), []
    if settled:
        return "", sorted(decisions)
    return "", []


def assemble_kernel_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one kernel event's ``ext`` out of its recorded rows."""
    event_rows = rows_for_event(parts.get(SECTION_EVENT) or [], event)
    header = event_rows[0] if event_rows else {}

    trace_runs = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_TRACE_ANALYZE) or [], event),
            keys=("ts", "run_id"),
        )
    )
    lane_rows = sort_rows(
        rows_for_event(parts.get(SECTION_LANE_RUN) or [], event),
        keys=("started_at", "run_id"),
    )
    rebench_rows = sort_rows(
        rows_for_event(parts.get(SECTION_REBENCH) or [], event),
        keys=("dispatched_at", "attempt_id"),
    )
    discovery_rows = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_GEAK_DISCOVERY) or [], event),
            keys=("ordinal", "source"),
        )
    )
    geak_kernel_rows = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_GEAK_ATTEMPT) or [], event),
            keys=("ordinal", "kernel_id"),
        )
    )
    acceptance_rows = sort_rows(
        rows_for_event(parts.get(SECTION_GEAK_ACCEPTANCE) or [], event),
        keys=("ordinal", "kernel_id", "selection"),
    )
    discovered_rows = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_DISCOVERED) or [], event),
            keys=("snapshot_id", "rank"),
        ),
        drop=("event_id", "rank"),
    )
    recommended_rows = [row for row in discovered_rows if row.get("selected")]
    integrate_rows = sort_rows(
        rows_for_event(parts.get(SECTION_INTEGRATE) or [], event),
        keys=("settled_at", "integration_id"),
    )

    lanes: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANE_BY_SOURCE.values()}
    for source, rows in group_rows(lane_rows, "source_kind").items():
        lane = LANE_BY_SOURCE.get(source)
        if lane:
            lanes[lane] = rows
    # The gate rules on a patch, and a patch belongs to a kernel -- the gate is
    # handed no attempt id, so the kernel is the only thing the two sides share.
    integrate_by_kernel = group_rows(integrate_rows, "kernel_id")
    for row in lanes.get("kernel_rewrites", []):
        row["e2e"] = _integrate_e2e(integrate_by_kernel.get(str(row.get("kernel_id") or ""), []))
    ledgers = group_rows(rebench_rows, "ledger")
    forge_ledger = ledgers.get(LEDGER_FORGE, [])
    geak_ledger = ledgers.get(LEDGER_GEAK, [])
    acceptances = group_rows(acceptance_rows, "acceptance_kind")

    verdicts = {str(row.get("attempt_id")): row for row in rebench_rows if row.get("attempt_id")}
    geak_ref, conflicting = _geak_settlement(geak_ledger)
    adopted, pending, by_source = _settle(
        lanes,
        verdicts,
        acceptances,
        geak_ref=geak_ref,
        geak_conflicted=bool(conflicting),
    )
    _fold_integrate_into_by_source(
        by_source,
        lanes=lanes,
        acceptances=acceptances,
        integrate_by_kernel=integrate_by_kernel,
    )

    # ``acceptance_kind`` and ``source_kind`` are the fields that chose which of the two arrays a row landed in, so on
    # the wire the array itself says it.
    acceptance_drop = ("event_id", "ordinal", "acceptance_kind", "source_kind")
    authored = wire_rows(acceptances.get(_ACCEPTANCE_AUTHORED, []), drop=acceptance_drop)
    env_selections = wire_rows(acceptances.get(_ACCEPTANCE_ENV, []), drop=acceptance_drop)

    reprofile = _as_dict(header.get("forge_reprofile")) or None
    if reprofile:
        # The re-profile dispatched the roofline executor inline, so its rows are in this event under the task id the
        # re-profile recorded.
        reprofile = {
            **reprofile,
            "run": assemble_roofline_action(parts, event=event, task_id=str(reprofile.get("task_id") or "")),
        }
    forge_engaged = bool(reprofile or trace_runs or forge_ledger or any(lanes.values()) or discovered_rows)
    forge: dict[str, Any] | None = None
    if forge_engaged:
        forge = {
            "engaged": True,
            "reprofile": reprofile,
            "trace_analyze_runs": trace_runs,
            "discovered_kernels": discovered_rows,
            "recommended_kernels": recommended_rows,
            "lanes": {lane: wire_rows(rows) for lane, rows in lanes.items()},
            "rebench_ledger": wire_rows(forge_ledger),
        }

    handoff = _as_dict(header.get("geak_handoff")) or None
    delegation = _as_dict(header.get("geak_delegation")) or None
    claim_block = _as_dict(header.get("geak_claim")) or None
    product = _as_dict(header.get("geak_product")) or None
    rebench_block = _as_dict(header.get("geak_rebench"))
    geak_engaged = bool(
        handoff or delegation or claim_block or product or rebench_block or geak_ledger or geak_kernel_rows or authored
    )
    geak: dict[str, Any] | None = None
    if geak_engaged:
        claim: dict[str, Any] | None = None
        if claim_block is not None or authored or env_selections:
            claim = {
                **(claim_block or {}),
                "authored_kernels": authored,
                "env_selections": env_selections,
                "kernels_optimized": len(authored),
                "accepted_heads_count": sum(1 for row in authored + env_selections if row.get("lane") == "headQueue"),
            }
        rebench = {
            "required": bool(geak_ledger) or bool(rebench_block),
            "max_attempts": _int_or_none(rebench_block.get("max_attempts")),
            "attempts_used": len(geak_ledger),
            "attempts": wire_rows(geak_ledger),
            "final_status": _text(rebench_block.get("final_status")),
            "final_error_class": _text(rebench_block.get("final_error_class")),
            "final_error": _text(rebench_block.get("final_error")),
        }
        if conflicting:
            rebench["conflicting_decisions"] = conflicting
        geak = {
            "engaged": True,
            "handoff": handoff,
            "delegation": delegation,
            "attempts": {
                "discovery_runs": discovery_rows,
                "kernels": geak_kernel_rows,
                "counts": _geak_counts(geak_kernel_rows),
            }
            if (discovery_rows or geak_kernel_rows)
            else None,
            "claim": claim,
            "product": product,
            "rebench": rebench,
        }

    outcome_block = _as_dict(header.get("outcome"))
    before = _float_or_none(outcome_block.get("tput_before"))
    after = _float_or_none(outcome_block.get("tput_after"))
    route = _text(header.get("route")) or _text(_as_dict(header.get("entry")).get("route"))
    # Derived here rather than taken from the phase, which has no verdict of its own to give: an entry is adopted
    # because a rebench validated something, and that join is what ``_settle`` above has just done.
    stated = _text(outcome_block.get("verdict"))
    outcome = {
        "route": route or "",
        "verdict": (stated or OUTCOME_ADOPTED) if adopted else None,
        "exit_reason": _text(outcome_block.get("exit_reason")),
        "tput_before": before,
        "tput_after": after,
        "net_gain_pct": round((after - before) / before * 100.0, 4)
        if before is not None and after is not None and before > 0
        else None,
        "session_baseline_tput": _float_or_none(outcome_block.get("session_baseline_tput")),
        "cumulative_gain_validated_out": _float_or_none(outcome_block.get("cumulative_gain_validated_out")),
        "stack_depth_out": _int_or_none(outcome_block.get("stack_depth_out")),
        "adopted": adopted,
        "pending_review": pending,
        "by_source": by_source,
        "stack_delta": _as_dict(outcome_block.get("stack_delta")) or {"added": [], "removed": []},
    }

    ext: dict[str, Any] = {
        "macro_cycle": _int_or_none(header.get("macro_cycle")) or 0,
        "in_flight_stage": _text(header.get("in_flight_stage")),
        "entry": _as_dict(header.get("entry")),
        "geak": geak,
        "forge": forge,
        # Sibling of the two routes rather than nested in either: the gate is
        # the orchestrator's and rules on whatever the visit produced, so a
        # patch from GEAK and one from forge go through the same one.
        "integrate": wire_rows(integrate_rows, drop=("event_id",)),
        "outcome": outcome,
        "failure": _as_dict(header.get("failure")) or None,
    }
    duration = _float_or_none(header.get("duration_sec"))
    if duration is not None:
        ext["duration_sec"] = duration

    candidate_count = sum(counters["attempted"] for counters in by_source.values())
    return ext, _derive_status(rebench_rows, candidate_count=candidate_count)


def _geak_counts(kernels: list[dict[str, Any]]) -> dict[str, int]:
    """Count GEAK's campaign off the rows it left, rather than while replaying."""
    counts = {
        "discovered": 0,
        "dispatched": 0,
        "skipped": 0,
        "backend_ok": 0,
        "backend_fail": 0,
        "integrated": 0,
    }
    for row in kernels:
        counts["discovered"] += 1
        counts["dispatched" if row.get("dispatched") else "skipped"] += 1
        backend = _as_dict(row.get("backend_result"))
        if backend:
            ok = str(backend.get("status") or "") in {"ok", "success", "succeeded"}
            counts["backend_ok" if ok else "backend_fail"] += 1
        if _as_dict(row.get("e2e")).get("integrated"):
            counts["integrated"] += 1
    return counts


def make_kernel_recorder(
    *,
    macro_cycle: int = 0,
    route: str = "",
    route_reason: str = "",
    resumed: bool = False,
    code_revision: str = "",
) -> KernelEventRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed.

    KERNEL behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating. An unbound session
    declines too: writing the timeline into whatever the working directory
    happens to be is worse than not recording.

    Returns:
        KernelEventRecorder | None: The recorder, or ``None`` when it could not
            be built.
    """
    from ...session.session_binding import session_is_bound

    try:
        if not session_is_bound():
            log.warning(
                "kernel timeline: no session bound; this phase entry's whole event will be "
                "missing from the breakdown. The coordinator binds at startup, so this means "
                "either that never happened or the entry ran outside the session's context"
            )
            return None
        return KernelEventRecorder(
            macro_cycle=macro_cycle,
            route=route,
            route_reason=route_reason,
            resumed=resumed,
            code_revision=code_revision,
        )
    except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
        log.warning(
            "kernel timeline: recorder construction failed; this phase entry's whole event "
            "will be missing from the breakdown",
            exc_info=True,
        )
        return None
