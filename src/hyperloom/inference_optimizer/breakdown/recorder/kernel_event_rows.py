# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Row writers for the SBD V6 ``kernel`` event that run outside a recorder: GEAK replay, integrate and bus analyses."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from hyperloom_common.coerce import (
    text_or_none as _text,
    to_float as _float_or_none,
    to_int as _int_or_none,
    to_mapping as _as_dict,
    to_list as _as_list,
)
from .assembler import event_parts
from .event_fields import clip as _clip
from .event_rows import rows_for_event
from .event_sink import EventSink
from .kernel_event import (
    PRODUCER,
    PRODUCER_GEAK,
    ROW_DISCOVERED,
    ROW_GEAK_ATTEMPT,
    ROW_GEAK_DISCOVERY,
    ROW_INTEGRATE,
    ROW_TRACE_ANALYZE,
    SECTION_DISCOVERED,
    SECTION_EVENT,
    SECTION_GEAK_ATTEMPT,
    SECTION_GEAK_DISCOVERY,
    SECTION_INTEGRATE,
    SECTION_TRACE_ANALYZE,
    TRACE_ANALYZE_TRIGGER_BUS_REQUEST,
    _republish_closed_event,
    kernel_event_id,
)
from .kernel_gain import claimed_gain
from .kernel_records import IntegrateVerdict, TraceAnalyzeRun

log = logging.getLogger(__name__)

__all__ = [
    "record_geak_attempts",
    "record_integrate_axis",
    "record_integrate_verdict",
    "record_trace_analyze_request",
    "reject_geak_attempts",
]


def record_geak_attempts(*, event: str, journey: dict[str, Any] | None) -> None:
    """Replay a GEAK journey into its originating event, including during recovery."""
    if not EventSink(event, producer=PRODUCER).has_row(SECTION_EVENT):
        return
    sink = EventSink(event, producer=PRODUCER_GEAK)
    parsed = _as_dict(journey)
    prior = {
        row.get("kernel_id"): _as_dict(row.get("e2e"))
        for row in rows_for_event(event_parts((SECTION_GEAK_ATTEMPT,), event=event)[SECTION_GEAK_ATTEMPT], event)
    }
    for ordinal, run in enumerate(_as_list(parsed.get("discovery_runs"))):
        row = _as_dict(run)
        if not row:
            continue
        source = _text(row.get("source"))
        sink.record(
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
        e2e = _as_dict(row.get("e2e"))
        # Re-reading the producer's file cannot undo the coordinator's verdict.
        if prior.get(kernel_id, {}).get("rejection_reason") and not e2e.get("rejection_reason"):
            e2e = prior[kernel_id]
        sink.record(
            SECTION_GEAK_ATTEMPT,
            _geak_attempt_row(ordinal, kernel_id, row, e2e),
            row_type=ROW_GEAK_ATTEMPT,
            natural_ids=kernel_id,
        )
    _republish_closed_event(event)


def _geak_attempt_row(ordinal: int, kernel_id: str, row: Mapping[str, Any], e2e: Mapping[str, Any]) -> dict[str, Any]:
    """The attempt row one journey kernel is recorded as, its verdict block ``e2e`` already settled."""
    dispatch = _as_dict(row.get("dispatch"))
    backend = _as_dict(row.get("backend_result"))
    verification = _as_dict(backend.get("verification"))
    return {
        "ordinal": ordinal,
        "kernel_id": kernel_id,
        "name": _text(row.get("name")) or kernel_id,
        # The journey states the op kind in whichever block resolved
        # it, so all three are read rather than only the kernel's.
        "op_kind": _text(row.get("op_kind")) or _text(dispatch.get("op_kind")) or _text(e2e.get("op_kind")),
        # Share of GPU time the kernel held in the profile that nominated it -- what makes an attempt worth its cost.
        "gpu_pct": _float_or_none(row.get("gpu_pct")),
        # The isolated speedup, which the journey states on the kernel or leaves to the backend's verification block.
        "micro_speedup": _float_or_none(row.get("micro_speedup"))
        if row.get("micro_speedup") is not None
        else _float_or_none(verification.get("micro_speedup")),
        "dispatched": bool(dispatch.get("dispatched", True)),
        "backends": [str(item) for item in _as_list(dispatch.get("backends"))],
        "skip_reason": _text(dispatch.get("skip_reason")),
        "task_group": _text(dispatch.get("task_group")),
        # Without these a GEAK campaign cannot be replayed in order: the forge lanes state them
        # and the journey has them, they were simply never read across.
        "started_at": _text(row.get("started_at") or dispatch.get("started_at")),
        "ended_at": _text(row.get("ended_at") or backend.get("ended_at")),
        "duration_sec": _float_or_none(row.get("duration_sec") or backend.get("duration_sec")),
        "backend_result": {
            "backend": _text(backend.get("backend")),
            "status": _text(backend.get("status")),
            "speedup": _float_or_none(backend.get("speedup")),
            "baseline_us": _float_or_none(backend.get("baseline_us")),
            "candidate_us": _float_or_none(backend.get("candidate_us")),
            "compile_status": _text(backend.get("compile_status")),
            "correctness": backend.get("correctness") if isinstance(backend.get("correctness"), bool) else None,
            "artifact_path": _text(backend.get("artifact_path")),
            "error_class": _text(backend.get("error_class")),
        }
        if backend
        else None,
        "e2e": {
            "integrated": bool(e2e.get("integrated")),
            "e2e_gain_pct": None,
            "validated": e2e.get("validated") if isinstance(e2e.get("validated"), bool) else None,
            "decision": _text(e2e.get("decision")),
            "self_reported_e2e_gain_pct": claimed_gain(e2e),
            "rejection_reason": _text(e2e.get("rejection_reason")),
            "revalidation_measured_tput": _float_or_none(e2e.get("revalidation_measured_tput")),
            "revalidation_current_best_tput": _float_or_none(e2e.get("revalidation_current_best_tput")),
            "revalidation_provenance": _text(e2e.get("revalidation_provenance")),
            "patch_path": _text(e2e.get("patch_path")),
            "target_file": _text(e2e.get("target_file")),
        }
        if e2e
        else None,
    }


def reject_geak_attempts(
    *,
    event: str,
    measured_tput: float,
    current_best_tput: float,
    provenance: str,
    rejection_reason: str,
) -> None:
    """Revoke imported KEEPs by event identity, including after KERNEL closes."""
    if not EventSink(event, producer=PRODUCER).has_row(SECTION_EVENT):
        return
    sink = EventSink(event, producer=PRODUCER_GEAK)
    parts = event_parts((SECTION_GEAK_ATTEMPT,), event=event)
    rows = rows_for_event(parts.get(SECTION_GEAK_ATTEMPT) or [], event)
    for row in rows:
        e2e = _as_dict(row.get("e2e"))
        if not (
            str(e2e.get("decision") or "").upper() in {"KEEP", "ADOPTED"}
            or e2e.get("integrated")
            or e2e.get("validated")
        ):
            continue
        sink.record(
            SECTION_GEAK_ATTEMPT,
            {
                **row,
                "e2e": {
                    **e2e,
                    "revalidation_measured_tput": measured_tput,
                    "revalidation_current_best_tput": current_best_tput,
                    "revalidation_provenance": provenance,
                    "rejection_reason": rejection_reason,
                    "decision": "REVERT",
                    "integrated": False,
                    "validated": False,
                    "e2e_gain_pct": None,
                },
            },
            row_type=ROW_GEAK_ATTEMPT,
            natural_ids=str(row["kernel_id"]),
        )
    _republish_closed_event(event)


def record_integrate_verdict(*, macro_cycle: Any, verdict: IntegrateVerdict) -> None:
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
    """
    if not str(verdict.integration_id or ""):
        return
    sink = EventSink(kernel_event_id(macro_cycle), producer=PRODUCER)
    if not sink.has_row(SECTION_EVENT):
        log.debug(
            "kernel timeline: no event %s to hold the integrate verdict for %s",
            sink.event_id,
            verdict.integration_id,
        )
        return
    sink.record(
        SECTION_INTEGRATE,
        verdict.row(macro_cycle),
        row_type=ROW_INTEGRATE,
        natural_ids=str(verdict.integration_id),
    )
    _republish_closed_event(sink.event_id)


def record_integrate_axis(*, macro_cycle: Any, integration_id: str, graded_objective: str) -> None:
    """Record the axis an integrate verdict's ``gain_pct`` was graded on.

    Merges onto the verdict's row by integration id; nothing is written for a
    verdict that has no row or an axis the gate did not report.
    """
    axis = _text(graded_objective)
    sink = EventSink(kernel_event_id(macro_cycle), producer=PRODUCER)
    if not (axis and integration_id and sink.has_row(SECTION_EVENT)):
        return
    row = {"integration_id": str(integration_id), "graded_objective": axis}
    sink.record(SECTION_INTEGRATE, row, row_type=ROW_INTEGRATE, natural_ids=str(integration_id))
    _republish_closed_event(sink.event_id)


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


def record_trace_analyze_request(*, macro_cycle: Any, run: TraceAnalyzeRun) -> None:
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
    if not str(run.run_id or ""):
        return
    sink = EventSink(kernel_event_id(macro_cycle), producer=PRODUCER)
    if not sink.has_row(SECTION_EVENT):
        log.debug(
            "kernel timeline: no event %s to hold the trace_analyze request %s",
            sink.event_id,
            run.run_id,
        )
        return
    sink.record(
        SECTION_TRACE_ANALYZE,
        run.row(TRACE_ANALYZE_TRIGGER_BUS_REQUEST),
        row_type=ROW_TRACE_ANALYZE,
        natural_ids=str(run.run_id or ""),
    )
    produced = _as_dict(run.snapshot)
    if produced:
        _write_discovered_kernels(sink, produced, provenance="trace_analyze_run")
    _republish_closed_event(sink.event_id)


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
