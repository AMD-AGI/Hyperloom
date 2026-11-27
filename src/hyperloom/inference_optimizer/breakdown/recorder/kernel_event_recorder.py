# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recorder that writes one KERNEL entry's ``kernel`` event rows, and its factory."""

from __future__ import annotations

import time
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
from .event_fields import clip as _clip, failure_row as _failure_row, now_iso_seconds as _now_iso
from .event_sink import EventSink
from .event_timeline import EventHeader, finish_event, open_event
from .kernel_event import (
    _ACCEPTANCE_AUTHORED,
    _ACCEPTANCE_ENV,
    EVENT_KIND,
    EVENT_TYPE,
    PRODUCER,
    PRODUCER_GEAK,
    ROW_GEAK_ACCEPTANCE,
    ROW_LANE_RUN,
    ROW_REBENCH,
    SECTION_EVENT,
    SECTION_GEAK_ACCEPTANCE,
    SECTION_LANE_RUN,
    SECTION_REBENCH,
    SOURCE_GEAK_ENV_SELECTION,
    STATUS_FAILED,
    _acceptance_identity,
    _republish_closed_event,
    assemble_kernel_ext,
    kernel_event_id,
)
from .kernel_event_rows import _write_discovered_kernels
from .kernel_records import (
    LEDGER_GEAK,
    SOURCE_GEAK_AUTHORED_KERNEL,
    FusionRun,
    GeakDelegation,
    GeakProduct,
    GeakRebenchAttempt,
    GemmTuningRun,
    KernelExit,
    KernelReprofile,
    KernelRewriteRun,
)
from .sections import EVENT_SECTIONS

__all__ = ["KernelEventRecorder", "make_kernel_recorder"]


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


class KernelEventRecorder:
    """Records the facts of one KERNEL entry, one fragment per row."""

    def __init__(
        self,
        *,
        macro_cycle: int = 0,
        route: str = "",
        route_reason: str = "",
        resumed: bool = False,
    ):
        """Bind a recorder to the event of one KERNEL entry."""
        self._event_id = kernel_event_id(macro_cycle)
        self._sink = EventSink(self._event_id, producer=PRODUCER)
        self._geak_sink = EventSink(self._event_id, producer=PRODUCER_GEAK)
        self._t0 = time.monotonic()
        self._start_time = _now_iso()
        self._sequence: int | None = None
        self._closed = False
        self._faulted = False
        self._route = str(route or "")
        self._stage = "entry"
        self._sink.record(
            SECTION_EVENT,
            {
                "macro_cycle": int(macro_cycle or 0),
                "in_flight_stage": self._stage,
                "entry": {
                    "route": self._route,
                    "route_reason": str(route_reason or ""),
                    "resumed": bool(resumed),
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
            EventHeader(EVENT_TYPE, self._event_id, EVENT_KIND, self._start_time),
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            ext={"route": self._route, "in_flight_stage": self._stage},
        )

    def record_discovered_kernels(
        self,
        snapshot: Mapping[str, Any] | None,
        *,
        provenance: str = "trace_analyze",
    ) -> None:
        """Record the profiling-rich kernel table this visit is targeting.

        The hot-kernel summary on a roofline event is intentionally thin; the
        KERNEL visit needs the per-kernel profiling fields that decide which
        targets are worth rewriting. This is recorded as its own table so a
        reader does not have to join against a capped top-15 summary missing
        ``kernel_id``.

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

    def record_reprofile(self, reprofile: KernelReprofile) -> None:
        """Record the entry re-profile that decides whether analysis is stale."""
        self._sink.record(SECTION_EVENT, {"forge_reprofile": reprofile.block()})

    def _record_lane_run(self, row: Mapping[str, Any]) -> None:
        """Write one lane row, keyed by the run it describes."""
        self._sink.record(
            SECTION_LANE_RUN,
            row,
            row_type=ROW_LANE_RUN,
            natural_ids=(str(row.get("source_kind") or ""), str(row.get("run_id") or "")),
        )

    def record_kernel_rewrite(self, run: KernelRewriteRun) -> None:
        """Record one forge source-level kernel rewrite."""
        self._record_lane_run(run.row())

    def record_fusion_run(self, run: FusionRun) -> None:
        """Record one forge-fusion run."""
        self._record_lane_run(run.row())

    def record_gemm_tuning_run(self, run: GemmTuningRun) -> None:
        """Record one GEMM shape-table tuning run.

        ``graded_objective`` names the axis ``gain_pct`` was measured on, so a
        run graded on total throughput or interactivity is not later read as an
        output gain. ``tuner`` names which backend produced the table, which is
        a separate question from which axis judged it.
        """
        self._record_lane_run(run.row())

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
                    "launch_digest": _text(_as_dict(payload.get("baseline_env_spec")).get("launch_digest")),
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

    def record_geak_delegation(self, delegation: GeakDelegation) -> None:
        """Record how the delegated runner itself ended."""
        self._sink.record(SECTION_EVENT, {"geak_delegation": delegation.block()})

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

    def record_geak_product(self, product: GeakProduct) -> None:
        """Record the reproducible configuration GEAK handed back."""
        self._sink.record(SECTION_EVENT, {"geak_product": product.block()})

    def record_geak_rebench_attempt(self, attempt: GeakRebenchAttempt) -> None:
        """Record one GEAK rebench attempt."""
        row = attempt.row()
        self._sink.record(
            SECTION_REBENCH,
            row,
            row_type=ROW_REBENCH,
            natural_ids=(LEDGER_GEAK, row["attempt_id"]),
        )

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
        _republish_closed_event(self._event_id)

    # ---- conclusion ------------------------------------------------------

    def finish(self, exit_facts: KernelExit = KernelExit(), *, failed: bool = False) -> None:
        """Record the exit facts, assemble the event and close it.

        The phase states what it measured on the way out, not how the visit
        ran: the verdict is read off the instruments that ruled on each
        candidate, and a phase allowed to state its own could contradict them.

        ``failed`` is the one thing only the phase knows -- that it could not
        get past the stage it was in, rather than reaching a conclusion the
        instruments can be asked about. It decides the event's status and is
        recorded, so a re-assembly after the close reaches the same one.
        """
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
                "closed_status": STATUS_FAILED if failed else "",
                "outcome": exit_facts.outcome(),
            },
        )
        # Both families of sections: an inline roofline recorded its rows into this event, and assembly needs them to
        # fill the re-profile block.
        from .recorder_warnings import RECORDING_ERRORS, note_failure

        try:
            ext, derived = assemble_kernel_ext(
                event_parts(EVENT_SECTIONS, event=self._event_id),
                event=self._event_id,
            )
            finish_event(
                EventHeader(EVENT_TYPE, self._event_id, EVENT_KIND, self._start_time),
                sequence=self._sequence,
                status=derived,
                ext=ext,
                end_time=end_time,
            )
        except RECORDING_ERRORS as exc:
            note_failure(section=SECTION_EVENT, error=exc, detail=f"closing kernel event {self._event_id}")

    def record_fault(
        self,
        *,
        stage: str,
        exc: BaseException | None = None,
        error_class: str = "",
        message: Any = "",
    ) -> None:
        """Name a fault that struck mid-visit, without ending the visit.

        A raising tick does not end a KERNEL visit: the coordinator files the
        exception against its crash count and the session keeps going, so the
        visit outlives the fault and may still settle candidates afterwards.
        Closing here would cut short a visit that survived, and recording
        nothing left the event closing clean -- stating a verdict for a visit
        that had blown up, with the exception readable nowhere.

        Only the first fault is kept: what follows a crash is generally its
        consequence, and the cause is the more useful of the two.
        """
        if self._closed or self._faulted:
            return
        self._faulted = True
        self._sink.record(
            SECTION_EVENT,
            {
                "failure": _failure_row(
                    stage=stage,
                    exc=exc,
                    error_class=error_class or ("" if exc is not None else f"{stage}_failed"),
                    message=message,
                )
            },
        )

    def finish_failed(self, *, stage: str, error_class: str = "", message: Any = "") -> None:
        """Close the event as failed, naming the stage that failed.

        This is the phase naming the stage it could not get past, rather than
        one it carried on through, so it decides the status where a fault
        recorded through :meth:`record_fault` does not: that visit survived and
        is judged on what its instruments went on to measure. The verdict is
        still derived either way -- a win delivered before the raise stays
        readable, beside the status that says the visit did not end well.
        """
        self._sink.record(
            SECTION_EVENT,
            {
                "failure": _failure_row(
                    stage=stage,
                    error_class=error_class or f"{stage}_failed",
                    message=message,
                )
            },
        )
        self.finish(KernelExit(exit_reason=str(stage or "")), failed=True)


def make_kernel_recorder(
    *,
    macro_cycle: int = 0,
    route: str = "",
    route_reason: str = "",
    resumed: bool = False,
) -> KernelEventRecorder | None:
    """Build a recorder, or ``None`` when no session is bound.

    An unbound session declines rather than writing the timeline into an
    arbitrary directory. Construction itself is not swallowed.
    """
    from .construct import decline_unbound

    if decline_unbound("kernel"):
        return None
    return KernelEventRecorder(
        macro_cycle=macro_cycle,
        route=route,
        route_reason=route_reason,
        resumed=resumed,
    )
