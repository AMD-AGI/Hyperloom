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
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
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

ROW_LANE_RUN = "lane_run"
ROW_REBENCH = "rebench"
ROW_TRACE_ANALYZE = "trace_analyze"
ROW_GEAK_ATTEMPT = "geak_attempt"
ROW_GEAK_DISCOVERY = "geak_discovery"
ROW_GEAK_ACCEPTANCE = "geak_acceptance"

ROUTE_GEAK = "geak"
ROUTE_FORGE = "forge"
ROUTE_COLLECTIVE_ONLY = "collective_only"

# The six candidate producers a KERNEL entry can adopt from. The first four are
# forge's independently gated lanes; the last two split GEAK's acceptances,
# which the ledger used to merge. A ``kind == "env"`` acceptance selects an
# existing library or environment variable and authors no kernel, so it belongs
# to the config half of GEAK's gain rather than to the per-kernel adoption
# ledger -- and before this recorder it was filtered out and lost entirely.
SOURCE_KERNEL_REWRITE = "kernel_rewrite"
SOURCE_FUSION = "fusion"
SOURCE_GEMM_TUNING = "gemm_tuning"
SOURCE_COLLECTIVE = "collective"
SOURCE_GEAK_AUTHORED_KERNEL = "geak_authored_kernel"
SOURCE_GEAK_ENV_SELECTION = "geak_env_selection"

_SOURCE_KINDS = (
    SOURCE_KERNEL_REWRITE,
    SOURCE_FUSION,
    SOURCE_GEMM_TUNING,
    SOURCE_COLLECTIVE,
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
    SOURCE_COLLECTIVE: "collective_runs",
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

# A rebench either validated the candidate, found its win immaterial, measured
# it truthfully without beating current_best, or could not conclude because the
# config it was asked to reproduce did not engage.
REBENCH_VALIDATED = "validated"
REBENCH_NO_MATERIAL = "no_material"
REBENCH_NO_PROMOTE = "no_promote"
REBENCH_FALLBACK = "fallback"

# A rebench that reports one of these measured nothing, so it concluded nothing
# about the candidate it was dispatched for.
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
    "ROUTE_COLLECTIVE_ONLY",
    "ROUTE_FORGE",
    "ROUTE_GEAK",
    "SECTION_EVENT",
    "SECTION_GEAK_ACCEPTANCE",
    "SECTION_GEAK_ATTEMPT",
    "SECTION_GEAK_DISCOVERY",
    "SECTION_LANE_RUN",
    "SECTION_REBENCH",
    "SECTION_TRACE_ANALYZE",
    "SOURCE_COLLECTIVE",
    "SOURCE_FUSION",
    "SOURCE_GEAK_AUTHORED_KERNEL",
    "SOURCE_GEAK_ENV_SELECTION",
    "SOURCE_GEMM_TUNING",
    "SOURCE_KERNEL_REWRITE",
    "KernelEventRecorder",
    "assemble_kernel_ext",
    "kernel_event_id",
    "make_kernel_recorder",
]


def kernel_event_id(macro_cycle: Any) -> str:
    """Build the event id of the KERNEL entry in one macro cycle.

    Args:
        macro_cycle (Any): The macro cycle the entry belongs to.

    Returns:
        str: The event id, ``kernel_agent:{macro_cycle}:kernel``.

    Raises:
        ValueError: If ``macro_cycle`` is not a non-negative integer.
    """
    return event_id(EVENT_PHASE, macro_cycle, EVENT_COMPONENT)


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

    Args:
        source_kind (str): One of the six producers.
        run_id (str): Lane-stable identifier for this candidate.
        status (str): How the candidate's own run ended.
        started_at (str | None): ISO timestamp the candidate started.
        ended_at (str | None): ISO timestamp the candidate ended.
        duration_sec (float | None): Wall-clock seconds the candidate took.
        micro_decision (str | None): The candidate layer's verdict on its own
            output.
        rebench_ref (str | None): The rebench attempt id that re-measured it.
        failure_reason (str | None): Normalized failure reason.

    Returns:
        dict[str, Any]: The shared lane-row block.
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

    Args:
        attempt_id (str): Ledger-stable identifier for this attempt.
        source_kind (str): The producer whose candidate this attempt
            re-measured.
        ledger (str): :data:`LEDGER_FORGE` or :data:`LEDGER_GEAK`.
        source_ref (str | None): The candidate's ``run_id``.
        idempotency_key (str | None): The dispatch idempotency key.
        task_id (str | None): The dispatched task id.
        dispatched_at (str | None): ISO timestamp the attempt was dispatched.
        settled_at (str | None): ISO timestamp the verdict landed.
        base_tput (float | None): The throughput the attempt measured against.
        measured_tput (float | None): The throughput the attempt measured.
        decision (str | None): The rebench verdict, or ``None`` while unsettled.
        decision_reason (str | None): Why the verdict landed that way.
        status (str | None): The attempt's own lifecycle status.
        engagement (dict[str, Any] | None): Config / overlay verification
            booleans.

    Returns:
        dict[str, Any]: The rebench attempt row.
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
    """Split GEAK's acceptances into authored kernels and env selections.

    GEAK routes an acceptance to ``accepted_kernels`` or ``accepted_heads``
    purely by which queue proposed it, and both lanes carry the same
    parity-checked ``e2e_delta_pct``; reading only the first drops most of the
    campaign. A candidate tag such as ``cand_c0_triton`` names the slot that
    proposed the acceptance, so the row records which of the two named it.

    Args:
        specs (Any): The combined ``accepted_kernels`` + ``accepted_heads``
            rows, each tagged with the lane it came from.

    Returns:
        list[dict[str, Any]]: One row per acceptance, each tagged with the kind
            it is and the position it was reported in. The position stands in
            for a timestamp GEAK does not report, and it is stable across a
            replay of the same report, which is what an ordering key has to be.
    """
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
            }
        )
    return rows


def _acceptance_identity(row: Mapping[str, Any], ordinal: int) -> str:
    """Name one acceptance stably enough to key its fragment.

    Args:
        row (Mapping[str, Any]): The acceptance row.
        ordinal (int): Its position in GEAK's report.

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
    """Records the facts of one KERNEL entry, one fragment per row.

    The recorder holds an event id and two sinks, and no event. Nothing it
    writes is read back until :meth:`finish`, which assembles the whole event
    out of the fragments rather than out of anything the recorder remembers --
    so an entry recorded across a resume assembles from both halves.
    """

    def __init__(
        self,
        *,
        macro_cycle: int = 0,
        route: str = "",
        route_reason: str = "",
        resumed: bool = False,
        code_revision: str = "",
    ):
        """Bind a recorder to the event of one KERNEL entry.

        Args:
            macro_cycle (int): The macro cycle this entry belongs to. It is a
                segment of the event id rather than a row field, because it is
                part of what identifies the event.
            route (str): The dispatch route the entry hook selected.
            route_reason (str): Why that route was selected.
            resumed (bool): Whether the phase was entered by a resume.
            code_revision (str): Orchestration commit the entry ran.

        Raises:
            ValueError: If ``macro_cycle`` is not a non-negative integer.
        """
        self._event_id = kernel_event_id(macro_cycle)
        self._sink = make_sink(self._event_id, producer=PRODUCER)
        self._geak_sink = make_sink(self._event_id, producer=PRODUCER_GEAK)
        self._t0 = time.monotonic()
        self._start_time = _now_iso()
        self._sequence: int | None = None
        self._closed = False
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
        """Record the entry measurements and put the event on the timeline.

        ``tput_before`` is the throughput the previous stage exited on, which is
        what this entry's net gain must be measured against. The session
        baseline is recorded beside it because the rebench path measures against
        that instead, and a reader comparing the two needs both on record.

        Args:
            stack_depth_in (Any): Optimization-stack depth on entry.
            budget_remaining_sec (Any): Phase budget left on entry.
            tput_before (Any): Throughput the previous stage exited on.
            session_baseline_tput (Any): The session's raw baseline throughput.
            snapshot (dict[str, Any] | None): The ``last_trace_analyze`` cache
                the entry inherited.
            snapshot_staleness (str): ``fresh`` / ``stale`` / ``absent``.
        """
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

    def enter_stage(self, stage: str) -> None:
        """Name the stage now in flight so a kill leaves it identifiable.

        This touches the fragment only. The timeline entry keeps the status it
        opened with until the phase ends, and a session killed before then is
        closed out of its fragments by finalize -- which reads this field.

        Args:
            stage (str): The stage the phase is entering.
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
        """Record the entry re-profile that decides whether analysis is stale.

        ``task_kind`` is the judgement the rest of the chain hangs on: a
        ``roofline`` task carries its own ``trace_analyze`` and refreshes the
        cache, while a plain ``profile`` task invalidates it and forces the phase
        to request analysis of its own. This runs on the forge and
        collective-only routes only -- GEAK profiles from scratch itself and is
        handed no trace, so re-profiling for it would buy nothing.

        Args:
            ran (bool): Whether a re-profile was actually dispatched.
            task_kind (str): ``roofline`` or ``profile``.
            trigger (str): ``gain`` / ``config_changed`` / ``workload_changed``.
            skipped_reason (str): Why it was skipped, when it was.
            idempotency_reason (str): The dispatch reason tag.
            snapshot_landed (bool): Whether a new snapshot actually landed.
            snapshot_id_before (Any): Snapshot counter before the attempt.
            snapshot_id_after (Any): Snapshot counter after the attempt.
            task_id (str): The dispatched task, which is how the re-profile's
                own rows are found again: the roofline executor records them
                into this event under that id, and assembly folds the action it
                names into ``forge.reprofile.run``.
        """
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
            run_id (str): Entry-stable identifier for this analysis.
            trigger (str): ``pre_run_optimization`` or ``llm_explicit``.
            status (str): ``ok`` or ``failed``.
            result (Any): The analysis tool's result dict.
            requested_by (str): The role that requested it.
            request_msg_id (str): The bus request message id.
            trace_input (str): The trace the run analysed.
            top_k (Any): The requested ranking depth.
            snapshot (dict[str, Any] | None): The ``last_trace_analyze`` cache
                the run produced.
            cache_hit (bool): Whether a cached result served the request.
        """
        produced = _as_dict(snapshot)
        self._sink.record(
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
                "roofline_baseline_gain_at_snapshot": _float_or_none(
                    produced.get("roofline_baseline_gain_at_snapshot")
                ),
                "steady_state_trace": _text(produced.get("steady_state_trace")),
                "analysis_md_path": _text(produced.get("analysis_md_path")),
                "reusable_native_kernel_ids": [
                    str(item) for item in _as_list(produced.get("reusable_native_kernel_ids"))
                ],
                "trace_validate_ref": None,
                **_analysis_detail(result),
            },
            row_type=ROW_TRACE_ANALYZE,
            natural_ids=str(run_id or ""),
        )

    def _record_lane_run(self, row: Mapping[str, Any]) -> None:
        """Write one lane row, keyed by the run it describes.

        Args:
            row (Mapping[str, Any]): The lane row, carrying its ``source_kind``
                and ``run_id``.
        """
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
        e2e: dict[str, Any] | None = None,
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

        Args:
            run_id (str): The real attempt id for this rewrite.
            kernel_id (str): The kernel the rewrite targeted.
            status (str): How the rewrite's own run ended.
            kernel_name (str): Human-readable kernel name.
            dispatched (bool): Whether a backend was actually dispatched.
            backends_tried (Any): The backends attempted.
            adopted_backend (str): The backend whose output was taken.
            skip_reason (str): Why dispatch was skipped, when it was.
            task_group (str): The dispatch task group.
            speedup (Any): Micro-benchmark speedup.
            baseline_us (Any): Micro-benchmark baseline microseconds.
            candidate_us (Any): Micro-benchmark candidate microseconds.
            compile_status (str): Compilation outcome.
            correctness (Any): Correctness verdict.
            artifact_path (str): The produced artifact.
            micro_decision (str): The candidate layer's own verdict.
            rebench_ref (str): The rebench attempt that re-measured it.
            trace_analyze_ref (str): The analysis that nominated this kernel.
            e2e (dict[str, Any] | None): The end-to-end integration sub-result.
            started_at (str): ISO timestamp the rewrite started.
            ended_at (str): ISO timestamp the rewrite ended.
            duration_sec (Any): Wall-clock seconds the rewrite took.
            failure_reason (str): Normalized failure reason.
        """
        integration = _as_dict(e2e)
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
                "e2e": {
                    "integrated": bool(integration.get("integrated")),
                    "e2e_gain_pct": _float_or_none(integration.get("e2e_gain_pct")),
                    "validated": integration.get("validated")
                    if isinstance(integration.get("validated"), bool)
                    else None,
                    "decision": _text(integration.get("decision")),
                    "patch_path": _text(integration.get("patch_path")),
                    "target_file": _text(integration.get("target_file")),
                }
                if integration
                else None,
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
        """Record one forge-fusion run.

        Args:
            run_id (str): Lane-stable identifier for this run.
            status (str): How the run ended.
            pattern (str): The fusion pattern attempted.
            target_module (str): The module the fusion targeted.
            applied (bool): Whether the fusion was applied.
            gain_pct (Any): The gain the run claimed.
            patch_path (str): The produced patch.
            micro_decision (str): The candidate layer's own verdict.
            rebench_ref (str): The rebench attempt that re-measured it.
            started_at (str): ISO timestamp the run started.
            ended_at (str): ISO timestamp the run ended.
            duration_sec (Any): Wall-clock seconds the run took.
            failure_reason (str): Normalized failure reason.
        """
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
        """Record one GEMM shape-table tuning run.

        Args:
            run_id (str): Lane-stable identifier for this run.
            status (str): How the run ended.
            shapes_total (Any): Shapes the run considered.
            shapes_tuned (Any): Shapes the run tuned.
            config_path (str): The produced shape-table.
            gain_pct (Any): The gain the run claimed.
            tuner (str): The tuner that ran.
            micro_decision (str): The candidate layer's own verdict.
            rebench_ref (str): The rebench attempt that re-measured it.
            started_at (str): ISO timestamp the run started.
            ended_at (str): ISO timestamp the run ended.
            duration_sec (Any): Wall-clock seconds the run took.
            failure_reason (str): Normalized failure reason.
        """
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

    def record_collective_run(
        self,
        *,
        run_id: str,
        status: str,
        op: str = "",
        algo: str = "",
        size_bytes: Any = None,
        world_size: Any = None,
        gain_pct: Any = None,
        withheld: bool = False,
        withhold_reason: str = "",
        micro_decision: str = "",
        rebench_ref: str = "",
        started_at: str = "",
        ended_at: str = "",
        duration_sec: Any = None,
        failure_reason: str = "",
    ) -> None:
        """Record one collective-tuning run.

        Args:
            run_id (str): Lane-stable identifier for this run.
            status (str): How the run ended.
            op (str): The collective operation tuned.
            algo (str): The algorithm selected.
            size_bytes (Any): The message size tuned for.
            world_size (Any): The participating rank count.
            gain_pct (Any): The gain the run claimed.
            withheld (bool): Whether the candidate was withheld from adoption.
            withhold_reason (str): Why it was withheld.
            micro_decision (str): The candidate layer's own verdict.
            rebench_ref (str): The rebench attempt that re-measured it.
            started_at (str): ISO timestamp the run started.
            ended_at (str): ISO timestamp the run ended.
            duration_sec (Any): Wall-clock seconds the run took.
            failure_reason (str): Normalized failure reason.
        """
        self._record_lane_run(
            {
                **_lane_row(
                    source_kind=SOURCE_COLLECTIVE,
                    run_id=run_id,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_sec=duration_sec,
                    micro_decision=micro_decision,
                    rebench_ref=rebench_ref,
                    failure_reason=failure_reason,
                ),
                "op": _text(op),
                "algo": _text(algo),
                "size_bytes": _int_or_none(size_bytes),
                "world_size": _int_or_none(world_size),
                "gain_pct": _float_or_none(gain_pct),
                "withheld": bool(withheld),
                "withhold_reason": _text(withhold_reason),
            }
        )

    def record_rebench_attempt(self, **fields: Any) -> None:
        """Record one forge rebench attempt.

        A dispatch and the verdict that lands minutes later are two calls on one
        ``attempt_id`` describing one attempt, and keying the fragment by that id
        is what keeps them one row instead of two. Both calls state the whole
        row, so the later one wins field by field -- a caller that knows only
        the verdict should pass the dispatch fields through rather than letting
        them default, because a stated ``None`` is a value like any other.

        Args:
            **fields: The :func:`_rebench_row` fields.
        """
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

        Args:
            handoff (dict[str, Any] | None): The handoff dict written for the
                runner.
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
        """Record how the delegated runner itself ended.

        Args:
            runner_status (str): The runner's own status.
            started_at (str): ISO timestamp the runner started.
            ended_at (str): ISO timestamp the runner ended.
            duration_sec (Any): Wall-clock seconds the runner took.
            error_class (str): The failure class, on a miss.
            error (str): The failure message, on a miss.
            returncode (Any): The runner's exit code.
            runner_timeout_sec (Any): The runner's budget.
            kill_timeout_sec (Any): The runner's hard kill budget.
            exp_root (str): The runner's experiment root.
            eval_dir (str): The macro-cycle-scoped eval dir.
            report_path (str): The human report the runner wrote.
            versions (dict[str, Any] | None): Tool version provenance.
            recovered_from_disk (bool): Whether the result was reconstructed
                on-disk.
            stages_reached (Any): Stages a crashed run reached.
        """
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
        """Replay what GEAK tried, from the journey it emits.

        GEAK's ``kernel_journey.json`` names every kernel it considered, which
        backends it dispatched, and what each one measured -- not just the
        acceptances that survived. Those rows land in sections of their own
        rather than merged into forge's lanes, which is what the projection did:
        they are a different producer's account of a different campaign, and the
        counts a reader wants are per-campaign.

        Args:
            journey (dict[str, Any] | None): The parsed ``kernel_journey.json``.
        """
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
            self._geak_sink.record(
                SECTION_GEAK_ATTEMPT,
                {
                    "ordinal": ordinal,
                    "kernel_id": kernel_id,
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
        """Record what GEAK reported about itself, before any re-measurement.

        Every number here is the optimizer's own account of its run. ``verified``
        is stored as a constant so a consumer cannot mistake this block for a
        conclusion: nothing in it has been re-measured by the orchestrator's own
        harness, and the adoption verdict rests solely on the rebench.

        Args:
            pending (dict[str, Any] | None): The recorded GEAK candidate slot.
            specs (Any): The combined acceptance rows from both proposal queues.
        """
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
        """Record the reproducible configuration GEAK handed back.

        Args:
            accepted_flags (Any): Server flags GEAK accepted.
            accepted_envs (dict[str, Any] | None): Environment variables GEAK
                accepted.
            accepted_config (dict[str, Any] | None): The runner's own
                accepted-config block.
            cfg_hash (str): Canonical fingerprint of the flags and envs.
            final_overlay (str): The overlay PYTHONPATH GEAK produced.
            final_overlay_digest (str): Digest of that overlay.
            final_launch_script (str): The optimized launch script.
            bench_script (str): The benchmark script GEAK measured with.
            final_patch (str): The aggregate source patch.
        """
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
        """Record one GEAK rebench attempt.

        Args:
            max_attempts (Any): The per-cycle attempt ceiling.
            **fields: The :func:`_rebench_row` fields.
        """
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
        """Record the terminal revalidation state of the GEAK candidate.

        Args:
            final_status (str): The revalidation status the result was stamped
                with.
            final_error_class (str): The revalidation failure class.
            final_error (str): The revalidation failure message.
        """
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
        """Record the exit facts, assemble the event and close it.

        The exit facts are written as rows before assembly rather than passed
        into it, so the event finalize would rebuild from the fragments alone is
        the same event this writes.

        Args:
            verdict (str): The entry's conclusion, for a caller that knows one
                assembly cannot derive -- the failure path, which closes before
                any candidate settles. Leave it empty otherwise: assembly reads
                the conclusion off the settled rows. Either way it is dropped
                when nothing was adopted, so an entry that adopted nothing
                cannot read as having concluded something about a candidate.
            status (str): The event status to close with. Derived from the
                rebench evidence when empty, so the ladder cannot drift per call
                site.
            exit_reason (str): The phase's own exit reason.
            tput_after (Any): Throughput the entry exited on.
            cumulative_gain_validated_out (Any): Validated cumulative gain on
                exit.
            stack_depth_out (Any): Optimization-stack depth on exit.
            stack_added (Any): Stack entries this visit added.
            stack_removed (Any): Stack entries this visit removed.
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
        # Both families of sections: an inline roofline recorded its rows into
        # this event, and assembly needs them to fill the re-profile block.
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

    def finish_failed(self, *, stage: str, error_class: str = "", message: Any = "") -> None:
        """Close the event as failed, naming the stage that failed.

        Args:
            stage (str): The stage that failed.
            error_class (str): The failure class.
            message (Any): The failure message.
        """
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
        """Close an event whose phase raised instead of returning.

        Args:
            exc (BaseException): The exception propagating out of the phase.
        """
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
    """Resolve every candidate row against the rebench that measured it.

    Args:
        lanes (Mapping[str, list[dict[str, Any]]]): The forge lane rows by wire
            array name, mutated in place with their settled ``outcome``.
        verdicts (Mapping[str, Mapping[str, Any]]): Rebench rows by attempt id.
        acceptances (Mapping[str, list[dict[str, Any]]]): GEAK's acceptance rows
            by kind, mutated in place with their settled ``outcome``.
        geak_ref (str): The GEAK attempt whose verdict the acceptances are
            settled against, or ``""`` when none may be honoured.
        geak_conflicted (bool): Whether GEAK's settled attempts disagreed.

    Returns:
        tuple: The adopted rows, the rows still awaiting review, and the
            per-source counters.
    """
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

    Args:
        attempts (list[dict[str, Any]]): Every rebench row of the event.
        candidate_count (int): How many candidates the entry produced.

    Returns:
        str: The event status.
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

    Args:
        attempts (list[dict[str, Any]]): GEAK's rebench rows, in order.

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
    """Assemble one kernel event's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The kernel sections as read
            back from the spool, section name to row list.
        event (str): The event id to assemble; rows of every other event in the
            same session are ignored.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload and the status derived
            from the rebench evidence. The caller may override the status -- a
            phase that failed outright knows something the rows do not -- but it
            never has to compute it.
    """
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

    lanes: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANE_BY_SOURCE.values()}
    for source, rows in group_rows(lane_rows, "source_kind").items():
        lane = LANE_BY_SOURCE.get(source)
        if lane:
            lanes[lane] = rows
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

    # ``acceptance_kind`` and ``source_kind`` are the fields that chose which of
    # the two arrays a row landed in, so on the wire the array itself says it.
    acceptance_drop = ("event_id", "ordinal", "acceptance_kind", "source_kind")
    authored = wire_rows(acceptances.get(_ACCEPTANCE_AUTHORED, []), drop=acceptance_drop)
    env_selections = wire_rows(acceptances.get(_ACCEPTANCE_ENV, []), drop=acceptance_drop)

    reprofile = _as_dict(header.get("forge_reprofile")) or None
    if reprofile:
        # The re-profile dispatched the roofline executor inline, so its rows
        # are in this event under the task id the re-profile recorded. Nesting
        # the assembled action here rather than leaving a sibling roofline event
        # is the whole reason the inline mode exists.
        reprofile = {
            **reprofile,
            "run": assemble_roofline_action(parts, event=event, task_id=str(reprofile.get("task_id") or "")),
        }
    forge_engaged = bool(reprofile or trace_runs or forge_ledger or any(lanes.values()))
    forge: dict[str, Any] | None = None
    if forge_engaged:
        forge = {
            "engaged": True,
            "reprofile": reprofile,
            "trace_analyze_runs": trace_runs,
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
    # Derived here rather than taken from the phase, which has no verdict of its
    # own to give: an entry is adopted because a rebench validated something, and
    # that join is what ``_settle`` above has just done. A caller's word is
    # honoured only when it names something assembly cannot see for itself --
    # the failure path's, which closes before any of this is settled.
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
        "outcome": outcome,
        "failure": _as_dict(header.get("failure")) or None,
    }
    duration = _float_or_none(header.get("duration_sec"))
    if duration is not None:
        ext["duration_sec"] = duration

    candidate_count = sum(counters["attempted"] for counters in by_source.values())
    return ext, _derive_status(rebench_rows, candidate_count=candidate_count)


def _geak_counts(kernels: list[dict[str, Any]]) -> dict[str, int]:
    """Count GEAK's campaign off the rows it left, rather than while replaying.

    Args:
        kernels (list[dict[str, Any]]): GEAK's per-kernel attempt rows.

    Returns:
        dict[str, int]: The per-campaign counters.
    """
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

    Args:
        macro_cycle (int): The macro cycle this entry belongs to.
        route (str): The dispatch route the entry hook selected.
        route_reason (str): Why that route was selected.
        resumed (bool): Whether the phase was entered by a resume.
        code_revision (str): Orchestration commit the entry ran.

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
