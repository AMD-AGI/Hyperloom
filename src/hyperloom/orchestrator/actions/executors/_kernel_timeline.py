# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real-time SBD V6 ``kernel`` timeline event recorder.

One ``kernel`` event is written per KERNEL_AGENT entry, at the moment the phase
runs. It replaces the export-time projection that reconstructed the same event
from V5 state, which could only infer what the phase had not persisted: the
adopted backend was guessed from a speedup plus an artifact path, a lost
``attempt_id`` fell back to a synthetic ``kernel_id:backend:sequence``, and a
lane was linked to its rebench by matching identity strings.

The event isolates the two dispatch routes rather than merging them. GEAK is a
delegated whole-pipeline run whose per-kernel attempts arrive as a replayed
``kernel_journey.json``; forge is an orchestrated sequence of independently
gated lanes. Both used to land in one undifferentiated ``kernel_rewrites``
array, because the projection read ``kernel_journey.kernels`` without filtering
on the ``route_strategy`` the recorder had stamped.

Two axes are kept apart throughout. A lane's ``micro_decision`` is what the
candidate layer concluded about its own output; ``outcome`` is whether the
end-to-end layer adopted it, and only a settled rebench that validated can
supply that. A candidate nothing re-measured stays ``needs_review`` rather than
inheriting its own verdict, so the event can never read as having adopted
something on the strength of the candidate's own claim.

Every write is best-effort and the recorder flushes after each state change, so
a session killed mid-phase leaves an event naming the stage that was in flight.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ._timeline_fields import (
    analysis_detail as _analysis_detail,
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    failure_row as _failure_row,
    float_or_none as _float_or_none,
    flush_event as _flush_event,
    int_or_none as _int_or_none,
    now_iso_seconds as _now_iso,
    text_or_none as _text,
)

log = logging.getLogger(__name__)

_EVENT_TYPE = "kernel"

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

_LANE_BY_SOURCE = {
    SOURCE_KERNEL_REWRITE: "kernel_rewrites",
    SOURCE_FUSION: "fusion_runs",
    SOURCE_GEMM_TUNING: "gemm_tuning_runs",
    SOURCE_COLLECTIVE: "collective_runs",
}

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

    ``outcome`` is derived rather than accepted from the caller: a row with no
    ``rebench_ref`` has nothing that re-measured it end to end, so it can only
    be ``needs_review`` however confident its own ``micro_decision`` was.

    Args:
        source_kind: One of the six producers.
        run_id: Lane-stable identifier for this candidate.
        status: How the candidate's own run ended.
        started_at: ISO timestamp the candidate started, or ``None``.
        ended_at: ISO timestamp the candidate ended, or ``None``.
        duration_sec: Wall-clock seconds the candidate took, or ``None``.
        micro_decision: The candidate layer's verdict on its own output.
        rebench_ref: The rebench attempt id that re-measured it, or ``None``.
        failure_reason: Normalized failure reason, or ``None``.

    Returns:
        The shared lane-row block.
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
    source_ref: str | None,
    idempotency_key: str | None,
    task_id: str | None,
    dispatched_at: str | None,
    settled_at: str | None,
    base_tput: float | None,
    measured_tput: float | None,
    decision: str | None,
    decision_reason: str | None,
    status: str | None,
    engagement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one rebench attempt row.

    ``engagement`` is the part the orchestrator already computed but never
    persisted: the GEAK verdict path compares the config fingerprint and the
    overlay digest to decide ``validated`` versus ``fallback``, and dropped both
    booleans on the floor once the decision was made.

    Args:
        attempt_id: Ledger-stable identifier for this attempt.
        source_kind: The producer whose candidate this attempt re-measured.
        source_ref: The candidate's ``run_id``, or ``None``.
        idempotency_key: The dispatch idempotency key, or ``None``.
        task_id: The dispatched task id, or ``None``.
        dispatched_at: ISO timestamp the attempt was dispatched, or ``None``.
        settled_at: ISO timestamp the verdict landed, or ``None``.
        base_tput: The throughput the attempt measured against, or ``None``.
        measured_tput: The throughput the attempt measured, or ``None``.
        decision: The rebench verdict, or ``None`` while unsettled.
        decision_reason: Why the verdict landed that way, or ``None``.
        status: The attempt's own lifecycle status, or ``None``.
        engagement: Config / overlay verification booleans, or ``None``.

    Returns:
        The rebench attempt row.
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


def _geak_acceptance_rows(specs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split GEAK's acceptances into authored kernels and env selections.

    GEAK routes an acceptance to ``accepted_kernels`` or ``accepted_heads``
    purely by which queue proposed it, and both lanes carry the same
    parity-checked ``e2e_delta_pct``; reading only the first drops most of the
    campaign. A candidate tag such as ``cand_c0_triton`` names the slot that
    proposed the acceptance, so the row records which of the two named it.

    Args:
        specs: The combined ``accepted_kernels`` + ``accepted_heads`` rows, each
            tagged with the lane it came from.

    Returns:
        The authored-kernel rows and the env-selection rows.
    """
    authored: list[dict[str, Any]] = []
    env: list[dict[str, Any]] = []
    for spec in _as_list(specs):
        row = _as_dict(spec)
        if not row:
            continue
        lane = str(row.get("lane") or "")
        delta = _float_or_none(row.get("e2e_delta_pct"))
        op_kind = _text(row.get("op_kind"))
        if str(row.get("kind") or "").strip().lower() == "env":
            env.append(
                {
                    "selection": str(row.get("short_name") or row.get("kernel_id") or row.get("cand_tag") or ""),
                    "op_kind": op_kind,
                    "lane": lane,
                    "e2e_delta_pct": delta,
                }
            )
            continue
        symbol = _text(row.get("short_name")) or _text(row.get("kernel_id"))
        authored.append(
            {
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
    return authored, env


class KernelTimelineRecorder:
    """Accumulates and flushes one SBD V6 ``kernel`` timeline event.

    The recorder owns a single event dict for the lifetime of one KERNEL entry.
    ``write_timeline_event`` stamps its storage sequence back onto that dict, so
    re-writing the same object updates the same file in place rather than
    appending a second event per flush.
    """

    def __init__(
        self,
        session_dir: Path | str,
        *,
        macro_cycle: int = 0,
        route: str = "",
        route_reason: str = "",
        resumed: bool = False,
        code_revision: str = "",
    ):
        """Start an in-flight kernel event.

        Args:
            session_dir: Session root the timeline lives under.
            macro_cycle: The macro cycle this entry belongs to. It identifies
                the visit, so it sits at the top of ``ext`` rather than being
                repeated inside each route's block.
            route: The dispatch route the entry hook selected.
            route_reason: Why that route was selected.
            resumed: Whether the phase was entered by a resume.
            code_revision: Orchestration commit the entry ran.
        """
        self._session_dir = Path(session_dir)
        self._t0 = time.monotonic()
        self._event: dict[str, Any] = {
            "type": _EVENT_TYPE,
            "kind": "kernel_agent",
            "status": "running",
            "start_time": _now_iso(),
            "end_time": "",
            "ext": {
                "macro_cycle": int(macro_cycle or 0),
                "in_flight_stage": "entry",
                "entry": {
                    "route": str(route or ""),
                    "route_reason": str(route_reason or ""),
                    "resumed": bool(resumed),
                    "code_revision": _text(code_revision),
                    "stack_depth_in": None,
                    "budget_remaining_sec": None,
                    "roofline_snapshot_id": None,
                    "roofline_snapshot_ts": None,
                    "roofline_baseline_gain_at_snapshot": None,
                    "snapshot_staleness": None,
                },
                "geak": None,
                "forge": None,
                "outcome": {
                    "route": str(route or ""),
                    "verdict": None,
                    "exit_reason": None,
                    "tput_before": None,
                    "tput_after": None,
                    "net_gain_pct": None,
                    "session_baseline_tput": None,
                    "cumulative_gain_validated_out": None,
                    "stack_depth_out": None,
                    "adopted": [],
                    "pending_review": [],
                    "by_source": _empty_by_source(),
                    "stack_delta": {"added": [], "removed": []},
                },
                "failure": None,
            },
        }

    # ---- internals -------------------------------------------------------

    @property
    def _ext(self) -> dict[str, Any]:
        return self._event["ext"]

    def _flush(self, component: str) -> None:
        """Persist the event, parking any writer failure for the next export."""
        _flush_event(self._session_dir, self._event, component=f"kernel.{component}")

    def _forge(self) -> dict[str, Any]:
        """Return the forge block, creating it on first use."""
        block = self._ext.get("forge")
        if not isinstance(block, dict):
            block = {
                "engaged": True,
                "reprofile": None,
                "trace_analyze_runs": [],
                "lanes": {
                    "kernel_rewrites": [],
                    "fusion_runs": [],
                    "gemm_tuning_runs": [],
                    "collective_runs": [],
                },
                "rebench_ledger": [],
            }
            self._ext["forge"] = block
        return block

    def _geak(self) -> dict[str, Any]:
        """Return the GEAK block, creating it on first use."""
        block = self._ext.get("geak")
        if not isinstance(block, dict):
            block = {
                "engaged": True,
                "handoff": None,
                "delegation": None,
                "attempts": None,
                "claim": None,
                "product": None,
                "rebench": {
                    "required": False,
                    "max_attempts": None,
                    "attempts_used": 0,
                    "attempts": [],
                    "final_status": None,
                    "final_error_class": None,
                    "final_error": None,
                },
            }
            self._ext["geak"] = block
        return block

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
        """Write the in-flight event so a killed session still shows the entry.

        ``tput_before`` is the throughput the previous stage exited on, which is
        what this entry's net gain must be measured against. The session
        baseline is recorded beside it because the rebench path measures against
        that instead, and a reader comparing the two needs both on record.

        Args:
            stack_depth_in: Optimization-stack depth on entry.
            budget_remaining_sec: Phase budget left on entry.
            tput_before: Throughput the previous stage exited on.
            session_baseline_tput: The session's raw baseline throughput.
            snapshot: The ``last_trace_analyze`` cache the entry inherited.
            snapshot_staleness: ``fresh`` / ``stale`` / ``absent``.
        """
        inherited = _as_dict(snapshot)
        entry = self._ext["entry"]
        entry["stack_depth_in"] = _int_or_none(stack_depth_in)
        entry["budget_remaining_sec"] = _float_or_none(budget_remaining_sec)
        entry["roofline_snapshot_id"] = _int_or_none(inherited.get("roofline_snapshot_id"))
        entry["roofline_snapshot_ts"] = _text(inherited.get("ts"))
        entry["roofline_baseline_gain_at_snapshot"] = _float_or_none(
            inherited.get("roofline_baseline_gain_at_snapshot")
        )
        entry["snapshot_staleness"] = _text(snapshot_staleness)
        outcome = self._ext["outcome"]
        outcome["tput_before"] = _float_or_none(tput_before)
        outcome["session_baseline_tput"] = _float_or_none(session_baseline_tput)
        self._flush("begin")

    def enter_stage(self, stage: str) -> None:
        """Name the stage now in flight so a kill leaves it identifiable.

        Args:
            stage: The stage the phase is entering.
        """
        self._ext["in_flight_stage"] = _text(stage)
        self._flush("stage")

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
    ) -> None:
        """Record the entry re-profile that decides whether analysis is stale.

        ``task_kind`` is the judgement the rest of the chain hangs on: a
        ``roofline`` task carries its own ``trace_analyze`` and refreshes the
        cache, while a plain ``profile`` task invalidates it and forces the phase
        to request analysis of its own. This runs on the forge and
        collective-only routes only -- GEAK profiles from scratch itself and is
        handed no trace, so re-profiling for it would buy nothing.

        Args:
            ran: Whether a re-profile was actually dispatched.
            task_kind: ``roofline`` or ``profile``.
            trigger: ``gain`` / ``config_changed`` / ``workload_changed``.
            skipped_reason: Why it was skipped, when it was.
            idempotency_reason: The dispatch reason tag.
            snapshot_landed: Whether a new snapshot actually landed.
            snapshot_id_before: Snapshot counter before the attempt.
            snapshot_id_after: Snapshot counter after the attempt.
        """
        self._forge()["reprofile"] = {
            "ran": bool(ran),
            "task_kind": _text(task_kind),
            "trigger": _text(trigger),
            "skipped_reason": _text(skipped_reason),
            "idempotency_reason": _text(idempotency_reason),
            "snapshot_landed": bool(snapshot_landed),
            "snapshot_id_before": _int_or_none(snapshot_id_before),
            "snapshot_id_after": _int_or_none(snapshot_id_after),
        }
        self._flush("reprofile")

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

        This array is normally empty. The entry re-profile dispatches a
        ``roofline`` task by default, which analyses the trace it just captured,
        so the phase's own request is skipped as cached. A non-empty array
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
            result: The analysis tool's result dict.
            requested_by: The role that requested it.
            request_msg_id: The bus request message id.
            trace_input: The trace the run analysed.
            top_k: The requested ranking depth.
            snapshot: The ``last_trace_analyze`` cache the run produced.
            cache_hit: Whether a cached result served the request.
        """
        produced = _as_dict(snapshot)
        row = {
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
        }
        self._forge()["trace_analyze_runs"].append(row)
        self._flush("trace_analyze")

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
            run_id: The real attempt id for this rewrite.
            kernel_id: The kernel the rewrite targeted.
            status: How the rewrite's own run ended.
            kernel_name: Human-readable kernel name.
            dispatched: Whether a backend was actually dispatched.
            backends_tried: The backends attempted.
            adopted_backend: The backend whose output was taken.
            skip_reason: Why dispatch was skipped, when it was.
            task_group: The dispatch task group.
            speedup: Micro-benchmark speedup.
            baseline_us: Micro-benchmark baseline microseconds.
            candidate_us: Micro-benchmark candidate microseconds.
            compile_status: Compilation outcome.
            correctness: Correctness verdict.
            artifact_path: The produced artifact.
            micro_decision: The candidate layer's own verdict.
            rebench_ref: The rebench attempt that re-measured it.
            trace_analyze_ref: The analysis that nominated this kernel.
            e2e: The end-to-end integration sub-result.
            started_at: ISO timestamp the rewrite started.
            ended_at: ISO timestamp the rewrite ended.
            duration_sec: Wall-clock seconds the rewrite took.
            failure_reason: Normalized failure reason.
        """
        integration = _as_dict(e2e)
        row = {
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
                "validated": integration.get("validated") if isinstance(integration.get("validated"), bool) else None,
                "decision": _text(integration.get("decision")),
                "patch_path": _text(integration.get("patch_path")),
                "target_file": _text(integration.get("target_file")),
            }
            if integration
            else None,
        }
        self._forge()["lanes"]["kernel_rewrites"].append(row)
        self._flush("kernel_rewrite")

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
            run_id: Lane-stable identifier for this run.
            status: How the run ended.
            pattern: The fusion pattern attempted.
            target_module: The module the fusion targeted.
            applied: Whether the fusion was applied.
            gain_pct: The gain the run claimed.
            patch_path: The produced patch.
            micro_decision: The candidate layer's own verdict.
            rebench_ref: The rebench attempt that re-measured it.
            started_at: ISO timestamp the run started.
            ended_at: ISO timestamp the run ended.
            duration_sec: Wall-clock seconds the run took.
            failure_reason: Normalized failure reason.
        """
        row = {
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
        self._forge()["lanes"]["fusion_runs"].append(row)
        self._flush("fusion")

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
            run_id: Lane-stable identifier for this run.
            status: How the run ended.
            shapes_total: Shapes the run considered.
            shapes_tuned: Shapes the run tuned.
            config_path: The produced shape-table.
            gain_pct: The gain the run claimed.
            tuner: The tuner that ran.
            micro_decision: The candidate layer's own verdict.
            rebench_ref: The rebench attempt that re-measured it.
            started_at: ISO timestamp the run started.
            ended_at: ISO timestamp the run ended.
            duration_sec: Wall-clock seconds the run took.
            failure_reason: Normalized failure reason.
        """
        row = {
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
        self._forge()["lanes"]["gemm_tuning_runs"].append(row)
        self._flush("gemm_tuning")

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
            run_id: Lane-stable identifier for this run.
            status: How the run ended.
            op: The collective operation tuned.
            algo: The algorithm selected.
            size_bytes: The message size tuned for.
            world_size: The participating rank count.
            gain_pct: The gain the run claimed.
            withheld: Whether the candidate was withheld from adoption.
            withhold_reason: Why it was withheld.
            micro_decision: The candidate layer's own verdict.
            rebench_ref: The rebench attempt that re-measured it.
            started_at: ISO timestamp the run started.
            ended_at: ISO timestamp the run ended.
            duration_sec: Wall-clock seconds the run took.
            failure_reason: Normalized failure reason.
        """
        row = {
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
        self._forge()["lanes"]["collective_runs"].append(row)
        self._flush("collective")

    def record_rebench_attempt(self, **fields: Any) -> None:
        """Append or update one forge rebench attempt.

        Re-recording the same ``attempt_id`` replaces the row rather than
        appending, so a dispatch and its later verdict describe one attempt.

        Args:
            **fields: The :func:`_rebench_row` fields.
        """
        row = _rebench_row(**fields)
        ledger = self._forge()["rebench_ledger"]
        for index, existing in enumerate(ledger):
            if existing.get("attempt_id") and existing["attempt_id"] == row["attempt_id"]:
                ledger[index] = row
                break
        else:
            ledger.append(row)
        self._flush("rebench")

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
            handoff: The handoff dict written for the runner.
        """
        payload = _as_dict(handoff)
        envs = payload.get("accepted_env")
        self._geak()["handoff"] = {
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
            "orchestrator_best_tput_same_config": _float_or_none(payload.get("orchestrator_best_tput_same_config")),
            "max_model_len": _int_or_none(payload.get("max_model_len")),
            "mem_fraction": _float_or_none(payload.get("mem_fraction")),
            "bench_client": _text(payload.get("bench_client")),
            "e2e_metric": _text(payload.get("e2e_metric")),
            "bench_protocol_present": bool(payload.get("bench_protocol")),
            "gpu_ids": _text(payload.get("gpu_ids")),
            "exp_root": _text(payload.get("exp_root")),
            "eval_dir": _text(payload.get("eval_dir")),
        }
        self._flush("geak_handoff")

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
            runner_status: The runner's own status.
            started_at: ISO timestamp the runner started.
            ended_at: ISO timestamp the runner ended.
            duration_sec: Wall-clock seconds the runner took.
            error_class: The failure class, on a miss.
            error: The failure message, on a miss.
            returncode: The runner's exit code.
            runner_timeout_sec: The runner's budget.
            kill_timeout_sec: The runner's hard kill budget.
            exp_root: The runner's experiment root.
            eval_dir: The macro-cycle-scoped eval dir.
            report_path: The human report the runner wrote.
            versions: Tool version provenance.
            recovered_from_disk: Whether the result was reconstructed on-disk.
            stages_reached: Stages a crashed run reached.
        """
        self._geak()["delegation"] = {
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
        self._flush("geak_delegation")

    def record_geak_attempts(self, journey: dict[str, Any] | None) -> None:
        """Record what GEAK tried, from the journey it emits.

        GEAK's ``kernel_journey.json`` names every kernel it considered, which
        backends it dispatched, and what each one measured -- not just the
        acceptances that survived. Those rows are shaped exactly like forge's
        because the orchestrator replays them through the same recorder calls,
        which is precisely why they must be stored under GEAK rather than merged
        into the forge lane the projection appended them to.

        Args:
            journey: The parsed ``kernel_journey.json``.
        """
        parsed = _as_dict(journey)
        discovery: list[dict[str, Any]] = []
        for run in _as_list(parsed.get("discovery_runs")):
            row = _as_dict(run)
            if not row:
                continue
            discovery.append(
                {
                    "source": _text(row.get("source")),
                    "status": _text(row.get("status")),
                    "hot_kernel_count": len(_as_list(row.get("hot_kernels"))),
                    "scan": _as_dict(row.get("scan")),
                }
            )
        kernels: list[dict[str, Any]] = []
        counts = {
            "discovered": 0,
            "dispatched": 0,
            "skipped": 0,
            "backend_ok": 0,
            "backend_fail": 0,
            "integrated": 0,
        }
        for kernel in _as_list(parsed.get("kernels")):
            row = _as_dict(kernel)
            kernel_id = _text(row.get("kernel_id"))
            if not kernel_id:
                continue
            dispatch = _as_dict(row.get("dispatch"))
            backend = _as_dict(row.get("backend_result"))
            e2e = _as_dict(row.get("e2e"))
            dispatched = bool(dispatch.get("dispatched", True))
            counts["discovered"] += 1
            counts["dispatched" if dispatched else "skipped"] += 1
            if backend:
                counts[
                    "backend_ok"
                    if str(backend.get("status") or "") in {"ok", "success", "succeeded"}
                    else "backend_fail"
                ] += 1
            if e2e.get("integrated"):
                counts["integrated"] += 1
            kernels.append(
                {
                    "kernel_id": kernel_id,
                    "dispatched": dispatched,
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
                }
            )
        self._geak()["attempts"] = {
            "discovery_runs": discovery,
            "kernels": kernels,
            "counts": counts,
        }
        self._flush("geak_attempts")

    def record_geak_claim(self, pending: dict[str, Any] | None, *, specs: Any = None) -> None:
        """Record what GEAK reported about itself, before any re-measurement.

        Every number here is the optimizer's own account of its run. ``verified``
        is stored as a constant so a consumer cannot mistake this block for a
        conclusion: nothing in it has been re-measured by the orchestrator's own
        harness, and the adoption verdict rests solely on the rebench.

        Args:
            pending: The recorded GEAK candidate slot.
            specs: The combined acceptance rows from both proposal queues.
        """
        slot = _as_dict(pending)
        authored, env = _geak_acceptance_rows(specs)
        self._geak()["claim"] = {
            "verified": False,
            "self_reported_tput": _float_or_none(slot.get("self_reported_tput")),
            "self_reported_speedup": _float_or_none(slot.get("self_reported_speedup")),
            "self_reported_gain_pct": _float_or_none(slot.get("self_reported_gain_pct")),
            "self_reported_basis": _text(slot.get("self_reported_basis")),
            "geak_status": _text(slot.get("geak_status")),
            "baseline_alignment_status": _text(slot.get("baseline_alignment_status")),
            "authored_kernels": authored,
            "env_selections": env,
            "kernels_optimized": len(authored),
            "accepted_heads_count": sum(1 for row in authored + env if row.get("lane") == "headQueue"),
            "validated_regimes": _as_list(slot.get("validated_regimes")),
        }
        self._flush("geak_claim")

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
            accepted_flags: Server flags GEAK accepted.
            accepted_envs: Environment variables GEAK accepted.
            accepted_config: The runner's own accepted-config block.
            cfg_hash: Canonical fingerprint of the flags and envs.
            final_overlay: The overlay PYTHONPATH GEAK produced.
            final_overlay_digest: Digest of that overlay.
            final_launch_script: The optimized launch script.
            bench_script: The benchmark script GEAK measured with.
            final_patch: The aggregate source patch.
        """
        flags = accepted_flags
        self._geak()["product"] = {
            "accepted_flags": [str(item) for item in _as_list(flags)] if not isinstance(flags, str) else _text(flags),
            "accepted_envs": _as_dict(accepted_envs),
            "accepted_config": _as_dict(accepted_config),
            "cfg_hash": _text(cfg_hash),
            "final_overlay": _text(final_overlay),
            "final_overlay_digest": _text(final_overlay_digest),
            "final_launch_script": _text(final_launch_script),
            "bench_script": _text(bench_script),
            "final_patch": _text(final_patch),
        }
        self._flush("geak_product")

    def record_geak_rebench_attempt(self, *, max_attempts: Any = None, **fields: Any) -> None:
        """Append or update one GEAK rebench attempt.

        Args:
            max_attempts: The per-cycle attempt ceiling.
            **fields: The :func:`_rebench_row` fields.
        """
        fields.setdefault("source_kind", SOURCE_GEAK_AUTHORED_KERNEL)
        row = _rebench_row(**fields)
        rebench = self._geak()["rebench"]
        rebench["required"] = True
        if max_attempts is not None:
            rebench["max_attempts"] = _int_or_none(max_attempts)
        attempts = rebench["attempts"]
        for index, existing in enumerate(attempts):
            if existing.get("attempt_id") and existing["attempt_id"] == row["attempt_id"]:
                attempts[index] = row
                break
        else:
            attempts.append(row)
        rebench["attempts_used"] = len(attempts)
        self._flush("geak_rebench")

    def record_geak_rebench_conclusion(
        self,
        *,
        final_status: str = "",
        final_error_class: str = "",
        final_error: str = "",
    ) -> None:
        """Record the terminal revalidation state of the GEAK candidate.

        Args:
            final_status: The revalidation status the result was stamped with.
            final_error_class: The revalidation failure class.
            final_error: The revalidation failure message.
        """
        rebench = self._geak()["rebench"]
        rebench["final_status"] = _text(final_status)
        rebench["final_error_class"] = _text(final_error_class)
        rebench["final_error"] = _clip(final_error) or None
        self._flush("geak_rebench_conclusion")

    # ---- conclusion ------------------------------------------------------

    def _settle_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
        """Resolve every candidate row against the rebench that measured it.

        Returns:
            The adopted rows, the rows still awaiting review, and the per-source
            counters.
        """
        by_source = _empty_by_source()
        adopted: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []

        forge = _as_dict(self._ext.get("forge"))
        verdicts: dict[str, dict[str, Any]] = {
            str(row.get("attempt_id")): row for row in _as_list(forge.get("rebench_ledger")) if row.get("attempt_id")
        }
        geak = _as_dict(self._ext.get("geak"))
        for row in _as_list(_as_dict(geak.get("rebench")).get("attempts")):
            if row.get("attempt_id"):
                verdicts[str(row["attempt_id"])] = row

        def _resolve(row: dict[str, Any], *, ref: str, gain: Any, unsettled_why: str = "") -> None:
            source = str(row.get("source_kind") or "")
            counters = by_source.setdefault(source, {"attempted": 0, "adopted": 0, "needs_review": 0, "rejected": 0})
            counters["attempted"] += 1
            if str(row.get("status") or "") in {"failed", "timeout"}:
                row["outcome"] = OUTCOME_REJECTED
                counters["rejected"] += 1
                return
            verdict = verdicts.get(ref) if ref else None
            decision = str(_as_dict(verdict).get("decision") or "")
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

        for lane in _LANE_BY_SOURCE.values():
            for row in _as_list(_as_dict(forge.get("lanes")).get(lane)):
                _resolve(
                    row,
                    ref=str(row.get("rebench_ref") or ""),
                    gain=row.get("gain_pct")
                    if row.get("gain_pct") is not None
                    else _as_dict(row.get("e2e")).get("e2e_gain_pct"),
                )

        claim = _as_dict(geak.get("claim"))
        # GEAK may rebench the same candidate up to its per-cycle ceiling, so
        # unlike a forge lane it can end the visit holding several settled
        # verdicts. Taking the newest would let a KEEP after a REVERT read as an
        # adoption; two settled verdicts that disagree is a fact worth seeing,
        # so the candidate stays pending and neither verdict is honoured.
        settled = [
            _as_dict(attempt)
            for attempt in _as_list(_as_dict(geak.get("rebench")).get("attempts"))
            if str(_as_dict(attempt).get("decision") or "")
        ]
        geak_ref = ""
        if settled and len({str(attempt.get("decision")) for attempt in settled}) == 1:
            geak_ref = str(settled[-1].get("attempt_id") or "")
        elif settled:
            self._ext["geak"]["rebench"]["conflicting_decisions"] = sorted(
                {str(attempt.get("decision")) for attempt in settled}
            )
        for source, rows in (
            (SOURCE_GEAK_AUTHORED_KERNEL, claim.get("authored_kernels")),
            (SOURCE_GEAK_ENV_SELECTION, claim.get("env_selections")),
        ):
            for row in _as_list(rows):
                entry = _as_dict(row)
                synthetic = {
                    "source_kind": source,
                    "run_id": entry.get("short_name") or entry.get("kernel_id") or entry.get("selection") or "",
                    "status": "success",
                    "rebench_ref": geak_ref,
                }
                _resolve(
                    synthetic,
                    ref=geak_ref,
                    gain=entry.get("e2e_delta_pct"),
                    unsettled_why="rebench_conflict" if len(settled) > 1 and not geak_ref else "",
                )
                entry["outcome"] = synthetic["outcome"]

        return adopted, pending, by_source

    def _derive_status(self, *, candidate_count: int) -> str:
        """Read the event status off what the rebenches actually measured.

        A rebench that concluded against its candidate still concluded, so it
        makes the visit ``succeeded``; the distinction that matters is between
        measuring and not measuring. A candidate that was built and never
        validated leaves the visit ``degraded`` -- the work happened but nothing
        settled it -- while a visit that produced no candidate at all and
        measured nothing is ``skipped`` rather than degraded, because there was
        nothing there to degrade.

        Args:
            candidate_count: How many candidates the visit produced.

        Returns:
            The event status.
        """
        forge = _as_dict(self._ext.get("forge"))
        geak = _as_dict(self._ext.get("geak"))
        attempts = [
            _as_dict(row)
            for row in (
                *_as_list(forge.get("rebench_ledger")),
                *_as_list(_as_dict(geak.get("rebench")).get("attempts")),
            )
        ]
        if any(_float_or_none(row.get("measured_tput")) is not None for row in attempts):
            return "succeeded"
        if attempts and all(str(row.get("status") or "").lower() in _REBENCH_FAULTED_STATUSES for row in attempts):
            return "failed"
        return "degraded" if candidate_count else "skipped"

    def finish(
        self,
        *,
        verdict: str,
        status: str = "",
        exit_reason: str = "",
        tput_after: Any = None,
        cumulative_gain_validated_out: Any = None,
        stack_depth_out: Any = None,
        stack_added: Any = None,
        stack_removed: Any = None,
    ) -> None:
        """Close the event and settle every candidate against its rebench.

        Args:
            verdict: The entry's conclusion. Left unstamped when nothing was
                adopted, so a visit that adopted nothing cannot read as having
                concluded something about a candidate.
            status: The event status to close with. Derived from the rebench
                evidence when empty, so the ladder cannot drift per call site.
            exit_reason: The phase's own exit reason.
            tput_after: Throughput the entry exited on.
            cumulative_gain_validated_out: Validated cumulative gain on exit.
            stack_depth_out: Optimization-stack depth on exit.
            stack_added: Stack entries this visit added.
            stack_removed: Stack entries this visit removed.
        """
        adopted, pending, by_source = self._settle_rows()
        outcome = self._ext["outcome"]
        before = _float_or_none(outcome.get("tput_before"))
        after = _float_or_none(tput_after)
        outcome["verdict"] = _text(verdict) if adopted else None
        outcome["exit_reason"] = _text(exit_reason)
        outcome["tput_after"] = after
        outcome["net_gain_pct"] = (
            round((after - before) / before * 100.0, 4)
            if before is not None and after is not None and before > 0
            else None
        )
        outcome["cumulative_gain_validated_out"] = _float_or_none(cumulative_gain_validated_out)
        outcome["stack_depth_out"] = _int_or_none(stack_depth_out)
        outcome["adopted"] = adopted
        outcome["pending_review"] = pending
        outcome["by_source"] = by_source
        outcome["stack_delta"] = {
            "added": [_as_dict(row) for row in _as_list(stack_added)],
            "removed": [_as_dict(row) for row in _as_list(stack_removed)],
        }
        self._ext["in_flight_stage"] = None
        candidate_count = sum(counters["attempted"] for counters in by_source.values())
        self._event["status"] = str(status) or self._derive_status(candidate_count=candidate_count)
        self._event["end_time"] = _now_iso()
        self._ext["duration_sec"] = round(time.monotonic() - self._t0, 3)
        self._flush("finish")

    def finish_failed(self, *, stage: str, error_class: str = "", message: Any = "") -> None:
        """Close the event as failed, naming the stage that failed.

        Args:
            stage: The stage that failed.
            error_class: The failure class.
            message: The failure message.
        """
        self._ext["failure"] = _failure_row(
            phase=stage,
            error_class=error_class or f"{stage}_failed",
            message=message,
        )
        self.finish(verdict="failed", status="failed", exit_reason=str(stage or ""))

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an event whose phase raised instead of returning.

        Args:
            exc: The exception propagating out of the phase.
        """
        if self._event.get("end_time"):
            return
        self.finish_failed(
            stage=str(self._ext.get("in_flight_stage") or "kernel"),
            error_class=type(exc).__name__,
            message=f"kernel phase raised: {exc!r}",
        )


def make_kernel_recorder(
    session_dir: Path | str,
    *,
    macro_cycle: int = 0,
    route: str = "",
    route_reason: str = "",
    resumed: bool = False,
    code_revision: str = "",
) -> KernelTimelineRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed.

    KERNEL behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating. An unresolved
    session directory declines too: writing the timeline into whatever the
    working directory happens to be is worse than not recording.

    Args:
        session_dir: Session root the timeline lives under.
        macro_cycle: The macro cycle this entry belongs to.
        route: The dispatch route the entry hook selected.
        route_reason: Why that route was selected.
        resumed: Whether the phase was entered by a resume.
        code_revision: Orchestration commit the entry ran.

    Returns:
        The recorder, or ``None`` when it could not be built.
    """
    try:
        root = Path(session_dir)
        if not root.name or not root.is_dir():
            log.debug("kernel timeline: unresolved session dir %r; not recording", str(session_dir))
            return None
        return KernelTimelineRecorder(
            root,
            macro_cycle=macro_cycle,
            route=route,
            route_reason=route_reason,
            resumed=resumed,
            code_revision=code_revision,
        )
    except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
        log.debug("kernel timeline: recorder construction failed; not recording", exc_info=True)
        return None
