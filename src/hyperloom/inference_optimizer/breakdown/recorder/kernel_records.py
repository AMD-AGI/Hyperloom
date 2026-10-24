# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The records a KERNEL visit writes, each normalized into the row or block it persists as."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .event_fields import (
    analysis_detail as _analysis_detail,
    as_dict as _as_dict,
    as_list as _as_list,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    now_iso_seconds as _now_iso,
    text_or_none as _text,
)

# The five candidate producers a KERNEL entry can adopt from.
SOURCE_KERNEL_REWRITE = "kernel_rewrite"
SOURCE_FUSION = "fusion"
SOURCE_GEMM_TUNING = "gemm_tuning"
SOURCE_GEAK_AUTHORED_KERNEL = "geak_authored_kernel"

#: Which route asked for a rebench. Only GEAK runs one: it re-measures a whole
#: delegated config against a per-cycle attempt ceiling. Forge has no rebench
#: step at all -- its candidates are settled by the integrate gate.
LEDGER_GEAK = "geak"

#: ``geak_pending.status`` values left by a candidate whose revalidation ended
#: without a verdict, so the win was abandoned rather than judged.
GEAK_DROPPED_STATUSES: frozenset[str] = frozenset({"rebench_cancelled", "rebench_declined", "overlay_unloadable"})

__all__ = [
    "GEAK_DROPPED_STATUSES",
    "LEDGER_GEAK",
    "SOURCE_FUSION",
    "SOURCE_GEAK_AUTHORED_KERNEL",
    "SOURCE_GEMM_TUNING",
    "SOURCE_KERNEL_REWRITE",
    "GeakRebenchAttempt",
    "IntegrateVerdict",
    "KernelReprofile",
    "KernelRewriteRun",
    "LaneRunFields",
    "TraceAnalyzeRun",
]


@dataclass(frozen=True)
class LaneRunFields:
    """The fields every candidate lane run carries.

    The row states what the lane did, not what became of it. A forge candidate
    is settled by the integrate gate, whose verdict is a different row written
    later, so the outcome is resolved at assembly once both halves are on disk.

    ``integrate_ref`` names the queued integration when the lane has one. A
    rewrite is joined to its verdict by ``kernel_id``, but fusion and GEMM
    tuning produce no kernel of their own, so without the integration id the
    gate that ruled on them could never be found.
    """

    run_id: str
    status: str
    started_at: str = ""
    ended_at: str = ""
    duration_sec: Any = None
    micro_decision: str = ""
    integrate_ref: str = ""
    failure_reason: str = ""
    error_class: str = ""

    def row(self, source_kind: str) -> dict[str, Any]:
        """The lane row's shared fields, for a run of ``source_kind``."""
        return {
            "source_kind": str(source_kind),
            "run_id": str(self.run_id or ""),
            "status": str(self.status or ""),
            "started_at": _text(self.started_at),
            "ended_at": _text(self.ended_at),
            "duration_sec": _float_or_none(self.duration_sec),
            "micro_decision": _text(self.micro_decision),
            "integrate_ref": _text(self.integrate_ref),
            "error_class": _text(self.error_class),
            "failure_reason": _text(self.failure_reason),
        }


@dataclass(frozen=True)
class KernelRewriteRun:
    """One forge source-level kernel rewrite.

    ``adopted_backend`` and ``lane.run_id`` are stated rather than derived. The
    projection had to guess the backend from a speedup plus an artifact path
    and to synthesize an identifier from ``kernel_id:backend:sequence``
    whenever the real attempt id had been lost.
    """

    lane: LaneRunFields
    kernel_id: str
    kernel_name: str = ""
    dispatched: bool = True
    backends_tried: Any = None
    adopted_backend: str = ""
    skip_reason: str = ""
    task_group: str = ""
    speedup: Any = None
    baseline_us: Any = None
    candidate_us: Any = None
    compile_status: str = ""
    correctness: Any = None
    artifact_path: str = ""
    trace_analyze_ref: str = ""

    def row(self) -> dict[str, Any]:
        """The lane row this rewrite persists as."""
        return {
            **self.lane.row(SOURCE_KERNEL_REWRITE),
            "kernel_id": str(self.kernel_id or ""),
            "kernel_name": _text(self.kernel_name),
            "dispatched": bool(self.dispatched),
            "backends_tried": [str(item) for item in _as_list(self.backends_tried)],
            "adopted_backend": _text(self.adopted_backend),
            "skip_reason": _text(self.skip_reason),
            "task_group": _text(self.task_group),
            "speedup": _float_or_none(self.speedup),
            "baseline_us": _float_or_none(self.baseline_us),
            "candidate_us": _float_or_none(self.candidate_us),
            "compile_status": _text(self.compile_status),
            "correctness": self.correctness if isinstance(self.correctness, bool) else None,
            "artifact_path": _text(self.artifact_path),
            "trace_analyze_ref": _text(self.trace_analyze_ref),
        }


@dataclass(frozen=True)
class KernelReprofile:
    """The entry re-profile that decides whether the cached analysis is stale."""

    ran: bool
    task_kind: str = ""
    trigger: str = ""
    skipped_reason: str = ""
    idempotency_reason: str = ""
    snapshot_landed: bool = False
    snapshot_id_before: Any = None
    snapshot_id_after: Any = None
    task_id: str = ""

    def block(self) -> dict[str, Any]:
        """The ``forge_reprofile`` block this re-profile persists as."""
        return {
            "ran": bool(self.ran),
            "task_id": _text(self.task_id),
            "task_kind": _text(self.task_kind),
            "trigger": _text(self.trigger),
            "skipped_reason": _text(self.skipped_reason),
            "idempotency_reason": _text(self.idempotency_reason),
            "snapshot_landed": bool(self.snapshot_landed),
            "snapshot_id_before": _int_or_none(self.snapshot_id_before),
            "snapshot_id_after": _int_or_none(self.snapshot_id_after),
        }


@dataclass(frozen=True)
class IntegrateVerdict:
    """The end-to-end integrate gate's verdict on one queued patch."""

    integration_id: str
    kernel_id: str
    #: The gate's verdict (``KEEP`` / ``REVERT``).
    decision: str = ""
    status: str = ""
    attempt_count: Any = None
    #: How many attempts never measured the patch fairly.
    fault_count: Any = None
    gain_pct: Any = None
    accuracy_pass: Any = None
    validation_tier: str = ""
    patch_path: str = ""
    target_file: str = ""
    error_class: str = ""
    #: Why the patch was rejected outright, when it was -- an exhausted budget
    #: rather than a verdict on the code.
    rejected_reason: str = ""
    retryable: bool = False
    settled_at: str = ""
    extra_server_args: str = ""
    #: Throughput basis the gain was measured on (``hot`` / ``cold``); a gain
    #: is meaningless without the baseline behind it.
    basis: str = ""
    alignment_status: str = ""
    #: Whether the measured gain is this one kernel's; a rebench carrying
    #: several kernels measured all of them together.
    gain_attributed: Any = None

    def row(self, macro_cycle: Any) -> dict[str, Any]:
        """The integrate row this verdict persists as, settled in ``macro_cycle``."""
        return {
            "integration_id": str(self.integration_id),
            "kernel_id": str(self.kernel_id or ""),
            "decision": _text(self.decision),
            "status": _text(self.status),
            "attempt_count": _int_or_none(self.attempt_count),
            "fault_count": _int_or_none(self.fault_count),
            "gain_pct": _float_or_none(self.gain_pct),
            "accuracy_pass": self.accuracy_pass if isinstance(self.accuracy_pass, bool) else None,
            "validation_tier": _text(self.validation_tier),
            "patch_path": _text(self.patch_path),
            "target_file": _text(self.target_file),
            "error_class": _text(self.error_class),
            "rejected_reason": _text(self.rejected_reason),
            "retryable": bool(self.retryable),
            "settled_at": _text(self.settled_at),
            "settled_in_macro_cycle": _int_or_none(macro_cycle),
            "extra_server_args": _text(self.extra_server_args),
            "basis": _text(self.basis),
            "alignment_status": _text(self.alignment_status),
            "gain_attributed": self.gain_attributed if isinstance(self.gain_attributed, bool) else None,
        }


@dataclass(frozen=True)
class TraceAnalyzeRun:
    """One trace analysis, the result it returned and the cache it left."""

    run_id: str
    status: str
    result: Any
    requested_by: str = ""
    request_msg_id: str = ""
    trace_input: str = ""
    top_k: Any = None
    #: A ``last_trace_analyze``-shaped cache.
    snapshot: dict[str, Any] | None = None
    cache_hit: bool = False

    def row(self, trigger: str) -> dict[str, Any]:
        """The analysis row this run persists as, for a run asked for by ``trigger``."""
        produced = _as_dict(self.snapshot)
        return {
            "run_id": str(self.run_id or ""),
            "trigger": _text(trigger),
            "requested_by": _text(self.requested_by),
            "request_msg_id": _text(self.request_msg_id),
            "ts": _now_iso(),
            "status": str(self.status or ""),
            "cache_hit": bool(self.cache_hit),
            "trace_input": _text(self.trace_input),
            "top_k": _int_or_none(self.top_k),
            "roofline_snapshot_id": _int_or_none(produced.get("roofline_snapshot_id")),
            "roofline_baseline_gain_at_snapshot": _float_or_none(produced.get("roofline_baseline_gain_at_snapshot")),
            "steady_state_trace": _text(produced.get("steady_state_trace")),
            "analysis_md_path": _text(produced.get("analysis_md_path")),
            "reusable_native_kernel_ids": [str(item) for item in _as_list(produced.get("reusable_native_kernel_ids"))],
            **_analysis_detail(self.result),
        }


@dataclass(frozen=True)
class GeakRebenchAttempt:
    """One end-to-end re-measurement of a GEAK candidate."""

    attempt_id: str
    source_kind: str = SOURCE_GEAK_AUTHORED_KERNEL
    source_ref: str | None = None
    idempotency_key: str | None = None
    task_id: str | None = None
    dispatched_at: str | None = None
    settled_at: str | None = None
    base_tput: float | None = None
    measured_tput: float | None = None
    decision: str | None = None
    decision_reason: str | None = None
    status: str | None = None
    #: The config fingerprint and overlay digest the verdict compared to decide
    #: ``validated`` versus ``fallback``.
    engagement: dict[str, Any] | None = None

    def row(self) -> dict[str, Any]:
        """The rebench row this attempt persists as."""
        verified = _as_dict(self.engagement)
        base = _float_or_none(self.base_tput)
        measured = _float_or_none(self.measured_tput)
        delta = None
        if base is not None and measured is not None and base > 0:
            delta = round((measured - base) / base * 100.0, 4)
        return {
            "attempt_id": str(self.attempt_id or ""),
            "source_kind": str(self.source_kind),
            "ledger": LEDGER_GEAK,
            "source_ref": _text(self.source_ref),
            "idempotency_key": _text(self.idempotency_key),
            "task_id": _text(self.task_id),
            "dispatched_at": _text(self.dispatched_at),
            "settled_at": _text(self.settled_at),
            "base_tput": base,
            "measured_tput": measured,
            "delta_pct": delta,
            "decision": _text(self.decision),
            "decision_reason": _text(self.decision_reason),
            "status": _text(self.status),
            "engagement": {
                "config_matched": verified.get("config_matched"),
                "overlay_loaded": verified.get("overlay_loaded"),
                "expected_cfg_hash": _text(verified.get("expected_cfg_hash")),
                "observed_cfg_hash": _text(verified.get("observed_cfg_hash")),
                "expected_overlay_digest": _text(verified.get("expected_overlay_digest")),
                "observed_overlay_digest": _text(verified.get("observed_overlay_digest")),
            },
        }
