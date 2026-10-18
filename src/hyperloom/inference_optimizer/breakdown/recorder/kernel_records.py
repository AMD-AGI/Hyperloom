# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The records a KERNEL visit writes, each normalized into the row or block it persists as."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .event_fields import (
    as_list as _as_list,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
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

__all__ = [
    "LEDGER_GEAK",
    "SOURCE_FUSION",
    "SOURCE_GEAK_AUTHORED_KERNEL",
    "SOURCE_GEMM_TUNING",
    "SOURCE_KERNEL_REWRITE",
    "FusionRun",
    "GemmTuningRun",
    "IntegrateVerdict",
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
class FusionRun:
    """One forge-fusion run."""

    lane: LaneRunFields
    pattern: str = ""
    target_module: str = ""
    applied: bool = False
    gain_pct: Any = None
    patch_path: str = ""

    def row(self) -> dict[str, Any]:
        """The lane row this run persists as."""
        return {
            **self.lane.row(SOURCE_FUSION),
            "pattern": _text(self.pattern),
            "target_module": _text(self.target_module),
            "applied": bool(self.applied),
            "gain_pct": _float_or_none(self.gain_pct),
            "patch_path": _text(self.patch_path),
        }


@dataclass(frozen=True)
class GemmTuningRun:
    """One GEMM shape-table tuning run.

    ``graded_objective`` names the axis ``gain_pct`` was measured on, so a
    run graded on total throughput or interactivity is not later read as an
    output gain. ``tuner`` names which backend produced the table, which is
    a separate question from which axis judged it.
    """

    lane: LaneRunFields
    shapes_total: Any = None
    shapes_tuned: Any = None
    config_path: str = ""
    gain_pct: Any = None
    graded_objective: str = ""
    tuner: str = ""

    def row(self) -> dict[str, Any]:
        """The lane row this run persists as."""
        return {
            **self.lane.row(SOURCE_GEMM_TUNING),
            "shapes_total": _int_or_none(self.shapes_total),
            "shapes_tuned": _int_or_none(self.shapes_tuned),
            "config_path": _text(self.config_path),
            "gain_pct": _float_or_none(self.gain_pct),
            "graded_objective": _text(self.graded_objective),
            "tuner": _text(self.tuner),
        }


@dataclass(frozen=True)
class IntegrateVerdict:
    """The end-to-end integrate gate's verdict on one patch.

    ``macro_cycle`` is the cycle the gate settled in, which is the cycle that
    ran the gate rather than necessarily the one that produced the patch.

    Attributes:
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

    macro_cycle: Any
    integration_id: str
    kernel_id: str
    decision: str = ""
    status: str = ""
    attempt_count: Any = None
    fault_count: Any = None
    gain_pct: Any = None
    accuracy_pass: Any = None
    validation_tier: str = ""
    patch_path: str = ""
    target_file: str = ""
    error_class: str = ""
    rejected_reason: str = ""
    retryable: bool = False
    settled_at: str = ""
    extra_server_args: str = ""
    basis: str = ""
    alignment_status: str = ""
    gain_attributed: Any = None

    def row(self) -> dict[str, Any]:
        """The integrate row this verdict persists as."""
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
            "settled_in_macro_cycle": _int_or_none(self.macro_cycle),
            "extra_server_args": _text(self.extra_server_args),
            "basis": _text(self.basis),
            "alignment_status": _text(self.alignment_status),
            "gain_attributed": self.gain_attributed if isinstance(self.gain_attributed, bool) else None,
        }


@dataclass(frozen=True)
class TraceAnalyzeRun:
    """One analysis run, as asked for and as it came back.

    ``status`` is ``ok`` or ``failed``.
    """

    run_id: str
    status: str
    result: Any
    requested_by: str = ""
    request_msg_id: str = ""
    trace_input: str = ""
    top_k: Any = None
    snapshot: dict[str, Any] | None = None
    cache_hit: bool = False
