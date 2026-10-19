# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate kernel-optimization attempts into a single forensic report."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from kernelforge.knowledge.kb_store.identity.implementation import normalize_operator_name

from hyperloom.common.coerce import to_float

from ._kernel_agent_results import (
    ERROR_CLASS_AGENT_ERROR,
    ERROR_CLASS_COMPILE_FAILED,
    ERROR_CLASS_CORRECTNESS_FAILED,
    ERROR_CLASS_PREPROCESS_FAILED,
    ERROR_CLASS_TIMEOUT,
    ERROR_CLASS_UNKNOWN,
    _backend_results_dir,
    _load_backend_ladder,
    _load_kernel_result,
)
from ._kernel_decisions import _forge_loop_entries_by_operator_in_optimization_stack


log = logging.getLogger(__name__)


# Per-kernel outcome bucket (closed set).
CATEGORY_INTEGRATED = "INTEGRATED"
CATEGORY_KEEP_PENDING = "KEEP_PENDING"
CATEGORY_ATTEMPTED_REJECTED = "ATTEMPTED_REJECTED"
CATEGORY_IN_FLIGHT = "IN_FLIGHT"

#: Closed terminal kernel-outcome bucket the dashboard reads directly.
#: ``IN_FLIGHT`` (no terminal decision) folds into ``fail``.
OUTCOME_SUCCESS = "success"
OUTCOME_UNVALIDATED = "unvalidated"
OUTCOME_FAIL = "fail"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_SKIP = "skip"

#: Catch-all bucket of both breakdowns, for a reason or failure mode their
#: vocabularies do not name.
_BUCKET_OTHER = "other"

#: ``kernel_opt_task_attempts`` rejection reasons we surface verbatim into
#: ``rejection_breakdown`` totals (anything else falls into ``other``).
KNOWN_REJECTION_REASONS = (
    "revert_decision",
    "max_partial_attempts_without_keep",
    "max_failures_without_keep",
    # Retired without a backend ever running, so it must not inflate the reverted count that reads as "optimization
    # was tried and lost".
    "unresolved_source",
)


FIELD_GLOSSARY: dict[str, str] = {
    "gpu_pct": (
        "Share of total GPU time spent in this kernel "
        "(kernel_duration / total_gpu_duration). Higher = more "
        "impactful to optimize."
    ),
    "efficiency_pct": (
        "Achieved throughput as a percentage of the kernel's roofline "
        "peak for its bound_type. Lower = more headroom to gain."
    ),
    "bound_type": ("Whether the kernel is limited by memory bandwidth (memory-bound) or compute (compute-bound)."),
    "compile_passed": (
        "True only if at least one backend in the ladder produced a "
        "usable patch. False means the whole backend ladder "
        "failed to produce any compiled artifact."
    ),
    "backend_ladder": (
        "Per-backend outcome of the kernel-agent dispatch. "
        "``produced_artifact=false`` across all rows is the dominant "
        "signal that the entire ladder failed for this kernel."
    ),
}


def _relative_to_session(p: Path, session_dir: Path) -> str:
    """Render ``p`` as a path relative to ``session_dir`` when possible."""
    try:
        return str(p.relative_to(session_dir))
    except ValueError:
        return str(p)


def _rejected_reason_of(entry: dict[str, Any]) -> str:
    """Return the rejection reason recorded on a ledger row (``\"\"`` when none)."""
    return str(entry.get("rejected_reason") or entry.get("integration_rejected_reason") or "").strip()


def _entry_integration_status(entry: dict[str, Any]) -> str:
    """Return the row's terminal integration status, lowercased (``""`` when unset)."""
    return str(entry.get("integration_status") or "").strip().lower()


def _rejection_bucket(reason: str) -> str:
    """Map a rejection reason onto a :data:`KNOWN_REJECTION_REASONS` bucket."""
    if reason in KNOWN_REJECTION_REASONS:
        return reason
    if reason.startswith("max_partial_attempts_"):
        return "max_partial_attempts_without_keep"
    if reason.startswith("max_failures_"):
        return "max_failures_without_keep"
    return _BUCKET_OTHER


def _synthetic_forge_loop_attempt(stack_entry: dict[str, Any]) -> dict[str, Any]:
    """A ledger-shaped attempt for a kernel-recipe-lane integration that never wrote ``kernel_opt_task_attempts``.

    Only called for a row already known (by operator name) to be integrated, so the fields below describe a KEEP —
    there is no partial/rejected state on this path, since a lane only ever lands a stack entry once it kept a
    patch.
    """
    return {
        "attempts": 1,
        "partial_count": 0,
        "failure_count": 0,
        "last_decision": "KEEP",
        "last_status": "integrated",
        # Absent, not 0.0: this row was never measured at the micro level, only end to end (see
        # _summary_integrated).
        "last_micro_speedup": None,
        "last_source_file": str(stack_entry.get("target_file") or stack_entry.get("source_file") or ""),
        "last_ts": str(stack_entry.get("ts") or ""),
        "rejected_reason": "",
        "compile_passed": True,
        "correctness_passed": stack_entry.get("accuracy") is not None,
        "integration_status": "integrated",
    }


def _synthetic_gemm_tuning_attempt(stack_entry: dict[str, Any]) -> dict[str, Any]:
    """A ledger-shaped attempt for a gemm_tuning KEEP, which never writes ``kernel_opt_task_attempts``.

    gemm_tuning retunes GEMM configs across many shapes through one CSV, not one named kernel, so
    unlike a forge-loop integration it cannot be reconciled against a specific roofline top15 row --
    it is surfaced as its own standalone entry instead of a match (see
    ``_gemm_tuning_entries_in_optimization_stack``).
    """
    return {
        "attempts": 1,
        "partial_count": 0,
        "failure_count": 0,
        "last_decision": "KEEP",
        "last_status": "integrated",
        "last_micro_speedup": None,
        "last_ts": str(stack_entry.get("ts") or ""),
        "rejected_reason": "",
        "compile_passed": True,
        "correctness_passed": None,
        "integration_status": "integrated",
    }


def _gemm_tuning_entries_in_optimization_stack(state: Any) -> list[dict[str, Any]]:
    """``optimization_stack`` entries a gemm_tuning KEEP landed, in stack order."""
    entries = []
    for e in getattr(state, "optimization_stack", []) or []:
        if isinstance(e, dict) and e.get("action") == "gemm_tuning":
            entries.append(e)
    return entries


def _classify_attempted(
    entry: dict[str, Any],
    *,
    integrated_ids: set[str],
    rejected_ids: set[str],
    kernel_id: str,
) -> str:
    """Decide the category for a kernel that has an attempts ledger row."""
    last_decision = str(entry.get("last_decision") or "").upper()
    integration_status = _entry_integration_status(entry)
    if kernel_id in integrated_ids or integration_status == "integrated":
        return CATEGORY_INTEGRATED
    if kernel_id in rejected_ids or integration_status == "rejected" or _rejected_reason_of(entry):
        return CATEGORY_ATTEMPTED_REJECTED
    if last_decision == "KEEP":
        return CATEGORY_KEEP_PENDING
    return CATEGORY_IN_FLIGHT


def _kernel_outcome_class(
    category: str,
    backend_ladder: list[dict[str, Any]],
) -> str:
    """Map a kernel's category + backend ladder to a terminal outcome bucket."""
    if category in (CATEGORY_INTEGRATED, CATEGORY_KEEP_PENDING):
        return OUTCOME_SUCCESS
    ladder = backend_ladder or []
    # Every recorded attempt self-skipped -> skip; a mixed ladder is not.
    if ladder and all(bool(r.get("skipped")) for r in ladder):
        return OUTCOME_SKIP
    if any(str(r.get("error_class") or "") == ERROR_CLASS_TIMEOUT for r in ladder):
        return OUTCOME_TIMEOUT
    return OUTCOME_FAIL


def _session_kernel_opt_outcome(by_kernel: list[dict[str, Any]]) -> str:
    """Roll per-kernel ``outcome_class`` up to one session-level verdict."""
    classes = [str(r.get("outcome_class") or "") for r in by_kernel if r.get("outcome_class")]
    if not classes:
        return OUTCOME_SKIP
    if OUTCOME_SUCCESS in classes:
        return OUTCOME_SUCCESS
    if all(c == OUTCOME_SKIP for c in classes):
        return OUTCOME_SKIP
    if OUTCOME_TIMEOUT in classes and OUTCOME_FAIL not in classes:
        return OUTCOME_TIMEOUT
    return OUTCOME_FAIL


def _lane_totals(
    attempted: int,
    success: int,
    unvalidated: int,
    failed: int,
    *,
    outcome: str = "",
) -> dict[str, Any]:
    """Build one lane's counters and terminal outcome."""
    if not outcome:
        if attempted == 0:
            outcome = OUTCOME_SKIP
        elif success:
            outcome = OUTCOME_SUCCESS
        elif unvalidated:
            outcome = OUTCOME_UNVALIDATED
        else:
            outcome = OUTCOME_FAIL
    return {
        "attempted": attempted,
        "success": success,
        "unvalidated": unvalidated,
        "failed": failed,
        "outcome": outcome,
    }


def _geak_lane_totals(state: Any) -> dict[str, Any]:
    """Summarize GEAK's E2E result independently of source rewrites."""
    result = getattr(state, "geak_result", {}) or {}
    if not isinstance(result, dict):
        return _lane_totals(0, 0, 0, 0)
    accepted = [
        row
        for key in ("accepted_kernels", "accepted_heads")
        for row in (result.get(key) or [])
        if isinstance(row, (dict, str)) and bool(row)
    ]
    if accepted:
        count = len(accepted)
        return _lane_totals(count, count, 0, 0)
    status = str(result.get("status") or "").strip().lower()
    if not status or status == OUTCOME_SKIP or status == "skipped":
        return _lane_totals(0, 0, 0, 0)
    return _lane_totals(1, 0, 0, 1)


class _LaneCounters(NamedTuple):
    """One lane's attempted/success/unvalidated/failed tallies."""

    attempted: int = 0
    success: int = 0
    unvalidated: int = 0
    failed: int = 0

    def plus(self, other: _LaneCounters) -> _LaneCounters:
        """These counters with ``other``'s added to each field."""
        return _LaneCounters(
            self.attempted + other.attempted,
            self.success + other.success,
            self.unvalidated + other.unvalidated,
            self.failed + other.failed,
        )


def _gemm_tuning_results(state: Any) -> list[Any]:
    """Every recorded gemm_tuning result, falling back to the last one on its own."""
    attempts = list(getattr(state, "gemm_tuning_attempts", []) or [])
    if attempts:
        return attempts
    last = getattr(state, "last_gemm_tuning", {}) or {}
    return [last] if isinstance(last, dict) and last else []


def _is_gemm_tuner_winner(row: dict[str, Any]) -> bool:
    """True when a tuner row kept its config or measured a micro speedup above 1.0x."""
    return row.get("kept") is True or (_to_float(row.get("best_micro_speedup")) or 0.0) > 1.0


def _gemm_tuner_row_counts(result: dict[str, Any]) -> _LaneCounters | None:
    """Counters for a gemm_tuning result reported per tuner row, ``None`` when it has no rows."""
    rows = [row for row in (result.get("tuners_run") or []) if isinstance(row, dict)]
    if not rows:
        return None
    winners = sum(1 for row in rows if _is_gemm_tuner_winner(row))
    losers = len(rows) - winners
    if result.get("requires_e2e_validation"):
        return _LaneCounters(attempted=len(rows), unvalidated=winners, failed=losers)
    return _LaneCounters(attempted=len(rows), success=winners, failed=losers)


def _gemm_campaign_counts(result: dict[str, Any]) -> _LaneCounters:
    """Counters for a gemm_tuning result that reports one campaign-wide decision."""
    status = str(result.get("status") or "").strip().lower()
    if not status or status in (OUTCOME_SKIP, "skipped"):
        return _LaneCounters()
    if str(result.get("decision") or "").upper() != "KEEP":
        return _LaneCounters(attempted=1, failed=1)
    if result.get("requires_e2e_validation"):
        return _LaneCounters(attempted=1, unvalidated=1)
    return _LaneCounters(attempted=1, success=1)


def _gemm_tuning_lane_totals(state: Any) -> dict[str, Any]:
    """Summarize GEMM micro-tuning without treating candidates as E2E wins."""
    totals = _LaneCounters()
    for result in _gemm_tuning_results(state):
        if not isinstance(result, dict):
            continue
        per_row = _gemm_tuner_row_counts(result)
        totals = totals.plus(per_row if per_row is not None else _gemm_campaign_counts(result))
    return _lane_totals(totals.attempted, totals.success, totals.unvalidated, totals.failed)


def _overall_lane_outcome(lanes: dict[str, dict[str, Any]]) -> str:
    """Roll lane outcomes up without calling unvalidated work successful."""
    outcomes = {str(row.get("outcome") or "") for row in lanes.values()}
    for outcome in (OUTCOME_SUCCESS, OUTCOME_UNVALIDATED, OUTCOME_TIMEOUT, OUTCOME_FAIL):
        if outcome in outcomes:
            return outcome
    return OUTCOME_SKIP


def _summary_integrated(
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line summary for an ``INTEGRATED`` kernel.

    ``last_micro_speedup`` is absent (``None``), not ``0.0``, for a synthetic row built from an
    optimization_stack entry a kernel-opt ledger never measured (forge-loop, fusion, gemm_tuning) --
    a real 0.000x would misreport a kernel this session's own stack shows was kept for a positive
    gain.
    """
    micro = entry.get("last_micro_speedup")
    if micro is None:
        return "integrated into optimization_stack; no kernel-level micro speedup recorded"
    return f"integrated into optimization_stack; micro_speedup={micro:.3f}x"


def _summary_keep_pending(
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line summary for a ``KEEP_PENDING`` kernel."""
    micro = entry.get("last_micro_speedup") or 0.0
    return f"KEEP awaiting integrate; micro_speedup={micro:.3f}x (pending integrate action)"


def _summary_attempted_rejected(
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line summary for an ``ATTEMPTED_REJECTED`` kernel."""
    all_failed = bool(backend_ladder) and all(
        row.get("status") == "failed" and not row.get("produced_artifact") for row in backend_ladder
    )
    if all_failed:
        backends = "/".join(row.get("backend") or "?" for row in backend_ladder)
        return (
            f"kernel-agent ladder ({backends}) all "
            f"{len(backend_ladder)} backends failed to produce a "
            f"usable patch; verification: {artifact_error or 'no usable artifact'}"
        )
    decision = str(entry.get("last_decision") or "").upper() or "rejected"
    return f"{decision}; rejected_reason={_rejected_reason_of(entry) or 'n/a'}"


def _summary_in_flight(
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line summary for an ``IN_FLIGHT`` kernel."""
    attempts = int(entry.get("attempts") or 0)
    return f"in-flight; {attempts} attempt(s) recorded, no terminal decision yet"


class _CategoryHandling(NamedTuple):
    """One row of :data:`CATEGORY_DISPATCH`."""

    count_key: str
    summary: Callable[[dict[str, Any], list[dict[str, Any]], str], str]


#: Single source of truth for per-category handling: each entry defines the
#: ``totals`` counter key and the one-line summary builder. A category absent
#: from this table falls back to the ``in_flight`` counter and an empty summary.
CATEGORY_DISPATCH: dict[str, _CategoryHandling] = {
    CATEGORY_INTEGRATED: _CategoryHandling("integrated", _summary_integrated),
    CATEGORY_KEEP_PENDING: _CategoryHandling("keep_pending", _summary_keep_pending),
    CATEGORY_ATTEMPTED_REJECTED: _CategoryHandling("rejected", _summary_attempted_rejected),
    CATEGORY_IN_FLIGHT: _CategoryHandling("in_flight", _summary_in_flight),
}


def _category_count_key(category: str) -> str:
    """Resolve the ``totals`` counter for ``category`` via :data:`CATEGORY_DISPATCH`."""
    handling = CATEGORY_DISPATCH.get(category)
    return handling.count_key if handling is not None else "in_flight"


def _summary_one_line(
    *,
    category: str,
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line natural-language summary, deterministic, never LLM."""
    handling = CATEGORY_DISPATCH.get(category)
    if handling is None:
        return ""
    return handling.summary(entry, backend_ladder, artifact_error)


class _SummaryInputs(NamedTuple):
    """Every per-session lookup the by-kernel row builders read, resolved once."""

    session_dir: Path
    results_dir: Path | None
    attempts_map: dict[str, dict[str, Any]]
    hot_kernel_ids: set[str]
    integrated_ids: set[str]
    rejected_ids: set[str]
    forge_loop_entries_by_operator: dict[str, dict[str, Any]]
    last_kernel_opt: dict[str, Any]
    keep_pending_kernel_id: str


class _AttemptMatch(NamedTuple):
    """The ledger row a hot kernel resolved to, with the category it lands in."""

    attempt: dict[str, Any]
    category: str


class _KernelTally:
    """The by-kernel summary rows of one session and the counters they roll up into."""

    def __init__(self, top_candidates: int) -> None:
        self.rows: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {
            "top_candidates": top_candidates,
            "attempted": 0,
            "integrated": 0,
            "keep_pending": 0,
            "rejected": 0,
            "in_flight": 0,
        }
        self.rejection_breakdown: dict[str, int] = dict.fromkeys(KNOWN_REJECTION_REASONS, 0)
        self.rejection_breakdown[_BUCKET_OTHER] = 0

    def record(self, row: dict[str, Any]) -> None:
        """Keep one attempted kernel's row and fold it into the counters."""
        category = str(row.get("category") or "")
        self.counts["attempted"] += 1
        self.counts[_category_count_key(category)] += 1
        if category == CATEGORY_ATTEMPTED_REJECTED:
            bucket = _rejection_bucket(str(row.get("rejected_reason") or ""))
            self.rejection_breakdown[bucket] = self.rejection_breakdown.get(bucket, 0) + 1
        self.rows.append(row)


def _roofline_top15(state: Any) -> list[dict[str, Any]]:
    """The session's hottest kernels, in the order trace analysis ranked them."""
    analysis = getattr(state, "last_trace_analyze", {}) or {}
    return list(analysis.get("kernel_roofline_top15") or [])


def _hot_kernel_ids(top15: list[dict[str, Any]]) -> set[str]:
    """The kernel ids the roofline top15 names."""
    ids = {str(entry.get("kernel_id") or "") for entry in top15 if isinstance(entry, dict)}
    ids.discard("")
    return ids


def _latest_attempt_per_kernel(state: Any) -> dict[str, dict[str, Any]]:
    """Attempts-ledger rows keyed by the kernel they now describe, newest ``last_ts`` winning."""
    latest: dict[str, dict[str, Any]] = {}
    for ledger_id, attempt in dict(getattr(state, "kernel_opt_task_attempts", {}) or {}).items():
        if not isinstance(attempt, dict):
            continue
        kernel_id = str(attempt.get("current_kernel_id") or attempt.get("kernel_id") or ledger_id)
        previous = latest.get(kernel_id)
        if previous is None or str(attempt.get("last_ts") or "") >= str(previous.get("last_ts") or ""):
            latest[kernel_id] = attempt
    return latest


def _integrated_kernel_ids(state: Any) -> set[str]:
    """Kernel ids an ``integrate`` entry landed on the optimization stack."""
    ids: set[str] = set()
    for entry in getattr(state, "optimization_stack", []) or []:
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("kernel_id") or "")
        if kid and entry.get("action") == "integrate":
            ids.add(kid)
    return ids


def _pending_keep_kernel_id(
    last_kernel_opt: dict[str, Any],
    integrated_ids: set[str],
    rejected_ids: set[str],
) -> str:
    """The kernel whose KEEP has reached neither integrate nor reject (``""`` when there is none)."""
    if str(last_kernel_opt.get("decision") or "").upper() != "KEEP":
        return ""
    kernel_id = str(last_kernel_opt.get("kernel_id") or "")
    if kernel_id and kernel_id not in integrated_ids and kernel_id not in rejected_ids:
        return kernel_id
    return ""


def _collect_summary_inputs(state: Any, session_dir: Path, top15: list[dict[str, Any]]) -> _SummaryInputs:
    """Resolve the session state every by-kernel row is built from."""
    last_kernel_opt = dict(getattr(state, "last_kernel_opt", {}) or {})
    integrated_ids = _integrated_kernel_ids(state)
    rejected_ids = {str(x) for x in (getattr(state, "rejected_kernel_ids", []) or [])}
    return _SummaryInputs(
        session_dir=session_dir,
        results_dir=_backend_results_dir(session_dir, str(getattr(state, "session_id", "") or "")),
        attempts_map=_latest_attempt_per_kernel(state),
        hot_kernel_ids=_hot_kernel_ids(top15),
        integrated_ids=integrated_ids,
        rejected_ids=rejected_ids,
        forge_loop_entries_by_operator=_forge_loop_entries_by_operator_in_optimization_stack(state),
        last_kernel_opt=last_kernel_opt,
        keep_pending_kernel_id=_pending_keep_kernel_id(last_kernel_opt, integrated_ids, rejected_ids),
    )


def _hot_kernel_attempt(top_entry: dict[str, Any], kernel_id: str, inputs: _SummaryInputs) -> _AttemptMatch | None:
    """Resolve a roofline top15 row to its attempt, or ``None`` when no recorded lane touched it.

    A kernel-recipe lane (forge-loop/flydsl/fusion) lands its optimization_stack entry under its own
    long-form recipe id, which never equals a top15 row's synthetic kNNN id and never touches
    kernel_opt_task_attempts — so without the operator-name match below, an integrated kernel
    silently reads as "never attempted" (see _forge_loop_entries_by_operator_in_optimization_stack's
    docstring for why the operator name is the shared identity).
    """
    attempt = inputs.attempts_map.get(kernel_id)
    if attempt is not None:
        category = _classify_attempted(
            attempt,
            integrated_ids=inputs.integrated_ids,
            rejected_ids=inputs.rejected_ids,
            kernel_id=kernel_id,
        )
        return _AttemptMatch(attempt, category)
    operator = normalize_operator_name(str(top_entry.get("name") or ""))
    forge_loop_entry = inputs.forge_loop_entries_by_operator.get(operator)
    if forge_loop_entry is None:
        return None
    return _AttemptMatch(_synthetic_forge_loop_attempt(forge_loop_entry), CATEGORY_INTEGRATED)


def _tally_hot_kernels(top15: list[dict[str, Any]], inputs: _SummaryInputs, tally: _KernelTally) -> None:
    """Record every roofline top15 kernel a recorded lane attempted, hottest first."""
    for top_entry in top15:
        if not isinstance(top_entry, dict):
            continue
        kernel_id = str(top_entry.get("kernel_id") or "")
        if not kernel_id:
            continue
        match = _hot_kernel_attempt(top_entry, kernel_id, inputs)
        if match is None:
            continue
        pending = inputs.last_kernel_opt if kernel_id == inputs.keep_pending_kernel_id else None
        where = ResultsLocation(inputs.results_dir, inputs.session_dir)
        tally.record(_render_attempted_row(top_entry, match.attempt, match.category, where, pending))


def _tally_ledger_only_kernels(inputs: _SummaryInputs, tally: _KernelTally) -> None:
    """Record every attempted kernel the roofline top15 does not name."""
    where = ResultsLocation(inputs.results_dir, inputs.session_dir)
    for kernel_id, attempt in inputs.attempts_map.items():
        if kernel_id in inputs.hot_kernel_ids:
            continue
        category = _classify_attempted(
            attempt,
            integrated_ids=inputs.integrated_ids,
            rejected_ids=inputs.rejected_ids,
            kernel_id=kernel_id,
        )
        tally.record(_render_attempted_row({"kernel_id": kernel_id}, attempt, category, where, None))


def _tally_gemm_tuning_keeps(state: Any, inputs: _SummaryInputs, tally: _KernelTally) -> None:
    """Record each gemm_tuning KEEP as a standalone entry.

    One campaign retunes many GEMM shapes at once rather than one named kernel, so there is no
    roofline top15 row to match it against and it would otherwise read as "never attempted".
    """
    where = ResultsLocation(inputs.results_dir, inputs.session_dir)
    for gemm_entry in _gemm_tuning_entries_in_optimization_stack(state):
        gemm_kid = str(gemm_entry.get("variant_name") or "gemm_tuning")
        top_entry = {"kernel_id": gemm_kid, "name": gemm_kid, "kernel_category": "gemm_tuning"}
        attempt = _synthetic_gemm_tuning_attempt(gemm_entry)
        tally.record(_render_attempted_row(top_entry, attempt, CATEGORY_INTEGRATED, where, None))


def _lane_totals_by_lane(
    state: Any,
    counts: dict[str, int],
    by_kernel: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Counters and terminal outcome for each kernel lane of one session."""
    unvalidated = counts["keep_pending"] + counts["in_flight"]
    source_outcome = (
        OUTCOME_UNVALIDATED
        if counts["attempted"] and not counts["integrated"] and unvalidated
        else _session_kernel_opt_outcome(by_kernel)
    )
    return {
        "source_level": _lane_totals(
            counts["attempted"],
            counts["integrated"],
            unvalidated,
            counts["rejected"],
            outcome=source_outcome,
        ),
        "geak": _geak_lane_totals(state),
        "gemm_tuning": _gemm_tuning_lane_totals(state),
    }


def _absent_geak_backends_takeaway(state: Any) -> str:
    """Takeaway naming the backends GEAK could not try (``""`` when it had them all)."""
    geak_result = getattr(state, "geak_result", None)
    absent = geak_result.get("absent_backends") if isinstance(geak_result, dict) else None
    if not isinstance(absent, dict) or not absent:
        return ""
    return (
        f"GEAK ran without these backends (unavailable): {', '.join(sorted(absent))}; "
        "any no-win it reports did not try them."
    )


def build_kernel_optimization_summary(state: Any, session_dir: Path | str) -> dict[str, Any]:
    """Build the full summary block for one session."""
    top15 = _roofline_top15(state)
    inputs = _collect_summary_inputs(state, Path(session_dir), top15)
    tally = _KernelTally(len(top15))
    _tally_hot_kernels(top15, inputs, tally)
    _tally_ledger_only_kernels(inputs, tally)
    _tally_gemm_tuning_keeps(state, inputs, tally)

    counts = tally.counts
    failure_reason_breakdown = _aggregate_failure_reasons(tally.rows)
    top_takeaways = _build_top_takeaways(
        counts=counts,
        by_kernel=tally.rows,
        rejection_breakdown=tally.rejection_breakdown,
        failure_reason_breakdown=failure_reason_breakdown,
    )
    lane_totals = _lane_totals_by_lane(state, counts, tally.rows)
    counts["attempted"] = sum(int(lane["attempted"]) for lane in lane_totals.values())
    if counts["attempted"] and lane_totals["source_level"]["attempted"] == 0:
        top_takeaways[0] = "No source-level kernel rewrites were attempted; other kernel lanes did run."
    absent_backends_takeaway = _absent_geak_backends_takeaway(state)
    if absent_backends_takeaway:
        top_takeaways.append(absent_backends_takeaway)

    return {
        "schema_version": 2,
        "session_id": str(getattr(state, "session_id", "") or ""),
        "model_name": str(getattr(state, "model_name", "") or ""),
        "cumulative_gain_validated_pct": float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0),
        "kernel_opt_outcome": _overall_lane_outcome(lane_totals),
        "totals": counts,
        "rejection_breakdown": tally.rejection_breakdown,
        "failure_reason_breakdown": failure_reason_breakdown,
        "field_glossary": FIELD_GLOSSARY,
        "by_kernel": tally.rows,
        "top_takeaways": top_takeaways,
        "lane_totals": lane_totals,
    }


class ResultsLocation(NamedTuple):
    """Where a session's per-kernel result files live and what their paths are relative to."""

    results_dir: Path | None
    session_dir: Path


#: ``results/<kid>.json`` verification fields that override the ledger's own, for kernels
#: that never populate the ledger's compile/correctness fields.
_VERIFICATION_DETAIL_KEYS = (
    "compile_passed",
    "correctness_passed",
    "correctness_source",
    "micro_speedup",
    "micro_speedup_source",
    "verification_status",
    "best_artifact_path",
    "best_backend",
    "best_attempt_id",
)

#: ``last_kernel_opt`` fields that win over both the ledger and the detail file.
_VERIFICATION_DECISION_KEYS = (
    "compile_passed",
    "correctness_passed",
    "best_artifact_path",
    "reasons",
)


def _overlay_present_keys(target: dict[str, Any], source: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Copy each of ``keys`` that ``source`` records as non-``None`` onto ``target``."""
    for key in keys:
        value = source.get(key)
        if value is not None:
            target[key] = value


def _verification_block(
    attempt: dict[str, Any],
    kernel_result: dict[str, Any] | None,
    last_kernel_opt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compile/correctness evidence for one kernel, strongest source applied last."""
    verification: dict[str, Any] = {
        "compile_passed": attempt.get("compile_passed"),
        "correctness_passed": attempt.get("correctness_passed"),
    }
    detail = (kernel_result or {}).get("verification")
    if isinstance(detail, dict):
        _overlay_present_keys(verification, detail, _VERIFICATION_DETAIL_KEYS)
    if last_kernel_opt:
        _overlay_present_keys(verification, last_kernel_opt, _VERIFICATION_DECISION_KEYS)
    return verification


def _artifact_error(verification: dict[str, Any], ladder: list[dict[str, Any]]) -> str:
    """Why this kernel has no usable artifact (``""`` when it compiled)."""
    if verification.get("compile_passed") is not False:
        return ""
    return "no usable backend attempt" if ladder else "ladder unavailable; compile_passed=false"


def _kernel_agent_result_path(where: ResultsLocation, kernel_id: str) -> str:
    """Session-relative path of the kernel-agent result file (``""`` when it was never written)."""
    if where.results_dir is None:
        return ""
    fpath = where.results_dir / f"{kernel_id}.json"
    return _relative_to_session(fpath, where.session_dir) if fpath.is_file() else ""


def _roofline_profile_fields(top_entry: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    """Row fields describing the kernel itself, as trace analysis profiled it."""
    return {
        "kernel_name": str(top_entry.get("name") or ""),
        "kernel_category": str(top_entry.get("kernel_category") or ""),
        "source_file": str(top_entry.get("source_file") or attempt.get("last_source_file") or ""),
        "gpu_pct": _to_float(top_entry.get("gpu_pct")),
        "efficiency_pct": _to_float(top_entry.get("efficiency_percent")),
        "bound_type": str(top_entry.get("bound_type") or ""),
        "arithmetic_intensity": _to_float(top_entry.get("arithmetic_intensity")),
    }


def _ledger_progress_fields(attempt: dict[str, Any]) -> dict[str, Any]:
    """Row fields describing how far the attempts ledger got with the kernel."""
    return {
        "attempts_total": int(attempt.get("attempts") or 0),
        "partial_count": int(attempt.get("partial_count") or 0),
        "failure_count": int(attempt.get("failure_count") or 0),
        "last_decision": str(attempt.get("last_decision") or ""),
        "last_status": str(attempt.get("last_status") or ""),
        # Absent stays absent: a row for a kernel never benchmarked at the micro level (see
        # _synthetic_forge_loop_attempt / _synthetic_gemm_tuning_attempt) must not publish 0.0
        # beside a category of INTEGRATED, which reads as "measured and zero".
        "last_micro_speedup": _to_float(attempt.get("last_micro_speedup")),
        "last_ts": str(attempt.get("last_ts") or ""),
    }


def _render_attempted_row(
    top_entry: dict[str, Any],
    attempt: dict[str, Any],
    category: str,
    where: ResultsLocation,
    last_kernel_opt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a summary row for a kernel that was attempted."""
    kid = str(top_entry.get("kernel_id") or attempt.get("kernel_id") or "")
    ladder, ladder_unavailable = _load_backend_ladder(where.results_dir, kid)
    kernel_result, _ = _load_kernel_result(where.results_dir, kid)
    verification = _verification_block(attempt, kernel_result, last_kernel_opt)
    artifact_error = _artifact_error(verification, ladder)

    return {
        "kernel_id": kid,
        **_roofline_profile_fields(top_entry, attempt),
        "category": category,
        "outcome_class": _kernel_outcome_class(category, ladder),
        "rejected_reason": _rejected_reason_of(attempt),
        "summary": _summary_one_line(
            category=category,
            entry=attempt,
            backend_ladder=ladder,
            artifact_error=artifact_error,
        ),
        **_ledger_progress_fields(attempt),
        "verification": verification,
        "backend_ladder": ladder,
        "backend_ladder_unavailable_reason": ladder_unavailable,
        "kernel_agent_result_path": _kernel_agent_result_path(where, kid),
    }


#: ``backend_ladder[].error_class`` -> ``failure_reason_breakdown`` bucket.
_ERROR_CLASS_TO_BUCKET = {
    ERROR_CLASS_TIMEOUT: "timeout",
    ERROR_CLASS_PREPROCESS_FAILED: "preprocess_failed",
    ERROR_CLASS_COMPILE_FAILED: "compile_failed",
    ERROR_CLASS_CORRECTNESS_FAILED: "correctness_failed",
    ERROR_CLASS_AGENT_ERROR: "agent_error",
}


def _dominant_error_class_bucket(ladder: list[dict[str, Any]]) -> str:
    """The bucket named by the ladder's most common ``error_class`` (``""`` when it records none)."""
    seen: dict[str, int] = {}
    for row in ladder:
        error_class = str(row.get("error_class") or "")
        if error_class and error_class != ERROR_CLASS_UNKNOWN:
            seen[error_class] = seen.get(error_class, 0) + 1
    if not seen:
        return ""
    return _ERROR_CLASS_TO_BUCKET.get(max(seen.items(), key=lambda kv: kv[1])[0], _BUCKET_OTHER)


def _structural_failure_bucket(row: dict[str, Any], ladder: list[dict[str, Any]]) -> str:
    """The bucket implied by the ladder's produced artifacts and the kernel's verification."""
    if not any(r.get("produced_artifact") for r in ladder):
        if all(r.get("status") == "failed" for r in ladder):
            return "ladder_all_failed"
        return "ladder_partial_no_artifact"
    if (row.get("verification") or {}).get("correctness_passed") is False:
        return "correctness_failed"
    if (row.get("last_micro_speedup") or 0.0) > 0.0:
        return "speedup_below_threshold"
    return _BUCKET_OTHER


def _rejected_kernel_failure_bucket(row: dict[str, Any]) -> str:
    """The ``failure_reason_breakdown`` bucket one rejected kernel belongs in."""
    ladder = row.get("backend_ladder") or []
    if not ladder:
        return "ladder_unavailable" if row.get("backend_ladder_unavailable_reason") else _BUCKET_OTHER
    return _dominant_error_class_bucket(ladder) or _structural_failure_bucket(row, ladder)


def _aggregate_failure_reasons(by_kernel: list[dict[str, Any]]) -> dict[str, int]:
    """Count high-level failure modes across attempted-rejected kernels."""
    breakdown: dict[str, int] = {
        # Structural buckets (used when no error_class is available).
        "ladder_all_failed": 0,
        "ladder_partial_no_artifact": 0,
        "speedup_below_threshold": 0,
        "ladder_unavailable": 0,
        # Root-cause buckets (from error_class).
        "timeout": 0,
        "preprocess_failed": 0,
        "compile_failed": 0,
        "correctness_failed": 0,
        "agent_error": 0,
        _BUCKET_OTHER: 0,
    }
    for row in by_kernel:
        if row.get("category") == CATEGORY_ATTEMPTED_REJECTED:
            breakdown[_rejected_kernel_failure_bucket(row)] += 1
    return breakdown


def _build_top_takeaways(
    *,
    counts: dict[str, int],
    by_kernel: list[dict[str, Any]],
    rejection_breakdown: dict[str, int],
    failure_reason_breakdown: dict[str, int],
) -> list[str]:
    """Deterministic 2-4 sentence summary, no LLM."""
    out: list[str] = []
    attempted = counts.get("attempted", 0)
    integrated = counts.get("integrated", 0)
    rejected = counts.get("rejected", 0)

    if attempted > 0:
        out.append(
            f"{integrated} of {attempted} attempted kernels reached KEEP and integrated; {rejected} were rejected."
        )
    else:
        # Says only what this ledger knows.
        out.append("No kernels were attempted through the lanes recorded here.")

    ladder_all = failure_reason_breakdown.get("ladder_all_failed", 0)
    if ladder_all >= 1:
        out.append(
            f"Dominant failure mode: kernel-agent backend ladder "
            f"(geak/forge) failed completely for {ladder_all} "
            "kernel(s) — no backend produced a usable patch. Inspect "
            "kernel-agent toolchain (build env, backend availability)."
        )

    highest_impact = _find_highest_impact_missed(by_kernel)
    if highest_impact is not None:
        gpu = highest_impact.get("gpu_pct") or 0.0
        eff = highest_impact.get("efficiency_pct") or 0.0
        name = highest_impact.get("kernel_name") or highest_impact.get("kernel_id")
        out.append(
            f"Highest-impact missed opportunity: {name} at "
            f"{gpu:.1f}% GPU time, {eff:.1f}% efficiency — "
            "substantial headroom remains."
        )

    return out


def _find_highest_impact_missed(
    by_kernel: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the missed kernel with the highest ``gpu_pct``."""
    best: dict[str, Any] | None = None
    best_gpu = -1.0
    for row in by_kernel:
        if row.get("category") in (CATEGORY_INTEGRATED, CATEGORY_KEEP_PENDING):
            continue
        gpu = row.get("gpu_pct")
        if not isinstance(gpu, (int, float)):
            continue
        if gpu > best_gpu:
            best_gpu = float(gpu)
            best = row
    return best


def _to_float(v: Any) -> float | None:
    """Coerce a value to a 4-decimal float, or ``None`` on failure."""
    parsed = to_float(v)
    return round(parsed, 4) if parsed is not None else None


__all__ = [
    "build_kernel_optimization_summary",
    "OUTCOME_SUCCESS",
    "OUTCOME_UNVALIDATED",
    "OUTCOME_FAIL",
    "OUTCOME_TIMEOUT",
    "OUTCOME_SKIP",
    "CATEGORY_INTEGRATED",
    "CATEGORY_KEEP_PENDING",
    "ERROR_CLASS_TIMEOUT",
    "ERROR_CLASS_PREPROCESS_FAILED",
    "ERROR_CLASS_COMPILE_FAILED",
    "ERROR_CLASS_CORRECTNESS_FAILED",
    "ERROR_CLASS_AGENT_ERROR",
    "ERROR_CLASS_UNKNOWN",
    "CATEGORY_ATTEMPTED_REJECTED",
    "CATEGORY_IN_FLIGHT",
    "FIELD_GLOSSARY",
]
