# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate kernel-optimization attempts into a single forensic report.

Combines the per-kernel ledger (:attr:`SharedState.kernel_opt_task_attempts`)
and the collective campaign history (:attr:`SharedState.collective_attempts`)
with the kernel-agent run results to explain why the kernel-agent did not
produce an optimized kernel. All public helpers are pure functions over
``SharedState`` + ``session_dir`` returning JSON-ready dicts; never raise on
missing files.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.common.coerce import to_float

from ..state.kernel_decision_settings import resolve_hot_kernel_min_gpu_pct

log = logging.getLogger(__name__)


# Per-kernel outcome bucket (closed set).
CATEGORY_INTEGRATED = "INTEGRATED"
CATEGORY_KEEP_PENDING = "KEEP_PENDING"
CATEGORY_ATTEMPTED_REJECTED = "ATTEMPTED_REJECTED"
CATEGORY_IN_FLIGHT = "IN_FLIGHT"
CATEGORY_UNATTEMPTED = "UNATTEMPTED"

#: Closed 4-value terminal kernel-outcome bucket the dashboard reads directly.
#: ``IN_FLIGHT`` (no terminal decision) folds into ``fail``.
OUTCOME_SUCCESS = "success"
OUTCOME_FAIL = "fail"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_SKIP = "skip"

#: Sub-reason vocabulary for ``UNATTEMPTED`` kernels, derived from the top15
#: entry's geometry so the front-end can filter by why.
UNATTEMPTED_NO_SOURCE = "no_source_file"
UNATTEMPTED_NOT_REUSABLE = "not_reusable_native_kernel"
UNATTEMPTED_NO_BACKEND = "no_recommended_backend"
#: The dispatcher's own gate. Named for the skip reason the batch filter logs
#: (``below_min_gpu_pct=<threshold>``) so a report row and a run log describe the
#: same event.
UNATTEMPTED_BELOW_MIN_GPU_PCT = "below_min_gpu_pct"
#: Cleared every gate and still never ran. This is the residual bucket, and it
#: must stay narrow: an 8h run reported five of these while the real cause was a
#: Coordinator that could not emit the kernel REQUEST at all, and the label sent
#: the investigation after a priority cutoff that had never fired.
UNATTEMPTED_NEVER_DISPATCHED = "never_dispatched"
UNATTEMPTED_UNKNOWN = "unknown"

#: ``kernel_opt_task_attempts`` rejection reasons we surface verbatim into
#: ``rejection_breakdown`` totals (anything else falls into ``other``).
KNOWN_REJECTION_REASONS = (
    "revert_decision",
    "max_partial_attempts_without_keep",
    "max_failures_without_keep",
)

#: ``backend_ladder[].error_class`` vocabulary surfaced into
#: ``failure_reason_breakdown``. Empty string is reserved for succeeded attempts.
ERROR_CLASS_TIMEOUT = "timeout"
ERROR_CLASS_PREPROCESS_FAILED = "preprocess_failed"
ERROR_CLASS_COMPILE_FAILED = "compile_failed"
ERROR_CLASS_CORRECTNESS_FAILED = "correctness_failed"
ERROR_CLASS_AGENT_ERROR = "agent_error"
ERROR_CLASS_UNKNOWN = "unknown"

#: On early failure kernel-agent points ``optimized_path`` at a stdout/stderr
#: dump; those must not flip ``produced_artifact=true``.
_ARTIFACT_LOG_SUFFIXES = (
    "_stdout.log",
    "_stderr.log",
    ".log",
    ".txt",
)


def _is_real_artifact_path(path: str) -> bool:
    """True only when ``path`` looks like a real kernel artifact.

    Excludes stdout/stderr/log dumps kernel-agent writes on early-failure
    paths and stuffs into ``optimized_path``.

    Args:
        path: Candidate artifact path string.

    Returns:
        True when the path looks like a real kernel artifact.
    """
    if not path:
        return False
    p = path.strip()
    if not p:
        return False
    low = p.lower()
    if any(low.endswith(suf) for suf in _ARTIFACT_LOG_SUFFIXES):
        return False
    fname = low.rsplit("/", 1)[-1]
    if "_stdout" in fname or "_stderr" in fname:
        return False
    return True


_RE_TIMEOUT = re.compile(r"Timed out after (\d+)s")
# stdout_tail is column-wrapped, so the signal can straddle newlines.
_RE_PREPROCESS_FAILED = re.compile(
    r"preprocess[\s\S]{0,300}?success=False"
    r"(?:[\s\S]{0,80}?errors=(\d+))?",
    re.IGNORECASE,
)
_RE_COMPILE_FAILED = re.compile(
    r"(compile|build).{0,30}(failed|error)|undefined reference",
    re.IGNORECASE,
)
_RE_CORRECTNESS_FAILED = re.compile(
    r"correctness.{0,30}(failed|mismatch)|accuracy mismatch",
    re.IGNORECASE,
)


def _classify_attempt_failure(
    attempt: dict[str, Any],
) -> tuple[str, str]:
    """Classify a failed/partial attempt into ``(error_class, error_message)``.

    Priority: timeout → preprocess → compile → correctness → agent_error →
    unknown. ``succeeded`` attempts get ``("", "")``.

    Args:
        attempt: One attempt record dict.

    Returns:
        An ``(error_class, error_message)`` tuple.
    """
    status = str(attempt.get("status") or "").strip().lower()
    if status == "succeeded":
        return "", ""
    stdout = str(attempt.get("stdout_tail") or "")
    explicit_err = str(attempt.get("error_message") or "")

    for blob in (explicit_err, stdout):
        m = _RE_TIMEOUT.search(blob)
        if m:
            secs = m.group(1)
            return ERROR_CLASS_TIMEOUT, f"Timed out after {secs}s"

    m = _RE_PREPROCESS_FAILED.search(stdout)
    if m:
        errs = m.group(1) or "?"
        return (
            ERROR_CLASS_PREPROCESS_FAILED,
            f"preprocess reported {errs} error(s)",
        )

    if _RE_COMPILE_FAILED.search(stdout):
        return ERROR_CLASS_COMPILE_FAILED, "compilation failed"

    if _RE_CORRECTNESS_FAILED.search(stdout):
        return ERROR_CLASS_CORRECTNESS_FAILED, "correctness check failed"

    rc = attempt.get("returncode")
    if isinstance(rc, int) and rc != 0:
        return ERROR_CLASS_AGENT_ERROR, f"agent exit code {rc}"

    return ERROR_CLASS_UNKNOWN, ""


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
    "lane": (
        "Which optimization lane produced the row. ``collective`` rows come "
        "from the multi-GPU comm lane and carry no roofline geometry, so "
        "efficiency_pct / bound_type / arithmetic_intensity stay null."
    ),
    "collective_op": (
        "The collective primitive that was optimized: ``all_reduce``, ``reduce_scatter`` or ``all_gather``."
    ),
    "world_size": ("Rank count the collective was measured across; the lane requires it to match the run's TP width."),
    "bandwidth": (
        "Per-case ``bytes`` / ``algbw_gbps`` / ``busbw_gbps`` from the final "
        "bench. Latency alone cannot separate a faster transfer from a cheaper "
        "barrier; bus bandwidth against the fabric peak says which one a "
        "kept kernel bought, and which regime still has headroom."
    ),
    "collective_attempt_id": (
        "Stable identity for one collective campaign, used to deduplicate "
        "resumed or salvaged attempts across a session."
    ),
    "salvaged": (
        "True when the validated best was recovered from the campaign sidecar "
        "after the wrapper timed out, rather than returned by a clean exit."
    ),
    "e2e_gain_pct": (
        "End-to-end throughput delta measured by the integrate gate. This, not "
        "micro_speedup, decides whether a collective KEEP is adopted."
    ),
    "speedup_basis": (
        "Whether the row's headline number was validated end to end (``e2e``) "
        "or is a microbenchmark ratio only. A collective's microbenchmark "
        "routinely overstates its end-to-end worth, so treat "
        "``microbenchmark`` rows as unproven."
    ),
}


def _backend_results_dir(session_dir: Path, session_id: str) -> Path | None:
    """Return ``<sd>/kernel-agent/runs/<key>/results`` or ``None``.

    ``key`` lookup order: ``session_dir.name``, then ``state.session_id``,
    then a lone subdir under ``kernel-agent/runs/`` (migrated-key recovery).

    Args:
        session_dir: Session directory root.
        session_id: State session id used as a fallback lookup key.

    Returns:
        The results directory path, or ``None`` when none is found.
    """
    from hyperloom.inference_optimizer.session.session_paths import kernel_agent_runs_root

    runs_root = kernel_agent_runs_root(Path(session_dir))
    if not runs_root.is_dir():
        return None
    for key in (session_dir.name, str(session_id or "").strip()):
        if not key:
            continue
        candidate = runs_root / key / "results"
        if candidate.is_dir():
            return candidate
    subdirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        candidate = subdirs[0] / "results"
        if candidate.is_dir():
            return candidate
    return None


def _load_kernel_result(
    results_dir: Path | None,
    kernel_id: str,
) -> tuple[dict[str, Any] | None, str]:
    """Read the raw kernel-agent ``results/<kid>.json`` payload.

    Returns ``(payload_dict_or_None, unavailable_reason)``; reused by ladder
    harvesting and verification passthrough.

    Args:
        results_dir: Directory holding ``<kid>.json`` result files, or ``None``.
        kernel_id: Kernel id whose result is loaded.

    Returns:
        A ``(payload_or_None, unavailable_reason)`` tuple.
    """
    if results_dir is None:
        return None, "kernel_agent_results_dir_missing"
    fpath = results_dir / f"{kernel_id}.json"
    if not fpath.is_file():
        return None, "kernel_agent_result_file_missing"
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "parse_error"
    if not isinstance(data, dict):
        return None, "parse_error"
    return data, ""


def _load_backend_ladder(
    results_dir: Path | None,
    kernel_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Parse one kernel's kernel-agent ``results/<kid>.json`` attempts.

    Returns ``(ladder, unavailable_reason)``:
    * ``ladder`` is the list of compact per-backend rows (empty when
      unavailable); each row carries ``backend / status / attempt_id /
      produced_artifact / elapsed_sec / error_class / error_message``.
    * ``unavailable_reason`` is empty on success or one of
      ``kernel_agent_results_dir_missing``,
      ``kernel_agent_result_file_missing``, ``parse_error``,
      ``no_attempts_recorded``.

    Args:
        results_dir: Directory holding ``<kid>.json`` result files, or ``None``.
        kernel_id: Kernel id whose ladder is parsed.

    Returns:
        A ``(ladder, unavailable_reason)`` tuple.
    """
    data, reason = _load_kernel_result(results_dir, kernel_id)
    if data is None:
        return [], reason
    raw_attempts = data.get("attempts") or []
    if not isinstance(raw_attempts, list) or not raw_attempts:
        return [], "no_attempts_recorded"
    ladder: list[dict[str, Any]] = []
    for a in raw_attempts:
        if not isinstance(a, dict):
            continue
        produced = _is_real_artifact_path(a.get("optimized_path") or "")
        row: dict[str, Any] = {
            "backend": str(a.get("backend") or ""),
            "status": str(a.get("status") or ""),
            "attempt_id": str(a.get("attempt_id") or ""),
            "produced_artifact": produced,
        }
        # Backend self-skip marker for the outcome classifier.
        if a.get("skipped"):
            row["skipped"] = True
        elapsed = a.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            row["elapsed_sec"] = float(elapsed)
        err_class, err_msg = _classify_attempt_failure(a)
        if err_class:
            row["error_class"] = err_class
        if err_msg:
            row["error_message"] = err_msg
        ladder.append(row)
    return ladder, ""


def _relative_to_session(p: Path, session_dir: Path) -> str:
    """Render ``p`` as a path relative to ``session_dir`` when possible.

    Args:
        p: The path to render.
        session_dir: Session directory to make ``p`` relative to.

    Returns:
        The relative path string, or the absolute string when not nested.
    """
    try:
        return str(p.relative_to(session_dir))
    except ValueError:
        return str(p)


def _rejected_reason_of(entry: dict[str, Any]) -> str:
    """Return the rejection reason recorded on a ledger row (``""`` when none).

    A grouped kernel's rejection is stamped on the task row as
    ``integration_rejected_reason``; a single-kernel rejection uses
    ``rejected_reason``. Both are the same fact and the summary must read either.

    Args:
        entry: The kernel's attempts ledger row.

    Returns:
        The rejection reason, or ``""`` when the row records none.
    """
    return str(entry.get("rejected_reason") or entry.get("integration_rejected_reason") or "").strip()


def _entry_integration_status(entry: dict[str, Any]) -> str:
    """Return the row's terminal integration status, lowercased (``""`` when unset)."""
    return str(entry.get("integration_status") or "").strip().lower()


def _rejection_bucket(reason: str) -> str:
    """Map a rejection reason onto a :data:`KNOWN_REJECTION_REASONS` bucket.

    Threshold-encoding reasons (``max_partial_attempts_3``) collapse onto their
    canonical key; anything unrecognised lands in ``other``.

    Args:
        reason: The recorded rejection reason.

    Returns:
        The breakdown key to increment.
    """
    if reason in KNOWN_REJECTION_REASONS:
        return reason
    if reason.startswith("max_partial_attempts_"):
        return "max_partial_attempts_without_keep"
    if reason.startswith("max_failures_"):
        return "max_failures_without_keep"
    return "other"


def _classify_attempted(
    entry: dict[str, Any],
    *,
    integrated_ids: set[str],
    rejected_ids: set[str],
    kernel_id: str,
) -> str:
    """Decide the category for a kernel that has an attempts ledger row.

    The row's own terminal decision is authoritative alongside the id sets: a
    rejected *group* task deliberately stays out of ``rejected_kernel_ids`` (its
    members can be re-dispatched under another task), so a summary keyed only on
    that set reports a terminally rejected kernel as ``IN_FLIGHT``.

    Args:
        entry: The kernel's attempts ledger row.
        integrated_ids: Kernel ids already integrated.
        rejected_ids: Kernel ids that were rejected.
        kernel_id: The kernel id being classified.

    Returns:
        The outcome category constant.
    """
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
    """Map a kernel's category + backend ladder to a terminal outcome bucket.

    Closed 4-value vocabulary (``success`` / ``fail`` / ``timeout`` / ``skip``),
    derived only from structured signals so the dashboard reads one uniform
    field across backends:

    * ``success`` — kept/integrated (a KEEP reached).
    * ``skip``    — never dispatched (``UNATTEMPTED``) OR every recorded attempt
      self-skipped before real work (``skipped`` marker, e.g. forge bailed on a
      compile-only/unsupported/non-git kernel).
    * ``timeout`` — at least one attempt timed out (``error_class == timeout``)
      and none of the above.
    * ``fail``    — anything else that was attempted (compile/correctness/agent
      errors, no measurable improvement, or a non-terminal ``IN_FLIGHT``).

    Args:
        category: The kernel's outcome category constant.
        backend_ladder: Per-backend attempt rows for the kernel.

    Returns:
        One of ``OUTCOME_SUCCESS`` / ``OUTCOME_SKIP`` / ``OUTCOME_TIMEOUT`` /
        ``OUTCOME_FAIL``.
    """
    if category in (CATEGORY_INTEGRATED, CATEGORY_KEEP_PENDING):
        return OUTCOME_SUCCESS
    if category == CATEGORY_UNATTEMPTED:
        return OUTCOME_SKIP
    ladder = backend_ladder or []
    # Every recorded attempt self-skipped -> skip; a mixed ladder is not.
    if ladder and all(bool(r.get("skipped")) for r in ladder):
        return OUTCOME_SKIP
    if any(str(r.get("error_class") or "") == ERROR_CLASS_TIMEOUT for r in ladder):
        return OUTCOME_TIMEOUT
    return OUTCOME_FAIL


def _session_kernel_opt_outcome(by_kernel: list[dict[str, Any]]) -> str:
    """Roll per-kernel ``outcome_class`` up to one session-level verdict.

    Precedence: any ``success`` -> ``success``; else if every kernel is
    ``skip`` (or there are no kernels) -> ``skip``; else ``timeout`` only when a
    timeout is present and no real ``fail``; otherwise ``fail``.

    Args:
        by_kernel: The per-kernel summary rows (each carrying ``outcome_class``).

    Returns:
        The session-level kernel-optimization outcome bucket.
    """
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


def _unattempted_reason(
    top_entry: dict[str, Any],
    *,
    min_gpu_pct: float,
) -> tuple[str, str]:
    """Pick ``(reason_code, human_detail)`` for an UNATTEMPTED kernel.

    Order mirrors the dispatcher's own gates, cheapest first: source-file
    resolve, patchability, backend recommendation, then the GPU-share threshold.
    Only a kernel that cleared all four lands in the residual bucket, so that
    bucket names the absence of a dispatch rather than inventing a gate.

    Args:
        top_entry: The kernel's top-list/roofline entry.
        min_gpu_pct: The threshold the dispatcher applied, from
            :func:`resolve_hot_kernel_min_gpu_pct`.

    Returns:
        A ``(reason_code, human_detail)`` tuple.
    """
    source = str(top_entry.get("source_file") or "").strip()
    reusable = bool(top_entry.get("reusable_native_kernel"))
    recommended = top_entry.get("recommended_backends") or []
    if not source:
        return (
            UNATTEMPTED_NO_SOURCE,
            "TraceLens could not resolve a rewritable source file "
            "(typically a vendor-library op like aten::mm backed by "
            "Tensile / hipBLASLt / rocBLAS). Switch backend via "
            "sglang flags instead of rewriting the kernel.",
        )
    if not reusable:
        return (
            UNATTEMPTED_NOT_REUSABLE,
            "Source resolved but classify_patchability rejected it "
            "(vendor dispatch wrapper / runtime-generated kernel / "
            "source outside a reusable framework root).",
        )
    if not recommended:
        return (
            UNATTEMPTED_NO_BACKEND,
            "Reusable kernel but no recommended backend in the top-15 row; kernel-agent will not auto-dispatch.",
        )
    try:
        gpu_pct = float(top_entry.get("gpu_pct") or 0.0)
    except (TypeError, ValueError):
        gpu_pct = 0.0
    if gpu_pct < min_gpu_pct:
        return (
            UNATTEMPTED_BELOW_MIN_GPU_PCT,
            f"Eligible but {gpu_pct:.3g}% of GPU time is under the "
            f"{min_gpu_pct:.3g}% dispatch threshold "
            "(HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT), so the batch filter skipped it.",
        )
    return (
        UNATTEMPTED_NEVER_DISPATCHED,
        f"Eligible and at {gpu_pct:.3g}% of GPU time it cleared the "
        f"{min_gpu_pct:.3g}% threshold, yet no attempt was recorded: the "
        "Coordinator never issued the kernel REQUEST, or the session ended "
        "before its turn came up. Check the run log for PolicyGate denials "
        "and for 'batch candidates filtered'.",
    )


def _summary_integrated(
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line summary for an ``INTEGRATED`` kernel."""
    micro = entry.get("last_micro_speedup") or 0.0
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
    """One row of :data:`CATEGORY_DISPATCH`.

    Attributes:
        count_key: The ``totals`` counter this category increments.
        summary: Deterministic ``(entry, backend_ladder, artifact_error) ->
            str`` one-line summary builder for this category.
    """

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
    """Resolve the ``totals`` counter for ``category`` via :data:`CATEGORY_DISPATCH`.

    Unknown or blank categories fall back to the ``in_flight`` counter.

    Args:
        category: The kernel outcome category constant.

    Returns:
        The ``totals`` dict key to increment for this category.
    """
    handling = CATEGORY_DISPATCH.get(category)
    return handling.count_key if handling is not None else "in_flight"


def _summary_one_line(
    *,
    category: str,
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line natural-language summary, deterministic, never LLM.

    Args:
        category: The kernel outcome category.
        entry: The kernel's attempt ledger row.
        backend_ladder: Per-backend attempt rows.
        artifact_error: Verification error detail, when any.

    Returns:
        A one-line summary string (``""`` for unknown categories).
    """
    handling = CATEGORY_DISPATCH.get(category)
    if handling is None:
        return ""
    return handling.summary(entry, backend_ladder, artifact_error)


def _stored_collective_attempt_id(record: dict[str, Any]) -> str:
    """Read the identity a campaign was already stamped with (``""`` if absent).

    The phase derives that value; this only reports it.
    """
    return str(record.get("collective_attempt_id") or "").strip()


def _collective_attempt_records(state: Any) -> list[dict[str, Any]]:
    """Return collective campaign history, dropping unusable records.

    The forensics for every dense kernel share this report, so one malformed
    collective row is logged and skipped rather than raised: raising here would
    delete the whole ``kernel_optimization_summary.json`` because both callers
    swallow the exception.
    """
    raw_history = getattr(state, "collective_attempts", None)
    if raw_history is None:
        return []
    if not isinstance(raw_history, list):
        log.warning(
            "kernel summary: ignoring collective_attempts of type %s (expected list)",
            type(raw_history).__name__,
        )
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_history:
        if not isinstance(item, dict):
            log.warning("kernel summary: dropping non-mapping collective campaign")
            continue
        identity = _stored_collective_attempt_id(item)
        if not identity:
            log.warning(
                "kernel summary: dropping collective campaign with no collective_attempt_id",
            )
            continue
        if identity in seen:
            log.warning(
                "kernel summary: dropping duplicate collective campaign %s",
                identity,
            )
            continue
        seen.add(identity)
        records.append(dict(item))
    return records


def _classify_collective_attempt(record: dict[str, Any]) -> str:
    """Map one Collective campaign to the existing summary categories."""
    integration_decision = str(record.get("integration_decision") or "").strip().upper()
    # Fall back to legacy field name for --resume compat.
    integration_status = (
        str(record.get("patch_cleanup_status") or record.get("integration_status") or "").strip().lower()
    )
    run_decision = str(record.get("decision") or "").strip().upper()
    run_status = str(record.get("status") or "").strip().lower()

    if integration_decision == "KEEP":
        if integration_status == "complete":
            return CATEGORY_INTEGRATED
        return CATEGORY_KEEP_PENDING
    if integration_status == "complete":
        return CATEGORY_ATTEMPTED_REJECTED
    if integration_status == "pending" or (record.get("kept") and record.get("requires_e2e_validation")):
        return CATEGORY_KEEP_PENDING
    if run_status in {"failed", "error", "crashed", "timeout"}:
        return CATEGORY_ATTEMPTED_REJECTED
    if run_decision in {"REVERT", "NEEDS_REVIEW"}:
        return CATEGORY_ATTEMPTED_REJECTED
    if run_decision == "KEEP" or record.get("kept"):
        return CATEGORY_KEEP_PENDING
    if run_status in {"ok", "complete", "succeeded"}:
        return CATEGORY_ATTEMPTED_REJECTED
    return CATEGORY_IN_FLIGHT


def _render_collective_attempt_row(
    record: dict[str, Any],
    category: str,
) -> dict[str, Any]:
    """Render one Collective campaign as a kernel-summary row."""
    run_status = str(record.get("status") or "").strip().lower()
    integration_decision = str(record.get("integration_decision") or "").strip().upper()
    final_decision = integration_decision or str(record.get("decision") or "").strip().upper()
    micro_speedup = _to_float(record.get("kernel_speedup")) or 0.0
    e2e_gain_pct = _to_float(record.get("integration_gain_pct"))
    patch_path = str(record.get("patch_path") or record.get("patch") or "")

    raw_error_class = str(record.get("integration_error_class") or record.get("error_class") or "").lower()
    if "timeout" in raw_error_class:
        error_class = ERROR_CLASS_TIMEOUT
    elif "compile" in raw_error_class or "build" in raw_error_class:
        error_class = ERROR_CLASS_COMPILE_FAILED
    elif "correct" in raw_error_class or "mismatch" in raw_error_class:
        error_class = ERROR_CLASS_CORRECTNESS_FAILED
    elif raw_error_class:
        error_class = ERROR_CLASS_AGENT_ERROR
    else:
        error_class = ""

    if run_status in {"ok", "complete", "succeeded"}:
        backend_status = "succeeded"
    elif run_status in {"failed", "error", "crashed", "timeout"}:
        backend_status = "failed"
    else:
        backend_status = run_status or "unknown"
    backend_row: dict[str, Any] = {
        "backend": "forge_collective",
        "status": backend_status,
        "attempt_id": str(record.get("experiment_id") or _stored_collective_attempt_id(record)),
        "produced_artifact": _is_real_artifact_path(patch_path),
    }
    duration_sec = _to_float(record.get("duration_sec"))
    if duration_sec is not None:
        backend_row["elapsed_sec"] = duration_sec
    if error_class:
        backend_row["error_class"] = error_class
    error_message = str(record.get("integration_error") or record.get("error") or "")
    if error_message:
        backend_row["error_message"] = error_message[-1200:]
    backend_ladder = [backend_row]

    if category == CATEGORY_INTEGRATED:
        summary = "collective E2E KEEP integrated"
    elif category == CATEGORY_KEEP_PENDING:
        recovery_action = str(record.get("integration_recovery_action") or "").strip()
        summary = (
            f"collective integration recovery pending: {recovery_action}"
            if recovery_action
            else "collective KEEP awaiting E2E integration"
        )
    elif integration_decision:
        summary = f"collective E2E {integration_decision}"
    elif error_message:
        summary = "collective campaign failed"
    else:
        summary = f"collective {final_decision or 'campaign'} did not integrate"
    # A collective's microbenchmark routinely overstates its end-to-end worth
    # (a 27.8%-of-GPU-time kernel at 1.11x micro landed at +0.39% E2E), so a row
    # that has no E2E number must not read as a measured gain.
    speedup_basis = "e2e" if e2e_gain_pct is not None else "microbenchmark"
    if micro_speedup > 0:
        summary += f"; micro_speedup={micro_speedup:.3f}x"
    if e2e_gain_pct is not None:
        summary += f"; e2e_gain={e2e_gain_pct:.3f}%"
    elif micro_speedup > 0:
        summary += " (microbenchmark only, not E2E validated)"

    rejected_reason = ""
    if category == CATEGORY_ATTEMPTED_REJECTED:
        if integration_decision:
            rejected_reason = f"collective_e2e_{integration_decision.lower()}"
        elif error_message:
            rejected_reason = "collective_run_failed"
        else:
            rejected_reason = "collective_no_keep"

    return {
        "kernel_id": str(record.get("kernel_id") or f"collective:{_stored_collective_attempt_id(record)}"),
        "kernel_name": str(record.get("kernel_name") or ""),
        "kernel_category": "collective",
        "source_file": str(record.get("source_file") or record.get("target_file") or ""),
        "gpu_pct": _to_float(record.get("gpu_pct")),
        "efficiency_pct": None,
        "bound_type": "",
        "arithmetic_intensity": None,
        "recommended_backends": ["forge_collective"],
        "lane": "collective",
        "backend": "forge_collective",
        "engine": str(record.get("engine") or "forge_collective"),
        "category": category,
        "outcome_class": _kernel_outcome_class(category, backend_ladder),
        "rejected_reason": rejected_reason,
        "summary": summary,
        "attempts_total": 1,
        "partial_count": 0,
        "failure_count": int(run_status in {"failed", "error", "crashed", "timeout"} or bool(error_class)),
        "last_decision": final_decision,
        "last_status": str(record.get("integration_result_status") or record.get("status") or ""),
        "last_micro_speedup": micro_speedup,
        "last_ts": str(record.get("integration_ts") or record.get("ts") or ""),
        "verification": {
            "compile_passed": None,
            "correctness_passed": True if record.get("kept") else None,
            "micro_speedup": micro_speedup,
            "e2e_gain_pct": e2e_gain_pct,
            "integration_decision": integration_decision,
        },
        "backend_ladder": backend_ladder,
        "backend_ladder_unavailable_reason": "",
        "kernel_agent_result_path": "",
        "collective_attempt_id": _stored_collective_attempt_id(record),
        "integration_id": str(record.get("integration_id") or ""),
        "experiment_id": str(record.get("experiment_id") or ""),
        "collective_op": str(record.get("collective_op") or ""),
        "world_size": record.get("world_size"),
        "iterations": record.get("iterations"),
        "salvaged": bool(record.get("salvaged")),
        "workspace": str(record.get("integration_workspace") or record.get("workspace") or ""),
        "patch_path": patch_path,
        "e2e_gain_pct": e2e_gain_pct,
        "speedup_basis": speedup_basis,
    }


def build_kernel_optimization_summary(
    state: Any,
    session_dir: Path | str,
    *,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Build the full summary block for one session.

    Combines the kernel ledger / Collective campaign history /
    optimization_stack / rejected ids / top15 with the per-kernel kernel-agent
    ``results/<kid>.json`` files. Returns a JSON-ready dict for atomic write to
    ``<session_dir>/reports/kernel_optimization_summary.json``.

    Args:
        state: The session ``SharedState`` instance.
        session_dir: Session directory (path or string).
        schema_version: Schema version stamped onto the output.

    Returns:
        A JSON-ready summary dict.
    """
    sd_path = Path(session_dir)
    session_id = str(getattr(state, "session_id", "") or "")
    results_dir = _backend_results_dir(sd_path, session_id)

    top15: list[dict[str, Any]] = list(
        (getattr(state, "last_trace_analyze", {}) or {}).get("kernel_roofline_top15") or []
    )

    raw_attempts: dict[str, dict[str, Any]] = dict(getattr(state, "kernel_opt_task_attempts", {}) or {})
    attempts_map: dict[str, dict[str, Any]] = {}
    for ledger_id, attempt in raw_attempts.items():
        if not isinstance(attempt, dict):
            continue
        current_kernel_id = str(attempt.get("current_kernel_id") or attempt.get("kernel_id") or ledger_id)
        previous = attempts_map.get(current_kernel_id)
        if previous is None or str(attempt.get("last_ts") or "") >= str(previous.get("last_ts") or ""):
            attempts_map[current_kernel_id] = attempt
    collective_attempts = [
        record
        for record in _collective_attempt_records(state)
        if str(record.get("status") or "").strip().lower() != "skipped"
        and str(record.get("decision") or "").strip().upper() != "SKIP"
    ]
    collective_kernel_ids = {
        str(record.get("kernel_id") or "").strip()
        for record in collective_attempts
        if str(record.get("kernel_id") or "").strip()
    }
    rejected_ids: set[str] = set(str(x) for x in (getattr(state, "rejected_kernel_ids", []) or []))
    integrated_ids: set[str] = set()
    for entry in getattr(state, "optimization_stack", []) or []:
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("kernel_id") or "")
        if kid and entry.get("action") in {"collective", "integrate"}:
            integrated_ids.add(kid)
    last_kernel_opt = dict(getattr(state, "last_kernel_opt", {}) or {})
    keep_pending_kid = ""
    if str(last_kernel_opt.get("decision") or "").upper() == "KEEP":
        cand_kid = str(last_kernel_opt.get("kernel_id") or "")
        if cand_kid and cand_kid not in integrated_ids and cand_kid not in rejected_ids:
            keep_pending_kid = cand_kid

    by_kernel: list[dict[str, Any]] = []
    rejection_breakdown: dict[str, int] = {r: 0 for r in KNOWN_REJECTION_REASONS}
    rejection_breakdown["other"] = 0
    unattempted_breakdown: dict[str, int] = {
        UNATTEMPTED_NO_SOURCE: 0,
        UNATTEMPTED_NOT_REUSABLE: 0,
        UNATTEMPTED_NO_BACKEND: 0,
        UNATTEMPTED_BELOW_MIN_GPU_PCT: 0,
        UNATTEMPTED_NEVER_DISPATCHED: 0,
        UNATTEMPTED_UNKNOWN: 0,
    }
    min_gpu_pct = resolve_hot_kernel_min_gpu_pct()
    counts = {
        "top_candidates": len(top15),
        "attempted": 0,
        "integrated": 0,
        "keep_pending": 0,
        "rejected": 0,
        "in_flight": 0,
        "unattempted": 0,
    }

    # Process top15 kernels first (pre-sorted by gpu_pct desc).
    processed_kids: set[str] = set()
    for top_entry in top15:
        if not isinstance(top_entry, dict):
            continue
        kid = str(top_entry.get("kernel_id") or "")
        if not kid:
            continue
        processed_kids.add(kid)
        if kid in collective_kernel_ids:
            continue
        attempt = attempts_map.get(kid)
        if attempt is None:
            reason_code, reason_detail = _unattempted_reason(
                top_entry,
                min_gpu_pct=min_gpu_pct,
            )
            counts["unattempted"] += 1
            unattempted_breakdown[reason_code] = unattempted_breakdown.get(reason_code, 0) + 1
            by_kernel.append(_render_unattempted_row(top_entry, reason_code, reason_detail))
            continue
        counts["attempted"] += 1
        category = _classify_attempted(
            attempt,
            integrated_ids=integrated_ids,
            rejected_ids=rejected_ids,
            kernel_id=kid,
        )
        counts[_category_count_key(category)] += 1
        if category == CATEGORY_ATTEMPTED_REJECTED:
            bucket = _rejection_bucket(_rejected_reason_of(attempt))
            rejection_breakdown[bucket] = rejection_breakdown.get(bucket, 0) + 1
        by_kernel.append(
            _render_attempted_row(
                top_entry,
                attempt,
                category,
                results_dir=results_dir,
                session_dir=sd_path,
                last_kernel_opt=last_kernel_opt if kid == keep_pending_kid else None,
            )
        )

    # Kernels with a ledger row but not in top15.
    for kid, attempt in attempts_map.items():
        if kid in processed_kids or kid in collective_kernel_ids:
            continue
        counts["attempted"] += 1
        category = _classify_attempted(
            attempt,
            integrated_ids=integrated_ids,
            rejected_ids=rejected_ids,
            kernel_id=kid,
        )
        counts[_category_count_key(category)] += 1
        # Same accounting as the top15 loop above: a rejected kernel that only
        # has a ledger row must land in the breakdown too, or the totals and the
        # per-reason split disagree.
        if category == CATEGORY_ATTEMPTED_REJECTED:
            bucket = _rejection_bucket(_rejected_reason_of(attempt))
            rejection_breakdown[bucket] = rejection_breakdown.get(bucket, 0) + 1
        by_kernel.append(
            _render_attempted_row(
                {"kernel_id": kid},
                attempt,
                category,
                results_dir=results_dir,
                session_dir=sd_path,
                last_kernel_opt=None,
            )
        )

    for collective_attempt in collective_attempts:
        counts["attempted"] += 1
        category = _classify_collective_attempt(collective_attempt)
        counts[_category_count_key(category)] += 1
        if category == CATEGORY_ATTEMPTED_REJECTED:
            rejection_breakdown["other"] += 1
        by_kernel.append(_render_collective_attempt_row(collective_attempt, category))

    failure_reason_breakdown = _aggregate_failure_reasons(by_kernel)
    dispatch_skip = dict(getattr(state, "last_kernel_opt_dispatch_skip", {}) or {})
    top_takeaways = _build_top_takeaways(
        counts=counts,
        by_kernel=by_kernel,
        rejection_breakdown=rejection_breakdown,
        failure_reason_breakdown=failure_reason_breakdown,
        dispatch_skip=dispatch_skip,
    )

    return {
        "schema_version": schema_version,
        "session_id": session_id,
        "model_name": str(getattr(state, "model_name", "") or ""),
        "cumulative_gain_validated_pct": float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0),
        "kernel_opt_outcome": _session_kernel_opt_outcome(by_kernel),
        "totals": counts,
        "rejection_breakdown": rejection_breakdown,
        "unattempted_reason_breakdown": unattempted_breakdown,
        "failure_reason_breakdown": failure_reason_breakdown,
        # Non-failure breadcrumb for a wholesale dispatch skip; {} otherwise.
        "dispatch_skip_reason": dispatch_skip,
        "field_glossary": FIELD_GLOSSARY,
        "by_kernel": by_kernel,
        "top_takeaways": top_takeaways,
    }


def _render_unattempted_row(
    top_entry: dict[str, Any],
    reason_code: str,
    reason_detail: str,
) -> dict[str, Any]:
    """Build a summary row for a kernel that was not attempted.

    Args:
        top_entry: The kernel's roofline/top-list entry.
        reason_code: Machine-readable reason the kernel was skipped.
        reason_detail: Human-readable explanation of the skip.

    Returns:
        A row dict tagged with the unattempted category and reason.
    """
    return {
        "kernel_id": str(top_entry.get("kernel_id") or ""),
        "kernel_name": str(top_entry.get("name") or ""),
        "kernel_category": str(top_entry.get("kernel_category") or ""),
        "source_file": str(top_entry.get("source_file") or ""),
        "gpu_pct": _to_float(top_entry.get("gpu_pct")),
        "efficiency_pct": _to_float(top_entry.get("efficiency_percent")),
        "bound_type": str(top_entry.get("bound_type") or ""),
        "arithmetic_intensity": _to_float(top_entry.get("arithmetic_intensity")),
        "reusable_native_kernel": bool(top_entry.get("reusable_native_kernel")),
        "recommended_backends": list(top_entry.get("recommended_backends") or []),
        "category": CATEGORY_UNATTEMPTED,
        "outcome_class": OUTCOME_SKIP,
        "unattempted_reason": reason_code,
        "unattempted_detail": reason_detail,
        "summary": f"not attempted: {reason_code}",
    }


def _render_attempted_row(
    top_entry: dict[str, Any],
    attempt: dict[str, Any],
    category: str,
    *,
    results_dir: Path | None,
    session_dir: Path,
    last_kernel_opt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a summary row for a kernel that was attempted.

    Loads the backend ladder and kernel result, then assembles a row
    capturing the attempt outcome and verification details.

    Args:
        top_entry: The kernel's roofline/top-list entry.
        attempt: The recorded attempt metadata.
        category: Outcome category for the row.
        results_dir: Directory holding per-kernel result artifacts.
        session_dir: Session directory for the run.
        last_kernel_opt: Most recent kernel-optimization record, if any.

    Returns:
        A row dict describing the attempt and its results.
    """
    kid = str(top_entry.get("kernel_id") or attempt.get("kernel_id") or "")
    ladder, ladder_unavailable = _load_backend_ladder(results_dir, kid)
    kernel_result, _ = _load_kernel_result(results_dir, kid)

    verification: dict[str, Any] = {
        "compile_passed": attempt.get("compile_passed"),
        "correctness_passed": attempt.get("correctness_passed"),
    }
    # Detail-file passthrough for kernels that don't populate ledger
    # compile/correctness fields.
    if isinstance(kernel_result, dict):
        ver_block = kernel_result.get("verification")
        if isinstance(ver_block, dict):
            for key in (
                "compile_passed",
                "correctness_passed",
                "correctness_source",
                "micro_speedup",
                "micro_speedup_source",
                "verification_status",
                "best_artifact_path",
                "best_backend",
                "best_attempt_id",
            ):
                v = ver_block.get(key)
                if v is not None:
                    verification[key] = v
    # last_kernel_opt wins over ledger + detail file.
    if isinstance(last_kernel_opt, dict) and last_kernel_opt:
        for key in (
            "compile_passed",
            "correctness_passed",
            "best_artifact_path",
            "reasons",
        ):
            v = last_kernel_opt.get(key)
            if v is not None:
                verification[key] = v
    artifact_error = ""
    if verification.get("compile_passed") is False and ladder:
        artifact_error = "no usable backend attempt"
    elif verification.get("compile_passed") is False:
        artifact_error = "ladder unavailable; compile_passed=false"

    summary_text = _summary_one_line(
        category=category,
        entry=attempt,
        backend_ladder=ladder,
        artifact_error=artifact_error,
    )

    row: dict[str, Any] = {
        "kernel_id": kid,
        "kernel_name": str(top_entry.get("name") or ""),
        "kernel_category": str(top_entry.get("kernel_category") or ""),
        "source_file": str(top_entry.get("source_file") or attempt.get("last_source_file") or ""),
        "gpu_pct": _to_float(top_entry.get("gpu_pct")),
        "efficiency_pct": _to_float(top_entry.get("efficiency_percent")),
        "bound_type": str(top_entry.get("bound_type") or ""),
        "arithmetic_intensity": _to_float(top_entry.get("arithmetic_intensity")),
        "category": category,
        "outcome_class": _kernel_outcome_class(category, ladder),
        "rejected_reason": _rejected_reason_of(attempt),
        "summary": summary_text,
        "attempts_total": int(attempt.get("attempts") or 0),
        "partial_count": int(attempt.get("partial_count") or 0),
        "failure_count": int(attempt.get("failure_count") or 0),
        "last_decision": str(attempt.get("last_decision") or ""),
        "last_status": str(attempt.get("last_status") or ""),
        "last_micro_speedup": _to_float(attempt.get("last_micro_speedup")) or 0.0,
        "last_ts": str(attempt.get("last_ts") or ""),
        "verification": verification,
        "backend_ladder": ladder,
        "backend_ladder_unavailable_reason": ladder_unavailable,
        "kernel_agent_result_path": (
            _relative_to_session(results_dir / f"{kid}.json", session_dir)
            if results_dir is not None and (results_dir / f"{kid}.json").is_file()
            else ""
        ),
    }
    return row


#: ``backend_ladder[].error_class`` -> ``failure_reason_breakdown`` bucket.
_ERROR_CLASS_TO_BUCKET = {
    ERROR_CLASS_TIMEOUT: "timeout",
    ERROR_CLASS_PREPROCESS_FAILED: "preprocess_failed",
    ERROR_CLASS_COMPILE_FAILED: "compile_failed",
    ERROR_CLASS_CORRECTNESS_FAILED: "correctness_failed",
    ERROR_CLASS_AGENT_ERROR: "agent_error",
}


def _aggregate_failure_reasons(by_kernel: list[dict[str, Any]]) -> dict[str, int]:
    """Count high-level failure modes across attempted-rejected kernels.

    Priority: ``error_class``-derived buckets trump legacy structural buckets
    so root causes don't get buried in ``other``; falls back to structural
    classification when no ladder attempt carries an error_class.

    Args:
        by_kernel: The per-kernel summary rows.

    Returns:
        Mapping of failure-mode bucket to count.
    """
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
        "other": 0,
    }
    for row in by_kernel:
        if row.get("category") != CATEGORY_ATTEMPTED_REJECTED:
            continue
        ladder = row.get("backend_ladder") or []
        ladder_unavail = row.get("backend_ladder_unavailable_reason") or ""
        if not ladder:
            breakdown["ladder_unavailable" if ladder_unavail else "other"] += 1
            continue

        # error_class wins: pick the most common failure mode.
        ec_counts: dict[str, int] = {}
        for r in ladder:
            ec = str(r.get("error_class") or "")
            if ec and ec != ERROR_CLASS_UNKNOWN:
                ec_counts[ec] = ec_counts.get(ec, 0) + 1
        if ec_counts:
            top_ec = max(ec_counts.items(), key=lambda kv: kv[1])[0]
            bucket = _ERROR_CLASS_TO_BUCKET.get(top_ec, "other")
            breakdown[bucket] += 1
            continue

        # Structural fallback: classify via produced artifacts and verification.
        any_artifact = any(r.get("produced_artifact") for r in ladder)
        all_failed = all(r.get("status") == "failed" for r in ladder)
        verification = row.get("verification") or {}
        if all_failed and not any_artifact:
            breakdown["ladder_all_failed"] += 1
        elif not any_artifact:
            breakdown["ladder_partial_no_artifact"] += 1
        elif verification.get("correctness_passed") is False:
            breakdown["correctness_failed"] += 1
        elif (row.get("last_micro_speedup") or 0.0) > 0.0:
            breakdown["speedup_below_threshold"] += 1
        else:
            breakdown["other"] += 1
    return breakdown


def _build_top_takeaways(
    *,
    counts: dict[str, int],
    by_kernel: list[dict[str, Any]],
    rejection_breakdown: dict[str, int],
    failure_reason_breakdown: dict[str, int],
    dispatch_skip: dict[str, Any] | None = None,
) -> list[str]:
    """Deterministic 2-4 sentence summary, no LLM.

    Args:
        counts: Per-category totals.
        by_kernel: The per-kernel summary rows.
        rejection_breakdown: Counts of rejection reasons.
        failure_reason_breakdown: Counts of failure modes.
        dispatch_skip: The recorded wholesale dispatch-skip breadcrumb, used to
            name the reason instead of guessing at it.

    Returns:
        A list of takeaway sentences.
    """
    out: list[str] = []
    attempted = counts.get("attempted", 0)
    integrated = counts.get("integrated", 0)
    rejected = counts.get("rejected", 0)
    unattempted = counts.get("unattempted", 0)

    if attempted > 0:
        out.append(
            f"{integrated} of {attempted} attempted kernels reached KEEP and integrated; {rejected} were rejected."
        )
    else:
        # The reason is recorded, so state it. Guessing between "disabled" and
        # "nothing qualified" left the third case -- the candidate table was
        # never produced, so nothing was ever asked -- unrepresented, and a
        # reader with every bucket at zero concluded the workload had no
        # headroom.
        skip_reason = str((dispatch_skip or {}).get("reason") or "")
        if skip_reason:
            out.append(f"No kernels were attempted in this session: kernel_opt was never dispatched ({skip_reason}).")
        else:
            out.append(
                "No kernels were attempted in this session "
                "(check if kernel_opt was disabled or no candidates qualified)."
            )

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

    if unattempted > 0:
        no_src = sum(
            1
            for r in by_kernel
            if r.get("category") == CATEGORY_UNATTEMPTED and r.get("unattempted_reason") == UNATTEMPTED_NO_SOURCE
        )
        if no_src > 0:
            out.append(
                f"{no_src} top candidate(s) were not attempted because "
                "no rewritable source file was resolved (vendor-library "
                "ops); address via backend swap (sglang flags), not "
                "kernel rewriting."
            )
    return out


def _find_highest_impact_missed(
    by_kernel: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the missed kernel with the highest ``gpu_pct``.

    "Missed" = anything not ``INTEGRATED`` / ``KEEP_PENDING`` (i.e.
    ``ATTEMPTED_REJECTED``, ``UNATTEMPTED``, or still ``IN_FLIGHT``).

    Args:
        by_kernel: The per-kernel summary rows.

    Returns:
        The highest-``gpu_pct`` missed row, or ``None`` when none qualify.
    """
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
    """Coerce a value to a 4-decimal float, or ``None`` on failure.

    Wraps :func:`hyperloom.common.coerce.to_float` (rejects bool/None/dirty
    input) and rounds the result to 4 decimals for the forensic report.

    Args:
        v: Arbitrary value to convert.

    Returns:
        The rounded float, or ``None`` if it cannot be parsed.
    """
    parsed = to_float(v)
    return round(parsed, 4) if parsed is not None else None


__all__ = [
    "build_kernel_optimization_summary",
    "OUTCOME_SUCCESS",
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
    "CATEGORY_UNATTEMPTED",
    "UNATTEMPTED_NO_SOURCE",
    "UNATTEMPTED_NOT_REUSABLE",
    "UNATTEMPTED_NO_BACKEND",
    "UNATTEMPTED_BELOW_MIN_GPU_PCT",
    "UNATTEMPTED_NEVER_DISPATCHED",
    "UNATTEMPTED_UNKNOWN",
    "FIELD_GLOSSARY",
]
