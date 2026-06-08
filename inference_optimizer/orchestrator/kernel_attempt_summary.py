# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Aggregate kernel-optimization attempts into a single forensic report.

Combines two data sources to answer "why did the kernel-agent not
produce an optimized kernel for this session?":

* :attr:`SharedState.kernel_opt_attempts` — per-kernel ledger written
  by :meth:`SharedState.record_kernel_opt` (one entry per kernel_id
  with history / decision / failure_count / rejected_reason).
* ``<sd>/kernel-agent/runs/<session_id>/results/<kernel_id>.json`` —
  per-kernel kernel-agent run result, carries the full backend ladder
  (geak / claude / codex) with per-backend ``status`` /
  ``optimized_path`` so we can show whether any backend produced an
  artifact.

The top-15 candidate list comes from
``last_trace_analyze.kernel_roofline_top15`` (set by the trace_analyze
sub-step of ``roofline``); kernels listed there but absent from
``kernel_opt_attempts`` are classified ``UNATTEMPTED`` with a
sub-reason explaining why kernel-agent did not dispatch them.

All public helpers are pure functions that take ``SharedState`` +
``session_dir`` and return JSON-ready dicts. They never raise on
missing files — the kernel-agent run dir is best-effort.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Category vocabulary
# ---------------------------------------------------------------------------
# Per-kernel outcome bucket. Front-end / downstream consumers should
# treat these as a closed set; new categories require a schema_version
# bump on the summary block.
CATEGORY_INTEGRATED = "INTEGRATED"
CATEGORY_KEEP_PENDING = "KEEP_PENDING"
CATEGORY_ATTEMPTED_REJECTED = "ATTEMPTED_REJECTED"
CATEGORY_IN_FLIGHT = "IN_FLIGHT"
CATEGORY_UNATTEMPTED = "UNATTEMPTED"

#: Sub-reason vocabulary for ``UNATTEMPTED`` kernels. Picked off the
#: top15 entry's geometry (``source_file``, ``reusable_native_kernel``,
#: ``recommended_backends``) so the front-end can filter by why.
UNATTEMPTED_NO_SOURCE = "no_source_file"
UNATTEMPTED_NOT_REUSABLE = "not_reusable_native_kernel"
UNATTEMPTED_NO_BACKEND = "no_recommended_backend"
UNATTEMPTED_BELOW_CUTOFF = "below_priority_cutoff"
UNATTEMPTED_UNKNOWN = "unknown"

#: ``kernel_opt_attempts`` rejection reasons we surface verbatim into
#: ``rejection_breakdown`` totals (anything else falls into ``other``).
KNOWN_REJECTION_REASONS = (
    "revert_decision",
    "max_partial_attempts_without_keep",
    "max_failures_without_keep",
)

#: ``backend_ladder[].error_class`` vocabulary surfaced into
#: ``failure_reason_breakdown`` so root causes (timeout / preprocess /
#: compile / correctness / agent error) are no longer buried in ``other``.
#: Empty string is reserved for succeeded attempts.
ERROR_CLASS_TIMEOUT = "timeout"
ERROR_CLASS_PREPROCESS_FAILED = "preprocess_failed"
ERROR_CLASS_COMPILE_FAILED = "compile_failed"
ERROR_CLASS_CORRECTNESS_FAILED = "correctness_failed"
ERROR_CLASS_AGENT_ERROR = "agent_error"
ERROR_CLASS_UNKNOWN = "unknown"

#: kernel-agent often points ``optimized_path`` at a stdout / stderr
#: dump when the run failed early (preprocess, build, harness). Those
#: paths are NOT real artifacts and must not flip
#: ``produced_artifact=true`` (which would mask ``ladder_all_failed``).
_ARTIFACT_LOG_SUFFIXES = (
    "_stdout.log", "_stderr.log", ".log", ".txt",
)


def _is_real_artifact_path(path: str) -> bool:
    """True only when ``path`` looks like a real kernel artifact.

    Excludes stdout/stderr/log dumps that kernel-agent writes even on
    early-failure paths (e.g. geak preprocess failure still writes
    ``<id>_stdout.log`` and stuffs it into ``optimized_path``).
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
# stdout_tail is wrapped at ~80 cols by the log formatter, so the
# "preprocess ... (success=False, errors=N)" signal often straddles a
# newline. Allow up to 300 chars of any-whitespace between the keyword
# and the failure marker, and let ``errors=`` sit on a wrapped line too.
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

    Deterministic, never LLM. Priority order: timeout → preprocess →
    compile → correctness → generic agent_error → unknown. ``succeeded``
    attempts get ``("", "")``.
    """
    status = str(attempt.get("status") or "").strip().lower()
    if status == "succeeded":
        return "", ""
    stdout = str(attempt.get("stdout_tail") or "")
    explicit_err = str(attempt.get("error_message") or "")

    # timeout — explicit oob CLI signal; check stdout AND error_message
    for blob in (explicit_err, stdout):
        m = _RE_TIMEOUT.search(blob)
        if m:
            secs = m.group(1)
            return ERROR_CLASS_TIMEOUT, f"Timed out after {secs}s"

    # preprocess failure (GEAK pattern)
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


# ---------------------------------------------------------------------------
# Glossary block (kept inline so the report self-documents)
# ---------------------------------------------------------------------------
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
    "bound_type": (
        "Whether the kernel is limited by memory bandwidth "
        "(memory-bound) or compute (compute-bound)."
    ),
    "compile_passed": (
        "True only if at least one backend in the ladder produced a "
        "usable patch. False means the whole geak->claude->codex "
        "ladder failed to produce any compiled artifact."
    ),
    "backend_ladder": (
        "Per-backend outcome of the kernel-agent dispatch. "
        "``produced_artifact=false`` across all rows is the dominant "
        "signal that the entire ladder failed for this kernel."
    ),
}


# ---------------------------------------------------------------------------
# kernel-agent results harvesting
# ---------------------------------------------------------------------------
def _backend_results_dir(session_dir: Path, session_id: str) -> Path | None:
    """Return ``<sd>/kernel-agent/runs/<key>/results`` or ``None``.

    Best-effort. ``key`` lookup order mirrors how
    ``kernel_request_handlers`` passes ``--session-id`` to the
    kernel-agent CLI:

      1. ``session_dir.name`` (the timestamp dir, e.g. ``20260529T104050Z``)
         — this is the default `payload.get("session_id") or session_dir.name`.
      2. ``state.session_id`` — the longer
         ``<model>_<ts>_<hash>`` form, used when an explicit override
         is supplied.
      3. Any single subdir under ``kernel-agent/runs/`` — recovery
         path for sessions whose key got migrated.

    Returns the first directory that exists; ``None`` when none match.
    """
    runs_root = Path(session_dir) / "kernel-agent" / "runs"
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
    results_dir: Path | None, kernel_id: str,
) -> tuple[dict[str, Any] | None, str]:
    """Read the raw kernel-agent ``results/<kid>.json`` payload.

    Returns ``(payload_dict_or_None, unavailable_reason)``. Reused by
    both ladder harvesting and verification passthrough so a single
    read covers both downstream needs.
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
    results_dir: Path | None, kernel_id: str,
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
        # produced_artifact: real kernel code, not a stdout/stderr dump.
        # kernel-agent writes ``optimized_path`` even on early-failure
        # paths (e.g. geak preprocess failure -> *_stdout.log).
        produced = _is_real_artifact_path(a.get("optimized_path") or "")
        row: dict[str, Any] = {
            "backend": str(a.get("backend") or ""),
            "status": str(a.get("status") or ""),
            "attempt_id": str(a.get("attempt_id") or ""),
            "produced_artifact": produced,
        }
        # kernel-agent results use ``elapsed_s`` (no -ec). Read both for
        # forward compatibility but the canonical source is elapsed_s.
        elapsed = a.get("elapsed_s")
        if elapsed is None:
            elapsed = a.get("elapsed_sec")
        if isinstance(elapsed, (int, float)):
            row["elapsed_sec"] = float(elapsed)
        # error_class / error_message: deterministic root-cause tag so the
        # frontend doesn't have to scrape stdout_tail to know why a
        # backend failed.
        err_class, err_msg = _classify_attempt_failure(a)
        if err_class:
            row["error_class"] = err_class
        if err_msg:
            row["error_message"] = err_msg
        ladder.append(row)
    return ladder, ""


def _relative_to_session(p: Path, session_dir: Path) -> str:
    """Render ``p`` as a path relative to ``session_dir`` when possible."""
    try:
        return str(p.relative_to(session_dir))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Per-kernel classification
# ---------------------------------------------------------------------------
def _classify_attempted(
    entry: dict[str, Any],
    *,
    integrated_ids: set[str],
    rejected_ids: set[str],
    kernel_id: str,
) -> str:
    """Decide the category for a kernel that has an attempts ledger row."""
    last_decision = str(entry.get("last_decision") or "").upper()
    if kernel_id in integrated_ids:
        return CATEGORY_INTEGRATED
    if kernel_id in rejected_ids:
        return CATEGORY_ATTEMPTED_REJECTED
    if last_decision == "KEEP":
        return CATEGORY_KEEP_PENDING
    return CATEGORY_IN_FLIGHT


def _unattempted_reason(top_entry: dict[str, Any]) -> tuple[str, str]:
    """Pick ``(reason_code, human_detail)`` for an UNATTEMPTED kernel.

    Order matters: ``no_source_file`` first because the kernel-agent
    dispatcher's first gate is the source-file resolve; only kernels
    that pass that gate ever reach the backend-recommendation step.
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
            "Reusable kernel but no recommended backend in the top-15 "
            "row; kernel-agent will not auto-dispatch.",
        )
    return (
        UNATTEMPTED_BELOW_CUTOFF,
        "Eligible candidate that was not dispatched within this "
        "session's budget (e.g. session ended before its turn came up).",
    )


def _summary_one_line(
    *,
    category: str,
    entry: dict[str, Any],
    backend_ladder: list[dict[str, Any]],
    artifact_error: str,
) -> str:
    """One-line natural-language summary, deterministic, never LLM."""
    if category == CATEGORY_INTEGRATED:
        micro = entry.get("last_micro_speedup") or 0.0
        return f"integrated into optimization_stack; micro_speedup={micro:.3f}x"
    if category == CATEGORY_KEEP_PENDING:
        micro = entry.get("last_micro_speedup") or 0.0
        return (
            f"KEEP awaiting integrate; micro_speedup={micro:.3f}x "
            "(pending integrate action)"
        )
    if category == CATEGORY_ATTEMPTED_REJECTED:
        all_failed = (
            bool(backend_ladder)
            and all(
                row.get("status") == "failed" and not row.get("produced_artifact")
                for row in backend_ladder
            )
        )
        if all_failed:
            backends = "/".join(row.get("backend") or "?" for row in backend_ladder)
            return (
                f"kernel-agent ladder ({backends}) all "
                f"{len(backend_ladder)} backends failed to produce a "
                f"usable patch; verification: {artifact_error or 'no usable artifact'}"
            )
        decision = str(entry.get("last_decision") or "").upper() or "rejected"
        return f"{decision}; rejected_reason={entry.get('rejected_reason') or 'n/a'}"
    if category == CATEGORY_IN_FLIGHT:
        attempts = int(entry.get("attempts") or 0)
        return f"in-flight; {attempts} attempt(s) recorded, no terminal decision yet"
    return ""


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_kernel_optimization_summary(
    state: Any,
    session_dir: Path | str,
    *,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Build the full summary block for one session.

    Combines :attr:`SharedState.kernel_opt_attempts` +
    :attr:`SharedState.optimization_stack` +
    :attr:`SharedState.rejected_kernel_ids` +
    ``last_trace_analyze.kernel_roofline_top15`` with the per-kernel
    kernel-agent ``results/<kid>.json`` files. Returns a JSON-ready
    dict suitable for atomic write to
    ``<session_dir>/reports/kernel_optimization_summary.json``.
    """
    sd_path = Path(session_dir)
    session_id = str(getattr(state, "session_id", "") or "")
    results_dir = _backend_results_dir(sd_path, session_id)

    top15: list[dict[str, Any]] = list(
        (getattr(state, "last_trace_analyze", {}) or {}).get(
            "kernel_roofline_top15"
        )
        or []
    )
    top_by_id: dict[str, dict[str, Any]] = {
        str(k.get("kernel_id")): k
        for k in top15
        if isinstance(k, dict) and k.get("kernel_id")
    }

    attempts_map: dict[str, dict[str, Any]] = dict(
        getattr(state, "kernel_opt_attempts", {}) or {}
    )
    rejected_ids: set[str] = set(
        str(x) for x in (getattr(state, "rejected_kernel_ids", []) or [])
    )
    integrated_ids: set[str] = set()
    for entry in getattr(state, "optimization_stack", []) or []:
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("kernel_id") or "")
        if kid and entry.get("action") == "integrate":
            integrated_ids.add(kid)
    last_kernel_opt = dict(getattr(state, "last_kernel_opt", {}) or {})
    keep_pending_kid = ""
    if str(last_kernel_opt.get("decision") or "").upper() == "KEEP":
        cand_kid = str(last_kernel_opt.get("kernel_id") or "")
        if (
            cand_kid
            and cand_kid not in integrated_ids
            and cand_kid not in rejected_ids
        ):
            keep_pending_kid = cand_kid

    by_kernel: list[dict[str, Any]] = []
    rejection_breakdown: dict[str, int] = {r: 0 for r in KNOWN_REJECTION_REASONS}
    rejection_breakdown["other"] = 0
    unattempted_breakdown: dict[str, int] = {
        UNATTEMPTED_NO_SOURCE: 0,
        UNATTEMPTED_NOT_REUSABLE: 0,
        UNATTEMPTED_NO_BACKEND: 0,
        UNATTEMPTED_BELOW_CUTOFF: 0,
        UNATTEMPTED_UNKNOWN: 0,
    }
    counts = {
        "top_candidates": len(top15),
        "attempted": 0,
        "integrated": 0,
        "keep_pending": 0,
        "rejected": 0,
        "in_flight": 0,
        "unattempted": 0,
    }

    # Process every kernel that appears in the top15 first (canonical
    # ordering by gpu_pct desc — top15 is already pre-sorted by
    # tracelens_analysis).
    processed_kids: set[str] = set()
    for top_entry in top15:
        if not isinstance(top_entry, dict):
            continue
        kid = str(top_entry.get("kernel_id") or "")
        if not kid:
            continue
        processed_kids.add(kid)
        attempt = attempts_map.get(kid)
        if attempt is None:
            reason_code, reason_detail = _unattempted_reason(top_entry)
            counts["unattempted"] += 1
            unattempted_breakdown[reason_code] = (
                unattempted_breakdown.get(reason_code, 0) + 1
            )
            by_kernel.append(
                _render_unattempted_row(top_entry, reason_code, reason_detail)
            )
            continue
        counts["attempted"] += 1
        category = _classify_attempted(
            attempt,
            integrated_ids=integrated_ids,
            rejected_ids=rejected_ids,
            kernel_id=kid,
        )
        if category == CATEGORY_INTEGRATED:
            counts["integrated"] += 1
        elif category == CATEGORY_KEEP_PENDING:
            counts["keep_pending"] += 1
        elif category == CATEGORY_ATTEMPTED_REJECTED:
            counts["rejected"] += 1
            rej_reason = str(attempt.get("rejected_reason") or "").strip()
            bucket = rej_reason if rej_reason in KNOWN_REJECTION_REASONS else None
            if bucket is None:
                # max_partial / max_failures encode the threshold in the
                # reason string (e.g. ``max_partial_attempts_2_without_keep``);
                # collapse them onto the canonical key.
                if rej_reason.startswith("max_partial_attempts_"):
                    bucket = "max_partial_attempts_without_keep"
                elif rej_reason.startswith("max_failures_"):
                    bucket = "max_failures_without_keep"
                else:
                    bucket = "other"
            rejection_breakdown[bucket] = rejection_breakdown.get(bucket, 0) + 1
        else:
            counts["in_flight"] += 1
        by_kernel.append(
            _render_attempted_row(
                top_entry, attempt, category,
                results_dir=results_dir, session_dir=sd_path,
                last_kernel_opt=last_kernel_opt if kid == keep_pending_kid else None,
            )
        )

    # Kernels that have an attempts ledger row but are NOT in top15
    # (e.g. top15 was refreshed by a later roofline and the original
    # candidate dropped out). Render them anyway with category derived
    # from the ledger and ``top_entry={}``.
    for kid, attempt in attempts_map.items():
        if kid in processed_kids:
            continue
        counts["attempted"] += 1
        category = _classify_attempted(
            attempt,
            integrated_ids=integrated_ids,
            rejected_ids=rejected_ids,
            kernel_id=kid,
        )
        if category == CATEGORY_INTEGRATED:
            counts["integrated"] += 1
        elif category == CATEGORY_KEEP_PENDING:
            counts["keep_pending"] += 1
        elif category == CATEGORY_ATTEMPTED_REJECTED:
            counts["rejected"] += 1
        else:
            counts["in_flight"] += 1
        by_kernel.append(
            _render_attempted_row(
                {"kernel_id": kid}, attempt, category,
                results_dir=results_dir, session_dir=sd_path,
                last_kernel_opt=None,
            )
        )

    failure_reason_breakdown = _aggregate_failure_reasons(by_kernel)
    top_takeaways = _build_top_takeaways(
        counts=counts,
        by_kernel=by_kernel,
        rejection_breakdown=rejection_breakdown,
        failure_reason_breakdown=failure_reason_breakdown,
    )

    return {
        "schema_version": schema_version,
        "session_id": session_id,
        "model_name": str(getattr(state, "model_name", "") or ""),
        "cumulative_gain_validated_pct": float(
            getattr(state, "cumulative_gain_validated", 0.0) or 0.0
        ),
        "totals": counts,
        "rejection_breakdown": rejection_breakdown,
        "unattempted_reason_breakdown": unattempted_breakdown,
        "failure_reason_breakdown": failure_reason_breakdown,
        "field_glossary": FIELD_GLOSSARY,
        "by_kernel": by_kernel,
        "top_takeaways": top_takeaways,
    }


# ---------------------------------------------------------------------------
# Row renderers
# ---------------------------------------------------------------------------
def _render_unattempted_row(
    top_entry: dict[str, Any],
    reason_code: str,
    reason_detail: str,
) -> dict[str, Any]:
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
    kid = str(top_entry.get("kernel_id") or attempt.get("kernel_id") or "")
    ladder, ladder_unavailable = _load_backend_ladder(results_dir, kid)
    # Pull the raw kernel-agent result once so verification passthrough
    # below doesn't re-read the file.
    kernel_result, _ = _load_kernel_result(results_dir, kid)

    verification: dict[str, Any] = {
        "compile_passed": attempt.get("compile_passed"),
        "correctness_passed": attempt.get("correctness_passed"),
    }
    # Detail-file passthrough: IN_FLIGHT / ATTEMPTED_REJECTED kernels
    # never populate ledger compile/correctness fields, so without this
    # block their verification dict stays {"compile_passed": null,
    # "correctness_passed": null} even though the detail file has the
    # real values. Surfaces the extended audit fields too.
    if isinstance(kernel_result, dict):
        ver_block = kernel_result.get("verification")
        if isinstance(ver_block, dict):
            for key in (
                "compile_passed", "correctness_passed",
                "correctness_source", "micro_speedup",
                "micro_speedup_source", "verification_status",
                "best_artifact_path", "best_backend", "best_attempt_id",
            ):
                v = ver_block.get(key)
                if v is not None:
                    verification[key] = v
    # last_kernel_opt is the KEEP_PENDING handoff and wins over both
    # the ledger and the detail file for the current pending kernel.
    if isinstance(last_kernel_opt, dict) and last_kernel_opt:
        for key in (
            "compile_passed", "correctness_passed", "best_artifact_path",
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
        "source_file": str(
            top_entry.get("source_file") or attempt.get("last_source_file") or ""
        ),
        "gpu_pct": _to_float(top_entry.get("gpu_pct")),
        "efficiency_pct": _to_float(top_entry.get("efficiency_percent")),
        "bound_type": str(top_entry.get("bound_type") or ""),
        "arithmetic_intensity": _to_float(top_entry.get("arithmetic_intensity")),
        "category": category,
        "rejected_reason": str(attempt.get("rejected_reason") or ""),
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


# ---------------------------------------------------------------------------
# Aggregations / takeaways
# ---------------------------------------------------------------------------
#: ``backend_ladder[].error_class`` -> ``failure_reason_breakdown`` bucket.
#: Lets root causes (timeout / preprocess / compile / correctness) land in
#: dedicated buckets instead of the legacy catch-all ``other``.
_ERROR_CLASS_TO_BUCKET = {
    ERROR_CLASS_TIMEOUT:            "timeout",
    ERROR_CLASS_PREPROCESS_FAILED:  "preprocess_failed",
    ERROR_CLASS_COMPILE_FAILED:     "compile_failed",
    ERROR_CLASS_CORRECTNESS_FAILED: "correctness_failed",
    ERROR_CLASS_AGENT_ERROR:        "agent_error",
}


def _aggregate_failure_reasons(by_kernel: list[dict[str, Any]]) -> dict[str, int]:
    """Count high-level failure modes across attempted-rejected kernels.

    Priority: dedicated ``error_class``-derived buckets (timeout /
    preprocess_failed / compile_failed / correctness_failed /
    agent_error) trump the legacy structural buckets so root causes
    don't get buried in ``other``. Falls back to the structural
    classification when no ladder attempt carries an error_class.
    """
    breakdown: dict[str, int] = {
        # Legacy structural buckets (kept for back-compat)
        "ladder_all_failed": 0,
        "ladder_partial_no_artifact": 0,
        "speedup_below_threshold": 0,
        "ladder_unavailable": 0,
        # Root-cause buckets derived from error_class
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

        # 1) error_class wins: pick the most common failure mode across
        #    failed/partial attempts (a kernel that timed out on 2 of 3
        #    backends is fundamentally a timeout problem).
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

        # 2) Structural fallback (legacy paths: pre-error_class data,
        #    correctness checked via verification block, etc.)
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
) -> list[str]:
    """Deterministic 2-4 sentence summary, no LLM."""
    out: list[str] = []
    attempted = counts.get("attempted", 0)
    integrated = counts.get("integrated", 0)
    rejected = counts.get("rejected", 0)
    unattempted = counts.get("unattempted", 0)

    if attempted > 0:
        out.append(
            f"{integrated} of {attempted} attempted kernels reached "
            f"KEEP and integrated; {rejected} were rejected."
        )
    else:
        out.append(
            "No kernels were attempted in this session "
            "(check if kernel_opt was disabled or no candidates qualified)."
        )

    ladder_all = failure_reason_breakdown.get("ladder_all_failed", 0)
    if ladder_all >= 1:
        out.append(
            f"Dominant failure mode: kernel-agent backend ladder "
            f"(geak/claude/codex) failed completely for {ladder_all} "
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
            1 for r in by_kernel
            if r.get("category") == CATEGORY_UNATTEMPTED
            and r.get("unattempted_reason") == UNATTEMPTED_NO_SOURCE
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

    "Missed" = ``ATTEMPTED_REJECTED`` (we tried and failed) OR
    ``UNATTEMPTED`` (we never tried). ``INTEGRATED`` / ``KEEP_PENDING``
    are excluded because they are not "missed".
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
    if v is None:
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_kernel_optimization_summary",
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
    "UNATTEMPTED_BELOW_CUTOFF",
    "UNATTEMPTED_UNKNOWN",
    "FIELD_GLOSSARY",
]
