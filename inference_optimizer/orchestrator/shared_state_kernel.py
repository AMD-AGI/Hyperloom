# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-decision write-owner functions extracted from :class:`SharedState`.

Part of the SharedState behavior-offload (phase 2): SharedState is becoming a
passive persisted record, and the methods that *own kernel decisions* (recording
kernel-opt / integrate / gemm-tuning outcomes, kernel-patch identity, pending-
keep bookkeeping) live here as free functions taking the ``state`` as their
first argument. ``SharedState`` keeps thin forwarding shims so existing callers
and the ~54 tests that call ``state.record_kernel_opt(...)`` are unchanged.

These functions were moved verbatim (AST ``self`` -> ``state`` rename); they
read and mutate ``state`` exactly as the original methods did.
"""

from __future__ import annotations

import os
from typing import Any

from .shared_state import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
    _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
    _DEFAULT_KERNEL_OPT_MAX_FAILURES,
    _DEFAULT_KERNEL_OPT_MAX_PARTIAL,
    _now_iso,
)

def _format_last_kernel_opt(state) -> str:
    """Single-line repr of last kernel-opt outcome for prompt injection.

    Returns:
        str: A compact ``kernel_id=... decision=... speedup=...`` line
            (with optional per-kernel attempts/retired history), or
            ``"(none)"`` when no kernel_opt has run.
    """
    if not state.last_kernel_opt:
        return "(none)"
    ko = state.last_kernel_opt
    kid = str(ko.get("kernel_id") or "")
    attempts_entry = state.kernel_opt_attempts.get(kid) or {}
    history_tag = ""
    if attempts_entry:
        history_tag = (
            f" history=attempts={attempts_entry.get('attempts', 0)}"
            f"/partial={attempts_entry.get('partial_count', 0)}"
        )
        rej_reason = attempts_entry.get("rejected_reason")
        if rej_reason:
            history_tag += f"/retired={rej_reason}"
    return (
        f"kernel_id={kid or '?'} "
        f"decision={ko.get('decision','?')} "
        f"speedup={ko.get('micro_speedup','?')}"
        f"{history_tag}"
    )


def _resolve_kernel_patch_identity(
    state, payload: dict[str, Any] | None,
) -> tuple[str, str, str, str]:
    """Resolve a kernel patch's identity tuple from a result/intent payload.

    Pulls ``kernel_id`` / patch path / target file / extra server args
    from the envelope, back-filling the patch path from
    :attr:`last_kernel_opt` when the payload omits it but names a
    matching kernel. The legacy ``extra_sglang_args`` alias is resolved
    through the compat helper.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or LLM
            intent envelope (``None`` treated as empty).

    Returns:
        tuple[str, str, str, str]: ``(kernel_id, patch_path,
            target_file, extra_args)``; any unresolved component is an
            empty string.
    """
    payload = payload or {}
    kernel_id = str(payload.get("kernel_id") or "")
    patch_path = str(
        payload.get("patch_path")
        or payload.get("best_artifact_path")
        or ""
    )
    if (
        not patch_path
        and kernel_id
        and str((state.last_kernel_opt or {}).get("kernel_id") or "") == kernel_id
    ):
        patch_path = str(
            (state.last_kernel_opt or {}).get("best_artifact_path")
            or (state.last_kernel_opt or {}).get("patch_path")
            or ""
        )
    target_file = str(
        payload.get("target_file")
        or payload.get("source_file")
        or ""
    )
    # External envelope; route through compat helper so legacy ``extra_sglang_args`` still resolves.
    from ..compat.payload_aliases import read_extra_server_args
    extra_args = read_extra_server_args(payload).strip()
    return kernel_id, patch_path, target_file, extra_args


def kernel_patch_key(state, payload: dict[str, Any] | None) -> str:
    """Compute the dedup key for a kernel patch.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or intent
            envelope.

    Returns:
        str: ``"<kernel_id>|<patch_path>|<extra_args>"``, or ``""`` when
            either ``kernel_id`` or ``patch_path`` cannot be resolved.
    """
    kernel_id, patch_path, _target_file, extra_args = (
        _resolve_kernel_patch_identity(state, payload)
    )
    if not kernel_id or not patch_path:
        return ""
    return "|".join([kernel_id, patch_path, extra_args])


def find_rejected_kernel_patch(
    state,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Look up a previously-rejected patch matching ``payload``.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or intent
            envelope identifying the patch.

    Returns:
        dict[str, Any] | None: The matching rejected-patch entry, or
            ``None`` when the key is unresolvable or not on record.
    """
    key = kernel_patch_key(state, payload)
    if not key:
        return None
    for entry in state.rejected_kernel_patches:
        if isinstance(entry, dict) and entry.get("key") == key:
            return entry
    return None


def record_kernel_integrate_result(
    state,
    result: dict[str, Any],
    *,
    max_attempts: int = 3,
    keep_threshold_pct: float = 1.0,
) -> dict[str, Any] | None:
    """Persist one integrate E2E result and reject exhausted patch attempts.

    Appends the attempt to the per-key ``kernel_integrate_attempts``
    ledger and, on a REVERT decision or once ``max_attempts`` is hit
    without a KEEP, moves the patch into ``rejected_kernel_patches`` and
    records its ``kernel_id`` in ``rejected_kernel_ids``.

    Args:
        result (dict[str, Any]): The integrate E2E result envelope.
        max_attempts (int): Max attempts before rejecting a non-KEEP
            patch (default 3).
        keep_threshold_pct (float): The gain threshold recorded on the
            rejection row for context (default 1.0).

    Returns:
        dict[str, Any] | None: The updated attempts entry (carrying a
            ``rejected`` sub-dict when rejection fired), or ``None`` when
            ``result`` is not a dict or its patch key is unresolvable.
    """
    if not isinstance(result, dict):
        return None
    key = kernel_patch_key(state, result)
    if not key:
        return None
    kernel_id, patch_path, target_file, extra_args = (
        _resolve_kernel_patch_identity(state, result)
    )
    entry = dict(state.kernel_integrate_attempts.get(key) or {})
    attempts = list(entry.get("attempts") or [])
    attempt = {
        "decision": result.get("decision"),
        "status": result.get("status"),
        "new_tput": result.get("new_tput"),
        "gain_pct": result.get("gain_pct"),
        "workspace": result.get("workspace"),
        "report_path": result.get("report_path"),
        "ts": _now_iso(),
    }
    attempts.append(attempt)
    best_gain = max(
        (
            float(a.get("gain_pct"))
            for a in attempts
            if isinstance(a, dict) and isinstance(a.get("gain_pct"), (int, float))
        ),
        default=0.0,
    )
    entry.update({
        "key": key,
        "kernel_id": kernel_id,
        "patch_path": patch_path,
        "target_file": target_file,
        "extra_server_args": extra_args,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "best_gain_pct": best_gain,
        "last_decision": result.get("decision"),
        "last_status": result.get("status"),
        "updated_at": _now_iso(),
    })
    state.kernel_integrate_attempts[key] = entry

    # kernel_journey stage 4 (end-to-end integrate outcome): additive,
    # idempotent per kernel_id, best-effort.
    try:
        from ..breakdown.recorder import instrument
        sdir = getattr(state, "_session_dir", None)
        if sdir and kernel_id:
            _dec = str(result.get("decision") or "").upper()
            instrument.record_kernel_e2e(
                sdir,
                kernel_id=kernel_id,
                integrated=(_dec == "KEEP"),
                e2e_gain_pct=result.get("gain_pct"),
                validated=True if _dec == "KEEP" else None,
                decision=_dec,
                patch_path=patch_path,
                target_file=target_file,
                extra_server_args=extra_args,
            )
    except Exception:  # noqa: BLE001
        pass

    if result.get("decision") == "KEEP":
        return entry

    should_reject = (
        result.get("decision") == "REVERT"
        or len(attempts) >= max_attempts
    )
    if not should_reject:
        return entry

    reason = (
        "revert_decision"
        if result.get("decision") == "REVERT"
        else f"max_e2e_attempts_{max_attempts}_without_keep"
    )
    rejected = {
        "key": key,
        "kernel_id": kernel_id,
        "patch_path": patch_path,
        "target_file": target_file,
        "extra_server_args": extra_args,
        "attempt_count": len(attempts),
        "best_gain_pct": best_gain,
        "keep_threshold_pct": keep_threshold_pct,
        "last_decision": result.get("decision"),
        "reason": reason,
        "ts": _now_iso(),
    }
    state.rejected_kernel_patches = [
        r for r in state.rejected_kernel_patches
        if not (isinstance(r, dict) and r.get("key") == key)
    ]
    state.rejected_kernel_patches.append(rejected)
    if kernel_id and kernel_id not in state.rejected_kernel_ids:
        state.rejected_kernel_ids.append(kernel_id)
    entry["rejected"] = rejected
    state.kernel_integrate_attempts[key] = entry
    return entry


def record_kernel_opt(state, result: dict[str, Any]) -> None:
    """Capture kernel_optimization_handler result for the next Orch turn; empty kernel_id no-op, non-KEEP can't overwrite a pending KEEP, retires kernel_id (r24 guard) after >= max_partial PARTIALs (INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL)."""
    if not isinstance(result, dict):
        return
    # Author-time breakdown capture: record geak/oob invocations (incl.
    # backend + pre-dispatch failures) before the metadata-less early
    # return so no failed attempt becomes invisible in the geak/oob view.
    try:
        from ..breakdown.recorder import instrument
        sdir = getattr(state, "_session_dir", None)
        instrument.record_kernel_invocations(sdir, result)
        # kernel_journey stage 2 (dispatch) + stage 3 (backend attempts):
        # additive, never overlaps the legacy geak/oob view above.
        _kid = str(result.get("kernel_id") or "")
        if sdir and _kid:
            _attempts = result.get("attempts")
            _attempts = _attempts if isinstance(_attempts, list) else []
            _backends = []
            for _a in _attempts:
                if isinstance(_a, dict):
                    _b = str(_a.get("backend") or "").lower()
                    if _b and _b not in _backends:
                        _backends.append(_b)
            if not _backends:
                _sel = result.get("selected_backends") or result.get("backends")
                if isinstance(_sel, list):
                    _backends = [str(b).lower() for b in _sel if b]
            # A backend that failed before dispatching attempts (e.g. geak
            # rejecting an empty/non-reusable kernel shape) still counts as
            # dispatched: the backend was invoked. Mirror the failure-detect
            # used by record_kernel_backend_result so the synthetic FAILED
            # attempt and the dispatch flag stay consistent.
            _status = str(result.get("status") or "").lower()
            _err_class = str(result.get("error_class") or "")
            _decision = str(
                (result.get("proposal") or {}).get("decision") or "").upper()
            _failed_predispatch = (not _attempts) and (
                _status in {"failed", "error", "crashed", "timeout"}
                or (_decision == "REVERT" and bool(_err_class))
            )
            if _failed_predispatch and not _backends:
                _b = str(result.get("backend") or "").lower() or "geak"
                _backends = [_b]
            _dispatched = bool(_attempts) or _failed_predispatch
            instrument.record_kernel_dispatch(
                sdir,
                kernel_id=_kid,
                dispatched=_dispatched,
                backends=_backends,
                skip_reason="" if _dispatched else str(
                    result.get("error_class") or result.get("status") or ""),
                orchestration_commit=str(getattr(state, "code_revision", "") or ""),
            )
            instrument.record_kernel_backend_result(sdir, result)
    except Exception:  # noqa: BLE001
        pass
    kernel_id = str(result.get("kernel_id") or "")
    if not kernel_id:
        # Metadata-less failure: preserve prior streaming-record KEEP.
        return

    verification = result.get("verification") or {}
    proposal = result.get("proposal") or {}
    decision = str(proposal.get("decision", "")).upper()
    micro_speedup = verification.get("micro_speedup", 0.0)
    try:
        micro_float = float(micro_speedup)
    except (TypeError, ValueError):
        micro_float = 0.0
    best_artifact_path = str(verification.get("best_artifact_path", "") or "")
    source_file = str(
        result.get("source_file")
        or (result.get("candidate") or {}).get("source_file")
        or ""
    )
    # Extract test_command from the first attempt that recorded one so
    # after_kernel_opt rocprof can reuse it without re-deriving from scratch.
    test_command = ""
    for _attempt in (result.get("attempts") or []):
        if isinstance(_attempt, dict):
            _tc = str((_attempt.get("backend_paths") or {}).get("test_command") or "").strip()
            if _tc:
                test_command = _tc
                break
    status = str(result.get("status") or "").lower()
    err_class = str(result.get("error_class") or "")
    # Pure infra failure = backend ladder with no verdict; kept distinct from REVERT/PARTIAL so retirement counters don't double-count.
    is_infra_failure = (
        decision == ""
        and (
            status in {"failed", "error", "timeout"}
            or err_class in {
                "subtask_exception",
                "handler_exception",
                "subprocess_timeout",
                "kernel_agent_root_missing",
                "missing_integration_inputs",
            }
        )
    )
    ts = _now_iso()

    entry = dict(state.kernel_opt_attempts.get(kernel_id) or {})
    history = list(entry.get("history") or [])
    history.append({
        "decision": decision, "micro": micro_float,
        "status": status, "ts": ts,
    })
    history = history[-10:]
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    # Per-source attempts so a Python wrapper and its device file don't share a retry quota.
    per_source = dict(entry.get("attempts_per_source") or {})
    src_key = source_file or ""
    per_source[src_key] = int(per_source.get(src_key, 0)) + 1
    entry["attempts_per_source"] = per_source
    if decision == "PARTIAL":
        entry["partial_count"] = int(entry.get("partial_count", 0)) + 1
    elif decision == "KEEP":
        # Success resets streaks so a future regression isn't auto-retired on stale history.
        entry["partial_count"] = 0
        entry["failure_count"] = 0
    if is_infra_failure:
        entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
    entry["last_decision"] = decision
    entry["last_status"] = status
    entry["last_micro_speedup"] = micro_float
    entry["last_artifact_path"] = best_artifact_path
    entry["last_source_file"] = source_file
    entry["last_ts"] = ts
    entry["history"] = history
    if test_command:
        entry["test_command"] = test_command

    # last_kernel_opt overwrite policy: KEEP always wins; non-KEEP writes only when no pending KEEP to protect.
    prev = state.last_kernel_opt or {}
    prev_decision = str(prev.get("decision", "")).upper()
    prev_kid = str(prev.get("kernel_id", ""))
    integrated_ids = _kernel_ids_in_optimization_stack(state)
    prev_pending = (
        prev_decision == "KEEP"
        and bool(prev_kid)
        and prev_kid not in (state.rejected_kernel_ids or [])
        and prev_kid not in integrated_ids
    )
    if decision == "KEEP" or not prev_pending:
        state.last_kernel_opt = {
            "kernel_id": kernel_id,
            "decision": decision,
            "reasons": proposal.get("reasons", []),
            "micro_speedup": micro_float,
            "compile_passed": verification.get("compile_passed"),
            "correctness_passed": verification.get("correctness_passed"),
            "best_artifact_path": best_artifact_path,
            "source_file": source_file,
            "ts": ts,
        }

    max_partial = _DEFAULT_KERNEL_OPT_MAX_PARTIAL
    env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL")
    if env_v:
        try:
            max_partial = max(1, int(env_v))
        except (TypeError, ValueError):
            pass

    # One backend ladder without a KEEP retires the kernel by default; raise threshold for flaky backends.
    max_failures = _DEFAULT_KERNEL_OPT_MAX_FAILURES
    env_f = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES")
    if env_f:
        try:
            max_failures = max(1, int(env_f))
        except (TypeError, ValueError):
            pass

    should_reject = (
        decision == "REVERT"
        or int(entry.get("partial_count", 0)) >= max_partial
        or int(entry.get("failure_count", 0)) >= max_failures
    )
    if should_reject:
        if kernel_id not in state.rejected_kernel_ids:
            state.rejected_kernel_ids.append(kernel_id)
        entry["rejected_reason"] = (
            "revert_decision"
            if decision == "REVERT"
            else (
                f"max_partial_attempts_{max_partial}_without_keep"
                if int(entry.get("partial_count", 0)) >= max_partial
                else f"max_failures_{max_failures}_without_keep"
            )
        )

    state.kernel_opt_attempts[kernel_id] = entry


def record_gemm_tuning(state, result: dict[str, Any]) -> None:
    """Capture the GEAK GEMM tuning result for sequencing and prompts.

    Snapshots the result into ``last_gemm_tuning`` and appends it to the
    capped ``gemm_tuning_attempts`` history. A non-dict result is
    normalized into a failure record.

    Args:
        result (dict[str, Any]): The GEMM tuning result envelope.
    """
    if not isinstance(result, dict):
        result = {"status": "failed", "error": "non-dict gemm tuning result"}
    entry = dict(result)
    entry.setdefault("ts", _now_iso())
    state.last_gemm_tuning = entry
    attempts = list(state.gemm_tuning_attempts or [])
    attempts.append(entry)
    state.gemm_tuning_attempts = attempts[-_DEFAULT_ATTEMPTS_HISTORY:]


def _kernel_ids_in_optimization_stack(state) -> set[str]:
    """kernel_ids already absorbed into optimization_stack as integrate entries.

    Returns:
        set[str]: The set of ``kernel_id`` values that appear on an
            ``integrate`` entry of :attr:`optimization_stack`.
    """
    return {
        str(e.get("kernel_id"))
        for e in (state.optimization_stack or [])
        if isinstance(e, dict)
        and e.get("action") == "integrate"
        and e.get("kernel_id")
    }


def _source_files_in_optimization_stack(state) -> set[str]:
    """source_file paths already touched by an integrate entry; enforces "same source_file, only strongest KEEP integrated" (apply_kernel_patch is a whole-file overwrite)."""
    sources: set[str] = set()
    for e in (state.optimization_stack or []):
        if not isinstance(e, dict) or e.get("action") != "integrate":
            continue
        src = str(e.get("target_file") or e.get("source_file") or "")
        if src:
            sources.add(src)
    return sources


def _kernel_ids_with_integrate_attempts(state) -> set[str]:
    """kernel_ids that already received an E2E integrate verdict."""
    ids: set[str] = set()
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("kernel_id") or "").strip()
        if kid:
            ids.add(kid)
    return ids


def _kernel_trace_impact_pct(state, kernel_id: str) -> float:
    """Return TraceLens gpu_pct for a kernel_id; unknown kernels sort last."""
    kid = str(kernel_id or "").strip()
    if not kid:
        return 0.0
    trace = state.last_trace_analyze or {}
    for row in trace.get("hot_kernels_top15") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("kernel_id") or "").strip() != kid:
            continue
        try:
            return float(row.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def next_pending_keep_kernel_id(state) -> str:
    """Return next KEEP kernel_id awaiting integrate ("" if drained).

    Ordering favors trace impact (``gpu_pct``) over kernel micro speedup:
    E2E validation should test the highest-impact hot kernel first, not
    merely the patch with the largest isolated microbenchmark win.
    """
    pending = pending_keep_kernel_ids(state)
    return pending[0] if pending else ""


def pending_keep_kernel_ids(state) -> list[str]:
    """All KEEP kernel_ids awaiting integrate, sorted impact-first.

    Kernels that already have an integrate attempt (including
    ``NEEDS_REVIEW``) are excluded so a noisy near-threshold result does not
    automatically rerun the same patch up to the historical max-attempt cap.
    Positive ``NEEDS_REVIEW`` rows are handled by stack validation instead.
    """
    integrated_ids = _kernel_ids_in_optimization_stack(state)
    integrated_sources = _source_files_in_optimization_stack(state)
    attempted_ids = _kernel_ids_with_integrate_attempts(state)
    rejected = set(state.rejected_kernel_ids or [])
    # Mirror next_pending_keep_kernel_id same-file guard: only strongest KEEP per source_file is queueable.
    claimed_sources: set[str] = set()
    ranked: list[tuple[float, float, str, str]] = []
    for kid, entry in (state.kernel_opt_attempts or {}).items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("last_decision", "")).upper() != "KEEP":
            continue
        if kid in integrated_ids or kid in attempted_ids or kid in rejected:
            continue
        src = str(entry.get("last_source_file") or "")
        if src and src in integrated_sources:
            continue
        try:
            micro = float(entry.get("last_micro_speedup") or 0.0)
        except (TypeError, ValueError):
            micro = 0.0
        ranked.append((_kernel_trace_impact_pct(state, kid), micro, kid, src))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    result: list[str] = []
    for _impact, _micro, kid, src in ranked:
        if src and src in claimed_sources:
            continue
        if src:
            claimed_sources.add(src)
        result.append(kid)
    return result


def has_keep_pending_integrate(state) -> bool:
    """Whether any KEEP kernel is still awaiting integrate.

    Returns:
        bool: ``True`` when :meth:`next_pending_keep_kernel_id` is
            non-empty.
    """
    return bool(next_pending_keep_kernel_id(state))


def kernel_opt_attempts_count(state) -> int:
    """Number of distinct kernels with recorded kernel_opt attempts.

    Returns:
        int: The size of the ``kernel_opt_attempts`` ledger.
    """
    return len(state.kernel_opt_attempts or {})


def untried_hot_reusable_kernels(
    state,
    *,
    min_gpu_pct: float | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Hot kernels still owing a ``kernel_opt`` attempt (reusable, gpu_pct >= min_gpu_pct, untouched); capped to top_n by gpu_pct, one kernel_id per task_group."""
    info = state.last_trace_analyze or {}
    hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
    task_groups = info.get("task_groups") or []
    if not isinstance(hot, list):
        return []

    if min_gpu_pct is None:
        try:
            min_gpu_pct = float(os.environ.get(
                "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT",
                _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
            ))
        except (TypeError, ValueError):
            min_gpu_pct = _DEFAULT_HOT_KERNEL_MIN_GPU_PCT
    if top_n is None:
        try:
            top_n = int(os.environ.get(
                "HYPERLOOM_KERNEL_OPT_GATE_TOP_N",
                _DEFAULT_HOT_KERNEL_GATE_TOP_N,
            ))
        except (TypeError, ValueError):
            top_n = _DEFAULT_HOT_KERNEL_GATE_TOP_N
    top_n = max(1, int(top_n))

    kid_to_group: dict[str, list[str]] = {}
    for g in task_groups:
        if not isinstance(g, dict):
            continue
        members = [str(m) for m in (g.get("kernel_ids") or []) if m]
        for m in members:
            kid_to_group[m] = members

    integrated_ids = _kernel_ids_in_optimization_stack(state)
    integrated_sources = _source_files_in_optimization_stack(state)
    rejected = set(state.rejected_kernel_ids or [])
    attempts = state.kernel_opt_attempts or {}

    # Sort by gpu_pct desc so dedup picks the strongest member of each task_group.
    rows: list[tuple[float, str, str, list[str]]] = []
    for k in hot:
        if not isinstance(k, dict):
            continue
        if k.get("reusable_native_kernel") is not True:
            continue
        try:
            gpu_pct = float(k.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            gpu_pct = 0.0
        if gpu_pct < min_gpu_pct:
            continue
        kid = str(k.get("kernel_id") or "")
        if not kid:
            continue
        src = str(k.get("source_file") or "")
        members = sorted(kid_to_group.get(kid, [kid]))
        rows.append((gpu_pct, kid, src, members))
    rows.sort(key=lambda x: x[0], reverse=True)

    ranked: list[tuple[float, str, str, list[str]]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for row in rows:
        group_key = tuple(row[3])
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        ranked.append(row)
    ranked = ranked[:top_n]

    untried: list[str] = []
    for _pct, kid, src, members in ranked:
        if members and all(m in rejected for m in members):
            continue
        if any(m in integrated_ids for m in members):
            continue
        if src and src in integrated_sources:
            continue
        if any(int((attempts.get(m) or {}).get("attempts", 0)) > 0
               for m in members):
            continue
        untried.append(kid)
    return untried

