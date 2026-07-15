# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-decision write-owner functions.

SharedState is a passive persisted record; the functions that *own kernel
decisions* (recording kernel-opt / integrate / gemm-tuning outcomes,
kernel-patch identity, pending-keep bookkeeping, hot-kernel reuse) live here.
They take ``state`` as their first argument and read/mutate it; SharedState
keeps thin forwarding shims so existing callers keep working.

Also carries the "honest E2E" hardening-flag helper (``_honest_flag`` + its
constants), shared by both this cluster and the request handlers that stayed
in the origin module.

Dependencies on retry/default settings come from
``state.kernel_decision_settings`` so this module does not import
``shared_state``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from hyperloom.common.env import env_bool

from ..state.kernel_decision_settings import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
    _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
    _DEFAULT_KERNEL_OPT_MAX_PARTIAL,
    _MAX_INTEGRATE_FAULT_ATTEMPTS,
    _now_iso,
    resolve_kernel_opt_max_failures,
)
from ..trace.trace_env import env_flag


log = logging.getLogger(__name__)

# "Honest E2E" hardening flags. The umbrella flag ``HL_HONEST_E2E`` turns the
# whole mode on; each fix also has a per-fix override that wins over the umbrella
# (set it to an explicit falsey value to opt a single fix out of the umbrella).
_HONEST_E2E_UMBRELLA_ENV = "HL_HONEST_E2E"

def _honest_flag(specific_env: str) -> bool:
    """Resolve a per-fix honest-E2E flag against the umbrella flag.

    Returns ``True`` when the per-fix env ``specific_env`` is truthy, OR when it
    is unset and the umbrella ``HL_HONEST_E2E`` is truthy. An explicit falsey
    per-fix value always wins (lets one fix opt out of the umbrella). The
    umbrella defaults ON. Opt the whole cohort back out with ``HL_HONEST_E2E=0``
    (or a single fix via its per-fix env).

    The per-fix layer uses :func:`trace_env.env_flag` (the canonical superset
    vocabulary that recognizes ``0/false/no/off`` as an *explicit* False and
    falls back to its ``default`` for unset/unrecognized values), so an
    unrecognized per-fix value defers to the umbrella exactly as before. The
    umbrella layer is :func:`common.env.env_bool` (default ON).

    Args:
        specific_env: The per-fix environment variable name.

    Returns:
        bool: Whether the gated behavior should be enabled.
    """
    return env_flag(specific_env, default=env_bool(_HONEST_E2E_UMBRELLA_ENV, True))

# ===========================================================================
# Kernel-decision write-owner functions. SharedState is a passive
# persisted record; the functions that *own kernel decisions* (recording
# kernel-opt / integrate / gemm-tuning outcomes, kernel-patch identity,
# pending-keep bookkeeping, hot-kernel reuse) belong to this kernel domain.
# They take ``state`` as their first argument and read/mutate it; SharedState
# keeps thin forwarding shims so existing callers
# (``state.record_kernel_opt(...)`` etc.) keep working.
#
# Retry/default settings are intentionally below both SharedState and kernel
# decision code to keep the dependency graph one-way.
# ===========================================================================
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
    matching kernel. Extra launch args are read from the canonical
    ``extra_server_args`` field.

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
    extra_args = str(payload.get("extra_server_args") or "").strip()
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
    max_fault_attempts: int | None = None,
) -> dict[str, Any] | None:
    """Persist one integrate E2E result and reject exhausted patch attempts.

    Appends the attempt to the per-key ``kernel_integrate_attempts``
    ledger. Two terminal paths are kept distinct:

    * **Gate verdict** — a genuine REVERT (gain below threshold / accuracy
      regression), or ``max_attempts`` non-fault attempts without a KEEP,
      moves the patch into ``rejected_kernel_patches`` and records its
      ``kernel_id`` in ``rejected_kernel_ids`` (terminal).
    * **Integration fault** — an environment/apply/bench crash (see
      :meth:`SharedState._is_integrate_fault`) that never fairly measured the
      patch. Faults do *not* consume the REVERT quota; they get an independent
      ``max_fault_attempts`` budget and are marked ``retryable`` so the
      pending-integrate driver re-enqueues them, only being rejected once
      that fault budget is exhausted.

    Args:
        result (dict[str, Any]): The integrate E2E result envelope.
        max_attempts (int): Max non-fault attempts before rejecting a
            non-KEEP patch (default 3).
        keep_threshold_pct (float): The gain threshold recorded on the
            rejection row for context (default 1.0).
        max_fault_attempts (int): Independent budget for total integration-
            fault attempts (initial + retries) before they are rejected as
            ``fault_attempts_exhausted`` (default 2 = one retry).

    Returns:
        dict[str, Any] | None: The updated attempts entry (carrying a
            ``rejected`` sub-dict when rejection fired, or
            ``retryable=True`` for an un-exhausted fault), or ``None`` when
            ``result`` is not a dict or its patch key is unresolvable.
    """
    if max_fault_attempts is None:
        max_fault_attempts = _MAX_INTEGRATE_FAULT_ATTEMPTS

    if not isinstance(result, dict):
        return None
    key = kernel_patch_key(state, result)
    if not key:
        return None
    kernel_id, patch_path, target_file, extra_args = (
        _resolve_kernel_patch_identity(state, result)
    )
    is_fault = state._is_integrate_fault(result)
    entry = dict(state.kernel_integrate_attempts.get(key) or {})
    attempts = list(entry.get("attempts") or [])
    attempt = {
        "decision": result.get("decision"),
        "status": result.get("status"),
        "error_class": result.get("error_class"),
        "is_fault": is_fault,
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
    # Quota accounting: faults and gate verdicts draw from separate budgets.
    fault_count = sum(
        1 for a in attempts if isinstance(a, dict) and a.get("is_fault")
    )
    verdict_attempt_count = len(attempts) - fault_count
    entry.update({
        "key": key,
        "kernel_id": kernel_id,
        "patch_path": patch_path,
        "target_file": target_file,
        "extra_server_args": extra_args,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "fault_count": fault_count,
        "verdict_attempt_count": verdict_attempt_count,
        "best_gain_pct": best_gain,
        "last_decision": result.get("decision"),
        "last_status": result.get("status"),
        "last_error_class": result.get("error_class"),
        "last_was_fault": is_fault,
        "updated_at": _now_iso(),
    })
    # Clear any stale retryable flag; re-set below only for un-exhausted faults.
    entry.pop("retryable", None)
    state.kernel_integrate_attempts[key] = entry

    # Record the integrate outcome into the breakdown recorder (idempotent per
    # kernel_id, best-effort).
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument
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

    # Integration fault: never measured fairly. Retry on its own budget instead
    # of burning the REVERT quota, only rejecting once that budget is exhausted.
    if is_fault:
        if fault_count < max_fault_attempts:
            entry["retryable"] = True
            state.kernel_integrate_attempts[key] = entry
            return entry
        reason = f"fault_attempts_exhausted_{max_fault_attempts}"
    else:
        # Gate verdict path: a genuine REVERT, or too many non-fault attempts
        # without a KEEP.
        should_reject = (
            result.get("decision") == "REVERT"
            or verdict_attempt_count >= max_attempts
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
        "fault_count": fault_count,
        "best_gain_pct": best_gain,
        "keep_threshold_pct": keep_threshold_pct,
        "last_decision": result.get("decision"),
        "last_error_class": result.get("error_class"),
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
    """Capture kernel_optimization_handler result for the next Orch turn; empty kernel_id no-op, non-KEEP can't overwrite a pending KEEP, retires kernel_id (r24 guard) after >= max_partial PARTIALs (INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL).

    Args:
        result (dict[str, Any]): The kernel_optimization_handler result
            envelope; non-dicts and empty ``kernel_id`` are no-ops.
    """
    if not isinstance(result, dict):
        return
    # Capture an empty-queue skip (no eligible kernels) as a non-failure
    # breadcrumb so the summary can surface it.
    is_no_eligible_dispatch_skip = (
        str(result.get("status") or "").lower() == "skipped"
        and str(result.get("reason") or "") == "no_eligible_kernels"
    )
    if is_no_eligible_dispatch_skip:
        state.last_kernel_opt_dispatch_skip = {
            "reason": "no_eligible_kernels",
            "kernels_considered": int(result.get("kernels_considered") or 0),
            "message": str(result.get("message") or ""),
            "ts": _now_iso(),
        }
    elif str(result.get("kernel_id") or ""):
        state.last_kernel_opt_dispatch_skip = {}
    # Author-time breakdown capture: record geak/forge invocations before the
    # metadata-less early return so no failed attempt becomes invisible.
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument
        sdir = getattr(state, "_session_dir", None)
        instrument.record_kernel_invocations(sdir, result)
        # Record dispatch and per-backend attempts.
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
            # A backend that failed before dispatching attempts still counts as
            # dispatched. Mirror record_kernel_backend_result's failure-detect
            # so the synthetic FAILED attempt and the dispatch flag stay
            # consistent.
            _status = str(result.get("status") or "").lower()
            _err_class = str(result.get("error_class") or "")
            _decision = str(
                (result.get("proposal") or {}).get("decision") or "").upper()
            _failed_predispatch = (not _attempts) and (
                _status in {"failed", "error", "crashed", "timeout"}
                or (_decision == "REVERT" and bool(_err_class))
            )
            if _failed_predispatch and not _backends:
                # Never default an unattributable failure to GEAK; "unknown"
                # reflects a pre-dispatch gating failure with no backend launched.
                _b = str(result.get("backend") or "").lower() or "unknown"
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
    deploy_snapshot_dir = str(verification.get("deploy_snapshot_dir", "") or "")
    deploy_patch_path = str(verification.get("deploy_patch_path", "") or "")
    deploy_repo_root = str(verification.get("deploy_repo_root", "") or "")
    best_artifact_bundle = dict(verification.get("best_artifact_bundle") or {})
    source_file = str(
        result.get("source_file")
        or (result.get("candidate") or {}).get("source_file")
        or ""
    )
    # Extract test_command from the first attempt that recorded one so
    # after_kernel_opt rocprof can reuse it.
    test_command = ""
    for _attempt in (result.get("attempts") or []):
        if isinstance(_attempt, dict):
            _tc = str((_attempt.get("backend_paths") or {}).get("test_command") or "").strip()
            if _tc:
                test_command = _tc
                break
    status = str(result.get("status") or "").lower()
    err_class = str(result.get("error_class") or "")
    # Pure infra failure = backend ladder with no verdict; kept distinct from
    # REVERT/PARTIAL so retirement counters don't double-count.
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
    # Per-source attempts so a Python wrapper and its device file don't share a
    # retry quota.
    per_source = dict(entry.get("attempts_per_source") or {})
    src_key = source_file or ""
    per_source[src_key] = int(per_source.get(src_key, 0)) + 1
    entry["attempts_per_source"] = per_source
    if decision == "PARTIAL":
        entry["partial_count"] = int(entry.get("partial_count", 0)) + 1
    elif decision == "KEEP":
        # Success resets streaks so a future regression isn't auto-retired on
        # stale history.
        entry["partial_count"] = 0
        entry["failure_count"] = 0
    if is_infra_failure:
        entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
    entry["last_decision"] = decision
    entry["last_status"] = status
    entry["last_micro_speedup"] = micro_float
    entry["last_artifact_path"] = best_artifact_path
    entry["last_artifact_bundle"] = best_artifact_bundle
    entry["last_snapshot_dir"] = deploy_snapshot_dir
    entry["last_deploy_patch_path"] = deploy_patch_path
    entry["last_deploy_repo_root"] = deploy_repo_root
    entry["last_source_file"] = source_file
    # Record backend + correctness so the GEAK-only verified-NEEDS_REVIEW
    # promotion gate can identify a correctness-verified GEAK win (consumed only
    # when HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW is enabled).
    entry["last_backend"] = str(verification.get("best_backend") or "")
    entry["last_correctness_passed"] = verification.get("correctness_passed")
    entry["last_ts"] = ts
    entry["history"] = history
    if test_command:
        entry["test_command"] = test_command

    # last_kernel_opt overwrite policy: KEEP always wins; non-KEEP writes only
    # when there is no pending KEEP to protect.
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
            "best_artifact_bundle": best_artifact_bundle,
            "deploy_snapshot_dir": deploy_snapshot_dir,
            "deploy_patch_path": deploy_patch_path,
            "deploy_repo_root": deploy_repo_root,
            "source_file": source_file,
            "ts": ts,
        }

    max_partial = _DEFAULT_KERNEL_OPT_MAX_PARTIAL
    env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL")
    if env_v:
        try:
            max_partial = max(1, int(env_v))
        except (TypeError, ValueError):
            # Malformed env override -> keep the default partial-attempt cap.
            pass

    # Backend ladder failures are often transient infra faults; real REVERT
    # still retires immediately below, but infra failures get a retry.
    max_failures = resolve_kernel_opt_max_failures()

    # High-impact infra-retry (flag-gated, default off). An infra non-finish
    # (timeout / preprocess / agent-crash; no verdict) means the attempt didn't
    # finish, not that the kernel can't be improved. Give a high-GPU%-share
    # kernel more attempts to COMPLETE rather than retiring it as a REVERT would.
    infra_failure_cap = max_failures
    if _honest_flag("HL_INFRA_RETRY_HIGH_IMPACT"):
        try:
            _impact_pct = float(_kernel_trace_impact_pct(state, kernel_id) or 0.0)
        except Exception:  # noqa: BLE001 - impact is best-effort
            _impact_pct = 0.0
        # Stamp impact so the dispatch-side cap widens from the same record.
        entry["last_gpu_pct"] = _impact_pct
        try:
            _min_gpu = float(os.environ.get("HL_INFRA_RETRY_MIN_GPU_PCT", "5.0") or 5.0)
        except ValueError:
            _min_gpu = 5.0
        try:
            _infra_max = int(os.environ.get("HL_INFRA_RETRY_MAX", "4") or 4)
        except ValueError:
            _infra_max = 4
        if _impact_pct >= _min_gpu:
            infra_failure_cap = max(max_failures, _infra_max)

    should_reject = (
        decision == "REVERT"
        or int(entry.get("partial_count", 0)) >= max_partial
        or int(entry.get("failure_count", 0)) >= infra_failure_cap
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
    """source_file paths already touched by an integrate entry; enforces "same source_file, only strongest KEEP integrated" (apply_kernel_patch is a whole-file overwrite).

    Returns:
        set[str]: The set of ``target_file`` / ``source_file`` paths
            referenced by ``integrate`` entries of
            :attr:`optimization_stack`.
    """
    sources: set[str] = set()
    for e in (state.optimization_stack or []):
        if not isinstance(e, dict) or e.get("action") != "integrate":
            continue
        src = str(e.get("target_file") or e.get("source_file") or "")
        if src:
            sources.add(src)
    return sources

def _kernel_ids_with_integrate_attempts(state) -> set[str]:
    """kernel_ids that already received a *terminal* E2E integrate verdict.

    A kernel_id whose only integrate attempts are un-exhausted integration
    faults (``retryable``) is intentionally excluded so the pending-integrate
    driver re-enqueues it for a fault retry. A kernel_id is treated as
    attempted once *any* of its entries reached a non-retryable terminal
    state (KEEP / real REVERT / fault budget exhausted); a terminal entry on
    one patch key wins over a retryable entry on another.

    Returns:
        set[str]: The kernel_ids with at least one non-retryable terminal
            integrate entry.
    """
    terminal: set[str] = set()
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("kernel_id") or "").strip()
        if not kid:
            continue
        if entry.get("retryable") and not entry.get("rejected"):
            continue
        terminal.add(kid)
    return terminal

def integrate_attempt_count_for_kernel(state, kernel_id: str) -> int:
    """Total *recorded* integrate attempts for a kernel_id.

    Sums ``attempt_count`` across every ``kernel_integrate_attempts`` entry
    sharing this ``kernel_id`` (one kernel may produce more than one patch
    key). The count only advances inside
    :func:`record_kernel_integrate_result`, so it is a reliable in-flight
    signal for the KERNEL-phase auto-integrate driver: an unchanged count
    means a dispatched integrate has not yet been recorded (still in flight),
    an advanced count means it completed.

    Args:
        kernel_id (str): The kernel identifier to total attempts for.

    Returns:
        int: Recorded integrate attempts (0 when unknown/blank).
    """
    kid = str(kernel_id or "").strip()
    if not kid:
        return 0
    total = 0
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("kernel_id") or "").strip() != kid:
            continue
        try:
            total += int(entry.get("attempt_count") or 0)
        except (TypeError, ValueError):
            continue
    return total

def _kernel_trace_impact_pct(state, kernel_id: str) -> float:
    """Return TraceLens gpu_pct for a kernel_id; unknown kernels sort last.

    Args:
        kernel_id (str): The kernel identifier to look up in the latest
            trace-analyze ``hot_kernels_top15``.

    Returns:
        float: The kernel's ``gpu_pct`` impact, or ``0.0`` when blank,
            unknown, or unparseable.
    """
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

    Returns:
        str: The highest-impact pending KEEP ``kernel_id``, or ``""``
            when the queue is drained.
    """
    pending = pending_keep_kernel_ids(state)
    return pending[0] if pending else ""

def pending_keep_kernel_ids(state) -> list[str]:
    """All KEEP kernel_ids awaiting integrate, sorted impact-first.

    Kernels that already have an integrate attempt (including
    ``NEEDS_REVIEW``) are excluded so a noisy near-threshold result does not
    automatically rerun the same patch up to the historical max-attempt cap.
    Positive ``NEEDS_REVIEW`` rows are handled by stack validation instead.

    Returns:
        list[str]: Pending KEEP ``kernel_id`` values sorted impact-first
            (trace ``gpu_pct``, then micro speedup), one per source file.
    """
    integrated_ids = _kernel_ids_in_optimization_stack(state)
    integrated_sources = _source_files_in_optimization_stack(state)
    attempted_ids = _kernel_ids_with_integrate_attempts(state)
    rejected = set(state.rejected_kernel_ids or [])
    # Mirror next_pending_keep_kernel_id same-file guard: only the strongest
    # KEEP per source_file is queueable.
    claimed_sources: set[str] = set()
    ranked: list[tuple[float, float, str, str]] = []
    for kid, entry in (state.kernel_opt_attempts or {}).items():
        if not isinstance(entry, dict):
            continue
        _dec = str(entry.get("last_decision", "")).upper()
        # GEAK-only verified-NEEDS_REVIEW promotion (flag-gated): admit a
        # correctness-verified, high-micro GEAK NEEDS_REVIEW into the integrate
        # queue so its win is E2E-measured. Off => only KEEP queues.
        _promote = _honest_flag("HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW")
        try:
            _thr = float(os.environ.get("HL_VERIFIED_MICRO_PROMOTE_THRESHOLD", "1.10") or 1.10)
        except ValueError:
            _thr = 1.10
        try:
            _m = float(entry.get("last_micro_speedup") or 0.0)
        except (TypeError, ValueError):
            _m = 0.0
        _geak_nr = (
            _promote
            and _dec == "NEEDS_REVIEW"
            and str(entry.get("last_backend") or "").lower() == "geak"
            and entry.get("last_correctness_passed") is True
            and _m >= _thr
        )
        if _dec != "KEEP" and not _geak_nr:
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
    """Hot kernels still owing a ``kernel_opt`` attempt (reusable, gpu_pct >= min_gpu_pct, untouched); capped to top_n by gpu_pct, one kernel_id per task_group.

    Args:
        min_gpu_pct (float | None): Minimum GPU-share threshold; when
            ``None`` it is read from ``HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT``.
        top_n (int | None): Cap on enforced kernels by gpu_pct; when
            ``None`` it is read from ``HYPERLOOM_KERNEL_OPT_GATE_TOP_N``.

    Returns:
        list[str]: The untried hot-reusable ``kernel_id`` values (one per
            task_group), sorted strongest-first.
    """
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

    # Sort by gpu_pct desc so dedup picks the strongest member of each
    # task_group.
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
