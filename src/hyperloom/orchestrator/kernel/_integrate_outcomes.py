# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The documents one integration attempt returns: its refusals, its failures and its verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

from hyperloom.common.coerce import to_str_list

from ..actions.executors._kernel_agent_tool import (
    HandlerResult,
    _maybe_apply_kernel_patch,
    _maybe_finalize_kernel_patch,
    _maybe_revert_kernel_patch,
)
from ..actions.stop_attribution import stopped_by_the_run_class
from ._integrate_payload import _integrate_mode, _is_paired_gemm_measurement, _preapplied_snapshot_payload
from ._source_import import _SourceImport
from .keep_thresholds import STACK_INCREMENTAL_KEEP_THRESHOLD_PCT
from .patch_lifecycle import cleanup_verdict as _cleanup_verdict


# A patch whose correctness was only established against a reference kernel;
# serving accuracy is what settles it.
_FRAMEWORK_APPLYBACK_ARTIFACT_KIND = "framework_applyback"
_INTEGRATE_ACCURACY_VALIDATION_TIER = "integrate_e2e_accuracy"


def _applied_patch_result(
    payload: dict[str, Any],
    *,
    session_dir: str,
    preapplied_git_patch: bool,
) -> dict[str, Any]:
    # A controller publication is git-applied to the worktree before the validator
    # runs, so its final bytes are already on disk; apply reads them as a snapshot
    # instead of mistaking the diff for replacement source. Skipping the apply
    # outright would also skip the cache invalidation, multi-node fan-out and
    # rebuild the measurement depends on. Only an in-process caller can set this:
    # an agent's integrate params land in ``payload`` verbatim, so the payload
    # cannot carry the trust.
    return _maybe_apply_kernel_patch(
        _preapplied_snapshot_payload(payload) if preapplied_git_patch else payload,
        session_dir=session_dir,
        kernel_id=payload.get("kernel_id"),
    )


@dataclass(frozen=True)
class _FailureVerdict:
    """Why one integration attempt failed and the decision on its patch."""

    error_class: str
    error: str
    decision: str


def _integrate_failure(
    payload: dict[str, Any],
    apply_result: dict[str, Any],
    verdict: _FailureVerdict,
    **extra: Any,
) -> HandlerResult:
    return {
        "status": "failed",
        "error_class": verdict.error_class,
        "error": verdict.error,
        "decision": verdict.decision,
        "kernel_id": payload.get("kernel_id"),
        "patch_path": payload.get("patch_path"),
        "target_file": payload.get("target_file") or payload.get("source_file"),
        "apply_result": apply_result,
        "revert_result": _maybe_revert_kernel_patch(apply_result),
        **extra,
    }


def _unmeasured_integration_failure(
    payload: dict[str, Any],
    *,
    apply_result: dict[str, Any],
    bench_result: Any,
) -> HandlerResult:
    rebaseline_error_class = (
        str((bench_result or {}).get("error_class") or "").strip() if isinstance(bench_result, dict) else ""
    ) or "bench_exception"
    # A run the stop class claims measured nothing, so the patch has no verdict to answer for.
    stopped = stopped_by_the_run_class(rebaseline_error_class)
    verdict = (
        _FailureVerdict(stopped.error_class, stopped.interrupted, "NEEDS_REVIEW")
        if stopped is not None
        else _FailureVerdict(rebaseline_error_class, "re-baseline did not succeed", "REVERT")
    )
    return _integrate_failure(payload, apply_result, verdict, rebaseline_detail=bench_result)


class _IntegrateVerdict(NamedTuple):
    payload: dict
    bench_result: dict[str, Any]
    performance: Any
    cleanup: dict[str, Any]
    accuracy_gate: dict[str, Any] | None
    source_import: _SourceImport
    rebuild_check: HandlerResult
    server_args: str


def _request_overrides_serving_controls(payload: Mapping[str, Any]) -> bool:
    return (
        bool(to_str_list(payload.get("remove_args")))
        or bool(to_str_list(payload.get("unset_envs")))
        or str(payload.get("args_mode") or "append").strip().lower() == "replace"
    )


def _integrate_request_refusal(payload: dict, *, requested_controls: bool) -> HandlerResult | None:
    if float(payload.get("base_tput", 0.0)) <= 0:
        return {
            "status": "failed",
            "error": "integrate_handler requires base_tput > 0 to compute KEEP/REVERT",
        }
    paired_measurement = _is_paired_gemm_measurement(payload)
    if paired_measurement and any(
        payload.get(key) for key in ("patch_path", "target_file", "source_file", "snapshot_dir")
    ):
        raise ValueError("GEMM paired measurement cannot apply a kernel artifact")
    if _integrate_mode(payload) != "env_only":
        return None
    if (
        payload.get("extra_envs")
        or str(payload.get("extra_server_args") or "").strip()
        or requested_controls
        or paired_measurement
    ):
        return None
    return {
        "status": "failed",
        "error_class": "env_only_missing_envs",
        "error": "env_only integrate requires runtime configuration or an explicit paired reference",
    }


def _unapplied_patch_failure(payload: dict, apply_result: dict) -> HandlerResult | None:
    status = apply_result.get("status")
    if status == "ok":
        return None
    if status == "failed":
        apply_reason = str(apply_result.get("error") or apply_result.get("reason") or "").strip()
        error_class = "apply_failed"
        error = f"kernel patch apply failed: {apply_reason}" if apply_reason else "kernel patch apply failed"
    else:
        error_class = "patch_not_applied"
        error = "kernel patch was not applied; refusing to run E2E benchmark"
    return {
        "status": "failed",
        "error_class": error_class,
        "error": error,
        "decision": "REVERT",
        "apply_result": apply_result,
        "kernel_id": payload.get("kernel_id"),
        "patch_path": payload.get("patch_path"),
        "target_file": payload.get("target_file") or payload.get("source_file"),
    }


def _applyback_verdict_pending(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("artifact_kind") or "") == _FRAMEWORK_APPLYBACK_ARTIFACT_KIND
        and str(payload.get("integration_validation_status") or "") != "passed"
    )


def _settled_patch(decision: str, apply_result: dict, *, caller_owns_durability: bool) -> dict[str, Any]:
    if decision != "KEEP":
        finalize_result: HandlerResult = {"status": "skipped", "reason": "non-KEEP decision"}
    elif caller_owns_durability:
        finalize_result = {"status": "skipped", "reason": "caller owns the KEEP's durability"}
    else:
        finalize_result = _maybe_finalize_kernel_patch(apply_result)
    revert_result = (
        {"status": "skipped", "reason": "KEEP decision"}
        if decision == "KEEP"
        else _maybe_revert_kernel_patch(apply_result)
    )
    top_status, patch_cleanup_action = _cleanup_verdict(
        decision=decision,
        revert_result=revert_result,
        finalize_result=finalize_result,
        revert_required=decision != "KEEP" and bool(apply_result.get("manifest_path")),
    )
    return {
        "status": top_status,
        "decision": decision,
        "patch_cleanup_action": patch_cleanup_action,
        "apply_result": apply_result,
        "revert_result": revert_result,
        "finalize_result": finalize_result,
    }


def _integrate_result(verdict: _IntegrateVerdict) -> HandlerResult:
    payload = verdict.payload
    bench_result = verdict.bench_result
    result: dict[str, Any] = {
        **verdict.cleanup,
        "kernel_id": payload.get("kernel_id"),
        "patch_path": payload.get("patch_path"),
        "target_file": payload.get("target_file") or payload.get("source_file"),
        "base_tput": float(payload.get("base_tput", 0.0)),
        "new_tput": float(bench_result.get("output_throughput") or 0.0),
        "gain_pct": verdict.performance.gain_pct,
        "graded_objective": verdict.performance.graded.objective,
        "bench_result": bench_result,
        "report_path": bench_result.get("report_path"),
        "workspace": bench_result.get("workspace"),
        "extra_server_args": verdict.server_args,
        "extra_envs": dict(payload.get("extra_envs") or {}),
        "rebuild_check": verdict.rebuild_check,
        "task_group_key": str(payload.get("task_group_key") or ""),
        "identity_route": str(payload.get("identity_route") or ""),
        "integration_id": str(payload.get("integration_id") or ""),
    }
    # Fusion provenance rides through so the KEEP writeback lifts the stack row as
    # ``fusion`` (not ``integrate``) and sets ``last_fusion_integrate`` -- both are
    # read by the idempotency short-circuit and the remote-recipe fusion export.
    if str(payload.get("source") or "") == "forge_fusion":
        result["source"] = "forge_fusion"
        result["action_label"] = str(payload.get("action_label") or "fusion")
    if _integrate_mode(payload) == "env_only":
        result["advisory"] = (
            "env_only mode: the E2E benchmark measured the configuration change only, not a code patch."
        )
    if result["status"] == "failed":
        result["error_class"] = "patch_revert_incomplete"
        result["error"] = str(result["revert_result"].get("error") or "Kernel patch revert did not complete")
    _explain_integrate_decision(result, verdict)
    return result


def _explain_integrate_decision(result: HandlerResult, verdict: _IntegrateVerdict) -> None:
    performance = verdict.performance
    graded = performance.graded
    source_import = verdict.source_import
    if graded.graded_on_intvty and graded.verdict == "REVERT":
        result["decision_reason"] = "intvty_regression"
    keep_threshold_pct = float(verdict.payload.get("keep_threshold_pct", 1.0))
    if performance.stack_positive_keep and performance.gain_pct <= keep_threshold_pct:
        result["decision_reason"] = "stack_positive_increment"
        result["stack_incremental_gain_pct"] = performance.stack_incremental_gain_pct
        result["stack_incremental_keep_threshold_pct"] = STACK_INCREMENTAL_KEEP_THRESHOLD_PCT
    if source_import.confirmed is not None:
        result["source_import_confirmed"] = source_import.confirmed
    if len(source_import.evidence) > 1:
        result["source_import_evidence"] = source_import.evidence
    if source_import.downgraded:
        result["decision_reason"] = "source_not_confirmed_imported"
    gate = verdict.accuracy_gate
    if gate is not None:
        result["accuracy"] = gate["accuracy"]
        result["baseline_accuracy"] = gate["baseline_accuracy"]
        result["accuracy_pass"] = gate["accuracy_pass"]
        result["accuracy_gate"] = gate
        if gate["blocked"]:
            result["decision_reason"] = (
                "accuracy_regression" if gate["accuracy_pass"] is False else "accuracy_evidence_missing"
            )
    if not _applyback_verdict_pending(verdict.payload):
        return
    result["artifact_kind"] = _FRAMEWORK_APPLYBACK_ARTIFACT_KIND
    # Only a KEEP settles the outstanding verdict. A non-KEEP is left unstamped:
    # the attempt ledger already distinguishes a rejection from a retryable
    # fault, and this field must not blur the two.
    if result["decision"] == "KEEP":
        result["integration_validation_status"] = "passed"
        result["validation_tier"] = _INTEGRATE_ACCURACY_VALIDATION_TIER
