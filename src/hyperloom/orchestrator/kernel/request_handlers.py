# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator-side programmatic handlers for kernel REQUEST kinds.

Handler signature::

    async def handler(payload: dict, *, session_dir: Path) -> dict:

Dispatch table is exposed via :data:`KERNEL_REQUEST_HANDLERS` for test monkey-patching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Awaitable, Callable

from ..actions.executors._kernel_agent_tool import HandlerResult
from ..actions.executors.trace_analyze import trace_analyze_handler
from ._forge_fusion_run import run_fusion_handler
from ._forge_gemm_tuning import _run_forge_gemm_tuning
from .integrate_accuracy import _integrate_accuracy_verdict
from ._integrate_outcomes import (
    _applied_patch_result,
    _FailureVerdict,
    _integrate_failure,
    _integrate_request_refusal,
    _integrate_result,
    _IntegrateVerdict,
    _request_overrides_serving_controls,
    _settled_patch,
    _unapplied_patch_failure,
    _unmeasured_integration_failure,
)
from ._integrate_payload import (
    _fill_integrate_defaults_from_state,
    _integrate_mode,
    _is_paired_gemm_measurement,
    _resolve_integrate_payload,
)
from ._integrate_rebaseline import (
    _cpp_itfs_rebuild_check,
    _forced_cpp_itfs_rebuild,
    _integrate_rebaseline,
    _multi_node_restart_failure,
    _run_integrate_rebaseline_with_lock_retry,
)
from ._source_import import _confirm_source_import
from ._tuning_audit_trace import _trace_gemm_tuning_run
from .keep_thresholds import STACK_INCREMENTAL_KEEP_THRESHOLD_PCT

HandlerFn = Callable[..., Awaitable[HandlerResult]]


async def run_gemm_tuning_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Run the forge GEMM tuner and record the run in the audit trace.

    Args:
        payload: The GEMM-tuning request payload.
        session_dir: Session directory for workspace and state.

    Returns:
        A ``HandlerResult`` describing the tuning outcome.
    """
    result = await _run_forge_gemm_tuning(payload, session_dir=session_dir)
    result.setdefault("task_id", payload.get("task_id"))
    result.setdefault("macro_cycle", payload.get("macro_cycle"))
    _trace_gemm_tuning_run(result, session_dir=session_dir)
    return result


log = logging.getLogger(__name__)


async def integrate_handler(
    payload: dict,
    *,
    session_dir: Path,
    preapplied_git_patch: bool = False,
) -> HandlerResult:
    """Apply a kernel patch + re-baseline + KEEP/REVERT decision.

    Applies an optimized kernel artifact, re-runs the active Magpie baseline,
    and KEEPs only when measured E2E throughput clears the threshold AND the
    re-baseline's accuracy holds (source + artifacts are backed up first so
    non-KEEP can restore without a rebuild). Accuracy is graded only for a
    candidate that already cleared the throughput bar -- see
    :func:`grade_integrate_accuracy`.

    Payload: ``base_tput`` must be > 0 at decision time, but is auto-filled from
    SharedState when a baseline has been recorded, so a bare ``{kernel_id}`` (or
    ``{integration_id}``) payload is accepted. Optional: patch_path,
    target_file, snapshot_dir, kernel_repo, config_path, extra_server_args,
    extra_envs, source, task_group_key, keep_threshold_pct (1.0). The baseline
    benchmark launch applies the invocation's fixed benchmark timeout policy.

    Returns the KEEP/REVERT decision and re-baseline metrics (``status``,
    ``decision``, ``base_tput``, ``new_tput``, ``gain_pct``, ``kernel_id``,
    ``patch_path``, ``report_path``, ``workspace``), plus ``accuracy`` /
    ``baseline_accuracy`` / ``accuracy_pass`` / ``accuracy_gate`` when the gate
    was graded. ``base_tput`` / ``new_tput`` remain output throughput;
    ``gain_pct`` follows ``graded_objective`` and ``bench_result`` retains the
    E2E measurement.
    """
    from ..actions.executors._accuracy_gate import EVAL_KIND_CONTEXT_TOO_SMALL
    from ..actions.executors.benchmark_result import is_valid_measurement
    from ..measurement.integrate_performance import assess_integrate_performance
    from ..state.shared_state import SharedState

    requested_controls = _request_overrides_serving_controls(payload)
    # Fill defaults from SharedState before the ``base_tput > 0`` check so a bare
    # {kernel_id} payload isn't failed with a phantom "missing base_tput".
    payload = _fill_integrate_defaults_from_state(payload, session_dir=session_dir)
    refusal = _integrate_request_refusal(payload, requested_controls=requested_controls)
    if refusal is not None:
        return refusal

    # Coordinator-internal: config-only measurements opt in explicitly; agent
    # integrate requests retain main's patch-resolution contract.
    mode = _integrate_mode(payload)
    paired_measurement = _is_paired_gemm_measurement(payload)
    if mode != "env_only":
        payload, missing_inputs = _resolve_integrate_payload(payload, session_dir=session_dir)
        if missing_inputs is not None:
            return missing_inputs

    state = SharedState.load_or_init(session_dir)
    apply_result = _applied_patch_result(payload, session_dir=session_dir, preapplied_git_patch=preapplied_git_patch)
    if mode == "env_only" and apply_result.get("status") == "skipped":
        apply_result = {
            "status": "ok",
            "reason": "env_only_no_patch_applied",
            "kernel_id": payload.get("kernel_id"),
        }
    log.info("integrate_handler: apply_result=%s", apply_result)
    unapplied = _unapplied_patch_failure(payload, apply_result)
    if unapplied is not None:
        return unapplied

    base_tput = float(payload.get("base_tput", 0.0))
    performance_policy = {
        "base_tput": base_tput,
        "keep_threshold_pct": float(payload.get("keep_threshold_pct", 1.0)),
        "stack_incremental_keep_threshold_pct": STACK_INCREMENTAL_KEEP_THRESHOLD_PCT,
    }
    rebaseline = _integrate_rebaseline(payload, state, session_dir=session_dir, performance_policy=performance_policy)
    with _forced_cpp_itfs_rebuild(apply_result):
        restart_failure = await _multi_node_restart_failure(rebaseline, payload, apply_result)
        if restart_failure is not None:
            return restart_failure
        try:
            bench_result = await _run_integrate_rebaseline_with_lock_retry(
                rebaseline.executor,
                rebaseline.ctx,
                workspace=rebaseline.workspace,
                reason=f"integrate {payload.get('kernel_id') or 'anonymous'} rebaseline",
            )
        except Exception as exc:  # isolation boundary: one integrate rebaseline, recorded as a REVERT
            log.warning("integrate_handler: rebaseline failed: %s", exc, exc_info=True)
            return _integrate_failure(
                payload, apply_result, _FailureVerdict("rebaseline_exception", repr(exc), "REVERT")
            )
    if not is_valid_measurement(bench_result):
        return _unmeasured_integration_failure(payload, apply_result=apply_result, bench_result=bench_result)

    rebuild_check = _cpp_itfs_rebuild_check(apply_result)
    if not rebuild_check.get("verified", True):
        return _integrate_failure(
            payload,
            apply_result,
            _FailureVerdict(
                "cpp_itfs_rebuild_not_verified",
                "re-baseline did not produce a freshly-built cpp_itfs lib.so; refusing to score a possibly-stale binary",
                "NEEDS_REVIEW",
            ),
            rebuild_check=rebuild_check,
        )

    if paired_measurement:
        return {
            "status": "ok",
            "decision": "NEEDS_REVIEW",
            "base_tput": base_tput,
            "new_tput": float(bench_result.get("output_throughput") or 0.0),
            "bench_result": bench_result,
            "workspace": bench_result.get("workspace"),
        }
    performance = assess_integrate_performance(state, bench_result, **performance_policy)
    if performance.graded.degrade_reason:
        log.info("integrate_handler: grading unavailable (%s)", performance.graded.degrade_reason)
    # Read after the re-baseline, not before it: the executor shares this state
    # object and a salvaged sibling accuracy during the run republishes the
    # reference this candidate is graded against.
    accuracy_gate, accuracy_decision = _integrate_accuracy_verdict(
        payload, bench_result, rebaseline, performance, float(state.baseline_accuracy or 0.0)
    )
    if accuracy_gate is not None and accuracy_gate["blocked"] and accuracy_gate["infeasible"]:
        # A gate that cannot run under this serving configuration measured nothing
        # about the patch, so it is an integration fault: faults carry their own
        # budget and never spend one of the attempts a patch gets to prove itself.
        return _integrate_failure(
            payload,
            apply_result,
            _FailureVerdict(EVAL_KIND_CONTEXT_TOO_SMALL, accuracy_gate["reason"], "NEEDS_REVIEW"),
            gain_pct=performance.gain_pct,
            accuracy_gate=accuracy_gate,
        )

    # Only the strict sub-flag enforces the confirmation; an unknown never penalizes.
    confirmation = _confirm_source_import(
        payload,
        workspace=bench_result.get("workspace"),
        decision=accuracy_decision,
    )
    return _integrate_result(
        _IntegrateVerdict(
            payload=payload,
            bench_result=bench_result,
            performance=performance,
            cleanup=_settled_patch(confirmation.decision, apply_result, caller_owns_durability=preapplied_git_patch),
            accuracy_gate=accuracy_gate,
            source_import=confirmation,
            rebuild_check=rebuild_check,
            server_args=rebaseline.server_args,
        )
    )


# Kernel-agent programmatic dispatch table.
KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "trace_analyze": trace_analyze_handler,
    # No run_gemm_tuning / run_fusion entries: KernelPhase awaits those handlers directly.
    "integrate": integrate_handler,
    "apply_patch": integrate_handler,  # alias — same flow
}


__all__ = [
    "KERNEL_REQUEST_HANDLERS",
    "integrate_handler",
    "run_fusion_handler",
    "run_gemm_tuning_handler",
]
