# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Promoting a settled integrate window into the optimization stack."""

from __future__ import annotations

import logging
from typing import Any, Mapping, NamedTuple

from hyperloom.common.coerce import to_str_list
from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ..actions.executors._integrate_vocabulary import KEEP_STATUSES
from ..measurement.integrate_performance import integrate_measurement_fields
from ..measurement.runtime_findings import KEEP_REASON_CORRECTNESS_FIX
from ..state.task_registry import Task
from ._keep_patch_column import _is_patch_column_keep, _lever_for_keep, _source_layer_handles
from ._promote_outcome import IntegrateRecoveryIncomplete, _PromoteOutcome

log = logging.getLogger(__name__)


# Statuses meaning a patch was applied and measured, then rolled back or rejected on measured grounds.
_INTEGRATE_REVERT_STATUSES: frozenset[str] = frozenset({"reverted", "accuracy_unavailable_reject", "regression"})


def _integrate_marker_verdict(status: str, phase: str) -> str:
    """Classify one integrate window from its verdict and its durable phase.

    The pair is the whole truth about an attempt's obligations, so the live
    promotion path and the resume recovery path read it here rather than each
    keeping its own table.

    Returns:
        ``"retained"`` when the candidate was deliberately left in the tree and
        the user's stash was popped, ``"settled"`` when the tree was restored
        to its pre-attempt state, ``"inflight"`` when an apply-only attempt is
        still waiting for its benchmark, and ``"incomplete"`` when the two
        disagree and no obligation can be discharged from them.
    """
    if phase == "accepted" and status in KEEP_STATUSES:
        return "retained"
    if phase == "restored" and status not in KEEP_STATUSES and status != "applied_no_bench":
        return "settled"
    if status == "applied_no_bench" and phase in {"applied", "applied_with_restored_stash"}:
        return "inflight"
    return "incomplete"


def _confirmed_task_outcome(task: Task) -> dict[str, Any] | None:
    """Read a task's own record of the result its executor produced.

    ``SubAgentRunner`` writes the full outcome onto the task row before the
    ``delegated_result`` reaches the bus, so a crash inside that window leaves
    the verdict durable here and nowhere else. An unconfirmed physical cleanup
    makes that outcome diagnostic only — the work it describes may still hold
    resources — so it is not returned.

    TODO: ``bringup.reconcile._terminal_by_observation`` reads the same evidence
    shape under slightly different rules; the two want one reader.
    """
    for entry in reversed(getattr(task, "history", None) or []):
        if not isinstance(entry, dict):
            continue
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("outcome"), dict):
            continue
        if evidence.get("cleanup_confirmed") is not True:
            return None
        result = evidence["outcome"].get("result")
        return result if isinstance(result, dict) else None
    return None


class _IntegrateMarker(NamedTuple):
    phase: str
    outcome: dict[str, Any] | None
    verdict: str


def _integrate_marker_of(pending: Mapping[str, Any], task: Task | None) -> _IntegrateMarker:
    recovery = pending.get("recovery")
    phase = str(recovery.get("phase") or "") if isinstance(recovery, dict) else ""
    outcome = _confirmed_task_outcome(task) if task is not None else None
    verdict = _integrate_marker_verdict(str((outcome or {}).get("status") or ""), phase) if outcome else ""
    return _IntegrateMarker(phase=phase, outcome=outcome, verdict=verdict)


class _KeptResultScan(NamedTuple):
    scanned: bool
    kept: dict[str, Any] | None


def _task_recorded_a_result(task: Task) -> bool:
    return any(
        any(key in (row.get("evidence") or {}) for key in ("result_keys", "result", "outcome"))
        for row in task.history
        if isinstance(row, dict)
    )


def _integrate_stack_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the members a KEEP promotes.

    Only the kernel-stack lane produces multi-member results, and it binds them to their ledger rows before
    promoting, so the members are read off the result rather than bound again here.
    """
    from ..phases.kernel_stack import resolve_stack_members

    raw_members = result.get("stack_kernel_ids")
    membership = {"stack_validation": isinstance(raw_members, list) and len(raw_members) > 1, **result}
    members = resolve_stack_members(membership)
    return {"stack_kernel_ids": list(members), "stack_validation": membership["stack_validation"]}


def _integrate_keep_tput(result: Mapping[str, Any], measurement: Mapping[str, Any]) -> float | None:
    if measurement is not result and "output_throughput" in measurement:
        measured = measurement["output_throughput"]
    else:
        measured = result.get("output_throughput")
        if measured is None:
            measured = result.get("new_tput")
    if not isinstance(measured, (int, float)) or measured <= 0:
        return None
    return float(measured)


def _integrate_verdict(status: str, *, lifted: bool, kept: bool) -> Verdict:
    if status == "kept_inert":
        return Verdict.RECORDED
    if kept:
        return Verdict.ADOPTED if lifted else Verdict.REFUSED
    if status in _INTEGRATE_REVERT_STATUSES:
        return Verdict.REVERTED
    return Verdict.FAILED


def _patch_recipe_delta(result: Mapping[str, Any], declared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "remove_args": to_str_list(declared.get("remove_args")),
        "unset_envs": to_str_list(declared.get("unset_envs")),
        "args_mode": str(result.get("args_mode") or declared.get("args_mode") or "append").strip().lower(),
    }


def _integrate_keep_origin(
    result: Mapping[str, Any], task_params: Mapping[str, Any], source_phase: str
) -> dict[str, Any]:
    domain = str(task_params.get("domain") or result.get("domain") or "").strip()
    provenance = str(task_params.get("provenance") or result.get("provenance") or "").strip()
    if domain and not provenance.startswith("specialist:"):
        provenance = f"specialist:{domain}"
    fields: dict[str, Any] = {"provenance": provenance or "integrate_patch", "scope": "source_patch"}
    lever_kind = _lever_for_keep(task_params, result)
    if lever_kind:
        fields["lever_kind"] = lever_kind
    if source_phase:
        fields["source_phase"] = source_phase
    if domain:
        fields["domain"] = domain
    gap_layer = str(task_params.get("gap_layer") or result.get("gap_layer") or "").strip()
    if gap_layer:
        fields["gap_layer"] = gap_layer
    if task_params.get("framework_agent_authoring") or result.get("framework_agent_authoring"):
        fields["framework_agent_authoring"] = True
    if result.get("enablement") or task_params.get("enablement"):
        fields["recipe_publishable"] = False
    # A verified correctness fix was bounded by the integrate gate, not by a gain.
    if result.get("keep_reason") == KEEP_REASON_CORRECTNESS_FIX:
        fields["keep_reason"] = KEEP_REASON_CORRECTNESS_FIX
        fields["resolves_finding"] = str(result.get("resolves_finding") or "")
        fields["attribution_eligible"] = False
    return fields


def _integrate_keep_variant(
    result: Mapping[str, Any],
    task: "Task | None",
    measurement: Mapping[str, Any],
    tput: float,
    source_phase: str,
) -> dict[str, Any]:
    """Build the stack variant an integrate_patch KEEP lifts, live (with its task) or on resume replay (``task=None``)."""
    task_params = (getattr(task, "params", None) or {}) if task is not None else {}
    specialist_task_id = str(result.get("specialist_task_id") or task_params.get("specialist_task_id") or "")
    return {
        "name": specialist_task_id or "integrate_patch_keep",
        "task_id": getattr(task, "task_id", "") if task is not None else "",
        "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "candidate_extra_envs": dict(result.get("extra_envs_applied") or {}),
        "recipe_delta": _patch_recipe_delta(result, task_params),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "tput": tput,
        **integrate_measurement_fields(measurement),
        **_source_layer_handles(result),
        **_integrate_keep_origin(result, task_params, source_phase),
    }


class IntegratePromotionMixin:
    """Promoting an integrate_patch KEEP."""

    async def record_integrate_keep(self, result: dict[str, Any]) -> None:
        """Promote a kernel integrate KEEP into the optimization stack.

        Stamps ``cumulative_gain_validated`` and fires a watermark roofline once
        the lift is accepted. No-op without a positive ``new_tput`` or when the
        lift refuses the winner.

        Args:
            result (dict[str, Any]): The integrate-patch executor result.
        """
        measurement = result.get("bench_result") or result
        new_tput = _integrate_keep_tput(result, measurement)
        if new_tput is None:
            return
        stack_fields = _integrate_stack_fields(result)
        # A fusion sibling drained through the shared integrate lane must still
        # land on the stack as ``action="fusion"``: the idempotency short-circuit
        # (``_active_forge_fusion_env_flags``) and the remote-recipe fusion export
        # (``build_kernel_fusion_value``) both key on that label, and the latter
        # also gates on ``last_fusion_integrate`` being a KEEP. The generic path
        # is otherwise unchanged.
        is_fusion = str(result.get("source") or "") == "forge_fusion"
        action_label = str(result.get("action_label") or "").strip() if is_fusion else ""
        lift_kind = action_label or "integrate"
        variant: dict[str, Any] = {
            "name": result.get("kernel_id"),
            "candidate_extra_server_args": result.get("extra_server_args"),
            "extra_envs": {str(k): str(v) for k, v in (result.get("extra_envs") or {}).items()},
            "source_phase": str(self.shared_state.phase or "KERNEL_AGENT"),
            **integrate_measurement_fields(measurement),
            **_source_layer_handles(result),
        }
        if is_fusion:
            variant["provenance"] = "forge_fusion"
        lifted = self.lift_to_current_best(
            lift_kind,
            float(new_tput),
            variant,
            gap_canonical_id=str(result.get("gap_canonical_id") or "").strip(),
            entry_extra={
                "integration_id": result.get("integration_id"),
                "kernel_id": result.get("kernel_id"),
                "task_group_key": result.get("task_group_key"),
                "identity_route": result.get("identity_route"),
                "patch_path": result.get("patch_path"),
                "target_file": result.get("target_file"),
                "gain_pct": result.get("gain_pct"),
                **stack_fields,
                "backend": result.get("backend"),
                "engine": result.get("engine"),
                # Provenance for a fusion sibling; readers key the stack row on
                # ``action == "fusion"`` above, this just records the producer.
                **({"backend": "forge", "engine": "forge_fusion"} if is_fusion else {}),
                # Preserve committed source-layer identity for recipe export.
                **{key: result[key] for key in ("scope", "operator_id", "base_sha", "keep_commit") if result.get(key)},
            },
        )
        if not lifted:
            return
        if is_fusion:
            self.shared_state.last_fusion_integrate = {**result, "decision": "KEEP"}
        self._coord.recipe_journal.journal_integrate_keep(result, lift_kind=lift_kind, new_tput=float(new_tput))
        await self._validate_and_watermark(
            new_tput, measurement, source="integrate_keep", watermark_reason="integrate_keep_watermark"
        )

    def _confirmed_integrate_completion(
        self, result: dict, task: "Task | None", *, kept_flag: bool
    ) -> dict[str, Any] | None:
        """Return the matching, completed marker without performing recovery."""
        pending = self.shared_state.pending_integrate
        task_id = str(getattr(task, "task_id", "") or "")
        matches = isinstance(pending, dict) and bool(task_id) and pending.get("task_id") == task_id
        failed = (
            result.get("error_class") == "integrate_restore_incomplete"
            or bool(result.get("recovery_errors"))
            or bool(result.get("stash_restore_error"))
        )
        if not matches and not failed:
            return None
        recovery = pending.get("recovery") if matches else None
        phase = str(recovery.get("phase") or "") if isinstance(recovery, dict) else ""
        status = str(result.get("status") or "")
        if not failed:
            if matches and "recovery" not in pending and kept_flag:
                return pending
            verdict = _integrate_marker_verdict(status, phase)
            if verdict in ("retained", "settled"):
                return pending
            if verdict == "inflight":
                return None
        self.shared_state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
        self.shared_state.save(self.session_dir)
        raise IntegrateRecoveryIncomplete(
            f"integrate recovery is incomplete or inconsistent for task={task_id!r}: "
            f"phase={phase!r}, status={status!r}, errors={result.get('recovery_errors')!r}, "
            f"stash_restore_error={result.get('stash_restore_error')!r}"
        )

    async def _promote_integrate_patch(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote only after recovery is confirmed; clear only its completed marker."""
        stack_len_before = len(self.shared_state.optimization_stack or [])
        status = str(result.get("status") or "")
        measurement = result.get("bench_result") or result
        new_tput = measurement.get("output_throughput", result.get("output_throughput"))
        kept_flag = status == "kept" and isinstance(new_tput, (int, float)) and float(new_tput) > 0
        completed_pending = self._confirmed_integrate_completion(result, task, kept_flag=kept_flag)
        changed = self._record_authored_levers(result)
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        enablement_landing = bool(result.get("enablement") or task_params.get("enablement"))
        lifted = False
        if kept_flag:
            lifted = await self._lift_integrate_keep(result, task, measurement, enablement_landing=enablement_landing)
            changed = True
        if completed_pending is not None and self.shared_state.pending_integrate is completed_pending:
            self.shared_state.pending_integrate = {}
            changed = True
        outcome.verdict = _integrate_verdict(status, lifted=lifted, kept=kept_flag)
        # A measured integrate counts either way; KEEP/REVERT is a later judgement.
        if new_tput is not None:
            self.shared_state.gain_gated_action_count += 1
        if (
            lifted
            and not enablement_landing
            and len(self.shared_state.optimization_stack or []) > stack_len_before
            and _is_patch_column_keep(task_params, result)
        ):
            self._enqueue_agent_keep_outbox(
                stack_index=len(self.shared_state.optimization_stack) - 1,
                result=result,
                task=task,
                include_patches=True,
            )
        outcome.changed = changed

    def _record_authored_levers(self, result: Mapping[str, Any]) -> bool:
        levers = result.get("framework_levers")
        if not (isinstance(levers, list) and levers):
            return False
        lever_outcome = str(result.get("framework_lever_outcome") or "")
        return bool(
            self.shared_state.record_authored_framework_levers(
                levers,
                default_on=(lever_outcome == "default_on"),
            )
        )

    async def _lift_integrate_keep(
        self,
        result: dict,
        task: "Task | None",
        measurement: Mapping[str, Any],
        *,
        enablement_landing: bool,
    ) -> bool:
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        new_tput = float(measurement.get("output_throughput", result.get("output_throughput")))
        source_phase = str(task_params.get("source_phase") or result.get("source_phase") or "").strip()
        variant = _integrate_keep_variant(result, task, measurement, new_tput, source_phase)
        prebaseline = enablement_landing and float(self.shared_state.baseline_tput or 0.0) <= 0.0
        if prebaseline:
            variant["baseline_enablement"] = True
            variant["attribution_eligible"] = False
        lifted = self.lift_to_current_best(
            "integrate_patch",
            new_tput,
            variant,
            gap_canonical_id=str(task_params.get("gap_canonical_id") or result.get("gap_canonical_id") or "").strip(),
        )
        if prebaseline:
            log.info(
                "integrate_patch KEEP accepted as pre-baseline enablement; "
                "retained in config stack without gain attribution "
                "(task=%s specialist=%s)",
                str(getattr(task, "task_id", "") or ""),
                str(result.get("specialist_task_id") or task_params.get("specialist_task_id") or ""),
            )
        elif lifted:
            await self._validate_and_watermark(
                new_tput,
                measurement,
                source="integrate_patch_keep",
                watermark_reason="integrate_patch_keep_watermark",
            )
        return lifted

    async def _promote_conc_sweep(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a conc_sweep result: record_conc_sweep + save; discovery-only.

        No audit attempt is written: ``record_action_attempt`` returns early for
        any kind outside ``_AUDIT_ACTIONS``, which conc_sweep is not in. The
        conc_sweep event carries the skip reason, the budget verdict and the
        summary instead.
        """
        outcome.early_return = True
        # Write last_conc_sweep so the SWEEP exit logic can distinguish an honest sweep_done from a no-pair sweep_failed.
        self.shared_state.record_conc_sweep(result)
        self.shared_state.save(self.session_dir)
