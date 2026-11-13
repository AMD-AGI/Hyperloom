# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Promoting a settled integrate window into the optimization stack."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, NamedTuple

from hyperloom_common.coerce import to_str_list
from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ..actions.executors._integrate_vocabulary import KEEP_STATUSES
from ..measurement.integrate_performance import integrate_measurement_fields
from ..measurement.runtime_findings import KEEP_REASON_CORRECTNESS_FIX, KEEP_REASON_EXPORT_LANDED
from ..state.task_registry import Task
from ._keep_patch_column import _is_patch_column_keep, _lever_for_keep, _source_layer_handles
from .launch_controls import LAUNCH_CONTROL_KEYS
from ._promote_outcome import IntegrateRecoveryIncomplete, _PromoteOutcome

log = logging.getLogger(__name__)

#: A KEEP whose patch landed with every switch off: adopted source, no stack entry.
_KEPT_INERT = "kept_inert"
#: The scope recorded for a layer that changed source rather than configuration.
_SOURCE_PATCH = "source_patch"


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
    if status == _KEPT_INERT:
        return Verdict.RECORDED
    if kept:
        return Verdict.ADOPTED if lifted else Verdict.REFUSED
    if status in _INTEGRATE_REVERT_STATUSES:
        return Verdict.REVERTED
    return Verdict.FAILED


def _patch_recipe_delta(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "remove_args": to_str_list(result.get("remove_args")),
        "unset_envs": to_str_list(result.get("unset_envs")),
        "args_mode": str(result.get("args_mode") or "append").strip().lower(),
    }


def _kept_launch_controls(result: Mapping[str, Any], measurement: Mapping[str, Any]) -> dict[str, Any]:
    """The removal controls a KEEP launched with and the args it keeps, as an explore KEEP lifts them.

    A persisted KEEP replaces the launch with the argv its bench ran; a native one keeps the stack relative to the
    recipe. Either way the previous args still hold what was removed, so the lift must not merge onto them.
    """
    controls = {key: result[key] for key in LAUNCH_CONTROL_KEYS if result.get(key)}
    if not controls:
        return {}
    launched = result.get("effective_extra_server_args")
    if launched is None:
        return {**controls, "extra_server_args": str(measurement["effective_config"]["extra_server_args"])}
    return {**controls, "extra_server_args": launched, "effective_extra_server_args": launched}


def _keep_on_correctness_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """The extra KEEP fields for a landing the integrate gate bounded rather than a gain.

    A verified correctness fix is one. So is the first export: it is kept on correctness
    too, and becomes the grading anchor, so what follows is measured against the exported
    path rather than the eager one it replaced. Neither is attributable to a gain.
    """
    reason = result.get("keep_reason")
    if reason == KEEP_REASON_CORRECTNESS_FIX:
        return {
            "keep_reason": KEEP_REASON_CORRECTNESS_FIX,
            "resolves_finding": str(result.get("resolves_finding") or ""),
            "attribution_eligible": False,
        }
    if reason == KEEP_REASON_EXPORT_LANDED:
        return {"keep_reason": KEEP_REASON_EXPORT_LANDED, "attribution_eligible": False}
    return {}


def _integrate_keep_origin(
    result: Mapping[str, Any], task_params: Mapping[str, Any], source_phase: str
) -> dict[str, Any]:
    domain = str(task_params.get("domain") or result.get("domain") or "").strip()
    provenance = str(task_params.get("provenance") or result.get("provenance") or "").strip()
    if domain and not provenance.startswith("specialist:"):
        provenance = f"specialist:{domain}"
    fields: dict[str, Any] = {"provenance": provenance or "integrate_patch", "scope": _SOURCE_PATCH}
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
    fields.update(_keep_on_correctness_fields(result))
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
        "recipe_delta": _patch_recipe_delta(result),
        **_kept_launch_controls(result, measurement),
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
        changed = self._record_keep_bookkeeping(result, kept_flag=kept_flag, measurement=measurement, status=status)
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

    def _record_keep_bookkeeping(
        self, result: Mapping[str, Any], *, kept_flag: bool, measurement: Mapping[str, Any], status: str
    ) -> bool:
        """Record what this integrate changed outside the stack; True when any of it moved state."""
        changed = self._record_authored_levers(result)
        if self._record_export_gate(kept_flag=kept_flag, workspace=str(measurement.get("workspace") or "")):
            changed = True
        if status == _KEPT_INERT and self._record_inert_source_layer(result):
            changed = True
        return changed

    def _record_export_gate(self, *, kept_flag: bool, workspace: str = "") -> bool:
        """Judge whether the module this KEEP accepted really exported.

        Inert unless ``aoti.enabled``. The evidence is the run's own call counters, not
        its source: a module that mentions the AOTI entry points in a helper nobody
        calls, in a dead branch, or in the ``except`` arm of a fallback to eager parses
        the same as one that exports, and benchmarks just as happily. Counting the calls
        separates them -- see ``measurement.aoti_evidence``.

        Every KEEP is judged, not just the ones before the first landing: ``current_best``
        moves whenever a KEEP lands, so "does the deliverable export?" has a fresh answer
        each time and is recorded as one in ``current_best_exports``.

        Losing an export is prevented rather than recorded -- ``_export_regression_refusal``
        reverts any candidate that cannot be shown to export, so an accepted export keeps
        the slot. This stays a judgement rather than an assertion because an enablement
        round keeps through a lane that never reaches that gate; if one lands eager, the
        dispatcher closing the gate is the right outcome, not a crash.

        ``export_landed`` stays one-way, and only the first-export throughput waiver reads
        it. Keeping the two apart matters: a flag that reopened when the gate closed would
        re-arm that waiver, and a slow export waived back in could trade the slot with
        eager indefinitely.

        There is no source-reading fallback. A promotion judges a benchmark that just ran
        under this build, so its counters are always present -- including on a resumed
        session, whose next KEEP is a fresh run.

        ``export_attempts`` counts rounds that cleared their gate without exporting. Most
        are now counted by ``_gate_perf``, where ``_export_regression_refusal`` turns them
        into REVERTs before they reach here; this increment covers the lane that bypasses
        that gate, where an enablement round keeps on bring-up rather than throughput. An
        ordinary REVERT is still not an export attempt -- it is a candidate that failed on
        its own merits -- and counting one would read as progress toward an export.

        Returns whether it mutated state, so the caller persists it; a failed attempt is
        the one this counter exists to record, and that path has no other reason to save.
        """
        from hyperloom.common.optimization_scope import aoti_config

        try:
            config = aoti_config()
        except ValueError:
            return False
        if config is None or not config.enabled:
            return False
        if not kept_flag:
            return False
        exported = self._keep_really_exported(workspace)
        changed = exported != self.shared_state.current_best_exports
        self.shared_state.current_best_exports = exported
        if exported:
            if not self.shared_state.export_landed:
                self.shared_state.export_landed = True
                changed = True
            return changed
        self.shared_state.export_attempts += 1
        return True

    def _keep_really_exported(self, workspace: str) -> bool:
        """Whether this KEEP's benchmark ran an exported module.

        ``workspace`` is the graded leg's own run directory -- ``bench_result``'s, not the
        verdict's. The verdict names the attempt's task root, under which ``read_evidence``
        would also sum the switch-off parity leg, which runs this same patch with the
        switches *unset* and reports ``invoked=0`` by design, and any counters an earlier
        attempt left there: the root is keyed by task id and reused. Either opens the gate
        for a module that never exported. The parity leg is the live one here -- it runs
        after the perf gate reads, but before this does.
        """
        from ..measurement.aoti_evidence import read_evidence

        evidence = read_evidence(workspace) if workspace else None
        if evidence is None or not evidence.observed:
            # The counters are armed for every run of an aoti workload, so their absence
            # is a broken harness, not a verdict -- most likely the start-up shim never
            # loaded. Hold the gate and say which it is.
            log.warning(
                "export gate: no run counters under %r; the start-up shim did not run, "
                "so this KEEP cannot be judged and the gate stays shut",
                workspace or "<no workspace>",
            )
            return False
        if not evidence.exported:
            log.info(
                "export gate: KEEP did not export (compiled=%d loaded=%d invoked=0 across %d process(es))",
                evidence.compiled,
                evidence.loaded,
                evidence.processes,
            )
        return evidence.exported

    def _record_inert_source_layer(self, result: Mapping[str, Any]) -> bool:
        """Record the source an inert KEEP committed, leaving the stack and ``current_best`` alone.

        Later explore entries switch the code on, and the export refuses a tree no snapshot
        covers, so the layer is kept beside the stack. It is not an adoption: on the stack it
        would count as an unvalidated KEEP and be published as one. Only ``custom`` is handled
        -- it is the one framework whose export reads this layer.
        """
        if not result.get("source_snapshot") or str(self.shared_state.framework or "").strip().lower() != "custom":
            return False
        specialist = str(result.get("specialist_task_id") or "")
        self.shared_state.inert_source_layers.append(
            {
                "variant_name": f"{specialist}:inert" if specialist else "inert",
                "specialist_task_id": specialist,
                "after_stack_len": len(self.shared_state.optimization_stack or []),
                "ts": datetime.now(timezone.utc).isoformat(),
                "scope": _SOURCE_PATCH,
                **_source_layer_handles(result),
            }
        )
        return True

    def _record_authored_levers(self, result: Mapping[str, Any]) -> bool:
        levers = result.get("framework_levers")
        if not (isinstance(levers, list) and levers):
            return False
        lever_outcome = str(result.get("framework_lever_outcome") or "")
        return bool(
            self.shared_state.record_authored_framework_levers(
                levers,
                default_on=(lever_outcome == "default_on"),
                source_layer=self._dormant_source_layer(result) if result.get("status") == _KEPT_INERT else None,
            )
        )

    def _dormant_source_layer(self, result: Mapping[str, Any]) -> dict[str, Any] | None:
        """The source layer of a dormant keep, placed after every stack entry committed before it."""
        handles = _source_layer_handles(result)
        if not handles["source_snapshot"]:
            return None
        return {
            "scope": _SOURCE_PATCH,
            "name": str(result.get("specialist_task_id") or ""),
            "stack_position": len(self.shared_state.optimization_stack or []),
            **handles,
        }

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
