# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from hyperloom.common.coerce import to_float, to_str_list
from ..state.optimization_journal import (
    Journal,
    JournalEntry,
    OUTCOME_KEEP,
    OUTCOME_NO_PROMOTE,
    OUTCOME_REVERT,
    classify_change_kind,
    derive_journal_outcome,
    operation_kind_for,
    summarize_change,
)
from ..actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from ..state.shared_state import SharedState, resolve_grading_anchor_tput
from hyperloom.inference_optimizer.protocol.intent import Intent
from ..bus.message_bus import Message
from .coordinator_helpers import (
    _MIN_KERNEL_ENGAGED_GAIN_PCT,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _merge_cumulative_extra_server_args,
    _parse_baseline_workload_extra,
    _geak_result_has_material,
    _geak_revalidation_decision,
    _geak_sweep_measured_tput,
    _normalize_geak_overlay_dir,
    _scrape_resolved_launch_flags,
    _split_env_and_flags,
)
from ..policy.gate import (
    PolicyDenied,
)
from ..state.task_registry import Task
from ..actions.executors.benchmark_result import is_valid_measurement
from ..actions.executors._accuracy_gate import (
    BASELINE_EVAL_ACCURACY_FLOOR_KEY,
    BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILED_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    EVAL_KIND_ACCURACY_UNAVAILABLE,
    accuracy_meets_floor,
)

from .coordinator import (
    _AUDIT_ACTIONS,
    _BASELINE_MAX_TOTAL_FAILURES,
    _DEFAULT_RESUME_DRIFT_FLOOR_PCT,
    _ENABLEMENT_MAX_STALL,
    _SEVERITY_CRASH,
    _SEVERITY_REGRESS,
    PendingProposal,
    _extract_enablement_launch_log,
)
import logging as _logging

log = _logging.getLogger(__name__)


@dataclass
class _PromoteOutcome:
    """Mutable carrier threaded through the per-kind promote handlers;
    ``early_return`` skips the shared audit/save tail (sweep / conc_sweep)."""

    changed: bool = False
    audit_decision: str | None = None
    audit_extras: dict[str, Any] = field(default_factory=dict)
    early_return: bool = False


def _predicted_gain(*sources: dict[str, Any] | None) -> float | None:
    """First non-zero ``predicted_gain_pct`` (``to_float``-parsed) across ordered sources.

    Sources are checked in order; a non-zero prediction wins. Returns ``None``
    when none carry a usable value so the journal row stays ``predicted``-free
    for unpredicted (default-grid) changes rather than recording a fake 0.
    """
    for src in sources:
        if not isinstance(src, dict):
            continue
        val = to_float(src.get("predicted_gain_pct"))
        if val is not None and val != 0.0:
            return val
    return None


class WritebackCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _emit_lifecycle(
        self,
        *,
        step: str,
        status: str,
        artifacts: dict[str, str] | None = None,
        detail: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Record + persist one operator-facing lifecycle event.

        Best-effort by design: operator-facing logging must never break the
        orchestration loop, so any failure is swallowed at debug level.

        Args:
            step: The machine step name (resolved to a human label downstream).
            status: The lifecycle status (e.g. START / END / ERROR / ENTER).
            artifacts: Optional mapping of produced artifact paths.
            detail: Optional free-text detail.
            duration_s: Optional elapsed seconds for the step.
        """
        try:
            self.shared_state.record_lifecycle_event(
                step=step,
                status=status,
                artifacts=artifacts,
                detail=detail,
                duration_s=duration_s,
            )
            # Terminal events (END/ERROR) always flush; non-terminal markers are
            # debounced by ``_lifecycle_save_min_interval_s``.
            terminal = status in ("END", "ERROR")
            now = time.monotonic()
            if terminal or (now - self._lifecycle_last_save >= self._lifecycle_save_min_interval_s):
                self.shared_state.save(self.session_dir)
                self._coord._lifecycle_last_save = now
        except Exception:  # noqa: BLE001
            log.debug(
                "Coordinator: lifecycle emit failed (step=%s status=%s)",
                step,
                status,
                exc_info=True,
            )

    async def _record_policy_denied(
        self,
        source: str,
        intent: Intent,
        denied: PolicyDenied,
        *,
        action_name: str | None = None,
    ) -> None:
        """Record a PolicyGate denial.

        Publishes a ``policy_denied`` observation and records the denial streak.
        The streak is a fact for LLM self-correction only: there is no
        auto-prune and no ``policy_loop`` stop triggered from it.

        Args:
            source (str): The agent whose intent was denied.
            intent (Intent): The denied intent.
            denied (PolicyDenied): The denial carrying rule / hint / reason.
            action_name (str | None): Explicit action name override; falls back
                to ``intent.payload['action_name']``.
        """
        # Surface every PolicyGate denial in the process log (not just the bus)
        # so security rejections are observable in ops logs.
        log.warning(
            "PolicyGate denied intent: source=%s type=%s rule=%s reason=%s",
            source,
            intent.type.value,
            denied.rule,
            str(denied),
        )
        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                source,
                "observation",
                {
                    "kind": "policy_denied",
                    "intent_type": intent.type.value,
                    "rule": denied.rule,
                    "hint": denied.hint,
                    "reason": str(denied),
                },
                priority=0,
            )
        )
        resolved_action = action_name or str((intent.payload or {}).get("action_name") or "")
        # Streak counter is a fact for LLM self-correction only; the system does not auto-prune or stop on it.
        self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            tick=int(self.shared_state.tick or 0),
            intent_payload=intent.payload,
        )

    async def _record_observation(self, source: str, topic: str, payload: dict) -> None:
        """Append a broadcast observation message to the bus.

        Args:
            source (str): The agent recording the observation.
            topic (str): The bus topic to publish under.
            payload (dict): The observation payload.
        """
        await self.bus.append_and_seq(Message.new(source, "*", topic, payload))

    def _record_kernel_opt_partial(self, result: dict[str, Any]) -> None:
        """Streaming callback for ``_run_optimization_batch`` sub-attempts: write each per-kernel entry to kernel_opt_attempts immediately so the next-tick prompt is accurate mid-batch.

        Args:
            result: One sub-attempt's per-kernel result dict.
        """
        try:
            self.shared_state.record_kernel_opt(result)
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            # Never let a per-sub-attempt hiccup poison the gather.
            log.exception(
                "_record_kernel_opt_partial failed for kernel_id=%s",
                (result or {}).get("kernel_id") if isinstance(result, dict) else None,
            )

    def _update_cumulative_gain_validated(self, new_tput: float) -> None:
        """Update cumulative_gain_validated, its timestamp, and stack-length watermark.

        Call only when ``baseline_tput > 0`` and ``new_tput`` is a positive
        measured throughput.  The caller remains responsible for any surrounding
        guard (e.g. ``if self.shared_state.baseline_tput > 0``).

        Args:
            new_tput: The newly measured throughput to promote as the validated
                gain anchor.
        """
        validated_gain = (float(new_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
        self.shared_state.cumulative_gain_validated = float(validated_gain)
        self.shared_state.cumulative_gain_validated_ts = datetime.now(timezone.utc).isoformat()
        self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)

    async def _record_integrate_keep(self, result: dict[str, Any]) -> None:
        """Promote a kernel integrate KEEP into the optimization stack.

        Appends a deduped ``integrate`` entry to the optimization stack, mirrors
        the gain into the per-entry gain ledger, updates ``current_best`` and
        ``cumulative_gain`` / ``cumulative_gain_validated``, and fires a
        watermark roofline when the gain crosses the threshold. No-op when the
        result lacks a positive ``new_tput``.

        Args:
            result (dict[str, Any]): The integrate-patch executor result.
        """
        new_tput = result.get("new_tput")
        if not isinstance(new_tput, (int, float)) or new_tput <= 0:
            return
        cb = self.shared_state.current_best or {}
        extra_args = (
            str(result.get("extra_server_args") or "")
            or (str(cb.get("extra_server_args") or "") if isinstance(cb, dict) else "")
        ).strip()
        apply_result = result.get("apply_result") or {}
        backup_manifest = apply_result.get("manifest_path") if isinstance(apply_result, dict) else None
        if not backup_manifest and isinstance(apply_result, dict):
            stack_applies = apply_result.get("stack_apply_results")
            if isinstance(stack_applies, list):
                for applied in stack_applies:
                    if isinstance(applied, dict) and applied.get("manifest_path"):
                        backup_manifest = applied.get("manifest_path")
                        break
        entry = {
            "action": "integrate",
            "source_phase": str(getattr(self.shared_state, "phase", "") or "KERNEL_AGENT"),
            "integration_id": result.get("integration_id"),
            "kernel_id": result.get("kernel_id"),
            "task_group_key": result.get("task_group_key"),
            "identity_route": result.get("identity_route"),
            "patch_path": result.get("patch_path"),
            "target_file": result.get("target_file"),
            "backup_manifest": backup_manifest,
            "gain_pct": result.get("gain_pct"),
            "tput": float(new_tput),
            "workspace": result.get("workspace"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        stack_kernel_ids = result.get("stack_kernel_ids")
        if isinstance(stack_kernel_ids, list) and stack_kernel_ids:
            entry["stack_kernel_ids"] = [str(kid) for kid in stack_kernel_ids if str(kid)]
        integrate_gap_cid = str(result.get("gap_canonical_id") or "").strip()
        if integrate_gap_cid:
            entry["gap_canonical_id"] = integrate_gap_cid
        key = (
            entry.get("integration_id"),
            entry["kernel_id"],
            entry["patch_path"],
            entry["target_file"],
        )
        existing = {
            (
                item.get("integration_id"),
                item.get("kernel_id"),
                item.get("patch_path"),
                item.get("target_file"),
            )
            for item in self.shared_state.optimization_stack
            if isinstance(item, dict) and item.get("action") == "integrate"
        }
        if key not in existing:
            self.shared_state.optimization_stack.append(entry)
            # Mirror into gain_per_stack_entry so breakdown attribution works without re-walking the event log.
            self.shared_state.append_stack_gain_entry(
                action="integrate",
                variant_name=entry.get("kernel_id"),
                new_tput=new_tput,
                extra_server_args=extra_args,
                ts=entry["ts"],
            )

        self.shared_state.current_best = {
            "action": "integrate",
            "tput": float(new_tput),
            "integration_id": result.get("integration_id"),
            "kernel_id": result.get("kernel_id"),
            "extra_server_args": extra_args,
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": result.get("ttft_mean_ms"),
            "e2el_mean_ms": result.get("e2el_mean_ms"),
            "tpot_mean_ms": result.get("tpot_mean_ms"),
            "workspace": result.get("workspace"),
        }
        if self.shared_state.baseline_tput > 0:
            self.shared_state.cumulative_gain = (
                (float(new_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
            )
            # Integrate KEEP is already rebench-validated: promote into cumulative_gain_validated + watermark.
            self._update_cumulative_gain_validated(new_tput)
            await self._maybe_enqueue_watermark_roofline(
                reason="integrate_keep_watermark",
            )

    def _is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
        """Decide whether a settled task result should be promoted.

        Per-kind rules: baseline/profile require a valid measurement, sweep
        requires ``status == "succeeded"``, ``replay_warm_recipe`` always routes
        through promotion (it owns its own failure bookkeeping), and everything
        else is promotable unless ``status == "failed"``.

        Args:
            task_kind (str): The task's kind.
            result (dict[str, Any]): The task result payload.

        Returns:
            bool: ``True`` when the result should go through
                :meth:`_promote_to_shared_state`.
        """
        if not isinstance(result, dict):
            return False
        if task_kind == "baseline":
            # A baseline whose accuracy eval failed measured throughput but must
            # not anchor; route it to _handle_unpromotable_result for enablement.
            if bool(result.get("baseline_eval_failed")):
                return False
            return is_valid_measurement(result)
        if task_kind == "profile":
            return is_valid_measurement(result)
        if task_kind == "sweep":
            return result.get("status") == "succeeded"
        # replay_warm_recipe always routes through _promote_warm_replay (owns its own failure bookkeeping).
        if task_kind == "replay_warm_recipe":
            return True
        return result.get("status") != "failed"

    def _record_intervention_for_task(
        self,
        task: "Task",
        result: Any,
    ) -> None:
        """Log a completed task's change_type into SharedState.intervention_mix (explore → config; integrate_patch → code_patch_attempt or code_patch when kept). Best-effort.

        Args:
            task: The completed task whose kind selects the intervention class.
            result: The task result dict; non-dict results are ignored.
        """
        if not isinstance(result, dict):
            return
        kind = (task.kind or "").strip()
        if kind == "explore":
            # Winner surrogate: result.winners present OR best_variant set.
            winners = result.get("winners") or []
            best = result.get("best_variant")
            if not winners and not best:
                # An explore round that KEPT nothing still counts as a config-only attempt.
                self.shared_state.record_intervention(
                    change_type="config_attempt",
                    action="explore",
                    task_id=task.task_id,
                    delta_pct=None,
                )
                return
            delta_pct = None
            if isinstance(best, dict):
                delta_pct = best.get("gain_pct")
            self.shared_state.record_intervention(
                change_type="config",
                action="explore",
                task_id=task.task_id,
                delta_pct=delta_pct if isinstance(delta_pct, (int, float)) else None,
            )
            return
        if kind == "integrate_patch":
            status = str(result.get("status") or "").strip().lower()
            if not status:
                return
            if status != "kept":
                self.shared_state.record_intervention(
                    change_type="code_patch_attempt",
                    action="integrate_patch",
                    task_id=task.task_id,
                    delta_pct=result.get("delta_pct"),
                )
                return
            self.shared_state.record_intervention(
                change_type="code_patch",
                action="integrate_patch",
                task_id=task.task_id,
                delta_pct=result.get("delta_pct"),
            )

    def _persist_eval_failure(self, result_payload: dict[str, Any]) -> None:
        """Persist an eval-rooted baseline failure so enablement can re-run it.

        Records origin, floor, probe config, contract fingerprint, evidence,
        kind and observed accuracy, and seeds ``enablement_launch_log`` so the
        FRAMEWORK pump dispatches even when the failure carries no boot log.

        A run that never executed the eval characterizes nothing: it reports
        ``accuracy_unavailable`` with no task/metric/source. Such a run must
        still register as a failed round (origin / pending / stall accounting
        below), but it must NOT overwrite a stored trigger that actually
        measured an accuracy -- otherwise an eval-less re-baseline downgrades
        real ``accuracy_below_floor`` evidence to an empty
        ``accuracy_unavailable`` and the next enablement attempt loses the
        measurement it is supposed to reproduce. Contract fingerprints cannot
        gate this: ``RUN_EVAL`` is itself a contract field, so the eval-less
        run's fingerprint never matches the measured one.
        """
        state = self.shared_state
        was_validation_pending = bool(getattr(state, "enablement_validation_pending", False))
        incoming_kind = result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY)
        measured_incoming = to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY)) is not None
        stored_kind = str(getattr(state, "enablement_baseline_eval_kind", "") or "")
        preserve_measured_trigger = (
            incoming_kind == EVAL_KIND_ACCURACY_UNAVAILABLE
            and not measured_incoming
            and bool(stored_kind)
            and stored_kind != EVAL_KIND_ACCURACY_UNAVAILABLE
        )
        state.enablement_origin = "eval"
        state.enablement_pending = True
        # A failed revalidation reopens the authoring loop and counts as a
        # no-progress round so the enablement_stalled cap can still terminate.
        if was_validation_pending:
            state.enablement_validation_pending = False
            state.enablement_stall_streak = int(getattr(state, "enablement_stall_streak", 0) or 0) + 1
            if state.enablement_stall_streak >= _ENABLEMENT_MAX_STALL and not state.stop_reason:
                state.set_stop_reason("enablement_stalled")
        floor = to_float(result_payload.get(BASELINE_EVAL_ACCURACY_FLOOR_KEY))
        if floor is not None:
            state.enablement_accuracy_floor = float(floor)
        if preserve_measured_trigger:
            log.info(
                "enablement: keeping measured trigger kind=%s (accuracy=%s task=%s); "
                "an eval-less baseline reported accuracy_unavailable and must not "
                "overwrite it",
                stored_kind,
                getattr(state, "enablement_observed_accuracy", None),
                getattr(state, "enablement_observed_task", ""),
            )
            return
        cfg = result_payload.get("materialized_config")
        if isinstance(cfg, str) and cfg:
            state.enablement_probe_config_path = cfg
        fp = result_payload.get(BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY)
        if isinstance(fp, str) and fp:
            state.enablement_eval_contract_fingerprint = fp
        if isinstance(incoming_kind, str) and incoming_kind:
            state.enablement_baseline_eval_kind = incoming_kind
        observed = to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY))
        if observed is not None:
            state.enablement_observed_accuracy = float(observed)
        state.enablement_observed_task = str(result_payload.get("accuracy_task") or "")
        state.enablement_observed_metric = str(result_payload.get("accuracy_metric") or "")
        evidence = str(result_payload.get(BASELINE_EVAL_EVIDENCE_KEY) or "")
        if evidence:
            state.enablement_baseline_eval_evidence = evidence[:4000]
            state.enablement_launch_log = evidence

    async def _handle_unpromotable_result(
        self,
        task: Task,
        result: dict[str, Any] | None,
    ) -> None:
        """Record a failed / unpromotable task result into SharedState: append to last_action_failures (+ a failed attempts row for _AUDIT_ACTIONS) and apply the baseline failure_streak/stop_reason gates.

        Args:
            task: The failed/unpromotable task.
            result: The task result payload; ``None`` is treated as an empty
                result.
        """
        result_payload = dict(result or {})
        if task.kind == "conc_sweep" and not result_payload.get("status"):
            result_payload["status"] = "failed"
        if task.kind in {"framework_agent", "conc_sweep", "replay_warm_recipe", "integrate_patch"}:
            try:
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                result_payload.setdefault(
                    "workload",
                    {
                        "framework": str(getattr(self.shared_state, "framework", "") or ""),
                        "model_name": str(getattr(self.shared_state, "model_name", "") or ""),
                        "gpu_type": str(getattr(self.shared_state, "gpu_type", "") or ""),
                        "precision": str(getattr(self.shared_state, "precision", "") or ""),
                        "tp": int(getattr(self.shared_state, "tp", 0) or 0),
                        "conc": int(getattr(self.shared_state, "conc", 0) or 0),
                        "isl": int(getattr(self.shared_state, "isl", 0) or 0),
                        "osl": int(getattr(self.shared_state, "osl", 0) or 0),
                    },
                )
                instrument.record_action_operation(
                    self.session_dir,
                    action=task.kind,
                    task_id=task.task_id,
                    status="failed",
                    decision="discarded",
                    result=result_payload,
                    phase=str(getattr(self.shared_state, "phase", "") or ""),
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    tick=int(getattr(self.shared_state, "tick", 0) or 0),
                )
            except Exception:  # noqa: BLE001
                log.debug("v4 action failure capture failed", exc_info=True)
        any_changed = False
        params = task.params or {}
        if task.kind == "explore" and bool(params.get("geak_fallback")):
            pending = getattr(self.shared_state, "geak_pending", None) or {}
            pending_task_id = str(pending.get("revalidation_task_id") or "") if isinstance(pending, dict) else ""
            if not pending_task_id or pending_task_id == task.task_id:
                geak_result = (
                    dict(self.shared_state.geak_result)
                    if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                    else {}
                )
                geak_result["revalidation_status"] = "failed"
                geak_result["revalidation_error_class"] = str(result_payload.get("error_class") or "")
                geak_result["revalidation_error"] = str(
                    result_payload.get("error") or result_payload.get("reason") or ""
                )[:500]
                self.shared_state.geak_result = geak_result
                self.shared_state.geak_pending = {}
                self.shared_state.resume_pending_revalidation = False
                any_changed = True
        # Per-action audit (failed attempt) for the in-scope kinds.
        if task.kind in _AUDIT_ACTIONS:
            audit_extras: dict[str, Any] = {}
            # Stamp baseline-params fingerprint for the self-loop denial helper.
            if task.kind == "baseline":
                audit_extras["fingerprint"] = _baseline_params_fingerprint(task.params)
            self.shared_state.record_action_attempt(
                action=task.kind,
                task_id=task.task_id,
                status="failed",
                decision="no_promote",
                result=result_payload,
                extras=audit_extras,
            )
            any_changed = True
        # Global rolling failure log (every kind, including kernel_agent-owned).
        self.shared_state.record_action_failure(
            action=task.kind,
            task_id=task.task_id,
            result=result_payload,
        )
        any_changed = True
        if task.kind == "conc_sweep":
            self.shared_state.record_action_attempt(
                action="conc_sweep",
                task_id=task.task_id,
                status=str(result_payload.get("status") or "failed"),
                decision="discarded",
                result=result_payload,
                extras={
                    "was_skipped": bool(result_payload.get("was_skipped", False)),
                    "skip_reason": result_payload.get("skip_reason"),
                    "budget_exhausted": bool(result_payload.get("budget_exhausted", False)),
                    "total_budget_sec": result_payload.get("total_budget_sec"),
                    "elapsed_sec": result_payload.get("elapsed_sec"),
                    "best_speedup": ((result_payload.get("summary") or {}).get("best_speedup")),
                    "best_conc": ((result_payload.get("summary") or {}).get("best_conc")),
                    "successful_pairs": ((result_payload.get("summary") or {}).get("successful_pairs")),
                    "report_path": result_payload.get("report_json_path"),
                },
            )
            self.shared_state.record_conc_sweep(result_payload)
        # A framework_agent task that settles failed/empty never reaches the
        # promote branch that writes the terminal progress row; stamp
        # no_result_failed so the pump does not re-select it every tick.
        if task.kind == "framework_agent":
            cand = (task.params or {}).get("candidate")
            cand_id = self._framework_candidate_key(cand if isinstance(cand, dict) else None)
            if cand_id:
                self._stamp_framework_progress(
                    candidate_id=cand_id,
                    batch_id=str((task.params or {}).get("batch_id") or ""),
                    status="no_result_failed",
                    kept=False,
                    rationale=str(result_payload.get("reason") or result_payload.get("error") or "")[:500],
                    provenance="executor",
                    extra={"status": str(result_payload.get("status") or "")},
                )
        # Baseline-specific gates: streak counter + stop_reason + baseline_not_promoted event.
        # Fast arg errors get their own streak so they don't burn the
        # slow-baseline retry budget on deterministic failures.
        baseline_event_payload: dict[str, Any] | None = None
        # Only arm/streak while no baseline has succeeded yet (tput <= 0).
        if task.kind == "baseline" and self.shared_state.baseline_tput <= 0:
            err_class = result_payload.get("error_class", "")
            # While a serial enablement is actively engaged, baseline boots
            # re-fail on purpose (each round clears a deeper gap), so the
            # ``baseline_failed`` fast-fail must NOT fire here; the
            # ``enablement_stalled`` cap is the correct fast-fail instead.
            # ``fast_exit_arg_error`` stays gated on its own streak regardless.
            from ..phases.machine_state import enablement_engaged as _enablement_engaged  # noqa: PLC0415

            enablement_engaged = _enablement_engaged(self.shared_state)
            eval_failed = bool(result_payload.get(BASELINE_EVAL_FAILED_KEY))
            # Revalidation task failed for any reason (boot/OOM/timeout/eval): clear
            # pending state, preserve the frozen trigger identity, increment stall.
            reval_tid = str(getattr(self.shared_state, "enablement_revalidation_task_id", "") or "").strip()
            is_revalidation = bool(
                (task.params or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON
                or (reval_tid and reval_tid == str(task.task_id or ""))
            )
            if is_revalidation and bool(getattr(self.shared_state, "enablement_validation_pending", False)):
                self.shared_state.enablement_validation_pending = False
                self.shared_state.enablement_revalidation_task_id = ""
                self.shared_state.enablement_stall_streak = (
                    int(getattr(self.shared_state, "enablement_stall_streak", 0) or 0) + 1
                )
                try:
                    from ..phases.framework import _ENABLEMENT_MAX_STALL as _max_stall
                except ImportError:
                    _max_stall = 5
                if self.shared_state.enablement_stall_streak >= _max_stall and not self.shared_state.stop_reason:
                    self.shared_state.set_stop_reason("enablement_stalled")
                else:
                    self.shared_state.enablement_dispatched = False
                launch_log = _extract_enablement_launch_log(result_payload)
                if launch_log:
                    self.shared_state.enablement_launch_log = launch_log
                log.warning(
                    "enablement revalidation task %s failed (error_class=%s); "
                    "stall_streak=%d rearm=%s",
                    task.task_id,
                    err_class,
                    self.shared_state.enablement_stall_streak,
                    not bool(self.shared_state.stop_reason),
                )
                any_changed = True
            from ..actions.executors._accuracy_gate import eval_enablement_allowed  # noqa: PLC0415
            from ..actions.executors._multi_node_env import is_multi_node  # noqa: PLC0415

            # Single-node eval-pending failure: throughput measured fine and the
            # eval is expected to re-run under enablement, so do not spend the
            # baseline_failed budget yet. Multi-node keeps the strict backstop,
            # and so does a session that never admitted the eval lane — nothing
            # would re-run the eval, so holding the budget just stalls the run.
            eval_pending_suppress = (
                eval_failed and not is_multi_node() and eval_enablement_allowed(self.shared_state)
            )
            if eval_failed:
                self._persist_eval_failure(result_payload)
            if err_class == "fast_exit_arg_error":
                self.shared_state.baseline_arg_error_streak += 1
                if self.shared_state.baseline_arg_error_streak >= 2:
                    self.shared_state.set_stop_reason("baseline_arg_error")
            else:
                self.shared_state.baseline_failure_streak += 1
                self.shared_state.baseline_arg_error_streak = 0
                if self.shared_state.baseline_failure_streak >= 3 and not enablement_engaged and not eval_pending_suppress:
                    self.shared_state.set_stop_reason("baseline_failed")
            # Combined backstop: count ALL baseline failures so mixed
            # error_classes that split the per-class streaks still fast-fail.
            self.shared_state.baseline_total_failures += 1
            if (
                self.shared_state.baseline_total_failures >= _BASELINE_MAX_TOTAL_FAILURES
                and not self.shared_state.stop_reason
                and not enablement_engaged
                and not eval_pending_suppress
            ):
                self.shared_state.set_stop_reason("baseline_failed")
            # One-shot eager fallback: a (non-OOM) cuda-graph capture failure is
            # often recoverable by disabling cuda-graph capture.
            if err_class == "cuda_graph_capture_failed" and not self.shared_state.baseline_eager_fallback:
                self.shared_state.baseline_eager_fallback = True
                log.warning(
                    "baseline %s hit cuda-graph capture failure; arming "
                    "disable-cuda-graph fallback for the next baseline retry",
                    task.task_id,
                )
            # Stash the launch/traceback text for the FRAMEWORK pump (fast arg errors excluded).
            if err_class != "fast_exit_arg_error":
                launch_log = _extract_enablement_launch_log(result_payload)
                if launch_log:
                    self.shared_state.enablement_launch_log = launch_log
            baseline_event_payload = {
                "kind": "baseline_not_promoted",
                "task_id": task.task_id,
                "failure_streak": self.shared_state.baseline_failure_streak,
                "arg_error_streak": self.shared_state.baseline_arg_error_streak,
                "stop_reason": self.shared_state.stop_reason,
                "result_status": result_payload.get("status"),
                "error_class": err_class,
                "baseline_eval_failed": eval_failed,
            }
            any_changed = True
        # Mirror the promote-path roofline failure handling: bump streak, clear gate, warn.
        if task.kind == "roofline":
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak += 1
            if self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
            any_changed = True
            log.warning(
                "Auto-roofline %s failed (reason=%s phase=%s "
                "error_class=%s); continuing in degraded mode "
                "(specialists / explore proceed without a fresh "
                "analysis_md). No retry, no fallback.",
                task.task_id,
                str((task.params or {}).get("reason") or ""),
                result_payload.get("phase"),
                result_payload.get("error_class"),
            )
        if any_changed:
            self.shared_state.save(self.session_dir)
        if baseline_event_payload is not None:
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "event",
                    baseline_event_payload,
                )
            )

    def _source_session_id(self) -> str:
        """Return the hyperloom-local session id used as source_session_id on KB fact writes.

        NOT a KB-side session id; prefers recipe_kb_session_id, falls back to session_dir.name.

        Returns:
            The hyperloom-local session id (recipe_kb_session_id when set, else
            ``session_dir.name``).
        """
        return str(getattr(self.shared_state, "recipe_kb_session_id", "") or "") or self.session_dir.name

    async def _fact_write_hook(
        self,
        *,
        task: "Task",
        result: Any,
        kept: bool,
    ) -> None:
        """Per-task fact-write entry point (per_variant for explore grids, else per-task); best-effort, never raises.

        Args:
            task: The completed task being recorded.
            result: The task's :class:`SubAgentResult` (or result dict).
            kept: Whether the task's result was KEEP-promoted.
        """
        result_dict = result.result if hasattr(result, "result") else (result or {})
        if not isinstance(result_dict, dict):
            result_dict = {}
        source_session_id = self._source_session_id()
        per_variant = result_dict.get("per_variant_outcomes")
        if task.kind == "explore" and isinstance(per_variant, list) and per_variant:
            for vo in per_variant:
                try:
                    self._record_fact_per_variant(
                        task=task,
                        source_session_id=source_session_id,
                        variant_outcome=vo,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "fact-write per-variant failed (task=%s)",
                        task.task_id,
                    )
        else:
            try:
                self._record_fact_per_task(
                    task=task,
                    source_session_id=source_session_id,
                    result_dict=result_dict,
                    kept=kept,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "fact-write per-task failed (task=%s)",
                    task.task_id,
                )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive; never crash on save
            log.exception("fact-write SharedState.save failed")

    def _ensure_journal(self) -> Journal:
        """Lazy-instantiate the per-session :class:`Journal` (load_or_create reads an existing file on resume).

        Returns:
            The per-session :class:`Journal` instance (created on first call,
            with the baseline backfilled on subsequent calls).
        """
        existing = getattr(self, "_journal", None)
        if existing is None:
            ss = self.shared_state
            self._coord._journal = Journal.load_or_create(
                self.session_dir,
                session_id=str(getattr(ss, "recipe_kb_session_id", "") or "")
                or str(getattr(ss, "session_id", "") or "")
                or self.session_dir.name,
                model=str(getattr(ss, "model_name", "") or ""),
                hardware=str(getattr(ss, "gpu_type", "") or ""),
                framework=str(getattr(ss, "framework", "") or ""),
                baseline_throughput=float(getattr(ss, "baseline_tput", 0.0) or 0.0),
            )
        else:
            # Backfill baseline once the baseline executor finishes.
            existing.update_baseline(float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0))
        return self._journal

    def _pitfall_severity_for(
        self,
        result_dict: dict[str, Any] | None,
    ) -> str | None:
        """Decide whether a failed result warrants a pitfall row.

        ``crash`` / ``oom`` / ``hang`` / ``detokenizer_stall`` on ``error_class``,
        or ``crash`` / ``oom`` / ``hang`` on ``status``, yield
        ``SEVERITY_CRASH``; a ``gain_pct`` at or below
        ``PITFALL_REGRESS_THRESHOLD_PCT`` (-5.0) yields ``SEVERITY_REGRESS``;
        otherwise ``None``.

        Args:
            result_dict: The failed task's result dict; non-dict yields ``None``.

        Returns:
            The pitfall severity (``SEVERITY_CRASH`` / ``SEVERITY_REGRESS``), or
            ``None`` when no pitfall is warranted.
        """
        if not isinstance(result_dict, dict):
            return None
        error_class = str(result_dict.get("error_class") or "").lower()
        # ``detokenizer_stall`` is a hang in all but name; record it as a
        # crash-severity pitfall so the offending config is not re-proposed.
        if error_class in ("crash", "oom", "hang", "detokenizer_stall"):
            return _SEVERITY_CRASH
        status = str(result_dict.get("status") or "").lower()
        if status in ("crash", "oom", "hang"):
            return _SEVERITY_CRASH
        gain = result_dict.get("gain_pct")
        try:
            gain_pct = float(gain) if gain is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        if gain_pct is not None and gain_pct <= self.PITFALL_REGRESS_THRESHOLD_PCT:
            return _SEVERITY_REGRESS
        return None

    def _journal_entry_phase(self) -> str:
        """Return the current phase label for journal entries.

        Returns:
            str: The uppercased phase name, or ``"UNKNOWN"`` when unset.
        """
        return str(getattr(self.shared_state, "phase", "") or "").strip().upper() or "UNKNOWN"

    def _record_fact_impl(
        self,
        *,
        task: "Task",
        source_session_id: str,
        is_keep: bool,
        change: str,
        gain_pct: float | None,
        throughput_after: float | None,
        best_config_candidate: dict[str, Any] | None,
        evidence_refs: list[str],
        pitfall_severity_dict: dict[str, Any],
        variant_name: str | None = None,
    ) -> None:
        """Shared KB write for _record_fact_per_task and _record_fact_per_variant.

        Writes one KB lesson (on KEEP with positive gain) or one KB pitfall
        (on REVERT/failure) to the recipe row, then returns.  Call only after
        the journal entry has been appended and ``recipe_kb`` is confirmed
        non-None by the caller.

        Args:
            task: The completed task (provides task_id).
            source_session_id: Hyperloom-local session id stamped on provenance.
            is_keep: True when the outcome is a validated KEEP.
            change: Summarized change string (used in the statement).
            gain_pct: Measured gain percentage, or ``None``.
            throughput_after: Measured throughput after the change, or ``None``.
            best_config_candidate: Pre-extracted best-config dict (differs
                between per-task and per-variant callers).
            evidence_refs: List of evidence reference strings to stamp on the
                provenance (caller builds task-only or task+variant refs).
            pitfall_severity_dict: The dict passed to ``_pitfall_severity_for``
                (per-task passes ``result_dict``; per-variant passes a merged
                metrics + outcome dict).
            variant_name: Variant name, present only for per-variant calls;
                added to ``provenance_details`` when non-None.
        """
        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        workload_tags = self._coord._collect_workload_tags()
        extra = workload_tags if workload_tags else None
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        provenance_base: dict[str, Any] = {
            "source_session_id": source_session_id,
            "source_task_id": task.task_id,
            "evidence": list(evidence_refs or []),
            "applicable_models": list(models or []),
            "applicable_hardware": list(hardware or []),
            "extra": dict(extra or {}),
            "now": now_iso,
        }
        if variant_name is not None:
            provenance_base["source_variant_name"] = variant_name

        if is_keep and gain_pct is not None and gain_pct > 0:
            statement = self._coord._build_statement(
                change=change,
                gain_pct=gain_pct,
                kind="lesson",
            )
            impact = self._coord._build_measured_impact(
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                stack_depth=len(getattr(self.shared_state, "optimization_stack", []) or []),
                measured_at=now_iso,
            )
            live = self._read_local_recipe_row()
            recipe_overrides = self._kb_best_config_overrides_for_keep(
                live=live,
                best_config_candidate=best_config_candidate,
                throughput_after=throughput_after,
            )
            self._kb_amend_recipe(
                append_lesson={
                    "statement": statement,
                    "measured_impact": impact,
                },
                recipe_overrides=recipe_overrides or None,
                provenance_details=provenance_base,
            )
            return

        severity = self._pitfall_severity_for(pitfall_severity_dict)
        if severity is not None:
            description = self._coord._build_statement(
                change=change,
                severity=severity,
                kind="pitfall",
            )
            self._kb_amend_recipe(
                append_pitfall={
                    "description": description,
                    "severity": severity,
                },
                provenance_details=provenance_base,
            )

    def _record_fact_per_task(
        self,
        *,
        task: "Task",
        source_session_id: str,
        result_dict: dict[str, Any],
        kept: bool,
    ) -> None:
        """Per-task fact write — one journal row + maybe one KB fact (source_session_id is hyperloom-local).

        Args:
            task: The completed task being recorded.
            source_session_id: The hyperloom-local session id stamped on the
                fact provenance.
            result_dict: The task result dict.
            kept: Whether the result was KEEP-promoted (KEEP → lesson, else
                pitfall/REVERT).
        """
        journal = self._ensure_journal()
        # integrate_patch / framework_agent report their delta under ``delta_pct``;
        # fall back to it so a reverted/kept patch shows its REAL measured delta
        # in the journal instead of a null gain.
        gain_pct = to_float(result_dict.get("gain_pct"))
        if gain_pct is None:
            gain_pct = to_float(result_dict.get("delta_pct"))
        throughput_after = to_float(result_dict.get("output_throughput"))
        kind = classify_change_kind(task.kind, None)
        change = summarize_change(task.kind, None, result_dict)
        # Journal outcome follows the executor's per-status verdict for source-
        # patch kinds (a ``reverted`` patch is promotable but NOT a KEEP); other
        # kinds keep the binary promotable→KEEP behaviour. See
        # ``derive_journal_outcome`` (fixes the "fake KEEP" bug).
        outcome = derive_journal_outcome(task.kind, result_dict, promotable=kept)
        is_keep = outcome == OUTCOME_KEEP
        if is_keep:
            error_class = None
            reason = None
        else:
            error_class = str(result_dict.get("error_class") or "") or None
            reason = str(result_dict.get("reason") or "") or None
        journal.append_entry(
            JournalEntry(
                phase=self._journal_entry_phase(),
                iter=int(self.shared_state.tick or 0),
                kind=kind,
                change=change,
                outcome=outcome,
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                error_class=error_class,
                reason=reason,
                task_id=task.task_id,
                tick=int(self.shared_state.tick or 0),
                predicted_gain_pct=_predicted_gain(
                    result_dict,
                    getattr(task, "params", None),
                ),
            )
        )

        if self.recipe_kb is None:
            return

        self._record_fact_impl(
            task=task,
            source_session_id=source_session_id,
            is_keep=is_keep,
            change=change,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            best_config_candidate=self._extract_kept_best_config(
                task=task,
                result_dict=result_dict,
            ),
            # evidence_refs (log:task-...) gives traceability since source_session_id lands in attrs.
            evidence_refs=[f"log:task-{task.task_id}"],
            pitfall_severity_dict=result_dict,
        )

    def _build_statement(
        self,
        *,
        change: str,
        kind: str,
        gain_pct: float | None = None,  # kept for backward call-signature compat
        severity: str | None = None,
    ) -> str:
        """Build the lesson statement / pitfall description hashed into the KB canonical_id; MUST exclude volatile fields (e.g. gain_pct) so N sessions merge instead of producing N rows. Identity = framework + change + model/hw.

        Args:
            change: The summarized change description.
            kind: ``"lesson"`` or ``"pitfall"`` — selects the rendered form.
            gain_pct: Kept for backward call-signature compat; intentionally not
                included in the statement.
            severity: The pitfall severity, rendered only when ``kind`` is
                ``"pitfall"``.

        Returns:
            The identity-stable statement / description string.
        """
        framework = str(getattr(self.shared_state, "framework", "") or "").strip()
        fw_tag = f"[{framework or '?'}] "
        model = self.shared_state.model_name or "?"
        hw = self.shared_state.gpu_type or "?"
        if kind == "lesson":
            # gain_pct intentionally NOT included — see docstring.
            return f"{fw_tag}{change} on {model}/{hw}"
        # kind == "pitfall"
        return f"{fw_tag}{change} → {severity or '?'} on {model}/{hw}"

    @staticmethod
    def _build_measured_impact(
        *,
        gain_pct: float | None,
        throughput_after: float | None,
        stack_depth: int,
        measured_at: str,
    ) -> dict[str, Any]:
        """Structured ``measured_impact`` payload (dict not legacy string so consumers parse without regex); stack_depth = stack length before this lesson lands.

        Args:
            gain_pct: The measured gain percent, or ``None``.
            throughput_after: Throughput after the change, or ``None``.
            stack_depth: Optimization-stack length before this lesson lands.
            measured_at: ISO timestamp of the measurement.

        Returns:
            A compact ``measured_impact`` dict with ``None`` fields stripped.
        """
        out: dict[str, Any] = {
            "gain_pct": float(gain_pct) if gain_pct is not None else None,
            "stack_depth_at_apply": int(stack_depth),
            "measured_at": measured_at,
        }
        if throughput_after is not None:
            out["throughput_after"] = float(throughput_after)
        # Strip None for compactness (prompt section uses .get).
        return {k: v for k, v in out.items() if v is not None}

    def _record_fact_per_variant(
        self,
        *,
        task: "Task",
        source_session_id: str,
        variant_outcome: dict[str, Any],
    ) -> None:
        """Per-variant fact write — mirror of _record_fact_per_task for explore per-variant decisions.

        Args:
            task: The completed explore task.
            source_session_id: The hyperloom-local session id stamped on the
                fact provenance.
            variant_outcome: One per-variant outcome row (name, outcome,
                metrics).
        """
        journal = self._ensure_journal()
        outcome_raw = str(variant_outcome.get("outcome") or "")
        if outcome_raw == "KEEP":
            outcome = OUTCOME_KEEP
        elif outcome_raw in ("REVERT", "FAILED", "KEEP_UNSTABLE"):
            outcome = OUTCOME_REVERT
        elif outcome_raw == "SKIPPED_DEDUP":
            return  # nothing to journal
        else:
            outcome = OUTCOME_NO_PROMOTE
        variant_name = str(variant_outcome.get("variant_name") or "")
        metrics = variant_outcome.get("metrics") or {}
        gain_pct = to_float(metrics.get("gain_pct") if isinstance(metrics, dict) else None)
        throughput_after = to_float(metrics.get("output_throughput") if isinstance(metrics, dict) else None)
        variant_attrs = variant_outcome.get("variant") or {}
        kind = classify_change_kind(
            task.kind,
            variant_attrs if isinstance(variant_attrs, dict) else None,
        )
        # Ensure the change summary is variant-specific (else every explore variant writes an identical row).
        change_attrs = dict(variant_attrs) if isinstance(variant_attrs, dict) else {}
        if (
            not (change_attrs.get("extra_server_args") or change_attrs.get("extra_envs") or change_attrs.get("name"))
            and variant_name
        ):
            change_attrs["name"] = variant_name
        change = summarize_change(task.kind, change_attrs, None)
        error_class = None
        reason = None
        if outcome == OUTCOME_REVERT:
            error_class = str(variant_outcome.get("error_class") or "") or None
            reason = str(variant_outcome.get("reason") or "") or None
        # Proposer attribution + per-variant measurement detail, carried from the
        # explore executor's per_variant_outcomes so the decision row records who
        # proposed the change and how it measured (beyond headline gain/tput).
        detail_metrics = {
            k: metrics[k]
            for k in (
                "runtime_sec",
                "wall_clock_ratio_vs_baseline",
                "stack_rebench_tput",
                "estimated_output_throughput",
            )
            if isinstance(metrics, dict) and metrics.get(k) is not None
        }
        journal.append_entry(
            JournalEntry(
                phase=self._journal_entry_phase(),
                iter=int(self.shared_state.tick or 0),
                kind=kind,
                change=change,
                outcome=outcome,
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                error_class=error_class,
                reason=reason,
                task_id=task.task_id,
                variant_name=variant_name,
                provenance=str(variant_outcome.get("provenance") or ""),
                scope=str(variant_outcome.get("scope") or ""),
                fingerprint=str(variant_outcome.get("fingerprint") or ""),
                metrics=detail_metrics,
                tick=int(self.shared_state.tick or 0),
                predicted_gain_pct=_predicted_gain(
                    variant_outcome,
                    variant_attrs if isinstance(variant_attrs, dict) else None,
                    getattr(task, "params", None),
                ),
            )
        )

        if self.recipe_kb is None:
            return

        self._record_fact_impl(
            task=task,
            source_session_id=source_session_id,
            is_keep=(outcome == OUTCOME_KEEP),
            change=change,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            best_config_candidate=self._extract_kept_best_config(
                task=task,
                variant_attrs=change_attrs,
            ),
            # Workload-shape tags — see _record_fact_per_task.
            evidence_refs=[f"log:task-{task.task_id}", f"variant:{variant_name}"],
            pitfall_severity_dict={
                **(metrics if isinstance(metrics, dict) else {}),
                "error_class": variant_outcome.get("error_class"),
                "status": variant_outcome.get("outcome"),
            },
            variant_name=variant_name,
        )

    def _collect_workload_tags(self) -> dict[str, Any]:
        """Return the workload-shape KB tag dict for the current session; shared by recipe attrs + lesson/pitfall writes so the warm-start reader filters symmetrically.

        Returns:
            A dict of workload-shape KB tags (framework, model, parallelism,
            runtime versions, baseline workload extras) with empty values
            omitted.
        """
        ss = self.shared_state
        out: dict[str, Any] = {}
        framework = str(getattr(ss, "framework", "") or "").strip()
        if framework:
            out["framework"] = framework
        model_class = str(getattr(ss, "model_class", "") or "").strip()
        if model_class:
            out["model_class"] = model_class
        # model_family (v1 fallback) no longer stamped: v2 uses the exact 5-tuple canonical_id.
        model_name = str(getattr(ss, "model_name", "") or "").strip()
        if model_name:
            out["model_name"] = model_name
        for src_attr, dst_key in (
            ("precision", "precision"),
            ("tp", "tp"),
            ("ep", "ep"),
            ("conc", "conc"),
            ("isl", "isl"),
            ("osl", "osl"),
            ("max_model_len", "max_model_len"),
        ):
            v = getattr(ss, src_attr, None)
            if v not in (None, "", 0):
                out[dst_key] = v
        # EP env fallback when SharedState.ep is unset (legacy SDK callers).
        if "ep" not in out:
            raw_ep = (os.environ.get("EP") or "").strip()
            try:
                n = int(raw_ep) if raw_ep else 0
            except ValueError:
                n = 0
            if n > 0:
                out["ep"] = n
        # PP — no SharedState field (no CLI surface); env-only.
        raw_pp = (os.environ.get("PP") or "").strip()
        try:
            pp_n = int(raw_pp) if raw_pp else 0
        except ValueError:
            pp_n = 0
        if pp_n > 0:
            out["pp"] = pp_n
        # runtime version tags from stack_fingerprint_meta (cli writes at boot, resume reads verbatim).
        fp_meta = getattr(ss, "stack_fingerprint_meta", None) or {}
        if isinstance(fp_meta, dict):
            # framework_version is whichever of sglang/vllm is active.
            fw_lc = framework.lower()
            if fw_lc in ("sglang", "vllm"):
                v = str(fp_meta.get(fw_lc) or "").strip()
                if v and v != "unknown":
                    out["framework_version"] = v
            for src_key, dst_key in (
                ("rocm", "rocm_version"),
                ("aiter", "aiter_version"),
                ("image_digest", "image_digest"),
            ):
                v = str(fp_meta.get(src_key) or "").strip()
                if v and v != "unknown":
                    out[dst_key] = v
        # per-baseline workload extras from materialized YAML; keep bool False (don't drop an "explicitly disabled" signal).
        wl_extra = getattr(ss, "baseline_workload_extra", None) or {}
        if isinstance(wl_extra, dict):
            for k in ("max_running_requests", "max_num_seqs"):
                v = wl_extra.get(k)
                if isinstance(v, int) and v > 0:
                    out[k] = v
            for k in ("chunked_prefill_enabled", "enable_torch_compile"):
                v = wl_extra.get(k)
                if isinstance(v, bool):
                    out[k] = v
            for k in ("quant_scheme", "workload_mode"):
                v = wl_extra.get(k)
                if isinstance(v, str) and v.strip():
                    out[k] = v.strip()
        return out

    def _build_kernel_optimizations_from_state(self) -> list[dict[str, Any]]:
        """Collect KEEP'd kernel optimizations + their E2E verdict by joining kernel_opt_attempts (micro) and kernel_integrate_attempts (E2E) on kernel_id; non-integrated KEEPs surface integrated=False. Returns KernelOptimization-shaped dicts.

        Returns:
            A list of KernelOptimization-shaped dicts for each KEEP'd kernel,
            joined with its E2E integrate verdict where available.
        """
        ss = self.shared_state
        opt_attempts = (
            getattr(ss, "kernel_opt_task_attempts", {})
            or getattr(ss, "kernel_opt_attempts", {})
            or {}
        )
        integ_attempts = getattr(ss, "kernel_integrate_attempts", {}) or {}
        if not isinstance(opt_attempts, dict):
            return []

        # Index integrate results by kernel_id (last write wins; entry carries rolled-up best_gain_pct).
        integ_by_kid: dict[str, dict[str, Any]] = {}
        integ_by_task: dict[str, dict[str, Any]] = {}
        if isinstance(integ_attempts, dict):
            for entry in integ_attempts.values():
                if not isinstance(entry, dict):
                    continue
                kid = str(entry.get("kernel_id") or "")
                if kid:
                    integ_by_kid[kid] = entry
                task_group_key = str(entry.get("task_group_key") or "")
                if task_group_key:
                    integ_by_task[task_group_key] = entry

        out: list[dict[str, Any]] = []
        for ledger_id, e in opt_attempts.items():
            if not isinstance(e, dict):
                continue
            if str(e.get("last_decision", "")).upper() != "KEEP":
                continue
            try:
                micro = float(e.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            kid = str(
                e.get("current_kernel_id")
                or e.get("kernel_id")
                or ledger_id
            )
            task_group_key = str(e.get("task_group_key") or "")
            integ = (
                integ_by_task.get(task_group_key)
                if task_group_key
                else integ_by_kid.get(kid)
            )
            e2e_gain = 0.0
            e2e_tput = 0.0
            e2e_decision = ""
            integrated = False
            if isinstance(integ, dict):
                integrated = True
                # Integrate-layer verdict (E2E); lets warm-start skip a micro-win/E2E-loss kernel.
                e2e_decision = str(integ.get("last_decision") or "").upper()
                try:
                    e2e_gain = float(integ.get("best_gain_pct") or 0.0)
                except (TypeError, ValueError):
                    e2e_gain = 0.0
                # Last attempt's E2E re-bench throughput.
                for att in reversed(list(integ.get("attempts") or [])):
                    if isinstance(att, dict) and att.get("new_tput") is not None:
                        try:
                            e2e_tput = float(att.get("new_tput") or 0.0)
                        except (TypeError, ValueError):
                            e2e_tput = 0.0
                        break
            out.append(
                {
                    "kernel_id": kid,
                    "source_file": str(e.get("last_source_file") or ""),
                    "artifact_path": str(e.get("last_artifact_path") or ""),
                    "micro_speedup": micro,
                    "decision": "KEEP",
                    "e2e_gain_pct": e2e_gain,
                    "e2e_tput": e2e_tput,
                    "e2e_decision": e2e_decision,
                    "integrated": integrated,
                    "ts": str(e.get("last_ts") or ""),
                }
            )
        return out

    def _collect_attempt_provenance(
        self,
    ) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
        """Map proven optimizations to their research-hint origin from the gaps[] attempts ledger; returns (kept_sources by name/kernel, kept_by_gap by canonical_id, reverted_rows). Fail-soft.

        Returns:
            A ``(kept_sources, kept_by_gap, reverted_rows)`` tuple: KEEP'd
            provenance keyed by variant/kernel name, KEEP'd provenance keyed by
            gap canonical_id, and reverted-attempt rows.
        """
        kept_sources: dict[str, str] = {}
        kept_by_gap: dict[str, str] = {}
        reverted_rows: list[dict[str, Any]] = []
        gaps = getattr(self.shared_state, "gaps", []) or []
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            provenance = str(gap.get("provenance") or "").strip()
            canonical = str(gap.get("canonical_id") or "").strip()
            for attempt in gap.get("attempts") or []:
                if not isinstance(attempt, dict):
                    continue
                variant = str(attempt.get("variant_name") or "").strip()
                kernel = str(attempt.get("kernel_id") or "").strip()
                outcome = str(attempt.get("outcome") or "").strip().upper()
                if outcome == "KEEP" and provenance:
                    if variant:
                        kept_sources.setdefault(variant, provenance)
                    if kernel:
                        kept_sources.setdefault(kernel, provenance)
                    if canonical:
                        kept_by_gap.setdefault(canonical, provenance)
                elif outcome == "REVERT" and (variant or kernel):
                    row: dict[str, Any] = {
                        "name": variant or kernel,
                        "reason": "reverted",
                        "gain_pct": attempt.get("gain_pct"),
                    }
                    if provenance:
                        row["source"] = provenance
                    reverted_rows.append(row)
        return kept_sources, kept_by_gap, reverted_rows

    def _build_recipe_attrs_from_state(self) -> dict[str, Any]:
        """Materialise the recipe-shaped view of :class:`SharedState` (defensive getattr).

        Returns:
            A recipe-shaped attrs dict (best_config, what_worked, what_failed,
            kernel_optimizations, workload tags, session row) for KB recipe
            writes.
        """
        ss = self.shared_state
        current_best = getattr(ss, "current_best", {}) or {}
        opt_stack = getattr(ss, "optimization_stack", []) or []
        gain_per_stack = getattr(ss, "gain_per_stack_entry", []) or []
        last_failures = getattr(ss, "last_action_failures", []) or []
        # RecipeKB best_config keys on the canonical extra_server_args field.
        best_config: dict[str, Any] = {}
        if isinstance(current_best, dict):
            cb_args = current_best.get("extra_server_args")
            if cb_args:
                best_config["extra_server_args"] = str(cb_args)
            for key in ("extra_envs", "name", "tput", "accuracy"):
                if key in current_best:
                    best_config[key] = current_best[key]
        # Prefer the last validated stack layer for launch args (current_best may carry a corrupted string).
        if opt_stack:
            last_entry = opt_stack[-1]
            if isinstance(last_entry, dict):
                stack_args = str(
                    last_entry.get("candidate_extra_server_args") or last_entry.get("extra_server_args") or "",
                ).strip()
                if stack_args:
                    best_config["extra_server_args"] = stack_args
        sediment_on = bool(getattr(ss, "recipe_sediment_enabled", True))
        kept_sources, kept_by_gap, reverted_rows = self._collect_attempt_provenance() if sediment_on else ({}, {}, [])
        what_worked: list[dict[str, Any]] = []
        for idx, entry in enumerate(opt_stack):
            if not isinstance(entry, dict):
                continue
            gain_per: float | None = None
            if idx < len(gain_per_stack):
                gain_per = gain_per_stack[idx]
            name = str(entry.get("variant_name") or entry.get("name") or entry.get("kernel_id") or "")
            row: dict[str, Any] = {
                "name": name,
                "extra_server_args": str(entry.get("extra_server_args") or ""),
                "extra_envs": dict(entry.get("extra_envs") or {}),
                "gain_pct": gain_per,
            }
            # Prefer the entry's gap-id provenance (naming-independent); fall back to name/kernel_id match.
            entry_gap = str(entry.get("gap_canonical_id") or "").strip()
            src = (
                (kept_by_gap.get(entry_gap) if entry_gap else None)
                or kept_sources.get(name)
                or kept_sources.get(str(entry.get("kernel_id") or ""))
            )
            if src:
                row["source"] = src
            what_worked.append(row)
        what_failed: list[dict[str, Any]] = []
        for failure in last_failures[-10:]:
            if isinstance(failure, dict):
                what_failed.append(
                    {
                        "name": str(failure.get("name") or failure.get("action") or ""),
                        "reason": str(failure.get("reason") or failure.get("error_class") or ""),
                    }
                )
        for rev in reverted_rows:
            what_failed.append(rev)
        kernel_optimizations = self._coord._build_kernel_optimizations_from_state()
        cumulative_validated = float(getattr(ss, "cumulative_gain_validated", 0.0) or 0.0)
        cumulative_total = float(getattr(ss, "cumulative_gain", 0.0) or 0.0)
        validated_stack_len = int(getattr(ss, "cumulative_gain_validated_stack_len", 0) or 0)
        stack_fingerprint = getattr(ss, "stack_fingerprint", "") or ""
        # Workload-shape tags for shape-filtered warm-start queries (shared via _collect_workload_tags).
        workload_tags = self._coord._collect_workload_tags()
        # framework_version left unset here (manifest-derived); the T0 backfill writes it.
        return {
            "best_config": best_config,
            "best_throughput": float(current_best.get("tput", 0.0)) if isinstance(current_best, dict) else 0.0,
            "what_worked": what_worked,
            "what_failed": what_failed,
            "kernel_optimizations": kernel_optimizations,
            "stack_fingerprint": {"sha": str(stack_fingerprint)} if stack_fingerprint else {},
            "last_profiled": str(getattr(ss, "cumulative_gain_validated_ts", "") or ""),
            "workload": workload_tags,
            "sessions": [
                {
                    "session_id": str(getattr(ss, "recipe_kb_session_id", "") or self.session_dir.name),
                    "gain_pct": cumulative_validated or cumulative_total,
                    "stack_len": validated_stack_len or len(opt_stack),
                    # arbor-shape provenance so the session row is self-describing (before/after tput + knobs).
                    "throughput_before": float(getattr(ss, "baseline_tput", 0.0) or 0.0),
                    "throughput_after": (
                        float(current_best.get("tput", 0.0)) if isinstance(current_best, dict) else 0.0
                    ),
                    "date": datetime.now(timezone.utc).isoformat(),
                    "actions_taken": [
                        nm
                        for nm in (
                            str(e.get("variant_name") or e.get("name") or e.get("action") or "").strip()
                            for e in opt_stack
                            if isinstance(e, dict)
                        )
                        if nm
                    ],
                }
            ],
        }

    def finalize_recipe_and_journal(self) -> None:
        """CLOSE-time fact finalize: final update_recipe + journal finalize (total_gain_pct + final_throughput); idempotent (CLOSE sequencer + _recipe_kb_t4_hook safety net)."""
        try:
            journal = self._ensure_journal()
            ss = self.shared_state
            cb = getattr(ss, "current_best", {}) or {}
            final_tput = float(cb.get("tput", 0.0)) if isinstance(cb, dict) else 0.0
            total_gain = float(
                getattr(ss, "cumulative_gain_validated", 0.0) or getattr(ss, "cumulative_gain", 0.0) or 0.0,
            )
            journal.finalize(
                final_throughput=final_tput if final_tput > 0 else None,
                total_gain_pct=total_gain,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("optimization_journal.finalize failed")

        if self.recipe_kb is None:
            return
        ss = self.shared_state
        model_name = getattr(ss, "model_name", "") or ""
        gpu_type = getattr(ss, "gpu_type", "") or ""
        if not model_name or not gpu_type:
            log.info(
                "recipe KB finalize_recipe: missing model/hardware (model=%r hardware=%r); skipping update_recipe",
                model_name,
                gpu_type,
            )
            return
        try:
            attrs = self._coord._build_recipe_attrs_from_state()
            # Hoist workload tags flat into top-level recipe attrs (shallow-merged) for warm-start filters.
            workload_tags = attrs.get("workload") or {}

            # sessions[] read-modify-write: read anchor, drop prior entry with our session_id (resume safety), append ours, write back.
            my_sessions = list(attrs["sessions"] or [])
            my_session_ids = {str((s or {}).get("session_id") or "") for s in my_sessions if isinstance(s, dict)}
            # v2: read-modify-write the recipe row; sessions[] merged in-process under the cid flock so concurrent finalises don't tear.
            merged_sessions: list[dict[str, Any]] = list(my_sessions)
            existing_row: dict[str, Any] = {}
            if self.recipe_kb is not None:
                try:
                    cid = self._workload_canonical_id()
                    # Read the LOCAL row (authoritative for writes) so the merge + guard compare against it.
                    existing_row = self.recipe_kb.local.get_recipe(canonical_id=cid) or {}
                    existing_sessions: list[dict[str, Any]] = []
                    for row in existing_row.get("sessions") or []:
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("session_id") or "") in my_session_ids:
                            # Resume/retry of the same session — our new entry supersedes the prior one.
                            continue
                        existing_sessions.append(dict(row))
                    merged_sessions = existing_sessions + my_sessions
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.info(
                        "recipe read failed (%s); finalize will append "
                        "the current session only; the next finalize "
                        "will catch up.",
                        exc,
                    )

            # KEEP'd kernel optimizations ride the extras channel; merge with prior rows, dedup by kernel_id.
            kopts_new = list(attrs.get("kernel_optimizations") or [])
            new_kids = {str((k or {}).get("kernel_id") or "") for k in kopts_new if isinstance(k, dict)}
            merged_kopts: list[dict[str, Any]] = list(kopts_new)
            for prior in existing_row.get("kernel_optimizations") or []:
                if not isinstance(prior, dict):
                    continue
                if str(prior.get("kernel_id") or "") in new_kids:
                    continue
                merged_kopts.append(dict(prior))

            extras_payload = dict(workload_tags or {})
            if merged_kopts:
                extras_payload["kernel_optimizations"] = merged_kopts

            overrides: dict[str, Any] = {
                "what_worked": attrs["what_worked"],
                "what_failed": attrs["what_failed"],
                "last_profiled": attrs["last_profiled"],
                "sessions": merged_sessions,
                "extras": extras_payload,
            }
            # Overwrite best_config/best_throughput only on a real improvement: requires has_validated_win AND my_tput > live_tput.
            my_tput = float(attrs.get("best_throughput") or 0.0)
            cb_now = getattr(ss, "current_best", {}) or {}
            cb_args_now = str(cb_now.get("extra_server_args") or "").strip() if isinstance(cb_now, dict) else ""
            validated_gain = float(getattr(ss, "cumulative_gain_validated", 0.0) or 0.0)
            has_validated_win = bool(
                (getattr(ss, "optimization_stack", []) or []) or validated_gain > 0.0 or cb_args_now
            )
            try:
                live_tput = float(existing_row.get("best_throughput") or 0.0)
            except (TypeError, ValueError):
                live_tput = 0.0
            if has_validated_win and my_tput > live_tput:
                overrides["best_config"] = attrs["best_config"]
                overrides["best_throughput"] = my_tput
            # Merge stack_fingerprint rather than replace (CLOSE only has the sha; T0 stamps version keys).
            merged_fp = dict(existing_row.get("stack_fingerprint") or {})
            for fp_key, fp_val in (attrs.get("stack_fingerprint") or {}).items():
                if fp_val not in (None, "", {}):
                    merged_fp[fp_key] = fp_val
            if merged_fp:
                overrides["stack_fingerprint"] = merged_fp

            self._kb_amend_recipe(
                recipe_overrides=overrides,
                provenance_details={
                    "phase": "close_finalize",
                    "evidence": [
                        f"log:session-{getattr(ss, 'recipe_kb_session_id', '') or self.session_dir.name}",
                    ],
                },
            )
        # Catch-all keeps CLOSE step 2.5 defensive against programmer bugs.
        except Exception:  # noqa: BLE001 — defensive
            log.exception("update_recipe raised unexpectedly")

    async def _record_specialist_result(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> None:
        """Common bookkeeping for any specialist task termination (dispatcher loop + intent routing); idempotent on round_id, failures logged not raised.

        Args:
            task: The terminated specialist task.
            done_payload: The specialist's done payload (proposal_set, domain,
                summary, etc.).
            source: The emitting agent string (``specialist:<task_id>``).
        """
        domain = str(done_payload.get("domain") or "").strip()
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        is_empty = bool(done_payload.get("empty")) or len(proposals) == 0

        round_entry = self._build_specialist_round_entry(
            task=task,
            done_payload=done_payload,
            source=source,
        )
        # Advisory multi-model scoring of the proposal_set; informational only, gates nothing. Defensive.
        _scorer = getattr(self, "_proposal_scorer", None)
        if _scorer is not None and proposals:
            try:
                scores = await _scorer.score(
                    gap={
                        "domain": domain,
                        "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                        "gap_symptom": (task.params or {}).get("gap_symptom"),
                        "gap_evidence": (task.params or {}).get("gap_evidence"),
                        "summary": done_payload.get("summary", ""),
                    },
                    proposals=proposals,
                    task_id=task.task_id,
                    tick=int(getattr(self.shared_state, "tick", 0) or 0),
                    phase=(getattr(self.shared_state, "phase", "") or "") or None,
                )
                if scores and scores.get("models"):
                    round_entry["ensemble_scores"] = scores
            except Exception:  # noqa: BLE001 — advisory; never block
                log.exception(
                    "specialist bookkeeping: proposal scoring failed for task=%s (continuing without scores)",
                    task.task_id,
                )
        try:
            self.shared_state.record_specialist_round(round_entry)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: record_specialist_round failed for task=%s",
                task.task_id,
            )

        # Per-anchor coverage ledger: every specialist completion is
        # one "round" — tick all anchors, then zero the one that just ran so a
        # long-idle domain's counter climbs until the hard-trigger forces it.
        try:
            self.shared_state.bump_domain_round_counters()
            self.shared_state.note_specialist_dispatched(domain)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: domain round-counter update failed for task=%s",
                task.task_id,
            )

        try:
            self.shared_state.update_last_specialist(
                {
                    "task_id": task.task_id,
                    "domain": domain,
                    "gap_canonical_id": str(done_payload.get("gap_canonical_id") or ""),
                    "empty": is_empty,
                    "proposals_total": len(proposals),
                    "confidence": done_payload.get("confidence"),
                    "summary": str(done_payload.get("summary") or "")[:480],
                    "reason": str(done_payload.get("reason") or "")[:480],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: update_last_specialist failed for task=%s",
                task.task_id,
            )

        # Persist so a resume picks up the bookkeeping without re-running the specialist.
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: SharedState.save failed for task=%s",
                task.task_id,
            )

        # Routed via ``_coord`` so a test / caller that overrides
        # ``coordinator._record_observation`` still wins (bare-name delegation
        # resolves it back onto this class otherwise).
        await self._coord._record_observation(
            source or "coordinator",
            "observation",
            {
                "kind": "specialist_done_recorded",
                "task_id": task.task_id,
                "domain": domain,
                "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                "proposals_total": len(proposals),
                "empty": is_empty,
            },
        )

        # Multi-node only: auto-materialise the proposal_set into a
        # benchmarked explore task. No-op single-node (LLM drives explore
        # directly there) and no-op when the proposal_set is empty / has
        # no applicable variants. See :meth:`_maybe_materialize_mn_explore`.
        try:
            await self._maybe_materialize_mn_explore(
                task=task,
                domain=domain,
                proposals=proposals,
            )
        except Exception:  # noqa: BLE001 — defensive; never block bookkeeping
            log.exception(
                "mn_auto_materialize: bridge raised for task=%s (continuing)",
                task.task_id,
            )

        # ``_route_steward_verdict`` has no definition anywhere, so this branch
        # cannot succeed — the except below swallows the AttributeError.
        if domain == "session_steward_specialist":
            try:
                await self._route_steward_verdict(
                    task=task,
                    done_payload=done_payload,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "steward routing failed for task=%s; no phase-routing change applied",
                    task.task_id,
                )

        # Harvest research-scout output (hints, competitor target, gap seeds, PR dedup). Fail-soft.
        if domain == "research_scout_specialist":
            try:
                await self._coord._harvest_research_scout(done_payload)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout harvest failed for task=%s",
                    task.task_id,
                )

        # Consume static-recon bridge candidates into gaps[] so the EXPLORE
        # freeform specialist picks them up with a precise mandate. Fail-soft.
        if domain == "static_recon_specialist":
            try:
                self._coord._consume_static_recon(done_payload)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "static-recon consume failed for task=%s",
                    task.task_id,
                )

        # Aggregate research evidence from any domain (e.g. pr_intel) that
        # self-reports a ``research`` block, so FRAMEWORK / explore lanes
        # reuse the session-wide seen-set. Idempotent for research_scout
        # (already harvested above). Fail-soft.
        try:
            self._coord._aggregate_research_evidence(done_payload)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "research evidence aggregation failed for task=%s",
                task.task_id,
            )

        # Refresh the gaps ledger after a specialist round closes; record the verdict as a gap attempt.
        gap_cid = str(done_payload.get("gap_canonical_id") or "").strip()
        if gap_cid:
            try:
                self.shared_state.append_gap_attempt(
                    gap_cid,
                    {
                        "action": "specialist",
                        "variant_name": domain,
                        "outcome": "EMPTY" if is_empty else "PROPOSALS",
                        "proposals_total": len(proposals),
                    },
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "specialist bookkeeping: append_gap_attempt failed for gap=%s",
                    gap_cid,
                )
        try:
            await self._refresh_gaps(reason="specialist_done")
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "specialist bookkeeping: _refresh_gaps failed for task=%s",
                task.task_id,
            )
        if bool((task.params or {}).get("enablement")) and isinstance(
            done_payload.get("needs_targeted_build"), dict
        ):
            try:
                await self._maybe_enqueue_specialist_requested_build(
                    task_id=str(task.task_id or ""),
                    payload=done_payload,
                )
            except Exception:  # noqa: BLE001
                log.exception(
                    "specialist build request failed for task=%s",
                    task.task_id,
                )
        # Push specialist-authored patches to the Critic so integrate_patch can pass.
        try:
            await self._maybe_autosubmit_specialist_patches(
                task=task,
                done_payload=done_payload,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "B3: specialist patch autosubmit failed for task=%s",
                task.task_id,
            )
        # Relaxed FRAMEWORK rule: a config-lever deliverable (no source patch,
        # but a proposal_set of serving flags / env vars) is routed through the
        # same integrate_patch gate via its config_changes channel.
        try:
            await self._maybe_autosubmit_framework_config(
                task=task,
                done_payload=done_payload,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "FRAMEWORK config autosubmit failed for task=%s",
                task.task_id,
            )

    def _aggregate_research_evidence(self, done_payload: dict[str, Any]) -> None:
        """Aggregate research evidence (PR ids / diffs / NVIDIA refs) into the
        session-wide seen-set, de-duped across the session.

        Applies to every domain that self-reports a ``research`` block
        (``pr_intel`` + ``research_scout``), so FRAMEWORK / explore lanes
        do not re-fetch the same references. Fail-soft: never raises (the caller
        also guards, but keep this self-contained so partial payloads degrade
        gracefully).
        """
        block = done_payload.get("research")
        if not isinstance(block, dict):
            return
        pr_ids: list[Any] = []
        for key in ("prs_fetched", "pr_diffs_read", "nvidia_refs"):
            vals = block.get(key)
            if isinstance(vals, list):
                pr_ids.extend(vals)
        if not pr_ids:
            return
        try:
            added = self.shared_state.register_seen_pr_ids(pr_ids)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "depth: register_seen_pr_ids failed during research aggregation",
            )
            return
        if added:
            log.info(
                "depth: aggregated %d new research reference(s) into seen-set",
                added,
            )

    async def _harvest_research_scout(self, done_payload: dict[str, Any]) -> None:
        """Persist top-level scout output and re-seed Orchestration.

        The scout is a text-hints-only collector. Any ``competitor_target``
        numbers it emits are intentionally ignored here: measured competitor
        baselines are sourced from InferenceX, not authored by the scout, so
        LLM-written numbers must never be persisted as a consumable target.

        Args:
            done_payload: The completed research-scout task payload.
        """
        from ..knowledge import research_hints as _research_hints

        hints = done_payload.get("new_findings") or []
        if not isinstance(hints, list):
            hints = []
        try:
            added, dropped = _research_hints.append_hints(
                self.session_dir,
                hints,
            )
            if dropped:
                log.info(
                    "research-scout: dropped %d sourceless hint(s)",
                    dropped,
                )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: append_hints failed")
            added = 0
        # Share inspected PR ids with the FRAMEWORK dedup set.
        pr_ids: list[Any] = []
        for hint in hints:
            if isinstance(hint, dict) and hint.get("source"):
                pr_ids.append(hint["source"])
        proposals = done_payload.get("proposal_set") or []
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                for key in ("pr_evidence", "source_evidence"):
                    refs = proposal.get(key)
                    if isinstance(refs, list):
                        pr_ids.extend(refs)
        try:
            self.shared_state.register_seen_pr_ids(pr_ids)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: register_seen_pr_ids failed")
        # Seed high-priority hints as gaps[] so EXPLORE tries them early.
        try:
            self._seed_gaps_from_research_hints()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: gap seeding failed")
        compacted = await self._coord._maybe_checkpoint_orchestration(
            tick=int(getattr(self.shared_state, "tick", 0) or 0),
            force=True,
        )
        if not compacted:
            self._coord._reset_orchestration_conversation()
        log.info(
            "research-scout harvested: hints_added=%d seen_pr_ids=%d",
            added,
            len(self.shared_state.research_scout_seen_pr_ids or []),
        )

    def _lift_to_current_best(
        self,
        task_kind: str,
        best_tput: float,
        bv: dict[str, Any],
        *,
        gap_canonical_id: str = "",
    ) -> bool:
        """Lift a winner only when it improves the current throughput anchor.

        Args:
            task_kind: The action kind that produced the winner (stamped on the
                stack entry / current_best).
            best_tput: The winning variant's measured throughput.
            bv: The winning variant dict (args, envs, metrics, provenance).
            gap_canonical_id: When known, stamped onto the stack entry so
                provenance resolves by gap id rather than name.

        Returns:
            ``True`` when the winner was lifted, ``False`` when it was refused
            for not beating the current anchor.
        """
        anchor = resolve_grading_anchor_tput(self.shared_state)
        if anchor > 0 and float(best_tput) <= anchor:
            log.info(
                "current_best held at %.1f: %s winner measured %.1f (no lift)",
                anchor,
                task_kind,
                float(best_tput),
            )
            return False
        previous = self.shared_state.current_best or {}
        base_args = ""
        if isinstance(previous, dict):
            base_args = str(previous.get("extra_server_args") or "").strip()
        candidate_args = ""
        if isinstance(bv, dict):
            candidate_args = str(bv.get("candidate_extra_server_args") or bv.get("extra_server_args") or "").strip()
        full_args = ""
        if isinstance(bv, dict):
            full_args = str(bv.get("extra_server_args") or "").strip()
        controls_effective = bool(
            isinstance(bv, dict)
            and (
                bv.get("remove_args")
                or bv.get("unset_envs")
                or str(bv.get("args_mode") or "").strip().lower() == "replace"
            )
        )
        # Build cumulative launch args without double-stacking; helper dedupes repeated --flag pairs (last wins).
        if controls_effective:
            # Removal/replace winners publish their effective cumulative config
            # from ExploreExecutor. Prepending the prior current_best would
            # reintroduce flags the variant deliberately removed.
            full_args = _dedupe_extra_server_args(full_args)
        else:
            full_args = _merge_cumulative_extra_server_args(
                base_args,
                candidate_args,
                full_args,
            )

        variant_name = bv.get("name") if isinstance(bv, dict) else None
        if candidate_args or variant_name:
            existing = {
                (str(e.get("action")), str(e.get("variant_name")))
                for e in self.shared_state.optimization_stack
                if isinstance(e, dict)
            }
            key = (task_kind, str(variant_name or ""))
            if key not in existing:
                source_phase = str(
                    (bv.get("source_phase") if isinstance(bv, dict) else "")
                    or (
                        getattr(self.shared_state, "phase", "")
                        if task_kind != "integrate_patch"
                        else ""
                    )
                    or ""
                ).strip()
                stack_entry: dict[str, Any] = {
                    "action": task_kind,
                    "variant_name": variant_name,
                    "candidate_extra_server_args": candidate_args,
                    "extra_server_args": full_args,
                    "extra_envs": (dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}),
                    "tput": float(best_tput),
                    "workspace": (bv.get("workspace") if isinstance(bv, dict) else None),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                if source_phase:
                    stack_entry["source_phase"] = source_phase
                if gap_canonical_id:
                    stack_entry["gap_canonical_id"] = gap_canonical_id
                # Stamp the variant's stable join key (and source) so breakdown
                # attribution maps this explore gain to its specialist provenance.
                fp_val = ""
                prov_val = ""
                if isinstance(bv, dict):
                    fp_val = str(bv.get("fingerprint") or "").strip()
                    if not fp_val:
                        from ..actions.executors._canonical_fingerprint import (
                            canonical_fingerprint,
                        )

                        fp_val = canonical_fingerprint(
                            candidate_args or full_args,
                            dict(bv.get("extra_envs") or {}),
                            remove_args=bv.get("remove_args"),
                            unset_envs=bv.get("unset_envs"),
                            args_mode=str(bv.get("args_mode") or "append"),
                        )
                    prov_val = str(bv.get("provenance") or "").strip()
                if fp_val:
                    stack_entry["fingerprint"] = fp_val
                if prov_val:
                    stack_entry["provenance"] = prov_val
                if isinstance(bv, dict):
                    for _ctrl_key in ("remove_args", "unset_envs", "args_mode"):
                        if bv.get(_ctrl_key):
                            stack_entry[_ctrl_key] = bv.get(_ctrl_key)
                    if bv.get("task_id"):
                        stack_entry["task_id"] = str(bv.get("task_id"))
                    if bv.get("effective_extra_server_args"):
                        stack_entry["effective_extra_server_args"] = _dedupe_extra_server_args(
                            str(bv.get("effective_extra_server_args") or "")
                        )
                # Stable filter label for "what kind of optimization" (backend /
                # param / env), so the stack can be sliced like the timeline.
                _stack_envs = dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}
                stack_entry["operation_kind"] = operation_kind_for(
                    task_kind,
                    classify_change_kind(
                        task_kind,
                        {"extra_server_args": candidate_args, "extra_envs": _stack_envs},
                    ),
                )
                _stack_scope = str(bv.get("scope") or "").strip() if isinstance(bv, dict) else ""
                if _stack_scope:
                    stack_entry["scope"] = _stack_scope
                if isinstance(bv, dict):
                    for _src_key in (
                        "source_snapshot",
                        "source_manifest",
                        "framework_root",
                        "base_sha",
                    ):
                        val = bv.get(_src_key)
                        if val:
                            stack_entry[_src_key] = str(val)
                    target_files = [
                        str(path)
                        for path in (bv.get("target_files") or [])
                        if str(path).strip()
                    ]
                    if target_files:
                        stack_entry["target_files"] = target_files
                    for _attr_key in ("baseline_enablement", "attribution_eligible"):
                        if _attr_key in bv:
                            stack_entry[_attr_key] = bool(bv.get(_attr_key))
                    if "framework_agent_authoring" in bv:
                        stack_entry["framework_agent_authoring"] = bool(
                            bv.get("framework_agent_authoring")
                        )
                    for _origin_key in ("domain", "gap_layer"):
                        if bv.get(_origin_key):
                            stack_entry[_origin_key] = str(bv.get(_origin_key))
                self.shared_state.optimization_stack.append(stack_entry)
                # Mirror append into gain_per_stack_entry so the two lists stay index-aligned.
                self.shared_state.append_stack_gain_entry(
                    action=task_kind,
                    variant_name=variant_name,
                    new_tput=best_tput,
                    extra_server_args=full_args,
                )

        # Merge envs: start from previous stack top envs so source-layer KEEPs
        # (config_changes_applied={}) do not clear prior explore/env layers.
        _prev_envs = dict((previous.get("extra_envs") or {}) if isinstance(previous, dict) else {})
        _new_envs = dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}
        _merged_envs = dict(_prev_envs)
        for _key in to_str_list(bv.get("unset_envs") if isinstance(bv, dict) else None):
            _merged_envs.pop(_key, None)
        _merged_envs.update(_new_envs)
        current_best = {
            "action": task_kind,
            "tput": float(best_tput),
            "variant_name": variant_name,
            "extra_server_args": full_args,
            "extra_envs": _merged_envs,
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": bv.get("ttft_mean_ms") if isinstance(bv, dict) else None,
            "e2el_mean_ms": bv.get("e2el_mean_ms") if isinstance(bv, dict) else None,
            "tpot_mean_ms": bv.get("tpot_mean_ms") if isinstance(bv, dict) else None,
            "workspace": bv.get("workspace") if isinstance(bv, dict) else None,
        }
        if isinstance(bv, dict):
            for _ctrl_key in ("remove_args", "unset_envs", "args_mode"):
                if bv.get(_ctrl_key):
                    current_best[_ctrl_key] = bv.get(_ctrl_key)
            if bv.get("effective_extra_server_args"):
                current_best["effective_extra_server_args"] = _dedupe_extra_server_args(
                    str(bv.get("effective_extra_server_args") or "")
                )
            if (bv.get("remove_args") or bv.get("unset_envs")) and not current_best.get("args_mode"):
                current_best["args_mode"] = "replace"
        self.shared_state.current_best = current_best
        if self.shared_state.baseline_tput > 0:
            self.shared_state.cumulative_gain = (
                (float(best_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
            )
        return True

    def _should_run_prelude_bootstrap(self, tput: Any) -> bool:
        """Whether to enqueue the post-baseline PRELUDE bootstrap chain.

        Returns ``False`` when there is no positive baseline throughput, when a
        roofline task is already pending, or when a stop is already pending
        (e.g. the baseline accuracy test produced no result ->
        ``baseline_accuracy_failed``). In the stop case the run is about to halt
        at the Coordinator's end-of-tick check, so no new bootstrap work (warm
        replay / roofline / scout / static recon) must be enqueued or dispatched
        in the meantime.

        Args:
            tput: The promoted baseline throughput.

        Returns:
            bool: True only when the bootstrap chain should run.
        """
        if not (isinstance(tput, (int, float)) and tput > 0):
            return False
        if (self.shared_state.auto_roofline_pending_task_id or "").strip():
            return False
        if (self.shared_state.stop_reason or "").strip():
            return False
        return True

    async def _promote_to_shared_state(
        self,
        task_kind: str,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> None:
        """Lift specific action-result fields into the persistent SharedState (baseline/profile/roofline/grid).

        Args:
            task_kind: The settled task's kind, selecting the promote branch.
            result: The task result dict; non-dict results are ignored.
            task: The originating task, used for audit fingerprints and
                pending-roofline gating.
        """
        if not isinstance(result, dict):
            return
        if task_kind in {"framework_agent", "conc_sweep", "replay_warm_recipe", "integrate_patch"}:
            try:
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                result_status = str(result.get("status") or "succeeded")
                kept = task_kind == "framework_agent" and result_status.lower() == "kept"
                v4_result = dict(result)
                v4_result.setdefault(
                    "workload",
                    {
                        "framework": str(getattr(self.shared_state, "framework", "") or ""),
                        "model_name": str(getattr(self.shared_state, "model_name", "") or ""),
                        "gpu_type": str(getattr(self.shared_state, "gpu_type", "") or ""),
                        "precision": str(getattr(self.shared_state, "precision", "") or ""),
                        "tp": int(getattr(self.shared_state, "tp", 0) or 0),
                        "conc": int(getattr(self.shared_state, "conc", 0) or 0),
                        "isl": int(getattr(self.shared_state, "isl", 0) or 0),
                        "osl": int(getattr(self.shared_state, "osl", 0) or 0),
                    },
                )
                instrument.record_action_operation(
                    self.session_dir,
                    action=task_kind,
                    task_id=getattr(task, "task_id", "") if task is not None else "",
                    status=result_status,
                    decision="promoted" if kept else "discarded",
                    result=v4_result,
                    extras={
                        "candidate_id": self._framework_candidate_key(result.get("candidate"))
                        if task_kind == "framework_agent" and isinstance(result.get("candidate"), dict)
                        else ""
                    },
                    phase=str(getattr(self.shared_state, "phase", "") or ""),
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    tick=int(getattr(self.shared_state, "tick", 0) or 0),
                )
            except Exception:  # noqa: BLE001
                log.debug("v4 action result capture failed", exc_info=True)
        outcome = _PromoteOutcome()
        handler_name = self._PROMOTE_HANDLERS.get(task_kind)
        if handler_name is not None:
            await getattr(self, handler_name)(result, task, outcome)
        # sweep / conc_sweep already recorded + saved + returned via their handler.
        if outcome.early_return:
            return
        # Audit trail: one succeeded-attempt record with branch-supplied decision/extras.
        if outcome.audit_decision is not None and task_kind in _AUDIT_ACTIONS:
            self.shared_state.record_action_attempt(
                action=task_kind,
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status="succeeded",
                decision=outcome.audit_decision,
                result=result,
                extras=outcome.audit_extras,
            )
            outcome.changed = True
        if outcome.changed:
            self.shared_state.save(self.session_dir)

    _PROMOTE_HANDLERS: dict[str, str] = {
        "baseline": "_promote_baseline",
        "replay_warm_recipe": "_promote_replay_warm_recipe",
        "profile": "_promote_profile",
        "roofline": "_promote_roofline",
        "explore": "_promote_explore",
        "integrate_patch": "_promote_integrate_patch",
        "framework_agent": "_promote_framework_agent",
        "sweep": "_promote_sweep",
        "conc_sweep": "_promote_conc_sweep",
    }

    async def _promote_baseline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a baseline result: anchor tput / accuracy / config and bootstrap PRELUDE."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        tput = result.get("output_throughput")
        warmup_anchor = result.get("warmup_round_tput")
        tracked_tid = str(getattr(self.shared_state, "enablement_revalidation_task_id", "") or "").strip()
        promoting_tid = str(getattr(task, "task_id", "") or "").strip() if task is not None else ""
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        is_revalidation = bool(
            task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON
            or (tracked_tid and tracked_tid == promoting_tid)
        )
        # The anchor is the best measurement of the unmodified stack. A later,
        # lower re-baseline must not replace it, or downstream gains end up
        # measured against a different reference than the one they claim. A
        # revalidation is exempt: the enablement patch changed the stack, so the
        # prior anchor no longer describes anything reproducible.
        prior_anchor = float(self.shared_state.baseline_tput or 0.0)
        anchor_accepted = bool(
            isinstance(tput, (int, float))
            and tput > 0
            and (prior_anchor <= 0.0 or float(tput) > prior_anchor or is_revalidation)
        )
        if isinstance(tput, (int, float)) and tput > 0:
            if anchor_accepted:
                # The anchor is the hot measure round; the cold warmup round is
                # discarded so gain math never mixes cold-before with hot-after.
                self.shared_state.baseline_tput = float(tput)
            else:
                log.info(
                    "baseline anchor: keeping %.1f; re-baseline measured %.1f "
                    "(task=%s)",
                    prior_anchor,
                    float(tput),
                    promoting_tid,
                )
            self.shared_state.baseline_failure_streak = 0
            self.shared_state.baseline_arg_error_streak = 0
            # A genuine baseline may revalidate an eval-origin enablement.
            if bool(getattr(self.shared_state, "enablement_validation_pending", False)):
                if is_revalidation:
                    acc = result.get("accuracy")
                    floor = float(getattr(self.shared_state, "enablement_accuracy_floor", 0.0) or 0.0)
                    if accuracy_meets_floor(acc, floor):
                        self.shared_state.enablement_succeeded = True
                        self.shared_state.enablement_validation_pending = False
                        self.shared_state.enablement_revalidation_task_id = ""
                        self.shared_state.enablement_origin = ""
                        self.shared_state.enablement_pending = False
                    else:
                        # Sub-floor accuracy on the tracked revalidation: rearm the
                        # specialist loop without clearing the frozen trigger identity.
                        log.warning(
                            "enablement revalidation: accuracy %.4f below floor %.4f; rearming",
                            acc if isinstance(acc, (int, float)) else float("nan"),
                            floor,
                        )
                        self.shared_state.enablement_validation_pending = False
                        self.shared_state.enablement_revalidation_task_id = ""
                        self.shared_state.enablement_stall_streak = (
                            int(getattr(self.shared_state, "enablement_stall_streak", 0) or 0) + 1
                        )
                        _max_stall = getattr(self, "_ENABLEMENT_MAX_STALL", None)
                        if _max_stall is None:
                            try:
                                from ..phases.framework import _ENABLEMENT_MAX_STALL as _max_stall
                            except ImportError:
                                _max_stall = 5
                        if self.shared_state.enablement_stall_streak >= _max_stall and not self.shared_state.stop_reason:
                            self.shared_state.set_stop_reason("enablement_stalled")
                        else:
                            self.shared_state.enablement_dispatched = False
                else:
                    # An unrelated baseline promoted while revalidation is pending.
                    # Only anchor tput; do not consume or clear the pending state.
                    log.info(
                        "enablement revalidation pending: unrelated baseline promoted "
                        "(task=%s tracked=%s); not consuming pending state",
                        promoting_tid,
                        tracked_tid,
                    )
                    self.shared_state.enablement_origin = ""
                    self.shared_state.enablement_pending = False
            else:
                self.shared_state.enablement_origin = ""
                self.shared_state.enablement_pending = False
            changed = True
        # Accuracy / config / wall-clock describe the anchor run, so they only
        # move when the anchor itself moves; otherwise the recorded reference
        # tput and the config it was measured with drift apart.
        if anchor_accepted:
            acc = result.get("accuracy")
            if isinstance(acc, (int, float)):
                self.shared_state.baseline_accuracy = float(acc)
                changed = True
            # Persist the materialized YAML so downstream tasks reuse the exact workload contract.
            materialized = result.get("materialized_config")
            if isinstance(materialized, str) and materialized:
                self.shared_state.baseline_config_path = materialized
                changed = True
                # Parse workload-shape extras from the YAML for lesson/pitfall attrs.
                try:
                    parsed = _parse_baseline_workload_extra(materialized)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "baseline workload extra parsing failed for %s",
                        materialized,
                    )
                    parsed = {}
                if parsed:
                    self.shared_state.baseline_workload_extra = parsed
            # Promote baseline wall-clock so ExploreExecutor derives the overtime kill deadline.
            runtime_sec_raw = result.get("subprocess_runtime_sec")
            if isinstance(runtime_sec_raw, (int, float)) and runtime_sec_raw > 0:
                self.shared_state.baseline_runtime_sec = float(runtime_sec_raw)
                changed = True
            # Promote the warm measure-round wall-clock as the anchor for the
            # explore decision-round overtime kill (present only on the
            # double-run baseline path; else explore uses the cold anchor).
            warm_runtime_raw = result.get("measure_round_runtime_sec")
            if isinstance(warm_runtime_raw, (int, float)) and warm_runtime_raw > 0:
                self.shared_state.baseline_warm_runtime_sec = float(warm_runtime_raw)
                changed = True
            elif float(getattr(self.shared_state, "baseline_warm_runtime_sec", 0.0) or 0.0) != 0.0:
                self.shared_state.baseline_warm_runtime_sec = 0.0
                changed = True
        # current_best.tput follows the same hot baseline contract so the
        # gain numerator and denominator stay aligned. Once the stack carries a
        # validated layer, current_best belongs to the stack top and a baseline
        # must not reset it back to the bare reference config.
        if anchor_accepted and not (getattr(self.shared_state, "optimization_stack", None) or []):
            anchor_tput = float(self.shared_state.baseline_tput or 0.0)
            self.shared_state.current_best = {
                "action": "baseline",
                "tput": (
                    anchor_tput if anchor_tput > 0 else (float(tput) if isinstance(tput, (int, float)) else None)
                ),
                "hot_tput": (float(tput) if isinstance(tput, (int, float)) else None),
                "cold_tput": (
                    float(warmup_anchor) if isinstance(warmup_anchor, (int, float)) and warmup_anchor > 0 else None
                ),
                "ttft_mean_ms": result.get("ttft_mean_ms"),
                "e2el_mean_ms": result.get("e2el_mean_ms"),
                "tpot_mean_ms": result.get("tpot_mean_ms"),
                "workspace": result.get("workspace"),
            }
            changed = True
        if anchor_accepted:
            audit_decision = "promoted"
        elif isinstance(tput, (int, float)) and tput > 0:
            audit_decision = "no_promote"
        else:
            audit_decision = "discarded"
        audit_extras = {
            "materialized_config": result.get("materialized_config"),
            "accuracy": result.get("accuracy"),
            "baseline_tput": (float(tput) if isinstance(tput, (int, float)) else None),
            # Stamp canonical params fingerprint for the self-loop denial helper.
            "fingerprint": _baseline_params_fingerprint(task_params),
            # Record revalidation context for history.
            "is_revalidation": bool(task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON),
            "enablement_succeeded": bool(getattr(self.shared_state, "enablement_succeeded", False)),
            "enablement_accuracy_floor": float(getattr(self.shared_state, "enablement_accuracy_floor", 0.0) or 0.0),
        }
        if not anchor_accepted and isinstance(tput, (int, float)) and tput > 0:
            audit_extras["anchor_kept_tput"] = prior_anchor
        # Present only when the probe cut a runaway eval short; explains a ~0 accuracy.
        if result.get("eval_probe"):
            audit_extras["eval_probe"] = result["eval_probe"]
        # seed the gaps[] ledger from baseline (best-effort).
        await self._refresh_gaps(reason="baseline_done")
        if self.shared_state.baseline_tput > 0:
            await self._drain_queued_baselines(reason="baseline_established")
        # Standalone baseline-arm roofline ceiling (pure CPU): backs up the
        # snapshot ceiling in case the later roofline step fails.
        if isinstance(tput, (int, float)) and tput > 0:
            try:
                self.shared_state.record_baseline_roofline_ceiling()
            except Exception as exc:  # noqa: BLE001 — best-effort backup
                log.warning(
                    "baseline roofline-ceiling backup failed: %r",
                    exc,
                )
        # PRELUDE bootstrap (post-baseline), ordering mandatory: (1) inject warm-recipe history, (2) warm-replay, (3) auto-analysis, (4) research scout.
        # Only the run that first establishes the anchor bootstraps; a later
        # re-baseline must not re-fire replay / scout / recon.
        if prior_anchor <= 0.0 and self._should_run_prelude_bootstrap(tput):
            # History injection (fires regardless of --no-warm-replay).
            try:
                self._inject_warm_recipe_history_into_ledger()
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception(
                    "PRELUDE: warm-recipe history injection failed: %r",
                    exc,
                )
            # Warm-recipe replay, anchored on the hot baseline_tput contract.
            try:
                await self._maybe_enqueue_warm_replay(
                    baseline_tput=float(self.shared_state.baseline_tput or tput),
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception(
                    "PRELUDE: failed to enqueue warm-replay task: %r",
                    exc,
                )
            # Auto-analysis (roofline / profile); may defer.
            await self._maybe_enqueue_prelude_initial_analysis_after_baseline(
                baseline_tput=float(tput),
            )
            # Research scout (parallel, read-only, CPU-only).
            await self._maybe_enqueue_prelude_research_scout()
            # Static-recon (parallel, read-only, CPU-only): seed bridge
            # candidates as gaps[] before EXPLORE starts.
            await self._maybe_enqueue_prelude_static_recon()
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _drain_queued_baselines(self, *, reason: str) -> list[str]:
        """Cancel redundant queued baselines, preserving enablement revalidation.

        Args:
            reason: Stamped onto the cancellation history and the observation.

        Returns:
            The cancelled task ids (empty when the queue held none).
        """
        spared = await self._enablement_revalidation_task_ids()
        try:
            cancelled = await self.tasks.cancel_family(
                ["baseline"],
                reason=reason,
                exclude_task_ids=spared,
            )
        except Exception:  # noqa: BLE001 — draining is best-effort
            log.exception("baseline drain: cancel_family failed")
            return []
        if not cancelled:
            return []
        log.info(
            "baseline drain: cancelled %d queued baseline task(s) (reason=%s, spared=%d)",
            len(cancelled),
            reason,
            len(spared),
        )
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "baseline_drain",
                "reason": reason,
                "cancelled_task_ids": cancelled,
                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
            },
        )
        return cancelled

    async def _enablement_revalidation_task_ids(self) -> set[str]:
        """Return queued enablement-revalidation baseline task IDs."""
        spared: set[str] = set()
        tracked = str(getattr(self.shared_state, "enablement_revalidation_task_id", "") or "").strip()
        if tracked:
            spared.add(tracked)
        try:
            for task in await self.tasks.queued():
                if str(getattr(task, "kind", "") or "") != "baseline":
                    continue
                if (getattr(task, "params", None) or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON:
                    spared.add(str(getattr(task, "task_id", "") or ""))
        except Exception:  # noqa: BLE001 — fall back to the tracked id alone
            log.exception("baseline drain: queued-task scan failed")
        return {t for t in spared if t}

    async def _promote_replay_warm_recipe(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Separate promote path so replay doesn't overwrite baseline_tput/current_best."""
        try:
            self._promote_warm_replay(result, task=task)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("warm-replay promote failed")
        # PRELUDE initial roofline was deferred while replay ran.
        await self._maybe_enqueue_prelude_initial_analysis_after_baseline()

    async def _promote_profile(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a profile result: trace path / status, optional current_best, roofline anchor."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # Defensive skipped arm: audit as skipped + drop the gate.
        if str(result.get("status") or "") == "skipped":
            audit_decision = "skipped"
            audit_extras = {
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
            if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        else:
            audit_decision = "promoted"
            audit_extras = {
                "trace_path": None,
                "profile_args": None,
                "output_throughput": result.get("output_throughput"),
            }
        # Surface the trace path so Orch passes a real path to trace_analyze.
        trace_path = result.get("main_trace_path") or (result.get("trace_files") or [None])[0]
        profile_status = str(result.get("status") or "")
        if profile_status == "failed" or result.get("error_class") == "no_trace_files":
            self.shared_state.last_profile_status = "failed"
            self.shared_state.last_profile_workload = {}
            if not trace_path:
                self.shared_state.last_profile_trace = ""
            self.shared_state.last_profile_args = ""
            self.shared_state.last_profile_workload_action = ""
            changed = True
        elif trace_path:
            self.shared_state.last_profile_trace = str(trace_path)
            self.shared_state.last_profile_status = "succeeded"
            # Record the server config in effect for this trace, tagged with the
            # arm it measured so a later same-arm check can trust it.
            profile_args = ""
            if task is not None:
                task_params = task.params or {}
                profile_args = str(task_params.get("base_extra_args") or "")
                self.shared_state.record_profile_workload(
                    task_params,
                    arm=("baseline" if str(task_params.get("reason") or "") == "prelude_initial" else ""),
                )
            else:
                self.shared_state.last_profile_workload = (
                    self.shared_state.current_profile_workload_context()
                )
                self.shared_state.last_profile_workload_action = str(
                    (self.shared_state.current_best or {}).get("action") or ""
                )
            self.shared_state.last_profile_args = str(
                self.shared_state.last_profile_workload.get("server_args")
                or profile_args
            )
            # New trace invalidates the stale trace_analyze cache.
            self.shared_state.last_trace_analyze = {}
            changed = True
            audit_extras["trace_path"] = str(trace_path)
            audit_extras["profile_args"] = profile_args
        # profile result may include a tput; promote into current_best on the +1% rule.
        tput = result.get("output_throughput")
        cb = self.shared_state.current_best or {}
        cb_tput = cb.get("tput") if isinstance(cb, dict) else None
        cur_best = (
            float(cb_tput)
            if isinstance(cb_tput, (int, float)) and cb_tput > 0
            else float(self.shared_state.baseline_tput or 0.0)
        )
        if (
            isinstance(tput, (int, float))
            and tput > 0
            and cur_best > 0
            and (tput - cur_best) / cur_best * 100.0 >= 1.0
        ):
            promoted = dict(cb) if isinstance(cb, dict) else {}
            promoted.update(
                {
                    "action": "profile",
                    "tput": float(tput),
                    "ttft_mean_ms": result.get("ttft_mean_ms"),
                    "e2el_mean_ms": result.get("e2el_mean_ms"),
                    "tpot_mean_ms": result.get("tpot_mean_ms"),
                    "workspace": result.get("workspace"),
                }
            )
            self.shared_state.current_best = promoted
            if self.shared_state.baseline_tput > 0:
                self.shared_state.cumulative_gain = (
                    (float(tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
                )
            changed = True
        # On a successful profile, re-anchor last_roofline_tput and clear the pending field.
        if profile_status == "succeeded":
            anchor_tput = self._current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
                changed = True
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
            changed = True
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_roofline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a roofline result: audit + failure streak + roofline anchor (reads last_trace_analyze)."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # The composite roofline action runs profile + trace_analyze atomically;
        # its executor writes last_profile_* + last_trace_analyze, so here we just record the audit row.
        status = str(result.get("status") or "")
        if status == "skipped":
            # Defensive arm: clean no-op, no streak/watermark touch.
            audit_decision = "skipped"
            audit_extras = {
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
            # Still clear the pending pointer so the watermark check can re-arm.
            if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        elif status == "succeeded":
            audit_decision = "promoted"
            # Prefer the executor's last_trace_analyze snapshot over the result dict.
            _last_ta = self.shared_state.last_trace_analyze or {}
            audit_extras = {
                "snapshot_id": (
                    _last_ta.get("roofline_snapshot_id")
                    if _last_ta.get("roofline_snapshot_id") is not None
                    else result.get("snapshot_id")
                ),
                "last_profile_trace": (self.shared_state.last_profile_trace or result.get("last_profile_trace")),
                "analysis_md_path": (_last_ta.get("analysis_md_path") or result.get("analysis_md_path")),
                "profile_workspace": result.get("profile_workspace"),
                "degraded": bool(result.get("degraded", False)),
            }
            # Reset the roofline failure streak on a successful snapshot.
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak = 0
            # Re-anchor the 10% watermark step on the projected current tput.
            anchor_tput = self._current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
            changed = True
        else:
            audit_decision = "discarded"
            audit_extras = {
                "phase": result.get("phase"),
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
            # Bump the failure streak (mirrors the audit ledger for prompt renderers).
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak += 1
            changed = True
            log.warning(
                "Auto-roofline %s failed (reason=%s phase=%s "
                "error_class=%s); continuing in degraded mode "
                "(specialists / explore proceed without a fresh "
                "analysis_md). No retry, no fallback.",
                task.task_id if task else "?",
                str((task.params or {}).get("reason") or "") if task is not None else "",
                result.get("phase"),
                result.get("error_class"),
            )
        # Clear the pending pointer (matched by task id).
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
            changed = True
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_explore(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote an explore result: ledger increment, winners, current_best lift, resume revalidation."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # The executor already did per-variant KEEP/REVERT + rebench, so winners
        # are authoritative; Coordinator is single-writer for explore_search.accepted +
        # current_best + optimization_stack and does not re-threshold.
        # 1. Apply the executor's ledger increment.
        update = result.get("explore_search_update")
        if isinstance(update, dict):
            self.shared_state.apply_explore_search_update(update)
            changed = True
        # 2. Search-space expansion bookkeeping (honoured defensively when an update is present).
        disc_update = result.get("discovered_flags_update")
        if isinstance(disc_update, dict):
            self.shared_state.record_discovered_flags(
                framework=str(disc_update.get("framework") or ""),
                backend_flags=disc_update.get("backend_flags"),
                param_flags=disc_update.get("param_flags"),
                source_path=str(disc_update.get("source_path") or ""),
            )
            err = disc_update.get("discovery_error")
            if err:
                self.shared_state.discovered_flags_error = str(err)
            changed = True
        # 3. Per-winner record_explore_accepted (Coordinator is sole writer).
        winners = result.get("winners") or []
        round_id = str(result.get("round_id") or "")
        best_winner = result.get("best_variant")
        best_tput = result.get("output_throughput")
        promoted = False
        # A post-resume revalidation task confirms the EXISTING stack/current
        # best rather than adding a variant, so it never "promotes".
        # Reconcile the validation watermark + clear the
        # ``resume_pending_revalidation`` flag from the measured tput — but
        # ONLY when the rebench actually produced a valid measurement, so a
        # failed/empty rebench leaves the flag set and reports keep warning.
        is_revalidation_task = task is not None and str((task.params or {}).get("source") or "") in {
            "resume_stack_revalidate",
            "resume_reverify_best",
        }
        if is_revalidation_task:
            measured = result.get("output_throughput")
            measured_ok = isinstance(measured, (int, float)) and measured > 0
            # A GEAK revalidation (2b) must assert config identity + that the
            # optimization engaged before stamping validated, else replay via
            # the GEAK harness (2a). Native revalidations keep the
            # unconditional watermark reconciliation below.
            if bool((task.params or {}).get("geak_fallback")):
                got_hash = ""
                if isinstance(best_winner, dict):
                    got_hash = str(best_winner.get("fingerprint") or "")
                if not got_hash and isinstance(winners, list) and winners and isinstance(winners[0], dict):
                    got_hash = str(winners[0].get("fingerprint") or "")
                cb_now = (
                    self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
                )
                cb_tput = cb_now.get("tput")
                decision = _geak_revalidation_decision(
                    measured=measured,
                    baseline=self.shared_state.baseline_tput,
                    got_hash=got_hash,
                    expected_hash=str((task.params or {}).get("expected_cfg_hash") or ""),
                    min_engaged_gain_pct=_MIN_KERNEL_ENGAGED_GAIN_PCT,
                    current_best=cb_tput,
                )
                ps = (
                    self.shared_state.geak_result
                    if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                    else {}
                )
                # A rebench that beats current_best is only a KERNEL gain when
                # GEAK actually produced something. Without a material product
                # (kernel/head/overlay/patch or a config delta vs the pre-KERNEL
                # best) the win is same-config measurement noise; drop it.
                # An empty geak_result cannot be judged by the helper, so
                # disambiguate here: a pre-existing ``geak_e2e`` stack entry
                # means this is a resume revalidation of an already-material win
                # (let it through); otherwise there is no material to validate.
                if decision == "validated":
                    stack_now = self.shared_state.optimization_stack or []
                    has_prior_geak_e2e = any(
                        isinstance(e, dict) and e.get("action") == "geak_e2e" for e in stack_now
                    )
                    # Escape hatch first: a pre-existing geak_e2e stack entry is
                    # an already-proven win, so this is a resume revalidation.
                    # It must short-circuit the material check regardless of
                    # geak_result (which is persisted and thus non-empty on
                    # resume) — by now current_best already carries the GEAK
                    # accepted_config, so the fingerprint would match and be
                    # mis-judged no_material, reverting a real win. Only apply
                    # the material gate on the FIRST validation (no prior entry),
                    # where current_best is still the pre-KERNEL config.
                    if not has_prior_geak_e2e:
                        if not ps:
                            decision = "no_material"
                        elif not _geak_result_has_material(
                            ps,
                            prev_best_flags=str(cb_now.get("extra_server_args") or ""),
                            prev_best_envs=cb_now.get("extra_envs") or {},
                        ):
                            decision = "no_material"
                if decision == "validated":
                    # Write the headline from the measured orchestrator-harness
                    # rebench: lift current_best + optimization_stack + the
                    # validated gain and clear geak_pending.
                    self._promote_geak_from_candidate(
                        ps,
                        measured_tput=float(measured),
                        provenance="geak_orch_harness_validated",
                    )
                elif decision == "no_material":
                    # No material GEAK product; the rebench beating current_best
                    # is same-config measurement noise. Do not touch the
                    # headline / stack / gain; record + clear the candidate.
                    log.info(
                        "geak 2b rebench beat current_best but GEAK shipped no "
                        "material product (measured=%r current_best=%r) -> "
                        "no_material drop",
                        measured,
                        cb_tput,
                    )
                    try:
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "geak_no_material",
                                "measured_tput": float(measured),
                                "current_best_tput": (
                                    float(cb_tput) if isinstance(cb_tput, (int, float)) else None
                                ),
                                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
                            },
                        )
                    except Exception:  # noqa: BLE001 - observation is best-effort
                        log.exception("geak no_material: observation emit failed")
                    # Stamp the drop on geak_result (always, so an empty {} is
                    # distinguishable from never-populated on resume/debug and
                    # acts as a tombstone against KERNEL crash-recovery
                    # re-enqueue) and reject any provisional KEEP in
                    # kernel_journey so a session audit does not read a dropped
                    # candidate as an accepted kernel (no-op when the journey
                    # has no KEEP / the file is absent).
                    ps_stamped = dict(ps) if isinstance(ps, dict) else {}
                    ps_stamped["revalidation_status"] = "no_material"
                    self.shared_state.geak_result = ps_stamped
                    try:
                        self.phase_kernel._reject_geak_kernel_journey(
                            ps_stamped,
                            measured_tput=float(measured),
                            current_best_tput=(
                                float(cb_tput) if isinstance(cb_tput, (int, float)) else 0.0
                            ),
                            provenance="geak_no_material",
                            rejection_reason="geak_no_material_product",
                        )
                    except Exception:  # noqa: BLE001 - journey reject is best-effort
                        log.exception("geak no_material: journey rejection failed")
                    self.shared_state.geak_pending = {}
                    self.shared_state.resume_pending_revalidation = False
                elif decision == "no_promote":
                    # Well-measured + engaged over baseline, but does not beat
                    # current_best. This is a real result, NOT inconclusive, so
                    # do not replay via the GEAK harness (2a); clear the pending
                    # candidate without touching the headline / stack / gain.
                    log.info(
                        "geak 2b rebench did not beat current_best "
                        "(measured=%r current_best=%r) -> no_promote",
                        measured,
                        cb_tput,
                    )
                    try:
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "geak_no_promote",
                                "measured_tput": float(measured),
                                "current_best_tput": (
                                    float(cb_tput) if isinstance(cb_tput, (int, float)) else None
                                ),
                                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
                            },
                        )
                    except Exception:  # noqa: BLE001 - observation is best-effort
                        log.exception("geak no_promote: observation emit failed")
                    self.shared_state.geak_pending = {}
                    self.shared_state.resume_pending_revalidation = False
                else:
                    # 2b inconclusive -> GEAK harness replay (2a), which
                    # clears the pending flag on success. Best-effort.
                    log.warning(
                        "geak 2b revalidation inconclusive "
                        "(measured=%r got_hash=%r expected=%r) -> GEAK-harness 2a fallback",
                        measured,
                        got_hash,
                        (task.params or {}).get("expected_cfg_hash"),
                    )
                    fallback_result: dict[str, Any]
                    try:
                        # Routed via ``_coord`` so a test / caller that overrides
                        # ``coordinator._validate_geak_via_geak_harness`` still wins
                        # (bare-name delegation resolves it back onto this class).
                        fallback_result = await self._coord._validate_geak_via_geak_harness(
                            reason="2b_inconclusive"
                        )
                    except Exception as exc:  # noqa: BLE001 - defensive
                        log.exception("geak 2a GEAK-harness fallback failed")
                        fallback_result = {
                            "validated": False,
                            "reason": repr(exc),
                        }
                        try:
                            from hyperloom.inference_optimizer.breakdown.recorder import instrument

                            geak_result = (
                                self.shared_state.geak_result
                                if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                                else {}
                            )
                            instrument.record_geak_operation(
                                self.session_dir,
                                stage="final_validation_failed",
                                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                                result={
                                    **geak_result,
                                    "failure_reason": "geak_harness_fallback_exception",
                                    "error": repr(exc),
                                },
                                status="failed",
                                validation_source="geak_same_harness_geak",
                            )
                        except Exception:  # noqa: BLE001
                            log.debug("geak v4 fallback-exception recording failed", exc_info=True)
                    if not bool(fallback_result.get("validated")):
                        geak_result = (
                            dict(self.shared_state.geak_result)
                            if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                            else {}
                        )
                        geak_result["revalidation_status"] = "fallback_failed"
                        geak_result["revalidation_error"] = str(
                            fallback_result.get("reason")
                            or fallback_result.get("status")
                            or "GEAK harness fallback did not validate"
                        )[:500]
                        self.shared_state.geak_result = geak_result
                        self.shared_state.geak_pending = {}
                        self.shared_state.resume_pending_revalidation = False
                changed = True
            else:
                if measured_ok and self.shared_state.baseline_tput > 0:
                    self._update_cumulative_gain_validated(measured)
                    cb_rec = (
                        self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
                    )
                    recorded = cb_rec.get("tput")
                    floor = _DEFAULT_RESUME_DRIFT_FLOOR_PCT
                    if (
                        isinstance(recorded, (int, float))
                        and recorded > 0
                        and float(measured) < float(recorded) * floor / 100.0
                    ):
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "current_best_drift",
                                "severity": "high",
                                "measured_tput": float(measured),
                                "recorded_tput": float(recorded),
                                "floor_pct": floor,
                            },
                        )
                if measured_ok:
                    self.shared_state.resume_pending_revalidation = False
                changed = True
        # A revalidation task only CONFIRMS the existing stack/current_best; its
        # winner is not a new discovery. Skip the accept/lift path so a rebench
        # (e.g. geak_revalidate) never appends a duplicate optimization_stack
        # entry or re-lifts current_best.
        if isinstance(winners, list) and winners and not is_revalidation_task:
            for winner in winners:
                if not isinstance(winner, dict):
                    continue
                accepted = dict(winner)
                accepted.setdefault("accepted_at_round", round_id)
                accepted.setdefault("provenance", winner.get("provenance") or "llm_direct")
                self.shared_state.record_explore_accepted(accepted)
                # A specialist-provenance KEEP zeroes that domain's rounds_since_last_keep counter.
                prov = str(accepted.get("provenance") or "")
                if prov.startswith("specialist:"):
                    try:
                        self.shared_state.note_domain_keep(prov.split(":", 1)[1].strip())
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "depth: note_domain_keep failed for provenance=%r",
                            prov,
                        )
                changed = True
            # 4. Lift the best winner into current_best / optimization_stack.
            if isinstance(best_winner, dict) and isinstance(best_tput, (int, float)) and best_tput > 0:
                best_winner = dict(best_winner)
                if task is not None:
                    best_winner["task_id"] = str(task.task_id or "")
                explore_gap_cid = (
                    str((task.params or {}).get("gap_canonical_id") or "").strip() if task is not None else ""
                )
                promoted = self._lift_to_current_best(
                    "explore",
                    float(best_tput),
                    best_winner,
                    gap_canonical_id=explore_gap_cid,
                )
                changed = True
        try:
            self.shared_state.note_explore_outcome(promoted=promoted)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("depth: note_explore_outcome failed")
        if promoted:
            # explore inlines the per-KEEP rebench: promote into cumulative_gain_validated +
            # advance validated_stack_len so the unvalidated-stack guard clears.
            if self.shared_state.baseline_tput > 0 and isinstance(best_tput, (int, float)) and best_tput > 0:
                self._update_cumulative_gain_validated(best_tput)
                # Watermark refresh: enqueue a fresh roofline once projected tput crosses +10%.
                await self._maybe_enqueue_watermark_roofline(
                    reason="explore_keep_watermark",
                )
        else:
            changed = True
        if promoted:
            audit_decision = "promoted"
        elif winners and not is_revalidation_task:
            audit_decision = "no_promote"
        else:
            audit_decision = "discarded"
        audit_extras = {
            "round_id": round_id,
            "winners_count": (len(winners) if isinstance(winners, list) else 0),
            "losers_count": len(result.get("losers") or []),
            "skipped_dup_count": len(result.get("skipped_dup") or []),
            "best_variant_name": (best_winner.get("name") if isinstance(best_winner, dict) else None),
            "best_gain_pct_vs_base": result.get("best_gain_pct"),
            "output_throughput": best_tput,
            "keep_unstable_count": len(result.get("keep_unstable_in_stack") or []),
            "explore_grid_exhausted": bool(result.get("explore_grid_exhausted")),
        }
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_integrate_patch(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote an integrate_patch result: on KEEP lift current_best; clear pending_integrate."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        status = str(result.get("status") or "")
        new_tput = result.get("output_throughput")
        kept_flag = status == "kept" and isinstance(new_tput, (int, float)) and float(new_tput) > 0
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        prebaseline_enablement = bool(
            kept_flag
            and float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0) <= 0.0
            and (
                result.get("enablement")
                or task_params.get("enablement")
                or task_params.get("enablement_landing")
            )
        )
        lifted = False
        if kept_flag:
            specialist_task_id = str(
                result.get("specialist_task_id")
                or task_params.get("specialist_task_id")
                or ""
            )
            origin_domain = str(
                task_params.get("domain")
                or task_params.get("source_domain")
                or result.get("domain")
                or ""
            ).strip()
            origin_provenance = str(
                task_params.get("provenance")
                or result.get("provenance")
                or ""
            ).strip()
            if origin_domain and not origin_provenance.startswith("specialist:"):
                origin_provenance = f"specialist:{origin_domain}"
            source_phase = str(
                task_params.get("source_phase")
                or result.get("source_phase")
                or ""
            ).strip()
            gap_canonical_id = str(
                task_params.get("gap_canonical_id")
                or result.get("gap_canonical_id")
                or ""
            ).strip()
            lift = {
                "name": specialist_task_id or "integrate_patch_keep",
                "task_id": getattr(task, "task_id", "") if task is not None else "",
                "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
                "extra_envs": dict(
                    result.get("extra_envs_applied")
                    or result.get("config_changes_applied")
                    or {}
                ),
                "tput": float(new_tput),
                "workspace": result.get("workspace"),
                "provenance": origin_provenance or "integrate_patch",
                "scope": "source_patch",
                # Durable source-layer handles so current_best stays relaunchable
                # and reproducible in the GEAK baseline.
                "source_snapshot": result.get("source_snapshot") or "",
                "source_manifest": result.get("source_manifest") or "",
                "target_files": [
                    str(path)
                    for path in (result.get("target_files") or [])
                    if str(path).strip()
                ],
                "framework_root": result.get("framework_root") or "",
                "base_sha": result.get("base_sha") or "",
            }
            if source_phase:
                lift["source_phase"] = source_phase
            if origin_domain:
                lift["domain"] = origin_domain
            if task_params.get("gap_layer"):
                lift["gap_layer"] = str(task_params.get("gap_layer"))
            if task_params.get("framework_agent_authoring"):
                lift["framework_agent_authoring"] = True
            if prebaseline_enablement:
                lift["baseline_enablement"] = True
                lift["attribution_eligible"] = False
                # This patch establishes the runnable baseline environment. Keep
                # it in the configuration stack for reproducibility, but mark it
                # ineligible for gain attribution because no runnable before
                # measurement exists.
                lifted = self._lift_to_current_best(
                    "integrate_patch",
                    float(new_tput),
                    lift,
                    gap_canonical_id=gap_canonical_id,
                )
                log.info(
                    "integrate_patch KEEP accepted as pre-baseline enablement; "
                    "retained in config stack without gain attribution "
                    "(task=%s specialist=%s)",
                    str(getattr(task, "task_id", "") or ""),
                    specialist_task_id,
                )
            else:
                lifted = self._lift_to_current_best(
                    "integrate_patch",
                    float(new_tput),
                    lift,
                    gap_canonical_id=gap_canonical_id,
                )
                if lifted and self.shared_state.baseline_tput > 0:
                    self._update_cumulative_gain_validated(new_tput)
                    self.shared_state.resume_pending_revalidation = False
                    await self._maybe_enqueue_watermark_roofline(
                        reason="integrate_keep_watermark",
                    )
            changed = True
        # Clear the pending_integrate sentinel after the task outcome is observed.
        if isinstance(getattr(self.shared_state, "pending_integrate", None), dict):
            pending = self.shared_state.pending_integrate
            if not pending or str(pending.get("task_id") or "") in {
                "",
                str(getattr(task, "task_id", "") or ""),
            }:
                self.shared_state.pending_integrate = {}
                changed = True
        if prebaseline_enablement:
            audit_decision = "enablement_accepted" if lifted else "no_promote"
        elif lifted:
            audit_decision = "promoted"
        elif kept_flag:
            audit_decision = "no_promote"
        else:
            audit_decision = "discarded"
        audit_extras = {
            "status": status,
            "specialist_task_id": result.get("specialist_task_id"),
            "output_throughput": new_tput,
            "delta_pct": result.get("delta_pct"),
            "prebaseline_enablement": prebaseline_enablement,
            "accuracy_pass": result.get("accuracy_pass"),
            "patches_applied": result.get("patches_applied") or [],
            "patches_reverted": result.get("patches_reverted") or [],
            # Enablement eval-origin verdict fields for history.
            "correctness_verified": result.get("correctness_verified"),
            "enablement_eval_failure_kind": result.get("enablement_eval_failure_kind"),
            "enablement_observed_accuracy": result.get("enablement_observed_accuracy"),
            "provisional": result.get("provisional"),
        }
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_framework_agent(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a framework_agent candidate: progress row, batch max-gain stat, KEEP lift."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # FRAMEWORK per-candidate result: append a progress row, update the batch
        # max-gain stat, and on KEEP lift to current_best + validated gain + watermark.
        status = str(result.get("status") or "")
        candidate = result.get("candidate") or {}
        cand_id = self._framework_candidate_key(candidate if isinstance(candidate, dict) else None)
        # Silent apply/bench failure: recover the candidate key from task
        # params and coerce the status so the row is a real terminal verdict
        # the pump can dedup on, not a blank row keyed on "".
        if not cand_id and task is not None:
            task_cand = (getattr(task, "params", None) or {}).get("candidate")
            cand_id = self._framework_candidate_key(task_cand if isinstance(task_cand, dict) else None)
        if not status:
            status = "no_result_failed"
        batch_id = str(
            result.get("batch_id")
            or candidate.get("batch_id")
            or ((getattr(task, "params", None) or {}).get("batch_id") if task is not None else "")
            or ""
        )
        delta_pct = result.get("delta_pct")
        new_tput = result.get("output_throughput")
        kept_flag = status == "kept"
        progress_entry = {
            "candidate_id": cand_id,
            "pr_url": str(candidate.get("pr_url") or ""),
            "status": status,
            "pre_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
            "post_tput": float(new_tput) if isinstance(new_tput, (int, float)) else 0.0,
            "gain_pct": float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0,
            "kept": kept_flag,
            "batch_id": batch_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle": int(getattr(self.shared_state, "macro_cycle", 0) or 0),
        }
        if not isinstance(self.shared_state.framework_agent_phase_progress, list):
            self.shared_state.framework_agent_phase_progress = []
        self.shared_state.framework_agent_phase_progress.append(progress_entry)
        try:
            from ..framework.artifacts import write_decision_json

            write_decision_json(
                self.session_dir,
                candidate_id=cand_id,
                batch_id=batch_id,
                status=status,
                kept=kept_flag,
                provenance="raw_diff",
                reason=str(result.get("reason") or ""),
                gain_pct=(float(delta_pct) if isinstance(delta_pct, (int, float)) else None),
                accuracy_pass=result.get("accuracy_pass"),
                extra={"workspace": str(result.get("workspace") or "")},
            )
        except Exception:  # noqa: BLE001
            log.debug("FRAMEWORK: executor decision.json write failed", exc_info=True)
        # Update batch max-gain rolling stat (for the plateau judge).
        batches = getattr(self.shared_state, "framework_agent_batches", None) or []
        if isinstance(batches, list) and batches:
            for entry in reversed(batches):
                if isinstance(entry, dict) and str(entry.get("batch_id") or "") == batch_id:
                    prev = float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
                    gain = float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0
                    if gain > prev:
                        entry["max_gain_pct_observed_in_batch"] = gain
                    break
        changed = True
        lifted = False
        if kept_flag and isinstance(new_tput, (int, float)) and new_tput > 0:
            lift = {
                "name": f"framework:{cand_id}",
                "variant_name": cand_id,
                "task_id": getattr(task, "task_id", "") if task is not None else "",
                "candidate_extra_server_args": "",
                "extra_envs": {},
                "workspace": result.get("workspace"),
                # Direct framework candidates are owned by FRAMEWORK_AGENT even
                # if writeback runs after the state machine has advanced.
                "source_phase": "FRAMEWORK_AGENT",
                "provenance": "framework_agent",
            }
            lifted = self._lift_to_current_best("framework", float(new_tput), lift)
            if lifted and self.shared_state.baseline_tput > 0:
                self._update_cumulative_gain_validated(new_tput)
                await self._maybe_enqueue_watermark_roofline(
                    reason="framework_keep_watermark",
                )
        if lifted:
            audit_decision = "promoted"
        elif kept_flag:
            audit_decision = "no_promote"
        else:
            audit_decision = "discarded"
        audit_extras = {
            "candidate_id": cand_id,
            "batch_id": batch_id,
            "status": status,
            "delta_pct": delta_pct,
            "output_throughput": new_tput,
            "kept": kept_flag,
        }
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_sweep(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a sweep result: self-audit + record_sweep + save; discovery-only, never promotes."""
        outcome.early_return = True
        pareto = result.get("pareto_front") or []
        self.shared_state.record_action_attempt(
            action="sweep",
            task_id=getattr(task, "task_id", "") if task is not None else "",
            status="succeeded",
            decision="discarded",
            result=result,
            extras={
                "grid_size": result.get("grid_size"),
                "best_overall": result.get("best_overall"),
                "best_for_each_conc": result.get("best_for_each_conc"),
                "pareto_front_size": (len(pareto) if isinstance(pareto, list) else None),
            },
        )
        self.shared_state.record_sweep(result)
        # Sweep is discovery-only (never promotes) and must not mutate params_no_promote_streak.
        self.shared_state.save(self.session_dir)
        # SWEEP post-hook: chain conc_sweep after a succeeded sweep when opted in.
        if getattr(self.shared_state, "conc_sweep_enabled", False) and result.get("status") == "succeeded":
            try:
                await self._enqueue_internal_conc_sweep_task(
                    reason="post_sweep",
                )
            except Exception:  # noqa: BLE001
                log.exception("conc_sweep: post-sweep enqueue raised (non-fatal)")

    async def _promote_conc_sweep(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a conc_sweep result: self-audit + record_conc_sweep + save; discovery-only."""
        outcome.early_return = True
        self.shared_state.record_action_attempt(
            action="conc_sweep",
            task_id=getattr(task, "task_id", "") if task is not None else "",
            status=str(result.get("status") or "succeeded"),
            decision="discarded",
            result=result,
            extras={
                "was_skipped": bool(result.get("was_skipped", False)),
                "skip_reason": result.get("skip_reason"),
                "budget_exhausted": bool(result.get("budget_exhausted", False)),
                "total_budget_sec": result.get("total_budget_sec"),
                "elapsed_sec": result.get("elapsed_sec"),
                "best_speedup": ((result.get("summary") or {}).get("best_speedup")),
                "best_conc": ((result.get("summary") or {}).get("best_conc")),
                "successful_pairs": ((result.get("summary") or {}).get("successful_pairs")),
                "report_path": result.get("report_json_path"),
            },
        )
        # Write last_conc_sweep so exit_normal_sweep can fire conc_sweep_done.
        self.shared_state.record_conc_sweep(result)
        self.shared_state.save(self.session_dir)

    # ------------------------------------------------------------------
    # Resume / replay (folded in from the former ResumeCollaborator).
    # Three semantic boundaries live below: the live-promote / replay path
    # (``_replay_keep_from_result``), the resume-reconcile path
    # (``_resume_consistency_pass`` + its recover helpers), and the
    # current_best lift path (``_materialize_stack_config_for_resume`` /
    # ``build_env_spec``). Methods keep bare ``self.<name>`` access; tests
    # monkeypatch them via ``coord.writeback.<name>`` (or bare-name
    # ``_DELEGATED`` on the coordinator).
    # ------------------------------------------------------------------
    def _detect_resume_state(self) -> dict[str, Any]:
        """Synchronously inspect persistence to determine if this is a resume (non-blocking).

        Returns:
            A dict with ``is_resume``, ``event_count``, ``state_json_present``
            and ``rebuilt`` (the last set later by :meth:`replay_for_resume`).
        """
        ev_count = self.bus.db.fetchone_sync("SELECT COUNT(*) AS c FROM events")
        events_present = (int(ev_count["c"]) if ev_count else 0) > 0
        state_path = SharedState.state_path(self.session_dir)
        return {
            "is_resume": events_present or state_path.exists(),
            "event_count": int(ev_count["c"]) if ev_count else 0,
            "state_json_present": state_path.exists(),
            "rebuilt": False,  # set by replay_for_resume()
        }

    async def replay_for_resume(self) -> dict[str, Any]:
        """Walk the event log to reconstruct ``CoordinatorState.pending_proposals``. Idempotent; a proposal is undecided when no review_verdict targets it.

        Returns:
            A dict summarising the replay: ``is_resume``, ``event_count``,
            ``state_json_present``, ``pending_restored`` (count rebuilt) and
            ``verdicts_seen``.
        """
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)

        decided_ids: set[str] = set()
        verdict_by_target: dict[str, str] = {}
        for v in verdicts:
            target = v.payload.get("target_proposal_msg_id")
            if not target:
                continue
            # Verdicts with a verdict_map but no summary are treated as needs_review.
            summary = v.payload.get("verdict") or ""
            if not summary and isinstance(v.payload.get("verdict_map"), dict):
                summary = "needs_review"
            verdict_by_target[target] = summary
            decided_ids.add(target)

        rebuilt = 0
        self.state.pending_proposals.clear()
        for p in proposal_msgs:
            if p.msg_id in decided_ids:
                continue
            payload = p.payload or {}
            self.state.pending_proposals[p.msg_id] = PendingProposal(
                proposal_msg_id=p.msg_id,
                from_agent=p.from_agent,
                action_name=str(payload.get("action_name", "")),
                predicted_gain_pct=float(payload.get("predicted_gain_pct", 0.0)),
                payload=dict(payload),
            )
            rebuilt += 1

        self._resumed_from["rebuilt"] = True
        self._resumed_from["pending_restored"] = rebuilt
        return {
            "is_resume": self._resumed_from["is_resume"],
            "event_count": self._resumed_from["event_count"],
            "state_json_present": self._resumed_from["state_json_present"],
            "pending_restored": rebuilt,
            "verdicts_seen": len(verdicts),
        }

    def _materialize_stack_config_for_resume(self) -> dict[str, Any]:
        """Rebuild cumulative launch args/envs from ``optimization_stack``."""
        stack = [e for e in (getattr(self.shared_state, "optimization_stack", []) or []) if isinstance(e, dict)]
        args = ""
        envs: dict[str, str] = {}
        overlay = ""
        tput: float | None = None
        variant_name = ""
        action = "resume_reconstructed"
        workspace = None
        for entry in stack:
            candidate = str(entry.get("candidate_extra_server_args") or "").strip()
            full = str(entry.get("extra_server_args") or "").strip()
            args = _merge_cumulative_extra_server_args(args, candidate, full)
            raw_envs = entry.get("extra_envs") or {}
            if isinstance(raw_envs, Mapping):
                for k, v in raw_envs.items():
                    ks = str(k)
                    # A flag mis-stored under extra_envs (e.g. a ``--compilation-config``
                    # key from an integrate_patch entry) is a SERVER ARG, not an env
                    # var — the grid runner would otherwise inject it verbatim as an
                    # env the backend ignores, silently dropping it from the rebuilt
                    # config. Route any ``-``-prefixed key back into extra_server_args
                    # so the materialized stack reproduces the real launch. General:
                    # keyed on the ``-`` prefix, never on a specific flag name.
                    if ks.startswith("-"):
                        tok = ks if v in ("", None) else f"{ks}={v}"
                        args = _merge_cumulative_extra_server_args(args, "", tok)
                    else:
                        envs[ks] = str(v)
            # Carry the authored-kernel overlay (PYTHONPATH prefix) so a native
            # rebuild of an overlay winner actually loads the built kernels
            # instead of measuring the un-optimized stack. Last non-empty wins.
            entry_overlay = str(entry.get("final_overlay") or "").strip()
            if entry_overlay:
                overlay = entry_overlay
            if isinstance(entry.get("tput"), (int, float)) and float(entry["tput"]) > 0:
                tput = float(entry["tput"])
            variant_name = str(entry.get("variant_name") or variant_name or "")
            action = str(entry.get("action") or action)
            workspace = entry.get("workspace") or workspace
        return {
            "action": action,
            "variant_name": variant_name,
            "extra_server_args": args,
            "extra_envs": envs,
            "final_overlay": overlay,
            "tput": tput,
            "workspace": workspace,
            "optimization_stack": stack,
        }

    def build_env_spec(self) -> dict[str, Any]:
        """Fully-reproducible descriptor of ``current_best``'s launch environment.

        Layers, in the order a consumer must apply them to reconstruct the exact
        stack ``current_best`` was measured on:

          * ``config``  — cumulative server args + env vars (the reversible layer).
          * ``source_snapshots`` — ordered durable source-layer snapshots
            (``scope=source_patch`` entries), each a self-contained directory
            (see :mod:`source_snapshot`) that reconstructs the patched framework
            tree independent of the mutable live checkout.
          * ``overlay_pythonpath`` — the authored-kernel overlay prefix.
          * ``launch_recipe`` — the baseline Magpie recipe to launch from.

        This is the single source of truth the GEAK handoff forwards so the
        baseline ref is materialized from the SAME layers as ``current_best``
        (not just its flags/env), closing the cross-harness baseline gap.
        """
        materialized = self._materialize_stack_config_for_resume()
        stack = [e for e in (getattr(self.shared_state, "optimization_stack", []) or []) if isinstance(e, dict)]
        source_snapshots: list[dict[str, Any]] = []
        for entry in stack:
            if entry.get("scope") != "source_patch":
                continue
            snap = str(entry.get("source_snapshot") or "").strip()
            if not snap:
                # A source_patch with no durable snapshot (e.g. a pre-fix legacy
                # KEEP) is surfaced so the consumer can flag an unreproducible
                # baseline rather than silently launch a weaker stock tree.
                source_snapshots.append(
                    {
                        "id": str(entry.get("variant_name") or entry.get("name") or ""),
                        "snapshot_dir": "",
                        "framework_root": str(entry.get("framework_root") or ""),
                        "base_sha": str(entry.get("base_sha") or ""),
                        "reproducible": False,
                    }
                )
                continue
            source_snapshots.append(
                {
                    "id": str(entry.get("variant_name") or entry.get("name") or ""),
                    "snapshot_dir": snap,
                    "framework_root": str(entry.get("framework_root") or ""),
                    "base_sha": str(entry.get("base_sha") or ""),
                    "reproducible": True,
                }
            )
        # FULL resolved engine config (not just the current_best delta): the
        # complete server-launch flag set the orchestrator actually ran, scraped
        # from the authoritative launched argv. This is what closes the CONFIG
        # layer of the cross-harness baseline gap — mem-fraction, radix cache,
        # chunked-prefill and every other engine knob the recipe/delta never
        # carried. ``extra_server_args``/``extra_envs`` remain the current_best
        # DELTA (a consumer merges the delta on top, delta winning on conflict).
        server_launch_flags = ""
        try:
            cb_now = getattr(self.shared_state, "current_best", None)
            _target_tput = float((cb_now or {}).get("tput") or 0.0) if isinstance(cb_now, Mapping) else 0.0
            server_launch_flags = _scrape_resolved_launch_flags(
                getattr(self, "session_dir", ""),
                str(os.environ.get("FRAMEWORK", "") or "sglang"),
                target_tput=_target_tput,
            )
        except Exception:  # noqa: BLE001 — additive; never block env_spec
            server_launch_flags = ""
        return {
            "schema_version": 1,
            "config": {
                "extra_server_args": materialized.get("extra_server_args") or "",
                "extra_envs": dict(materialized.get("extra_envs") or {}),
                # Authoritative, COMPLETE engine flags (run-specific stripped);
                # empty => consumer keeps its own adapter defaults (prior behavior).
                "server_launch_flags": server_launch_flags,
            },
            "source_snapshots": source_snapshots,
            "overlay_pythonpath": materialized.get("final_overlay") or "",
            "launch_recipe": str(getattr(self.shared_state, "baseline_config_path", "") or ""),
        }

    async def _resume_consistency_pass(self) -> dict[str, Any]:
        """One-shot resume audit + recovery for stack/current_best consistency.

        Order matters: recover half-applied / orphaned KEEPs FIRST (they mutate
        the stack), then reconcile ``current_best`` against the resulting stack,
        then compensate the validation watermark by enqueuing a single
        full-stack end-to-end rebench. Idempotent — only runs on a resumed
        session and every recovery step dedupes, so a second pass is a no-op.
        """
        if not self._resumed_from.get("is_resume"):
            return {"skipped": True, "reason": "not_resume"}
        state = self.shared_state
        report: dict[str, Any] = {
            "skipped": False,
            "fixes": [],
            "warnings": [],
        }
        # (1) Half-applied integrate window: replay the
        # missing stack append or roll back the partial patch BEFORE anything
        # reads the stack, so the rest of the pass sees the recovered truth.
        await self._resume_recover_pending_integrate(report)
        # (1b) In-flight targeted build: an off-loop compile cannot survive a
        # coordinator restart, so kill the orphan group, GC its attempt dir,
        # sweep its jit locks, fail the row, and clear the sentinel.
        await self._resume_recover_pending_targeted_build(report)
        # (1c) Orphaned revalidation tasks: if enablement_validation_pending is set
        # but the tracked revalidation task is already terminal, clear the pending
        # flag and rearm the stall counter so a fresh revalidation can be enqueued.
        await self._resume_recover_pending_revalidation(report)
        # (2) Orphaned KEEPs: replay integrate_patch KEEPs
        # that crashed before the append landed; surface ambiguous ones loudly.
        await self._resume_recover_orphaned_keeps(report)

        # (3) current_best <-> stack reconcile (after 1/2 may have grown stack).
        stack = [e for e in (getattr(state, "optimization_stack", []) or []) if isinstance(e, dict)]
        cb = state.current_best if isinstance(state.current_best, dict) else {}
        if stack:
            rebuilt = self._materialize_stack_config_for_resume()
            cb_args = str(cb.get("extra_server_args") or "")
            cb_envs = (
                {str(k): str(v) for k, v in (cb.get("extra_envs") or {}).items()}
                if isinstance(cb.get("extra_envs"), Mapping)
                else {}
            )
            if cb_args != rebuilt["extra_server_args"] or cb_envs != rebuilt["extra_envs"]:
                # The append-only stack is authoritative; a disagreeing
                # current_best is the inconsistency, recorded distinctly from the
                # rebuild fix so operators can see a stale best was detected.
                report["warnings"].append(
                    {
                        "kind": "resume_inconsistent_current_best",
                        "current_best_args": cb_args,
                        "stack_args": rebuilt["extra_server_args"],
                    }
                )
                new_cb = dict(cb)
                new_cb.update(
                    {
                        "action": rebuilt["action"],
                        "variant_name": rebuilt["variant_name"],
                        "extra_server_args": rebuilt["extra_server_args"],
                        "extra_envs": rebuilt["extra_envs"],
                        "optimization_stack": list(stack),
                        "source": "resume_consistency_rebuild_from_stack",
                    }
                )
                if rebuilt["tput"] is not None and not isinstance(new_cb.get("tput"), (int, float)):
                    new_cb["tput"] = rebuilt["tput"]
                if rebuilt["workspace"] and not new_cb.get("workspace"):
                    new_cb["workspace"] = rebuilt["workspace"]
                state.current_best = new_cb
                report["fixes"].append("rebuilt_current_best_config_from_stack")
        elif cb:
            report["warnings"].append({"kind": "current_best_without_stack"})

        # (4) Validation-watermark compensation: unvalidated
        # KEEPs (claimed gain not yet end-to-end confirmed) → flag + enqueue ONE
        # full-stack rebench. The flag + watermark are reconciled from the
        # measured tput when that rebench promotes (see _promote_to_shared_state).
        stack = [e for e in (getattr(state, "optimization_stack", []) or []) if isinstance(e, dict)]
        vlen = int(getattr(state, "cumulative_gain_validated_stack_len", 0) or 0)
        if vlen < len(stack):
            state.resume_pending_revalidation = True
            report["warnings"].append(
                {
                    "kind": "resume_unvalidated_keeps",
                    "validated_stack_len": vlen,
                    "stack_len": len(stack),
                }
            )
            try:
                fix = await self._enqueue_internal_stack_rebench(reason="resume_unvalidated_keeps")
                report["fixes"].append({"kind": "queued_resume_stack_rebench", **fix})
            except Exception:  # noqa: BLE001
                log.exception("Coordinator: failed to enqueue resume stack rebench")
                report["warnings"].append({"kind": "resume_stack_rebench_enqueue_failed"})

        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: resume consistency save failed")
        await self._record_observation("coordinator", "observation", {"kind": "resume_consistency", **report})
        return report

    def _replay_keep_from_result(self, kind: str, result: dict[str, Any]) -> bool:
        """Replay a recorded KEEP delegated-result into current_best/stack.

        Reconstructs the winning-variant dict from a persisted ``delegated_result``
        and routes it through :meth:`_lift_to_current_best`, which dedupes by
        ``(action, variant_name)`` — so replay is idempotent. Used by both the
        pending-integrate (Gap C) and orphaned-KEEP (Gap B) resume recovery
        paths. Returns ``True`` only when a new stack entry was appended.

        Args:
            kind: The originating action kind (``integrate_patch`` / ``explore``
                / ``framework``).
            result: The recorded delegated result payload for that KEEP.

        Returns:
            ``True`` when the replay appended a new stack entry, else ``False``.
        """
        if not isinstance(result, dict):
            return False
        tput = result.get("output_throughput")
        if not (isinstance(tput, (int, float)) and float(tput) > 0):
            return False
        if kind == "explore":
            bv_src = result.get("best_variant")
            if not isinstance(bv_src, dict) or not bv_src.get("name"):
                return False
            bv = dict(bv_src)
        elif kind == "integrate_patch":
            sid = str(result.get("specialist_task_id") or "")
            if not sid:
                return False
            domain = str(
                result.get("domain") or result.get("source_domain") or ""
            ).strip()
            provenance = str(result.get("provenance") or "").strip()
            if domain and not provenance.startswith("specialist:"):
                provenance = f"specialist:{domain}"
            bv = {
                "name": sid,
                "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
                "extra_envs": dict(
                    result.get("extra_envs_applied")
                    or result.get("config_changes_applied")
                    or {}
                ),
                "tput": float(tput),
                "workspace": result.get("workspace"),
                "provenance": provenance or "integrate_patch",
                "scope": "source_patch",
                # Same durable source-layer handles as the primary KEEP lift so a
                # source_patch recovered on THIS path is equally reproducible in
                # the GEAK baseline (no path is left snapshot-less).
                "source_snapshot": result.get("source_snapshot") or "",
                "source_manifest": result.get("source_manifest") or "",
                "target_files": [
                    str(path)
                    for path in (result.get("target_files") or [])
                    if str(path).strip()
                ],
                "framework_root": result.get("framework_root") or "",
                "base_sha": result.get("base_sha") or "",
            }
            source_phase = str(result.get("source_phase") or "").strip()
            gap_layer = str(result.get("gap_layer") or "").strip()
            if source_phase:
                bv["source_phase"] = source_phase
            if domain:
                bv["domain"] = domain
            if gap_layer:
                bv["gap_layer"] = gap_layer
            if result.get("framework_agent_authoring"):
                bv["framework_agent_authoring"] = True
        else:
            return False
        before = len(self.shared_state.optimization_stack or [])
        if not self._lift_to_current_best(
            kind,
            float(tput),
            bv,
            gap_canonical_id=str(result.get("gap_canonical_id") or ""),
        ):
            return False
        return len(self.shared_state.optimization_stack or []) > before

    def _resume_rollback_pending_integrate(self, pending: dict[str, Any]) -> dict[str, Any]:
        """Reverse-apply a half-applied integrate patch set (Gap C rollback).

        Best-effort ``git apply -R`` of every patch recorded on the
        ``pending_integrate`` sentinel into the framework source tree, so a
        crash AFTER ``git apply`` but BEFORE the bench/KEEP cannot leak a partial
        change into later launches. A patch that is not currently applied simply
        fails the reverse ``--check`` and is reported, not retried.

        Args:
            pending: The ``pending_integrate`` sentinel dict.

        Returns:
            A summary ``{"reversed": [...], "failed": [...]}``.
        """
        from ..actions.executors.integrate_patch import _git_apply_reverse

        summary: dict[str, Any] = {"reversed": [], "failed": []}
        # Discard a half-provisioned attempt venv so a crash mid-provision
        # cannot leak a multi-GB dir. Independent of the patch rollback below.
        attempt_venv_root = str(pending.get("attempt_venv_root") or "").strip()
        if attempt_venv_root:
            gc_root = str(Path(attempt_venv_root).parent)
            if self._gc_attempt_runtime(gc_root):
                summary["attempt_runtime_gc"] = gc_root
        root = str(pending.get("framework_source_root") or "").strip()
        patches = [str(p) for p in (pending.get("patches") or []) if str(p).strip()]
        if not root or not patches:
            return summary
        root_path = Path(root)
        for patch in patches:
            try:
                ok, err = _git_apply_reverse(root_path, Path(patch))
            except Exception as exc:  # noqa: BLE001 — rollback is best-effort
                summary["failed"].append({"patch": patch, "error": repr(exc)})
                continue
            if ok:
                summary["reversed"].append(patch)
            else:
                summary["failed"].append({"patch": patch, "error": err})
        return summary

    @staticmethod
    def _gc_attempt_runtime(attempt_dir: str) -> bool:
        """Remove an attempt-runtime dir (best-effort).

        Returns True when a directory was present and removal was attempted.
        """
        import shutil

        path = Path(str(attempt_dir or "").strip())
        if not attempt_dir or not path.exists():
            return False
        try:
            shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception:  # noqa: BLE001 — GC is best-effort
            return False

    async def _resume_recover_pending_integrate(self, report: dict[str, Any]) -> None:
        """Recover a crashed integrate_patch window from the sentinel (Gap C).

        Three-way decision keyed on whether a ``kept`` delegated-result exists
        for the sentinel's task: replay the missing append (crashed after KEEP),
        roll back the half-applied patch (crashed after apply, before KEEP), or
        clear a stale sentinel. The sentinel is always cleared afterwards.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        state = self.shared_state
        pending = getattr(state, "pending_integrate", {}) or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        kept_res: dict[str, Any] | None = None
        try:
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                if task_id and str(payload.get("task_id") or "") != task_id:
                    continue
                res = payload.get("result") or {}
                # Require an explicit integrate_patch kind: an empty-kind wildcard
                # could misclassify a non-integrate event that happens to share
                # this task_id as a kept integrate result, skipping rollback of a
                # half-applied patch.
                if (
                    isinstance(res, dict)
                    and str(res.get("kind") or payload.get("kind") or "") == "integrate_patch"
                    and str(res.get("status") or "").lower() == "kept"
                ):
                    kept_res = res
                    break
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: pending_integrate kept-result scan failed")
        if kept_res is not None:
            appended = self._replay_keep_from_result("integrate_patch", kept_res)
            report["fixes"].append(
                {"kind": "replayed_pending_integrate", "task_id": task_id, "appended": bool(appended)}
            )
        else:
            summary = self._resume_rollback_pending_integrate(pending)
            if summary.get("reversed"):
                report["fixes"].append({"kind": "rolled_back_pending_integrate", "task_id": task_id, **summary})
            elif summary.get("failed"):
                report["warnings"].append({"kind": "pending_integrate_rollback_failed", "task_id": task_id, **summary})
            else:
                report["fixes"].append({"kind": "cleared_stale_pending_integrate", "task_id": task_id})
        state.pending_integrate = {}

    async def _resume_recover_pending_targeted_build(self, report: dict[str, Any]) -> None:
        """Reclaim an off-loop build that was in flight when the coordinator died.

        A detached compile cannot be re-adopted across a restart: kill its
        recorded process group, rmtree the attempt dir, sweep its per-attempt
        aiter JIT locks (a killed compile leaves a pid-less lock that wedges
        every later build of that module), mark the row failed with a
        ``timeout`` failure_class for the framework channel, and clear the
        sentinel. Best-effort throughout.
        """
        import shutil
        import signal

        from ..framework.targeted_build import kill_build_pgroup

        state = self.shared_state
        pending = getattr(state, "pending_targeted_build", {}) or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        summary: dict[str, Any] = {"kind": "reclaimed_pending_targeted_build", "task_id": task_id}

        try:
            pgid = int(pending.get("pgid") or 0)
        except (TypeError, ValueError):
            pgid = 0
        if pgid > 0:
            kill_build_pgroup(pgid, sig=signal.SIGKILL)
            summary["killed_pgid"] = pgid

        attempt_root = str(pending.get("attempt_root") or "").strip()
        if attempt_root and Path(attempt_root).exists():
            shutil.rmtree(attempt_root, ignore_errors=True)
            summary["removed_attempt_root"] = attempt_root

        jit_dir = str(pending.get("aiter_jit_dir") or "").strip()
        if jit_dir:
            try:
                from ..actions.executors._aiter_jit import sweep_stale_aiter_locks_if_dead

                sweep_stale_aiter_locks_if_dead(aiter_jit_dir=Path(jit_dir))
                summary["swept_jit_dir"] = jit_dir
            except Exception:  # noqa: BLE001 — sweep is best-effort
                log.debug("resume: targeted-build jit sweep failed for %s", jit_dir, exc_info=True)

        state.enablement_last_build_failure = {
            "failure_class": "timeout",
            "failure_summary": "targeted build interrupted by coordinator restart",
        }
        if task_id:
            try:
                task = await self.tasks.get(task_id)
                if getattr(task, "state", "") == "running":
                    await self.tasks.transition(
                        task_id, "failed", evidence={"failure_class": "resume_interrupted"}
                    )
                    summary["failed_row"] = True
            except Exception:  # noqa: BLE001 — reclaim backstop still applies
                log.debug("resume: targeted-build row fail raced for %s", task_id, exc_info=True)

        state.pending_targeted_build = {}
        report["fixes"].append(summary)

    async def _resume_recover_pending_revalidation(self, report: dict[str, Any]) -> None:
        """Clear stale enablement_validation_pending when the tracked revalidation task is terminal.

        If the coordinator died while a revalidation baseline was running, the
        task row may already be in a terminal state on resume.  Without this
        recovery the pending flag stays set indefinitely and the next
        revalidation cannot be enqueued (tracked_tid is still the old row).
        """
        state = self.shared_state
        if not bool(getattr(state, "enablement_validation_pending", False)):
            return
        tracked_tid = str(getattr(state, "enablement_revalidation_task_id", "") or "").strip()
        if not tracked_tid:
            return
        try:
            from ..state.task_registry import TERMINAL_STATES, TaskNotFound

            try:
                row = await self.tasks.get(tracked_tid)
                is_terminal = row.state in TERMINAL_STATES
            except TaskNotFound:
                is_terminal = True
            if is_terminal:
                state.enablement_validation_pending = False
                state.enablement_revalidation_task_id = ""
                state.enablement_stall_streak = (
                    int(getattr(state, "enablement_stall_streak", 0) or 0) + 1
                )
                state.enablement_dispatched = False
                report["fixes"].append(
                    {"kind": "cleared_orphaned_revalidation_pending", "task_id": tracked_tid}
                )
                log.info(
                    "resume: cleared stale enablement_validation_pending for terminal revalidation task %s",
                    tracked_tid,
                )
        except Exception:  # noqa: BLE001 — best-effort
            log.debug("resume: revalidation pending recovery check failed", exc_info=True)

    async def _resume_recover_orphaned_keeps(self, report: dict[str, Any]) -> None:
        """Recover / surface KEEPs present in the event log but absent from the stack (Gap B).

        ``integrate_patch`` KEEPs are well-defined (a ``kept`` status means the
        single-variant bench + accuracy gate passed and the patch was committed),
        so a kept-but-absent one is a crash before the append landed → replay it
        (idempotent), unless its run workspace is gone → discard + alert. ``explore``
        / ``framework`` KEEPs are ambiguous (KEEP_UNSTABLE eviction can drop a
        kept explore variant from the stack), so they are surfaced as a
        ``medium`` alert rather than resurrected. Whatever the stack ends up as
        is re-validated by the Gap A full-stack rebench.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        state = self.shared_state
        try:
            stack_keys = {
                (str(e.get("action") or ""), str(e.get("variant_name") or ""))
                for e in (state.optimization_stack or [])
                if isinstance(e, dict)
            }
            seen: set[tuple[str, str]] = set()
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                kind = str(payload.get("kind") or "")
                res = payload.get("result") or {}
                if not isinstance(res, dict) or str(res.get("status") or "").lower() != "kept":
                    continue
                if kind == "integrate_patch":
                    variant = str(res.get("specialist_task_id") or "")
                elif kind == "framework_agent":
                    cand = res.get("candidate") or {}
                    variant = str(
                        (cand.get("candidate_id") if isinstance(cand, dict) else "")
                        or (cand.get("pr_url") if isinstance(cand, dict) else "")
                        or ""
                    )
                elif kind == "explore":
                    bv = res.get("best_variant") or {}
                    variant = str((bv.get("name") if isinstance(bv, dict) else "") or "")
                else:
                    continue
                key = (kind, variant)
                if not variant or key in stack_keys or key in seen:
                    continue
                seen.add(key)
                if kind == "integrate_patch":
                    workspace = str(res.get("workspace") or "").strip()
                    if workspace and not Path(workspace).exists():
                        report["warnings"].append(
                            {
                                "kind": "orphaned_keep_discarded",
                                "orphan_kind": kind,
                                "variant": variant,
                                "task_id": payload.get("task_id"),
                                "reason": "workspace_missing",
                            }
                        )
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "orphaned_keep_discarded",
                                "severity": "medium",
                                "orphan_kind": kind,
                                "variant": variant,
                            },
                        )
                    elif self._replay_keep_from_result(kind, res):
                        stack_keys.add(key)
                        report["fixes"].append(
                            {"kind": "replayed_orphaned_keep", "orphan_kind": kind, "variant": variant}
                        )
                    else:
                        report["warnings"].append(
                            {"kind": "orphaned_keep_replay_noop", "orphan_kind": kind, "variant": variant}
                        )
                else:
                    # explore / framework: ambiguous vs eviction — never
                    # resurrect; surface for the operator.
                    report["warnings"].append(
                        {
                            "kind": "orphaned_keep",
                            "orphan_kind": kind,
                            "variant": variant,
                            "task_id": payload.get("task_id"),
                        }
                    )
                    await self._record_observation(
                        "coordinator",
                        "observation",
                        {
                            "kind": "orphaned_keep",
                            "severity": "medium",
                            "orphan_kind": kind,
                            "variant": variant,
                        },
                    )
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: orphaned KEEP resume recovery failed")

    async def _enqueue_internal_stack_rebench(self, *, reason: str) -> dict[str, Any]:
        """Enqueue one full-stack end-to-end rebench of the cumulative config (Gap A).

        Builds a single-variant ``explore`` task from the stack-materialized
        launch args/envs, benched against ``baseline_tput`` so the measured
        delta becomes the validated cumulative gain. Tagged
        ``source=resume_stack_revalidate`` so ``_promote_to_shared_state``
        reconciles ``cumulative_gain_validated_stack_len`` + clears
        ``resume_pending_revalidation`` from the measured throughput. Idempotent
        via a fixed idempotency key.

        Args:
            reason: Human-readable reason stamped on the task params.

        Returns:
            A summary ``{"task_id", "existing"}`` or ``{"skipped", "reason"}``.
        """
        # fix-point 7 (2b) — when the win is a GEAK e2e result, source the
        # revalidation config from result.json (the SINGLE source of truth), NOT
        # from stack materialization. This guarantees the same-harness rebench
        # launches byte-for-byte the config GEAK optimized (flags + parsed env +
        # authored overlay), independent of whether the optimization is a MoE
        # tuned-config / kernel / flag winner — no case-by-case markers. The
        # consumer (_promote_to_shared_state) asserts config identity + effect
        # before stamping validated, and falls back to 2a (GEAK harness) on miss.
        ps = self.shared_state.geak_result if isinstance(getattr(self.shared_state, "geak_result", None), dict) else {}
        ps_cfg = ps.get("accepted_config") or {}
        ps_overlay = _normalize_geak_overlay_dir(str(ps.get("final_overlay") or "").strip())
        if str(ps.get("status") or "") == "ok" and (ps_cfg.get("flags") or ps_cfg.get("env") or ps_overlay):
            from ..actions.executors._canonical_fingerprint import canonical_fingerprint

            ps_flags = str(ps_cfg.get("flags") or "").strip()
            ps_envs, _ps_extra_flags = _split_env_and_flags(str(ps_cfg.get("env") or ""))
            if _ps_extra_flags:
                ps_flags = (ps_flags + " " + _ps_extra_flags).strip()
            if ps_flags or ps_envs or ps_overlay:
                # Identity hash uses the SAME (args, envs) contract the grid
                # executor fingerprints with (overlay is NOT part of the hash,
                # matching canonical_fingerprint) so expected == the
                # ran variant's fingerprint by construction, and any executor-side
                # drop/alter of config is caught downstream.
                expected_cfg_hash = canonical_fingerprint(ps_flags, ps_envs)
                params_ps: dict[str, Any] = {
                    "source": "resume_stack_revalidate",
                    "reason": reason,
                    "geak_fallback": True,
                    "expected_cfg_hash": expected_cfg_hash,
                    "grid": [
                        {
                            "name": "geak_revalidate",
                            "extra_args": ps_flags,
                            "extra_envs": dict(ps_envs),
                            "overlay_pythonpath": ps_overlay,
                            "provenance": "geak_revalidate",
                            "note": "same-harness config-identity revalidation of the geak e2e win",
                        }
                    ],
                    # Revalidation reproduces the whole stack, so its gain is
                    # cumulative-vs-baseline, not a delta over current_best.
                    "base_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
                    "enable_stack_rebench": False,
                }
                if self.shared_state.baseline_config_path:
                    params_ps["config_path"] = self.shared_state.baseline_config_path
                task, existing = await self.tasks.create_or_return_existing(
                    kind="explore",
                    params=params_ps,
                    idempotency_key="geak-revalidate",
                )
                try:
                    from hyperloom.inference_optimizer.breakdown.recorder import instrument

                    instrument.record_geak_operation(
                        self.session_dir,
                        stage="rebench_started",
                        macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                        result={
                            **ps,
                            "rebench": {
                                "task_id": task.task_id,
                                "existing": bool(existing),
                                "mode": "orchestrator_same_harness",
                                "expected_cfg_hash": expected_cfg_hash,
                            },
                        },
                        status="running",
                    )
                except Exception:  # noqa: BLE001
                    log.debug("geak v4 rebench recording failed", exc_info=True)
                return {
                    "task_id": task.task_id,
                    "task_state": task.state,
                    "existing": bool(existing),
                    "mode": "geak_2b",
                }

        rebuilt = self._materialize_stack_config_for_resume()
        args = str(rebuilt.get("extra_server_args") or "").strip()
        envs = rebuilt.get("extra_envs") or {}
        overlay = str(rebuilt.get("final_overlay") or "").strip()
        cb_now = self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
        cb_remove = cb_now.get("remove_args")
        cb_unset = cb_now.get("unset_envs")
        cb_replace = str(cb_now.get("args_mode") or "").strip().lower() == "replace"
        if not (args or envs or cb_remove or cb_unset or cb_replace):
            return {"skipped": True, "reason": "empty_stack"}
        params: dict[str, Any] = {
            "source": "resume_stack_revalidate",
            "reason": reason,
            "grid": [
                {
                    "name": "resume_stack_revalidate",
                    "extra_args": args,
                    "extra_envs": dict(envs),
                    # Carry the overlay so an authored-kernel native stack rebuild
                    # loads the built kernels (inert when empty).
                    "overlay_pythonpath": overlay,
                    "provenance": "resume_stack_revalidate",
                    "note": "post-resume full-stack end-to-end revalidation",
                }
            ],
            # Cumulative-vs-baseline, same as the geak revalidation above.
            "base_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
            "enable_stack_rebench": False,
        }
        if cb_remove:
            params["base_remove_args"] = [cb_remove] if isinstance(cb_remove, str) else list(cb_remove or [])
        if cb_unset:
            params["base_unset_envs"] = [cb_unset] if isinstance(cb_unset, str) else list(cb_unset or [])
        if cb_replace:
            params["base_args_mode"] = "replace"
        if self.shared_state.baseline_config_path:
            params["config_path"] = self.shared_state.baseline_config_path
        task, existing = await self.tasks.create_or_return_existing(
            kind="explore",
            params=params,
            idempotency_key="resume-stack-revalidate",
        )
        return {"task_id": task.task_id, "existing": bool(existing)}

    async def _validate_geak_via_geak_harness(self, *, reason: str) -> dict[str, Any]:
        """2a fallback - validate the geak win by REPLAYING it through
        GEAK's own ``bench_e2e.sh`` (the harness that produced the headline
        result), so the optimized config engages BY CONSTRUCTION regardless of
        winner kind (tuned-config / kernel / overlay / flag). Because the replay
        reproduces the optimized config from ``result.json`` directly, a
        ``succeeded`` status is itself the engagement proof. The validated gain
        is the MEASURED replay throughput over the orchestrator's raw baseline
        (``(measured - baseline) / baseline``); GEAK's own ``throughput_speedup``
        / ``hot_geak_speedup`` serves only as a ``> 1.0`` sanity gate, not as the
        reported number. Recorded under a distinct provenance
        (``geak_same_harness_geak``) because the measurement came from GEAK's
        harness rather than the orchestrator's. Used only when 2b (orchestrator
        harness) is inconclusive.

        Args:
            reason: Human-readable reason stamped in logs/return.

        Returns:
            A summary dict describing whether validation succeeded.
        """
        ps = self.shared_state.geak_result if isinstance(getattr(self.shared_state, "geak_result", None), dict) else {}
        if str(ps.get("status") or "") != "ok":
            return {"validated": False, "skipped": True, "reason": "no_geak_result"}
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="geak_harness_fallback",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result={**ps, "fallback_reason": reason},
                status="running",
                validation_source="geak_same_harness_geak",
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 fallback recording failed", exc_info=True)
        am = ps.get("alignment_metrics") or {}
        # Read GEAK's OWN within-harness speedup on the SAME basis it promoted
        # (result.throughput_speedup == cold_geak_speedup when final_basis=="cold",
        # else the hot within-GEAK ratio; see run_e2e final-basis selection). It is
        # used solely to sanity-check that GEAK claimed a win on its promoted basis
        # before the replay measurement is promoted as the headline.
        # Falls back to the explicit within-GEAK ratios when throughput_speedup is
        # missing (older result.json), preferring the promoted basis.
        try:
            geak_sp = float(ps.get("throughput_speedup") or 0.0)
        except (TypeError, ValueError):
            geak_sp = 0.0
        if geak_sp <= 0:
            basis = str(am.get("final_basis") or ps.get("final_throughput_basis") or "hot")
            fallback_key = "cold_geak_speedup" if basis == "cold" else "hot_geak_speedup"
            try:
                geak_sp = float(am.get(fallback_key) or am.get("hot_geak_speedup") or 0.0)
            except (TypeError, ValueError):
                geak_sp = 0.0
        regimes = ps.get("validated_regimes") or []
        reg = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
        try:
            conc = int(reg.get("conc") or 64)
            isl = int(reg.get("isl") or 1024)
            osl = int(reg.get("osl") or 1024)
        except (TypeError, ValueError):
            conc, isl, osl = 64, 1024, 1024
        from hyperloom.inference_optimizer.session.session_paths import runs_dir
        from ..actions.executors._geak_sweep import sweep_via_geak

        try:
            timeout = int(os.environ.get("SWEEP_VARIANT_TIMEOUT_SEC", "").strip() or "2400")
        except (TypeError, ValueError):
            timeout = 2400
        res = await sweep_via_geak(
            result=ps,
            conc_values=[conc],
            isl_osl_configs=[f"{isl}:{osl}"],
            output_root=runs_dir(self.session_dir, "sweep", "revalidate_geak"),
            variant_timeout_sec=timeout,
            repeats=3,
            # Single-point validated replay pins the headline protocol (num_prompts
            # etc.) so it is protocol-identical to the reported result.
            pin_num_prompts=True,
        )
        if str(res.get("status") or "") == "succeeded" and geak_sp > 1.0:
            # Rebench-first: write the headline from the GEAK-harness MEASURED
            # throughput (engages by construction via the launch-script replay),
            # keeping the leaderboard number a same-harness total rather than a
            # self-reported speedup.
            measured = _geak_sweep_measured_tput(res)
            if measured is None:
                log.warning("geak 2a: succeeded sweep but no measurable throughput; candidate stays pending")
                try:
                    from hyperloom.inference_optimizer.breakdown.recorder import instrument

                    instrument.record_geak_operation(
                        self.session_dir,
                        stage="final_validation_failed",
                        macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                        result={**ps, "fallback_result": res, "failure_reason": "missing_measured_throughput"},
                        status="failed",
                        validation_source="geak_same_harness_geak",
                    )
                except Exception:  # noqa: BLE001
                    log.debug("geak v4 missing-measurement recording failed", exc_info=True)
                return {"validated": False, "status": res.get("status"), "reason": reason}
            self._promote_geak_from_candidate(
                ps,
                measured_tput=measured,
                provenance="geak_same_harness_geak",
            )
            base = float(self.shared_state.baseline_tput or 0.0)
            gain_out = ((measured - base) / base * 100.0) if base > 0 else 0.0
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 - defensive
                log.exception("geak 2a: SharedState.save failed")
            return {"validated": True, "gain": gain_out, "reason": reason}
        log.warning(
            "geak 2a fallback did not validate (status=%r geak_speedup=%r reason=%s)",
            res.get("status"),
            geak_sp,
            reason,
        )
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="final_validation_failed",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result={
                    **ps,
                    "fallback_result": res,
                    "failure_reason": reason,
                    "geak_speedup": geak_sp,
                },
                status="failed",
                validation_source="geak_same_harness_geak",
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 failed-validation recording failed", exc_info=True)
        return {"validated": False, "status": res.get("status"), "reason": reason}

    async def _resume_reenter_kernel_if_needed(self) -> None:
        """Idempotently re-fire the KERNEL_AGENT entry hook on resume.

        Phase-entry side effects (the GEAK delegation + its ``result.json``
        crash-recovery) are bound to a phase *transition* via
        ``_on_phase_entered``; a resume only restores ``phase`` from state.json
        and never re-enters the current phase. Without this, a session that
        crashed mid ``KERNEL_AGENT`` sits idle until the phase budget cap fires,
        then hands SWEEP an empty result — the whole delegation is silently lost.

        General across every crash timing (not case-by-case): the decision is
        driven purely by whether THIS KERNEL phase's history row already carries
        a ``geak`` completion record, so it self-classifies:

          * completed-this-phase -> only re-arm (+persist) the ``skip_to_sweep``
            hint the delegation sets, so the phase machine winds down to SWEEP
            with no e2e re-run;
          * not-completed -> re-enter ``_on_enter_kernel``; its own entry guard
            promotes an existing OK ``result.json`` (crash-before-handback) and
            re-runs the e2e only when there is genuinely nothing to recover
            (run_e2e itself then continues from the pinned eval_dir on disk).

        No-op unless resumed while parked in ``KERNEL_AGENT`` with the GEAK
        backend selected.
        """
        from ..phases.machine_state import (
            ESCALATE_HINT_SKIP_TO_SWEEP,
            PHASE_KERNEL_AGENT,
        )

        if not self._resumed_from.get("is_resume"):
            return
        state = self.shared_state
        if (state.phase or "").strip().upper() != PHASE_KERNEL_AGENT:
            return
        if not (self._kernel_enabled() and self._geak_enabled()):
            return
        history = state.phase_history or []
        row = history[-1] if history else {}
        evidence = row.get("evidence") if isinstance(row, dict) else {}
        completed_this_phase = isinstance(evidence, dict) and isinstance(evidence.get("geak"), dict)
        if completed_this_phase:
            # The delegation landed during this phase but the SWEEP transition
            # never persisted (crash between the hook and the next tick). Re-arm
            # the wind-down hint + persist so the phase machine advances.
            cur = str(getattr(state, "pending_escalate_hint", "") or "").strip()
            if cur != ESCALATE_HINT_SKIP_TO_SWEEP:
                state.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)
                try:
                    state.save(self.session_dir)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception("resume: save after re-arming skip_to_sweep failed")
                log.info(
                    "resume: KERNEL GEAK already completed this phase; "
                    "re-armed skip_to_sweep hint (lost before SWEEP transition)."
                )
            return
        log.info(
            "resume: re-entering KERNEL GEAK delegation (no completion "
            "evidence on the current phase row); recover-from-disk or re-run."
        )
        try:
            await self._on_enter_kernel(from_phase="resume")
        except Exception:  # noqa: BLE001 — resume re-entry must never kill the session
            log.exception("resume: KERNEL re-entry hook failed")

    @property
    def resumed_from(self) -> dict[str, Any]:
        """Read-only snapshot of resume detection (set by ``__init__``).

        Returns:
            A copy of the resume-detection dict so callers cannot mutate
            internal state.
        """
        return dict(self._resumed_from)

    # Bounded test interface
    async def _replay_resume_if_needed(self) -> None:
        """Rebuild in-memory state once for a resumed session (replay log + abandon orphan dispatches)."""
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        await self.replay_for_resume()
        await self._resume_consistency_pass()
        # Re-fire the KERNEL delegation hook when resuming parked in KERNEL_AGENT
        # (phase-entry side effects are bound to transitions, not resume). Runs
        # after the consistency pass so current_best/stack are already rebuilt.
        await self._resume_reenter_kernel_if_needed()
