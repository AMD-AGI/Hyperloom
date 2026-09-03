# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-stack validation handler: draining pending KEEP integrates and
running/recovering the positive-needs-review stack e2e validation."""

from __future__ import annotations
import logging as _logging
from datetime import datetime, timezone
from typing import Any
from ..bus.message_bus import Message
from ..kernel._kernel_decisions import _entry_by_kernel_id
from ..state.shared_state import resolve_graded_comparison
from ..state.task_registry import Task
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class KernelStackPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    async def _drain_pending_keep_integrates(self) -> None:
        """Drain pending KEEP integrates inherited from KERNEL so sweep measures full current_best. Cap 10; a dispatch failure sets ``rejected_reason=integrate_dispatch_exception`` on the per-kernel and per-task_key attempt ledgers and flips the queued record to ``dispatch_failed``; only records with no ``task_key`` are also appended to ``rejected_kernel_ids``."""
        from ..kernel.request_handlers import integrate_handler

        state = self.shared_state
        drained = 0
        max_drain = 10
        while drained < max_drain:
            pending_records = state.pending_kernel_integration_records()
            if not pending_records:
                break
            pending = pending_records[0]
            kid = str(pending.get("kernel_id") or "")
            integration_id = str(pending.get("integration_id") or "")
            log.info(
                "SWEEP entry: draining pending KEEP integrate for kernel_id=%s integration_id=%s (drained %d so far)",
                kid,
                integration_id,
                drained,
            )
            try:
                base = float((state.current_best or {}).get("tput") or state.baseline_tput or 0.0)
                result = await integrate_handler(
                    {
                        "kernel_id": kid,
                        "integration_id": integration_id,
                        "task_group_key": str(pending.get("task_group_key") or ""),
                        "identity_route": str(pending.get("identity_route") or ""),
                        "base_tput": base,
                    },
                    session_dir=self.session_dir,
                )
                if isinstance(result, dict) and result.get("status") != "skipped":
                    state.record_kernel_integrate_result(result)
                    if str(result.get("decision") or "").upper() == "KEEP":
                        await self._record_integrate_keep(result)
                state.save(self.session_dir)
            except Exception as exc:  # noqa: BLE001 — never block SWEEP entry
                log.exception(
                    "SWEEP entry: integrate(%s) raised %r; marking rejected to prevent drain loop deadlock",
                    kid,
                    exc,
                )
                if state.rejected_kernel_ids is None:
                    state.rejected_kernel_ids = []
                pending_task_key = str(pending.get("task_key") or "")
                if not pending_task_key and kid not in state.rejected_kernel_ids:
                    state.rejected_kernel_ids.append(kid)
                attempt = _entry_by_kernel_id(state, kid)
                if isinstance(attempt, dict):
                    attempt["rejected_reason"] = "integrate_dispatch_exception"
                stable_attempt = (state.kernel_opt_task_attempts or {}).get(pending_task_key)
                if isinstance(stable_attempt, dict):
                    stable_attempt["rejected_reason"] = "integrate_dispatch_exception"
                queued = (state.pending_kernel_integrations or {}).get(integration_id)
                if isinstance(queued, dict):
                    queued["status"] = "dispatch_failed"
                state.save(self.session_dir)
            drained += 1
        if drained >= max_drain:
            log.warning(
                "SWEEP entry: drain cap (%d) reached; remaining pending "
                "KEEPs will be visible in summary.by_kernel as KEEP_PENDING",
                max_drain,
            )

    def _positive_needs_review_integrates(self) -> list[dict[str, Any]]:
        """Return positive NEEDS_REVIEW integrate entries eligible for stack validation.

        Returns:
            Integrate-attempt entries with a positive best gain that are not
            yet stack-resolved or in progress, sorted by gain descending.
        """
        out: list[dict[str, Any]] = []
        stack_resolved_ids = self._stack_resolved_kernel_ids()
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            kernel_id = str(entry.get("kernel_id") or "").strip()
            if (
                bool(entry.get("stack_resolved"))
                or bool(entry.get("stack_validation_in_progress"))
                or kernel_id in stack_resolved_ids
            ):
                continue
            if str(entry.get("last_decision") or "").upper() != "NEEDS_REVIEW":
                continue
            try:
                best_gain = float(entry.get("best_gain_pct") or 0.0)
            except (TypeError, ValueError):
                best_gain = 0.0
            if best_gain <= 0:
                continue
            patch_path = str(entry.get("patch_path") or "").strip()
            target_file = str(entry.get("target_file") or "").strip()
            if patch_path and target_file and kernel_id:
                out.append(entry)
        out.sort(key=lambda e: float(e.get("best_gain_pct") or 0.0), reverse=True)
        return out

    def _stack_resolved_kernel_ids(self) -> set[str]:
        """Kernel ids already covered by a kept stack validation.

        Returns:
            The set of kernel ids resolved by kept ``integrate`` stack entries.
        """
        resolved: set[str] = set()
        for item in self.shared_state.optimization_stack or []:
            if not isinstance(item, dict):
                continue
            if item.get("action") != "integrate":
                continue
            stack_ids = item.get("stack_kernel_ids")
            if isinstance(stack_ids, list):
                resolved.update(str(kid) for kid in stack_ids if str(kid))
                continue
            kernel_id = str(item.get("kernel_id") or "")
            if "+" in kernel_id:
                resolved.update(kid for kid in kernel_id.split("+") if kid)
        return resolved

    def _mark_stack_validation_entries_resolved(
        self,
        entries: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Mark component NEEDS_REVIEW entries as handled by a kept stack.

        Args:
            entries: The component integrate entries that formed the stack.
            result: The stack-validation result; only a ``KEEP`` decision with
                a stack kernel id triggers marking.
        """
        stack_id = str(result.get("kernel_id") or "")
        decision = str(result.get("decision") or "").upper()
        if decision != "KEEP" or not stack_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        wanted = {
            (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            for entry in entries
            if isinstance(entry, dict)
        }
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            identity = (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            if identity not in wanted:
                continue
            entry["stack_resolved"] = True
            entry["stack_validation_kernel_id"] = stack_id
            entry["stack_decision"] = decision
            entry["stack_resolved_at"] = now
            entry.pop("stack_validation_in_progress", None)

    def _stack_component_identities(
        self,
        entries: list[dict[str, Any]],
    ) -> set[tuple[str, str, str]]:
        """Return (kernel_id, patch_path, target_file) tuples for stack members.

        Args:
            entries: The stack component integrate entries.

        Returns:
            A set of ``(kernel_id, patch_path, target_file)`` identity tuples.
        """
        return {
            (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            for entry in entries
            if isinstance(entry, dict)
        }

    def _mark_stack_validation_in_progress(
        self,
        entries: list[dict[str, Any]],
        stack_id: str,
    ) -> None:
        """Persist an in-flight stack guard before applying patches.

        Args:
            entries: The component integrate entries to guard.
            stack_id: The combined stack kernel id stamped onto each entry.
        """
        now = datetime.now(timezone.utc).isoformat()
        wanted = self._stack_component_identities(entries)
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            identity = (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            if identity not in wanted:
                continue
            entry["stack_validation_in_progress"] = True
            entry["stack_validation_kernel_id"] = stack_id
            entry["stack_validation_started_at"] = now

    def _clear_stack_validation_in_progress(
        self,
        entries: list[dict[str, Any]],
    ) -> None:
        """Clear the in-flight stack guard for component integrate entries.

        Args:
            entries: The component integrate entries whose in-progress guard
                should be cleared.
        """
        wanted = self._stack_component_identities(entries)
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            identity = (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            if identity not in wanted:
                continue
            entry.pop("stack_validation_in_progress", None)

    def _clear_pending_stack_validation_checkpoints(self) -> None:
        """Drop crash-recovery checkpoints once a stack attempt is finished."""
        self.shared_state.pending_stack_validation_result = {}
        self.shared_state.pending_stack_validation_apply_results = []

    async def _recover_interrupted_stack_validation(self) -> bool:
        """Resume or abort a stack validation interrupted by crash.

        Returns:
            ``True`` if an interrupted stack validation was finalized or rolled
            back, ``False`` when there was nothing to recover.
        """
        from ..kernel.request_handlers import _maybe_revert_kernel_patch

        pending = self.shared_state.pending_stack_validation_result
        if isinstance(pending, dict) and pending:
            stack = self._stack_entries_for_validation(
                pending.get("stack_kernel_ids") or [],
                stack_id=str(pending.get("kernel_id") or ""),
            )
            if len(stack) >= 2:
                await self._finalize_stack_validation_outcome(stack, pending)
                return True

        partial_applies = list(
            self.shared_state.pending_stack_validation_apply_results or [],
        )
        in_progress = [
            entry
            for entry in (self.shared_state.kernel_integrate_attempts or {}).values()
            if isinstance(entry, dict) and entry.get("stack_validation_in_progress")
        ]
        if not partial_applies and not in_progress:
            return False

        if partial_applies:
            for applied in reversed(partial_applies):
                revert_r = _maybe_revert_kernel_patch(applied)
                if str(revert_r.get("status") or "") not in {"ok", "skipped"}:
                    log.warning(
                        "stack recovery: revert of partial apply %s returned %s",
                        applied.get("manifest_path"),
                        revert_r.get("status"),
                    )
        if in_progress:
            self._clear_stack_validation_in_progress(in_progress)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)
        log.warning(
            "Recovered interrupted stack validation: reverted partial applies and cleared in-progress guards",
        )
        return True

    def _stack_entries_for_validation(
        self,
        kernel_ids: list[Any],
        *,
        stack_id: str = "",
    ) -> list[dict[str, Any]]:
        """Rebuild component integrate ledger rows for a stack id.

        Args:
            kernel_ids: The component kernel ids to recover.
            stack_id: Fallback ``+``-joined stack id parsed for component ids
                when ``kernel_ids`` is empty.

        Returns:
            The matching integrate-attempt entries, sorted by kernel id.
        """
        wanted_ids = {str(kid) for kid in kernel_ids if str(kid)}
        if not wanted_ids and stack_id:
            wanted_ids = {kid for kid in stack_id.split("+") if kid}
        out: list[dict[str, Any]] = []
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            kid = str(entry.get("kernel_id") or "")
            if kid in wanted_ids:
                out.append(entry)
        out.sort(key=lambda e: str(e.get("kernel_id") or ""))
        return out

    async def _finalize_stack_validation_outcome(
        self,
        stack: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Record stack validation, promote KEEP, and clear recovery checkpoints.

        Args:
            stack: The component integrate entries that formed the stack.
            result: The stack-validation result; a ``KEEP`` decision promotes
                the stack and marks entries resolved.
        """
        self.shared_state.record_kernel_integrate_result(result)
        decision = str(result.get("decision") or "").upper()
        if decision == "KEEP":
            self._mark_stack_validation_entries_resolved(stack, result)
            self.shared_state.save(self.session_dir)
            await self._record_integrate_keep(result)
        else:
            self._clear_stack_validation_in_progress(stack)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)

    async def _maybe_validate_positive_needs_review_stack(self) -> None:
        """Run one E2E stack validation for multiple small positive kernel patches.

        Single-patch ``NEEDS_REVIEW`` is not retried automatically. When two or
        more pending kernel patches individually show positive but sub-threshold
        E2E gain, validate their combined effect once before moving to SWEEP.
        """
        if await self._recover_interrupted_stack_validation():
            return
        entries = self._positive_needs_review_integrates()
        if len(entries) < 2:
            return
        # Avoid two whole-file patches on the same target file.
        seen_targets: set[str] = set()
        stack: list[dict[str, Any]] = []
        for entry in entries:
            target = str(entry.get("target_file") or "")
            if target in seen_targets:
                continue
            seen_targets.add(target)
            stack.append(entry)
        if len(stack) < 2:
            return
        stack_id = "+".join(str(e.get("kernel_id") or "") for e in stack)
        self._mark_stack_validation_in_progress(stack, stack_id)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)
        result = await self._run_kernel_stack_validation_e2e(stack)
        if not isinstance(result, dict):
            self._clear_stack_validation_in_progress(stack)
            self._clear_pending_stack_validation_checkpoints()
            self.shared_state.save(self.session_dir)
            return
        self.shared_state.pending_stack_validation_result = result
        self.shared_state.save(self.session_dir)
        await self._finalize_stack_validation_outcome(stack, result)

    async def _run_kernel_stack_validation_e2e(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply multiple kernel patches, run one E2E benchmark, then keep or revert the stack.

        Args:
            entries: The component integrate entries whose patches are applied
                together for the combined benchmark.

        Returns:
            A result dict with the KEEP/REVERT decision, measured throughput,
            incremental gain, apply/revert sub-results and stack metadata.
        """
        from ..actions.executors.baseline import SBD_INNER_STEP_PARAM, BaselineExecutor

        # Lazy (re-)import so tests can monkeypatch it on the source module.
        from ..actions.executors.benchmark_result import is_valid_measurement  # noqa: F811
        from ..kernel.patch_lifecycle import cleanup_verdict, lifecycle_complete
        from ..kernel.request_handlers import (
            KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT,
            _grade_integrate_accuracy,
            _maybe_apply_kernel_patch,
            _maybe_finalize_kernel_patch,
            _maybe_revert_kernel_patch,
        )
        from ..loop.sub_agent_runner import RunnerContext
        from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir

        kernel_ids = [str(e.get("kernel_id") or "") for e in entries]
        stack_id = "+".join(kernel_ids)
        apply_results: list[dict[str, Any]] = []
        try:
            for entry in entries:
                payload = {
                    "kernel_id": entry.get("kernel_id"),
                    "patch_path": entry.get("patch_path"),
                    "target_file": entry.get("target_file"),
                    "allow_unknown_target": True,
                }
                applied = _maybe_apply_kernel_patch(
                    payload,
                    session_dir=self.session_dir,
                    kernel_id=str(entry.get("kernel_id") or ""),
                )
                apply_results.append(applied)
                self.shared_state.pending_stack_validation_apply_results = list(
                    apply_results,
                )
                self.shared_state.save(self.session_dir)
                if applied.get("status") != "ok":
                    raise RuntimeError(f"stack patch apply failed for {entry.get('kernel_id')}: {applied}")

            workspace = unique_runs_dir(self.session_dir, "integrate", f"integrate-stack-{stack_id}")
            fake_task = Task(
                task_id=f"integrate-stack-{stack_id}",
                kind="baseline",
                state="running",
                params={
                    "config_path": self.shared_state.baseline_config_path,
                    "output_dir": str(workspace),
                    "timeout_sec": 20 * 60,
                    "extra_server_args": ((self.shared_state.current_best or {}).get("extra_server_args") or ""),
                    # Synthetic kind="baseline": validates the stacked kernels
                    # against the already-anchored baseline on throughput alone.
                    # Exempt from the genuine-baseline accuracy guard -- this ctx
                    # carries the live SharedState, so without the exemption a
                    # missing accuracy here would stamp an eval-failure contract
                    # and drag a healthy run back into enablement.
                    "quality_ref_exempt": True,
                    # A sub-step of the KERNEL phase's own event, not a
                    # dispatched measurement, so it leaves no baseline event.
                    SBD_INNER_STEP_PARAM: True,
                },
                idempotency_key=f"integrate-stack-{stack_id}-rebaseline",
            )
            # Inject the live SharedState via ctx.extra (not the constructor).
            bench_result = await BaselineExecutor(session_dir=self.session_dir)(
                RunnerContext(
                    task=fake_task,
                    lease=None,
                    extra={"shared_state": self.shared_state},
                )
            )
            if not is_valid_measurement(bench_result):
                decision = "REVERT"
                new_tput = 0.0
                gain_pct = -100.0
                incremental_gain_pct = -100.0
            else:
                base_tput = float(self.shared_state.baseline_tput or 0.0)
                new_tput = float(bench_result.get("output_throughput") or 0.0)
                gain_pct = (new_tput - base_tput) / base_tput * 100.0 if base_tput > 0 else 0.0
                # The stack is applied on top of current_best, so the KEEP
                # decision is the incremental gain over current_best rather than
                # the total gain over the baseline. Reported ``gain_pct`` stays
                # on the output axis the stack ledger is denominated in.
                graded = resolve_graded_comparison(self.shared_state, bench_result)
                if graded.degrade_reason:
                    log.info("stack-validate: %s graded on output throughput (%s)", stack_id, graded.degrade_reason)
                incremental_gain_pct = (
                    (graded.candidate - graded.reference) / graded.reference * 100.0 if graded.reference > 0 else 0.0
                )
                if graded.vetoed:
                    log.info("stack-validate: %s failed the interactivity constraint", stack_id)
                clears = incremental_gain_pct > KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT
                decision = "KEEP" if clears and not graded.vetoed else "REVERT"

            # bench_result already carries accuracy (RUN_EVAL defaults true here).
            if decision == "KEEP" and isinstance(bench_result, dict):
                try:
                    accuracy_gate = _grade_integrate_accuracy(
                        bench_result,
                        session_dir=self.session_dir,
                        workspace=workspace,
                        # The args the bench server ran under, so a serving
                        # context too small to host an eval is not read as a
                        # broken eval.
                        server_args=str((self.shared_state.current_best or {}).get("extra_server_args") or ""),
                    )
                    if accuracy_gate.get("blocked"):
                        decision = "NEEDS_REVIEW"
                        log.info(
                            "stack-validate: accuracy gate blocked KEEP for %s: %s",
                            stack_id,
                            accuracy_gate.get("reason"),
                        )
                except Exception:  # noqa: BLE001
                    log.debug("stack-validate: accuracy gate failed", exc_info=True)

            finalize_results: list[dict[str, Any]] = []
            stack_reverts: list[dict[str, Any]] = []
            if decision == "KEEP":
                for applied in apply_results:
                    finalize_results.append(_maybe_finalize_kernel_patch(applied))
                all_finalized = all(lifecycle_complete(fr) for fr in finalize_results)
                cs = "complete" if all_finalized else "recovery_required"
                ca = "" if all_finalized else "finalize"
                revert_result: dict[str, Any] = {"status": "skipped", "reason": "KEEP decision"}
                top_status = "ok"
            else:
                stack_reverts = [_maybe_revert_kernel_patch(applied) for applied in reversed(apply_results)]
                all_reverted = all(lifecycle_complete(r) for r in stack_reverts)
                top_status, cs, ca = cleanup_verdict(
                    decision=decision,
                    revert_result={"status": "ok" if all_reverted else "failed"},
                    finalize_result={"status": "skipped"},
                    revert_required=bool(apply_results),
                )
                revert_result = {
                    "status": "ok" if all_reverted else "failed",
                    "stack_reverts": stack_reverts,
                }

            result = {
                "status": top_status,
                "decision": decision,
                "patch_cleanup_status": cs,
                "patch_cleanup_action": ca,
                "kernel_id": stack_id,
                "patch_path": "+".join(str(e.get("patch_path") or "") for e in entries),
                "target_file": "+".join(str(e.get("target_file") or "") for e in entries),
                "base_tput": float(self.shared_state.baseline_tput or 0.0),
                "new_tput": new_tput,
                "gain_pct": gain_pct,
                "stack_incremental_gain_pct": incremental_gain_pct,
                "stack_incremental_keep_threshold_pct": (KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT),
                "report_path": bench_result.get("report_path") if isinstance(bench_result, dict) else None,
                "workspace": bench_result.get("workspace") if isinstance(bench_result, dict) else str(workspace),
                "apply_result": {"status": "ok", "stack_apply_results": apply_results},
                "revert_result": revert_result,
                "finalize_results": finalize_results,
                "stack_kernel_ids": kernel_ids,
                "stack_validation": True,
            }
            if top_status == "failed":
                result["error_class"] = "patch_revert_incomplete"
                result["error"] = "Stack patch revert did not fully complete"
            for metric in ("ttft_mean_ms", "e2el_mean_ms", "tpot_mean_ms"):
                if isinstance(bench_result, dict) and metric in bench_result:
                    result[metric] = bench_result.get(metric)
            return result
        except Exception as exc:  # noqa: BLE001
            reverts = [_maybe_revert_kernel_patch(applied) for applied in reversed(apply_results)]
            any_failed = any(str(r.get("status") or "") not in {"ok", "skipped"} for r in reverts)
            revert_status = "failed" if any_failed else "ok"
            return {
                "status": "failed",
                "decision": "REVERT",
                "patch_cleanup_status": "recovery_required" if any_failed else "complete",
                "patch_cleanup_action": "revert" if any_failed else "",
                "kernel_id": stack_id,
                "error": repr(exc),
                "apply_result": {"status": "failed", "stack_apply_results": apply_results},
                "revert_result": {"status": revert_status, "stack_reverts": reverts},
                "stack_kernel_ids": kernel_ids,
                "stack_validation": True,
            }

    async def _auto_enqueue_pending_integrations(self) -> None:
        """Auto-dispatch integrate for KEEP'd kernels awaiting integration.

        The candidate set is :meth:`SharedState.pending_kernel_integration_records`,
        which includes kernels whose only prior integrate attempts were un-exhausted
        (retryable) faults. Duplicate dispatch is guarded per ``integration_id``
        (falling back to ``kernel_id`` when absent) by the recorded
        integrate-attempt count (``_auto_integrate_attempt_marks``): a
        kernel is re-dispatched only once its prior integrate has been recorded,
        never while one is in flight. Idempotent.
        """
        state = self.shared_state
        pending_records = state.pending_kernel_integration_records()
        if not pending_records:
            return

        # Per-kernel in-flight guard, keyed on recorded integrate-attempt count.
        if not hasattr(self, "_auto_integrate_attempt_marks"):
            self._coord._auto_integrate_attempt_marks: dict[str, int] = {}

        for pending in pending_records:
            kid = str(pending.get("kernel_id") or "")
            integration_id = str(pending.get("integration_id") or "")
            dispatch_key = integration_id or kid
            recorded = (
                state.integrate_attempt_count_for_integration(integration_id)
                if integration_id
                else state.integrate_attempt_count_for_kernel(kid)
            )
            mark = self._auto_integrate_attempt_marks.get(dispatch_key)
            if mark is not None and recorded <= mark:
                # A prior integrate for this kernel is still in flight.
                continue
            log.info(
                "auto-integrate: dispatching integrate for KEEP'd kernel %s "
                "(IR-3 mandatory integration; recorded_attempts=%d)",
                kid,
                recorded,
            )
            await self.bus.append_and_seq(
                Message.new(
                    "orchestration",
                    "kernel_agent",
                    "request",
                    {
                        "kind": "integrate",
                        "kernel_id": kid,
                        "integration_id": integration_id,
                        "task_group_key": str(pending.get("task_group_key") or ""),
                        "identity_route": str(pending.get("identity_route") or ""),
                        "source": "auto_integrate_after_kernel_opt",
                    },
                    priority=2,
                )
            )
            self._auto_integrate_attempt_marks[dispatch_key] = recorded
