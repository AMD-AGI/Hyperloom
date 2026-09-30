# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-stack validation handler: draining pending KEEP integrates and running/recovering the positive-needs-review stack e2e validation."""

from __future__ import annotations
import logging as _logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, NoReturn
from hyperloom.common.perf_metric import VERDICT_KEEP, VERDICT_REVERT
from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from ..bus.message_bus import Message
from ..kernel._kernel_decisions import _entry_by_kernel_id
from ..kernel.patch_lifecycle import (
    CLEANUP_ACTION_NONE,
    CLEANUP_ACTION_REVERT,
    CLEANUP_COMPLETE,
    CLEANUP_RECOVERY_REQUIRED,
    cleanup_verdict,
    lifecycle_complete,
    revert_owed,
)
from ..state.shared_state import resolve_graded_comparison
from ..state.task_registry import Task
from ..collaborator import CoordinatorCollaborator

log = _logging.getLogger(__name__)


def resolve_stack_members(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve atomic member ids without interpreting a display id as a delimiter format."""
    flag = record.get("stack_validation")
    if "stack_validation" in record and not isinstance(flag, bool):
        raise ValueError("stack_validation must be a bool")
    if "stack_kernel_ids" in record:
        raw = record["stack_kernel_ids"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("stack members must be a non-empty list")
        if any(not isinstance(kid, str) or not kid.strip() for kid in raw):
            raise ValueError("stack members must be non-empty strings")
        members = tuple(raw)
        if len(set(members)) != len(members):
            raise ValueError("stack members must be unique")
        if flag is True and len(members) < 2:
            raise ValueError("stack validation requires at least two members")
        if flag is False and len(members) != 1:
            raise ValueError("single-kernel integration must have exactly one member")
    else:
        kid = record.get("kernel_id")
        if flag is True or not isinstance(kid, str) or not kid.strip() or ("+" in kid and flag is not False):
            raise ValueError("stack membership is unavailable; a display id is not member evidence")
        members = (kid,)
    return members


def _matching_stack_entries(
    members: tuple[str, ...],
    entries: Mapping[str, Any],
    *,
    identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind members by independent patch identities before testing ledger ambiguity."""
    identity_keys = ("kernel_id", "patch_path", "target_file")
    if (
        not isinstance(identities, list)
        or any(
            not isinstance(row, dict)
            or any(not isinstance(row.get(key), str) or not row[key].strip() for key in identity_keys)
            for row in identities
        )
        or tuple(row["kernel_id"] for row in identities) != members
    ):
        raise ValueError("stack member identities do not match the requested members")
    expected = {row["kernel_id"]: tuple(row[key] for key in identity_keys) for row in identities}
    found: dict[str, dict[str, Any]] = {}
    for entry in entries.values():
        if not isinstance(entry, dict) or entry.get("kernel_id") not in members:
            continue
        kid = entry["kernel_id"]
        if tuple(entry.get(key) for key in identity_keys) != expected[kid]:
            continue
        if kid in found:
            raise ValueError(f"stack member {kid!r} matches multiple ledger entries")
        found[kid] = entry
    if set(found) != set(members):
        raise ValueError(f"stack members missing from ledger: {sorted(set(members) - set(found))!r}")
    ordered = [found[kid] for kid in members]
    if len({entry["target_file"] for entry in ordered}) != len(ordered):
        raise ValueError("stack members must have distinct target files")
    return ordered


def _revert_incomplete(stack_id: str) -> RuntimeError:
    """The error a stack whose revert did not finish halts with."""
    return RuntimeError(f"stack {stack_id} revert incomplete; checkpoints retained for the next resume")


class KernelStackPhase(CoordinatorCollaborator):
    """Coordinator mixin; its methods run with the Coordinator as ``self``."""

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
            except Exception as exc:
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
        """Return positive NEEDS_REVIEW integrate entries eligible for stack validation."""
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
        """Kernel ids already covered by an explicitly identified kept integration."""
        resolved: set[str] = set()
        for item in self.shared_state.optimization_stack or []:
            if isinstance(item, dict) and item.get("action") == "integrate":
                resolved.update(resolve_stack_members(item))
        return resolved

    def _mark_stack_validation_entries_resolved(
        self,
        entries: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Mark the kept stack's bound ledger rows as handled by it."""
        decision = str(result.get("decision") or "").upper()
        now = datetime.now(timezone.utc).isoformat()
        for entry in entries:
            entry.update(stack_resolved=True, stack_decision=decision, stack_resolved_at=now)

    def _mark_stack_validation_in_progress(
        self,
        entries: list[dict[str, Any]],
        stack_id: str,
    ) -> list[dict[str, Any]]:
        """Persist an in-flight stack guard before applying patches; return the ledger rows the members bind to."""
        members = resolve_stack_members(
            {"stack_validation": True, "stack_kernel_ids": [e.get("kernel_id") for e in entries]}
        )
        rows = _matching_stack_entries(members, self.shared_state.kernel_integrate_attempts, identities=entries)
        for row in rows:
            row["stack_validation_in_progress"] = True
        self.shared_state.pending_stack_validation_result = {
            "kernel_id": stack_id,
            "stack_validation": True,
            "stack_kernel_ids": [entry["kernel_id"] for entry in entries],
            "stack_member_identities": [
                {key: entry[key] for key in ("kernel_id", "patch_path", "target_file")} for entry in entries
            ],
        }
        self.shared_state.pending_stack_validation_apply_results = []
        return rows

    def _clear_pending_stack_validation_checkpoints(self) -> None:
        """Drop crash-recovery checkpoints and every in-flight guard once a stack attempt is finished.

        One checkpoint slot means one attempt in flight, so once it is over no ledger row is in flight either.
        """
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if isinstance(entry, dict):
                entry.pop("stack_validation_in_progress", None)
        self.shared_state.pending_stack_validation_result = {}
        self.shared_state.pending_stack_validation_apply_results = []

    async def _recover_interrupted_stack_validation(self) -> bool:
        """Settle a stack attempt a crash or halt left behind; return whether there was one.

        The record, an apply row and a ledger row's in-flight guard each mean an attempt started. A guard alone is
        what v1.1.2 left when it crashed before its first apply checkpoint; with no apply row there is nothing to
        revert, so recovery only releases the members.
        """
        state = self.shared_state
        pending = state.pending_stack_validation_result
        guarded = any(
            isinstance(entry, dict) and entry.get("stack_validation_in_progress")
            for entry in (state.kernel_integrate_attempts or {}).values()
        )
        if not (pending or state.pending_stack_validation_apply_results or guarded):
            return False
        try:
            if pending and not isinstance(pending, dict):
                raise ValueError("stack recovery checkpoint must be a mapping")
            if pending.get("decision") and not revert_owed(pending):
                # Only a KEEP goes on to promote its members, so only a KEEP needs them bound; a settled REVERT has
                # already left the tree as it found it.
                keep = str(pending["decision"]).upper() == "KEEP"
                stack = self._settled_stack_members(pending) if keep else []
                await self._finalize_stack_validation_outcome(stack, pending)
                return True
            # Either the attempt never reached a decision, or its decision's revert did not finish: the members are
            # still on the tree, so tear them down before anything else measures it.
            self._unwind_stack_patches()
        except ValueError as exc:
            self._halt_stack_recovery(exc)
        self._clear_pending_stack_validation_checkpoints()
        state.save(self.session_dir)
        return True

    def _settled_stack_members(self, pending: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Bind a record whose decision is already carried out to the rows its attempt marked.

        A KEEP has finalized its patches and left no backup to roll back to, so an unbindable record can only halt.
        """
        return _matching_stack_entries(
            resolve_stack_members(pending),
            self.shared_state.kernel_integrate_attempts,
            identities=pending.get("stack_member_identities"),
        )

    def _unwind_stack_patches(self) -> None:
        """Revert every checkpointed apply in reverse order, or halt with the checkpoints intact.

        The apply rows alone record what reached the tree, so a record that no longer binds still unwinds.
        """
        from ..actions.executors._kernel_agent_tool import _maybe_revert_kernel_patch

        partial_applies = self.shared_state.pending_stack_validation_apply_results or []
        if not isinstance(partial_applies, list) or any(not isinstance(row, dict) for row in partial_applies):
            raise ValueError("stack apply checkpoints must be a list of apply results")
        for applied in reversed(partial_applies):
            if not lifecycle_complete(_maybe_revert_kernel_patch(applied)):
                stack_id = str((self.shared_state.pending_stack_validation_result or {}).get("kernel_id") or "")
                self._halt_stack_recovery(_revert_incomplete(stack_id))

    def _halt_stack_recovery(self, error: Exception) -> NoReturn:
        """Stop the session on stack state no resume can account for, keeping the evidence, then raise ``error``.

        ``_on_phase_entered`` logs and swallows whatever a phase hook raises, so the raise alone would let SWEEP
        benchmark the tree. The stop reason is what actually ends the run, and it is deliberately not an
        infrastructure one: a tree that no longer matches the ledger is a failed session, not an aborted one.
        """
        self.shared_state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
        self.shared_state.save(self.session_dir)
        log.error("stack recovery halted the session: %r", error)
        raise error

    async def _finalize_stack_validation_outcome(
        self,
        stack: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Record stack validation, promote KEEP, and clear recovery checkpoints."""
        decision = str(result.get("decision") or "").upper()
        self.shared_state.record_kernel_integrate_result(result)
        if revert_owed(result):
            self._halt_stack_recovery(_revert_incomplete(str(result.get("kernel_id") or "")))
        if decision == "KEEP":
            # Marked only once promoted, so a KEEP that dies on the way leaves no
            # row claiming a promotion that never happened.
            await self._record_integrate_keep(result)
            self._mark_stack_validation_entries_resolved(stack, result)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)

    async def _maybe_validate_positive_needs_review_stack(self) -> None:
        """Run one E2E stack validation for multiple small positive kernel patches."""
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
        stack = self._mark_stack_validation_in_progress(stack, stack_id)
        self.shared_state.save(self.session_dir)
        result = await self._run_kernel_stack_validation_e2e(stack)
        self.shared_state.pending_stack_validation_result = result
        self.shared_state.save(self.session_dir)
        await self._finalize_stack_validation_outcome(stack, result)

    async def _run_kernel_stack_validation_e2e(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply multiple kernel patches, run one E2E benchmark, then keep or revert the stack."""
        from ..actions.executors.baseline import BaselineExecutor
        from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import kernel_event_id

        # Lazy (re-)import so tests can monkeypatch it on the source module.
        from ..actions.executors.benchmark_result import is_valid_measurement
        from ..kernel.request_handlers import (
            KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT,
            _grade_integrate_accuracy,
        )
        from ..actions.executors._kernel_agent_tool import (
            _maybe_apply_kernel_patch,
            _maybe_finalize_kernel_patch,
            _maybe_revert_kernel_patch,
        )
        from ..loop.sub_agent_runner import RunnerContext
        from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir

        kernel_ids = [entry["kernel_id"] for entry in entries]
        stack_id = "+".join(kernel_ids)
        identities = [{key: entry[key] for key in ("kernel_id", "patch_path", "target_file")} for entry in entries]
        apply_results: list[dict[str, Any]] = []
        try:
            for entry in entries:
                payload = {
                    "kernel_id": entry.get("kernel_id"),
                    "patch_path": entry.get("patch_path"),
                    "target_file": entry.get("target_file"),
                }
                applied = _maybe_apply_kernel_patch(
                    payload,
                    session_dir=self.session_dir,
                    kernel_id=str(entry.get("kernel_id") or ""),
                )
                applied = {**applied, **payload}
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
                    # Synthetic kind="baseline": validates the stacked kernels against the already-anchored baseline
                    # on throughput alone.
                    "quality_ref_exempt": True,
                    # A sub-step of the KERNEL phase's own event, not a dispatched measurement, so it records into
                    # that event rather than leaving a baseline event of its own.
                    INLINE_EVENT_PARAM: kernel_event_id(int(getattr(self.shared_state, "macro_cycle", 0) or 0)),
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
            graded = None
            if not is_valid_measurement(bench_result):
                decision = "REVERT"
                graded_verdict = VERDICT_REVERT
                new_tput = 0.0
                gain_pct = -100.0
                incremental_gain_pct = -100.0
            else:
                base_tput = float(self.shared_state.baseline_tput or 0.0)
                new_tput = float(bench_result.get("output_throughput") or 0.0)
                gain_pct = (new_tput - base_tput) / base_tput * 100.0 if base_tput > 0 else 0.0
                # The stack is applied on top of current_best, so the KEEP decision is the incremental gain over
                # current_best rather than the total gain over the baseline.
                # The verdict is the chokepoint's: it already applies the AgentX keep-threshold floor and the
                # throughput guard, so the stack lane must not re-derive a KEEP from the raw incremental gain.
                graded = resolve_graded_comparison(
                    self.shared_state,
                    bench_result,
                    keep_threshold_pct=KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT,
                )
                incremental_gain_pct = (
                    (graded.candidate - graded.reference) / graded.reference * 100.0 if graded.reference > 0 else 0.0
                )
                if not graded.comparable:
                    # Fail closed rather than REVERT. A stack that could not be graded on the axis the session asked
                    # for has an output-axis figure only; promoting or discarding a kernel stack on a substitute axis
                    # is a call for a human, and the revert path below still leaves the tree clean either way.
                    log.info(
                        "stack-validate: %s performance comparison unavailable (%s)", stack_id, graded.degrade_reason
                    )
                    graded_verdict = graded.verdict
                    decision = "NEEDS_REVIEW"
                elif graded.verdict != VERDICT_KEEP:
                    log.info(
                        "stack-validate: %s %s intvty %.1f->%.1f tput %.1f->%.1f",
                        stack_id,
                        graded.verdict,
                        graded.reference,
                        graded.candidate,
                        graded.tput_reference,
                        graded.tput_candidate,
                    )
                    graded_verdict = graded.verdict
                    decision = "REVERT"
                else:
                    graded_verdict = graded.verdict
                    decision = "KEEP"

            # bench_result already carries accuracy (RUN_EVAL defaults true here).
            if decision == "KEEP" and isinstance(bench_result, dict):
                accuracy_gate = _grade_integrate_accuracy(
                    bench_result,
                    session_dir=self.session_dir,
                    workspace=workspace,
                    # The args the bench server ran under, so a serving context too small to host an eval is not
                    # read as a broken eval.
                    server_args=str((self.shared_state.current_best or {}).get("extra_server_args") or ""),
                )
                if accuracy_gate.get("blocked"):
                    decision = "NEEDS_REVIEW"
                    log.info(
                        "stack-validate: accuracy gate blocked KEEP for %s: %s",
                        stack_id,
                        accuracy_gate.get("reason"),
                    )

            finalize_results: list[dict[str, Any]] = []
            stack_reverts: list[dict[str, Any]] = []
            if decision == "KEEP":
                for applied in apply_results:
                    finalize_results.append(_maybe_finalize_kernel_patch(applied))
                all_finalized = all(lifecycle_complete(fr) for fr in finalize_results)
                revert_result: dict[str, Any] = {"status": "skipped", "reason": "KEEP decision"}
                top_status, cs, ca = cleanup_verdict(
                    decision=decision,
                    revert_result=revert_result,
                    finalize_result={"status": "ok" if all_finalized else "failed"},
                    revert_required=False,
                )
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
                "graded_objective": graded.objective if graded is not None else None,
                "bench_result": bench_result,
                "stack_incremental_gain_pct": incremental_gain_pct,
                "stack_incremental_keep_threshold_pct": (KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT),
                # A stack cannot be left half-applied, so anything short of KEEP reverts it whole.
                "graded_verdict": graded_verdict,
                "report_path": bench_result.get("report_path") if isinstance(bench_result, dict) else None,
                "workspace": bench_result.get("workspace") if isinstance(bench_result, dict) else str(workspace),
                "apply_result": {"status": "ok", "stack_apply_results": apply_results},
                "revert_result": revert_result,
                "finalize_results": finalize_results,
                "stack_kernel_ids": kernel_ids,
                "stack_validation": True,
                "stack_member_identities": identities,
            }
            if graded is not None and not graded.comparable:
                result["reason"] = f"performance comparison unavailable: {graded.degrade_reason}"
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
                "patch_cleanup_status": CLEANUP_RECOVERY_REQUIRED if any_failed else CLEANUP_COMPLETE,
                "patch_cleanup_action": CLEANUP_ACTION_REVERT if any_failed else CLEANUP_ACTION_NONE,
                "kernel_id": stack_id,
                "error": repr(exc),
                "apply_result": {"status": "failed", "stack_apply_results": apply_results},
                "revert_result": {"status": revert_status, "stack_reverts": reverts},
                "stack_kernel_ids": kernel_ids,
                "stack_validation": True,
                "stack_member_identities": identities,
            }

    async def _auto_enqueue_pending_integrations(self) -> None:
        """Auto-dispatch integrate for KEEP'd kernels awaiting integration."""
        state = self.shared_state
        pending_records = state.pending_kernel_integration_records()
        if not pending_records:
            return

        for pending in pending_records:
            kid = str(pending.get("kernel_id") or "")
            integration_id = str(pending.get("integration_id") or "")
            dispatch_key = integration_id or kid
            recorded = (
                state.integrate_attempt_count_for_integration(integration_id)
                if integration_id
                else state.integrate_attempt_count_for_kernel(kid)
            )
            mark = self._attempt_marks.get(dispatch_key)
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
                        "mode": "patch",
                    },
                )
            )
            self._attempt_marks[dispatch_key] = recorded
