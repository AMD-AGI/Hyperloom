# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-stack validation handler: draining pending KEEP integrates and running/recovering the positive-needs-review stack e2e validation."""

from __future__ import annotations
import logging as _logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, NoReturn
from hyperloom.common.perf_metric import VERDICT_KEEP, VERDICT_REVERT
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
from .machine_state import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

log = _logging.getLogger(__name__)


def resolve_stack_members(
    record: Mapping[str, Any],
    *,
    entries: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
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
    if entries is not None:
        _matching_stack_entries(members, entries, validation=record if flag is True else None)
    return members


def _matching_stack_entries(
    members: tuple[str, ...],
    entries: Mapping[str, Any],
    *,
    identities: list[dict[str, Any]] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bind members by independent patch identities before testing ledger ambiguity."""
    identity_keys = ("kernel_id", "patch_path", "target_file")
    if validation is not None:
        stack_id = validation.get("kernel_id")
        started = validation.get("stack_validation_started_at")
        if not isinstance(stack_id, str) or not stack_id or not isinstance(started, str) or not started:
            raise ValueError("stack members lack matching validation evidence")
        identities = validation.get("stack_member_identities")
        if not isinstance(identities, list):
            raise ValueError("stack recovery lacks independent member identities")
    expected = None
    if identities is not None:
        if (
            any(
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
        if expected is not None and tuple(entry.get(key) for key in identity_keys) != expected[kid]:
            continue
        if validation is not None and (
            entry.get("stack_validation_kernel_id") != stack_id or entry.get("stack_validation_started_at") != started
        ):
            continue
        if kid in found:
            raise ValueError(f"stack member {kid!r} matches multiple ledger entries")
        if any(not isinstance(entry.get(key), str) or not entry[key].strip() for key in ("patch_path", "target_file")):
            raise ValueError(f"stack member {kid!r} lacks a complete patch identity")
        found[kid] = entry
    if set(found) != set(members):
        raise ValueError(f"stack members missing from ledger: {sorted(set(members) - set(found))!r}")
    ordered = [found[kid] for kid in members]
    if len({entry["target_file"] for entry in ordered}) != len(ordered):
        raise ValueError("stack members must have distinct target files")
    return ordered


class KernelStackPhase(CoordinatorCollaborator):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    async def _drain_pending_keep_integrates(self) -> None:
        """Drain pending KEEP integrates inherited from KERNEL so sweep measures full current_best. Cap 10; a dispatch failure sets ``rejected_reason=integrate_dispatch_exception`` on the per-kernel and per-task_key attempt ledgers and flips the queued record to ``dispatch_failed``; only records with no ``task_key`` are also appended to ``rejected_kernel_ids``."""
        from ..kernel.request_handlers import integrate_handler

        state = self.shared_state
        self._stack_resolved_kernel_ids()
        self._pending_stack_members()
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

    def _pending_stack_members(self) -> tuple[str, ...]:
        """Validate persisted recovery evidence without changing any checkpoint."""
        pending = self.shared_state.pending_stack_validation_result
        if not pending:
            if self.shared_state.pending_stack_validation_apply_results or any(
                isinstance(entry, dict) and entry.get("stack_validation_in_progress")
                for entry in (self.shared_state.kernel_integrate_attempts or {}).values()
            ):
                raise ValueError("stack recovery lacks structured member evidence")
            return ()
        if not isinstance(pending, dict):
            raise ValueError("stack recovery checkpoint must be a mapping")
        members = resolve_stack_members(pending, entries=self.shared_state.kernel_integrate_attempts)
        if pending.get("stack_validation") is not True or len(members) < 2:
            raise ValueError("stack recovery checkpoint must identify a complete validation")
        if any(
            isinstance(entry, dict)
            and entry.get("stack_validation_in_progress")
            and entry.get("kernel_id") not in members
            for entry in self.shared_state.kernel_integrate_attempts.values()
        ):
            raise ValueError("stack recovery has unrelated in-progress members")
        return members

    def _mark_stack_validation_entries_resolved(
        self,
        entries: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Mark component NEEDS_REVIEW entries as handled by a kept stack."""
        stack_id = str(result.get("kernel_id") or "")
        decision = str(result.get("decision") or "").upper()
        if decision != "KEEP" or not stack_id:
            return
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
            entry["stack_resolved"] = True
            entry["stack_validation_kernel_id"] = stack_id
            entry["stack_decision"] = decision
            entry["stack_resolved_at"] = now
            entry.pop("stack_validation_in_progress", None)

    def _stack_component_identities(
        self,
        entries: list[dict[str, Any]],
    ) -> set[tuple[str, str, str]]:
        """Return (kernel_id, patch_path, target_file) tuples for stack members."""
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("stack members must be ledger rows")
        members = resolve_stack_members(
            {"stack_validation": True, "stack_kernel_ids": [e.get("kernel_id") for e in entries]}
        )
        supplied = {str(index): entry for index, entry in enumerate(entries)}
        _matching_stack_entries(members, supplied)
        _matching_stack_entries(members, self.shared_state.kernel_integrate_attempts, identities=entries)
        return {(e["kernel_id"], e["patch_path"], e["target_file"]) for e in entries}

    def _mark_stack_validation_in_progress(
        self,
        entries: list[dict[str, Any]],
        stack_id: str,
    ) -> None:
        """Persist an in-flight stack guard before applying patches."""
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
        self.shared_state.pending_stack_validation_result = {
            "kernel_id": stack_id,
            "stack_validation": True,
            "stack_kernel_ids": [entry["kernel_id"] for entry in entries],
            "stack_validation_started_at": now,
            "stack_member_identities": [
                {key: entry[key] for key in ("kernel_id", "patch_path", "target_file")} for entry in entries
            ],
        }
        self.shared_state.pending_stack_validation_apply_results = []

    def _clear_stack_validation_in_progress(
        self,
        entries: list[dict[str, Any]],
    ) -> None:
        """Clear the in-flight stack guard for component integrate entries."""
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
        """Resume or abort a stack validation interrupted by crash."""
        self._stack_resolved_kernel_ids()
        members = self._pending_stack_members()
        if not members:
            return False
        pending = self.shared_state.pending_stack_validation_result
        stack = _matching_stack_entries(members, self.shared_state.kernel_integrate_attempts, validation=pending)
        if pending.get("decision") and not revert_owed(pending):
            await self._finalize_stack_validation_outcome(stack, pending)
            return True

        # Either the attempt never reached a decision, or its decision's revert did not finish: the members are
        # still on the tree, so tear them down before anything else measures it.
        self._unwind_stack_patches(stack)
        self._clear_stack_validation_in_progress(stack)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)
        return True

    def _unwind_stack_patches(self, stack: list[dict[str, Any]]) -> None:
        """Revert every checkpointed apply in reverse order, or halt with the checkpoints intact."""
        from ..kernel.request_handlers import _maybe_revert_kernel_patch

        pending = self.shared_state.pending_stack_validation_result
        partial_applies = self.shared_state.pending_stack_validation_apply_results
        if not isinstance(partial_applies, list) or len(partial_applies) > len(stack):
            raise ValueError("stack apply checkpoints do not match its members")
        for entry, applied in zip(stack, partial_applies):
            if (
                not isinstance(applied, dict)
                or any(applied.get(key) != entry[key] for key in ("kernel_id", "patch_path", "target_file"))
                or applied.get("stack_validation_started_at") != pending["stack_validation_started_at"]
                or not isinstance(applied.get("manifest_path"), str)
                or not applied["manifest_path"]
            ):
                raise ValueError("stack apply checkpoint lacks matching member evidence")
        for applied in reversed(partial_applies):
            if not lifecycle_complete(_maybe_revert_kernel_patch(applied)):
                self._halt_on_owed_revert(str(pending.get("kernel_id") or ""))

    def _halt_on_owed_revert(self, stack_id: str) -> NoReturn:
        """Stop the session on a patched tree, keeping the evidence the next resume retries from.

        ``_on_phase_entered`` logs and swallows whatever a phase hook raises, so the raise alone would let SWEEP
        benchmark the patched tree. The stop reason is what actually ends the run, and it is deliberately not an
        infrastructure one: a tree that no longer matches the ledger is a failed session, not an aborted one.
        """
        self.shared_state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
        self.shared_state.save(self.session_dir)
        raise RuntimeError(f"stack {stack_id} revert incomplete; checkpoints retained for the next resume")

    def _stack_entries_for_validation(
        self,
        kernel_ids: list[Any],
    ) -> list[dict[str, Any]]:
        """Rebuild all component rows from explicit ids; the display id supplies no members."""
        members = resolve_stack_members({"stack_kernel_ids": kernel_ids, "stack_validation": True})
        return _matching_stack_entries(members, self.shared_state.kernel_integrate_attempts)

    async def _finalize_stack_validation_outcome(
        self,
        stack: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Record stack validation, promote KEEP, and clear recovery checkpoints."""
        self._stack_resolved_kernel_ids()
        self._stack_component_identities(stack)
        members = resolve_stack_members(result, entries=self.shared_state.kernel_integrate_attempts)
        if result.get("stack_validation") is not True or members != tuple(entry["kernel_id"] for entry in stack):
            raise ValueError("stack result does not describe the validated members")
        self.shared_state.record_kernel_integrate_result(result)
        if revert_owed(result):
            self._halt_on_owed_revert(str(result.get("kernel_id") or ""))
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
        """Run one E2E stack validation for multiple small positive kernel patches."""
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
        self.shared_state.save(self.session_dir)
        result = await self._run_kernel_stack_validation_e2e(stack)
        if not isinstance(result, dict):
            raise ValueError("stack validation produced no result; checkpoints retained")
        checkpoint = self.shared_state.pending_stack_validation_result
        result = {
            **result,
            "stack_validation_started_at": checkpoint["stack_validation_started_at"],
            "stack_member_identities": checkpoint["stack_member_identities"],
        }
        resolve_stack_members(result, entries=self.shared_state.kernel_integrate_attempts)
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
        from ..actions.executors.benchmark_result import is_valid_measurement  # noqa: F811
        from ..kernel.request_handlers import (
            KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT,
            _grade_integrate_accuracy,
            _maybe_apply_kernel_patch,
            _maybe_finalize_kernel_patch,
            _maybe_revert_kernel_patch,
        )
        from ..loop.sub_agent_runner import RunnerContext
        from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir

        self._stack_resolved_kernel_ids()
        self._stack_component_identities(entries)
        kernel_ids = [entry["kernel_id"] for entry in entries]
        stack_id = "+".join(kernel_ids)
        pending_members = self._pending_stack_members()
        if pending_members and (
            self.shared_state.pending_stack_validation_apply_results
            or self.shared_state.pending_stack_validation_result.get("decision")
        ):
            raise ValueError("stack validation must recover its pending result before applying again")
        if pending_members and pending_members != tuple(kernel_ids):
            raise ValueError("stack validation conflicts with pending members")
        if not pending_members:
            self._mark_stack_validation_in_progress(entries, stack_id)
            self.shared_state.save(self.session_dir)
        checkpoint = self.shared_state.pending_stack_validation_result
        started = checkpoint["stack_validation_started_at"]
        identities = checkpoint["stack_member_identities"]
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
                applied = {**applied, **payload, "stack_validation_started_at": started}
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
                try:
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
                except Exception:  # noqa: BLE001
                    log.debug("stack-validate: accuracy gate failed", exc_info=True)

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
                # A stack cannot be left half-applied, so RECORDED reverts like
                # REVERT does; the verdict says which one it was.
                "graded_verdict": graded_verdict,
                "report_path": bench_result.get("report_path") if isinstance(bench_result, dict) else None,
                "workspace": bench_result.get("workspace") if isinstance(bench_result, dict) else str(workspace),
                "apply_result": {"status": "ok", "stack_apply_results": apply_results},
                "revert_result": revert_result,
                "finalize_results": finalize_results,
                "stack_kernel_ids": kernel_ids,
                "stack_validation": True,
                "stack_validation_started_at": started,
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
                "stack_validation_started_at": started,
                "stack_member_identities": identities,
            }

    async def _auto_enqueue_pending_integrations(self) -> None:
        """Auto-dispatch integrate for KEEP'd kernels awaiting integration."""
        state = self.shared_state
        self._stack_resolved_kernel_ids()
        self._pending_stack_members()
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
                        "mode": "patch",
                    },
                )
            )
            self._auto_integrate_attempt_marks[dispatch_key] = recorded
