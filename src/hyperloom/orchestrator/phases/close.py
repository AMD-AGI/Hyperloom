# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLOSE phase handler: the 5-step close sequencer, post-opt roofline, and the
closing-grace / report-terminal helpers used by ``Coordinator.run``."""

from __future__ import annotations
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any
import logging as _logging
from . import machine_state as _phase_state
from ..bus.message_bus import Message
from ..state.task_registry import Task
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class ClosePhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    def _derive_close_stop_reason(self) -> str:
        """Best-effort ``stop_reason`` for a CLOSE reached blank: recover from the newest CLOSE-bound phase_history row, else time_exhausted.

        Returns:
            A valid stop reason recovered from the newest CLOSE-bound
            phase_history row, or ``"time_exhausted"`` as the fallback.
        """
        history = self.shared_state.phase_history or []
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            if (row.get("to_phase") or "").strip().upper() != _phase_state.PHASE_CLOSE:
                continue
            reason = (row.get("reason") or "").strip()
            if reason and _phase_state.is_valid_stop_reason(reason):
                return reason
            # Newest CLOSE-bound row had no usable reason — stop rather than use a stale older one.
            break
        return "time_exhausted"

    def _session_integrated_kernel_patch(self) -> bool:
        """True iff this session landed a kernel-level optimization (optimization_stack has an integrate/gemm_tuning/geak_e2e entry). Gates the CLOSE post-opt roofline so pure param-search sessions skip the extra profile.

        Returns:
            ``True`` when at least one kernel-level optimization landed.
        """
        stack = getattr(self.shared_state, "optimization_stack", None) or []
        if not isinstance(stack, list):
            return False
        for entry in stack:
            if isinstance(entry, dict) and str(entry.get("action") or "") in self._POST_OPT_ROOFLINE_ACTIONS:
                return True
        return False

    async def _maybe_run_close_post_opt_roofline(self) -> None:
        """Best-effort: run one final post-opt roofline at CLOSE when a kernel/source patch was integrated.

        Profiles the final optimized service once and writes
        reports/kernel_roofline_opt.json, giving the before/after kernel roofline
        chart its optimized snapshot. No-op for sessions without an
        integrate-class optimization; wrapped by the caller so a failure never
        blocks close.
        """
        if not self._session_integrated_kernel_patch():
            return
        # Skip on the wall-clock-deadline close path (its grace window is too
        # short for a full profile+TraceLens); only run on a normal converged close.
        if bool(getattr(self.shared_state, "closing_phase", False)):
            log.info("CLOSE step 0: skipped post-opt roofline (wall-clock closing grace window)")
            return
        if self._internal_analysis_kind() != "roofline":
            # Roofline disabled for this run; nothing to profile.
            return
        task = await self._enqueue_internal_analysis_task(reason="close_post_opt")
        log.info(
            "CLOSE step 0: running post-opt roofline task=%s (timeout=%.0fs)",
            task.task_id,
            self.CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC,
        )
        # Hard timeout so a slow profile+TraceLens can't stall the close
        # sequence; on timeout the chart degrades to baseline-only.
        try:
            result = await asyncio.wait_for(
                self.sub.run_task(task),
                timeout=self.CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.warning(
                "CLOSE step 0: post-opt roofline timed out after %.0fs; skipping (kernel_roofline_opt.json absent)",
                self.CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC,
            )
            try:
                current = await self.tasks.get(task.task_id)
                if current.state == "queued":
                    await self.tasks.transition(
                        task.task_id,
                        "cancelled",
                        {"reason": "close_post_opt_roofline_timeout"},
                    )
                elif current.state == "running":
                    await self.tasks.transition(
                        task.task_id,
                        "failed",
                        {"reason": "close_post_opt_roofline_timeout"},
                    )
            except Exception:  # noqa: BLE001
                log.debug(
                    "CLOSE step 0: failed to mark timed-out post-opt roofline task",
                    exc_info=True,
                )
            return
        state = getattr(result, "state", None)
        log.info("CLOSE step 0: post-opt roofline finished (state=%s)", state)

    async def _on_enter_close(self, *, from_phase: str) -> None:
        """CLOSE 5-step sequencer (fixed order): report → session_breakdown → fact_finalize → ndjson_drain (no-op) → mark close_sequence_done + stop_reason. Best-effort steps; final done step always runs.

        Args:
            from_phase: The phase being left, used only for logging.
        """
        log.info("CLOSE entered (from=%s); starting 5-step close sequence", from_phase or "<unknown>")
        await self._record_close_step("sequencer_started", status="running")

        # stop_reason must persist before step 2's breakdown (collector derives it from state.json); fill only when blank.
        if not self.shared_state.stop_reason:
            derived = self._derive_close_stop_reason()
            self.shared_state.set_stop_reason(derived)
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("CLOSE: early stop_reason persist failed; step 5 will retry")

        # Post-optimization roofline (best-effort): profile the final optimized
        # service once so the before/after kernel roofline chart has its "after"
        # column. Wrapped so a slow/failed run never blocks the steps below.
        try:
            await self._maybe_run_close_post_opt_roofline()
        except Exception as exc:  # noqa: BLE001
            log.warning("CLOSE step 0 (post-opt roofline) failed: %r", exc)

        # Report.
        try:
            self._emit_lifecycle(
                step="report",
                status="START",
                detail="close_phase_entry",
            )
            report_task = await self._enqueue_internal_report_task(
                reason="close_phase_entry",
            )
            report_result = await self.sub.run_task(report_task)
            terminal_state = report_result.state
            if terminal_state in {"succeeded", None}:
                await self._record_close_step(
                    "report",
                    status="done",
                    task_id=report_task.task_id,
                )
                # Surface the final report location; advertise whichever of
                # final.{json,md} exist under reports_dir(session_dir).
                from hyperloom.inference_optimizer.session.session_paths import reports_dir as _reports_dir

                _rd = _reports_dir(self.session_dir)
                _artifacts = {
                    "json_path": str(_rd / "final.json") if (_rd / "final.json").exists() else "",
                    "md_path": str(_rd / "final.md") if (_rd / "final.md").exists() else "",
                }
                self._emit_lifecycle(
                    step="report",
                    status="END",
                    artifacts=_artifacts,
                    detail="close_phase_entry",
                )
            else:
                detail = f"task_state={terminal_state!r}"
                self._emit_lifecycle(
                    step="report",
                    status="ERROR",
                    detail=detail,
                )
                await self._record_close_step(
                    "report",
                    status="failed",
                    task_id=report_task.task_id,
                    detail=detail,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("CLOSE step 1 (report) failed")
            self._emit_lifecycle(
                step="report",
                status="ERROR",
                detail=repr(exc)[:240],
            )
            await self._record_close_step(
                "report",
                status="failed",
                detail=repr(exc)[:240],
            )

        # Session breakdown.
        try:
            bd_task = await self._enqueue_internal_session_breakdown_task(
                reason="close_phase_entry",
            )
            bd_result = await self.sub.run_task(bd_task)
            terminal_state = bd_result.state
            if terminal_state in {"succeeded", None}:
                await self._record_close_step(
                    "session_breakdown",
                    status="done",
                    task_id=bd_task.task_id,
                )
            else:
                await self._record_close_step(
                    "session_breakdown",
                    status="failed",
                    task_id=bd_task.task_id,
                    detail=f"task_state={terminal_state!r}",
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("CLOSE step 2 (session_breakdown) failed")
            await self._record_close_step(
                "session_breakdown",
                status="failed",
                detail=repr(exc)[:240],
            )

        # ---------------- Langfuse flush + receipt splice -------------------
        # Must run before the artifact package: flush_session flips the receipt
        # to final counts and patch_breakdown_langfuse splices it back into
        # session_breakdown.json, so the bundled SBD carries final counts.
        # No-op unless live push is enabled; idempotent; best-effort.
        try:
            from ..trace.langfuse_emitter import (
                flush_session,
                record_session_breakdown,
            )

            flush_session(self.session_dir)
            from hyperloom.inference_optimizer.breakdown import patch_breakdown_langfuse

            patch_breakdown_langfuse(self.session_dir)
            # Attach the final breakdown JSON to the trace as a
            # ``session_breakdown`` observation (no-op when live push is disabled).
            record_session_breakdown(self.session_dir)
        except Exception as exc:  # noqa: BLE001
            log.debug("CLOSE step 2.5 (langfuse flush) failed", exc_info=True)
            await self._record_close_step(
                "langfuse_flush",
                status="failed",
                detail=repr(exc)[:240],
            )

        # ---------------- Artifact package -> /workspace ------------------
        # Bundle the curated result/report/analysis files into a single zip under
        # ``/workspace`` so the Claw sandbox sync ships it to object storage even
        # when ``$USER_DATA_PATH`` points outside ``/workspace``. Best-effort.
        try:
            from hyperloom.inference_optimizer.breakdown import package_session_artifacts

            pkg_path = package_session_artifacts(
                self.session_dir,
                session_id=str(getattr(self.shared_state, "session_id", "") or ""),
            )
            if pkg_path is not None:
                await self._record_close_step(
                    "artifact_package",
                    status="done",
                    detail=str(pkg_path),
                )
            else:
                await self._record_close_step(
                    "artifact_package",
                    status="skipped",
                    detail="no artifacts matched or dest unwritable",
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("CLOSE step 2.6 (artifact_package) failed")
            await self._record_close_step(
                "artifact_package",
                status="failed",
                detail=repr(exc)[:240],
            )

        # ---------------- Fact finalize (Cortex commit) -------------------
        # Writes update_recipe + finalises the local journal (final_throughput /
        # total_gain_pct). Recorded as the ``fact_finalize`` close_step.
        try:
            self.cortex_finalize_recipe_and_journal()
            await self._record_close_step("fact_finalize", status="done")
        except Exception as exc:  # noqa: BLE001
            log.exception("CLOSE step 4 (fact_finalize) failed")
            await self._record_close_step(
                "fact_finalize",
                status="failed",
                detail=repr(exc)[:240],
            )

        # Record a skipped ``ndjson_drain`` close-step for ledger consumers (RecipeKB is local-only).
        await self._record_close_step("ndjson_drain", status="skipped")

        # Mark done.
        self.shared_state.close_sequence_done = True
        # Set stop_reason so the main run loop terminates next tick (idempotent backstop to the early persist).
        if not self.shared_state.stop_reason:
            self.shared_state.set_stop_reason(self._derive_close_stop_reason())
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "CLOSE step 5 (close_sequence_done save) failed; cli.finally will still write a safety-net breakdown"
            )
        await self._record_close_step("done", status="done")
        log.info("CLOSE 5-step sequencer complete")

    async def _enqueue_internal_report_task(
        self,
        *,
        reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``report`` task (idempotency_key internal-report-<reason>).

        Reuses closing_report_task_id when set so the wall-clock + CLOSE-sequencer paths don't race.

        Args:
            reason: Tag used in the task's idempotency key and logging.

        Returns:
            The created or reused ``report`` :class:`Task`.
        """
        existing_id = (self.shared_state.closing_report_task_id or "").strip()
        if existing_id:
            try:
                task = await self.tasks.get(existing_id)
                log.info(
                    "internal-report task already enqueued by wall-clock "
                    "deadline path (task_id=%s, state=%s); sequencer will "
                    "wait for it",
                    task.task_id,
                    task.state,
                )
                return task
            except Exception:  # noqa: BLE001 — TaskNotFound + friends
                # Stale id; fall through to fresh enqueue.
                pass

        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            "session_dir": str(self.session_dir),
            "max_highlights": 50,
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="report",
            params=params,
            idempotency_key=f"internal-report-{reason}",
            requires_lanes=[],
            allowed_tools=["Read"],
            side_effects=["writes_results"],
            lease_ttl_sec=120,
        )
        # Mirror onto closing_report_task_id.
        if not self.shared_state.closing_report_task_id:
            self.shared_state.closing_report_task_id = task.task_id
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("internal-report: closing_report_task_id save failed")
        if was_existing:
            log.info(
                "internal-report task reused (idempotent: task_id=%s, state=%s)",
                task.task_id,
                task.state,
            )
        return task

    async def _enqueue_internal_session_breakdown_task(
        self,
        *,
        reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``session_breakdown`` task; same idempotency contract as the report helper.

        Args:
            reason: Tag used in the task's idempotency key and logging.

        Returns:
            The created (or existing idempotent) ``session_breakdown``
            :class:`Task`.
        """
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            "session_dir": str(self.session_dir),
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="session_breakdown",
            params=params,
            idempotency_key=f"internal-session_breakdown-{reason}",
            requires_lanes=[],
            allowed_tools=["Read"],
            side_effects=["writes_results"],
            lease_ttl_sec=120,
        )
        if was_existing:
            log.info(
                "internal-session_breakdown task reused (idempotent: task_id=%s, state=%s)",
                task.task_id,
                task.state,
            )
        return task

    async def _record_close_step(
        self,
        step: str,
        *,
        status: str,
        task_id: str = "",
        detail: str = "",
    ) -> None:
        """Append one row to ``phase_history[-1].evidence.close_steps`` (best-effort, per-step persist).

        Args:
            step: The close-step name.
            status: The step's status (e.g. "running", "done", "failed",
                "skipped").
            task_id: Optional task id associated with the step.
            detail: Optional free-text detail recorded on the row.
        """
        history = self.shared_state.phase_history or []
        if not history:
            return
        row = history[-1]
        if not isinstance(row, dict):
            return
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            row["evidence"] = evidence
        steps = evidence.get("close_steps")
        if not isinstance(steps, list):
            steps = []
            evidence["close_steps"] = steps
        entry: dict[str, Any] = {
            "step": step,
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if task_id:
            entry["task_id"] = task_id
        if detail:
            entry["detail"] = detail
        steps.append(entry)
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "close_step save failed for step=%r status=%r",
                step,
                status,
            )

    async def _enter_closing_phase(self, *, grace_sec: float) -> float:
        """Enter report-flush phase after the wall-clock deadline (enqueue deterministic report task).

        Args:
            grace_sec: Seconds the closing phase may run before the report task
                is abandoned.

        Returns:
            The monotonic deadline by which the closing phase must complete.
        """
        closing_started = time.time()
        closing_deadline = time.monotonic() + float(grace_sec)
        self.shared_state.closing_phase = True
        self.shared_state.closing_started_unix = closing_started
        self.shared_state.save(self.session_dir)

        log.info(
            "Coordinator: entering closing phase (grace=%.0fs); enqueueing deterministic report task",
            grace_sec,
        )

        try:
            for q in await self.tasks.queued():
                if q.kind == "report":
                    continue
                await self.tasks.transition(
                    q.task_id,
                    "cancelled",
                    evidence={"reason": "closing_phase"},
                )
        except Exception:  # noqa: BLE001
            log.exception(
                "closing_phase: cancel of queued tasks failed (non-fatal)",
            )

        idempotency_key = f"closing-report-{int(closing_started)}-{uuid.uuid4().hex[:6]}"
        task, _existing = await self.tasks.create_or_return_existing(
            kind="report",
            params={
                "session_dir": str(self.session_dir),
                "max_highlights": 50,
            },
            idempotency_key=idempotency_key,
            requires_lanes=[],
            allowed_tools=["Read"],
            side_effects=["writes_results"],
            lease_ttl_sec=120,
        )
        self.shared_state.closing_report_task_id = task.task_id
        self.shared_state.save(self.session_dir)

        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                "*",
                "event",
                {
                    "kind": "closing_phase_entered",
                    "task_id": task.task_id,
                    "grace_sec": float(grace_sec),
                    "closing_started_unix": closing_started,
                },
            )
        )
        return closing_deadline

    async def _closing_report_terminal(self) -> bool:
        """Report whether the closing-phase report task has finished.

        Returns:
            bool: ``True`` when the report task reached a terminal state (or is
                missing); ``False`` while it is still queued or running.
        """
        task_id = self.shared_state.closing_report_task_id
        if not task_id:
            return False
        from ..state.task_registry import TaskNotFound

        try:
            task = await self.tasks.get(task_id)
        except TaskNotFound:
            return True
        return task.state in {
            "succeeded",
            "failed",
            "cancelled",
            "needs_manual_review",
        }
