# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE phase handler: the close sequencer, post-opt roofline, and the closing-grace / report-terminal helpers used by ``Coordinator.run``."""

from __future__ import annotations
import asyncio
import time
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any
from hyperloom.common.deadline import Deadline
import logging as _logging
from hyperloom.inference_optimizer.breakdown.recorder import close_out as _close_out
from hyperloom.inference_optimizer.breakdown.stop_reasons import is_valid_stop_reason

from . import machine_state as _phase_state
from ..bus.message_bus import Message
from ..state.task_registry import IllegalTransition, Task, TaskNotFound
from .base import PhaseHandler

log = _logging.getLogger(__name__)

# Terminal task states, split by what the CLOSE sequencer can still do with a task in one.
_TASK_STATE_DONE: str = "succeeded"
_DEAD_TASK_STATES: frozenset[str] = frozenset({"cancelled", "failed"})
# A row the wall-clock deadline path already dispatched.
_TASK_STATE_RUNNING: str = "running"
# Appended to a close step's idempotency key when its first row is dead, so ``create_or_return_existing`` mints a new
# task instead of returning the corpse.
_RETRY_KEY_SUFFIX: str = "retry"
# Fallback registry poll interval, for a caller with no dispatcher poll set.
_DEFAULT_TASK_POLL_SEC: float = 10.0

# Floor on how long CLOSE waits for a step it found already running, for a step the catalogue prices at almost nothing
# or does not carry at all.
_CLOSE_STEP_WAIT_FLOOR_SEC: float = 60.0

# Ceiling on the same wait.
_CLOSE_STEP_WAIT_CEILING_SEC: float = 600.0

# Stops that ask the process to go away now. A full-stack benchmark would outlive the request, so CLOSE publishes
# nothing for an unvalidated stack instead of measuring it.
_NO_REVALIDATION_STOP_REASONS: frozenset[str] = frozenset(
    {
        "signal",
        "emergency",
        "coordinator_exception",
        "supervisor_coordinator_died",
        "supervisor_tick_stalled",
        "unknown",
    }
)


def _task_is_dead(task: Task | None) -> bool:
    """True when ``task`` reached a terminal state without producing its artifact."""
    if task is None:
        return False
    return str(getattr(task, "state", "") or "") in _DEAD_TASK_STATES


class ClosePhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    def _derive_close_stop_reason(self) -> str:
        """Best-effort ``stop_reason`` for a CLOSE reached blank: recover from the newest CLOSE-bound phase_history row, else time_exhausted."""
        history = self.shared_state.phase_history or []
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            if (row.get("to_phase") or "").strip().upper() != _phase_state.PHASE_CLOSE:
                continue
            reason = (row.get("reason") or "").strip()
            evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
            if (
                reason == "sweep_done"
                and evidence.get("sweep_was_skipped")
                and evidence.get("sweep_skip_budget_exhausted")
                and str(evidence.get("sweep_skip_reason") or "") == "budget_exhausted_no_successful_pairs"
            ):
                return "sweep_failed"
            if reason and is_valid_stop_reason(reason):
                return reason
            # Newest CLOSE-bound row had no usable reason — stop rather than use a stale older one.
            break
        return "time_exhausted"

    def _session_integrated_kernel_patch(self) -> bool:
        """True iff this session landed a kernel-level optimization (optimization_stack has an integrate/gemm_tuning/geak_e2e entry). Gates the CLOSE post-opt roofline so pure param-search sessions skip the extra profile."""
        stack = getattr(self.shared_state, "optimization_stack", None) or []
        if not isinstance(stack, list):
            return False
        for entry in stack:
            if isinstance(entry, dict) and str(entry.get("action") or "") in self._POST_OPT_ROOFLINE_ACTIONS:
                return True
        return False

    async def _maybe_run_close_post_opt_roofline(self) -> None:
        """Best-effort: run one final post-opt roofline at CLOSE when a kernel/source patch was integrated."""
        if not self._session_integrated_kernel_patch():
            return
        # Skip on the wall-clock-deadline close path (its grace window is too short for a full profile+TraceLens);
        # only run on a normal converged close.
        if bool(getattr(self.shared_state, "closing_phase", False)):
            log.info("CLOSE step 0: skipped post-opt roofline (wall-clock closing grace window)")
            return
        if self._internal_analysis_kind() != "roofline":
            # Roofline disabled for this run; nothing to profile.
            return
        task = await self._enqueue_internal_analysis_task(reason="close_post_opt")
        if task is None:
            return
        log.info(
            "CLOSE step 0: running post-opt roofline task=%s (timeout=%.0fs)",
            task.task_id,
            self.CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC,
        )
        # Hard timeout so a slow profile+TraceLens can't stall the close sequence; on timeout no post-opt snapshot
        # lands and the chart degrades to baseline-only.
        try:
            result = await asyncio.wait_for(
                self.run_task_registered(task),
                timeout=self.CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.warning(
                "CLOSE step 0: post-opt roofline timed out after %.0fs; skipping (no post-opt snapshot)",
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
            except Exception:
                log.debug(
                    "CLOSE step 0: failed to mark timed-out post-opt roofline task",
                    exc_info=True,
                )
            return
        state = getattr(result, "state", None)
        log.info("CLOSE step 0: post-opt roofline finished (state=%s)", state)

    def _close_stack_revalidation_timeout_sec(self) -> float:
        """How long CLOSE waits for its full-stack revalidation to settle."""
        runtime_sec = float(getattr(self.shared_state, "baseline_runtime_sec", 0.0) or 0.0)
        return max(float(self.CLOSE_STACK_REVALIDATION_TIMEOUT_SEC), 2.0 * runtime_sec)

    async def _abandon_close_task(self, task: Task, *, reason: str) -> None:
        """Settle a close-owned task CLOSE stopped waiting for, so nothing dispatches or promotes it later."""
        try:
            current = await self.tasks.get(task.task_id)
            if current.state == "queued":
                await self.tasks.transition(task.task_id, "cancelled", {"reason": reason})
            elif current.state == "running":
                await self.tasks.transition(task.task_id, "failed", {"reason": reason})
        except (TaskNotFound, IllegalTransition):
            log.debug("CLOSE: failed to settle abandoned task %s", task.task_id, exc_info=True)

    async def _revalidate_stack_for_close(self) -> None:
        """Measure the working stack once when it changed after the last validation.

        Fact finalize publishes only a Recipe whose gain was measured on it, so a
        KEEP lifted after the last validation would otherwise leave the session
        with nothing to publish. The rebench promotes through the ordinary
        ``resume_stack_revalidate`` path, which is what moves the validated
        generation; a failed or incomparable run leaves it where it was and
        fact finalize skips.
        """
        state = self.shared_state
        has_unvalidated_keeps = getattr(state, "optimization_stack_has_unvalidated_keeps", None)
        if not (callable(has_unvalidated_keeps) and has_unvalidated_keeps()):
            return
        step = "stack_revalidation"
        stop_reason = str(getattr(state, "stop_reason", "") or "")
        if bool(getattr(state, "closing_phase", False)) or stop_reason in _NO_REVALIDATION_STOP_REASONS:
            await self._record_close_step(step, status="skipped", detail=f"stop_reason={stop_reason or '<none>'}")
            return
        if float(getattr(state, "baseline_tput", 0.0) or 0.0) <= 0.0:
            await self._record_close_step(step, status="skipped", detail="no_baseline")
            return
        # Explore refuses a variant the budget cannot fit, so asking here only
        # names the reason in the close section instead of in a no-op task.
        usable_sec = _phase_state.session_usable_seconds(state)
        needed_sec = _phase_state.one_more_measurement_sec(state) or _phase_state.measured_seconds(
            state, "baseline_runtime_sec"
        )
        if usable_sec is not None and needed_sec is not None and usable_sec < needed_sec:
            await self._record_close_step(
                step,
                status="skipped",
                detail=f"session_budget usable={usable_sec:.0f}s needed={needed_sec:.0f}s",
            )
            return
        generation = int(getattr(state, "working_recipe_generation", 0) or 0)
        summary = await self._enqueue_internal_stack_rebench(
            reason="close_unvalidated_stack",
            idempotency_key=f"close-stack-revalidate-g{generation}",
        )
        task_id = str(summary.get("task_id") or "")
        if not task_id:
            await self._record_close_step(step, status="skipped", detail=str(summary.get("reason") or "not_enqueued"))
            return
        task = await self.tasks.get(task_id)
        if task.state != "queued":
            # Only a row this CLOSE can run end to end is worth waiting on; a
            # settled one already promoted (or failed to) under its own run.
            await self._record_close_step(step, status="skipped", task_id=task_id, detail=f"task_state={task.state}")
            return
        timeout_sec = self._close_stack_revalidation_timeout_sec()
        log.info("CLOSE: revalidating the working stack task=%s (timeout=%.0fs)", task_id, timeout_sec)
        try:
            result = await asyncio.wait_for(
                self.run_task_registered(
                    task,
                    on_complete=partial(self._reap_dispatched_task, task, gpu_lease=None),
                ),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            log.warning("CLOSE: stack revalidation timed out after %.0fs; the stack stays unvalidated", timeout_sec)
            await self._abandon_close_task(task, reason="close_stack_revalidation_timeout")
            await self._record_close_step(step, status="failed", task_id=task_id, detail="timeout")
            return
        if result is None:
            await self._abandon_close_task(task, reason="close_stack_revalidation_lanes_busy")
            await self._record_close_step(step, status="skipped", task_id=task_id, detail="lanes_busy")
            return
        validated = not has_unvalidated_keeps()
        await self._record_close_step(
            step,
            status="done" if validated else "failed",
            task_id=task_id,
            detail=f"task_state={getattr(result, 'state', '')} validated={validated}",
        )

    def _record_close_roofline_progress(self) -> None:
        """Snapshot the session's roofline progress into the close section."""
        try:
            state = self.shared_state
            snapshots = state.roofline_snapshots if isinstance(state.roofline_snapshots, list) else []
            _close_out.record_roofline_progress(
                self.session_dir,
                baseline_tput=state.baseline_tput,
                baseline_ts=state.start_ts,
                optimization_stack=state.optimization_stack,
                latest_snapshot=snapshots[-1] if snapshots else None,
                current_best_tput=(state.current_best or {}).get("tput"),
                cumulative_gain_pct=state.cumulative_gain_validated,
                failure_streak=state.roofline_failure_streak,
            )
        except Exception:
            log.debug("CLOSE: roofline progress record failed", exc_info=True)

    def _close_stack_ledger(self) -> None:
        """Settle the stack ledger's timeline event.

        A session that never reaches here leaves the ledger open, and finalize
        reports it interrupted -- the truthful reading, since the reconciliation
        never ran.
        """
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import stack_event

            stack_event.finish()
        except Exception:
            log.debug("CLOSE: stack ledger close failed", exc_info=True)

    def _record_close_baseline_progress(self) -> None:
        """Snapshot the session's baseline failure tally into the close section."""
        try:
            state = self.shared_state
            _close_out.record_baseline_progress(
                self.session_dir,
                failure_streak=state.baseline_failure_streak,
                total_failures=state.baseline_total_failures,
                arg_error_streak=state.baseline_arg_error_streak,
            )
        except Exception:
            log.debug("CLOSE: baseline progress record failed", exc_info=True)

    def _record_close_final_recipe(self) -> None:
        """Snapshot the configuration the session ended on into the close section."""
        try:
            state = self.shared_state
            best = state.current_best if isinstance(state.current_best, dict) else {}
            config = self._current_best_launch_config()
            action_path: list[str] = []
            for entry in state.optimization_stack or []:
                if not isinstance(entry, dict):
                    continue
                action = str(entry.get("action") or "")
                variant = str(entry.get("variant_name") or "")
                action_path.append(f"{action}:{variant}" if variant else action)
            _close_out.record_final_recipe(
                self.session_dir,
                throughput=best.get("tput"),
                ttft_mean_ms=best.get("ttft_mean_ms"),
                e2el_mean_ms=best.get("e2el_mean_ms"),
                action_path=action_path,
                extra_server_args=config.get("extra_server_args") or "",
                extra_envs=config.get("extra_envs") or {},
            )
        except Exception:
            log.debug("CLOSE: final recipe record failed", exc_info=True)

    def _record_close_geak_candidate(self) -> None:
        """Snapshot where the GEAK candidate stood into the close section."""
        try:
            state = self.shared_state
            _close_out.record_geak_candidate(
                self.session_dir,
                pending=state.geak_pending if isinstance(getattr(state, "geak_pending", None), dict) else {},
                revalidation_pending=getattr(state, "resume_pending_revalidation", False),
            )
        except Exception:
            log.debug("CLOSE: geak candidate record failed", exc_info=True)

    async def _on_enter_close(self, *, from_phase: str) -> None:
        """CLOSE sequencer (fixed order): stack revalidation → post-opt roofline → fact_finalize → report → session_breakdown → langfuse flush → artifact_package → ndjson_drain (no-op) → mark close_sequence_done + stop_reason. Best-effort steps; final done step always runs. The ``CLOSE step N`` log labels are non-contiguous for historical reasons."""
        log.info("CLOSE entered (from=%s); starting 7-step close sequence", from_phase or "<unknown>")
        # Opened before anything can record a step into it. It stands at
        # ``running`` until the verdict below, so a session killed mid-sequence
        # is reported as interrupted rather than judged on the steps it reached.
        _close_out.record_close_opened(self.session_dir)
        await self._record_close_step("sequencer_started", status="running")

        # stop_reason must persist before step 2's breakdown (collector derives it from state.json); fill only when blank.
        if not self.shared_state.stop_reason:
            derived = self._derive_close_stop_reason()
            self.shared_state.set_stop_reason(derived)
            try:
                self.shared_state.save(self.session_dir)
            except Exception:
                log.exception("CLOSE: early stop_reason persist failed; step 5 will retry")

        # Ahead of the roofline and every close-section record, so they all
        # describe the stack after its last validation settled.
        try:
            await self._revalidate_stack_for_close()
        except Exception as exc:
            log.exception("CLOSE: stack revalidation failed")
            await self._record_close_step("stack_revalidation", status="failed", detail=repr(exc)[:240])

        # Post-optimization roofline (best-effort): profile the final optimized service once so the before/after
        # kernel roofline chart has its "after" column.
        try:
            await self._maybe_run_close_post_opt_roofline()
        except Exception as exc:  # noqa: BLE001
            log.warning("CLOSE step 0 (post-opt roofline) failed: %r", exc)

        # Recorded here rather than derived by the exporter: this is the first
        # moment each of these is final, and the snapshot history the exporter
        # would re-walk is capped and may have evicted what it needs.
        self._record_close_roofline_progress()
        self._record_close_baseline_progress()
        self._close_stack_ledger()
        # A revert leaves its adoption row standing, so what the stack ended as
        # is stated here or nowhere.
        self._record_close_final_recipe()

        # ---------------- Fact finalize (Recipe KB commit) -------------------
        # Publish before report/breakdown/Langfuse so the terminal outcome and
        # audit row are captured by the session's final telemetry.
        try:
            outcome = self.ensure_recipe_finalized(source="close") or {}
            kb_status = str(outcome.get("status") or "done")
            close_status = (
                "failed" if kb_status == "error" else "skipped" if kb_status in {"disabled", "skipped"} else "done"
            )
            detail = " ".join(
                f"{key}={outcome[key]}"
                for key in (
                    "status",
                    "reason",
                    "backend",
                    "canonical_id",
                    "session_id",
                )
                if outcome.get(key) not in (None, "")
            )
            await self._record_close_step(
                "fact_finalize",
                status=close_status,
                detail=detail,
            )
        except Exception as exc:
            log.exception("CLOSE step 0.5 (fact_finalize) failed")
            await self._record_close_step(
                "fact_finalize",
                status="failed",
                detail=repr(exc)[:240],
            )

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
            terminal_state = await self._run_close_task(report_task, step="1 (report)")
            if terminal_state in {"succeeded", None}:
                await self._record_close_step(
                    "report",
                    status="done",
                    task_id=report_task.task_id,
                )
                # Surface the final report location; advertise whichever of final.{json,md} exist under
                # reports_dir(session_dir).
                from hyperloom.inference_optimizer.session.session_paths import reports_dir as _reports_dir

                _rd = _reports_dir(self.session_dir)
                _json_path = _rd / "final.json" if (_rd / "final.json").exists() else None
                _md_path = _rd / "final.md" if (_rd / "final.md").exists() else None
                # Recorded where the step that wrote the files knows which of
                # them landed, so the export need not probe the filesystem.
                _close_out.record_close_artifacts(
                    self.session_dir,
                    final_json_path=_json_path,
                    final_md_path=_md_path,
                )
                _artifacts = {
                    "json_path": str(_json_path) if _json_path else "",
                    "md_path": str(_md_path) if _md_path else "",
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
        except Exception as exc:
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
            terminal_state = await self._run_close_task(bd_task, step="2 (session_breakdown)")
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
        except Exception as exc:
            log.exception("CLOSE step 2 (session_breakdown) failed")
            await self._record_close_step(
                "session_breakdown",
                status="failed",
                detail=repr(exc)[:240],
            )

        # ---------------- Langfuse flush + receipt splice ------------------- Must run before the artifact package:
        # flush_session flips the receipt to final counts and patch_breakdown_langfuse splices it back into
        # session_breakdown.json, so the bundled SBD carries final counts.
        try:
            from hyperloom.inference_optimizer.trace.langfuse_emitter import (
                flush_session,
                record_session_breakdown,
            )

            flush_session(self.session_dir)
            from hyperloom.inference_optimizer.breakdown import patch_breakdown_langfuse

            patch_breakdown_langfuse(self.session_dir)
            # Attach the final breakdown JSON to the trace as a ``session_breakdown`` observation (no-op when live
            # push is disabled).
            record_session_breakdown(self.session_dir)
        except Exception as exc:
            log.debug("CLOSE step 2.5 (langfuse flush) failed", exc_info=True)
            await self._record_close_step(
                "langfuse_flush",
                status="failed",
                detail=repr(exc)[:240],
            )

        # ---------------- Artifact package -> /workspace ------------------ Bundle the curated result/report/analysis
        # files into a single zip under ``/workspace`` so the Claw sandbox sync ships it to object storage even when
        # ``$USER_DATA_PATH`` points outside ``/workspace``.
        session_id = str(getattr(self.shared_state, "session_id", "") or "")
        pkg_path = None
        try:
            from hyperloom.inference_optimizer.breakdown import package_session_artifacts

            # Zipping a large session walks thousands of files; off the loop so it does not stall the Coordinator's
            # other shutdown work.
            pkg_path = await asyncio.to_thread(
                package_session_artifacts,
                self.session_dir,
                session_id=session_id,
            )
            if pkg_path is not None:
                # A field rather than something the export parses back out of
                # the step's free-text ``detail``, which also carries the skip
                # and failure reasons.
                _close_out.record_close_artifacts(self.session_dir, artifact_package_path=pkg_path)
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
        except Exception as exc:
            log.exception("CLOSE step 2.6 (artifact_package) failed")
            await self._record_close_step(
                "artifact_package",
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
        except Exception:
            log.exception(
                "CLOSE step 5 (close_sequence_done save) failed; cli.finally will still write a safety-net breakdown"
            )
        await self._record_close_step("done", status="done")

        # The verdict, recorded last because reaching this line is the evidence
        # for it. A reader looking at the steps alone cannot tell the ones that
        # had not happened yet from the ones that never will.
        _close_out.record_close_settled(
            self.session_dir,
            stop_reason=str(getattr(self.shared_state, "stop_reason", "") or ""),
        )

        # Refresh the breakdown's ``close`` key now that the sequence is on
        # disk: step 2 wrote the breakdown, so the copy it produced describes
        # only the close-out up to itself. Splices one key; best-effort and
        # last, after stop_reason and close_sequence_done are settled, so it
        # cannot affect the run.
        #
        # Re-package when that changed something: ``session_breakdown.json`` is
        # bundled into the zip and the package is what external sync ships. The
        # refresh has to come after ``artifact_package`` has an outcome to
        # report, so the bundle is rebuilt rather than reordered. A rebuild that
        # fails leaves the shipped zip holding the step-2 snapshot, and there is
        # no close step to record that against -- the rebuild rewrites the path
        # ``artifact_package`` already names -- so it is said in the log, at a
        # level the default configuration prints.
        try:
            from hyperloom.inference_optimizer.breakdown import patch_breakdown_close

            if patch_breakdown_close(self.session_dir) and pkg_path is not None:
                from hyperloom.inference_optimizer.breakdown import package_session_artifacts

                rebuilt = await asyncio.to_thread(
                    package_session_artifacts,
                    self.session_dir,
                    session_id=session_id,
                )
                if rebuilt is None:
                    log.warning(
                        "CLOSE step 6: close section refreshed but the artifact package rebuild produced "
                        "nothing; %s still carries the pre-refresh close section",
                        pkg_path,
                    )
        except Exception:
            log.warning(
                "CLOSE step 6 (close section refresh) failed; the artifact package may still carry "
                "the pre-refresh close section",
                exc_info=True,
            )

        log.info("CLOSE 7-step sequencer complete")

    async def _enqueue_runnable_internal_task(
        self,
        *,
        kind: str,
        params: dict[str, Any],
        idempotency_key: str,
    ) -> Task:
        """Enqueue a Coordinator-internal close-step task the sequencer can still run."""
        task: Task | None = None
        for key in (idempotency_key, f"{idempotency_key}-{_RETRY_KEY_SUFFIX}"):
            task, was_existing = await self.tasks.create_or_return_existing(
                kind=kind,
                params=params,
                idempotency_key=key,
                requires_lanes=[],
                side_effects=["writes_results"],
                lease_ttl_sec=120,
            )
            if not was_existing:
                return task
            if not _task_is_dead(task):
                log.info(
                    "internal-%s task reused (idempotent: task_id=%s, state=%s)",
                    kind,
                    task.task_id,
                    task.state,
                )
                return task
            log.warning(
                "internal-%s task %s is %s and cannot be run; re-enqueueing under a fresh key",
                kind,
                task.task_id,
                task.state,
            )
        return task  # type: ignore[return-value]  # loop body always binds it

    async def _enqueue_internal_report_task(
        self,
        *,
        reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``report`` task (idempotency_key internal-report-<reason>)."""
        existing_id = (self.shared_state.closing_report_task_id or "").strip()
        if existing_id:
            try:
                task = await self.tasks.get(existing_id)
            except TaskNotFound:
                task = None
            if task is not None and not _task_is_dead(task):
                log.info(
                    "internal-report task already enqueued by wall-clock "
                    "deadline path (task_id=%s, state=%s); sequencer will "
                    "wait for it",
                    task.task_id,
                    task.state,
                )
                return task
            # Dead or vanished: the id names a report that will never be written, so drop it before the fresh enqueue
            # mirrors its own.
            if task is not None:
                log.warning(
                    "internal-report task %s recorded on closing_report_task_id is %s; re-enqueueing",
                    task.task_id,
                    task.state,
                )
            self.shared_state.closing_report_task_id = ""

        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            "session_dir": str(self.session_dir),
            "max_highlights": 50,
        }
        task = await self._enqueue_runnable_internal_task(
            kind="report",
            params=params,
            idempotency_key=f"internal-report-{reason}",
        )
        # Mirror onto closing_report_task_id.
        if not self.shared_state.closing_report_task_id:
            self.shared_state.closing_report_task_id = task.task_id
            try:
                self.shared_state.save(self.session_dir)
            except Exception:
                log.exception("internal-report: closing_report_task_id save failed")
        return task

    async def _enqueue_internal_session_breakdown_task(
        self,
        *,
        reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``session_breakdown`` task; same idempotency contract as the report helper."""
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            "session_dir": str(self.session_dir),
        }
        return await self._enqueue_runnable_internal_task(
            kind="session_breakdown",
            params=params,
            idempotency_key=f"internal-session_breakdown-{reason}",
        )

    def _close_step_wait_sec(self, task: Task) -> float:
        """How long CLOSE waits for a close-step task to reach a terminal state."""
        from ..loop.coordinator_helpers import expected_action_cost_minutes

        registry = getattr(self, "action_registry", None)
        kind = str(getattr(task, "kind", "") or "")
        meta = registry.get(kind) if registry is not None else None
        typical_sec = expected_action_cost_minutes(meta) * 60.0
        return min(_CLOSE_STEP_WAIT_CEILING_SEC, max(_CLOSE_STEP_WAIT_FLOOR_SEC, typical_sec))

    async def _await_running_close_task(self, task: Task, *, step: str) -> str:
        """Wait for an already-dispatched close-step task to reach a terminal state."""
        bound_sec = self._close_step_wait_sec(task)
        poll_sec = float(getattr(self, "_dispatcher_poll_sec", _DEFAULT_TASK_POLL_SEC))
        deadline = time.monotonic() + bound_sec
        log.info(
            "CLOSE step %s: task_id=%s is already running; waiting up to %.0fs for it",
            step,
            task.task_id,
            bound_sec,
        )
        state = _TASK_STATE_RUNNING
        while True:
            try:
                state = str(getattr(await self.tasks.get(task.task_id), "state", "") or "")
            except TaskNotFound:
                log.warning("CLOSE step %s: task_id=%s vanished while the sequencer waited for it", step, task.task_id)
                return state
            if state != _TASK_STATE_RUNNING:
                log.info(
                    "CLOSE step %s: task_id=%s finished as %s while the sequencer waited",
                    step,
                    task.task_id,
                    state,
                )
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                log.warning(
                    "CLOSE step %s: task_id=%s still running after %.0fs; recording the step as failed",
                    step,
                    task.task_id,
                    bound_sec,
                )
                return state
            await asyncio.sleep(min(poll_sec, remaining))

    async def _run_close_task(self, task: Task, *, step: str) -> str | None:
        """Run one close-step task and return the state it ended in."""
        state = str(getattr(task, "state", "") or "")
        if state == _TASK_STATE_DONE:
            log.info(
                "CLOSE step %s: task_id=%s already succeeded; keeping its artifact",
                step,
                task.task_id,
            )
            return state
        if state in _DEAD_TASK_STATES:
            log.warning(
                "CLOSE step %s: task_id=%s is %s and cannot be run; recording the step as failed",
                step,
                task.task_id,
                state,
            )
            return state
        if state == _TASK_STATE_RUNNING:
            return await self._await_running_close_task(task, step=step)
        return await self._run_fresh_close_task(task, step=step)

    async def _run_fresh_close_task(self, task: Task, *, step: str) -> str:
        """Run a queued close-step task, bounded by the same wait as an in-flight one."""
        bound_sec = self._close_step_wait_sec(task)
        log.info(
            "CLOSE step %s: task_id=%s starting; waiting up to %.0fs for it",
            step,
            task.task_id,
            bound_sec,
        )
        try:
            result = await asyncio.wait_for(self.run_task_registered(task), timeout=bound_sec)
        except asyncio.TimeoutError:
            log.warning(
                "CLOSE step %s: task_id=%s still running after %.0fs; recording the step as failed",
                step,
                task.task_id,
                bound_sec,
            )
            return _TASK_STATE_RUNNING
        return result.state

    async def _record_close_step(
        self,
        step: str,
        *,
        status: str,
        task_id: str = "",
        detail: str = "",
    ) -> None:
        """Record one settled close step (best-effort, per-step persist).

        The row goes to the breakdown's ``close`` section, which is what the
        exported ``close.steps`` is built from, and to
        ``phase_history[-1].evidence.close_steps``, which is the phase evidence
        ledger. The two are written independently: a session with no phase
        history row to append to still has a close-out to report.
        """
        ts = datetime.now(timezone.utc).isoformat()
        _close_out.record_close_step(
            self.session_dir,
            step=step,
            status=status,
            ts=ts,
            task_id=task_id,
            detail=detail,
        )
        entry: dict[str, Any] = {
            "step": step,
            "status": status,
            "ts": ts,
        }
        if task_id:
            entry["task_id"] = task_id
        if detail:
            entry["detail"] = detail
        if not _phase_state.append_phase_evidence_row(
            self.shared_state.phase_history,
            key="close_steps",
            row=entry,
        ):
            return
        try:
            self.shared_state.save(self.session_dir)
        except Exception:
            log.exception(
                "close_step save failed for step=%r status=%r",
                step,
                status,
            )

    async def _enter_closing_phase(self, *, grace_sec: float) -> Deadline:
        """Enter report-flush phase after the wall-clock deadline (enqueue deterministic report task).

        Args:
            grace_sec: Seconds the closing phase may run before the report task
                is abandoned.

        Returns:
            Deadline: The instant by which the closing phase must complete.
        """
        closing_started = time.time()
        closing_deadline = Deadline.after(grace_sec)
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
        except Exception:
            log.exception(
                "closing_phase: cancel of queued tasks failed (non-fatal)",
            )

        idempotency_key = f"closing-report-{int(closing_started)}-{uuid.uuid4().hex[:6]}"
        task_id = ""
        task, _existing = await self.tasks.create_or_return_existing(
            kind="report",
            params={
                "session_dir": str(self.session_dir),
                "max_highlights": 50,
            },
            idempotency_key=idempotency_key,
            requires_lanes=[],
            side_effects=["writes_results"],
            lease_ttl_sec=120,
        )
        task_id = task.task_id
        self.shared_state.closing_report_task_id = task_id
        self.shared_state.save(self.session_dir)

        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                "*",
                "event",
                {
                    "kind": "closing_phase_entered",
                    "task_id": task_id,
                    "grace_sec": float(grace_sec),
                    "closing_started_unix": closing_started,
                },
            )
        )
        return closing_deadline

    async def ensure_close_sequence(self, *, reason: str) -> bool:
        """Run the close sequencer if nothing else has, whatever stopped the run.

        The sequencer is otherwise reached only when the phase machine
        transitions into CLOSE, so a run that ends without advancing a phase --
        a signal, an exception, an objective met at the deadline -- would write
        no report at all. Idempotent via ``close_sequence_done``.

        Args:
            reason: What terminated the run, recorded as the entry's from-phase.

        Returns:
            bool: ``True`` when this call ran the sequence.
        """
        if self.shared_state.close_sequence_done:
            return False
        log.info("CLOSE: no close sequence has run (reason=%s); running it now", reason)
        await self._on_enter_close(from_phase=reason)
        return True

    async def _closing_report_terminal(self) -> bool:
        """Report whether the closing phase has a finished report to wait on.

        An absent report task counts as finished, so the loop drops out of the
        closing branch and onto the terminal close sequence that writes the
        report itself rather than waiting out its grace on nothing.

        Returns:
            bool: ``True`` when the report task reached a terminal state, is
                missing, or was never enqueued; ``False`` while it is still
                queued or running.
        """
        task_id = self.shared_state.closing_report_task_id
        if not task_id:
            log.warning("closing_phase: no report task was enqueued; closing without waiting on one")
            return True
        try:
            task = await self.tasks.get(task_id)
        except TaskNotFound:
            return True
        return task.state in {
            "succeeded",
            "failed",
            "cancelled",
        }
