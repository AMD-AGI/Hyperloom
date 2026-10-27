# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measuring one attempt and ruling on it: build, validate, bench, canonical acceptance, and the keep or revert that follows."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time

from kernelforge.evaluation.benchmark.runner import BenchRun, measure_wallclock
from kernelforge.evaluation.correctness.canonical import accept_candidate
from kernelforge.evaluation.correctness.pipeline import run_validation_pipeline
from kernelforge.evaluation.profiling.registers import check_registers
from kernelforge.evaluation.scoring import KEEP_MEASUREMENT_COUNT, runs_task_suite_acceptance
from kernelforge.loop.iteration.iteration_types import (
    IterationResult,
    LoopCallbacks,
    _Attempt,
    _IterationFrame,
    _Ruling,
    _decision_label,
)
from kernelforge.loop.jit_rebuild import force_jit_rebuild
from kernelforge.loop.run_state import make_event
from kernelforge.loop.iteration.scoring_runtime import _bench_failure_detail, _build_failure_tail
from kernelforge.tools.process_group import communicate_process_group


@contextlib.contextmanager
def _defer_termination_signals(enabled: bool):
    """Delay SIGTERM/SIGINT across best-commit checkpoint publication."""
    if not enabled:
        yield
        return
    pending: list[int] = []
    previous_handlers: dict[int, object] = {}

    def _defer(signum, _frame) -> None:
        pending.append(signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _defer)
    except ValueError:
        # signal.signal is available only on the process main thread.
        yield
        return
    try:
        yield
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if pending:
            os.kill(os.getpid(), pending[0])


class AttemptVerdictMixin:
    """Run one candidate through validation and benchmarking, rule on what came back, and settle what the iteration owes the campaign."""

    async def run_one_iteration(
        self,
        iteration: int,
        plan: str = "",
        *,
        benchmark_measurement: dict | None = None,
    ) -> IterationResult:
        """Execute a single build→validate→bench→canonical→decide iteration."""
        iter_start = time.time()
        force_jit_rebuild(self._jit_source_files())

        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *self.ic.build_command,
                cwd=self.ic.build_dir or self.ic.workspace_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await communicate_process_group(
                proc,
                timeout=self.ic.build_timeout_sec,
            )
            if proc.returncode != 0:
                return IterationResult(
                    iteration=iteration,
                    duration_sec=time.time() - iter_start,
                    validation_passed=False,
                    validation_summary=f"BUILD FAILED: {_build_failure_tail(stdout, stderr, 500)}",
                    kept=False,
                )

        measurement_started = time.time()
        print("  [validate] Running full correctness suite...")
        report = await run_validation_pipeline(
            driver_script=self.ic.driver_script,
            snr_threshold=self.ic.snr_threshold,
            timeout_per_stage=self.ic.validate_stage_timeout_sec,
        )
        for result in report.results:
            snr_str = f" SNR={result.snr_db:.1f}dB" if result.snr_db is not None else ""
            print(f"  [validate] Stage {result.stage} {result.stage_name}: {result.status_label}{snr_str}")
        if not report.all_passed:
            print(f"  [validate] FAILED at stage {report.failed_stage} — skipping bench")
            self._observe_measurement(measurement_started)
            return IterationResult(
                iteration=iteration,
                duration_sec=time.time() - iter_start,
                validation_passed=False,
                validation_summary=report.summary(),
                validation_outcome=report.failed_outcome,
                error_output=report.failed_output,
                kept=False,
            )

        if benchmark_measurement is not None and self._can_reuse_insession_benchmark(
            benchmark_measurement,
            attempt_diff=self._working_tree_diff(),
        ):
            print("  [bench] Reusing in-session three-measurement result...")
            bench_result = dict(benchmark_measurement)
            bench_result["reused_from_insession"] = True
        else:
            print("  [bench] Running three independent benchmark suites...")
            bench_result = await measure_wallclock(
                BenchRun(self.ic.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
                measurements=KEEP_MEASUREMENT_COUNT,
                timeout_sec=self.ic.bench_timeout_sec,
            )
        self._observe_measurement(measurement_started)
        self.last_case_bandwidth = dict(bench_result.get("case_bandwidth") or {})
        return await self._decide_iteration_keep(iteration, iter_start, report, bench_result)

    def _rule_on_attempt(
        self,
        result,
        attempt: _Attempt,
        frame: _IterationFrame,
        *,
        on_best_committed,
    ) -> _Ruling:
        """Keep or revert this attempt, and say what the verdict left behind.

        Returns the pending KEEP record, whether its checkpoint was finalized,
        and the commit the verdict settled on -- a KEEP commits here, so the
        hash the caller came in with is not the one it leaves with.
        """
        commit_hash, attempt_diff = attempt.commit_hash, attempt.attempt_diff
        ruling = _Ruling(None, False, commit_hash)
        elapsed = result.duration_sec
        raw_wall_txt = f"{result.wall_ms:.3f} ms" if result.wall_ms is not None else "unavailable"
        if not result.validation_passed or not result.kept:
            if commit_hash:
                self._git_revert_last()
            elif attempt_diff or self._new_paths_need_discard():
                self._git_discard_worktree()
            if not result.validation_passed:
                label = "Iteration crashed" if result.crashed else "Validation failed"
                print(f"  [REVERT] {label} ({elapsed:.0f}s)")
                print(f"           {result.validation_summary.splitlines()[-1] if result.validation_summary else ''}")
            else:
                speedup_txt = f"{result.mean_case_speedup:.6f}x" if result.mean_case_speedup is not None else "None"
                best_txt = f"{self.best_mean_case_speedup:.6f}x" if self.best_mean_case_speedup is not None else "?"
                print(
                    f"  [REVERT] mean case speedup={speedup_txt} not better than "
                    f"best={best_txt}; raw mean={raw_wall_txt} ({elapsed:.0f}s)"
                )
        else:
            ruling = self._commit_keep(result, attempt, frame, on_best_committed=on_best_committed)
            if result.kept:
                improvement = ""
                best_mean_case_speedup_before = frame.best_mean_case_speedup_before
                if best_mean_case_speedup_before and result.mean_case_speedup:
                    pct = (result.mean_case_speedup / best_mean_case_speedup_before - 1.0) * 100
                    improvement = f" ({pct:+.1f}% vs previous best)"
                snr_str = f" SNR={result.snr_db:.1f}dB" if result.snr_db is not None else ""
                print(
                    f"  [KEEP] mean case speedup={result.mean_case_speedup:.6f}x "
                    f"— NEW BEST{improvement}; raw mean={raw_wall_txt}"
                    f"{snr_str} ({elapsed:.0f}s)"
                )
        return ruling

    async def _close_iteration(
        self,
        result: IterationResult,
        attempt: _Attempt,
        frame: _IterationFrame,
        *,
        ruling: _Ruling,
        callbacks: LoopCallbacks,
    ) -> None:
        """Settle what this iteration owes the campaign now that it has a verdict.

        The publication, the lesson, the ledger entry, the archive and the
        handoff are all derived views of the same verdict, so they are
        written together and in that order: a resume reads them back as one
        iteration or not at all.
        """
        iteration, session_sink, best_before = frame.iteration, frame.session_sink, frame.best_before
        merge_pair, merge_diff = frame.merge
        pending_keep, keep_checkpoint_finalized, commit_hash = ruling
        # Remote/external work belongs outside the SIGTERM deferral window but still precedes potentially long
        # post-KEEP profiling.
        if keep_checkpoint_finalized and pending_keep is not None:
            self._publish_best_result(
                result,
                plan=session_sink.get("plan", ""),
                best_before=best_before,
                pending=pending_keep,
            )
        if result.kept and callbacks.on_best_ready:
            callbacks.on_best_ready(result)

        if self.experiment:
            self.tracker.log_iteration(
                self.experiment.experiment_id,
                config={"iteration": iteration, "kept": result.kept},
                snr_db=result.snr_db,
                wall_ms=result.wall_ms,
                mean_case_speedup=result.mean_case_speedup,
                pmc_diagnosis=result.pmc_diagnosis,
                vgpr=result.vgpr,
                decision="KEEP" if result.kept else "REVERT",
                notes=session_sink.get("plan", ""),
            )

        self.results.append(result)

        # Reduce this finished iteration into the durable run state + append a factual event, then checkpoint.
        if not keep_checkpoint_finalized:
            self._record_iteration_outcome(
                result,
                plan=session_sink.get("plan", ""),
            )

        # The verdict is now known, so ask the just-finished implementer session to record what it explored, then
        # stamp the measured outcome onto the same document.
        decision_label = _decision_label(result)
        if merge_diff and merge_pair is not None:
            print(
                f"  [merge] iterations {merge_pair[0].iteration}+"
                f"{merge_pair[1].iteration} measured stacked: {decision_label}"
            )
            if decision_label == "KEEP":
                # How often the mechanism changed an outcome, which is the other of the two numbers
                # ``merge_attempt_staged`` carries.
                self.state_store.append_event(
                    make_event(
                        "merge_attempt_kept",
                        iteration,
                        first_iteration=merge_pair[0].iteration,
                        second_iteration=merge_pair[1].iteration,
                        mean_case_speedup=result.mean_case_speedup,
                    )
                )
        iteration_diff_summary = (
            self._diff_summary(commit_hash) if commit_hash else self._diff_summary_from_diff(attempt.attempt_diff)
        )
        await self._record_lesson(
            iteration=iteration,
            result=result,
            decision=decision_label,
            session_sink=session_sink,
            diff_summary=iteration_diff_summary,
        )

        # Record this iteration into the cross-iteration experience ledger.
        self._record_ledger_iteration(
            result,
            frame,
            iteration_diff_summary=iteration_diff_summary,
            commit_hash=commit_hash,
            attempt_diff=attempt.attempt_diff,
        )

        # Archive the full solution and measurements as a derived view.
        archived_path = self._archive_attempt(result, attempt, frame, ruling=ruling, decision_label=decision_label)

        if not keep_checkpoint_finalized:
            self._publish_best_result(
                result,
                plan=session_sink.get("plan", ""),
                best_before=best_before,
            )
        self._record_iteration_handoff(
            iteration=iteration,
            decision=decision_label,
            optimization_plan_path=attempt.optimization_plan_path,
            session_sink=session_sink,
            archived_path=archived_path,
        )
        self._publish_optimization_history()

        if callbacks.on_iteration:
            callbacks.on_iteration(result)

        # A KEEP makes the prior evidence stale but does not discard its paths.
        if result.kept:
            self._analysis_bundle = None

    async def _decide_iteration_keep(
        self,
        iteration: int,
        iter_start: float,
        report,
        bench_result: dict,
    ) -> IterationResult:
        """Score the bench, check registers, apply the KEEP bar, and build the IterationResult."""
        # A candidate whose bench crashed is reverted for "no speedup" unless the crash itself reaches the agent; the
        # tool's output tail is the only place the traceback exists.
        bench_error_output = (
            "" if bench_result.get("success") else "BENCH FAILED: " + _bench_failure_detail(bench_result)
        )
        # Collapse per-case times into an equal-weight mean of per-case speedups, rather than allowing expensive cases
        # to dominate an aggregate ratio.
        self._apply_mean_case_speedup_metric(bench_result)
        measurement_scores = list(bench_result.get("measurement_mean_case_speedups") or [])
        sigma_resolution = await self._resolve_keep_sigma(bench_result, measurement_scores)
        self._emit_bench_scores(bench_result, sigma_resolution, measurement_scores, bench_error_output)
        selected_raw_mean_ms = bench_result.get("median_ms")
        mean_case_speedup = bench_result.get("mean_case_speedup")
        snr_db = report.results[-1].snr_db if report.results else None

        reg_result = await check_registers(build_dir=self.ic.build_dir)
        vgpr = reg_result.get("vgpr") if reg_result.get("success") else None
        if vgpr:
            print(f"  [registers] VGPR={vgpr}")

        improved = self._clears_keep_bar(bench_result, measurement_scores, sigma_resolution)

        # The numerical contract, which only assembly declares and only it needs. Every other backend was judged by
        # the driver and measured through it since; re-running that verdict here would answer the same question with
        # the same command, while reading a task configuration whose shape the engine has no business knowing.
        canonical_summary = ""
        if improved and runs_task_suite_acceptance(self.ic.kernel_backend):
            canonical_started = time.time()
            canonical = await accept_candidate(
                self.ic.workspace_dir,
                timeout_cap_sec=self.ic.validate_stage_timeout_sec,
                candidate_label=f"iteration {iteration}",
                kernel_backend=self.ic.kernel_backend,
            )
            # The suite only runs for a candidate the round produced, so it is part of that round's measurement and
            # has to be priced into the next round's admission alongside the validate-and-bench cycle.
            self._observe_measurement(canonical_started)
            canonical_summary = f"\n  Canonical correctness suite: {canonical.detail}"
            if canonical.numerical_evidence is not None:
                bench_result["numerical_validation"] = canonical.numerical_evidence
            if not canonical.passed:
                return IterationResult(
                    iteration=iteration,
                    duration_sec=time.time() - iter_start,
                    validation_passed=False,
                    validation_summary=(
                        f"{report.summary()}\n  Canonical correctness suite: FAILED — {canonical.detail}"
                    ),
                    validation_outcome=(canonical.outcome or "canonical_correctness_failure"),
                    wall_ms=selected_raw_mean_ms,
                    mean_case_speedup=mean_case_speedup,
                    snr_db=snr_db,
                    vgpr=vgpr,
                    error_output=canonical.output,
                    kept=False,
                    bench_detail=bench_result,
                )

        return IterationResult(
            iteration=iteration,
            duration_sec=time.time() - iter_start,
            validation_passed=True,
            validation_summary=report.summary() + canonical_summary,
            wall_ms=selected_raw_mean_ms,
            mean_case_speedup=mean_case_speedup,
            snr_db=snr_db,
            vgpr=vgpr,
            kept=improved,
            bench_detail=bench_result,
            error_output=bench_error_output,
        )
