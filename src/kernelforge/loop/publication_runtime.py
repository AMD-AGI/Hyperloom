# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Best-result publication, optimization history, terminal state and the final run report for the forge-loop."""

from __future__ import annotations

import logging
import subprocess
import time


from kernelforge.loop.run_state import (
    SESSION_COMPLETED,
    SESSION_PAUSED,
    finish_session,
    make_event,
)
from kernelforge.loop.reporting import KeepBundle, KeepIdentity, KeepScore
from kernelforge.loop.iteration_types import IterationResult

log = logging.getLogger(__name__)


def llm_spend_lines(usage: dict) -> list[str]:
    """Render a campaign's LLM spend: the total, then the split by role.

    Four token columns, not two. Priced across the 316 recorded end-to-end
    campaigns, ``cache_read`` is 39.5% of the bill and ``cache_creation`` 32.9%,
    against 1.6% for uncached input -- so a summary that reports only ``in`` and
    ``out`` hides roughly three quarters of what was actually paid for. It also
    hides the effect of any change that shrinks the prompt, because what such a
    change moves is exactly these two columns: the campaign whose prefix fell
    83% reported the same ``in``/``out`` line as the one whose prefix did not.

    Counters are read with a default so a usage dict recorded by an older run --
    or a partial one checkpointed mid-campaign -- renders as 0 rather than
    raising while reporting a result that has already been computed.
    """

    def _row(counters: dict, cost_available: bool) -> str:
        cost = f"${counters.get('total_cost_usd', 0.0):.2f}" if cost_available else "cost unavailable"
        return (
            f"{counters.get('input_tokens', 0):,} in / "
            f"{counters.get('output_tokens', 0):,} out / "
            f"{counters.get('cache_creation_input_tokens', 0):,} cache-write / "
            f"{counters.get('cache_read_input_tokens', 0):,} cache-read tokens, "
            f"{cost} ({counters.get('calls', 0)} calls)"
        )

    cost_available = usage.get("cost_available", "total_cost_usd" in usage)
    lines = [f"  LLM spend: {_row(usage, cost_available)}"]
    # The total alone says a campaign was expensive; it never says what was
    # expensive. Print the split so the next cut can be aimed.
    for name, counters in (usage.get("by_role") or {}).items():
        lines.append(f"    {name}: {_row(counters, cost_available)}")
    return lines


class PublicationRuntimeMixin:
    """Own what the loop publishes: the best result, its history, the terminal state and the report."""

    def _checkpoint_llm_usage(self) -> None:
        """Best-effort checkpoint of the latest cumulative LLM usage."""
        if self._usage is None:
            return
        try:
            self.llm_usage = dict(self._usage.totals())
            has_usage = bool(self.llm_usage.get("calls")) or any(
                self.llm_usage.get(key)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            if self.experiment is not None and has_usage:
                self.tracker.set_llm_usage(
                    self.experiment.experiment_id,
                    self.llm_usage,
                )
        except (OSError, TypeError, ValueError):
            log.debug("failed to checkpoint LLM usage", exc_info=True)

    def _publish_best_result(
        self,
        result: IterationResult,
        *,
        plan: str,
        best_before: float | None,
        pending: dict | None = None,
    ) -> bool:
        """Publish one KEEP before another Agent session may start."""
        if not result.kept or not result.commit_hash:
            return False
        baseline = (
            (pending or {}).get("pristine_baseline_wall_ms")
            or (pending or {}).get("baseline_wall_ms")
            or self.ic.pristine_baseline_wall_ms
            or self.ic.publication_baseline_wall_ms
            or self.ic.baseline_wall_ms
            or self.run_state.baseline_wall_ms
            or best_before
        )
        if baseline is None or result.wall_ms is None or result.mean_case_speedup is None:
            return False
        validation_text = (
            (pending or {}).get("validation_text") or result.validation_summary or "canonical validation passed"
        )
        if result.error_output and not pending:
            validation_text = f"{validation_text}\n\n{result.error_output}".strip()
        benchmark = dict((pending or {}).get("benchmark") or result.bench_detail or {})
        benchmark.setdefault("median_ms", result.wall_ms)
        benchmark.setdefault("mean_case_speedup", result.mean_case_speedup)
        try:
            self.best_publisher.publish(
                KeepIdentity(
                    campaign_id=self.run_state.campaign_id,
                    session_index=int((pending or {}).get("session_index", self.run_state.session_index)),
                    experiment_id=(
                        str((pending or {}).get("experiment_id") or "")
                        or (self.experiment.experiment_id if self.experiment else "")
                    ),
                    iteration=result.iteration,
                    commit_hash=result.commit_hash,
                    plan=plan,
                ),
                KeepScore(
                    baseline_wall_ms=baseline,
                    best_wall_ms=result.wall_ms,
                    mean_case_speedup=result.mean_case_speedup,
                    search_start_mean_case_speedup=(
                        self.ic.warm_start_mean_case_speedup or self.search_start_mean_case_speedup or 1.0
                    ),
                    snr_db=result.snr_db,
                    search_start_ms=(self.ic.warm_start_wall_ms or self.ic.baseline_wall_ms),
                ),
                KeepBundle(
                    validation_text=validation_text,
                    benchmark=benchmark,
                    changed_files=(
                        list((pending or {}).get("publication_changed_files") or [])
                        or self._publication_changed_files(result.commit_hash)
                    ),
                    patch=(
                        str((pending or {}).get("publication_patch") or "")
                        or self._publication_patch(result.commit_hash)
                    ),
                ),
                round_budget=self._round_budget_summary(),
            )
            return True
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            first_failure = not self.persistence_degraded
            self.persistence_degraded = True
            self.persistence_errors.append(f"publish best iteration {result.iteration}: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            # The first drop into degraded persistence is the one an operator can still act on; the 12-hour run buried
            # it at debug and the run looked healthy until the final report.
            if first_failure:
                log.warning("failed to publish best result", exc_info=True)
            else:
                log.debug("failed to publish best result", exc_info=True)
            return False

    def _publish_optimization_history(self) -> None:
        """Regenerate history from durable events and candidate metadata."""
        events = self.state_store.read_events()
        metadata: dict[int, dict] = {}
        for event in events:
            if event.get("type") != "iteration_result":
                continue
            iteration = int(event.get("iter", 0) or 0)
            candidate = self.archive.load_meta(iteration)
            if not candidate:
                continue
            candidate["archive_path"] = f"candidates/iter_{iteration:03d}/"
            candidate["change_diff"] = self.archive.read_candidate_file(
                iteration,
                "change.diff",
            )
            metadata[iteration] = candidate
        try:
            self.best_publisher.publish_history(
                events=events,
                candidate_metadata=metadata,
            )
        except (OSError, ValueError) as error:
            self.persistence_degraded = True
            self.persistence_errors.append(f"publish optimization history: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            log.debug("failed to publish optimization history", exc_info=True)

    def _announce_run(self, supervisor_fn) -> None:
        """Arm the supervision monitor and print what this run is about to do.

        The finalize reserve is an absolute admission guard, so on a short budget
        it can swallow most of the window; the warning says so in the operator's
        own units rather than leaving them to divide.
        """
        # Self-supervision monitor (AVO): tracks stall / unproductive-cycle signals so the loop can call the
        # supervisor to redirect the search instead of stopping at the first plateau.
        self.monitor = None
        if supervisor_fn is not None:
            from kernelforge.loop.supervisor import SupervisionMonitor

            self.monitor = SupervisionMonitor(
                supervise_after=self.ic.supervise_after,
                cooldown=self.ic.supervise_cooldown,
            )
            print(self._supervisor_banner())

        print("Starting autonomous iteration loop")
        print(f"  Kernel: {self.ic.kernel_file}")
        print(f"  Target: {self.ic.target_wall_ms} ms")
        print(
            f"  Budget: {self.ic.max_time_hours}h "
            f"(finalize reserve: {self.ic.budget_reserve_sec / 60:.0f} min; "
            "a round is admitted only when what remains also covers its "
            "estimated cost)"
        )
        # The finalize reserve is an absolute admission guard; on a SHORT budget it can swallow most of the window
        # (e.g. a 30-min reserve on a 1h run leaves only 30 min for iterations).
        _budget_sec = self.ic.max_time_hours * 3600.0
        if _budget_sec > 0 and self.ic.budget_reserve_sec >= 0.5 * _budget_sec:
            _pct = 100.0 * self.ic.budget_reserve_sec / _budget_sec
            print(
                f"  WARNING: finalize reserve ({self.ic.budget_reserve_sec / 60:.0f} min) "
                f"consumes {_pct:.0f}% of the {self.ic.max_time_hours}h budget; "
                f"the effective iteration window is only "
                f"{max(0.0, _budget_sec - self.ic.budget_reserve_sec) / 60:.0f} min. "
                f"Raise --max-hours for a longer run."
            )
        print(f"  Experiment: {self.experiment.experiment_id}")
        print()

    def _persist_pr_knowledge(self) -> None:
        """Persist the PR-knowledge snapshot and its provenance event.

        Only after the fresh-campaign guard has completed, so a refused run
        leaves the workspace it refused untouched. A snapshot that cannot be
        written degrades the recorded reason rather than failing the campaign,
        because the references are context, not the objective.
        """
        # Persist only after the fresh-campaign guard has completed.
        if self.ic.pr_kb_snapshot:
            from kernelforge.knowledge.pr_knowledge.references import commit_snapshot
            from kernelforge.knowledge.pr_knowledge.context import REASON_LOCAL_FAILURE

            try:
                commit_snapshot(self.ic.workspace_dir, self.ic.pr_kb_snapshot)
            except (OSError, ValueError) as error:
                print(f"  [pr-kb] warning: snapshot not persisted ({error})")
                if self.ic.pr_kb_event:
                    self.ic.pr_kb_event = dict(self.ic.pr_kb_event)
                    self.ic.pr_kb_event["degraded_reason"] = REASON_LOCAL_FAILURE
                else:
                    self.ic.pr_kb_event = {
                        "position": "A",
                        "reason": REASON_LOCAL_FAILURE,
                        "degraded_reason": REASON_LOCAL_FAILURE,
                    }
            self.ic.pr_kb_snapshot = {}
        if self.ic.pr_kb_event:
            try:
                self.state_store.append_event(make_event("pr_refs_refreshed", 0, **self.ic.pr_kb_event))
            except (OSError, ValueError) as error:
                print(f"  [pr-kb] warning: event not recorded ({error})")
            self.ic.pr_kb_event = {}

    def _print_final_report(self) -> None:
        """Print what the campaign did, and what it cost.

        The round figures are campaign totals rather than this session's, so
        they are labelled as such and their share is taken against the campaign
        clock rather than the session wall-clock printed above them.
        """
        # Final report
        total_time = time.time() - self.start_time
        kept_count = sum(1 for r in self.results if r.kept)
        print(f"\n{'=' * 60}")
        print("Autonomous loop complete")
        print(f"  Iterations: {len(self.results)}")
        print(f"  Kept: {kept_count}, Reverted: {len(self.results) - kept_count}")
        if self.monitor is not None:
            print(f"  Supervisor interventions: {self.monitor.intervention_count}")
        print(f"  Best mean case speedup: {self.best_mean_case_speedup}x")
        print(
            "  Selected candidate raw mean_ms (diagnostic; not monotonic, but "
            "the published manifest withdraws its improvement badge when it "
            f"contradicts the score): {self.best_wall_ms}"
        )
        print(f"  Total time: {total_time / 60:.1f} minutes")
        costs = self.run_state.round_costs
        if costs.rounds:
            # Campaign totals, not this session's, so they are labelled as such and the share is taken against the
            # campaign clock rather than the ``total_time`` printed just above.
            self._advance_campaign_clock()
            share = costs.planning_share_pct()
            share_text = f" ({share:.0f}% of campaign wall-clock)" if share is not None else ""
            print(
                f"  Rounds planned across the campaign: {costs.rounds}, "
                f"planning {costs.planning_total_sec / 60:.1f} min"
                f"{share_text}, "
                f"round wall-clock {costs.total_sec / 60:.1f} min"
            )
        if self._refused_round:
            print(
                "  ROUND REFUSED FOR BUDGET: the campaign stopped because no "
                "round fit the time left, not because it found nothing — "
                f"{self._refused_round}"
            )
        if self.llm_usage.get("calls"):
            for line in llm_spend_lines(self.llm_usage):
                print(line)
        print(f"  Experiment: {self.experiment.experiment_id}")

    def _persist_terminal_state(self) -> None:
        """Close the last round and record why the run ended.

        Whatever the last iteration cost belongs to this campaign's history even
        though no further round will read it: a resumed session will, and so
        will an inspection asking what the final best was.
        """
        # Whatever the last iteration cost belongs to this campaign's history even though no further round will read
        # it: a resumed session will.
        self._close_round()
        self.best_publisher.refresh_round_budget(self._round_budget_summary())

        # Persist the terminal control state so a resume/inspection sees why the run ended and what the final best
        # was.
        try:
            terminal_reason = self.termination_reason or self.run_state.termination_reason or "unknown"
            finish_session(
                self.run_state,
                status=(SESSION_COMPLETED if terminal_reason == "gate_met" else SESSION_PAUSED),
                reason=terminal_reason,
            )
            head_out = self._git("rev-parse", "HEAD").strip()
            if head_out:
                self.run_state.head_commit = head_out.splitlines()[0]
            self.state_store.append_event(
                make_event(
                    "run_terminated",
                    self.run_state.iteration,
                    reason=terminal_reason,
                    best_wall_ms=self.best_wall_ms,
                    best_mean_case_speedup=self.best_mean_case_speedup,
                )
            )
            self.state_store.save(self.run_state)
        except (OSError, subprocess.SubprocessError):
            log.debug("run_state: terminal save failed", exc_info=True)
        self.persistence_degraded = self.persistence_degraded or self.state_store.degraded
        self.persistence_errors = (self.persistence_errors + self.state_store.persistence_errors)[-10:]

        # Persist the run's total LLM token spend onto the experiment record so external callers can read the token
        # cost.
        self._checkpoint_llm_usage()

        if self.experiment:
            self.tracker.mark_complete(self.experiment.experiment_id)
