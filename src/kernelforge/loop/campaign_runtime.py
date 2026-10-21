# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Starting or resuming a campaign: the state it hydrates, the warm start it adopts, and how it ends."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from kernelforge.loop.iteration_types import IterationResult
from kernelforge.loop.run_state import (
    BestRecord,
    ORCHESTRATION_CIRCUIT_OPEN,
    RunState,
    SESSION_COMPLETED,
    SESSION_PAUSED,
    begin_orchestration_probe,
    finish_session,
    make_event,
    should_resume,
)

log = logging.getLogger(__name__)


class CampaignMixin:
    """Bring a campaign up: seed or hydrate its run state, adopt a validated warm start, and record how it ended."""

    def _seed_and_hydrate_run_state(self) -> None:
        """Seed a fresh session or hydrate an explicitly validated resume."""
        head_out = self._git("rev-parse", "HEAD").strip()
        head = head_out.splitlines()[0] if head_out else ""
        if self.resume and self.run_state.baseline_case_times:
            self._baseline_case_times = dict(self.run_state.baseline_case_times)
        if self.resume and should_resume(self.run_state, head):
            self.best_wall_ms = self.run_state.best.wall_ms
            self.best_mean_case_speedup = self.run_state.best.mean_case_speedup
            print(
                f"  [run-state] resumed best from {self.run_state.best.commit_hash[:8]}: "
                f"mean case speedup={self.run_state.best.mean_case_speedup:.6f}x, "
                f"raw mean={self.run_state.best.wall_ms} ms "
                f"(iter {self.run_state.best.iteration})"
            )
        if self.monitor is not None:
            self.monitor.no_improve_streak = self.run_state.stall.no_improvement_iters
            self.monitor.last_attempt_iter = self.run_state.stall.last_supervisor_attempt_iter or -10_000
            self.monitor.intervention_count = self.run_state.intervention_count
        if self.resume:
            # Without this the resumed session re-derives its own noise floor, incumbents and SNR reference, so it
            # judges candidates by different rules than the session it continues.
            if not self._scoring_state_restored:
                self._restore_scoring_state()
        if self.ic.baseline_wall_ms is not None:
            self.run_state.baseline_wall_ms = self.ic.baseline_wall_ms
        if self._baseline_case_times:
            self.run_state.baseline_case_times = dict(self._baseline_case_times)
        if self.ic.pristine_baseline_wall_ms is not None:
            self.run_state.pristine_baseline_wall_ms = self.ic.pristine_baseline_wall_ms
        if not self.resume:
            self.state_store.append_event(
                make_event(
                    "baseline_measured",
                    0,
                    baseline_wall_ms=self.ic.baseline_wall_ms,
                    mean_case_speedup=1.0,
                )
            )
        self.state_store.save(self.run_state)

    def _validate_pre_published_warm_start(
        self,
        *,
        commit_hash: str,
        baseline_ms: float,
        best_ms: float,
        mean_case_speedup: float,
    ) -> bool:
        """Validate the CLI's kill-recoverable warm-start publication."""
        publication = self.ic.warm_start_publication
        if not publication:
            return False

        try:
            manifest_path = Path(str(publication["best_manifest"]))
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("best manifest is not an object")
            checks = (
                int(publication.get("best_iteration", -1)) == 0,
                str(publication.get("best_commit") or "") == commit_hash,
                float(publication.get("baseline_ms")) == float(baseline_ms),
                float(publication.get("best_ms")) == float(best_ms),
                float(publication.get("mean_case_speedup")) == float(mean_case_speedup),
                int(manifest.get("iteration", -1)) == 0,
                str(manifest.get("commit_hash") or "") == commit_hash,
                float(manifest.get("baseline_wall_ms")) == float(baseline_ms),
                float(manifest.get("best_wall_ms")) == float(best_ms),
                float(manifest.get("mean_case_speedup")) == float(mean_case_speedup),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError("pre-published warm-start best artifact is unreadable") from error
        if not all(checks):
            raise RuntimeError("pre-published warm-start best artifact does not match the validated workspace state")
        return True

    def _stage_validated_warm_start_state(self) -> None:
        """Make iteration-zero warm-start state durable on the first save."""
        if self.resume or not self.ic.warm_start_commit:
            return
        head = self._git("rev-parse", "HEAD").strip()
        if head != self.ic.warm_start_commit:
            raise RuntimeError("validated warm-start commit is not the current workspace HEAD")
        if (
            self.ic.baseline_wall_ms is None
            or self.ic.pristine_baseline_wall_ms is None
            or self.ic.warm_start_wall_ms is None
            or self.ic.warm_start_mean_case_speedup is None
        ):
            raise RuntimeError("validated warm-start is missing performance baselines")
        warm_start_bench = dict(self.ic.warm_start_bench or {})
        self.run_state.best = BestRecord(
            iteration=0,
            wall_ms=self.ic.warm_start_wall_ms,
            mean_case_speedup=self.ic.warm_start_mean_case_speedup,
            commit_hash=head,
            plan=f"Validated start: {self.ic.warm_start_solution_slug}".strip(),
            source="warm_start",
        )
        self.run_state.head_commit = head
        self.run_state.baseline_wall_ms = self.ic.baseline_wall_ms
        self.run_state.pristine_baseline_wall_ms = self.ic.pristine_baseline_wall_ms
        self.run_state.baseline_case_times = dict(self.ic.baseline_case_times)
        self.run_state.best_case_times = dict(warm_start_bench.get("case_times") or {})
        self.run_state.unscored_cases = [str(case_id) for case_id in (warm_start_bench.get("unscored_cases") or [])]

    def _adopt_validated_warm_start(self) -> None:
        """Persist an applied KB seed as the recoverable local best at iteration 0."""
        if self.resume or not self.ic.warm_start_commit:
            return
        head = self._git("rev-parse", "HEAD").strip()
        if head != self.ic.warm_start_commit:
            raise RuntimeError("validated warm-start commit is not the current workspace HEAD")
        if (
            self.ic.baseline_wall_ms is None
            or self.ic.pristine_baseline_wall_ms is None
            or self.ic.warm_start_wall_ms is None
            or self.ic.warm_start_mean_case_speedup is None
        ):
            raise RuntimeError("validated warm-start is missing performance baselines")
        incumbent_wall_ms = self.ic.warm_start_wall_ms
        incumbent_mean_case_speedup = self.ic.warm_start_mean_case_speedup
        self.best_wall_ms = incumbent_wall_ms
        self.best_mean_case_speedup = incumbent_mean_case_speedup
        warm_start_bench = dict(self.ic.warm_start_bench or {})
        if not self._best_case_times:
            self._best_case_times = dict(warm_start_bench.get("case_times") or {})
        if not self._unscored_cases:
            self._unscored_cases = {str(case_id) for case_id in (warm_start_bench.get("unscored_cases") or [])}
        self.run_state.best = BestRecord(
            iteration=0,
            wall_ms=incumbent_wall_ms,
            mean_case_speedup=incumbent_mean_case_speedup,
            commit_hash=head,
            plan=f"Validated start: {self.ic.warm_start_solution_slug}".strip(),
            source="warm_start",
        )
        self.run_state.head_commit = head
        self.state_store.append_event(
            make_event(
                "warm_start_adopted",
                0,
                commit_hash=head,
                solution_slug=self.ic.warm_start_solution_slug,
                pristine_baseline_ms=self.ic.pristine_baseline_wall_ms,
                search_start_ms=incumbent_wall_ms,
                mean_case_speedup=incumbent_mean_case_speedup,
            )
        )
        self.state_store.save(self.run_state)
        self._persist_scoring_state()
        persisted = self.state_store.load()
        if (
            persisted.best.commit_hash != head
            or persisted.best.wall_ms != incumbent_wall_ms
            or persisted.best.mean_case_speedup != incumbent_mean_case_speedup
        ):
            raise RuntimeError("validated warm-start best state was not persisted")
        if self._validate_pre_published_warm_start(
            commit_hash=head,
            baseline_ms=self.ic.pristine_baseline_wall_ms,
            best_ms=self.ic.warm_start_wall_ms,
            mean_case_speedup=self.ic.warm_start_mean_case_speedup,
        ):
            return
        result = IterationResult(
            iteration=0,
            duration_sec=0.0,
            validation_passed=True,
            validation_summary=f"Validated start: {self.ic.warm_start_solution_slug}; correctness passed, timings recorded",
            wall_ms=incumbent_wall_ms,
            mean_case_speedup=incumbent_mean_case_speedup,
            kept=True,
            commit_hash=head,
            bench_detail={
                **warm_start_bench,
                "case_times": dict(self._best_case_times),
                "unscored_cases": sorted(self._unscored_cases),
                "median_ms": incumbent_wall_ms,
                "mean_case_speedup": incumbent_mean_case_speedup,
            },
        )
        if not self._publish_best_result(
            result,
            plan=self.run_state.best.plan,
            best_before=self.ic.pristine_baseline_wall_ms,
        ):
            raise RuntimeError("failed to publish validated warm-start best artifact")

    def _restore_or_begin_campaign(
        self, *, state_exists, current_ruling_path, agent_fn, orchestration_service, on_best_committed
    ) -> None:
        """Bring the run state up to where the campaign left off, or start a fresh one.

        A fresh campaign refuses a workspace that already holds one, because
        overwriting it would silently discard the earlier campaign's evidence.
        """
        if self.resume:
            self._validate_resume_scoring_state(self.run_state)
            # Pending KEEP reconciliation promotes the committed candidate and checkpoints scoring state.
            self._restore_scoring_state()
            # Publication reconciliation consumes both baseline anchors.
            if self.run_state.baseline_wall_ms is not None:
                self.ic.baseline_wall_ms = self.run_state.baseline_wall_ms
            if self.run_state.pristine_baseline_wall_ms is not None:
                self.ic.pristine_baseline_wall_ms = self.run_state.pristine_baseline_wall_ms
            self._restore_published_analysis_commit()
            pending = self._load_pending_keep()
            planned, pending_status, _, _ = self._plan_resume_recovery(
                self.run_state,
                pending,
            )
            self._validate_resume_state(
                planned,
                allow_dirty=pending_status == "uncommitted",
            )
            self._coordinate_resume_recovery(on_best_committed)
            self._restore_resume_baseline_case_times(self.run_state)
            if self._recovered_pending_keep is None:
                self._reconcile_best_publication()
            archive_next = self.archive.reconcile_next_iteration(
                self.run_state.next_iteration,
            )
            event_next = max(
                (
                    int(event.get("iter", 0) or 0) + 1
                    for event in self.state_store.read_events()
                    if isinstance(event.get("iter"), (int, float))
                ),
                default=1,
            )
            self.run_state.next_iteration = max(archive_next, event_next)
            if self.run_state.termination_reason == "orchestration_failed":
                if self.run_state.orchestration_circuit_state != ORCHESTRATION_CIRCUIT_OPEN:
                    raise ValueError("orchestration_failed resume requires an open circuit")
                if agent_fn is None or orchestration_service is None:
                    raise ValueError("orchestration_failed resume requires one orchestration probe")
                begin_orchestration_probe(self.run_state)
                self.state_store.append_event(
                    make_event(
                        "orchestration_circuit_half_open",
                        self.run_state.iteration,
                    )
                )
                self.state_store.save(self.run_state)
        else:
            self._validate_driver_integrity(self.run_state)
            if (
                state_exists
                or self.state_store.events_path.exists()
                or self._pending_keep_path.exists()
                or self.archive.max_iteration() > 0
                or current_ruling_path.exists()
                or (self.handoff_store is not None and self.handoff_store.latest() is not None)
            ):
                raise ValueError("workspace already contains a campaign; pass --resume to continue it")
            result = self._git("checkout", "-b", self.ic.git_branch)
            if "already exists" in result:
                self._git("checkout", self.ic.git_branch)
            current_branch = self._git("branch", "--show-current").splitlines()[0]
            if current_branch != self.ic.git_branch:
                raise ValueError(f"failed to switch workspace to branch {self.ic.git_branch}")
            self.run_state = RunState()

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
