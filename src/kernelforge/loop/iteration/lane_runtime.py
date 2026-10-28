# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Implementer lanes: fanning a round out across workspaces, queueing what they wrote, and the merge a stalled search may buy."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from kernelforge.loop.experiments_layout import LANE_PLAN_GLOB, ROUND_CONTEXT, ExperimentsLayout, lane_plan_name
from kernelforge.loop.lessons import SummaryOutcome
from kernelforge.loop.fanout import LanePlan, LaneResult, run_lanes
from kernelforge.loop.iteration.iteration_types import HeldRound, IterationResult, _StagedMerge, _append_finding
from kernelforge.loop.merge_candidates import (
    MERGE_ATTEMPT_STALL_THRESHOLD,
    MERGE_PRECEDENCE_STREAK_LIMIT,
    MergeCandidate,
    attempted_pairs,
    eligible_candidates,
    select_merge_pair,
)
from kernelforge.loop.run_state import make_event
from kernelforge.tools.durable_io import atomic_write_json, atomic_write_text
from kernelforge.tools.git import git
from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS

log = logging.getLogger(__name__)


def _lane_prompt(plan: str, *, serialized_driver: Path) -> str:
    """One lane's plan, with the one thing about the wrapper the session cannot see."""
    return (
        f"`python3 {serialized_driver}` takes an exclusive lock on the GPU this "
        "round shares, so it may sit silent before it starts. That wait is "
        "another lane's benchmark, not a hang -- wait for it rather than looking "
        "for another way to run the driver.\n\n"
        f"{plan}"
    )


def _patch_paths(patch: str, *, cwd: str) -> list[str]:
    """Every workspace path a patch writes, as git itself reads them."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".diff", encoding="utf-8", delete=False)
    try:
        handle.write(patch if patch.endswith("\n") else patch + "\n")
        handle.close()
        completed = git("apply", "--numstat", "-z", handle.name, cwd=cwd, check=False)
    finally:
        os.unlink(handle.name)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or f"git apply --numstat exited {completed.returncode}"
        raise ValueError(f"the paths in the diff could not be read: {detail}")
    paths: list[str] = []
    records = [record for record in completed.stdout.split("\0") if record]
    for record in records:
        fields = record.split("\t")
        if len(fields) < 3:
            raise ValueError(
                f"the paths in the diff could not be read: unreadable git apply --numstat record {record!r}"
            )
        paths.append(fields[2])
    for line in patch.splitlines():
        for prefix in ("rename from ", "copy from "):
            if line.startswith(prefix):
                paths.append(line[len(prefix) :].strip())
    return sorted({path for path in paths if path})


class LaneRuntimeMixin:
    """Run a round as concurrent lanes: plan them, persist what they bought, apply the next candidate, and stage a merge attempt."""

    async def _fan_out_round(
        self,
        *,
        iteration: int,
        orchestration_service,
        agent_factory,
        lanes: int | None = None,
    ) -> HeldRound | None:
        """Plan one round as lanes and run their sessions concurrently."""
        self._last_lane_plans = []
        # Reset here rather than at the ordinary path's planning call, which the round may now stand in for: a reused
        # plan must report its own executability and not the previous iteration's.
        self._last_orchestration_plan_executable = None
        plan_path: Path | None = None
        recovered = self._recoverable_lane_plans(iteration)
        if recovered is not None:
            planned_iteration, plans = recovered
            print(f"  [lanes] resuming {len(plans)} plans iteration {planned_iteration} paid for and never dispatched")
            self._last_lane_plans = plans
        else:
            width = max(1, int(self.ic.lanes if lanes is None else lanes))
            print(f"  [lanes] planning {width} concurrent Implementer lanes...")
            plan_path, error = await self._plan_round(
                iteration=iteration,
                orchestration_service=orchestration_service,
                lanes=width,
            )
            if plan_path is None:
                print(f"  [lanes] planning unavailable ({error}); falling back")
                return HeldRound(None, error)
        if len(self._last_lane_plans) < 2:
            print("  [lanes] one plan available; running the ordinary session")
            return HeldRound(plan_path, "")
        try:
            if recovered is not None:
                # Republished under the iteration that actually runs them, so this round is as recoverable as the one
                # it inherited from and an iteration's artifacts still describe what it did.
                plan_path = self._persist_lane_plans(
                    iteration,
                    self._last_lane_plans,
                    analysis_commit=self._canonical_commit(),
                )
                self._latest_optimization_plan_path = str(plan_path)
            # The round's plans exist and their cost is spent; what is priced here is only the sessions and the
            # measurement still to come.
            if not self._admit_dispatch(iteration):
                # The plans stay on disk and this iteration records no result, so the next session runs them instead
                # of buying them again.
                return HeldRound(plan_path, "")
            await self._fill_lane_queue(
                iteration=iteration,
                agent_factory=agent_factory,
                lane_plans=self._last_lane_plans,
            )
        except (OSError, RuntimeError) as error:
            self._lane_queue = []
            print(f"  [lanes] fan-out unavailable ({error}); falling back")
        # A recovered round that could not even republish holds nothing, and nothing was spent on it: that iteration
        # plans for itself as usual.
        return None if plan_path is None else HeldRound(plan_path, "")

    async def _fill_lane_queue(self, *, iteration: int = 0, agent_factory, lane_plans) -> None:
        """Run this round's lane sessions concurrently and queue what they wrote."""
        kernel_relative = self._workspace_path(self.ic.kernel_file)

        async def _session(
            lane: LanePlan,
            lane_dir: Path,
            serialized_driver: Path,
        ) -> SummaryOutcome:
            # The session edits the lane's own copy, so it is handed that copy's kernel rather than the canonical one.
            # The record turn resumes that session now: the round deletes this copy as soon as it returns.
            sink: dict = {}
            call = agent_factory(str(lane_dir), str(serialized_driver))
            await call(
                str(lane_dir / kernel_relative),
                _lane_prompt(lane.plan, serialized_driver=serialized_driver),
                session_sink=sink,
            )
            lesson = await self._request_lesson(iteration, sink.get("end_reason", ""), sink.get("summarize"))
            if not lesson:
                print(f"  [lane {lane.lane_id}] no lesson from its session ({lesson.reason})")
            return lesson

        results = await run_lanes(
            workspace_dir=self.ic.workspace_dir,
            lanes=[LanePlan(lane_id=str(index + 1), plan=plan) for index, plan in enumerate(lane_plans)],
            session=_session,
            # Beside the workspace, not in /tmp: a lane copy carries the build outputs and the whole experiment
            # archive, and /tmp is typically a smaller local filesystem than the one sized for the campaign.
            parent_dir=str(Path(self.ic.workspace_dir).resolve().parent),
            driver=self._workspace_path(self.ic.driver_script),
        )
        for result in results:
            if result.error:
                print(f"  [lane {result.lane_id}] session lost: {result.error}")
        active_profile_id = self.run_state.agent_relay.active_profile_id
        self._lane_queue = [
            replace(item, agent_profile_id=active_profile_id) for item in results if item.produced_candidate
        ]
        print(f"  [lanes] {len(self._lane_queue)} of {len(results)} lanes produced a candidate")
        self._persist_lane_queue()
        # The device is not per-lane.
        contended = [item for item in results if item.contended]
        if contended:
            hazard = self.device_hazard.record(
                iteration=iteration,
                detail="; ".join(f"lane {item.lane_id}: {item.reaped.describe()}" for item in contended),
                pids={pid for item in contended for pid in item.reaped.blockers},
            )
            print(
                f"  [lanes] {len(contended)} of {len(results)} lanes left the "
                f"device contended; this round measures nothing. "
                f"{hazard.describe()}"
            )

    def _unmeasurable_on_a_held_device(
        self,
        *,
        iteration: int,
        detail: str,
        session_sink: dict,
    ) -> IterationResult:
        """File an iteration that refused to measure on a device it does not own."""
        summary = (
            "REVERT (workspace contention): canonical correctness and benchmark "
            "were skipped because the device is still held by processes this "
            f"campaign could not clear. {detail}"
        )
        _append_finding(session_sink, summary)
        print("  [REVERT] Device still contended; nothing planned, run or measured this iteration")
        return IterationResult(
            iteration=iteration,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary=summary,
            kept=False,
            workspace_contention=detail,
        )

    def _lane_rejection(self, patch: str) -> str:
        """Why a lane candidate must not reach the canonical tree, or \"\"."""
        try:
            paths = _patch_paths(patch, cwd=self.ic.workspace_dir)
        except ValueError as error:
            return str(error)
        protected = sorted(
            path
            for path in paths
            if FORGE_PROTECTED_PATHS.is_protected(
                path,
                workspace=self.ic.workspace_dir,
                # The campaign driver carries no protected name of its own.
                exact_paths=(self.ic.driver_script,),
            )
        )
        if protected:
            return "it changes the measurement surface: " + ", ".join(protected)
        return ""

    def _take_lane_candidate(self) -> LaneResult | None:
        """Apply the next queued lane candidate to the canonical tree."""
        try:
            return self._next_lane_candidate()
        finally:
            self._persist_lane_queue()

    def _next_lane_candidate(self) -> LaneResult | None:
        while self._lane_queue:
            lane = self._lane_queue.pop(0)
            rejection = self._lane_rejection(lane.diff)
            if rejection:
                print(f"  [lane {lane.lane_id}] candidate rejected: {rejection}")
                continue
            if not self._git_apply_patch(lane.diff):
                self._git_discard_worktree()
                print(f"  [lane {lane.lane_id}] candidate no longer applies; dropped")
                continue
            try:
                # The driver is the measurement boundary and must remain byte-for-byte canonical.
                self._validate_driver_integrity(self.run_state)
            except ValueError as error:
                # Take the candidate back off the tree before anything else, so the next candidate does not inherit
                # it.
                self._git_discard_worktree()
                self._validate_driver_integrity(self.run_state)
                print(f"  [lane {lane.lane_id}] candidate rejected: {error}")
                continue
            return lane
        return None

    def _select_merge_attempt(
        self,
    ) -> tuple[MergeCandidate, MergeCandidate] | None:
        """Two rejected gains worth measuring stacked, once single patches stall."""
        if not self.ic.merge_stacking:
            return None
        if self.run_state.stall.unresolved_stall_iters < MERGE_ATTEMPT_STALL_THRESHOLD:
            return None
        incumbent_case_times = self._scored_incumbent_case_times()
        if not incumbent_case_times:
            return None
        index = self.archive.load_index()
        metas = []
        for row in index:
            meta = self.archive.load_meta(int(row.get("iter") or 0))
            if meta:
                metas.append(meta)
        return select_merge_pair(
            eligible_candidates(metas, incumbent_case_times),
            already_attempted=(
                attempted_pairs([str(row.get("plan") or "") for row in index]) | frozenset(self._declined_merge_pairs)
            ),
        )

    def _merge_attempt_refusal(self) -> str:
        """Why a selected pair may not be measured this iteration, or \"\"."""
        if self._merge_precedence_streak >= MERGE_PRECEDENCE_STREAK_LIMIT:
            return (
                f"{self._merge_precedence_streak} stacked iterations have run "
                "back to back without the queue being reached"
            )
        return ""

    def _stage_merge_attempt(
        self,
        pair: tuple[MergeCandidate, MergeCandidate] | None,
    ) -> tuple[str, str]:
        """Put both candidates' diffs in the tree; the diff, or why there is none."""
        if pair is None:
            return "", ""
        if self._working_tree_diff().strip():
            return "", self.TREE_ALREADY_DIRTY_OBSTACLE
        for candidate in pair:
            patch = self.archive.read_candidate_file(candidate.iteration, "change.diff")
            if not str(patch or "").strip():
                # Reported apart from a conflict because the two ask for opposite responses.
                self._git_discard_worktree()
                return "", (f"iteration {candidate.iteration}'s archived diff is missing or unreadable")
            if not self._git_apply_patch(patch):
                self._git_discard_worktree()
                return "", (f"iteration {candidate.iteration}'s diff would not apply over the other's")
        staged = self._working_tree_diff()
        if not staged.strip():
            self._git_discard_worktree()
            return "", "both diffs applied but changed nothing against HEAD"
        return staged, ""

    def _decline_merge_attempt(
        self,
        iteration: int,
        pair: tuple[MergeCandidate, MergeCandidate],
        obstacle: str,
        *,
        about_the_iteration: bool = False,
    ) -> None:
        """Report a pair that was selected and not measured, and drop it or not."""
        print(f"  [merge] declined: {obstacle}")
        if not about_the_iteration and obstacle != self.TREE_ALREADY_DIRTY_OBSTACLE:
            self._declined_merge_pairs.add(frozenset({pair[0].iteration, pair[1].iteration}))
        self.state_store.append_event(
            make_event(
                "merge_attempt_declined",
                iteration,
                first_iteration=pair[0].iteration,
                second_iteration=pair[1].iteration,
                obstacle=obstacle,
            )
        )

    def _stage_merge(self, *, iteration, hazard, fan_out_plan) -> _StagedMerge:
        """Stack two rejected gains, and say which pair and what it staged.

        Stacking costs a measurement but no session, so it is tried before
        spending another Implementer round on a search that has stopped
        producing a new best -- and, once that search has stalled, before
        draining a candidate the same search bought.
        """
        # Stacking two rejected gains costs a measurement but no session, so it is tried before spending another
        # Implementer round on a search that has stopped producing a new best -- and, once that search has
        # stalled, before draining a candidate the same search bought.
        merge_pair = None if hazard or fan_out_plan is not None else self._select_merge_attempt()
        merge_refusal = "" if merge_pair is None else self._merge_attempt_refusal()
        lane_queue_depth = 0 if hazard else len(self._lane_queue)
        if merge_refusal:
            merge_diff, merge_obstacle = "", merge_refusal
        else:
            merge_diff, merge_obstacle = self._stage_merge_attempt(merge_pair)
        self._merge_precedence_streak = self._merge_precedence_streak + 1 if merge_diff else 0
        if merge_diff and lane_queue_depth:
            # Distinct from ``merge_attempt_staged``, and not foldable into it: that counts every stack measured,
            # this counts the ones that went ahead of a queue a round already paid for, which is the only thing
            # precedence can cost.
            print(f"  [merge] precedence over a lane queue {lane_queue_depth} deep")
            self.state_store.append_event(
                make_event(
                    "merge_took_precedence",
                    iteration,
                    lane_queue_depth=lane_queue_depth,
                    first_iteration=merge_pair[0].iteration,
                    second_iteration=merge_pair[1].iteration,
                    unresolved_stall_iters=(self.run_state.stall.unresolved_stall_iters),
                )
            )
        if merge_pair is not None and merge_obstacle:
            self._decline_merge_attempt(
                iteration,
                merge_pair,
                merge_obstacle,
                # A refusal did not reach the pair's diffs, so it is not evidence about them and must not be
                # remembered against them.
                about_the_iteration=bool(merge_refusal),
            )
        return _StagedMerge(merge_pair, merge_diff)

    def _orchestration_root(self, iteration: int) -> Path:
        return ExperimentsLayout.of(Path(self.ic.workspace_dir).resolve()).round(iteration)

    def _lane_plan_path(self, iteration: int, lane: int) -> Path:
        """Where one lane's plan lives."""
        return self._orchestration_root(iteration) / lane_plan_name(lane)

    def _lane_queue_path(self) -> Path:
        """Where a round's unspent candidates wait for the iteration that measures them."""
        return ExperimentsLayout.of(Path(self.ic.workspace_dir).resolve()).lane_queue

    def _persist_lane_queue(self) -> None:
        """Publish what this round has bought and not yet measured."""
        path = self._lane_queue_path()
        try:
            if not self._lane_queue:
                path.unlink(missing_ok=True)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {"candidates": [lane.to_dict() for lane in self._lane_queue]})
        except OSError as error:
            print(
                f"  [lanes] queue not durable ({error}); the "
                f"{len(self._lane_queue)} candidate(s) still queued are lost "
                "if this process does not measure them"
            )

    def _restore_lane_queue(self) -> None:
        """Pick up candidates a previous process bought and never measured."""
        path = self._lane_queue_path()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            queued = [LaneResult.from_dict(entry) for entry in record["candidates"]]
        except (OSError, ValueError, KeyError, TypeError):
            log.debug("no readable lane queue at %s", path)
            return
        queued = [lane for lane in queued if lane.produced_candidate]
        if not queued:
            return
        self._lane_queue = queued
        print(f"  [lanes] resuming {len(queued)} candidate(s) an earlier round bought and never measured")

    def _lane_plan_manifest_path(self, iteration: int) -> Path:
        """The record that says a round's plans are complete and what they are."""
        return self._orchestration_root(iteration) / "lane_plans.json"

    def _persist_lane_plans(
        self,
        iteration: int,
        plans: Sequence[str],
        *,
        analysis_commit: str,
    ) -> Path:
        """Atomically publish every lane's plan and return lane 1's path."""
        published = [str(plan).strip() for plan in plans]
        if not published or not published[0]:
            raise ValueError("optimization plan must not be empty")
        if not all(published):
            raise ValueError("every lane plan must be non-empty")
        if not analysis_commit:
            raise ValueError("lane plans must record the commit they describe")
        manifest = self._lane_plan_manifest_path(iteration)
        manifest.unlink(missing_ok=True)
        keep = {self._lane_plan_path(iteration, lane) for lane in range(2, len(published) + 1)}
        for stale in self._orchestration_root(iteration).glob(LANE_PLAN_GLOB):
            if stale not in keep:
                stale.unlink()
        for lane, plan in enumerate(published, start=1):
            atomic_write_text(self._lane_plan_path(iteration, lane), plan + "\n")
        atomic_write_json(
            manifest,
            {
                "analysis_commit": analysis_commit,
                "lanes": len(published),
                "agent_profile_id": self.run_state.agent_relay.active_profile_id,
            },
        )
        return self._lane_plan_path(iteration, 1)

    def _load_lane_plans(self, iteration: int) -> tuple[str, list[str]] | None:
        """One round's commit and plans in lane order, or None if it has none."""
        try:
            manifest = json.loads(self._lane_plan_manifest_path(iteration).read_text(encoding="utf-8"))
            lanes = int(manifest["lanes"])
            analysis_commit = str(manifest["analysis_commit"])
        except (OSError, ValueError, KeyError, TypeError):
            log.debug("no readable lane plan manifest for iteration %s", iteration)
            return None
        plans: list[str] = []
        for lane in range(1, lanes + 1):
            try:
                plan = self._lane_plan_path(iteration, lane).read_text(encoding="utf-8").strip()
            except OSError:
                log.debug("lane %s of iteration %s is unreadable", lane, iteration)
                return None
            if not plan:
                return None
            plans.append(plan)
        if not plans or not analysis_commit:
            return None
        return analysis_commit, plans

    def _unfinished_iteration(self, before: int) -> int | None:
        """The iteration before ``before`` that started and reported no result."""
        started = 0
        finished: set[int] = set()
        for event in self.state_store.read_events():
            iteration = event.get("iter")
            if not isinstance(iteration, (int, float)):
                continue
            if int(iteration) >= before:
                continue
            if event.get("type") == "iteration_started":
                started = max(started, int(iteration))
            elif event.get("type") == "iteration_result":
                finished.add(int(iteration))
        if started and started not in finished:
            return started
        return None

    def _recoverable_lane_plans(
        self,
        iteration: int,
    ) -> tuple[int, list[str]] | None:
        """A previous round's plans that were paid for and never dispatched."""
        planned_iteration = self._unfinished_iteration(before=iteration)
        if planned_iteration is None:
            return None
        published = self._load_lane_plans(planned_iteration)
        if published is None:
            return None
        if not self._lane_plans_match_active_profile(planned_iteration):
            return None
        planned_commit, plans = published
        if len(plans) < 2:
            return None
        if planned_commit != self._canonical_commit():
            print(
                f"  [lanes] iteration {planned_iteration} planned against "
                f"{planned_commit}, which the tree has moved off; "
                "planning this round afresh"
            )
            return None
        return planned_iteration, plans

    def _persist_orchestration_result(self, iteration, context, result) -> None:
        """Persist planning diagnostics before publishing the executable plan."""
        root = self._orchestration_root(iteration)
        atomic_write_json(
            root / ROUND_CONTEXT,
            context.to_prompt_dict(),
        )
        if result.dispatch_plan is not None:
            atomic_write_json(
                root / "dispatch.json",
                result.dispatch_plan.to_dict(),
            )
        atomic_write_json(
            root / "specialists.json",
            {
                "analysis_commit": context.analysis_commit,
                "outcomes": [outcome.to_dict() for outcome in result.specialist_outcomes],
            },
        )
        diagnostics = dict(result.structured_output_diagnostics or {})
        artifact_paths = {}
        draft = str(result.optimization_plan_draft or "").strip()
        if draft:
            draft_path = root / "draft_plan.md"
            atomic_write_text(draft_path, draft + "\n")
            artifact_paths["draft_plan"] = str(draft_path.resolve())
        critic = result.plan_critic
        if critic is not None:
            critic_path = root / "critic_review.md"
            atomic_write_text(
                critic_path,
                critic.render_artifact().rstrip() + "\n",
            )
            artifact_paths["critic_review"] = str(critic_path.resolve())
        if artifact_paths:
            diagnostics["artifact_paths"] = artifact_paths
            diagnostics["plan_revised"] = bool(result.plan_revised)
        if diagnostics:
            atomic_write_json(
                root / "structured_output.json",
                diagnostics,
            )

    def _record_orchestration_final_plan(
        self,
        iteration: int,
        plan_path: Path,
    ) -> None:
        """Publish the final-plan pointer only after the plan exists."""
        diagnostics_path = self._orchestration_root(iteration) / "structured_output.json"
        if not diagnostics_path.is_file():
            return
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if not isinstance(diagnostics, dict):
            raise ValueError(f"invalid orchestration diagnostics: {diagnostics_path}")
        artifact_paths = diagnostics.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            return
        artifact_paths["final_plan"] = str(plan_path.resolve())
        atomic_write_json(diagnostics_path, diagnostics)

    def _record_probe_hazard(self, iteration: int, result) -> None:
        """Turn a probe the analysis phase could not clear into a live hazard."""
        diagnostics = getattr(result, "structured_output_diagnostics", None)
        finding = (diagnostics or {}).get("probe_device_hazard")
        if not isinstance(finding, dict):
            return
        hazard = self.device_hazard.record(
            iteration=iteration,
            detail=f"probe round: {finding.get('describe', '')}",
            pids=finding.get("pids") or (),
        )
        print(
            "  [probe] the round's probe scratch tree left the device "
            f"contended; this round measures nothing. {hazard.describe()}"
        )
