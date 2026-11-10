# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a tree round's sessions, measure every candidate where it was written, and admit the correct ones as nodes."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom_common.io import DURABLE, atomic_write_text
from kernelforge.agent_backends.session_resume import EXHAUSTED_END_REASON
from kernelforge.workflows.forge_loop.iteration.fanout import LanePlan, LaneResult, run_lanes
from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationResult
from kernelforge.workflows.forge_loop.iteration.lane_runtime import _lane_prompt
from kernelforge.workflows.forge_loop.memory.lessons import SummaryOutcome
from kernelforge.workflows.forge_loop.tree.model import (
    Attempt,
    AttemptOutcome,
    DirectionProposal,
    NodeMeasurement,
    NodeSource,
    SearchTree,
    Slot,
    SlotType,
    TreeNode,
)
from kernelforge.workflows.forge_loop.tree.planning import SlotPlan
from kernelforge.workflows.forge_loop.tree.profiles import SlotActors
from kernelforge.workflows.forge_loop.tree.refs import node_holding_tree, pin_node, point_direction_branch

_PLAN_SUMMARY_CHARS = 240


@dataclass(frozen=True)
class SlotOutcome:
    """What one slot came to: its attempt record, the node it added, and the measurement behind both."""

    attempt: Attempt
    node: TreeNode | None
    result: IterationResult | None


@dataclass(frozen=True)
class SessionAgent:
    """What builds one Implementer session, and the Agent profile it runs on; empty when the session has none."""

    factory: Callable
    profile_id: str = ""


@dataclass(frozen=True)
class LaneSession:
    """One Implementer session of a round: its lane, what it is asked, and the agent that runs it."""

    lane: LanePlan
    prompt: str
    agent: SessionAgent


@dataclass(frozen=True)
class StagedCandidate:
    """A candidate staged in the worktree of the node it was made on, the plan it came from, and where its records go."""

    worktree: Path
    start: str
    plan: str
    records: Path
    # The Agent profile whose session wrote the candidate; empty when the framework did.
    producer: str = ""


@dataclass(frozen=True)
class MeasuredCandidate:
    """What a staged candidate came to: the node that already holds it, or its measurement and the node it made."""

    holder: str = ""
    result: IterationResult | None = None
    node: TreeNode | None = None


def measured_correct(result: IterationResult | None) -> bool:
    """Whether a measurement makes its candidate a node: correct, and scored on every case three times."""
    if result is None or not result.validation_passed or result.mean_case_speedup is None:
        return False
    detail = result.bench_detail or {}
    return bool(detail.get("success")) and bool(detail.get("measurement_mean_case_speedups"))


def node_measurement(result: IterationResult) -> NodeMeasurement:
    """What a correct measurement adds to a node: the score and each case's time of every bench run."""
    detail = result.bench_detail or {}
    runs = (_case_run(item) for item in detail.get("measurements") or () if isinstance(item, dict))
    return NodeMeasurement(
        samples=tuple(float(item) for item in detail["measurement_mean_case_speedups"]),
        case_runs=tuple(run for run in runs if run),
    )


def _case_run(measurement: dict) -> dict[str, float]:
    return {
        str(case): float(ms)
        for case, ms in (measurement.get("case_times") or {}).items()
        if isinstance(ms, (int, float)) and float(ms) > 0.0
    }


def _summary(text: str) -> str:
    return " ".join(str(text or "").split())[:_PLAN_SUMMARY_CHARS]


def slot_prompt(plan: SlotPlan, start: TreeNode) -> str:
    """What one slot's Implementer is asked: where it starts, what counts, its lane's approach and focus, its plan."""
    preamble = (
        f"This session starts from search-tree node {start.commit} at {start.score:.6f}x (equal-weight mean "
        "pristine-relative case speedup). Aim to exceed it. The candidate must be correct; a correct but slower first "
        "step of the planned route is acceptable, is measured, and is kept in the search tree.\n\n"
    )
    return preamble + _lane_focus(plan) + plan.text


def _lane_focus(plan: SlotPlan) -> str:
    """What the round planner set this slot's lane apart by: its approach and the cases it aims at."""
    notes = []
    if plan.approach:
        notes.append(f"This lane's approach, which the other lanes at this node do not share: {plan.approach}")
    if plan.focus_cases:
        notes.append(
            f"This lane is aimed at these cases: {', '.join(plan.focus_cases)}. Make them faster; every case must "
            "stay correct, and the other cases may get slower, since the round merges its lanes' candidates so that "
            "each case runs on the fastest of them. While you iterate, you can time one of your cases alone by adding "
            "`--bench-case <case_id>` to the driver's bench-mode command (`--warmup <n> --iters <n> --bench-mode`); "
            "a driver that does not support the flag ignores or rejects it, and then times only its whole suite. Run "
            "the whole driver once before you finish."
        )
    return "".join(f"{note}\n\n" for note in notes)


class TreeAdmissionMixin:
    """Sessions, measurement and node admission for one tree round.

    The host supplies ``ic``, ``run_one_iteration``, ``_canonical_site``, ``_lane_rejection``, ``_workspace_path``,
    ``_request_lesson``, ``_check_profile_again``, ``_start_worktrees``, ``_tree_store``, ``_experiments`` and
    ``device_hazard``.
    """

    async def _run_slot_lanes(
        self, tree: SearchTree, iteration: int, plans: Sequence[SlotPlan], actors: SlotActors
    ) -> dict[int, LaneResult]:
        """Run one lane per planned slot on its own profile, copied from its start's worktree; results by slot."""
        sessions = [
            LaneSession(
                LanePlan(
                    lane_id=str(plan.slot.index),
                    plan=plan.text,
                    source_dir=str(self._start_worktrees.ensure(plan.slot.start)),
                ),
                slot_prompt(plan, tree.require_node(plan.slot.start)),
                SessionAgent(actors.of(plan.slot).agent_factory, plan.slot.agent_profile_id),
            )
            for plan in plans
        ]
        results = await self._run_sessions(iteration, sessions)
        return {int(lane_id): result for lane_id, result in results.items()}

    async def _run_sessions(self, iteration: int, sessions: Sequence[LaneSession]) -> dict[str, LaneResult]:
        """Run one Implementer session per lane, each asked its prompt on a copy of its lane's checkout."""
        kernel_relative = self._workspace_path(self.ic.kernel_file)
        by_lane = {session.lane.lane_id: session for session in sessions}

        async def _session(lane: LanePlan, lane_dir: Path, serialized_driver: Path) -> SummaryOutcome:
            session = by_lane[lane.lane_id]
            # The record turn resumes this session now: the round deletes the lane's copy as soon as it returns.
            sink: dict = {}
            call = session.agent.factory(str(lane_dir), str(serialized_driver))
            prompt = _lane_prompt(session.prompt, serialized_driver=serialized_driver)
            await call(str(lane_dir / kernel_relative), prompt, session_sink=sink)
            end_reason = sink.get("end_reason", "")
            if end_reason == EXHAUSTED_END_REASON:
                # The profile's line stopped answering: the next round's boundary checks it before it gets slots again.
                self._check_profile_again(session.agent.profile_id)
            return await self._request_lesson(iteration, end_reason, sink.get("summarize"))

        results = await run_lanes(
            workspace_dir=self.ic.workspace_dir,
            lanes=[session.lane for session in sessions],
            session=_session,
            parent_dir=str(Path(self.ic.workspace_dir).resolve().parent),
            driver=self._workspace_path(self.ic.driver_script),
        )
        return {item.lane_id: item for item in results}

    async def _measure_slots(
        self, tree: SearchTree, iteration: int, plans: Sequence[SlotPlan], lanes: dict[int, LaneResult]
    ) -> list[SlotOutcome]:
        """Measure every slot's candidate in turn, on the one device, at its own start."""
        outcomes = []
        for plan in plans:
            lane = lanes.get(plan.slot.index)
            if lane is not None:
                lesson = lane.lesson.text or f"(no lesson: {lane.lesson.reason})"
                self._write_slot_record(iteration, plan.slot, "lesson.md", lesson)
            outcomes.append(await self._measure_slot(tree, iteration, plan, lane))
        return outcomes

    async def _measure_slot(
        self, tree: SearchTree, iteration: int, plan: SlotPlan, lane: LaneResult | None
    ) -> SlotOutcome:
        slot = plan.slot
        refusal = self._candidate_refusal(lane)
        if refusal:
            outcome = AttemptOutcome.NO_CHANGES if lane is not None and not lane.error else AttemptOutcome.ERROR
            return self._unmeasured(iteration, slot, outcome, refusal)
        self._write_slot_record(iteration, slot, "change.diff", lane.diff)
        worktree = self._start_worktrees.ensure(slot.start)
        if not self._start_worktrees.apply(worktree, lane.diff):
            return self._unmeasured(iteration, slot, AttemptOutcome.ERROR, "the lane's diff did not apply to its start")
        staged = StagedCandidate(worktree, slot.start, plan.text, self._slot_dir(iteration, slot))
        measured = await self._measure_and_admit(
            tree, iteration, staged, lambda result: self._admit(tree, iteration, plan, result, worktree)
        )
        if measured.holder:
            return self._unmeasured(
                iteration, slot, AttemptOutcome.NO_CHANGES, f"the candidate is node {measured.holder}"
            )
        node = measured.node
        outcome = AttemptOutcome.INCORRECT if node is None else AttemptOutcome.NODE
        attempt = self._attempt(iteration, slot, outcome, measured.result, "" if node is None else node.commit)
        return SlotOutcome(attempt, node, measured.result)

    async def _measure_and_admit(
        self, tree: SearchTree, iteration: int, staged: StagedCandidate, admit: Callable[[IterationResult], TreeNode]
    ) -> MeasuredCandidate:
        """Measure a staged candidate and ``admit`` it when it is correct; its worktree ends back on its start.

        A candidate some node already holds is not measured: that node is named instead.
        """
        try:
            holder = self._node_holding(tree, staged.worktree)
            if holder:
                return MeasuredCandidate(holder=holder)
            result = await self._measure_at(iteration, staged.plan, staged.worktree)
            self._write_record(staged.records, "measurement.json", _measurement_record(result))
            if not measured_correct(result):
                return MeasuredCandidate(result=result)
            return MeasuredCandidate(result=result, node=admit(result))
        finally:
            self._start_worktrees.restore(staged.worktree, staged.start)

    def _node_holding(self, tree: SearchTree, worktree: Path) -> str:
        """The node whose files a worktree's staged candidate already is, or an empty string."""
        return node_holding_tree(self.ic.workspace_dir, tree, self._start_worktrees.staged_tree(worktree))

    def _candidate_refusal(self, lane: LaneResult | None) -> str:
        if lane is None:
            return "the lane never ran"
        if lane.error:
            return f"the lane was lost: {lane.error}"
        if not lane.diff.strip():
            return "the lane changed nothing"
        return self._lane_rejection(lane.diff)

    async def _measure_at(self, iteration: int, plan: str, worktree: Path) -> IterationResult:
        """The ordinary measurement of what is staged in ``worktree``, against the published best's KEEP bar."""
        try:
            return await self.run_one_iteration(iteration, plan, site=self._canonical_site().rebased(str(worktree)))
        except (OSError, RuntimeError, ValueError) as error:
            return IterationResult(
                iteration=iteration,
                duration_sec=0.0,
                validation_passed=False,
                validation_summary=f"measurement crashed: {type(error).__name__}: {error}",
                crashed=True,
            )

    def _admit(
        self, tree: SearchTree, iteration: int, plan: SlotPlan, result: IterationResult, worktree: Path
    ) -> TreeNode:
        """Commit a correct candidate in its worktree and add it to the tree on its direction."""
        slot = plan.slot
        direction_id = self._direction_for(tree, iteration, plan)
        commit = self._start_worktrees.commit(worktree, _commit_message(iteration, slot, direction_id, plan, result))
        pin_node(self.ic.workspace_dir, commit)
        provenance = (slot.source,) if slot.slot_type is SlotType.PORT else ()
        node = replace(
            _node_of(result, _Placement(commit, slot.start, direction_id, provenance), slot=slot, plan=plan.title),
            approach=plan.approach,
            focus_cases=plan.focus_cases,
            agent_profile_id=slot.agent_profile_id,
        )
        self._add_node(tree, node)
        return node

    def _admit_merge(
        self, tree: SearchTree, staged: StagedCandidate, result: IterationResult, source: TreeNode
    ) -> TreeNode:
        """Commit a correct merge in its base's worktree and add it under the base, on the base's direction."""
        base = tree.require_node(staged.start)
        message = f"iter-{result.iteration}-merge: {source.commit[:12]} onto {base.commit[:12]}"
        commit = self._start_worktrees.commit(staged.worktree, message)
        pin_node(self.ic.workspace_dir, commit)
        placement = _Placement(commit, base.commit, base.direction, (source.commit,))
        node = replace(_node_of(result, placement, slot=None, plan=staged.plan), agent_profile_id=staged.producer)
        self._add_node(tree, node)
        return node

    def _add_node(self, tree: SearchTree, node: TreeNode) -> None:
        """Add a committed node, move its direction's branch to the direction's best, and save the tree."""
        tree.add_node(node)
        self._point_direction(tree, node.direction)
        # Saved now, not when the round closes: a session killed later in the round would otherwise leave a pinned
        # commit no record names.
        self._tree_store.save(tree)

    def _direction_for(self, tree: SearchTree, iteration: int, plan: SlotPlan) -> str:
        slot = plan.slot
        if slot.slot_type is SlotType.DEEPEN:
            return slot.direction
        if slot.slot_type is SlotType.PORT:
            return tree.require_node(slot.start).direction
        proposal = plan.proposal or DirectionProposal(
            name=f"route opened in iteration {iteration}, slot {slot.index}",
            hypothesis=_summary(plan.text),
            expected_ceiling="not stated by the plan",
            failure_criterion="not stated by the plan",
        )
        direction = tree.allocate_direction(proposal, origin=slot.start, iteration=iteration)
        if slot.hint and "refut" in plan.hint_answer.lower():
            tree.record_hint_rebuttal(direction.direction_id, f"{slot.hint}: {plan.hint_answer}")
        return direction.direction_id

    def _point_direction(self, tree: SearchTree, direction_id: str) -> None:
        best = tree.direction_best(direction_id)
        direction = tree.direction(direction_id)
        if best is not None and direction is not None:
            point_direction_branch(self.ic.workspace_dir, direction.branch, best.commit)

    async def _remeasure(self, tree: SearchTree, iteration: int, commits: Sequence[str]) -> None:
        """Fold another measurement of each named node into its samples."""
        for commit in commits:
            worktree = self._start_worktrees.ensure(commit)
            result = await self._measure_at(iteration, f"re-measure {commit}", worktree)
            if measured_correct(result):
                tree.add_measurement(commit, node_measurement(result))
                self._tree_store.save(tree)

    def _attempt(
        self, iteration: int, slot: Slot, outcome: AttemptOutcome, result: IterationResult | None, node: str
    ) -> Attempt:
        return Attempt(
            iteration=iteration,
            slot=slot.index,
            slot_type=slot.slot_type,
            start=slot.start,
            direction=slot.direction,
            outcome=outcome,
            score=None if result is None else result.mean_case_speedup,
            node=node,
            archive_path=str(self._slot_dir(iteration, slot)),
            reason=slot.reason,
            agent_profile_id=slot.agent_profile_id,
        )

    def _unmeasured(self, iteration: int, slot: Slot, outcome: AttemptOutcome, reason: str) -> SlotOutcome:
        print(f"  [tree] slot {slot.index} ({slot.slot_type.value} at {slot.start[:12]}): {reason}")
        self._write_slot_record(iteration, slot, "outcome.txt", reason)
        return SlotOutcome(replace(self._attempt(iteration, slot, outcome, None, ""), detail=reason), None, None)

    def _slot_dir(self, iteration: int, slot: Slot) -> Path:
        return self._experiments.tree_round(iteration) / f"slot_{slot.index:02d}"

    def _write_slot_record(self, iteration: int, slot: Slot, name: str, content) -> None:
        self._write_record(self._slot_dir(iteration, slot), name, content)

    def _write_record(self, directory: Path, name: str, content) -> None:
        text = content if isinstance(content, str) else json.dumps(content, indent=2, sort_keys=True, default=str)
        atomic_write_text(directory / name, text, options=DURABLE)


@dataclass(frozen=True)
class _Placement:
    """Where a new node sits: its commit, the node it grew from, its direction, and the nodes it carries changes of."""

    commit: str
    parent: str
    direction: str
    provenance: tuple[str, ...] = ()


def _node_of(result: IterationResult, placement: _Placement, *, slot: Slot | None, plan: str) -> TreeNode:
    return TreeNode(
        commit=placement.commit,
        parent=placement.parent,
        direction=placement.direction,
        slot_type=None if slot is None else slot.slot_type,
        iteration=result.iteration,
        slot=0 if slot is None else slot.index,
        samples=(),
        wall_ms=result.wall_ms,
        case_times={},
        case_spreads={},
        snr_db=result.snr_db,
        plan=_summary(plan),
        source=_source_of(slot),
        provenance=placement.provenance,
    ).with_measurement(node_measurement(result))


def _source_of(slot: Slot | None) -> NodeSource:
    if slot is None:
        return NodeSource.COMPOSE
    return NodeSource.PORT if slot.slot_type is SlotType.PORT else NodeSource.ITERATION


def _commit_message(iteration: int, slot: Slot, direction: str, plan: SlotPlan, result: IterationResult) -> str:
    case_times = (result.bench_detail or {}).get("case_times") or {}
    return "\n".join(
        [
            f"iter-{iteration}-s{slot.index}: {_summary(plan.title)[:72]}",
            "",
            f"plan: {_summary(plan.text)}",
            f"start: {slot.start}",
            f"direction: {direction}",
            f"slot type: {slot.slot_type.value}",
            f"mean case speedup: {result.mean_case_speedup:.6f}",
            f"case ms: {json.dumps(case_times, sort_keys=True)}",
            f"snr_db: {result.snr_db}",
        ]
    )


def _measurement_record(result: IterationResult) -> dict:
    return {
        "validation_passed": result.validation_passed,
        "validation_summary": result.validation_summary,
        "mean_case_speedup": result.mean_case_speedup,
        "kept": result.kept,
        "wall_ms": result.wall_ms,
        "snr_db": result.snr_db,
        "bench": result.bench_detail,
    }


__all__ = [
    "LaneSession",
    "MeasuredCandidate",
    "SessionAgent",
    "SlotOutcome",
    "StagedCandidate",
    "TreeAdmissionMixin",
    "measured_correct",
    "node_measurement",
    "slot_prompt",
]
