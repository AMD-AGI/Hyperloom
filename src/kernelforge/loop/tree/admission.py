# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a tree round's lanes, measure every candidate where it was written, and admit the correct ones as nodes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.loop.fanout import LanePlan, LaneResult, run_lanes
from kernelforge.loop.iteration_result import IterationResult
from kernelforge.loop.merge_candidates import case_spreads, cases_beating_reference
from kernelforge.loop.tree.model import (
    Attempt,
    AttemptOutcome,
    DirectionProposal,
    NodeSource,
    SearchTree,
    Slot,
    SlotType,
    TreeNode,
)
from kernelforge.loop.tree.planning import SlotPlan
from kernelforge.loop.tree.refs import is_ancestor, node_holding_tree, pin_node, point_direction_branch
from kernelforge.tools.durable_io import atomic_write_text
from kernelforge.tools.git import git

_PLAN_SUMMARY_CHARS = 240


@dataclass(frozen=True)
class SlotOutcome:
    """What one slot came to: its attempt record, the node it added, and the measurement behind both."""

    attempt: Attempt
    node: TreeNode | None
    result: IterationResult | None


@dataclass(frozen=True)
class Composition:
    """The nodes mechanical stacking added, with their measurements, and the pairs whose diff did not apply."""

    nodes: tuple[tuple[TreeNode, IterationResult], ...]
    unstacked: tuple[tuple[str, str], ...]


def measured_correct(result: IterationResult | None) -> bool:
    """Whether a measurement makes its candidate a node: correct, and scored on every case three times."""
    if result is None or not result.validation_passed or result.mean_case_speedup is None:
        return False
    detail = result.bench_detail or {}
    return bool(detail.get("success")) and bool(detail.get("measurement_mean_case_speedups"))


def _summary(text: str) -> str:
    return " ".join(str(text or "").split())[:_PLAN_SUMMARY_CHARS]


def slot_prompt(plan: SlotPlan, start: TreeNode, lane_prompt) -> str:
    """One slot's Implementer prompt: where it starts, what counts, then its plan."""
    preamble = (
        f"This session starts from search-tree node {start.commit} at {start.score:.6f}x (equal-weight mean "
        "pristine-relative case speedup). Aim to exceed it. The candidate must be correct; a correct but slower first "
        "step of the planned route is acceptable, is measured, and is kept in the search tree.\n\n"
    )
    return lane_prompt(preamble + plan.text)


class TreeAdmissionMixin:
    """Lanes, measurement and node admission for one tree round.

    The host supplies ``ic``, ``run_one_iteration``, ``_canonical_site``, ``_lane_rejection``, ``_workspace_path``,
    ``_start_worktrees``, ``_tree_round_dir`` and ``device_hazard``.
    """

    async def _run_slot_lanes(self, tree: SearchTree, plans: Sequence[SlotPlan], agent_factory, lane_prompt) -> dict:
        """Run one lane per planned slot, each copied from its start's worktree; their results by slot index."""
        kernel_relative = self._workspace_path(self.ic.kernel_file)
        by_lane = {str(plan.slot.index): plan for plan in plans}
        transcripts: dict[str, str] = {}

        async def _session(lane: LanePlan, lane_dir: Path, serialized_driver: Path) -> None:
            plan = by_lane[lane.lane_id]
            prompt = slot_prompt(
                plan,
                tree.require_node(plan.slot.start),
                lambda text: lane_prompt(text, serialized_driver=serialized_driver),
            )
            answer = await agent_factory(str(lane_dir), str(serialized_driver))(str(lane_dir / kernel_relative), prompt)
            transcripts[lane.lane_id] = str(answer or "")

        results = await run_lanes(
            workspace_dir=self.ic.workspace_dir,
            lanes=[
                LanePlan(
                    lane_id=str(plan.slot.index),
                    plan=plan.text,
                    source_dir=str(self._start_worktrees.ensure(plan.slot.start)),
                )
                for plan in plans
            ],
            session=_session,
            parent_dir=str(Path(self.ic.workspace_dir).resolve().parent),
            driver=self._workspace_path(self.ic.driver_script),
        )
        return {int(item.lane_id): (item, transcripts.get(item.lane_id, "")) for item in results}

    async def _measure_slots(
        self, tree: SearchTree, iteration: int, plans: Sequence[SlotPlan], lanes: dict
    ) -> list[SlotOutcome]:
        """Measure every slot's candidate in turn, on the one device, at its own start."""
        outcomes = []
        for plan in plans:
            lane, transcript = lanes.get(plan.slot.index, (None, ""))
            self._write_slot_record(iteration, plan.slot, "session.md", transcript)
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
        holder = self._node_holding(tree, worktree)
        if holder:
            self._start_worktrees.restore(worktree, slot.start)
            return self._unmeasured(iteration, slot, AttemptOutcome.NO_CHANGES, f"the candidate is node {holder}")
        result = await self._measure_at(iteration, plan.text, worktree)
        self._write_slot_record(iteration, slot, "measurement.json", _measurement_record(result))
        if not measured_correct(result):
            self._start_worktrees.restore(worktree, slot.start)
            return SlotOutcome(self._attempt(iteration, slot, AttemptOutcome.INCORRECT, result, ""), None, result)
        node = self._admit(tree, iteration, plan, result, worktree)
        self._start_worktrees.restore(worktree, slot.start)
        return SlotOutcome(self._attempt(iteration, slot, AttemptOutcome.NODE, result, node.commit), node, result)

    def _node_holding(self, tree: SearchTree, worktree: Path) -> str:
        """The node whose files a worktree's staged candidate already is, or an empty string."""
        return node_holding_tree(self.ic.workspace_dir, tree, self._start_worktrees.staged_tree(worktree))

    def _candidate_refusal(self, lane: LaneResult | None) -> str:
        if lane is None:
            return "the slot's lane never ran"
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
        node = _node_of(result, _Placement(commit, slot.start, direction_id, provenance), slot=slot, plan=plan.title)
        tree.add_node(node)
        self._point_direction(tree, direction_id)
        return node

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

    async def _compose_onto_published(
        self, tree: SearchTree, iteration: int, new_nodes: Sequence[TreeNode]
    ) -> Composition:
        """Stack each new node that wins cases the published best loses onto the published best.

        A node grown from the published best already holds everything the published best has; stacking its change
        back onto the published best would only reproduce it.
        """
        published = tree.require_node(tree.published_best)
        composed: list[tuple[TreeNode, IterationResult]] = []
        unstacked: list[tuple[str, str]] = []
        if published.is_root:
            return Composition((), ())
        for node in new_nodes:
            if is_ancestor(self.ic.workspace_dir, published.commit, node.commit):
                continue
            if not _wins_both_ways(node, published):
                continue
            outcome = await self._compose(tree, iteration, node, published)
            if outcome is None:
                unstacked.append((node.commit, published.commit))
            elif outcome[0] is not None:
                composed.append(outcome)
        return Composition(tuple(composed), tuple(unstacked))

    async def _compose(self, tree: SearchTree, iteration: int, source: TreeNode, target: TreeNode):
        ancestor = git("merge-base", source.commit, target.commit, cwd=self.ic.workspace_dir).stdout.strip()
        patch = git("diff", ancestor, source.commit, cwd=self.ic.workspace_dir).stdout
        worktree = self._start_worktrees.ensure(target.commit)
        if not self._start_worktrees.apply(worktree, patch):
            return None
        holder = self._node_holding(tree, worktree)
        if holder:
            print(f"  [tree] stacking {source.commit[:12]} onto {target.commit[:12]} reproduces node {holder[:12]}")
            self._start_worktrees.restore(worktree, target.commit)
            return (None, None)
        plan = f"compose: the change of {source.commit} since {ancestor} onto {target.commit}"
        result = await self._measure_at(iteration, plan, worktree)
        if not measured_correct(result):
            self._start_worktrees.restore(worktree, target.commit)
            return (None, result)
        message = f"iter-{iteration}-compose: {source.commit[:12]} onto {target.commit[:12]}"
        commit = self._start_worktrees.commit(worktree, message)
        pin_node(self.ic.workspace_dir, commit)
        placement = _Placement(commit, target.commit, target.direction, (source.commit,))
        node = _node_of(result, placement, slot=None, plan=plan)
        tree.add_node(node)
        self._point_direction(tree, target.direction)
        self._start_worktrees.restore(worktree, target.commit)
        return (node, result)

    async def _remeasure(self, tree: SearchTree, iteration: int, commits: Sequence[str]) -> None:
        """Fold another measurement of each named node into its samples."""
        for commit in commits:
            worktree = self._start_worktrees.ensure(commit)
            result = await self._measure_at(iteration, f"re-measure {commit}", worktree)
            if measured_correct(result):
                tree.add_samples(commit, tuple(result.bench_detail["measurement_mean_case_speedups"]))

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
        )

    def _unmeasured(self, iteration: int, slot: Slot, outcome: AttemptOutcome, reason: str) -> SlotOutcome:
        print(f"  [tree] slot {slot.index} ({slot.slot_type.value} at {slot.start[:12]}): {reason}")
        self._write_slot_record(iteration, slot, "outcome.txt", reason)
        return SlotOutcome(replace(self._attempt(iteration, slot, outcome, None, ""), detail=reason), None, None)

    def _slot_dir(self, iteration: int, slot: Slot) -> Path:
        return self._tree_round_dir(iteration) / f"slot_{slot.index:02d}"

    def _write_slot_record(self, iteration: int, slot: Slot, name: str, content) -> None:
        text = content if isinstance(content, str) else json.dumps(content, indent=2, sort_keys=True, default=str)
        atomic_write_text(self._slot_dir(iteration, slot) / name, text)


def _wins_both_ways(node: TreeNode, published: TreeNode) -> bool:
    """Whether each of two nodes runs some case faster than the other beyond its own noise."""
    wins = cases_beating_reference(node.case_times, published.case_times, node.case_spreads)
    losses = cases_beating_reference(published.case_times, node.case_times, published.case_spreads)
    return bool(wins) and bool(losses)


@dataclass(frozen=True)
class _Placement:
    """Where a new node sits: its commit, the node it grew from, its direction, and the nodes it carries changes of."""

    commit: str
    parent: str
    direction: str
    provenance: tuple[str, ...] = ()


def _node_of(result: IterationResult, placement: _Placement, *, slot: Slot | None, plan: str) -> TreeNode:
    detail = result.bench_detail or {}
    return TreeNode(
        commit=placement.commit,
        parent=placement.parent,
        direction=placement.direction,
        slot_type=None if slot is None else slot.slot_type,
        iteration=result.iteration,
        slot=0 if slot is None else slot.index,
        samples=tuple(float(item) for item in detail["measurement_mean_case_speedups"]),
        wall_ms=result.wall_ms,
        case_times={str(key): float(value) for key, value in dict(detail.get("case_times") or {}).items()},
        case_spreads=case_spreads(detail.get("measurements")),
        snr_db=result.snr_db,
        plan=_summary(plan),
        source=_source_of(slot),
        provenance=placement.provenance,
    )


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


__all__ = ["Composition", "SlotOutcome", "TreeAdmissionMixin", "measured_correct", "slot_prompt"]
