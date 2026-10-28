# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-loop with the tree strategy: each iteration expands several nodes of a search tree at once.

Everything outside one iteration's search -- stop conditions, round pricing, Analysis, publication of a best, resume --
is the iteration loop's own. A round ends in the loop's ordinary settle path: the best node the round found that
cleared the published best's KEEP bar is staged in the campaign workspace and measured there once more before it is
committed on the campaign branch, so the campaign branch is the chain of published bests and each of its commits holds
exactly the tree of one node.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from kernelforge.loop.iteration_result import IterationResult
from kernelforge.loop.merge_runtime import StagedMerge
from kernelforge.loop.runner import (
    IterationLoop,
    IterationTurn,
    LoopAgents,
    LoopCallbacks,
    _Attempt,
    _IterationFrame,
    _lane_prompt,
    _RoundPlan,
)
from kernelforge.loop.tree.admission import SlotOutcome, TreeAdmissionMixin, measured_correct
from kernelforge.loop.tree.facts import FactSources, RoundOutlook, render_facts
from kernelforge.loop.tree.model import (
    Attempt,
    AttemptOutcome,
    NodeSource,
    Round,
    SearchTree,
    TreeNode,
)
from kernelforge.loop.tree.planning import RoundPlans, TreePlanningMixin
from kernelforge.loop.tree.refs import node_holding_tree, pin_node, reconcile_node_refs
from kernelforge.loop.tree.selection import Selection, read_selection
from kernelforge.loop.tree.store import SearchTreeStore
from kernelforge.loop.tree.worktrees import StartWorktrees, untracked_outside
from kernelforge.orchestrator.slot_selector import SelectorAnswer
from kernelforge.tools.durable_io import atomic_write_text
from kernelforge.tools.git import git

EXPERIMENTS_DIR = "forge_experiments"
SELECTIONS_DIR = "tree_selection"


class TreeSearchLoop(TreePlanningMixin, TreeAdmissionMixin, IterationLoop):
    """The iteration loop whose search step expands a tree of measured nodes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start_worktrees = StartWorktrees(self.ic.workspace_dir)
        self._tree_store = SearchTreeStore(Path(self.ic.workspace_dir) / EXPERIMENTS_DIR)
        self._tree: SearchTree | None = None
        self._unstacked: tuple[tuple[str, str], ...] = ()

    async def _begin_locked_run(self, agents: LoopAgents, *, callbacks: LoopCallbacks, usage=None) -> LoopAgents:
        actors = await super()._begin_locked_run(agents, callbacks=callbacks, usage=usage)
        self._begin_search_tree()
        return actors

    async def _run_supervisor_review(self, *_args, **_kwargs) -> None:
        """A tree campaign has no Supervisor: the selector owns every change of direction."""

    def _begin_search_tree(self) -> None:
        """Root a fresh campaign's tree at the published best, or reconcile a resumed one with the repository."""
        untracked = untracked_outside(self.ic.workspace_dir, [EXPERIMENTS_DIR])
        if untracked:
            raise ValueError(
                "a tree campaign measures candidates in worktrees, which would not see these untracked sources: "
                + ", ".join(untracked[:10])
            )
        self._start_worktrees.clear()
        if self._tree_store.exists():
            self._tree = self._resumed_tree()
        else:
            self._tree = SearchTree.create(self._root_node(), root_branch=self.ic.git_branch)
            pin_node(self.ic.workspace_dir, self._tree.root.commit)
        self._tree_store.save(self._tree)
        print(f"  [tree] {len(self._tree.nodes)} node(s), published best {self._tree.published_best[:12]}")

    def _rev_parse(self, revision: str) -> str:
        return git("rev-parse", revision, cwd=self.ic.workspace_dir).stdout.strip()

    def _root_node(self) -> TreeNode:
        head = self._rev_parse("HEAD")
        score = self.best_mean_case_speedup or 1.0
        return TreeNode(
            commit=head,
            parent="",
            direction="",
            slot_type=None,
            iteration=0,
            slot=0,
            samples=(score,),
            wall_ms=self.best_wall_ms,
            case_times=dict(self._best_case_times or self._baseline_case_times),
            case_spreads={},
            snr_db=None,
            plan=self.run_state.best.plan,
            source=NodeSource.WARM_START if self.ic.warm_start_commit else NodeSource.BASELINE,
        )

    def _resumed_tree(self) -> SearchTree:
        tree = self._tree_store.load()
        repair = reconcile_node_refs(self.ic.workspace_dir, tree)
        if repair.lost:
            raise ValueError(f"search tree nodes whose commits no longer exist: {', '.join(repair.lost)}")
        if tree.round is not None:
            reason = "the session ended before this round finished"
            tree.close_round([_lost_attempt(tree.round.iteration, slot, reason) for slot in tree.round.slots])
        head = self._rev_parse("HEAD")
        if head != tree.published_head:
            # The campaign branch moved past what the tree recorded: a publication finished and the tree was not saved.
            tree.publish(self._node_with_tree_of(tree, head), head=head)
        return tree

    def _node_with_tree_of(self, tree: SearchTree, commit: str) -> str:
        holder = node_holding_tree(self.ic.workspace_dir, tree, self._rev_parse(f"{commit}^{{tree}}"))
        if not holder:
            raise ValueError(f"the campaign branch is at {commit}, whose tree no node of the search tree holds")
        return holder

    async def _search_iteration(self, turn: IterationTurn, *, agents: LoopAgents, callbacks: LoopCallbacks) -> bool:
        if turn.hazard is not None:
            # A device this campaign may not measure on refuses the round the same way it refuses a linear one.
            return await super()._search_iteration(turn, agents=agents, callbacks=callbacks)
        tree = self._tree
        selection = await self._select(turn)
        round_ = Round(turn.iteration, selection.slots, selection.remeasure, selection.refresh_analysis)
        for change in selection.status_changes:
            tree.set_direction_status(change.direction_id, change.status, change.reason)
        tree.open_round(round_)
        self._tree_store.save(tree)
        plans = await self._plan_round_slots(tree, round_, turn.actors.orchestration_factory)
        if not self._admit_dispatch(turn.iteration):
            self._close_tree_round(tree, [], plans, "the round could not pay for its sessions")
            return False
        outcomes = await self._expand(tree, turn, plans)
        composition = await self._compose_onto_published(tree, turn.iteration, [o.node for o in outcomes if o.node])
        await self._remeasure(tree, turn.iteration, round_.remeasure)
        self._unstacked = composition.unstacked
        self._close_tree_round(tree, outcomes, plans, "")
        candidates = [(o.node, o.result) for o in outcomes if o.node is not None] + list(composition.nodes)
        await self._settle_round(turn, candidates, outcomes, callbacks)
        self._start_worktrees.release([tree.published_best])
        return True

    async def _expand(self, tree: SearchTree, turn: IterationTurn, plans: RoundPlans) -> list[SlotOutcome]:
        """Run the round's lanes and measure what they wrote; nothing is measured on a device a lane left held."""
        if not plans.plans:
            return []
        lanes = await self._run_slot_lanes(tree, plans.plans, turn.actors.agent_factory, _lane_prompt)
        contended = [lane for lane, _ in lanes.values() if lane.contended]
        if contended:
            self.device_hazard.record(
                iteration=turn.iteration,
                detail="; ".join(f"lane {lane.lane_id}: {lane.reaped.describe()}" for lane in contended),
                pids={pid for lane in contended for pid in lane.reaped.blockers},
            )
            return []
        return await self._measure_slots(tree, turn.iteration, plans.plans, lanes)

    async def _select(self, turn: IterationTurn) -> Selection:
        """Ask the selector for this round's slots and keep the legal part of its answer."""
        tree = self._tree
        facts = render_facts(
            tree,
            outlook=RoundOutlook(turn.round_lanes, self._time_remaining(), self._rounds_left()),
            sources=FactSources(
                archive_dir=str(self._tree_round_dir(0).parent),
                lessons_dir=str(Path(self.ic.workspace_dir) / EXPERIMENTS_DIR / "lessons"),
                selections_dir=str(self._selections_dir()),
            ),
            compose_failures=self._unstacked,
        )
        started_at = time.time()
        selector = turn.actors.slot_selector
        if selector is None:
            answer = SelectorAnswer(failure="no slot selector is configured")
        else:
            answer = await selector.select(
                workspace=self.ic.workspace_dir, facts=facts, lanes=turn.round_lanes, usage=self._usage
            )
            self._checkpoint_llm_usage()
        self._round_planning_sec += max(0.0, time.time() - started_at)
        selection = read_selection(tree, answer.payload, lanes=turn.round_lanes, failure=answer.failure)
        self._record_selection(turn.iteration, facts, answer, selection)
        return selection

    def _rounds_left(self) -> int | None:
        recent = [item.total_sec for item in self.run_state.round_costs.recent if item.total_sec > 0]
        if not recent:
            return None
        return int(self._time_remaining() // statistics.median(recent))

    def _selections_dir(self) -> Path:
        return Path(self.ic.workspace_dir) / EXPERIMENTS_DIR / SELECTIONS_DIR

    def _record_selection(self, iteration: int, facts: str, answer: SelectorAnswer, selection: Selection) -> None:
        slots = "\n".join(
            f"- slot {slot.index}: {slot.slot_type.value} at {slot.start} {slot.direction or slot.source or ''} "
            f"-- {slot.reason}"
            for slot in selection.slots
        )
        record = "\n".join(
            [
                f"# Slot selection, iteration {iteration}",
                "",
                f"Fallback: {selection.fallback or 'none'}",
                "",
                "## Slots",
                slots,
                "",
                "## Dropped",
                "\n".join(f"- {item}" for item in selection.dropped) or "- none",
                "",
                "## Facts given to the selector",
                facts,
                "",
                "## Selector answer",
                answer.text or f"(no answer: {answer.failure})",
                "",
            ]
        )
        atomic_write_text(self._selections_dir() / f"iter_{iteration:03d}.md", record)
        for slot in selection.slots:
            print(f"  [tree] slot {slot.index}: {slot.slot_type.value} at {slot.start[:12]} {slot.direction}")
        if selection.fallback:
            print(f"  [tree] selector fallback: {selection.fallback}")

    def _close_tree_round(self, tree: SearchTree, outcomes: list[SlotOutcome], plans: RoundPlans, reason: str):
        """Record every slot's attempt, including the slots that never ran, and clear the round."""
        attempts = [outcome.attempt for outcome in outcomes]
        measured = {attempt.slot for attempt in attempts}
        for item in plans.unplanned:
            attempts.append(_lost_attempt(tree.round.iteration, item.slot, item.reason))
        for plan in plans.plans:
            if plan.slot.index not in measured:
                attempts.append(_lost_attempt(tree.round.iteration, plan.slot, reason or "the slot was not measured"))
        tree.close_round(sorted(attempts, key=lambda item: item.slot))
        self._tree_store.save(tree)

    async def _settle_round(self, turn: IterationTurn, candidates: list, outcomes: list[SlotOutcome], callbacks):
        """Close the iteration: publish the round's best candidate that cleared the bar, or record why none did."""
        clearing = [(node, result) for node, result in candidates if result.kept]
        frame = _IterationFrame(turn.iteration, {"plan": _round_summary(outcomes)}, StagedMerge(None, ""))
        # The relay switches profile only between iterations, so every slot of this round ran on the active one.
        self._bind_session_producer(frame.session_sink, self.run_state.agent_relay.active_profile_id)
        if clearing:
            node, _ = max(clearing, key=lambda item: item[0].score)
            await self._publish_node(node, frame, callbacks)
            return
        measured = [o.result for o in outcomes if o.result is not None]
        if not measured:
            summary = IterationResult(turn.iteration, 0.0, False, "no slot produced a candidate to measure")
            self._stamp_result_profile(summary, frame.session_sink)
            await self._abandon_iteration(
                summary,
                frame,
                decision_label="NO_CHANGES",
                on_iteration=callbacks.on_iteration,
                round_plan=_RoundPlan(),
            )
            return
        best = max(measured, key=lambda item: (measured_correct(item), item.mean_case_speedup or 0.0))
        attempt = _Attempt(best, "", "", "", "", None, "", True)
        await self._settle_locked_attempt(attempt, frame, callbacks=callbacks)

    async def _publish_node(self, node: TreeNode, frame: _IterationFrame, callbacks) -> None:
        """Stage ``node``'s tree in the campaign workspace and settle it as this iteration's candidate."""
        git("read-tree", "--reset", "-u", node.commit, cwd=self.ic.workspace_dir)
        frame.session_sink["plan"] = node.plan
        attempt = _Attempt(None, "", node.plan, self._read_kernel_source(), self._working_tree_diff(), None, "", True)
        await self._settle_locked_attempt(attempt, frame, callbacks=callbacks)
        if self.results and self.results[-1].kept:
            self._tree.publish(node.commit, head=self.run_state.best.commit_hash)
            self._tree_store.save(self._tree)
            print(f"  [tree] published node {node.commit[:12]} as the new best")


def _lost_attempt(iteration: int, slot, reason: str) -> Attempt:
    return Attempt(
        iteration=iteration,
        slot=slot.index,
        slot_type=slot.slot_type,
        start=slot.start,
        direction=slot.direction,
        outcome=AttemptOutcome.ERROR,
        score=None,
        node="",
        archive_path="",
        reason=f"{slot.reason} [{reason}]",
    )


def _round_summary(outcomes: list[SlotOutcome]) -> str:
    parts = [
        f"slot {o.attempt.slot} {o.attempt.slot_type.value} at {o.attempt.start[:12]}: {o.attempt.outcome.value}"
        for o in outcomes
    ]
    return "tree round: " + ("; ".join(parts) or "no slot was measured")


__all__ = ["TreeSearchLoop"]
