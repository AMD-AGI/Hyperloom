# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-loop with the tree strategy: each iteration expands several nodes of a search tree at once.

Everything outside one iteration's search -- stop conditions, round pricing, Analysis, publication of a best, resume --
is the iteration loop's own. A round's new nodes are merged before it settles (see ``tree.merge``), and the round ends
in the loop's ordinary settle path: the best node the round found that cleared the published best's KEEP bar, a merge
included, is staged in the campaign workspace and measured there once more before it is committed on the campaign
branch, so the campaign branch is the chain of published bests and each of its commits holds exactly the tree of one
node.
"""

from __future__ import annotations

import statistics
import time

from hyperloom_common.io import DURABLE, atomic_write_text
from kernelforge.utils.git import git
from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.iteration.iteration_types import (
    IterationResult,
    IterationTurn,
    LoopAgents,
    LoopCallbacks,
    _Attempt,
    _IterationFrame,
    _RoundPlan,
    _StagedMerge,
)
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop
from kernelforge.workflows.forge_loop.roles.slot_selector import SelectorAnswer
from kernelforge.workflows.forge_loop.tree.admission import (
    SessionAgent,
    SlotOutcome,
    TreeAdmissionMixin,
    measured_correct,
    node_measurement,
)
from kernelforge.workflows.forge_loop.tree.facts import FactSources, RoundOutlook, render_facts
from kernelforge.workflows.forge_loop.tree.merge import TreeMergeMixin
from kernelforge.workflows.forge_loop.tree.model import (
    MergeFailure,
    NodeSource,
    Round,
    SearchTree,
    TreeNode,
    lost_attempt,
)
from kernelforge.workflows.forge_loop.tree.planning import RoundPlans, TreePlanningMixin
from kernelforge.workflows.forge_loop.tree.profiles import SlotActors, profile_at, round_profile_cycle
from kernelforge.workflows.forge_loop.tree.refs import node_holding_tree, pin_node, reconcile_node_refs
from kernelforge.workflows.forge_loop.tree.selection import Selection, read_selection
from kernelforge.workflows.forge_loop.tree.store import SearchTreeStore
from kernelforge.workflows.forge_loop.tree.worktrees import StartWorktrees, untracked_outside


class TreeSearchLoop(TreePlanningMixin, TreeMergeMixin, TreeAdmissionMixin, IterationLoop):
    """The iteration loop whose search step expands a tree of measured nodes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._experiments = ExperimentsLayout.of(self.ic.workspace_dir)
        self._start_worktrees = StartWorktrees(self.ic.workspace_dir)
        self._tree_store = SearchTreeStore(self._experiments.search_tree)
        self._tree: SearchTree | None = None
        # The last round's merges that did not land, which the next round's selector may carry over by PORT.
        self._merge_failures: tuple[MergeFailure, ...] = ()

    async def _begin_locked_run(self, agents: LoopAgents, *, callbacks: LoopCallbacks, usage=None) -> LoopAgents:
        actors = await super()._begin_locked_run(agents, callbacks=callbacks, usage=usage)
        await self._begin_search_tree()
        if self.ic.split_relay and len(self._split_profiles()) < 2:
            print(
                "  [tree] relay mode split: fewer than two Agent profiles can run this session, so every slot runs on "
                "the active one until another passes its check"
            )
        return actors

    def _switches_profile_at_blockage(self) -> bool:
        """A split relay runs every profile each round, so a blockage has no other profile to hand the search to."""
        return not self.ic.split_relay

    async def _run_supervisor_review(self, *_args, **_kwargs) -> None:
        """A tree campaign has no Supervisor: the selector owns every change of direction."""

    def _update_search_policy(self, _iteration: int) -> None:
        """A tree campaign has no campaign-wide search mode: each start's slots decide how it is planned."""

    async def _begin_search_tree(self) -> None:
        """Root a fresh campaign's tree at a measurement of the published best, or reconcile a resumed one."""
        untracked = untracked_outside(self.ic.workspace_dir, [self._experiments.root.name])
        if untracked:
            raise ValueError(
                "a tree campaign measures candidates in worktrees, which would not see these untracked sources: "
                + ", ".join(untracked[:10])
            )
        self._start_worktrees.clear()
        if self._tree_store.exists():
            self._tree = self._resumed_tree()
        else:
            self._tree = SearchTree.create(await self._measured_root(), root_branch=self.ic.git_branch)
            pin_node(self.ic.workspace_dir, self._tree.root.commit)
        self._tree_store.save(self._tree)
        print(f"  [tree] {len(self._tree.nodes)} node(s), published best {self._tree.published_best[:12]}")

    def _rev_parse(self, revision: str) -> str:
        return git("rev-parse", revision, cwd=self.ic.workspace_dir).stdout.strip()

    async def _measured_root(self) -> TreeNode:
        """The node a fresh campaign starts from, measured in its own worktree as every other node is."""
        head = self._rev_parse("HEAD")
        result = await self._measure_at(0, "measure the search tree's root", self._start_worktrees.ensure(head))
        if not measured_correct(result):
            raise RuntimeError(
                f"the search tree's root {head[:12]} did not measure correct, so the search has no per-case noise to "
                f"read its nodes against: {result.validation_summary}"
            )
        root = TreeNode(
            commit=head,
            parent="",
            direction="",
            slot_type=None,
            iteration=0,
            slot=0,
            samples=(),
            wall_ms=result.wall_ms,
            case_times={},
            case_spreads={},
            snr_db=result.snr_db,
            plan=self.run_state.best.plan,
            source=NodeSource.WARM_START if self.ic.warm_start_commit else NodeSource.BASELINE,
        )
        return root.with_measurement(node_measurement(result))

    def _resumed_tree(self) -> SearchTree:
        tree = self._tree_store.load()
        repair = reconcile_node_refs(self.ic.workspace_dir, tree)
        if repair.lost:
            raise ValueError(f"search tree nodes whose commits no longer exist: {', '.join(repair.lost)}")
        if tree.round is not None:
            tree.close_interrupted_round("the session ended before this round finished")
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
        actors = self._slot_actors(turn)
        plans = await self._plan_round_slots(tree, round_, actors)
        if not self._admit_dispatch(turn.iteration):
            self._close_tree_round(tree, [], plans, "the round could not pay for its sessions")
            return False
        outcomes = await self._expand(tree, turn, plans, actors)
        admitted = [(o.node, o.result) for o in outcomes if o.node is not None]
        merge_agent = SessionAgent(turn.actors.agent_factory, self.run_state.agent_relay.active_profile_id)
        merges = await self._merge_round(tree, turn.iteration, admitted, merge_agent)
        # Nothing is measured on a device a session of this round left held.
        if self.device_hazard.live is None:
            await self._remeasure(tree, turn.iteration, round_.remeasure)
        self._merge_failures = merges.failures
        self._close_tree_round(tree, outcomes, plans, "")
        await self._settle_round(turn, admitted + list(merges.nodes), outcomes, callbacks)
        self._start_worktrees.release([tree.published_best])
        return True

    async def _expand(
        self, tree: SearchTree, turn: IterationTurn, plans: RoundPlans, actors: SlotActors
    ) -> list[SlotOutcome]:
        """Run the round's lanes and measure what they wrote; nothing is measured on a device a lane left held."""
        if not plans.plans:
            return []
        lanes = await self._run_slot_lanes(tree, turn.iteration, plans.plans, actors)
        contended = [lane for lane in lanes.values() if lane.contended]
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
        cycle = self._profile_cycle(turn.iteration)
        labels = self._profile_labels()
        slot_models = tuple(labels.get(profile_at(cycle, position), "") for position in range(turn.round_lanes))
        facts = render_facts(
            tree,
            outlook=RoundOutlook(
                turn.round_lanes,
                self._time_remaining(),
                self._rounds_left(),
                slot_models=slot_models if len(set(slot_models)) > 1 else (),
            ),
            sources=FactSources(
                archive_dir=str(self._experiments.tree_rounds),
                selections_dir=str(self._experiments.tree_selections),
            ),
            merge_failures=self._merge_failures,
            models=labels,
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
        selection = read_selection(
            tree, answer.answer, lanes=turn.round_lanes, failure=answer.failure, profile_cycle=cycle
        )
        self._record_selection(turn.iteration, facts, answer, selection)
        return selection

    def _split_profiles(self) -> tuple[str, ...]:
        """The profiles a split relay gives this round's slots to: every one this session built that is available."""
        pool = self._search_actors
        if not self.ic.split_relay or pool is None:
            return ()
        statuses = self.run_state.agent_availability_snapshot.profiles
        return tuple(
            profile.profile_id
            for profile in pool.profiles
            if pool.has_bundle(profile.profile_id)
            and statuses.get(profile.profile_id) is not None
            and statuses[profile.profile_id].available
        )

    def _profile_cycle(self, iteration: int) -> tuple[str, ...]:
        """The profiles this round's slot positions take in turn: under a split relay every one that can run."""
        split = self._split_profiles()
        if len(split) >= 2:
            return round_profile_cycle(split, iteration)
        active = self.run_state.agent_relay.active_profile_id
        return (active,) if active else ()

    def _slot_actors(self, turn: IterationTurn) -> SlotActors:
        """The search actor of each slot: its own profile's under a split relay, else the round's."""
        split = self._split_profiles()
        if len(split) < 2:
            return SlotActors(turn.actors)
        return SlotActors(turn.actors, {profile_id: self._search_actors.bundle(profile_id) for profile_id in split})

    def _profile_labels(self) -> dict[str, str]:
        """Each Agent profile this campaign knows, as its provider and model."""
        statuses = self.run_state.agent_availability_snapshot.profiles
        return {profile_id: f"{status.provider}/{status.model}" for profile_id, status in statuses.items()}

    def _rounds_left(self) -> int | None:
        recent = [item.total_sec for item in self.run_state.round_costs.recent if item.total_sec > 0]
        if not recent:
            return None
        return int(self._time_remaining() // statistics.median(recent))

    def _record_selection(self, iteration: int, facts: str, answer: SelectorAnswer, selection: Selection) -> None:
        slots = "\n".join(
            f"- slot {slot.index}: {slot.slot_type.value} at {slot.start} {slot.direction or slot.source or ''} "
            f"on {slot.agent_profile_id or 'the session agent'} -- {slot.reason}"
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
                answer.answer.model_dump_json(indent=2)
                if answer.answer is not None
                else f"(no answer: {answer.failure})",
                "",
            ]
        )
        atomic_write_text(self._experiments.tree_selection(iteration), record, options=DURABLE)
        for slot in selection.slots:
            print(
                f"  [tree] slot {slot.index}: {slot.slot_type.value} at {slot.start[:12]} {slot.direction} "
                f"on {slot.agent_profile_id or 'the session agent'}"
            )
        if selection.fallback:
            print(f"  [tree] selector fallback: {selection.fallback}")

    def _close_tree_round(self, tree: SearchTree, outcomes: list[SlotOutcome], plans: RoundPlans, reason: str):
        """Record every slot's attempt, including the slots that never ran, and clear the round."""
        attempts = [outcome.attempt for outcome in outcomes]
        measured = {attempt.slot for attempt in attempts}
        for item in plans.unplanned:
            attempts.append(lost_attempt(tree.round.iteration, item.slot, item.reason))
        for plan in plans.plans:
            if plan.slot.index not in measured:
                attempts.append(lost_attempt(tree.round.iteration, plan.slot, reason or "the slot was not measured"))
        tree.close_round(sorted(attempts, key=lambda item: item.slot))
        self._tree_store.save(tree)

    async def _settle_round(self, turn: IterationTurn, candidates: list, outcomes: list[SlotOutcome], callbacks):
        """Close the iteration: publish the round's best candidate that cleared the bar, or record why none did."""
        clearing = [(node, result) for node, result in candidates if result.kept]
        published = max(clearing, key=lambda item: item[0].score)[0] if clearing else None
        frame = _IterationFrame(turn.iteration, {"plan": _round_summary(outcomes)}, _StagedMerge(None, ""))
        # The round's outcome is the profile's that wrote what it publishes, else the active profile's, which ran it.
        producer = published.agent_profile_id if published is not None else ""
        self._bind_session_producer(frame.session_sink, producer or self.run_state.agent_relay.active_profile_id)
        hazard = self.device_hazard.live
        if hazard is not None:
            # A session of this round left the device held: nothing after it is measured, the publication included.
            attempt = _Attempt(_held_device_result(turn.iteration, hazard.describe()), "", "", "", "", None, "", True)
            await self._settle_locked_attempt(attempt, frame, callbacks=callbacks)
            return
        if published is not None:
            await self._publish_node(published, frame, callbacks)
            return
        measured = [o.result for o in outcomes if o.result is not None]
        if not measured:
            summary = IterationResult(
                iteration=turn.iteration,
                duration_sec=0.0,
                validation_passed=False,
                validation_summary="no slot produced a candidate to measure",
            )
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
        """Stage ``node``'s tree in the campaign workspace and settle it as this iteration's candidate.

        The settle measures the node once more, KEEP or not, and that measurement joins the node's own.
        """
        git("read-tree", "--reset", "-u", node.commit, cwd=self.ic.workspace_dir)
        frame.session_sink["plan"] = node.plan
        attempt = _Attempt(None, "", node.plan, self._read_kernel_source(), self._working_tree_diff(), None, "", True)
        await self._settle_locked_attempt(attempt, frame, callbacks=callbacks)
        result = self.results[-1] if self.results and self.results[-1].iteration == frame.iteration else None
        if measured_correct(result):
            self._tree.add_measurement(node.commit, node_measurement(result))
        if result is not None and result.kept:
            self._tree.publish(node.commit, head=self.run_state.best.commit_hash)
            print(f"  [tree] published node {node.commit[:12]} as the new best")
        self._tree_store.save(self._tree)


def _held_device_result(iteration: int, detail: str) -> IterationResult:
    return IterationResult(
        iteration=iteration,
        duration_sec=0.0,
        validation_passed=False,
        validation_summary=(
            "REVERT (workspace contention): a session of this round left the device held by processes this campaign "
            f"could not clear, so nothing after it was measured and nothing was published. {detail}"
        ),
        kept=False,
        workspace_contention=detail,
    )


def _round_summary(outcomes: list[SlotOutcome]) -> str:
    parts = [
        f"slot {o.attempt.slot} {o.attempt.slot_type.value} at {o.attempt.start[:12]}: {o.attempt.outcome.value}"
        for o in outcomes
    ]
    return "tree round: " + ("; ".join(parts) or "no slot was measured")


__all__ = ["TreeSearchLoop"]
