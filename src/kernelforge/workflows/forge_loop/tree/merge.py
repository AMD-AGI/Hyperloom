# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merge what a tree round found, before the round settles.

The lanes of a round are distinct attempts, often aimed at different cases, so what the round gained is spread over its
new nodes. The node the round would publish -- its best node that cleared the KEEP bar, else its best -- takes in, best
first, every other new node that runs some case faster than it beyond noise, and the published best when it does not
already hold it. A change that applies to it is measured as it is; one that does not is merged by an Implementer session
the framework plans from both changes and their per-case times. A merge that measures correct is a node under the node
it was merged onto, and one that scores higher than that node is what the next merge builds on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kernelforge.utils.git import git
from kernelforge.workflows.forge_loop.iteration.fanout import LanePlan, LaneResult
from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationResult
from kernelforge.workflows.forge_loop.iteration.merge_candidates import cases_beating_reference
from kernelforge.workflows.forge_loop.policy.round_budget import admit_dispatch
from kernelforge.workflows.forge_loop.tree.admission import (
    LaneSession,
    MeasuredCandidate,
    SessionAgent,
    StagedCandidate,
)
from kernelforge.workflows.forge_loop.tree.model import MergeFailure, SearchTree, TreeNode
from kernelforge.workflows.forge_loop.tree.refs import is_ancestor

_MECHANICAL = "mechanical"
_SESSION = "session"
_BASE = "base"
_SOURCE = "source"


@dataclass(frozen=True)
class MergeRound:
    """The nodes a round's merges added, with their measurements, and the merges that did not land."""

    nodes: tuple[tuple[TreeNode, IterationResult], ...] = ()
    failures: tuple[MergeFailure, ...] = ()


@dataclass(frozen=True)
class _Merge:
    """One merge: the node a change is carried onto, the node it comes from, their common ancestor, its records."""

    base: TreeNode
    source: TreeNode
    ancestor: str
    records: Path

    @property
    def title(self) -> str:
        return f"merge of {self.source.commit} onto {self.base.commit}"


@dataclass(frozen=True)
class _Step:
    """What one merge came to: the node it added with its measurement, or why it added none."""

    node: TreeNode | None = None
    result: IterationResult | None = None
    # Why the change did not land, which a later round can still carry over by PORT.
    failure: str = ""
    # What a merge that added no node and did not fail came to.
    note: str = ""

    @property
    def outcome(self) -> str:
        return self.failure or self.note or f"node {self.node.commit}"


@dataclass(frozen=True)
class _CaseRow:
    """One case of a merge: its time on each side, and the side that runs it faster beyond noise, if either does."""

    case: str
    base_ms: float | None
    source_ms: float | None
    faster: str

    def to_dict(self) -> dict:
        return {"case": self.case, "base_ms": self.base_ms, "source_ms": self.source_ms, "faster": self.faster}


class TreeMergeMixin:
    """The merges that close a tree round.

    The host supplies ``ic``, ``device_hazard``, ``_experiments``, ``_start_worktrees``, ``_time_remaining`` and
    ``_measurement_estimate_sec``, and the admission mixin's ``_run_sessions``, ``_candidate_refusal``,
    ``_measure_and_admit``, ``_admit_merge``, ``_write_lesson`` and ``_write_record``.
    """

    async def _merge_round(
        self,
        tree: SearchTree,
        iteration: int,
        admitted: Sequence[tuple[TreeNode, IterationResult]],
        agent: SessionAgent,
    ) -> MergeRound:
        """Merge the round's gains into the node the round would publish, any merge session running on ``agent``."""
        if not admitted:
            return MergeRound()
        base = max([item for item in admitted if item[1].kept] or admitted, key=lambda item: item[0].score)[0]
        sources = [node for node, _ in admitted if node.commit != base.commit]
        sources.append(tree.require_node(tree.published_best))
        nodes: list[tuple[TreeNode, IterationResult]] = []
        failures: list[MergeFailure] = []
        attempted = 0
        for source in sorted(sources, key=lambda node: node.score, reverse=True):
            # A session that left the device held ends every measurement of the round.
            if self.device_hazard.live is not None:
                break
            if not self._has_gains_for(base, source):
                continue
            attempted += 1
            step = await self._merge(tree, iteration, self._merge_of(iteration, attempted, base, source), agent)
            if step.failure:
                failures.append(MergeFailure(source.commit, base.commit, step.failure))
            if step.node is not None:
                nodes.append((step.node, step.result))
                if step.node.score > base.score:
                    base = step.node
        return MergeRound(tuple(nodes), tuple(failures))

    def _has_gains_for(self, base: TreeNode, source: TreeNode) -> bool:
        """Whether ``source`` runs some case faster than ``base`` beyond its own noise, in code ``base`` does not hold."""
        if is_ancestor(self.ic.workspace_dir, source.commit, base.commit):
            return False
        return bool(cases_beating_reference(source.case_times, base.case_times, source.case_spreads))

    def _merge_of(self, iteration: int, index: int, base: TreeNode, source: TreeNode) -> _Merge:
        ancestor = git("merge-base", source.commit, base.commit, cwd=self.ic.workspace_dir).stdout.strip()
        return _Merge(base, source, ancestor, self._experiments.tree_round(iteration) / f"merge_{index:02d}")

    async def _merge(self, tree: SearchTree, iteration: int, merge: _Merge, agent: SessionAgent) -> _Step:
        """Carry the source's change onto the base as it is when it applies, else by a merge session, and measure it."""
        source_change = self._change_since_ancestor(merge, merge.source)
        self._write_record(merge.records, "source.diff", source_change)
        if not self._can_measure_a_merge():
            return self._recorded(merge, "", _Step(failure="the campaign had too little time left to measure a merge"))
        worktree = self._start_worktrees.ensure(merge.base.commit)
        method, producer = _MECHANICAL, ""
        if not self._start_worktrees.apply(worktree, source_change):
            method, producer = _SESSION, agent.profile_id
            failure = await self._stage_merge_session(iteration, merge, worktree, agent)
            if failure:
                return self._recorded(merge, method, _Step(failure=failure))
        staged = StagedCandidate(worktree, merge.base.commit, merge.title, merge.records, producer)
        measured = await self._measure_and_admit(
            tree, iteration, staged, lambda result: self._admit_merge(tree, staged, result, merge.source)
        )
        return self._recorded(merge, method, _measured_step(measured))

    async def _stage_merge_session(self, iteration: int, merge: _Merge, worktree: Path, agent: SessionAgent) -> str:
        """Have an Implementer merge the two nodes on a copy of the base's worktree, and stage what it wrote there.

        Returns why nothing was staged, or an empty string.
        """
        if not self._can_run_a_merge_session():
            return (
                "its change does not apply to the base, and the campaign had too little time left for a merge session"
            )
        self._write_record(merge.records, "base.diff", self._change_since_ancestor(merge, merge.base))
        plan = _merge_plan(merge)
        self._write_record(merge.records, "plan.md", plan)
        lane_id = merge.records.name
        lanes = await self._run_sessions(iteration, [LaneSession(LanePlan(lane_id, plan, str(worktree)), plan, agent)])
        lane = lanes.get(lane_id)
        if lane is not None:
            self._write_lesson(merge.records, lane)
            if lane.contended:
                self._record_held_device(iteration, lane)
                return "the merge session left the device held"
        refusal = self._candidate_refusal(lane)
        if refusal:
            return f"the merge session produced no candidate: {refusal}"
        self._write_record(merge.records, "change.diff", lane.diff)
        if not self._start_worktrees.apply(worktree, lane.diff):
            return "the merge session's diff did not apply to the base"
        return ""

    def _record_held_device(self, iteration: int, lane: LaneResult) -> None:
        self.device_hazard.record(
            iteration=iteration, detail=f"{lane.lane_id}: {lane.reaped.describe()}", pids=set(lane.reaped.blockers)
        )

    def _change_since_ancestor(self, merge: _Merge, node: TreeNode) -> str:
        return git("diff", merge.ancestor, node.commit, cwd=self.ic.workspace_dir).stdout

    def _can_measure_a_merge(self) -> bool:
        """Whether one more measurement fits beside the measurement the round's publication still needs."""
        return self._time_remaining() >= 2 * self._measurement_estimate_sec()

    def _can_run_a_merge_session(self) -> bool:
        """Whether a merge session and its measurement fit beside the measurement the round's publication needs."""
        measurement = self._measurement_estimate_sec()
        return admit_dispatch(remaining_sec=self._time_remaining() - measurement, measurement_sec=measurement).admitted

    def _recorded(self, merge: _Merge, method: str, step: _Step) -> _Step:
        """Write down what one merge came to, in its records and in the log."""
        self._write_record(
            merge.records,
            "merge.json",
            {
                "base": merge.base.commit,
                "source": merge.source.commit,
                "ancestor": merge.ancestor,
                "method": method,
                "outcome": step.outcome,
                "node": "" if step.node is None else step.node.commit,
                "cases": [row.to_dict() for row in _case_rows(merge.base, merge.source)],
            },
        )
        print(
            f"  [tree] merge {merge.source.commit[:12]} onto {merge.base.commit[:12]} "
            f"({method or 'not attempted'}): {step.outcome}"
        )
        return step


def _measured_step(measured: MeasuredCandidate) -> _Step:
    if measured.holder:
        return _Step(note=f"the merged code is node {measured.holder}")
    if measured.node is None:
        summary = (measured.result.validation_summary or "").strip().splitlines()
        return _Step(result=measured.result, failure=f"the merge measured incorrect: {summary[0] if summary else ''}")
    return _Step(node=measured.node, result=measured.result)


def _case_rows(base: TreeNode, source: TreeNode) -> list[_CaseRow]:
    gains = cases_beating_reference(source.case_times, base.case_times, source.case_spreads)
    holds = cases_beating_reference(base.case_times, source.case_times, base.case_spreads)
    return [
        _CaseRow(
            case,
            base.case_times.get(case),
            source.case_times.get(case),
            _SOURCE if case in gains else _BASE if case in holds else "",
        )
        for case in sorted({*base.case_times, *source.case_times})
    ]


_FASTER_LABELS = {_BASE: "this node", _SOURCE: "the other node", "": "neither"}


def _merge_plan(merge: _Merge) -> str:
    """The plan a merge session's Implementer is given: both changes, and which side runs each case faster."""
    base, source = merge.base, merge.source
    rows = [
        f"| {row.case} | {_ms(row.base_ms)} | {_ms(row.source_ms)} | {_FASTER_LABELS[row.faster]} |"
        for row in _case_rows(base, source)
    ]
    return "\n".join(
        [
            "# Merge plan",
            "",
            f"This checkout is search-tree node {base.commit} ({base.score:.6f}x). Node {source.commit} "
            f"({source.score:.6f}x) runs some cases faster than it beyond noise, and its change does not apply to this "
            "checkout as it is. Merge the two so that every case runs on the faster of the two implementations.",
            "",
            f"Both grew from {merge.ancestor}. This node's change since then is at {merge.records / 'base.diff'}; the "
            f"other node's is at {merge.records / 'source.diff'}.",
            "",
            "Change files only. A git command that changes the index, HEAD or a branch -- `git checkout <commit> -- "
            "<path>`, `git apply --index`, `git add`, `git stash`, `git merge`, `git cherry-pick`, `git reset`, "
            "`git commit` -- makes the loop discard everything this session wrote. Read the other node's version of a "
            f"file with `git show {source.commit}:<path>`.",
            "",
            f"- This node: {_described(base)}",
            f"- The other node: {_described(source)}",
            "",
            "| case | this node (ms) | other node (ms) | faster beyond noise |",
            "|---|---|---|---|",
            *rows,
            "",
            "Keep this node's code for the cases it runs faster, and bring in the other node's for the cases that one "
            "runs faster; a dispatch on the shape is the usual way to give each case its own path. Every case must stay "
            "correct. Check the merged kernel with the driver.",
        ]
    )


def _described(node: TreeNode) -> str:
    parts = [node.plan or "no plan recorded"]
    if node.approach:
        parts.append(f"approach: {node.approach}")
    if node.focus_cases:
        parts.append(f"aimed at: {', '.join(node.focus_cases)}")
    return "; ".join(parts)


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


__all__ = ["MergeRound", "TreeMergeMixin"]
