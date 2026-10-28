# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mechanical stacking and duplicate detection, on a real repository with a stand-in measurement."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

from kernelforge.loop.fanout import LaneResult
from kernelforge.loop.iteration_result import IterationResult
from kernelforge.loop.tree.admission import TreeAdmissionMixin
from kernelforge.loop.tree.model import (
    AttemptOutcome,
    DirectionProposal,
    NodeSource,
    SearchTree,
    Slot,
    SlotType,
    TreeNode,
)
from kernelforge.loop.tree.planning import SlotPlan
from kernelforge.loop.tree.worktrees import StartWorktrees

# The published best wins the small case; every challenger here wins the large one.
_PUBLISHED_TIMES = {"small": 1.0, "large": 2.0}
_CHALLENGER_TIMES = {"small": 1.2, "large": 1.5}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _source(values: dict[str, int]) -> str:
    # Far enough apart that a change to one constant is a hunk of its own, as in a real kernel.
    return "".join(f"{key} = {value}\n" + "#\n" * 8 for key, value in values.items())


def _commit(repo: Path, base: str, values: dict[str, int]) -> str:
    _git(repo, "checkout", "-q", "--detach", base)
    (repo / "kernel.py").write_text(_source(values))
    _git(repo, "commit", "-qam", str(values))
    return _git(repo, "rev-parse", "HEAD")


def _node(commit: str, parent: str, direction: str, case_times: dict[str, float]) -> TreeNode:
    return TreeNode(
        commit=commit,
        parent=parent,
        direction=direction,
        slot_type=None,
        iteration=1,
        slot=1,
        samples=(1.0,),
        wall_ms=None,
        case_times=dict(case_times),
        case_spreads=dict.fromkeys(case_times, 0.001),
        snr_db=None,
        plan="",
        source=NodeSource.ITERATION,
    )


class _Host(TreeAdmissionMixin):
    """Just what stacking needs from the loop, with every measurement recorded instead of run."""

    def __init__(self, repo: Path):
        self.ic = SimpleNamespace(workspace_dir=str(repo))
        self._start_worktrees = StartWorktrees(repo)
        self.measured: list[str] = []
        self._tree_round_dir = lambda iteration: repo.parent / "rounds" / f"iter_{iteration:03d}"
        self._lane_rejection = lambda _diff: ""

    async def _measure_at(self, iteration: int, plan: str, worktree: Path) -> IterationResult:
        self.measured.append((worktree / "kernel.py").read_text())
        runs = [{"success": True, "case_times": dict(_CHALLENGER_TIMES)} for _ in range(3)]
        detail = {
            "success": True,
            "measurement_mean_case_speedups": [1.1, 1.1, 1.1],
            "case_times": dict(_CHALLENGER_TIMES),
            "measurements": runs,
        }
        return IterationResult(iteration, 0.0, True, "PASS", mean_case_speedup=1.1, bench_detail=detail)


def _lineage(tmp_path: Path):
    """root -> published -> child -> grandchild on one route, and a sibling route that left before the published best."""
    repo = tmp_path / "workspace"
    repo.mkdir()
    (repo / "kernel.py").write_text(_source({"A": 1, "B": 1, "C": 1}))
    _git(repo, "init", "-q", "-b", "opt")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "root")
    root = _git(repo, "rev-parse", "HEAD")
    published = _commit(repo, root, {"A": 2, "B": 1, "C": 1})
    child = _commit(repo, published, {"A": 2, "B": 2, "C": 1})
    grandchild = _commit(repo, child, {"A": 2, "B": 2, "C": 2})
    sibling = _commit(repo, root, {"A": 1, "B": 1, "C": 3})

    tree = SearchTree.create(_node(root, "", "", _PUBLISHED_TIMES), root_branch="opt")
    for name, origin in (("route", root), ("sibling route", root)):
        tree.allocate_direction(DirectionProposal(name, "", "", ""), origin=origin, iteration=1)
    tree.add_node(_node(published, root, "d001", _PUBLISHED_TIMES))
    tree.add_node(_node(child, published, "d001", _CHALLENGER_TIMES))
    tree.add_node(_node(grandchild, child, "d001", _CHALLENGER_TIMES))
    tree.add_node(_node(sibling, root, "d002", _CHALLENGER_TIMES))
    tree.publish(published, head=published)
    return repo, tree, SimpleNamespace(published=published, grandchild=grandchild, sibling=sibling)


def test_a_node_grown_from_the_published_best_is_not_stacked_back_onto_it(tmp_path):
    repo, tree, ids = _lineage(tmp_path)
    host = _Host(repo)

    composition = asyncio.run(host._compose_onto_published(tree, 2, [tree.require_node(ids.grandchild)]))

    # The grandchild already holds everything the published best has; stacking would only reproduce it.
    assert composition.nodes == ()
    assert host.measured == []


def test_a_route_that_left_before_the_published_best_is_stacked_and_records_its_source(tmp_path):
    repo, tree, ids = _lineage(tmp_path)
    host = _Host(repo)

    composition = asyncio.run(host._compose_onto_published(tree, 2, [tree.require_node(ids.sibling)]))

    ((node, _result),) = composition.nodes
    assert host.measured == [_source({"A": 2, "B": 1, "C": 3})]
    assert node.parent == ids.published and node.source is NodeSource.COMPOSE
    assert node.provenance == (ids.sibling,)


def test_a_lane_that_reproduces_an_existing_node_adds_nothing(tmp_path):
    repo, tree, ids = _lineage(tmp_path)
    child = tree.require_node(ids.grandchild).parent
    diff = _git(repo, "diff", child, ids.grandchild) + "\n"
    slot = Slot(1, SlotType.DEEPEN, child, "continue the route", direction="d001")
    host = _Host(repo)

    outcome = asyncio.run(host._measure_slot(tree, 2, SlotPlan(slot, "plan", ""), LaneResult("1", "plan", diff)))

    assert outcome.node is None and outcome.attempt.outcome is AttemptOutcome.NO_CHANGES
    assert outcome.attempt.detail == f"the candidate is node {ids.grandchild}"
    assert host.measured == [] and len(tree.nodes) == 5


def test_stacking_that_reproduces_an_existing_node_measures_nothing(tmp_path):
    repo, tree, ids = _lineage(tmp_path)
    already = _commit(repo, ids.published, {"A": 2, "B": 1, "C": 3})
    tree.add_node(_node(already, ids.published, "d001", _CHALLENGER_TIMES))
    host = _Host(repo)

    composition = asyncio.run(host._compose_onto_published(tree, 2, [tree.require_node(ids.sibling)]))

    assert composition.nodes == () and composition.unstacked == ()
    assert host.measured == []
