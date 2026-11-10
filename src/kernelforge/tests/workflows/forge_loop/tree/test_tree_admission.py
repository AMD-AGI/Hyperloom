# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Admitting a slot's candidate as a node, on a real repository with a stand-in measurement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from kernelforge.workflows.forge_loop.iteration.fanout import LaneResult
from kernelforge.workflows.forge_loop.tree.model import (
    AttemptOutcome,
    DirectionProposal,
    Round,
    SearchTree,
    Slot,
    SlotType,
)
from kernelforge.workflows.forge_loop.tree.planning import SlotPlan
from kernelforge.tests.workflows.forge_loop.tree._tree_repo import _Host, _commit, _git, _node, _repo

_ROOT = {"A": 10, "B": 10, "C": 10}
_PUBLISHED = {"A": 9, "B": 10, "C": 10}
_CHILD = {"A": 9, "B": 9, "C": 10}
_GRANDCHILD = {"A": 9, "B": 9, "C": 9}


def _lineage(tmp_path: Path):
    """root -> published -> child -> grandchild on one route."""
    repo, root = _repo(tmp_path, _ROOT)
    published = _commit(repo, root, _PUBLISHED)
    child = _commit(repo, published, _CHILD)
    grandchild = _commit(repo, child, _GRANDCHILD)
    tree = SearchTree.create(_node(root, "", "", _ROOT), root_branch="opt")
    tree.allocate_direction(DirectionProposal("route", "", "", ""), origin=root, iteration=1)
    tree.add_node(_node(published, root, "d001", _PUBLISHED))
    tree.add_node(_node(child, published, "d001", _CHILD))
    tree.add_node(_node(grandchild, child, "d001", _GRANDCHILD))
    tree.publish(published, head=published)
    return repo, tree, SimpleNamespace(published=published, child=child, grandchild=grandchild)


def test_an_admitted_node_is_saved_before_its_round_closes(tmp_path):
    repo, tree, ids = _lineage(tmp_path)
    candidate = _commit(repo, ids.published, {"A": 9, "B": 10, "C": 8})
    slot = Slot(1, SlotType.DEEPEN, ids.published, "continue the route", direction="d001")
    tree.open_round(Round(iteration=2, slots=(slot,)))
    host = _Host(repo)
    diff = _git(repo, "diff", ids.published, candidate) + "\n"

    outcome = asyncio.run(host._measure_slot(tree, 2, SlotPlan(slot, "plan", ""), LaneResult("1", "plan", diff)))

    saved = host._tree_store.load()
    assert saved.round is not None and saved.round.iteration == 2
    assert saved.node(outcome.node.commit) is not None


def test_a_lane_that_reproduces_an_existing_node_adds_nothing(tmp_path):
    repo, tree, ids = _lineage(tmp_path)
    diff = _git(repo, "diff", ids.child, ids.grandchild) + "\n"
    slot = Slot(1, SlotType.DEEPEN, ids.child, "continue the route", direction="d001")
    host = _Host(repo)

    outcome = asyncio.run(host._measure_slot(tree, 2, SlotPlan(slot, "plan", ""), LaneResult("1", "plan", diff)))

    assert outcome.node is None and outcome.attempt.outcome is AttemptOutcome.NO_CHANGES
    assert outcome.attempt.detail == f"the candidate is node {ids.grandchild}"
    assert host.measured == [] and len(tree.nodes) == 4
