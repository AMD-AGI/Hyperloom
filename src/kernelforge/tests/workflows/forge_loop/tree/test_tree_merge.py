# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merging a round's gains onto the node it would publish, on a real repository with a stand-in measurement."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from kernelforge.utils.process_reaping import ReapReport
from kernelforge.workflows.forge_loop.iteration.fanout import LaneResult
from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationResult
from kernelforge.workflows.forge_loop.policy.round_budget import FIRST_ROUND_MEASUREMENT_SEC
from kernelforge.workflows.forge_loop.tree.merge import MergeRound
from kernelforge.workflows.forge_loop.tree.model import DirectionProposal, NodeSource, SearchTree, TreeNode
from kernelforge.tests.workflows.forge_loop.tree._tree_repo import _Host, _commit, _git, _node, _repo, _source

_EVEN = {"SMALL": 10, "LARGE": 10}


def _tree(tmp_path: Path, values: dict[str, int]) -> tuple[Path, SearchTree]:
    repo, root = _repo(tmp_path, values)
    return repo, SearchTree.create(_node(root, "", "", values), root_branch="opt")


def _grow(repo: Path, tree: SearchTree, parent: str, values: dict[str, int], *, iteration: int = 1) -> TreeNode:
    """A node holding ``values`` under ``parent``, on a direction of its own opened there."""
    commit = _commit(repo, parent, values)
    proposal = DirectionProposal(f"route to {values}", "", "", "")
    direction = tree.allocate_direction(proposal, origin=parent, iteration=iteration)
    node = replace(_node(commit, parent, direction.direction_id, values), iteration=iteration)
    tree.add_node(node)
    return node


def _admitted(*nodes: TreeNode, kept: tuple[TreeNode, ...] = ()) -> list[tuple[TreeNode, IterationResult]]:
    """The round's new nodes, each with the measurement it was admitted on."""
    return [
        (node, IterationResult(node.iteration, 0.0, True, "PASS", mean_case_speedup=node.score, kept=node in kept))
        for node in nodes
    ]


def _merge(host: _Host, tree: SearchTree, admitted, *, iteration: int = 1) -> MergeRound:
    return asyncio.run(host._merge_round(tree, iteration, admitted, agent_factory=None))


def _record(host: _Host, index: int = 1, iteration: int = 1) -> dict:
    return json.loads((host._experiments.tree_round(iteration) / f"merge_{index:02d}" / "merge.json").read_text())


def _session_writing(repo: Path, base: TreeNode, values: dict[str, int]) -> LaneResult:
    """A merge session's result: the change from ``base`` to ``values``."""
    return LaneResult("", "", diff=_git(repo, "diff", base.commit, _commit(repo, base.commit, values)) + "\n")


def test_a_change_that_applies_is_merged_as_it_is(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 10})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 5})
    host = _Host(repo)

    merges = _merge(host, tree, _admitted(small, large))

    ((merged, _result),) = merges.nodes
    assert host.prompts == [] and host.measured == [_source({"SMALL": 4, "LARGE": 5})]
    assert (merged.parent, merged.direction, merged.provenance) == (small.commit, small.direction, (large.commit,))
    assert merged.source is NodeSource.COMPOSE and host._tree_store.load().node(merged.commit) is not None
    assert (_record(host)["method"], _record(host)["node"]) == ("mechanical", merged.commit)
    assert merges.failures == ()


def test_a_change_that_does_not_apply_is_merged_by_a_session_told_which_node_runs_each_case_faster(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    # Each node wins its own case at the other's expense, in the same lines.
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 14})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 15, "LARGE": 5})
    host = _Host(repo)
    host.session_results = [_session_writing(repo, small, {"SMALL": 4, "LARGE": 5})]

    merges = _merge(host, tree, _admitted(small, large))

    (prompt,) = host.prompts
    records = host._experiments.tree_round(1) / "merge_01"
    assert prompt.startswith("# Merge plan")
    assert "| large | 1.4000 | 0.5000 | the other node |" in prompt
    assert "| small | 0.4000 | 1.5000 | this node |" in prompt
    assert f"{records / 'base.diff'}" in prompt and (records / "source.diff").read_text().strip()
    ((merged, _result),) = merges.nodes
    assert host.measured == [_source({"SMALL": 4, "LARGE": 5})] and merged.provenance == (large.commit,)
    assert _record(host)["method"] == "session"


def test_a_merge_that_reproduces_a_node_is_not_measured(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 10})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 5})
    both = {"SMALL": 4, "LARGE": 5}
    already = _node(_commit(repo, small.commit, both), small.commit, small.direction, both)
    tree.add_node(already)
    host = _Host(repo)

    merges = _merge(host, tree, _admitted(small, large))

    assert merges == MergeRound() and host.measured == []
    assert _record(host)["outcome"] == f"the merged code is node {already.commit}"


def test_a_merge_that_scores_higher_is_what_the_next_merge_builds_on(tmp_path):
    repo, tree = _tree(tmp_path, {"SMALL": 10, "LARGE": 10, "WIDE": 10})
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 10, "WIDE": 10})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 5, "WIDE": 10})
    wide = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 10, "WIDE": 6})
    host = _Host(repo)

    merges = _merge(host, tree, _admitted(small, large, wide))

    first, second = (node for node, _ in merges.nodes)
    assert (first.parent, first.provenance) == (small.commit, (large.commit,))
    assert (second.parent, second.provenance) == (first.commit, (wide.commit,))
    assert host.measured[-1] == _source({"SMALL": 4, "LARGE": 5, "WIDE": 6})


def test_the_base_is_the_round_s_best_node_that_cleared_the_bar(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 10})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 5})
    host = _Host(repo)

    merges = _merge(host, tree, _admitted(small, large, kept=(large,)))

    ((merged, _result),) = merges.nodes
    assert (merged.parent, merged.provenance) == (large.commit, (small.commit,))


def test_the_published_best_is_merged_onto_a_base_that_does_not_hold_it(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    published = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 6})
    tree.publish(published.commit, head=published.commit)
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 10}, iteration=2)
    host = _Host(repo)

    merges = _merge(host, tree, _admitted(small), iteration=2)

    ((merged, _result),) = merges.nodes
    assert (merged.parent, merged.provenance) == (small.commit, (published.commit,))
    assert host.measured == [_source({"SMALL": 4, "LARGE": 6})]


def test_a_base_grown_from_the_published_best_does_not_take_it_again(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    published = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 6})
    tree.publish(published.commit, head=published.commit)
    deeper = _grow(repo, tree, published.commit, {"SMALL": 4, "LARGE": 6}, iteration=2)
    host = _Host(repo)

    merges = _merge(host, tree, _admitted(deeper), iteration=2)

    assert merges == MergeRound() and host.measured == []


def test_a_node_that_runs_no_case_faster_beyond_noise_is_not_merged(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 10})
    slower = _grow(repo, tree, tree.root.commit, {"SMALL": 6, "LARGE": 10})
    host = _Host(repo)

    merges = _merge(host, tree, _admitted(small, slower))

    assert merges == MergeRound() and host.measured == []


def test_a_merge_the_campaign_cannot_pay_for_is_left_to_a_later_port(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 14})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 15, "LARGE": 5})
    # Room for a measurement beside the publication's, but not for a merge session as well.
    host = _Host(repo, remaining_sec=3 * FIRST_ROUND_MEASUREMENT_SEC)

    merges = _merge(host, tree, _admitted(small, large))

    (failure,) = merges.failures
    assert (failure.source, failure.base) == (large.commit, small.commit)
    assert failure.reason.endswith("too little time left for a merge session")
    assert merges.nodes == () and host.prompts == [] and host.measured == []


def test_a_merge_that_measures_incorrect_is_left_to_a_later_port(tmp_path):
    repo, tree = _tree(tmp_path, _EVEN)
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 10})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 10, "LARGE": 5})
    host = _Host(repo)
    host.incorrect = {_source({"SMALL": 4, "LARGE": 5})}

    merges = _merge(host, tree, _admitted(small, large))

    (failure,) = merges.failures
    assert merges.nodes == () and failure.reason.startswith("the merge measured incorrect: FAIL")


def test_a_merge_session_that_leaves_the_device_held_ends_the_round_s_merges(tmp_path):
    repo, tree = _tree(tmp_path, {"SMALL": 10, "LARGE": 10, "WIDE": 10})
    small = _grow(repo, tree, tree.root.commit, {"SMALL": 4, "LARGE": 14, "WIDE": 14})
    large = _grow(repo, tree, tree.root.commit, {"SMALL": 15, "LARGE": 5, "WIDE": 15})
    wide = _grow(repo, tree, tree.root.commit, {"SMALL": 16, "LARGE": 16, "WIDE": 6})
    host = _Host(repo)
    host.session_results = [LaneResult("", "", error="RuntimeError: held", reaped=ReapReport(unkillable=(4242,)))]

    merges = _merge(host, tree, _admitted(small, large, wide))

    assert [item.reason for item in merges.failures] == ["the merge session left the device held"]
    assert host.device_hazard.live.pids == {4242}
    assert len(host.prompts) == 1 and host.measured == [] and merges.nodes == ()
