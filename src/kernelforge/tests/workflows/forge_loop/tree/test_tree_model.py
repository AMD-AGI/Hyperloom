# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The search tree's model, its persisted schema and the git references that pin it."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from kernelforge.workflows.forge_loop.tree.model import (
    Attempt,
    AttemptOutcome,
    DirectionProposal,
    DirectionStatus,
    NodeMeasurement,
    NodeSource,
    Round,
    SearchTree,
    Slot,
    SlotStatus,
    SlotType,
    TreeNode,
    lost_attempt,
)
from kernelforge.workflows.forge_loop.tree.refs import (
    pin_node,
    pinned_nodes,
    point_direction_branch,
    reconcile_node_refs,
)
from kernelforge.workflows.forge_loop.tree.store import SearchTreeStore, tree_from_dict, tree_to_dict

PROPOSAL = DirectionProposal(
    name="persistent grid",
    hypothesis="48 of 304 CUs idle on large M",
    expected_ceiling="1.19x on m_4096",
    failure_criterion="no large-M case improves after three steps",
)


def _node(commit, parent, direction, *, iteration, samples=(1.0, 1.0, 1.0), slot_type=SlotType.BRANCH):
    return TreeNode(
        commit=commit,
        parent=parent,
        direction=direction,
        slot_type=slot_type,
        iteration=iteration,
        slot=1,
        samples=tuple(samples),
        wall_ms=1.0,
        case_times={"m_1": 1.0},
        case_spreads={"m_1": 0.01},
        snr_db=40.0,
        plan=f"plan {commit}",
        source=NodeSource.ITERATION,
    )


def _root(commit="a" * 40):
    return TreeNode(
        commit=commit,
        parent="",
        direction="",
        slot_type=None,
        iteration=0,
        slot=0,
        samples=(1.0, 1.0, 1.0),
        wall_ms=2.0,
        case_times={"m_1": 2.0},
        case_spreads={"m_1": 0.02},
        snr_db=None,
        plan="",
        source=NodeSource.BASELINE,
    )


def _grown_tree():
    """root -> d001: n1 (0.9) -> n2 (1.2); root -> d002: n3 (1.1)."""
    tree = SearchTree.create(_root(), root_branch="opt")
    first = tree.allocate_direction(PROPOSAL, origin=tree.root.commit, iteration=1)
    tree.add_node(_node("1" * 40, tree.root.commit, first.direction_id, iteration=1, samples=(0.9, 0.9, 0.9)))
    tree.add_node(_node("2" * 40, "1" * 40, first.direction_id, iteration=2, samples=(1.2, 1.2, 1.2)))
    second = tree.allocate_direction(PROPOSAL, origin=tree.root.commit, iteration=2)
    tree.add_node(_node("3" * 40, tree.root.commit, second.direction_id, iteration=2, samples=(1.1, 1.1, 1.1)))
    return tree


def test_directions_are_named_off_the_campaign_branch_in_sequence():
    tree = _grown_tree()

    assert [item.direction_id for item in tree.directions] == ["d001", "d002"]
    assert [item.branch for item in tree.directions] == ["opt-d001", "opt-d002"]


def test_a_slower_node_is_kept_and_the_direction_best_is_its_highest_node():
    tree = _grown_tree()

    assert tree.node("1" * 40).score < tree.root.score
    assert tree.direction_best("d001").commit == "2" * 40
    assert tree.best_measured().commit == "2" * 40
    assert tree.published_best == tree.root.commit


def test_every_measurement_joins_the_node_and_its_case_statistics_cover_every_run():
    tree = _grown_tree()
    unmeasured = replace(_node("4" * 40, "3" * 40, "d002", iteration=3), samples=(), case_times={}, case_spreads={})
    tree.add_node(unmeasured.with_measurement(NodeMeasurement((1.0, 1.1), ({"m_1": 1.0}, {"m_1": 1.2}))))

    node = tree.add_measurement("4" * 40, NodeMeasurement((0.9,), ({"m_1": 1.4, "m_2": 3.0},)))

    assert tree.node("4" * 40) == node
    assert node.samples == (1.0, 1.1, 0.9)
    assert node.case_samples == {"m_1": (1.0, 1.2, 1.4), "m_2": (3.0,)}
    assert node.case_times == {"m_1": pytest.approx(1.2), "m_2": 3.0}
    # One run cannot say how noisy its case is.
    assert node.case_spreads == {"m_1": pytest.approx(0.2)}


def test_a_node_recorded_without_its_runs_takes_its_case_statistics_from_its_next_measurement():
    recorded = _node("1" * 40, "a" * 40, "d001", iteration=1)

    measured = recorded.with_measurement(NodeMeasurement((1.0, 1.0), ({"m_1": 2.0}, {"m_1": 2.2})))

    assert len(measured.samples) == 5
    assert measured.case_samples == {"m_1": (2.0, 2.2)}
    assert measured.case_times == {"m_1": pytest.approx(2.1)}


def test_a_node_must_grow_from_a_known_parent_on_a_known_direction():
    tree = _grown_tree()

    with pytest.raises(ValueError, match="not a node"):
        tree.add_node(_node("9" * 40, "f" * 40, "d001", iteration=3))
    with pytest.raises(ValueError, match="unknown direction"):
        tree.add_node(_node("9" * 40, "2" * 40, "d999", iteration=3))
    with pytest.raises(ValueError, match="already holds"):
        tree.add_node(_node("2" * 40, "1" * 40, "d001", iteration=3))


def test_history_scope_covers_the_path_its_failures_and_every_attempt_from_the_start():
    tree = _grown_tree()
    tree.open_round(Round(iteration=3, slots=(Slot(1, SlotType.DEEPEN, "2" * 40, "r", direction="d001"),)))
    tree.close_round(
        [
            Attempt(3, 1, SlotType.DEEPEN, "2" * 40, "d001", AttemptOutcome.INCORRECT, None, "", "a", "r"),
        ]
    )
    tree.open_round(Round(iteration=4, slots=(Slot(1, SlotType.DEEPEN, "3" * 40, "r", direction="d002"),)))
    tree.close_round([Attempt(4, 1, SlotType.DEEPEN, "3" * 40, "d002", AttemptOutcome.INCORRECT, None, "", "a", "r")])

    assert tree.history_scope("2" * 40) == frozenset({1, 2, 3})


def test_an_interrupted_round_keeps_the_node_a_slot_already_admitted():
    tree = _grown_tree()
    deepen = Slot(1, SlotType.DEEPEN, "2" * 40, "continue", direction="d001")
    branch = Slot(2, SlotType.BRANCH, tree.root.commit, "open a route")
    tree.open_round(Round(iteration=3, slots=(deepen, branch)))
    tree.add_node(_node("4" * 40, "2" * 40, "d001", iteration=3, samples=(1.3, 1.3, 1.3)))

    tree.close_interrupted_round("the session ended")

    assert tree.round is None
    kept, lost = tree.attempts
    assert (kept.slot, kept.outcome, kept.node, kept.score) == (1, AttemptOutcome.NODE, "4" * 40, pytest.approx(1.3))
    assert (lost.slot, lost.outcome, lost.node, lost.detail) == (2, AttemptOutcome.ERROR, "", "the session ended")


def test_a_prefix_resolves_only_when_it_names_one_node():
    tree = _grown_tree()

    assert tree.resolve("1111111") == "1" * 40
    assert tree.resolve("111111") == ""
    assert tree.resolve("ffffffff") == ""


def test_the_tree_round_trips_through_its_store(tmp_path):
    tree = _grown_tree()
    tree.set_direction_status("d002", DirectionStatus.CLOSED, "converged")
    slot = Slot(1, SlotType.PORT, "3" * 40, "r", source="2" * 40, agent_profile_id="codex")
    tree.open_round(Round(iteration=3, slots=(slot,)))
    tree.mark_slot(1, SlotStatus.DISPATCHED)
    focused = replace(
        _node("4" * 40, "3" * 40, "d002", iteration=3),
        approach="stage via LDS",
        focus_cases=("m_1",),
        agent_profile_id="codex",
        case_samples={"m_1": (1.0, 1.01, 0.99)},
    )
    tree.add_node(focused)
    store = SearchTreeStore(tmp_path / "search_tree.json")

    store.save(tree)

    assert tree_to_dict(store.load()) == tree_to_dict(tree)
    assert store.load().round.slots[0] == replace(slot, status=SlotStatus.DISPATCHED)
    assert store.load().node(focused.commit) == focused


def test_a_tree_saved_before_it_recorded_lane_focus_agent_profiles_and_bench_runs_still_loads():
    tree = _grown_tree()
    tree.open_round(Round(iteration=3, slots=(Slot(1, SlotType.DEEPEN, "3" * 40, "r", direction="d002"),)))
    tree.close_round([lost_attempt(3, tree.round.slots[0], "lost")])
    tree.open_round(Round(iteration=4, slots=(Slot(1, SlotType.DEEPEN, "3" * 40, "r", direction="d002"),)))
    payload = json.loads(json.dumps(tree_to_dict(tree)))
    for node in payload["nodes"]:
        del node["approach"], node["focus_cases"], node["agent_profile_id"], node["case_samples"]
    for record in (*payload["attempts"], *payload["round"]["slots"]):
        del record["agent_profile_id"]

    loaded = tree_from_dict(payload)

    assert [(node.approach, node.focus_cases, node.agent_profile_id) for node in loaded.nodes] == [("", (), "")] * 4
    assert [node.case_samples for node in loaded.nodes] == [{}] * 4
    assert [item.agent_profile_id for item in (*loaded.attempts, *loaded.round.slots)] == ["", ""]


def test_the_store_refuses_a_field_set_it_did_not_write(tmp_path):
    payload = tree_to_dict(_grown_tree())
    payload["nodes"][1]["extra"] = 1

    with pytest.raises(ValueError, match="fields do not match"):
        tree_from_dict(payload)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda payload: payload.update(published_best="e" * 40), "published best"),
        (lambda payload: payload["nodes"].reverse(), "root node"),
        (lambda payload: payload["directions"][0].update(origin="e" * 40), "not a node"),
    ],
)
def test_the_store_refuses_references_that_do_not_close(corrupt, message):
    payload = json.loads(json.dumps(tree_to_dict(_grown_tree())))
    corrupt(payload)

    with pytest.raises(ValueError, match=message):
        tree_from_dict(payload)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    commits = []
    for index in range(2):
        (repo / "k.py").write_text(f"v = {index}\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", f"c{index}"], cwd=repo, check=True)
        commits.append(
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    return repo, commits


def test_node_refs_and_direction_branches_point_where_the_tree_says(tmp_path):
    repo, (first, second) = _repo(tmp_path)

    pin_node(repo, first)
    point_direction_branch(repo, "opt-d001", second)

    assert pinned_nodes(repo) == {first}
    branch_head = subprocess.run(["git", "rev-parse", "opt-d001"], cwd=repo, check=True, capture_output=True, text=True)
    assert branch_head.stdout.strip() == second


def test_reconcile_repins_a_missing_ref_and_names_a_lost_commit(tmp_path):
    repo, (first, second) = _repo(tmp_path)
    tree = SearchTree.create(_root(first), root_branch="opt")
    direction = tree.allocate_direction(PROPOSAL, origin=first, iteration=1)
    tree.add_node(_node(second, first, direction.direction_id, iteration=1))
    tree.add_node(_node("d" * 40, second, direction.direction_id, iteration=2))

    repair = reconcile_node_refs(repo, tree)

    assert set(repair.repinned) == {first, second}
    assert repair.lost == ("d" * 40,)
    assert pinned_nodes(repo) == {first, second}
