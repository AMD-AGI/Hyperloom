# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading the selector's answer into legal slots, and the facts it is given."""

from __future__ import annotations

from kernelforge.loop.tree.facts import FactSources, RoundOutlook, render_facts
from kernelforge.loop.tree.model import (
    Attempt,
    AttemptOutcome,
    DirectionProposal,
    DirectionStatus,
    NodeSource,
    Round,
    SearchTree,
    Slot,
    SlotType,
    TreeNode,
)
from kernelforge.loop.tree.selection import read_selection

ROOT, N1, N2, N3 = "a" * 40, "1" * 40, "2" * 40, "3" * 40
PROPOSAL = DirectionProposal("tile table", "retune per-M tiles", "1.1x", "no case moves")


def _node(commit, parent, direction, score, *, case_times, iteration):
    return TreeNode(
        commit=commit,
        parent=parent,
        direction=direction,
        slot_type=SlotType.BRANCH if parent else None,
        iteration=iteration,
        slot=1,
        samples=(score,) * 3,
        wall_ms=1.0,
        case_times=case_times,
        case_spreads=dict.fromkeys(case_times, 0.01),
        snr_db=40.0,
        plan=f"plan of {commit[:4]}",
        source=NodeSource.ITERATION if parent else NodeSource.BASELINE,
    )


def _tree():
    """root -> d001: n1 -> n2 (published); root -> d002: n3, which wins case big."""
    tree = SearchTree.create(
        _node(ROOT, "", "", 1.0, case_times={"small": 1.0, "big": 4.0}, iteration=0), root_branch="opt"
    )
    tree.allocate_direction(PROPOSAL, origin=ROOT, iteration=1)
    tree.add_node(_node(N1, ROOT, "d001", 1.05, case_times={"small": 0.9, "big": 4.0}, iteration=1))
    tree.add_node(_node(N2, N1, "d001", 1.10, case_times={"small": 0.8, "big": 4.0}, iteration=2))
    tree.allocate_direction(PROPOSAL, origin=ROOT, iteration=2)
    tree.add_node(_node(N3, ROOT, "d002", 0.98, case_times={"small": 1.2, "big": 3.0}, iteration=2))
    tree.publish(N2, head="c" * 40)
    return tree


def test_a_legal_answer_becomes_slots_in_order():
    selection = read_selection(
        _tree(),
        {
            "slots": [
                {"type": "deepen", "start": N2[:8], "direction": "d001", "reason": "still gaining"},
                {"type": "BRANCH", "start": N2, "hint": "persistent grid", "reason": "one direction here"},
                {"type": "PORT", "start": N3, "source": N2, "gains": "small-M tiles", "reason": "big-M structure"},
            ]
        },
        lanes=4,
    )

    assert [(item.index, item.slot_type, item.start) for item in selection.slots] == [
        (1, SlotType.DEEPEN, N2),
        (2, SlotType.BRANCH, N2),
        (3, SlotType.PORT, N3),
    ]
    assert selection.slots[1].hint == "persistent grid"
    assert selection.slots[2].source == N2 and selection.fallback == "" and selection.dropped == ()


def test_illegal_slots_are_dropped_with_their_reason():
    selection = read_selection(
        _tree(),
        {
            "slots": [
                {"type": "DEEPEN", "start": N3, "direction": "d001", "reason": "wrong line"},
                {"type": "DEEPEN", "start": N2, "direction": "d009", "reason": "unknown"},
                {"type": "PORT", "start": N3, "source": N3, "gains": "x", "reason": "self"},
                {"type": "BRANCH", "start": "deadbeefdead", "reason": "no such node"},
                {"type": "DEEPEN", "start": N2, "direction": "d001", "reason": "ok"},
                {"type": "DEEPEN", "start": N2, "direction": "d001", "reason": "twice"},
            ]
        },
        lanes=4,
    )

    assert [item.reason for item in selection.slots] == ["ok"]
    assert len(selection.dropped) == 5
    assert any("not on direction d001" in item for item in selection.dropped)
    assert any("repeats" in item for item in selection.dropped)


def test_a_direction_closed_in_the_same_answer_cannot_be_deepened_but_a_reopened_one_can():
    tree = _tree()
    tree.set_direction_status("d002", DirectionStatus.CLOSED, "stalled")

    selection = read_selection(
        tree,
        {
            "direction_status": [
                {"direction": "d001", "status": "CLOSED", "reason": "converged"},
                {"direction": "d002", "status": "OPEN", "reason": "new evidence from the port"},
            ],
            "slots": [
                {"type": "DEEPEN", "start": N2, "direction": "d001", "reason": "closed now"},
                {"type": "DEEPEN", "start": N3, "direction": "d002", "reason": "reopened"},
            ],
        },
        lanes=2,
    )

    assert [item.direction for item in selection.slots] == ["d002"]
    assert [(item.direction_id, item.status) for item in selection.status_changes] == [
        ("d001", DirectionStatus.CLOSED),
        ("d002", DirectionStatus.OPEN),
    ]


def test_slots_beyond_the_lanes_are_dropped():
    slots = [{"type": "BRANCH", "start": N2, "reason": str(index)} for index in range(3)]

    selection = read_selection(_tree(), {"slots": slots}, lanes=2)

    assert len(selection.slots) == 2
    assert "beyond the round's 2 lane(s)" in selection.dropped[0]


def test_no_legal_slot_falls_back_to_the_published_best_and_says_why():
    tree = _tree()

    deepened = read_selection(tree, None, lanes=3, failure="TimeoutError: selector")
    tree.set_direction_status("d001", DirectionStatus.CLOSED, "converged")
    branched = read_selection(tree, {"slots": []}, lanes=3)

    assert [(item.slot_type, item.start, item.direction) for item in deepened.slots] == [(SlotType.DEEPEN, N2, "d001")]
    assert deepened.fallback == "TimeoutError: selector"
    assert [(item.slot_type, item.start) for item in branched.slots] == [(SlotType.BRANCH, N2)]
    assert branched.fallback == "the selector named no legal slot"


def test_requests_keep_only_nodes_of_the_tree():
    selection = read_selection(
        _tree(),
        {"slots": [], "remeasure": [N2[:9], "ffffffff", N2], "refresh_analysis": [N3]},
        lanes=1,
    )

    assert selection.remeasure == (N2,)
    assert selection.refresh_analysis == (N3,)
    assert any("ffffffff" in item for item in selection.dropped)


def test_the_facts_carry_directions_nodes_case_standing_and_earlier_rounds():
    tree = _tree()
    tree.open_round(Round(iteration=3, slots=(Slot(1, SlotType.DEEPEN, N3, "keep going", direction="d002"),)))
    tree.close_round([Attempt(3, 1, SlotType.DEEPEN, N3, "d002", AttemptOutcome.INCORRECT, None, "", "", "keep going")])

    facts = render_facts(
        tree,
        outlook=RoundOutlook(lanes=3, remaining_sec=3600, rounds_left=2),
        sources=FactSources("/a", "/l", "/s"),
        compose_failures=((N3, N2),),
    )

    assert f"Published best: {N2}" in facts
    assert 'd002 "tile table" [OPEN]' in facts
    assert "beats published beyond noise on: big; loses on: small" in facts
    assert "iter 3 slot 1 DEEPEN d002 INCORRECT" in facts
    assert "iteration 3: DEEPEN at 333333333333 (keep going) -> INCORRECT" in facts
    assert "333333333333 onto 222222222222" in facts
    assert "about 2 round(s)" in facts
