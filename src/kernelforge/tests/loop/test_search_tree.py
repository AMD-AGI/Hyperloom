"""Tests for the search tree a tree-strategy campaign persists."""

from __future__ import annotations

import json

import pytest

from kernelforge.loop.search_tree import (
    ROOT_SOURCE_BASELINE,
    SearchTree,
    SearchTreeStore,
    TreeAttempt,
    TreeNode,
    TreeSeat,
    render_tree_view,
)

ROOT = "a" * 40


def _node(commit, parent, branch, *, iteration, score, depth=1):
    return TreeNode(
        commit=commit,
        parent=parent,
        branch=branch,
        depth=depth,
        iteration=iteration,
        mean_case_speedup=score,
        wall_ms=1.0 / score,
        case_times={"case": 1.0 / score},
        plan=f"plan {iteration}",
        source="iteration",
    )


def _attempt(seat, decision, *, child="", score=None):
    return TreeAttempt(
        iteration=seat.iteration,
        start_commit=seat.start_commit,
        branch=seat.branch,
        decision=decision,
        plan=f"attempt {seat.iteration}",
        mean_case_speedup=score,
        child_commit=child,
        selector_reason="",
        selector_fallback="",
    )


def _root():
    return TreeNode(
        commit=ROOT,
        parent="",
        branch="opt",
        depth=0,
        iteration=0,
        mean_case_speedup=1.0,
        wall_ms=1.0,
        case_times={"case": 1.0},
        plan="",
        source=ROOT_SOURCE_BASELINE,
    )


def _grow(tree, iteration, start, decision, *, child=None):
    """Seat ``iteration`` on ``start`` and close it, branching off when ``start`` heads no branch."""
    branch = tree.branch_headed_by(start) or tree.allocate_branch(start, iteration)
    seat = TreeSeat(iteration, start, branch, "", "")
    tree.open_seat(seat)
    node = None
    if child is not None:
        commit, score = child
        parent = tree.node(start)
        node = _node(commit, start, branch, iteration=iteration, score=score, depth=parent.depth + 1)
    tree.close_seat(_attempt(seat, decision, child=node.commit if node else "", score=child and child[1]), node)
    return branch


@pytest.fixture
def tree():
    """root -> A(1.2, opt) ; root -> B(1.1, opt-b001) -> C(1.15, opt-b001), with failed attempts around them."""
    tree = SearchTree.create(_root())
    _grow(tree, 1, ROOT, "KEEP", child=("b" * 40, 1.2))
    _grow(tree, 2, ROOT, "KEEP", child=("c" * 40, 1.1))
    _grow(tree, 3, "c" * 40, "REVERT_PERF")
    _grow(tree, 4, "c" * 40, "KEEP", child=("d" * 40, 1.15))
    _grow(tree, 5, "b" * 40, "REVERT_VALIDATION")
    return tree


def test_a_start_that_heads_no_branch_gets_a_new_numbered_branch(tree):
    assert [branch.name for branch in tree.branches] == ["opt", "opt-b001"]
    assert tree.branch("opt").head == "b" * 40
    assert tree.branch("opt-b001").head == "d" * 40
    assert tree.branch("opt-b001").start_commit == ROOT
    assert tree.node("d" * 40).depth == 2
    assert tree.next_branch_seq == 2


def test_the_global_best_is_the_highest_node_and_always_a_branch_head(tree):
    best = tree.global_best()

    assert best.commit == "b" * 40
    assert tree.branch_headed_by(best.commit) == "opt"


def test_history_scope_is_the_line_ending_at_the_start(tree):
    """The path's nodes and its failures, every attempt from the start itself, and nothing from a sibling branch."""
    # From D: B and D are its lineage, iteration 3 failed from B; root's KEEP that made A is another branch.
    assert tree.history_scope("d" * 40) == frozenset({2, 3, 4})
    # From root: every attempt made from it, KEEP or not, so the selector-chosen start knows what it already spawned.
    assert tree.history_scope(ROOT) == frozenset({1, 2})
    # From A: its own creation and the failure from A; B's line is not A's.
    assert tree.history_scope("b" * 40) == frozenset({1, 5})


def test_branch_stall_counts_attempts_since_the_branch_last_grew(tree):
    assert tree.branch_stall("opt") == 1
    assert tree.branch_stall("opt-b001") == 0


def test_resolve_accepts_an_unambiguous_prefix_only(tree):
    assert tree.resolve(("d" * 40)[:12]) == "d" * 40
    assert tree.resolve("dddddd") == ""
    assert tree.resolve("e" * 12) == ""


def test_closing_a_seat_another_iteration_holds_is_refused(tree):
    tree.open_seat(TreeSeat(6, ROOT, "opt", "", ""))

    with pytest.raises(ValueError, match="does not hold the seat"):
        tree.close_seat(_attempt(TreeSeat(7, ROOT, "opt", "", ""), "REVERT_PERF"), None)


def test_store_round_trips_and_refuses_a_best_that_disagrees_with_the_nodes(tree, tmp_path):
    store = SearchTreeStore(str(tmp_path))
    tree.open_seat(TreeSeat(6, "d" * 40, "opt-b001", "why", "selector failed"))
    store.save(tree)

    assert store.load() == tree
    payload = json.loads(store.path.read_text())
    assert payload["best_commit"] == "b" * 40

    payload["best_commit"] = "d" * 40
    store.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="best_commit"):
        store.load()


def test_a_missing_store_reads_as_no_tree(tmp_path):
    assert SearchTreeStore(str(tmp_path)).load() is None


def test_tree_view_says_where_every_attempt_and_lesson_lives(tree):
    view = render_tree_view(tree, archive_dir="/ws/forge_experiments/candidates", lessons_dir="/ws/lessons")

    assert "/ws/forge_experiments/candidates/iter_NNN/" in view
    assert "/ws/lessons/iter_NNN.md" in view
    assert "git diff <parent> <commit>" in view
    assert view.index("## Where the detail lives") < view.index("## Search tree")


def test_tree_view_names_every_node_its_attempts_and_each_branch_stall(tree):
    view = render_tree_view(tree, archive_dir="candidates", lessons_dir="lessons")

    node_lines = {
        line.split()[1]: line for line in view.splitlines() if line.lstrip().startswith("- ") and "depth" in line
    }
    assert set(node_lines) == {ROOT, "b" * 40, "c" * 40, "d" * 40}
    assert "GLOBAL BEST" in node_lines["b" * 40]
    assert "HEAD" in node_lines["d" * 40] and "HEAD" not in node_lines["c" * 40]
    assert "iter 3 REVERT_PERF" in view
    assert "iter 5 REVERT_VALIDATION" in view
    assert "opt-b001: HEAD dddddddddddd" in view
    assert "opt: HEAD bbbbbbbbbbbb" in view and "1 attempt(s) since its last node" in view
