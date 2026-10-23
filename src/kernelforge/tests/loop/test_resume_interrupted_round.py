# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A resume runs again the round a lost process started and never judged, under its own number and plan."""

from __future__ import annotations

from kernelforge.loop.run_state import LoopStateStore, RunState, make_event
from kernelforge.tests.test_loop_runner import _make_loop


def _interrupted(tmp_path, monkeypatch, *, judged=(1, 2), started=3):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    store = LoopStateStore(str(workspace))
    loop.state_store, loop.run_state = store, RunState(campaign_id="c", iteration=started, next_iteration=started + 1)
    for iteration in judged:
        store.append_event(make_event("iteration_started", iteration))
        store.append_event(make_event("iteration_result", iteration, decision="REVERT_PERF"))
    if started not in judged:
        store.append_event(make_event("iteration_started", started))
        store.append_event(make_event("round_dispatch", started, admitted=True))
    return loop, workspace, store


def test_the_unjudged_round_runs_again_under_its_own_number(tmp_path, monkeypatch):
    loop, _, store = _interrupted(tmp_path, monkeypatch)
    assert loop._resume_iteration(archive_next=3, event_next=4) == 3
    assert loop._restarted_iteration == 3
    assert store.read_events()[-1]["type"] == "iteration_restarted"


def test_a_judged_round_or_an_archived_one_is_moved_past(tmp_path, monkeypatch):
    loop, _, _ = _interrupted(tmp_path, monkeypatch, judged=(1, 2, 3))
    assert loop._resume_iteration(archive_next=4, event_next=4) == 4
    (tmp_path / "archived").mkdir()
    loop, _, _ = _interrupted(tmp_path / "archived", monkeypatch)
    assert loop._resume_iteration(archive_next=4, event_next=4) == 4
    assert loop._restarted_iteration is None


def test_the_interrupted_edits_are_kept_with_their_round_and_the_tree_returns_to_head(tmp_path, monkeypatch):
    loop, workspace, store = _interrupted(tmp_path, monkeypatch)
    (workspace / "kernel.py").write_text("def kernel():\n    return 3\n")
    loop._set_aside_interrupted_round()
    saved = workspace / "forge_experiments" / "orchestration" / "iter_003" / "interrupted.diff"
    assert "+    return 3" in saved.read_text()
    assert loop._git("status", "--porcelain", "--untracked-files=no") == ""
    assert "return 1" in (workspace / "kernel.py").read_text()
    event = store.read_events()[-1]
    assert event["type"] == "interrupted_round_set_aside" and event["files"] == ["kernel.py"]


def test_a_clean_tree_or_no_unjudged_round_leaves_everything_alone(tmp_path, monkeypatch):
    loop, workspace, store = _interrupted(tmp_path, monkeypatch, judged=(1, 2, 3))
    (workspace / "kernel.py").write_text("def kernel():\n    return 3\n")
    count = len(store.read_events())
    loop._set_aside_interrupted_round()
    assert "return 3" in (workspace / "kernel.py").read_text() and len(store.read_events()) == count


def test_the_rerun_uses_the_plan_it_bought_while_the_commit_stands(tmp_path, monkeypatch):
    loop, _, _ = _interrupted(tmp_path, monkeypatch)
    path = loop._persist_lane_plans(3, ["Direction: d1\nprefetch B"], analysis_commit=loop._canonical_commit())
    assert loop._restarted_round_plans(3, lanes=1) is None  # not a rerun
    loop._restarted_iteration = 3
    assert loop._restarted_round_plans(3, lanes=1) == path
    assert loop._last_lane_plans == ["Direction: d1\nprefetch B"]
    assert loop._restarted_round_plans(3, lanes=2) is None
    loop._persist_lane_plans(3, ["Direction: d1\nprefetch B"], analysis_commit="moved-on")
    assert loop._restarted_round_plans(3, lanes=1) is None


def test_the_implementer_of_a_rerun_is_pointed_at_the_interrupted_edits(tmp_path, monkeypatch):
    loop, workspace, _ = _interrupted(tmp_path, monkeypatch)
    assert loop._interrupted_round_block() == ""
    (workspace / "kernel.py").write_text("def kernel():\n    return 3\n")
    loop._set_aside_interrupted_round()
    loop._restarted_iteration = 3
    block = loop._interrupted_round_block()
    assert "interrupted once" in block and "iter_003/interrupted.diff" in block
