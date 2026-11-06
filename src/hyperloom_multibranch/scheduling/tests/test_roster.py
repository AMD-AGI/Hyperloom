# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the live roster: the branch cap, K limits, stops, the quota it derives, and its snapshot file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.roster import (
    BranchOrigin,
    BranchStatus,
    BranchStop,
    BranchStopped,
    Roster,
    RosterCaps,
    RosterFull,
    StopCause,
)
from hyperloom_multibranch.toolbox.queue import BRANCH_STOPPED, WorkQueue

NOW = 1000.0
ROOT = "root"
NO_GAIN = "no gain"
MAX_K = 3


def _roster(tmp_path: Path, max_branches: int = 2) -> Roster:
    caps = RosterCaps(max_branches=max_branches, max_k_per_branch=MAX_K, max_parallel_tests=8)
    return Roster(caps, tmp_path / "MB-roster.json", Clock(now=lambda: NOW))


def _open(roster: Roster, start: str = ROOT, k: int = 1) -> str:
    branch = roster.next_branch_id()
    roster.open(branch, origin=BranchOrigin.FIRST_BATCH, start_commit=start, k=k)
    return branch


def _judge_stop(reason: str = NO_GAIN) -> BranchStop:
    return BranchStop(StopCause.JUDGE, reason, NOW)


def test_branch_ids_are_sequential_and_never_reused(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    first = _open(roster)
    roster.stop(first, _judge_stop())
    assert [first, roster.next_branch_id(), roster.next_branch_id()] == ["b01", "b02", "b03"]


def test_cap_counts_registered_branches_that_are_not_stopped(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    roster.add_integration(ROOT, k=1)
    first = _open(roster)
    fork = roster.next_branch_id()
    roster.open(fork, origin=BranchOrigin.FORK, start_commit="", k=1, from_branch=first)
    assert roster.free_slots() == 0
    assert "slots are taken" in roster.open_refusal()
    with pytest.raises(RosterFull, match="slots are taken"):
        _open(roster)
    roster.stop(fork, _judge_stop())
    assert (roster.free_slots(), roster.open_refusal()) == (1, "")
    _open(roster)


def test_closed_roster_opens_nothing(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    roster.close()
    assert roster.closed
    assert (roster.free_slots(), roster.open_refusal()) == (0, "the session is stopping")
    with pytest.raises(RosterFull, match="stopping"):
        _open(roster)


def test_k_is_clamped_on_open_and_set_k(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    branch = _open(roster, k=MAX_K + 5)
    assert roster.entry(branch).k == MAX_K
    assert roster.set_k(branch, 0).k == 1
    assert roster.add_integration(ROOT, k=MAX_K + 1).k == MAX_K
    assert roster.set_k(INTEGRATION_BRANCH, 2).k == 2


def test_stopped_branch_is_never_restarted_or_resized(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    branch = _open(roster)
    fork = roster.next_branch_id()
    roster.open(fork, origin=BranchOrigin.FORK, start_commit="", k=1, from_branch=branch)
    roster.stop(fork, _judge_stop())
    with pytest.raises(BranchStopped):
        roster.started(fork, "fix")
    with pytest.raises(BranchStopped):
        roster.set_k(fork, 2)


def test_first_stop_is_kept(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    branch = _open(roster)
    first = roster.stop(branch, _judge_stop("plateau"))
    again = roster.stop(branch, BranchStop(StopCause.SESSION_END, "time", NOW + 1))
    assert again == first
    assert roster.entry(branch).stop == _judge_stop("plateau")


def test_unknown_branch_and_duplicate_open_raise(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    branch = _open(roster)
    with pytest.raises(KeyError):
        roster.entry("b99")
    with pytest.raises(ValueError, match="already"):
        roster.open(branch, origin=BranchOrigin.FIRST_BATCH, start_commit=ROOT, k=1)


def _submitted_by(queue: WorkQueue, branch: str) -> None:
    queue.check_branch_listed(TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, ROOT))


def test_quota_lists_only_started_live_branches_and_the_toolbox_refuses_the_rest(tmp_path: Path) -> None:
    roster = _roster(tmp_path, max_branches=3)
    roster.add_integration(ROOT, k=1)
    running = _open(roster, k=2)
    stopped = _open(roster)
    roster.stop(stopped, _judge_stop())
    fork = roster.next_branch_id()
    roster.open(fork, origin=BranchOrigin.FORK, start_commit="", k=1, from_branch=running)
    policy = roster.quota_policy((), {})
    assert policy.global_concurrency == 8
    assert dict(policy.branch_k) == {INTEGRATION_BRANCH: 1, running: 2}
    queue = WorkQueue(policy)
    _submitted_by(queue, running)
    for refused in (stopped, fork):
        with pytest.raises(RequestRejected) as rejection:
            _submitted_by(queue, refused)
        assert rejection.value.code == BRANCH_STOPPED
    assert roster.started(fork, "fix").status is BranchStatus.RUNNING
    assert roster.quota_policy((), {}).k_for(fork) == 1


def test_a_stopped_branch_keeps_its_k_while_the_toolbox_holds_tests_its_stop_let_finish(tmp_path: Path) -> None:
    roster = _roster(tmp_path, max_branches=3)
    judged, ended, idle = _open(roster, k=3), _open(roster, k=3), _open(roster, k=3)
    roster.stop(judged, _judge_stop())
    roster.stop(ended, BranchStop(StopCause.SESSION_END, "time_exhausted", 0.0))
    roster.stop(idle, _judge_stop())

    assert dict(roster.quota_policy({judged, ended}, {}).branch_k) == {judged: 3}


def test_every_k_of_the_quota_is_at_least_one_and_under_the_caps(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    roster.add_integration(ROOT, k=0)
    _open(roster, k=-4)
    _open(roster, k=MAX_K + 5)
    policy = roster.quota_policy((), {})
    assert policy.per_branch_k >= 1 and policy.global_concurrency >= 1
    assert sorted(policy.branch_k.values()) == [1, 1, MAX_K]


def test_snapshot_follows_every_change(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    snapshot = tmp_path / "MB-roster.json"
    assert json.loads(snapshot.read_text())["entries"] == []
    branch = _open(roster)
    roster.stop(branch, _judge_stop())
    roster.close()
    saved = json.loads(snapshot.read_text())
    assert saved["closed"] is True
    assert saved["caps"]["max_branches"] == 2
    [record] = saved["entries"]
    assert record["branch"] == branch
    assert record["status"] == "stopped"
    assert record["stop"] == {"cause": "judge", "reason": NO_GAIN, "at": NOW}
