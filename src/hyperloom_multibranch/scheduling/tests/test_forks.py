# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fork admission: an accepted fork takes a slot at once and waits for its fix commit; past the cap it is refused."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.scheduling import AdmissionStatus, ForkRequest
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import CommittedFork, ForkDesk
from hyperloom_multibranch.scheduling.roster import (
    BranchOrigin,
    BranchStatus,
    BranchStop,
    Roster,
    RosterCaps,
    StopCause,
)

CLOCK = Clock(now=lambda: 1_700_000_000.0)


def _notebook(tmp_path: Path, role: Role = Role.SCHEDULER) -> LocalNotebook:
    return LocalNotebook(tmp_path / "nb", Writer(role, role.value), clock=CLOCK.now)


def _desk(tmp_path: Path, max_branches: int) -> tuple[Roster, ForkDesk]:
    roster = Roster(RosterCaps(max_branches, 2, 8), tmp_path / "MB-roster.json", CLOCK)
    roster.open(roster.next_branch_id(), origin=BranchOrigin.FIRST_BATCH, start_commit="c0", k=2)
    return roster, ForkDesk(roster, _notebook(tmp_path), DecisionLog(tmp_path / "MB-decisions.jsonl", CLOCK))


async def test_an_accepted_fork_is_admitted_with_its_parent_k_until_its_fix_commit(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=2)

    answer = await desk.admit(ForkRequest("b01", "c1", "fp8 fix regresses"))

    assert (answer.status, answer.branch) == (AdmissionStatus.ACCEPTED, "b02")
    entry = roster.entry("b02")
    assert (entry.status, entry.origin, entry.from_branch, entry.k) == (
        BranchStatus.ADMITTED,
        BranchOrigin.FORK,
        "b01",
        2,
    )
    assert "b02" not in roster.quota_policy().branch_k
    assert desk.drain() == ()
    await desk.fork_committed("b02", "f1")
    assert desk.drain() == (CommittedFork("b02", "f1"),)
    assert desk.drain() == ()


async def test_a_fork_is_refused_at_the_cap_and_once_the_session_stops(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=1)

    at_cap = await desk.admit(ForkRequest("b01", "c1", "fix"))
    roster.close()
    closing = await desk.admit(ForkRequest("b01", "c1", "fix"))

    assert (at_cap.status, at_cap.reason) == (AdmissionStatus.REJECTED, "all 1 branch slots are taken")
    assert (closing.status, closing.reason) == (AdmissionStatus.REJECTED, "the session is stopping")
    assert [entry.branch for entry in roster.entries()] == ["b01"]
    logged = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    assert [row["detail"]["status"] for row in logged] == ["rejected", "rejected"]


async def test_a_fork_of_a_stopped_or_unknown_branch_is_refused(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=3)
    roster.stop("b01", BranchStop(StopCause.JUDGE, "flat", CLOCK.now()))

    stopped = await desk.admit(ForkRequest("b01", "c1", "fix"))
    unknown = await desk.admit(ForkRequest("b09", "c1", "fix"))

    assert (stopped.status, stopped.reason) == (AdmissionStatus.REJECTED, "parent branch is stopped")
    assert unknown.status is AdmissionStatus.REJECTED and "b09" in unknown.reason
    assert [entry.branch for entry in roster.entries()] == ["b01"]


async def test_a_fork_stopped_before_its_split_is_stopped_in_the_lineage_and_never_started(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=2)
    notebook = _notebook(tmp_path)
    await notebook.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    answer = await desk.admit(ForkRequest("b01", "c1", "fix"))
    roster.stop(answer.branch, BranchStop(StopCause.JUDGE, "flat", CLOCK.now()))

    split = LineageEvent(LineageKind.SPLIT, answer.branch, commit_id="f1", from_commit="c1", from_branch="b01")
    await _notebook(tmp_path, Role.GATE).record_lineage(split)
    await desk.fork_committed(answer.branch, "f1")

    assert desk.drain() == ()
    branches = {branch.branch: branch for branch in (await notebook.lineage()).branches}
    assert branches[answer.branch].stop_reason == "judge: flat"
