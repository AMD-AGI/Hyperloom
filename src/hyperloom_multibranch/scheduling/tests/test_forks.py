# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The branch requests the rule gate makes: an admitted fork or repair takes a slot at once, with the K of the branch
that asked, and waits on the desk until it is released; past the cap it waits in a queue, first in first out, and
enters a freed slot before anything new; a stopped or unknown branch, a stopping session and a full queue refuse it,
and a stopped branch or session voids what it queued."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission, ForkRequest, RepairRequest
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import QUEUE_FULL, WAITING, AdmittedBranch, ForkDesk, PendingBranch
from hyperloom_multibranch.scheduling.roster import (
    BranchOrigin,
    BranchStatus,
    BranchStop,
    Roster,
    RosterCaps,
    StopCause,
)
from hyperloom_multibranch.scheduling.tests.peer_fakes import tagged

CLOCK = Clock(now=lambda: 1_700_000_000.0)
FIX = ForkRequest("b01", "c1", "fp8 fix regresses")
BLOCKED = Judgement(Verdict.BLOCKED, 6.0, 2.0, ("r1", "r2"), {"output_throughput": 1060.0}, accuracy_ok=False)


def _repair(branch: str = "b01", mechanism: str = "fp8 kv cache") -> RepairRequest:
    intent = Intent(branch, "c1", "attention", mechanism, IntentLayer.CONFIG)
    return RepairRequest(intent, tagged("fp8-kv"), BLOCKED, "per-tensor KV scales; scale per head")


def _desk(tmp_path: Path, max_branches: int) -> tuple[Roster, ForkDesk]:
    roster = Roster(RosterCaps(max_branches, 2, 8), tmp_path / "MB-roster.json", CLOCK)
    roster.open(roster.next_branch_id(), origin=BranchOrigin.FIRST_BATCH, start_commit="c0", k=2)
    return roster, ForkDesk(roster, DecisionLog(tmp_path / "MB-decisions.jsonl", CLOCK), CLOCK)


def _logged(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]


async def test_an_accepted_fork_names_its_branch_and_waits_on_the_desk_until_released(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=2)

    answer = await desk.admit(FIX)

    assert answer == ForkAdmission(AdmissionStatus.ACCEPTED, FIX.reason, "b02")
    entry = roster.entry("b02")
    assert (entry.status, entry.origin, entry.from_branch, entry.k) == (
        BranchStatus.ADMITTED,
        BranchOrigin.FORK,
        "b01",
        2,
    )
    assert "b02" not in roster.quota_policy().branch_k
    assert desk.waiting() == desk.waiting() == (AdmittedBranch("b02", "c1"),)
    desk.release("b02")
    assert desk.waiting() == ()
    [logged] = _logged(tmp_path)
    assert (logged["kind"], logged["branch"], logged["detail"]["status"]) == ("fork", "b02", "accepted")


async def test_an_accepted_repair_inherits_its_branch_k_and_carries_its_request(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=3)
    request = _repair()

    answer = await desk.admit_repair(request)

    assert (answer.status, answer.branch) == (AdmissionStatus.ACCEPTED, "b02")
    entry = roster.entry("b02")
    assert (entry.origin, entry.from_branch, entry.k) == (BranchOrigin.REPAIR, "b01", 2)
    assert desk.waiting() == (AdmittedBranch("b02", "c1", request),)
    logged = _logged(tmp_path)[-1]
    assert (logged["kind"], logged["detail"]["diagnosis"]) == ("repair", "per-tensor KV scales; scale per head")


async def test_an_intent_is_repaired_once_per_commit(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=4)

    first = await desk.admit_repair(_repair())
    again = await desk.admit_repair(_repair())
    other = await desk.admit_repair(_repair(mechanism="bf16 kv cache"))

    assert [first.status, again.status, other.status] == [
        AdmissionStatus.ACCEPTED,
        AdmissionStatus.REJECTED,
        AdmissionStatus.ACCEPTED,
    ]
    assert "already open or queued" in again.reason
    assert [admitted.branch for admitted in desk.waiting()] == ["b02", "b03"]


async def test_a_request_past_the_cap_waits_and_enters_the_first_freed_slot(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=2)
    roster.open(roster.next_branch_id(), origin=BranchOrigin.FIRST_BATCH, start_commit="c0", k=1)

    pending = await desk.admit(FIX)
    desk.fill_free_slots()

    assert pending == ForkAdmission(AdmissionStatus.PENDING, WAITING)
    assert desk.pending() == (PendingBranch("b03", BranchOrigin.FORK, "b01", "c1", CLOCK.now()),)
    assert [entry.branch for entry in roster.entries()] == ["b01", "b02"] and desk.waiting() == ()

    roster.stop("b02", BranchStop(StopCause.JUDGE, "flat", CLOCK.now()))
    desk.fill_free_slots()

    assert desk.pending() == () and desk.waiting() == (AdmittedBranch("b03", "c1"),)
    entry = roster.entry("b03")
    assert (entry.status, entry.origin, entry.from_branch, entry.k) == (
        BranchStatus.ADMITTED,
        BranchOrigin.FORK,
        "b01",
        2,
    )
    logged = _logged(tmp_path)
    assert [(row["kind"], row["branch"], row["detail"]["status"]) for row in logged] == [
        ("fork", "b03", "pending"),
        ("fork", "b03", "dequeued"),
    ]
    assert logged[0]["reason"] == FIX.reason


async def test_a_request_arriving_while_others_wait_queues_behind_them_even_on_a_free_slot(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=2)
    roster.open(roster.next_branch_id(), origin=BranchOrigin.FIRST_BATCH, start_commit="c0", k=1)

    first = await desk.admit(FIX)
    roster.stop("b02", BranchStop(StopCause.JUDGE, "flat", CLOCK.now()))
    second = await desk.admit_repair(_repair())

    assert [first.status, second.status] == [AdmissionStatus.PENDING, AdmissionStatus.PENDING]
    assert [pending.branch for pending in desk.pending()] == ["b03", "b04"]
    desk.fill_free_slots()
    assert [admitted.branch for admitted in desk.waiting()] == ["b03"]
    assert [pending.branch for pending in desk.pending()] == ["b04"]


async def test_a_full_queue_and_a_stopping_session_refuse_a_request(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=1)

    queued = await desk.admit(FIX)
    full = await desk.admit_repair(_repair())
    roster.close()
    closing = await desk.admit(FIX)

    assert queued.status is AdmissionStatus.PENDING
    assert (full.status, full.reason) == (AdmissionStatus.REJECTED, QUEUE_FULL)
    assert (closing.status, closing.reason) == (AdmissionStatus.REJECTED, "the session is stopping")
    assert [entry.branch for entry in roster.entries()] == ["b01"]
    assert [row["detail"]["status"] for row in _logged(tmp_path)] == ["pending", "rejected", "rejected"]
    assert len(desk.pending()) == 1 and desk.waiting() == ()


async def test_a_queued_repair_of_an_intent_refuses_the_same_intent_until_it_is_voided(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=1)

    queued = await desk.admit_repair(_repair())
    again = await desk.admit_repair(_repair())
    await desk.void_all("time_exhausted")
    retried = await desk.admit_repair(_repair())

    assert [queued.status, again.status, retried.status] == [
        AdmissionStatus.PENDING,
        AdmissionStatus.REJECTED,
        AdmissionStatus.PENDING,
    ]
    assert "already open or queued" in again.reason
    assert [pending.branch for pending in desk.pending()] == ["b03"]


async def test_a_stopped_branch_voids_what_it_queued_and_the_session_stop_voids_the_rest(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=2)
    roster.open(roster.next_branch_id(), origin=BranchOrigin.FIRST_BATCH, start_commit="c0", k=1)
    await desk.admit(FIX)
    await desk.admit(ForkRequest("b02", "c2", "fix f2 regresses"))

    await desk.void_from("b01", "parent b01 stopped: flat")

    assert [pending.branch for pending in desk.pending()] == ["b04"]
    await desk.void_all("time_exhausted")
    assert desk.pending() == ()
    roster.stop("b02", BranchStop(StopCause.JUDGE, "flat", CLOCK.now()))
    desk.fill_free_slots()
    assert desk.waiting() == () and [entry.branch for entry in roster.entries()] == ["b01", "b02"]
    voided = [row for row in _logged(tmp_path) if row["detail"]["status"] == "voided"]
    assert [(row["branch"], row["reason"]) for row in voided] == [
        ("b03", "voided: parent b01 stopped: flat"),
        ("b04", "voided: time_exhausted"),
    ]


@pytest.mark.parametrize("asks", ["fork", "repair"])
@pytest.mark.parametrize("branch", ["b01", "b09"], ids=["stopped", "unknown"])
async def test_a_request_of_a_stopped_or_unknown_branch_is_refused(tmp_path: Path, branch: str, asks: str) -> None:
    roster, desk = _desk(tmp_path, max_branches=3)
    roster.stop("b01", BranchStop(StopCause.JUDGE, "flat", CLOCK.now()))

    if asks == "fork":
        answer = await desk.admit(ForkRequest(branch, "c1", "fp8 fix regresses"))
    else:
        answer = await desk.admit_repair(_repair(branch))

    assert answer.status is AdmissionStatus.REJECTED
    assert answer.reason == ("parent branch is stopped" if branch == "b01" else f"unknown parent branch {branch!r}")
    assert [entry.branch for entry in roster.entries()] == ["b01"]
