# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The branch requests the rule gate makes: an admitted fork or repair takes a slot at once and waits to be started;
past the cap it is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.fork import Admission, AdmissionStatus, ForkRequest, RepairRequest
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import FORK_K, ForkDesk
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
FIX = ForkRequest("c1", tagged("bounds-check"), "repro/moe-oob.py", "fp8 fix regresses")
BLOCKED = Judgement(Verdict.BLOCKED, 6.0, 2.0, ("r1", "r2"), {"output_throughput": 1060.0}, accuracy_ok=False)


def _repair(branch: str = "b01", mechanism: str = "fp8 kv cache") -> RepairRequest:
    intent = Intent(branch, "c1", "attention", mechanism, IntentLayer.CONFIG)
    return RepairRequest(intent, tagged("fp8-kv"), BLOCKED, "per-tensor KV scales; scale per head")


def _desk(tmp_path: Path, max_branches: int) -> tuple[Roster, ForkDesk]:
    roster = Roster(RosterCaps(max_branches, 2, 8), tmp_path / "MB-roster.json", CLOCK)
    roster.open(roster.next_branch_id(), origin=BranchOrigin.FIRST_BATCH, start_commit="c0", k=2)
    return roster, ForkDesk(roster, DecisionLog(tmp_path / "MB-decisions.jsonl", CLOCK))


def _logged(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]


async def test_an_accepted_fork_is_admitted_and_handed_on_once_with_the_fix_in_its_brief(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=2)

    answer = await desk.request_fork(FIX)

    assert answer == Admission(AdmissionStatus.ACCEPTED, FIX.reason)
    entry = roster.entry("b02")
    assert (entry.status, entry.origin, entry.from_branch, entry.k) == (
        BranchStatus.ADMITTED,
        BranchOrigin.FORK,
        "",
        FORK_K,
    )
    assert "b02" not in roster.quota_policy().branch_k
    [admitted] = desk.drain()
    assert (admitted.branch, admitted.start_commit) == ("b02", "c1")
    assert all(text in admitted.brief for text in (FIX.reason, FIX.repro_case, "bounds-check"))
    assert desk.drain() == ()
    [logged] = _logged(tmp_path)
    assert (logged["kind"], logged["branch"], logged["detail"]["status"]) == ("fork", "b02", "accepted")


async def test_an_accepted_repair_inherits_its_branch_k_and_starts_from_the_blocked_candidate(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=3)

    answer = await desk.request_repair(_repair())

    assert answer.status is AdmissionStatus.ACCEPTED
    entry = roster.entry("b02")
    assert (entry.origin, entry.from_branch, entry.k) == (BranchOrigin.REPAIR, "b01", 2)
    [admitted] = desk.drain()
    assert (admitted.branch, admitted.start_commit) == ("b02", "c1")
    assert all(text in admitted.brief for text in ("per-tensor KV scales", "fp8-kv", "+6.00%"))
    assert _logged(tmp_path)[-1]["kind"] == "repair"


async def test_an_intent_is_repaired_once_per_commit(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=4)

    first = await desk.request_repair(_repair())
    again = await desk.request_repair(_repair())
    other = await desk.request_repair(_repair(mechanism="bf16 kv cache"))

    assert [first.status, again.status, other.status] == [
        AdmissionStatus.ACCEPTED,
        AdmissionStatus.REJECTED,
        AdmissionStatus.ACCEPTED,
    ]
    assert "already open" in again.reason
    assert [admitted.branch for admitted in desk.drain()] == ["b02", "b03"]


async def test_a_request_is_refused_at_the_cap_and_once_the_session_stops(tmp_path: Path) -> None:
    roster, desk = _desk(tmp_path, max_branches=1)

    at_cap = await desk.request_fork(FIX)
    roster.close()
    closing = await desk.request_repair(_repair())

    assert (at_cap.status, at_cap.reason) == (AdmissionStatus.REJECTED, "all 1 branch slots are taken")
    assert (closing.status, closing.reason) == (AdmissionStatus.REJECTED, "the session is stopping")
    assert [entry.branch for entry in roster.entries()] == ["b01"]
    assert [row["detail"]["status"] for row in _logged(tmp_path)] == ["rejected", "rejected"]
    assert desk.drain() == ()


@pytest.mark.parametrize("branch", ["b01", "b09"], ids=["stopped", "unknown"])
async def test_a_repair_of_a_stopped_or_unknown_branch_is_refused(tmp_path: Path, branch: str) -> None:
    roster, desk = _desk(tmp_path, max_branches=3)
    roster.stop("b01", BranchStop(StopCause.JUDGE, "flat", CLOCK.now()))

    answer = await desk.request_repair(_repair(branch))

    assert answer.status is AdmissionStatus.REJECTED
    assert answer.reason == ("parent branch is stopped" if branch == "b01" else f"unknown parent branch {branch!r}")
    assert [entry.branch for entry in roster.entries()] == ["b01"]
