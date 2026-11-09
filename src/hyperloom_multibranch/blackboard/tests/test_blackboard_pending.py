# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A measured candidate waiting for its confirm, as its branch's progress row shows it to the frontier and the judge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom.contracts.measurement import RoundValue
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role
from hyperloom_multibranch.contracts.blackboard.peers import Purpose, TestRequest, TestResult
from hyperloom_multibranch.contracts.blackboard.records import (
    BaselineRecord,
    LineageEvent,
    PendingCandidate,
    Withdrawal,
)
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import intent, judgement, profile, request, result

Open = Callable[..., LocalBlackboard]
METRIC = "output_tps"
TRIED = intent("s1", "c0")


def _measured(request_id: str, value: float) -> TestResult:
    rounds = (RoundValue("warmup", 64, {METRIC: 1.0}), RoundValue("measure", 64, {METRIC: value}))
    return replace(result(request_id), rounds=rounds)


def _confirm(request_id: str, of: str) -> TestRequest:
    return replace(request(request_id, TRIED), purpose=Purpose.CONFIRM, retest_of=of)


@pytest.fixture
async def session(open_as: Open) -> Open:
    """The root ``c0`` baselined at 1000 with a bar of 1.6%, and ``s1`` open on it."""

    scheduler = open_as(Role.SCHEDULER)
    await scheduler.write_session_profile(profile())
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    baseline = BaselineRecord("c0", {METRIC: 1000.0}, {METRIC: 0.8}, {METRIC: 1.6}, ("b1", "b2"))
    await open_as(Role.GATE).write_baseline(baseline)
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    return open_as


async def _pending(open_as: Open) -> tuple[PendingCandidate, ...]:
    rows = {row.branch: row for row in await open_as(Role.OBSERVER).progress()}
    return rows["s1"].pending


async def test_a_measured_gain_waiting_for_its_confirm_shows_on_its_branch_until_the_confirm_ends(
    session: Open,
) -> None:
    toolbox = session(Role.TOOLBOX)
    await toolbox.write_request(request("r1", TRIED))
    await toolbox.write_result(_measured("r1", 1050.0))
    unconfirmed = await _pending(session)
    await toolbox.write_request(_confirm("r1c", "r1"))
    waiting = await _pending(session)
    await toolbox.write_result(_measured("r1c", 1048.0))

    assert unconfirmed == ()
    assert waiting == (PendingCandidate("r1", "r1c", "c0", METRIC, 1050.0, pytest.approx(5.0)),)
    assert await _pending(session) == ()


async def test_a_candidate_below_the_bar_with_its_confirm_withdrawn_or_off_the_head_is_not_pending(
    session: Open,
) -> None:
    toolbox = session(Role.TOOLBOX)
    for request_id, value in (("r1", 1010.0), ("r2", 1050.0), ("r3", 1050.0)):
        await toolbox.write_request(request(request_id, TRIED))
        await toolbox.write_result(_measured(request_id, value))
        await toolbox.write_request(_confirm(f"{request_id}c", request_id))
    await toolbox.write_withdrawal(Withdrawal("r2c", "branch stopped"))
    on_the_head = await _pending(session)
    await toolbox.write_request(request("r0", intent("s1", "c0", "attention", "fused rope")))
    await toolbox.write_result(_measured("r0", 1040.0))
    await session(Role.GATE).write_judgement("r0", judgement(4.0, ("r0",)), Conclusion.COMMITTED, "c1")

    assert [candidate.request_id for candidate in on_the_head] == ["r3"]
    assert await _pending(session) == ()
