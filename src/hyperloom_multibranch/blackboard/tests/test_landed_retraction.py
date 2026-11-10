# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A retraction the gate landed: committed, it names what it undoes, and no reader takes it for a gain or progress."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom_multibranch.contracts.blackboard import Role
from hyperloom_multibranch.contracts.blackboard.peers import TestRequest, Verdict
from hyperloom_multibranch.contracts.blackboard.records import (
    BaselineRecord,
    BlackboardSnapshot,
    LineageEvent,
    RetractionRef,
)
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.knowledge.experience import candidate_row
from hyperloom_multibranch.blackboard.knowledge.lessons import LessonKind, lesson_of
from hyperloom_multibranch.blackboard.rsi.analysis import failures_at
from hyperloom_multibranch.blackboard.rsi.frontier import STALLED_COMPONENT, StallBar, StopRule, stall_evidence
from hyperloom_multibranch.blackboard.rsi.learning_eval import branch_tests
from hyperloom_multibranch.blackboard.tests.factories import intent, judgement, profile, request, result

Open = Callable[..., LocalBlackboard]
METRIC = "output_tps"
RETRACTS = RetractionRef(iteration=7, keep=1, finding="iter7-selection-invalid", retracted_commit="c1")


def _imported(request_id: str, parent: str) -> TestRequest:
    """An autonomous branch's import: no intent."""
    return replace(request(request_id, intent("s1", parent)), intent_fingerprint=None)


async def _judged(open_as: Open, tried: TestRequest, gain: float, verdict: Verdict, conclusion: Conclusion, **made):
    await open_as(Role.TOOLBOX).write_request(tried)
    await open_as(Role.TOOLBOX).write_result(result(tried.request_id))
    decided = judgement(gain, (tried.request_id,), recorded={METRIC: 1000.0 * (1 + gain / 100)}, verdict=verdict)
    await open_as(Role.GATE).write_judgement(tried.request_id, decided, conclusion, **made)


@pytest.fixture(params=[True, False], ids=["retraction", "ordinary-commit"])
async def session(request: pytest.FixtureRequest, open_as: Open) -> tuple[BlackboardSnapshot, bool]:
    """``s1`` committed ``c1`` on the root and failed ``miss`` on it; ``r-c2`` landed ``c2`` on ``c1``, a retraction of
    ``c1`` at -3% or, for comparison, an ordinary commit at +3%; ``after`` was judged on ``c2``."""

    retracting = request.param
    scheduler = open_as(Role.SCHEDULER)
    await scheduler.write_session_profile(profile())
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, "integration", commit_id="c0"))
    await open_as(Role.GATE).write_baseline(
        BaselineRecord("c0", {METRIC: 1000.0}, {METRIC: 0.8}, {METRIC: 1.6}, ("b",))
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    await _judged(open_as, _imported("r-c1", "c0"), 2.0, Verdict.IMPROVED, Conclusion.COMMITTED, commit_id="c1")
    await _judged(open_as, _imported("miss", "c1"), -2.0, Verdict.REGRESSED, Conclusion.RECORDED_ONLY)
    if retracting:
        made = {"commit_id": "c2", "retracts": RETRACTS}
        await _judged(open_as, _imported("r-c2", "c1"), -3.0, Verdict.REGRESSED, Conclusion.COMMITTED, **made)
    else:
        await _judged(open_as, _imported("r-c2", "c1"), 3.0, Verdict.IMPROVED, Conclusion.COMMITTED, commit_id="c2")
    await _judged(open_as, _imported("after", "c2"), 0.1, Verdict.FLAT, Conclusion.RECORDED_ONLY)
    return await open_as(Role.OBSERVER).snapshot(), retracting


def _record(snapshot: BlackboardSnapshot, request_id: str):
    return next(record for record in snapshot.records if record.request_id == request_id)


async def test_the_record_names_what_it_retracts(session) -> None:
    snapshot, retracting = session
    landed = _record(snapshot, "r-c2")

    assert (landed.conclusion, landed.commit_id) == (Conclusion.COMMITTED, "c2")
    assert (landed.retracts, landed.landed_retraction) == ((RETRACTS, True) if retracting else (None, False))
    assert [record.request_id for record in snapshot.records if record.landed_retraction] == (
        ["r-c2"] if retracting else []
    )


async def test_a_retraction_teaches_a_pitfall_naming_what_it_undid_not_a_worked_change(session) -> None:
    snapshot, retracting = session
    lesson = lesson_of(_record(snapshot, "r-c2"), snapshot)

    if retracting:
        assert lesson.kind is LessonKind.PITFALL
        assert (
            "retracts c1, the agent's iteration 7 (its keep 1), found invalid by iter7-selection-invalid" in lesson.text
        )
    else:
        assert lesson.kind is LessonKind.WORKED


async def test_a_retraction_is_not_published_as_an_experience(session) -> None:
    snapshot, retracting = session
    row = candidate_row(_record(snapshot, "r-c2"), snapshot, snapshot.profile, METRIC)

    assert (row is None) if retracting else row["decision"] == "keep"


async def test_a_retraction_is_no_committed_test_for_the_learning_evaluation(session) -> None:
    snapshot, retracting = session
    committed = [test.committed for test in branch_tests(snapshot)["s1"]]

    assert committed == [True, False, not retracting, False]


async def test_a_retraction_redeems_no_failure_on_the_state_it_was_made_on(session) -> None:
    snapshot, retracting = session

    failed = {failed.request_id for failed in failures_at(snapshot, "c2")}

    assert ("miss" in failed, "r-c2" in failed) == (retracting, retracting)


async def test_a_retraction_does_not_restart_the_stall_count(session) -> None:
    snapshot, retracting = session
    row = next(row for row in snapshot.progress if row.branch == "s1")
    stalled = stall_evidence(snapshot, row, METRIC, StallBar(StopRule(), 10), young=False)[STALLED_COMPONENT]

    assert (row.judged_since_commit, stalled) == ((3, 3.0) if retracting else (1, 1.0))


async def test_only_a_committed_judgement_names_what_it_retracts(open_as: Open) -> None:
    await open_as(Role.TOOLBOX).write_request(_imported("r1", "c0"))
    await open_as(Role.TOOLBOX).write_result(result("r1"))
    decided = judgement(-3.0, ("r1",), verdict=Verdict.REGRESSED)

    with pytest.raises(ValueError, match="only a committed retraction"):
        await open_as(Role.GATE).write_judgement("r1", decided, Conclusion.RECORDED_ONLY, retracts=RETRACTS)
