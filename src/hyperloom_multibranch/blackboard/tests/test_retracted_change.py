# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The change a landed retraction undoes: a commit that holds it and not the retraction reaches no best gain, and the
tests that brought it in say it was retracted."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role
from hyperloom_multibranch.contracts.blackboard.peers import TestRequest, Verdict
from hyperloom_multibranch.contracts.blackboard.records import (
    BaselineRecord,
    BlackboardSnapshot,
    LineageEvent,
    MergeAttempt,
    RetractionRef,
)
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.knowledge.experience import candidate_row
from hyperloom_multibranch.blackboard.knowledge.lessons import LessonKind, lesson_of
from hyperloom_multibranch.blackboard.rsi.archive import tree_from_snapshot
from hyperloom_multibranch.blackboard.tests.factories import intent, judgement, profile, request, result

Open = Callable[..., LocalBlackboard]
METRIC = "output_tps"
RETRACTS = RetractionRef(iteration=7, keep=1, finding="iter7-selection-invalid", retracted_commit="c1")
SAID = "later retracted by c3: c1, the agent's iteration 7 (its keep 1), found invalid by iter7-selection-invalid"


def _imported(request_id: str, parent: str) -> TestRequest:
    """An autonomous branch's import: no intent."""
    return replace(request(request_id, intent("a1", parent)), intent_fingerprint=None)


def _merge(request_id: str, parent: str, merged: str) -> TestRequest:
    """The integration branch's test of a merge of ``merged`` onto ``parent``."""
    return request(request_id, intent(INTEGRATION_BRANCH, parent, "merge", merged))


async def _judged(open_as: Open, request_id: str, gain: float, value: float, conclusion: Conclusion, **made):
    """Measure the request written as ``request_id`` and judge it at ``gain``, its candidate at ``value``."""
    writer = open_as(Role.INTEGRATOR if conclusion is Conclusion.MERGED else Role.GATE)
    await open_as(Role.TOOLBOX).write_result(result(request_id))
    verdict = Verdict.IMPROVED if gain > 0 else Verdict.REGRESSED
    decided = judgement(gain, (request_id,), recorded={METRIC: value}, verdict=verdict)
    await writer.write_judgement(request_id, decided, conclusion, **made)


@pytest.fixture(params=[True, False], ids=["retraction", "ordinary-commit"])
async def session(request: pytest.FixtureRequest, open_as: Open) -> tuple[BlackboardSnapshot, bool]:
    """``a1`` imported ``c1`` at +6% on the root and ``c2`` at +1% on it, and integration merged ``c1`` as ``i1``;
    then ``r-c3`` landed ``c3`` on ``c2``: a retraction of ``c1``'s keep at -5% or, for comparison, an ordinary commit
    at +1%."""

    retracting = request.param
    scheduler = open_as(Role.SCHEDULER)
    await scheduler.write_session_profile(profile())
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await open_as(Role.GATE).write_baseline(
        BaselineRecord("c0", {METRIC: 1000.0}, {METRIC: 0.8}, {METRIC: 1.6}, ("b",))
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "a1", from_commit="c0"))
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(_imported("r-c1", "c0"))
    await _judged(open_as, "r-c1", 6.0, 1060.0, Conclusion.COMMITTED, commit_id="c1")
    await toolbox.write_request(_imported("r-c2", "c1"))
    await _judged(open_as, "r-c2", 1.0, 1070.0, Conclusion.COMMITTED, commit_id="c2")
    await toolbox.write_request(_merge("m1", "c0", "c1"))
    await open_as(Role.INTEGRATOR).record_merge_attempt(MergeAttempt(("c1",), "c0", request_id="m1"))
    await _judged(open_as, "m1", 5.8, 1058.0, Conclusion.MERGED, commit_id="i1")
    await toolbox.write_request(_imported("r-c3", "c2"))
    if retracting:
        await _judged(open_as, "r-c3", -5.0, 1020.0, Conclusion.COMMITTED, commit_id="c3", retracts=RETRACTS)
    else:
        await _judged(open_as, "r-c3", 1.0, 1080.0, Conclusion.COMMITTED, commit_id="c3")
    return await open_as(Role.OBSERVER).snapshot(), retracting


def _record(snapshot: BlackboardSnapshot, request_id: str):
    return next(record for record in snapshot.records if record.request_id == request_id)


async def test_the_commits_that_hold_the_retracted_change_without_its_retraction(session) -> None:
    snapshot, retracting = session

    held = {commit_id: [item.request_id for item in found] for commit_id, found in snapshot.unretracted().items()}

    assert held == ({"c1": ["r-c3"], "c2": ["r-c3"], "i1": ["r-c3"]} if retracting else {})


async def test_a_commit_that_holds_a_retracted_change_reaches_no_best_gain_and_its_retraction_does(session) -> None:
    snapshot, retracting = session

    tree = tree_from_snapshot(snapshot, METRIC)

    assert tree.final_gain_pct == pytest.approx(2.0 if retracting else 8.0)


async def test_the_tests_that_brought_the_change_in_still_worked_and_name_its_retraction(session) -> None:
    snapshot, retracting = session

    lessons = {request_id: lesson_of(_record(snapshot, request_id), snapshot) for request_id in ("r-c1", "m1", "r-c2")}

    assert {request_id: lesson.kind for request_id, lesson in lessons.items()} == dict.fromkeys(
        lessons, LessonKind.WORKED
    )
    assert [request_id for request_id, lesson in lessons.items() if SAID in lesson.text] == (
        ["r-c1", "m1"] if retracting else []
    )
    assert "retracted" not in lessons["r-c2"].text


async def test_the_import_that_brought_the_change_in_is_still_a_keep_whose_reflection_names_its_retraction(
    session,
) -> None:
    snapshot, retracting = session

    brought, built = (
        candidate_row(_record(snapshot, name), snapshot, snapshot.profile, METRIC) for name in ("r-c1", "r-c2")
    )

    assert (brought["decision"], built["decision"]) == ("keep", "keep")
    assert (brought["learn"], brought["learn_source"]) == ((SAID, "retraction") if retracting else ("", ""))
    assert (built["learn"], built["learn_source"]) == ("", "")
