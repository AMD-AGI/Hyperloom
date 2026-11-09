# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Eval points: unobserved they only forward; observed they record every call, inner units included, and an
evaluator may answer a call in a unit's place."""

from __future__ import annotations

import pytest

from hyperloom.brain_api import (
    BrainCall,
    BrainContractError,
    CallSite,
    Candidate,
    ConfigDelta,
    EvalPoint,
    Proposal,
    ProposeAsk,
    Review,
    ReviewAsk,
    SearchState,
    Workload,
    eval_point,
    observing,
    point_of,
    points_of,
)

_STATE = SearchState(Workload("m", "sglang"), baseline_tput=1.0, current_tput=1.0, keep_threshold_pct=1.0)
_ASK = ProposeAsk(domain="kv")


class _Unit:
    @eval_point("specialist.kv.propose", version="0.2")
    async def propose(self, state: SearchState, ask: ProposeAsk, ports: object) -> Proposal:
        if ask.task == "boom":
            raise RuntimeError("unit failed")
        return Proposal((Candidate("kv-fp8", ConfigDelta("--kv-cache-dtype fp8")),))


class _Brain:
    name = "fake"

    def __init__(self) -> None:
        self.unit = _Unit()

    @eval_point("main.propose")
    async def propose(self, state: SearchState, ask: ProposeAsk, ports: object) -> Proposal:
        return await self.unit.propose(state, ask, ports)

    @eval_point("main.review")
    async def review(self, state: SearchState, ask: ReviewAsk, ports: object) -> Review:
        return Review()

    def eval_points(self) -> dict:
        return {"specialist.kv.propose": self.unit.propose}


class _Recorder:
    def __init__(self, answers: dict | None = None) -> None:
        self.calls: list[BrainCall] = []
        self.answers = answers or {}

    async def replace(self, point: EvalPoint, state: SearchState, ask: object) -> Proposal | Review | None:
        return self.answers.get(point.name)

    async def observe(self, call: BrainCall) -> None:
        self.calls.append(call)


async def test_an_unobserved_point_only_forwards() -> None:
    proposal = await _Brain().propose(_STATE, _ASK, None)
    assert [c.name for c in proposal.candidates] == ["kv-fp8"]
    assert point_of(_Brain.propose) == EvalPoint("main.propose", "0.1", CallSite.PROPOSE)


async def test_an_observed_run_records_the_unit_then_the_main_point() -> None:
    recorder = _Recorder()
    with observing(recorder, brain="fake"):
        await _Brain().propose(_STATE, _ASK, None)
    await _Brain().propose(_STATE, _ASK, None)
    assert [(c.point, c.point_version, c.brain) for c in recorder.calls] == [
        ("specialist.kv.propose", "0.2", "fake"),
        ("main.propose", "0.1", "fake"),
    ]
    unit = recorder.calls[0]
    assert unit.answer is not None and unit.duration_sec >= 0.0
    assert BrainCall.from_wire(unit.to_wire()) == unit


async def test_a_failing_point_is_recorded_and_still_raises() -> None:
    recorder = _Recorder()
    with observing(recorder, brain="fake"), pytest.raises(RuntimeError, match="unit failed"):
        await _Brain().propose(_STATE, ProposeAsk(domain="kv", task="boom"), None)
    assert [(c.point, c.error) for c in recorder.calls] == [
        ("specialist.kv.propose", "RuntimeError: unit failed"),
        ("main.propose", "RuntimeError: unit failed"),
    ]


async def test_an_evaluator_can_answer_a_unit_in_its_place() -> None:
    fixed = Proposal((Candidate("fixed", ConfigDelta("--x")),))
    recorder = _Recorder({"specialist.kv.propose": fixed})
    with observing(recorder, brain="fake"):
        assert await _Brain().propose(_STATE, ProposeAsk(domain="kv", task="boom"), None) == fixed
    assert [c.point for c in recorder.calls] == ["main.propose"]
    with observing(_Recorder({"specialist.kv.propose": Review()}), brain="fake"), pytest.raises(TypeError):
        await _Brain().propose(_STATE, _ASK, None)


def test_points_of_lists_the_main_points_and_the_units() -> None:
    brain = _Brain()
    assert sorted(points_of(brain)) == ["main.propose", "main.review", "specialist.kv.propose"]

    class Unmarked(_Brain):
        def eval_points(self) -> dict:
            return {"specialist.kv.propose": self.review}

    with pytest.raises(BrainContractError, match="lists 'specialist.kv.propose'"):
        points_of(Unmarked())


@pytest.mark.parametrize(
    ("name", "version", "match"),
    [("Main.Propose", "0.1", "lower_snake_case"), ("propose", "0.1", "lower_snake_case"), ("a.b", "1", "major")],
)
def test_a_bad_name_or_version_is_refused(name, version, match) -> None:
    with pytest.raises(ValueError, match=match):
        eval_point(name, version=version)


def test_a_point_must_be_async_and_speak_a_contract_pair() -> None:
    with pytest.raises(TypeError, match="async"):

        @eval_point("a.b")
        def sync(self, state: SearchState, ask: ProposeAsk, ports: object) -> Proposal: ...

    with pytest.raises(TypeError, match="contract pair"):

        @eval_point("a.b")
        async def mixed(self, state: SearchState, ask: ProposeAsk, ports: object) -> Review: ...
