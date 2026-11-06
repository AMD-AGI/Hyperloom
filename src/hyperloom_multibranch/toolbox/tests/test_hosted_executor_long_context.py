# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A succeeded test that ran the long-context check carries what its landed outputs say.

The executor is the real one over the hosted backend and the local sandbox provider; the stand-in runner leaves the outputs a
check whose server behaves as the point says would (``fake_long_context``) in the files the bundle signs. What is read is
what the bundle landed, and the record is verified against it again.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.long_context import LONG_CONTEXT_COMPLETE, LONG_CONTEXT_INCOMPLETE
from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.long_context.tests.support import LENGTH
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.mcp_tools import jsonable
from hyperloom_multibranch.toolbox.tests.helpers import build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import POLICY, PointPlanner

PLAN = LongContextPlan("long-context-test", "a" * 64, LENGTH, trigger=("request long_context_check",))


@pytest.fixture
def rig(tmp_path: Path):
    made = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


class _Planner(PointPlanner):
    """Plans every request to one point file and the long-context check ``check`` names."""

    def __init__(self, point: Path, check: LongContextPlan | None) -> None:
        super().__init__(point)
        self.check = check

    def plan(self, request, parent) -> AcceptedMeasurement:  # type: ignore[no-untyped-def]
        return replace(super().plan(request, parent), long_context=self.check)


class _Session:
    def __init__(self, tmp_path: Path, rig: Rig, check: LongContextPlan | None) -> None:
        self.point = tmp_path / "planned.yaml"
        self.executor = HostedExecutor(rig.backend, tmp_path / "session")
        self.tb = build(tmp_path, executor=self.executor, planner=_Planner(self.point, check), policy=POLICY)
        self.tmp_path, self.rig = tmp_path, rig

    async def run(self, behaviour: dict | None, **fake: object) -> TestResult:
        fake = {"fake_accuracy": "0.64", **fake, **({} if behaviour is None else {"fake_long_context": behaviour})}
        point_yaml(self.tmp_path, self.point.name, **fake)
        await self.tb.toolbox.start()
        request_id = await self.tb.toolbox.submit_test(req())
        await asyncio.wait_for(self.tb.toolbox.drain(), 60)
        entry = self.tb.toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None
        return TestResult.from_document(entry.outcome["result"])


async def test_a_succeeded_test_carries_the_check_of_the_outputs_its_bundle_landed(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run({"broken": ["words"], "served": 4096})

    assert result.validity is Validity.SUCCEEDED
    check = result.long_context
    assert check is not None and check.status == LONG_CONTEXT_COMPLETE and check.trigger == PLAN.trigger
    assert {t.task: t.score for t in check.tasks}["common_words"] == 0.0 and check.score == 0.75
    assert (check.served_context, check.length) == (4096, LENGTH)
    landed = tmp_path / "session" / result.output_dir / check.outputs
    assert landed.is_file()
    archived = ResultManifest.from_bytes((Path(rig.record_archive(result.request_id)) / "manifest.json").read_bytes())
    assert check.outputs in {entry.path for entry in archived.files}, "the signature covers the outputs"
    assert session.executor.difference(result) is None, "the record follows from its own bundle"


async def test_a_launch_that_served_too_little_is_recorded_incomplete_never_as_a_score(tmp_path: Path, rig) -> None:
    result = await _Session(tmp_path, rig, PLAN).run({"served": LENGTH})

    check = result.long_context
    assert check is not None and check.status == LONG_CONTEXT_INCOMPLETE and check.served_context == LENGTH
    assert check.passed == 0 and result.validity is Validity.SUCCEEDED, "the measurement is still a measurement"


async def test_an_edited_score_with_untouched_hashes_is_named(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)
    result = await session.run({"broken": ["niah", "tracking"]})
    check = result.long_context
    assert check is not None and check.score == 0.25
    tasks = tuple(replace(t, passed=t.n, score=1.0) for t in check.tasks)
    prompts = tuple(replace(p, passed=True) for p in check.prompts)
    edited = replace(result, long_context=replace(check, passed=64, score=1.0, tasks=tasks, prompts=prompts))

    assert edited.manifest_sha256 == result.manifest_sha256
    assert session.executor.difference(edited) == ("long_context.passed: recorded 64, the bundle's outputs give 16")


async def test_a_runner_that_left_no_outputs_gives_an_incomplete_check_with_no_score(tmp_path: Path, rig) -> None:
    result = await _Session(tmp_path, rig, PLAN).run(None)

    check = result.long_context
    assert check is not None and check.status == LONG_CONTEXT_INCOMPLETE and check.score is None and check.n == 0
    assert "no readable long_context/outputs.json" in check.reason


async def test_a_test_that_asked_for_no_check_carries_none(tmp_path: Path, rig) -> None:
    result = await _Session(tmp_path, rig, None).run({})

    assert result.validity is Validity.SUCCEEDED and result.long_context is None
    assert jsonable(result)["long_context"] is None


async def test_a_test_that_did_not_succeed_makes_no_claim_about_long_context(tmp_path: Path, rig) -> None:
    result = await _Session(tmp_path, rig, PLAN).run({}, fake_mode="fail_test")

    assert result.validity is not Validity.SUCCEEDED and result.long_context is None
