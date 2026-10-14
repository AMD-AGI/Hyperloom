# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted executor and the test kinds: accuracy values reach the result, and a sweep is one server arm (W13)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import AccuracyValue, SandboxFacts, TestKind, TestResult, Validity
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome
from hyperloom.measure.tests.fake_runner import REPORT_PATH
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.toolbox.attempts import MeasurementRefused, TestJob
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor, _round_values
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner

FLEX = "exact_match,flexible-extract"


@pytest.fixture
def rig(tmp_path: Path):
    made: Rig = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


async def _run_accuracy(tmp_path: Path, r: Rig, **fake: object) -> tuple[TestResult, TestRecord]:
    point = point_yaml(tmp_path, "planned.yaml", **fake)
    executor = HostedExecutor(r.backend, tmp_path / "session")
    tb = build(tmp_path, executor=executor, planner=PointPlanner(point), policy=QueuePolicy(1, 1))
    await tb.toolbox.start()
    request_id = await tb.toolbox.submit_test(req(kind=TestKind.ACCURACY, accuracy=True))
    await asyncio.wait_for(tb.toolbox.drain(), 60)
    entry = tb.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None
    return TestResult.from_document(entry.outcome["result"]), await tb.notebook.record(request_id)


async def test_an_accuracy_test_carries_the_signed_score_into_the_notebook_result(tmp_path: Path, rig) -> None:
    result, record = await _run_accuracy(tmp_path, rig, fake_accuracy="0.64")
    assert result.validity is Validity.SUCCEEDED
    assert result.accuracy == (AccuracyValue("gsm8k", FLEX, 0.64),)
    assert record.result is not None and record.result.accuracy == (AccuracyValue("gsm8k", FLEX, 0.64),)
    assert [r.phase for r in result.rounds] == ["measure"], "the accuracy round is not a throughput round"


async def test_the_score_version_reaches_the_result_and_the_notebook(tmp_path: Path, rig) -> None:
    version = "lm_eval=0.4.13;gsm8k=3"
    result, record = await _run_accuracy(tmp_path, rig, fake_accuracy="0.64", fake_accuracy_version=version)
    assert [v.version for v in result.accuracy] == [version]
    assert record.result is not None and [v.version for v in record.result.accuracy] == [version]
    assert TestResult.from_document(result.to_document()).accuracy[0].version == version


async def test_an_accuracy_test_whose_round_holds_no_score_is_not_a_result(tmp_path: Path, rig) -> None:
    point = point_yaml(tmp_path, "planned.yaml", fake_accuracy="null")
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    accepted = AcceptedMeasurement(None, (point,))  # type: ignore[arg-type]
    job = TestJob(req(kind=TestKind.ACCURACY, accuracy=True).accepted("b1-000001"), 1, (), 2, 2, accepted)
    with pytest.raises(MeasurementRefused, match="accuracy round holds no accuracy score"):
        await executor.run(job)


async def test_an_accuracy_request_whose_result_has_no_accuracy_round_is_not_a_result(tmp_path: Path, rig) -> None:
    point = point_yaml(tmp_path, "planned.yaml", fake_rounds="true")
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    accepted = AcceptedMeasurement(None, (point,))  # type: ignore[arg-type]
    job = TestJob(req(kind=TestKind.ACCURACY, accuracy=True).accepted("b1-000001"), 1, (), 2, 2, accepted)
    with pytest.raises(MeasurementRefused, match="has no accuracy round"):
        await executor.run(job)


async def test_a_plain_test_ignores_a_score_it_did_not_ask_for(tmp_path: Path, rig) -> None:
    point = point_yaml(tmp_path, "planned.yaml", fake_accuracy="0.64")
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    accepted = AcceptedMeasurement(None, (point,))  # type: ignore[arg-type]
    result = await executor.run(TestJob(req().accepted("b1-000001"), 1, (), 2, 2, accepted))
    assert result.validity is Validity.SUCCEEDED and result.accuracy == ()


# -- the sweep ---------------------------------------------------------------------------------------------------


class _Journal:
    def latest(self, request_id: str) -> None:
        return None

    def next_attempt(self, request_id: str) -> int:
        return 1


class _Capture:
    """Records the call and ends it as a candidate failure, so the executor returns without a verified bundle."""

    name = "hosted"

    def __init__(self) -> None:
        self.calls: list[TestCall] = []
        self.journal = _Journal()

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call)
        return TestOutcome(1, "", "", (), SandboxFacts("sbx"), call.task_id, "candidate", "why")  # type: ignore[arg-type]


def _points(tmp_path: Path, *levels: int) -> tuple[Path, ...]:
    out = []
    for level in levels:
        path = tmp_path / f"c{level}.yaml"
        path.write_text(f"benchmark:\n  envs:\n    CONC: '{level}'\n    RUN_EVAL: 'false'\n")
        out.append(path)
    return tuple(out)


@pytest.mark.parametrize(
    ("kind", "levels", "prep"),
    [
        (TestKind.TEST, (8,), "grid"),
        (TestKind.TEST, (64, 16, 4), "conc_sweep"),
        (TestKind.SWEEP, (8,), "conc_sweep"),
        (TestKind.SWEEP, (64, 16, 4), "conc_sweep"),
    ],
)
async def test_several_points_or_a_sweep_request_run_as_one_server_arm(tmp_path, kind, levels, prep) -> None:
    backend = _Capture()
    executor = HostedExecutor(backend, tmp_path)  # type: ignore[arg-type]
    accepted = AcceptedMeasurement(None, _points(tmp_path, *levels))  # type: ignore[arg-type]
    await executor.run(TestJob(req(kind=kind).accepted("b1-000001"), 1, (), 2, 2, accepted))
    (seen,) = backend.calls
    assert seen.prep == prep and seen.points == accepted.points and seen.kind == "test"


def _round(tmp_path: Path, name: str, throughput: float) -> Path:
    path = tmp_path / name / REPORT_PATH
    path.parent.mkdir(parents=True)
    report = {
        "success": True,
        "throughput": {"output_throughput": throughput, "completed_requests": 64, "num_prompts": 64},
    }
    path.write_text(json.dumps(report))
    return path.parent


def test_a_sweep_yields_one_measured_value_per_level_with_its_concurrency(tmp_path: Path) -> None:
    points = _points(tmp_path, 4, 16, 64)
    # the arm measures from the highest level down, so the rounds come in that order
    rounds = tuple(
        RoundOutcome(index, "measure", _round(tmp_path, f"r{index}", 1000.0 * (index + 1)), 0) for index in (2, 1, 0)
    )
    values = _round_values(rounds, points)
    assert [(v.phase, v.concurrency, v.metrics["output_throughput"]) for v in values] == [
        ("measure", 64, 3000.0),
        ("measure", 16, 2000.0),
        ("measure", 4, 1000.0),
    ]
