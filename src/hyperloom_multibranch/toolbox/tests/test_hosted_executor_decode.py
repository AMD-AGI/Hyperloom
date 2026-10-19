# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The decode pass through the hosted executor: which tests carry it, what the result records, what reverify refuses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestResult, Validity
from hyperloom.measure.decode import OUTPUTS_PATH, read_decode_outputs
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.socket import TestCall
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.attempts import MeasurementRefused, TestJob
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.tests.helpers import BackendSpy, req

ACCURACY_PURPOSES = (Purpose.BASELINE, Purpose.EXPLORE, Purpose.CONFIRM, Purpose.INTEGRATION, Purpose.FINAL)


@pytest.fixture
def rig(tmp_path: Path):
    made: Rig = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


def _job(tmp_path: Path, *, purpose: Purpose = Purpose.EXPLORE, points: int = 1, **fake: object) -> TestJob:
    paths = tuple(point_yaml(tmp_path, f"p{i}.yaml", **fake) for i in range(points))
    request = req(purpose=purpose, accuracy=True, retest_of="b1-000000" if purpose is Purpose.CONFIRM else None)
    return TestJob(request.accepted("b1-000001"), 1, (), 2, 2, AcceptedMeasurement(None, paths))  # type: ignore[arg-type]


async def _run(tmp_path: Path, rig: Rig, **kwargs: object):  # type: ignore[no-untyped-def]
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    fake = {"fake_accuracy": "0.6", "fake_decode": "true", **kwargs}
    return executor, await executor.run(_job(tmp_path, **fake))  # type: ignore[arg-type]


@pytest.mark.parametrize("purpose", ACCURACY_PURPOSES)
async def test_every_purpose_that_carries_accuracy_asks_for_the_pass_and_records_what_the_bundle_holds(
    tmp_path: Path, rig: Rig, purpose: Purpose
) -> None:
    spy = BackendSpy(rig.backend)
    result = await HostedExecutor(spy, tmp_path / "session").run(  # type: ignore[arg-type]
        _job(tmp_path, purpose=purpose, fake_accuracy="0.6", fake_decode="true")
    )

    assert spy.calls[0].decode_pass is True
    assert result.decode_outputs == OUTPUTS_PATH and result.decode_identity.startswith("sha256:")
    assert read_decode_outputs(result, tmp_path / "session") == ((1, 2), (3,))


@pytest.mark.parametrize(
    ("purpose", "points", "workload"),
    [
        (Purpose.HEALTH, 1, WorkloadKind.INFERENCEX),
        (Purpose.EXPLORE, 2, WorkloadKind.INFERENCEX),
        (Purpose.EXPLORE, 1, WorkloadKind.AGENTX),
    ],
    ids=["other-purpose", "sweep", "agentx"],
)
def test_a_test_that_is_not_an_accuracy_point_of_an_inferencex_session_does_not_ask_for_it(
    tmp_path: Path, rig: Rig, purpose: Purpose, points: int, workload: WorkloadKind
) -> None:
    job = _job(tmp_path, purpose=purpose, points=points, fake_accuracy="0.6")
    executor = HostedExecutor(rig.backend, tmp_path / "session", workload_kind=workload)

    call = executor._call(job, job.environment, tmp_path / "out")  # type: ignore[arg-type]

    assert call.decode_pass is False and call.accuracy is True


async def test_a_result_with_no_pass_records_none_and_a_failed_pass_is_only_a_warning(tmp_path: Path, rig: Rig) -> None:
    _, result = await _run(tmp_path, rig, fake_decode="false", fake_warnings="['decode_pass_failed:http_500']")

    assert result.validity is Validity.SUCCEEDED and result.rounds, "the measurement is intact"
    assert (result.decode_outputs, result.decode_identity) == ("", "")
    assert result.nonfatal_warnings == ("decode_pass_failed:http_500",)


async def test_a_result_with_a_pass_verifies_again_from_its_bundle(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _run(tmp_path, rig)

    await executor.reverify(result)  # does not raise
    assert TestResult.from_document(result.to_document()) == result


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda r: replace(r, decode_identity="sha256:other|temperature=0|top_k=1"), "decode_identity: recorded"),
        (lambda r: replace(r, decode_outputs="decode/elsewhere.jsonl"), "decode_outputs: recorded"),
        (lambda r: replace(r, decode_outputs="", decode_identity=""), "decode_outputs: recorded ''"),
    ],
    ids=["identity", "path", "stripped"],
)
async def test_a_record_that_does_not_follow_the_signed_pass_is_not_verified(
    tmp_path: Path,
    rig: Rig,
    change,
    match,  # type: ignore[no-untyped-def]
) -> None:
    executor, result = await _run(tmp_path, rig)

    with pytest.raises(ResultUnverified, match=match):
        await executor.reverify(change(result))


async def test_a_pass_forged_onto_a_bundle_that_holds_none_is_not_verified(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _run(tmp_path, rig)
    plain_executor, plain = await _run(tmp_path, rig, fake_decode="false")
    forged = replace(plain, decode_outputs=OUTPUTS_PATH, decode_identity=result.decode_identity)

    with pytest.raises(ResultUnverified, match="the bundle lists no decode/outputs.jsonl"):
        await plain_executor.reverify(forged)


async def test_an_output_edited_in_the_archived_bundle_is_caught_by_the_signature(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _run(tmp_path, rig)
    (edited,) = (tmp_path / "archive").rglob("outputs.jsonl")
    edited.chmod(0o644)
    edited.write_text(edited.read_text().replace('"tokens": [1, 2]', '"tokens": [1, 9]'))

    with pytest.raises(ResultUnverified, match="rejected"):
        await executor.reverify(result)


async def test_an_unreadable_pass_the_sandbox_signed_is_refused_not_recorded(tmp_path: Path, rig: Rig) -> None:
    from hyperloom_multibranch.toolbox import hosted_executor

    executor = HostedExecutor(rig.backend, tmp_path / "session")
    call = TestCall(
        points=(), output_dir=tmp_path / "landed", timeout_sec=1, cwd=str(tmp_path), magpie_python="", task_id="t"
    )
    (tmp_path / "landed" / "decode").mkdir(parents=True)
    (tmp_path / "landed" / OUTPUTS_PATH).write_text("garbage\n")

    with pytest.raises(MeasurementRefused, match="unreadable decode pass"):
        hosted_executor._landed_decode(call)
    assert executor is not None
