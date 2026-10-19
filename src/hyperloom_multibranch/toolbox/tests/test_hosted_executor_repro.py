# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A repro test through the hosted executor: a verified verdict is its result, it measures nothing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, RoundValue, TestKind, TestResult, Validity
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom.measure.tests.repro_fixtures import case_attachment
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.toolbox.attempts import MeasurementRefused, TestJob
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.tests.helpers import BackendSpy, req


@pytest.fixture
def rig(tmp_path: Path):
    made: Rig = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


def _job(tmp_path: Path, rig: Rig, script: str, *, kind: TestKind = TestKind.REPRO, **fake: object) -> TestJob:
    point = point_yaml(tmp_path, "planned.yaml", **fake)
    case = case_attachment(rig.config.spec.blobs, script)
    purpose = Purpose.REPRO if kind is TestKind.REPRO else Purpose.EXPLORE
    request = req(kind=kind, purpose=purpose, attachments=(case,) if kind is TestKind.REPRO else ())
    return TestJob(request.accepted("b1-000001"), 1, (), 2, 2, AcceptedMeasurement(None, (point,)))  # type: ignore[arg-type]


async def _repro(tmp_path: Path, rig: Rig, script: str = "exit 1\n", **fake: object):  # type: ignore[no-untyped-def]
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    fake.setdefault("fake_repro", "true")
    return executor, await executor.run(_job(tmp_path, rig, script, **fake))


@pytest.mark.parametrize(("script", "verdict"), [("exit 0\n", "not_reproduced"), ("exit 1\n", "reproduced")])
async def test_a_repro_test_with_a_signed_verdict_is_a_result_with_the_verdict_and_no_measurement(
    tmp_path: Path, rig: Rig, script: str, verdict: str
) -> None:
    _, result = await _repro(tmp_path, rig, script, fake_report="false")

    assert result.validity is Validity.SUCCEEDED and result.manifest_sha256
    assert result.repro_verdict == verdict
    assert result.rounds == () and result.accuracy == () and result.profile_location == ""
    log = tmp_path / "session" / result.output_dir / "repro" / "case.log"
    assert log.is_file(), "the case's log tail lands in the session with the verdict"


async def test_the_numbers_a_repro_server_printed_are_not_reported(tmp_path: Path, rig: Rig) -> None:
    _, result = await _repro(tmp_path, rig, "exit 0\n")  # the fake also writes a throughput report

    assert result.rounds == ()


async def test_the_case_reaches_the_sandbox_as_an_attachment_and_runs_there_with_no_warmup(
    tmp_path: Path, rig: Rig
) -> None:
    spy = BackendSpy(rig.backend)
    job = _job(tmp_path, rig, "echo from-the-attachment\nexit 1\n", fake_repro="true")
    result = await HostedExecutor(spy, tmp_path / "session").run(job)  # type: ignore[arg-type]

    (call,) = spy.calls
    assert call.kind == "repro" and call.rounds.warmup_rounds == 0 and call.attachments == job.request.attachments
    assert (tmp_path / "session" / result.output_dir / "repro" / "case.log").read_text() == "from-the-attachment\n"


async def test_a_repro_result_that_holds_no_verdict_is_refused(tmp_path: Path, rig: Rig) -> None:
    executor = HostedExecutor(rig.backend, tmp_path / "session")

    with pytest.raises(MeasurementRefused, match="holds no verdict"):
        await executor.run(_job(tmp_path, rig, "exit 0\n", fake_report="false"))  # the sandbox ran no case


async def test_a_plain_test_has_no_repro_verdict_and_ships_no_attachment(tmp_path: Path, rig: Rig) -> None:
    spy = BackendSpy(rig.backend)
    result = await HostedExecutor(spy, tmp_path / "session").run(  # type: ignore[arg-type]
        _job(tmp_path, rig, "", kind=TestKind.TEST)
    )

    assert result.repro_verdict == "" and spy.calls[0].attachments == ()


async def test_a_repro_result_verifies_again_from_its_bundle(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _repro(tmp_path, rig)

    await executor.reverify(result)  # does not raise


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda r: replace(r, repro_verdict="not_reproduced"), "repro_verdict: recorded 'not_reproduced'"),
        (lambda r: replace(r, repro_verdict=""), "repro_verdict: recorded ''"),
        (
            lambda r: replace(r, rounds=(RoundValue("measure", None, {"output_throughput": 1.0}),)),
            "records no measured",
        ),
    ],
)
async def test_a_repro_record_that_does_not_follow_its_bundle_is_not_verified(
    tmp_path: Path, rig: Rig, change, match
) -> None:  # type: ignore[no-untyped-def]
    executor, result = await _repro(tmp_path, rig, "exit 1\n", fake_report="false")  # the bundle says reproduced

    with pytest.raises(ResultUnverified, match=match):
        await executor.reverify(change(result))


async def test_a_verdict_forged_onto_a_bundle_with_no_verdict_file_is_not_verified(tmp_path: Path, rig: Rig) -> None:
    executor, _ = await _repro(tmp_path, rig)
    plain = await executor.run(_job(tmp_path, rig, "", kind=TestKind.TEST, fake_report="true"))
    forged: TestResult = replace(plain, repro_verdict="not_reproduced", rounds=())

    with pytest.raises(ResultUnverified, match="lists no verdict file"):
        await executor.reverify(forged)
