# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A profile test through the hosted executor: a verified trace is its result, its numbers are not a measurement."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, RoundValue, TestKind, TestResult, Validity
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.toolbox.attempts import MeasurementRefused, TestJob
from hyperloom_multibranch.toolbox.hosted_executor import PROFILE_WITHOUT_STEPS, HostedExecutor
from hyperloom_multibranch.toolbox.tests.helpers import BackendSpy, req


_ROUND = RoundValue("measure", None, {"output_throughput": 1.0})


@pytest.fixture
def rig(tmp_path: Path):
    made: Rig = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


def _job(tmp_path: Path, kind: TestKind, **fake: object) -> TestJob:
    point = point_yaml(tmp_path, "planned.yaml", **fake)
    purpose = Purpose.PROFILE if kind is TestKind.PROFILE else Purpose.EXPLORE
    accepted = AcceptedMeasurement(None, (point,))  # type: ignore[arg-type]
    return TestJob(req(kind=kind, purpose=purpose).accepted("b1-000001"), 1, (), 2, 2, accepted)


async def _profile(tmp_path: Path, rig: Rig, **fake: object) -> tuple[HostedExecutor, TestResult]:
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    return executor, await executor.run(_job(tmp_path, TestKind.PROFILE, **fake))


async def test_a_profile_test_with_a_verified_trace_is_a_result_with_its_location_and_no_rounds(
    tmp_path: Path, rig: Rig
) -> None:
    _, result = await _profile(tmp_path, rig, fake_trace="true", fake_report="false")

    assert result.validity is Validity.SUCCEEDED and result.manifest_sha256
    assert result.rounds == () and result.accuracy == (), "a profiled round is not a measurement"
    assert result.profile_location == f"{result.output_dir}/profile"
    trace = tmp_path / "session" / result.profile_location / "torch_trace" / "rank0.pt.trace.json.gz"
    assert trace.is_file(), "the location resolves under the session directory, where the analysis reads it"


async def test_the_numbers_of_the_profiled_round_are_not_reported_even_when_it_printed_a_report(
    tmp_path: Path, rig: Rig
) -> None:
    _, result = await _profile(tmp_path, rig, fake_trace="true")  # the fake also writes a throughput report

    assert result.rounds == ()


async def test_a_profile_test_runs_one_round_with_no_warmup(tmp_path: Path, rig: Rig) -> None:
    spy = BackendSpy(rig.backend)
    await HostedExecutor(spy, tmp_path / "session").run(  # type: ignore[arg-type]
        _job(tmp_path, TestKind.PROFILE, fake_trace="true")
    )

    (call,) = spy.calls
    assert call.kind == "profile" and call.rounds.warmup_rounds == 0 and call.rounds.measure_rounds == 1


async def test_a_profile_result_that_holds_no_trace_is_refused(tmp_path: Path, rig: Rig) -> None:
    executor = HostedExecutor(rig.backend, tmp_path / "session")

    with pytest.raises(MeasurementRefused, match="holds no trace file"):
        await executor.run(_job(tmp_path, TestKind.PROFILE))


async def test_a_profile_whose_traces_are_all_graph_capture_sidecars_is_refused_with_the_reason(
    tmp_path: Path, rig: Rig
) -> None:
    """A NEXTN head's profile: the server captured its CUDA graphs at boot, then the run ended before the capture window,
    so ``graph_capture_profile/`` is all that landed. It is no result, and says why, rather than a success no analysis
    can read op shares from."""
    executor = HostedExecutor(rig.backend, tmp_path / "session")

    with pytest.raises(MeasurementRefused) as refused:
        await executor.run(_job(tmp_path, TestKind.PROFILE, fake_trace="capture_only", fake_report="false"))
    assert str(refused.value) == PROFILE_WITHOUT_STEPS


async def test_a_plain_test_without_a_measured_round_is_still_refused(tmp_path: Path, rig: Rig) -> None:
    executor = HostedExecutor(rig.backend, tmp_path / "session")

    with pytest.raises(MeasurementRefused, match="no readable measured round"):
        await executor.run(_job(tmp_path, TestKind.TEST, fake_trace="true", fake_report="false"))


async def test_a_plain_test_has_no_profile_location_and_the_default_warmup_plan(tmp_path: Path, rig: Rig) -> None:
    spy = BackendSpy(rig.backend)
    result = await HostedExecutor(spy, tmp_path / "session").run(  # type: ignore[arg-type]
        _job(tmp_path, TestKind.TEST, fake_trace="true")
    )

    assert result.profile_location == "" and spy.calls[0].rounds.warmup_rounds is None


async def test_a_profile_result_verifies_again_from_its_bundle(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _profile(tmp_path, rig, fake_trace="true", fake_report="false")

    await executor.reverify(result)  # does not raise


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda r: replace(r, profile_location="tests/elsewhere/profile"), "profile_location: recorded"),
        (lambda r: replace(r, rounds=(_ROUND,)), "records no measured value"),
    ],
)
async def test_a_profile_record_that_does_not_follow_its_bundle_is_not_verified(
    tmp_path: Path, rig: Rig, change, match
) -> None:  # type: ignore[no-untyped-def]
    executor, result = await _profile(tmp_path, rig, fake_trace="true", fake_report="false")

    with pytest.raises(ResultUnverified, match=match):
        await executor.reverify(change(result))


async def test_a_profile_record_whose_bundle_lists_no_trace_is_not_verified(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _profile(tmp_path, rig, fake_trace="true", fake_report="false")
    other = await executor.run(_job(tmp_path, TestKind.TEST, fake_report="true"))  # a bundle with no trace in it
    forged = replace(other, profile_location=f"{other.output_dir}/profile", rounds=())

    with pytest.raises(ResultUnverified, match="lists no trace file"):
        await executor.reverify(forged)
    await executor.reverify(result)  # the genuine one still verifies


async def test_a_profile_record_is_held_to_the_warnings_of_its_bundle_like_any_other(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _profile(tmp_path, rig, fake_trace="true", fake_report="false", fake_warnings='["w:1"]')
    assert result.nonfatal_warnings == ("w:1",)
    await executor.reverify(result)

    with pytest.raises(ResultUnverified, match=r"nonfatal_warnings: recorded \[\], the bundle has \['w:1'\]"):
        await executor.reverify(replace(result, nonfatal_warnings=()))
    with pytest.raises(ResultUnverified, match=r"nonfatal_warnings: recorded \['w:1', 'x'\]"):
        await executor.reverify(replace(result, nonfatal_warnings=("w:1", "x")))
