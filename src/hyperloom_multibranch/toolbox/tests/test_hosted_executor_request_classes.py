# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A succeeded AgentX test carries, for each measured round, how its failed requests split and where the export is (#117).

The executor is the real one over the hosted backend and the local sandbox provider. The stand-in runner writes the
round's per-request export (the lines of ``agentx_export``, in the shapes the pinned client writes) into the files the
bundle signs; what is read is what the bundle landed and the blackboard stored. The pointer is checked against the
archived manifest, which is the proof that the runner kept the export in the signed bundle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.tests import agentx_export as export
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.mcp_tools import jsonable
from hyperloom_multibranch.toolbox.tests.helpers import build
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import POLICY, PointPlanner, _run

EXPORT_IN_BUNDLE = "benchmark_fake_20261014_000000/aiperf_artifacts/profile_export.jsonl"


@pytest.fixture
def rig(tmp_path: Path):
    made = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


def _lines() -> list[str]:
    """Ten profiling requests (six served, three transport failures, one reply with nothing in it) and warm-up ones."""
    records = [export.served(n) for n in range(6)]
    records += [export.refused(6), export.unreachable(7), export.cancelled(8), export.no_content(9)]
    records += [export.refused(30, phase="warmup"), export.served(31, phase="warmup")]
    return export.text(*records)


async def _result(
    tmp_path: Path, r: Rig, workload: WorkloadKind = WorkloadKind.AGENTX, **fake: object
) -> tuple[HostedExecutor, TestResult]:
    point = point_yaml(tmp_path, "planned.yaml", **fake)
    executor = HostedExecutor(r.backend, tmp_path / "session", workload_kind=workload)
    tb = build(tmp_path, executor=executor, planner=PointPlanner(point), policy=POLICY, workload_kind=workload)
    _, result = await _run(tb)
    return executor, result


def _with_export(lines: list[str]) -> dict[str, object]:
    return {"fake_export": json.dumps(lines)}


async def test_a_result_counts_each_failed_request_by_class_and_the_blackboard_keeps_the_counts(
    tmp_path: Path, rig
) -> None:
    _, result = await _result(tmp_path, rig, **_with_export(_lines()))
    assert result.validity is Validity.SUCCEEDED
    (classes,) = result.request_classes
    assert (classes.point, classes.requests_total) == (0, 10)
    assert (classes.failed_total, classes.failed_transport, classes.failed_no_content) == (4, 3, 1)
    assert (classes.failed_empty, classes.failed_short) == (None, None), "the client as released cannot tell them"
    assert "cannot be told empty from short" in classes.reason


async def test_the_export_of_the_patched_client_splits_the_no_content_failures_exactly_and_reverifies(
    tmp_path: Path, rig
) -> None:
    records = [export.served(n) for n in range(6)] + [export.refused(6), export.cancelled(7)]
    records += [export.no_content(8, output_tokens=0), export.no_content(9, output_tokens=0)]
    records += [export.no_content(10, output_tokens=4)]
    executor, result = await _result(tmp_path, rig, **_with_export(export.text(*records)))
    (classes,) = result.request_classes
    assert (classes.requests_total, classes.failed_transport, classes.failed_no_content) == (11, 2, 3)
    assert (classes.failed_empty, classes.failed_short, classes.reason) == (2, 1, "")
    await executor.reverify(result)


async def test_the_runner_keeps_the_export_in_the_signed_bundle_and_the_result_points_at_it(
    tmp_path: Path, rig
) -> None:
    r = rig
    _, result = await _result(tmp_path, r, **_with_export(_lines()))
    (classes,) = result.request_classes
    assert classes.export == EXPORT_IN_BUNDLE, "relative to the test's output directory, as the manifest lists it"
    landed = tmp_path / "session" / result.output_dir / classes.export
    expected = "".join(f"{line}\n" for line in _lines()).encode()
    assert landed.read_bytes() == expected

    archived = ResultManifest.from_bytes((Path(r.record_archive(result.request_id)) / "manifest.json").read_bytes())
    listed = {entry.path: entry for entry in archived.files}
    assert listed[classes.export].sha256 == hashlib.sha256(expected).hexdigest(), "the signature covers the export"
    assert (Path(r.record_archive(result.request_id)) / classes.export).read_bytes() == expected


async def test_test_result_shows_the_counts_and_the_pointer_to_the_export(tmp_path: Path, rig) -> None:
    _, result = await _result(tmp_path, rig, **_with_export(_lines()))
    (shown,) = jsonable(result)["request_classes"]
    assert shown["export"] == EXPORT_IN_BUNDLE
    assert {name: shown[name] for name in ("requests_total", "failed_transport", "failed_no_content")} == {
        "requests_total": 10,
        "failed_transport": 3,
        "failed_no_content": 1,
    }
    assert shown["failed_empty"] is None and shown["failed_short"] is None


async def test_a_round_the_runner_left_no_export_for_has_an_explicit_no_count_not_zeros(tmp_path: Path, rig) -> None:
    _, result = await _result(tmp_path, rig)
    assert result.validity is Validity.SUCCEEDED, "a test without the export is still a measurement"
    (classes,) = result.request_classes
    assert (classes.export, classes.requests_total, classes.failed_total) == ("", None, None)
    assert classes.reason == "the round left no per-request export"


async def test_a_truncated_export_is_a_result_that_says_its_counts_cannot_be_trusted(tmp_path: Path, rig) -> None:
    lines = [*_lines(), '{"metadata": {"benchmark_phase": "profiling"']
    _, result = await _result(tmp_path, rig, **_with_export(lines))
    (classes,) = result.request_classes
    assert classes.failed_total is None and classes.export == EXPORT_IN_BUNDLE
    assert classes.reason.startswith("1 line(s) of the export are not a request record")


async def test_only_an_agentx_campaigns_test_carries_classes(tmp_path: Path, rig) -> None:
    # an InferenceX test asks for accuracy, which the stand-in runner then scores
    _, result = await _result(tmp_path, rig, WorkloadKind.INFERENCEX, fake_accuracy="0.64", **_with_export(_lines()))
    assert result.validity is Validity.SUCCEEDED and result.request_classes == ()


async def test_a_test_that_did_not_succeed_makes_no_claim_about_its_requests(tmp_path: Path, rig) -> None:
    _, result = await _result(tmp_path, rig, fake_mode="fail_test", **_with_export(_lines()))
    assert result.validity is not Validity.SUCCEEDED and result.request_classes == ()


async def test_a_recorded_result_is_reverified_against_the_exports_of_its_bundle(tmp_path: Path, rig) -> None:
    executor, result = await _result(tmp_path, rig, **_with_export(_lines()))
    assert executor.difference(result) is None


async def test_an_edited_count_with_untouched_hashes_is_named(tmp_path: Path, rig) -> None:
    executor, result = await _result(tmp_path, rig, **_with_export(_lines()))
    (classes,) = result.request_classes
    lighter = replace(classes, failed_total=3, failed_transport=2)  # still self-consistent, and one request short
    edited = replace(result, request_classes=(lighter,))
    assert (edited.manifest_sha256, edited.signing_key_fingerprint) == (
        result.manifest_sha256,
        result.signing_key_fingerprint,
    )
    assert executor.difference(edited) == "request_classes[0].failed_total: recorded 3, the bundle has 4"
    moved = replace(result, request_classes=(replace(classes, export="elsewhere/profile_export.jsonl"),))
    assert executor.difference(moved) == (
        f"request_classes[0].export: recorded 'elsewhere/profile_export.jsonl', the bundle has {EXPORT_IN_BUNDLE!r}"
    )
    extra = replace(result, request_classes=(classes, replace(classes, point=1)))
    assert executor.difference(extra) == "request_classes: recorded 2, the bundle has 1"


async def test_a_record_written_before_the_classes_existed_still_reverifies(tmp_path: Path, rig) -> None:
    executor, result = await _result(tmp_path, rig, **_with_export(_lines()))
    document = result.to_document()
    del document["request_classes"]
    assert executor.difference(TestResult.from_document(document)) is None, "it claims nothing, so nothing is wrong"
