# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Results and bundles written while the greedy decode pass existed still load and verify again.

The pass is gone, but a blackboard keeps the results it recorded (``decode_outputs``, ``decode_identity``), the archive
keeps the bundles (they list ``decode/outputs.jsonl``) and the journal keeps the sheets (``procedure.decode_pass``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestResult, Validity
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.spec import MeasurementSpec, canonical_json
from hyperloom.measure.tests.fake_runner import LEGACY_DECODE_IDENTITY, LEGACY_DECODE_PATH
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.toolbox.attempts import TestJob
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.tests.helpers import req


@pytest.fixture
def rig(tmp_path: Path):
    made: Rig = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


def _job(tmp_path: Path, **fake: object) -> TestJob:
    path = point_yaml(tmp_path, "p0.yaml", **fake)
    request = req(purpose=Purpose.EXPLORE, accuracy=True)
    return TestJob(request.accepted("b1-000001"), 1, (), 2, 2, AcceptedMeasurement(None, (path,)))  # type: ignore[arg-type]


async def _run(tmp_path: Path, rig: Rig, **fake: object):  # type: ignore[no-untyped-def]
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    result = await executor.run(_job(tmp_path, fake_accuracy="0.6", **{"fake_legacy_decode": "true", **fake}))  # type: ignore[arg-type]
    return executor, result


def _as_recorded_then(result: TestResult) -> TestResult:
    """The result as the blackboard of that time persisted it: the document carried both decode keys."""
    document = {**result.to_document(), "decode_outputs": LEGACY_DECODE_PATH, "decode_identity": LEGACY_DECODE_IDENTITY}
    return TestResult.from_document(json.loads(json.dumps(document)))


async def test_a_bundle_that_lists_the_old_decode_outputs_verifies_again(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _run(tmp_path, rig)
    (listed,) = (tmp_path / "archive").rglob("outputs.jsonl")
    assert listed.parent.name == "decode", "the bundle holds the old outputs file"

    await executor.reverify(result)  # does not raise


async def test_a_result_recorded_with_the_decode_keys_loads_and_verifies_again(tmp_path: Path, rig: Rig) -> None:
    executor, result = await _run(tmp_path, rig)

    recorded = _as_recorded_then(result)

    assert recorded == result
    await executor.reverify(recorded)  # does not raise


async def test_an_output_edited_in_the_archived_bundle_is_still_caught_by_the_signature(
    tmp_path: Path, rig: Rig
) -> None:
    executor, result = await _run(tmp_path, rig)
    (edited,) = (tmp_path / "archive").rglob("outputs.jsonl")
    edited.chmod(0o644)
    edited.write_text(edited.read_text().replace('"tokens": [1, 2]', '"tokens": [1, 9]'))

    with pytest.raises(ResultUnverified, match="rejected"):
        await executor.reverify(_as_recorded_then(result))


async def test_a_sheet_that_carries_decode_pass_is_read_again_when_its_result_is_verified(
    tmp_path: Path, rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    keep = MeasurementSpec.to_bytes

    def written_then(self: MeasurementSpec) -> bytes:
        document = json.loads(keep(self))
        document["procedure"]["decode_pass"] = True
        return canonical_json(document).encode()

    monkeypatch.setattr(MeasurementSpec, "to_bytes", written_then)
    # a runner that cannot read the sheet never answers: the short limit ends the attempt instead of the test hanging
    executor = HostedExecutor(rig.backend, tmp_path / "session", timeout_sec=30)
    result = await executor.run(_job(tmp_path, fake_accuracy="0.6"))  # type: ignore[arg-type]
    monkeypatch.undo()
    assert result.validity is Validity.SUCCEEDED, result.failure_reason
    kept = rig.backend.journal.load_spec(result.request_id, rig.backend.journal.latest(result.request_id).attempt)
    assert json.loads(kept)["procedure"]["decode_pass"] is True, "the journal holds the old sheet"

    await executor.reverify(result)  # does not raise
