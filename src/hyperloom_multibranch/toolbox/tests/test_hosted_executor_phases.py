# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A hosted test says where its wall time went: the executor is the real one over the hosted backend and the local
sandbox provider, and the stand-in runner signs the phases of the sheet's ``fake_phases`` in its manifest.

What is read is what the blackboard stored: the phases of the host and of the runner in order, the card time each held, and
the seconds that none of them covers, so the parts add up to the wall time of the attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.mcp_tools import _result_document
from hyperloom_multibranch.toolbox.tests.helpers import build
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import POLICY, PointPlanner, _run

RUNNER_PHASES = [["bootstrap", 0.2], ["sheet_wait", 0.1], ["server_boot", 0.4], ["measured", 0.3]]
HOST_AFTER = ["runner_handoff", "result_landing", "sandbox_release", "result_read"]


@pytest.fixture
def rig(tmp_path: Path):
    made = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


async def _attempt(tmp_path: Path, r: Rig, *, timeout_sec: int = 60, **fake: object):
    point = point_yaml(tmp_path, "planned.yaml", **fake)
    executor = HostedExecutor(r.backend, tmp_path / "session", timeout_sec=timeout_sec)
    tb = build(
        tmp_path, executor=executor, planner=PointPlanner(point), policy=POLICY, workload_kind=WorkloadKind.AGENTX
    )
    request_id, result = await _run(tb)
    return tb, request_id, result


def _names(result: TestResult) -> list[str]:
    return [phase.name for phase in result.phases]


async def test_a_hosted_test_goes_through_its_phases_in_order_and_they_add_up_to_its_wall_time(
    tmp_path: Path, rig
) -> None:
    _, _, result = await _attempt(tmp_path, rig, fake_phases=json.dumps(RUNNER_PHASES), fake_delay=1.3)

    assert result.validity is Validity.SUCCEEDED
    assert _names(result) == ["sandbox_start", "bootstrap", "sheet_wait", "server_boot", "measured", *HOST_AFTER]
    wall = result.finished_at - result.started_at
    assert sum(p.seconds for p in result.phases) + result.unaccounted_seconds == pytest.approx(wall, abs=1e-6)
    assert 0.0 <= result.unaccounted_seconds < 0.5, (
        "the host's phases cover its whole call; what is left is a thread starting"
    )
    stated = {p.name: p.seconds for p in result.phases}
    assert [stated[name] for name in ("bootstrap", "sheet_wait", "server_boot", "measured")] == [0.2, 0.1, 0.4, 0.3]
    assert stated["runner_handoff"] > 0.2, "the runner stated 1.0 s of the 1.3 s it was waited for"
    started = [p.started_at for p in result.phases]
    assert started == sorted(started) and started[0] >= result.started_at


async def test_every_phase_holds_the_cards_the_test_holds_so_the_card_time_adds_up_too(tmp_path: Path, rig) -> None:
    _, _, result = await _attempt(tmp_path, rig, fake_phases=json.dumps(RUNNER_PHASES), fake_delay=1.3)

    assert result.sandbox.gpu_count == 1
    assert all(p.gpu_seconds == p.seconds * 1 for p in result.phases)
    held = sum(p.gpu_seconds for p in result.phases) + result.unaccounted_seconds * result.sandbox.gpu_count
    assert held == pytest.approx(result.gpu_seconds, abs=1e-6)


async def test_the_blackboard_keeps_the_phases_and_test_result_shows_them(tmp_path: Path, rig) -> None:
    tb, request_id, result = await _attempt(tmp_path, rig, fake_phases=json.dumps(RUNNER_PHASES), fake_delay=1.3)

    record = await tb.blackboard.record(request_id)
    assert record.result is not None and record.result.phases == result.phases
    shown = _result_document(request_id, record)["result"]
    assert [p["name"] for p in shown["phases"]] == _names(result)
    assert shown["unaccounted_seconds"] == result.unaccounted_seconds
    assert shown["phases"][3] == {**shown["phases"][3], "name": "server_boot", "seconds": 0.4, "gpu_seconds": 0.4}


async def test_a_test_that_failed_in_the_runner_has_the_phases_up_to_its_failure(tmp_path: Path, rig) -> None:
    phases = [["bootstrap", 0.2], ["sheet_wait", 0.1], ["server_boot", 0.4]]

    _, _, result = await _attempt(tmp_path, rig, fake_mode="fail_test", fake_phases=json.dumps(phases), fake_delay=0.8)

    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
    assert _names(result) == ["sandbox_start", "bootstrap", "sheet_wait", "server_boot", *HOST_AFTER]
    assert "measured" not in _names(result)
    wall = result.finished_at - result.started_at
    assert sum(p.seconds for p in result.phases) + result.unaccounted_seconds == pytest.approx(wall, abs=1e-6)


async def test_a_runner_that_stated_no_phases_is_one_wait_between_the_sandbox_and_the_result(
    tmp_path: Path, rig
) -> None:
    _, _, result = await _attempt(tmp_path, rig, fake_delay=0.5)

    assert _names(result) == ["sandbox_start", "runner_wait", "result_landing", "sandbox_release", "result_read"]
    assert {p.name: p.seconds for p in result.phases}["runner_wait"] >= 0.5


async def test_a_test_the_runner_reported_timed_out_keeps_the_phases_it_signed(tmp_path: Path, rig) -> None:
    phases = [["sheet_wait", 0.1], ["server_boot", 0.4]]

    _, _, result = await _attempt(tmp_path, rig, fake_mode="timed_out", fake_phases=json.dumps(phases), fake_delay=0.6)

    assert result.failure_class is FailureClass.UNCLEAR
    assert _names(result) == [
        "sandbox_start",
        "sheet_wait",
        "server_boot",
        "runner_handoff",
        "result_landing",
        "sandbox_release",
    ]
    wall = result.finished_at - result.started_at
    assert sum(p.seconds for p in result.phases) + result.unaccounted_seconds == pytest.approx(wall, abs=1e-6)


async def test_a_test_that_outlived_the_host_wait_has_what_the_host_saw_of_it(tmp_path: Path, rig) -> None:
    _, _, result = await _attempt(tmp_path, rig, timeout_sec=1, fake_mode="hang")

    assert result.failure_class is FailureClass.UNCLEAR
    assert _names(result) == ["sandbox_start", "runner_wait", "result_landing", "sandbox_release"]
    stated = {p.name: p.seconds for p in result.phases}
    assert 0.9 <= stated["runner_wait"] < 1.3, (
        "the host waited for the runner until its time ran out, not for the close"
    )
    assert result.sandbox.gpu_count == 0 and all(p.gpu_seconds == 0 for p in result.phases)
