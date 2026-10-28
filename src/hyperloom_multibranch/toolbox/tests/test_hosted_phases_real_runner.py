# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the time of a hosted test goes, from the shipped runner to the result the blackboard stores.

The hosted executor runs the real hosted backend over the local sandbox provider, whose runner is the shipped one with the
real local backend; only ``run_grid`` is a stand-in (``phased_runner_entry``), which spends a scaled-down share of the
seconds a real round spends and leaves the marks the launcher leaves. A graded InferenceX test, an AgentX test and a
test whose server never got ready each go through the whole path, and each result lists its phases in the order they
happened, ending in the host's, with the seconds none of them covers kept apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestKind, TestResult, Validity
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests.test_hosted_real_runner import _rigs
from hyperloom_multibranch.toolbox.attempts import TestJob
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.tests.helpers import req

PHASED_RUNNER = (sys.executable, "-m", "hyperloom.measure.tests.phased_runner_entry")
AFTER_THE_RUNNER = ["result_signing", "runner_handoff", "result_landing", "sandbox_release", "result_read"]
BEFORE_THE_ROUNDS = ["sandbox_start", "sheet_wait", "preflight", "env_rebuild"]


@pytest.fixture
def phased(tmp_path: Path):
    yield from _rigs(tmp_path, PHASED_RUNNER)


def _point(tmp_path: Path, *, run_eval: str = "false", agentx: bool = False) -> Path:
    lines = ["benchmark:", "  model: /nonexistent/model", "  framework: sglang"]
    lines += ["  agentx:", "    enabled: true"] if agentx else []
    lines += ["  envs:", "    TP: '1'", f"    RUN_EVAL: '{run_eval}'"]
    path = tmp_path / "phased.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


async def _attempt(tmp_path: Path, make, scenario: str, scale: float, point: Path, **fields: object) -> TestResult:
    r = make(envs={"HL_PHASE_SCENARIO": scenario, "HL_PHASE_SCALE": str(scale)})
    executor = HostedExecutor(r.backend, tmp_path / "session", timeout_sec=120)
    request = req(kind=TestKind.TEST, **fields).accepted("req-1")
    return await executor.run(TestJob(request, 1, (), 2, 1, AcceptedMeasurement(None, (point,))))  # type: ignore[arg-type]


def _in_order(expected: list[str], names: list[str]) -> bool:
    """Whether ``expected`` is in ``names``, in that order (the names between them are the ones not asked about)."""
    remaining = iter(names)
    return all(name in remaining for name in expected)


def _adds_up(result: TestResult) -> None:
    wall = result.finished_at - result.started_at
    assert sum(p.seconds for p in result.phases) + result.unaccounted_seconds == pytest.approx(wall, abs=1e-6)
    assert -0.001 <= result.unaccounted_seconds < 1.0, "what is left is the host's thread starting, not a phase"
    starts = [p.started_at for p in result.phases]
    assert starts == sorted(starts)


async def test_a_graded_inferencex_test_has_every_phase_of_its_path_in_order(tmp_path: Path, phased) -> None:
    point = _point(tmp_path, run_eval="true")

    result = await _attempt(tmp_path, phased, "graded", 0.004, point, accuracy=True)

    assert result.validity is Validity.SUCCEEDED
    names = [p.name for p in result.phases]
    assert _in_order(
        [
            *BEFORE_THE_ROUNDS,
            "round_setup",
            "server_boot.other",
            "server_boot.weights_load",
            "server_boot.compile",
            "server_boot.graph_capture",
            "warmup",
            "accuracy_eval",
            "round_collect",
            "round_setup",
            "measured",
            "teardown",
            "teardown",
            *AFTER_THE_RUNNER,
        ],
        names,
    ), names
    assert {p.point for p in result.phases if p.name in ("warmup", "accuracy_eval", "measured")} == {0}
    _adds_up(result)


async def test_an_agentx_test_has_the_phases_of_its_warmup_run_and_of_its_clients_measured_run(
    tmp_path: Path, phased
) -> None:
    result = await _attempt(tmp_path, phased, "agentx", 0.004, _point(tmp_path, agentx=True))

    assert result.validity is Validity.SUCCEEDED
    names = [p.name for p in result.phases]
    assert _in_order(
        [
            *BEFORE_THE_ROUNDS,
            "round_setup",
            "server_boot",
            "warmup",
            "round_collect",
            "client_start",
            "warmup",
            "measured",
            "round_collect",
            "teardown",
            *AFTER_THE_RUNNER,
        ],
        names,
    ), names
    _adds_up(result)


async def test_a_test_that_failed_at_boot_has_its_phases_up_to_the_failure_and_the_hosts_after_it(
    tmp_path: Path, phased
) -> None:
    result = await _attempt(tmp_path, phased, "failing_boot", 0.0005, _point(tmp_path))

    assert result.validity is not Validity.SUCCEEDED
    names = [p.name for p in result.phases]
    assert _in_order([*BEFORE_THE_ROUNDS, "server_boot", *AFTER_THE_RUNNER], names), names
    assert not {"warmup", "accuracy_eval", "measured"} & set(names), "nothing ran after the boot that failed"
    assert [p.seconds for p in result.phases if p.name == "server_boot"][0] == pytest.approx(0.855, abs=0.3)
    _adds_up(result)
