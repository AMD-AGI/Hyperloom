# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure import socket as measurement_socket
from hyperloom.measure.socket import RoundPlan, TestCall, TestOutcome, run_test, set_active_backend


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    set_active_backend(None)


def _call(tmp_path: Path) -> TestCall:
    return TestCall(
        points=(tmp_path / "p.yaml",),
        output_dir=tmp_path,
        timeout_sec=60,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t",
    )


class _Backend:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[TestCall] = []

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call)
        return TestOutcome(0, "", "", (), SandboxFacts(sandbox_id="s"))


def test_run_test_goes_to_the_active_backend(tmp_path):
    backend = _Backend()
    set_active_backend(backend)

    outcome = run_test(_call(tmp_path))

    assert [c.task_id for c in backend.calls] == ["t"] and outcome.returncode == 0


def test_without_a_chosen_backend_the_local_one_runs(tmp_path):
    assert measurement_socket.active_backend().name == "local"


def test_set_active_backend_none_returns_to_the_local_backend(tmp_path):
    set_active_backend(_Backend())
    set_active_backend(None)

    assert measurement_socket.active_backend().name == "local"


@pytest.mark.parametrize("warmup", [2, -1])
def test_a_round_plan_allows_at_most_one_warmup_round(warmup):
    with pytest.raises(ValueError, match="at most one"):
        RoundPlan(warmup_rounds=warmup)


def test_a_round_plan_measures_one_round():
    with pytest.raises(ValueError, match="measures one round"):
        RoundPlan(measure_rounds=2)


def test_a_call_defaults_to_a_single_point_test_with_no_accuracy(tmp_path):
    call = _call(tmp_path)

    assert (call.kind, call.prep, call.accuracy, call.rounds) == ("test", "grid", False, RoundPlan())


def test_a_call_without_points_or_a_task_id_is_refused_before_any_backend_runs(tmp_path):
    backend = _Backend()
    set_active_backend(backend)
    call = _call(tmp_path)

    with pytest.raises(ValueError, match="at least one measurement point"):
        run_test(TestCall(**{**call.__dict__, "points": ()}))
    with pytest.raises(ValueError, match="task id"):
        run_test(TestCall(**{**call.__dict__, "task_id": ""}))
    assert backend.calls == []


def test_a_backend_that_returns_something_else_is_a_reported_defect(tmp_path):
    class Wrong:
        name = "wrong"

        def run_test(self, call):
            return 0

    set_active_backend(Wrong())

    with pytest.raises(TypeError, match="must return a TestOutcome, got int"):
        run_test(_call(tmp_path))
