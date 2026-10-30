# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phases the local backend stamps for the tests it runs, against a ``run_grid`` that spends a clock's seconds the way
a real round does and leaves what the launcher leaves beside each run's server log (``phase_rounds``).

The clock is the test's, so every length asserted here is exact: a graded test, an AgentX test, a test whose server never
got ready, a sweep arm, a test that timed out.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.contracts.measurement import PhaseTiming
from hyperloom.measure import local
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.phases import PhaseRecorder
from hyperloom.measure.socket import MeasurementTimeout, TestCall
from hyperloom.measure.tests.phase_rounds import READY_ONLY_LOG, FakeClock, Rounds, Run, Script, StagedGrid, result
from hyperloom.measure.tests.test_local_backend import (  # noqa: F401 - magpie_backend is the module's autouse fixture
    _call,
    _graded_call,
    _point,
    _sweep_call,
    magpie_backend,
)


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(local, "time", SimpleNamespace(time=clock, monotonic=time.monotonic))
    monkeypatch.setattr(
        local, "resolve_lifecycle_params", lambda path: {"eligible": True, "port": 8123, "framework": "sglang"}
    )
    monkeypatch.setattr(local, "teardown_lifecycle_server", lambda **kw: clock.advance(3.0))
    return clock


def _run(monkeypatch, clock: FakeClock, call: TestCall, scripts: dict[str, Script]):
    """Run ``call`` with a recorder on the test's clock, as the runner passes its own; give the outcome and the recorder."""
    monkeypatch.setattr(local, "run_grid", StagedGrid(scripts))
    recorder = PhaseRecorder(clock)
    outcome = LocalMeasurementBackend().run_test(replace(call, phases=recorder))
    return outcome, recorder


def _stamped(phases: tuple[PhaseTiming, ...]) -> list[tuple[str, float, int | None]]:
    return [(p.name, round(p.seconds, 3), p.point) for p in phases]


# -- a graded InferenceX test -------------------------------------------------------------------------------------


def test_a_graded_test_goes_through_its_phases_in_order_and_they_add_up_to_the_time_the_backend_ran(
    monkeypatch, clock, tmp_path
) -> None:
    clock.advance(1000.0)  # whatever ran before the backend was called is not its time
    began = clock.now

    outcome, recorder = _run(monkeypatch, clock, _graded_call(tmp_path), Rounds(clock).graded())

    assert _stamped(outcome.phases) == [
        ("round_setup", 7.779, 0),
        ("server_boot.other", 20.221, 0),
        ("server_boot.weights_load", 40.0, 0),
        ("server_boot.compile", 10.0, 0),
        ("server_boot.graph_capture", 12.0, 0),
        ("warmup", 60.0, 0),
        ("accuracy_eval", 50.0, 0),
        ("round_collect", 5.0, 0),
        ("round_setup", 4.0, 0),
        ("measured", 56.0, 0),
        ("teardown", 40.0, 0),
        ("teardown", 3.0, None),
    ]
    assert sum(p.seconds for p in outcome.phases) == pytest.approx(clock.now - began)
    assert recorder.phases == outcome.phases, "a recorder the runner passed in is the one the backend stamped"


def test_the_time_before_the_first_round_is_the_kernel_tables_the_rebuilt_framework_and_the_launchers(
    monkeypatch, clock, tmp_path
) -> None:
    def slow_round(slot: Path, kwargs: dict[str, Any]):
        clock.advance(2.0)
        return result("test", slot)

    monkeypatch.setattr(local, "install_launchers", lambda points: clock.advance(1.5))

    outcome, _ = _run(monkeypatch, clock, _call(tmp_path), {"test": slow_round})

    assert _stamped(outcome.phases) == [("kernel_tables", 1.5, None), ("round", 2.0, 0)]


# -- an AgentX test -----------------------------------------------------------------------------------------------


def _agentx_point(tmp_path: Path) -> Path:
    path = tmp_path / "agentx.yaml"
    benchmark = {"framework": "sglang", "agentx": {"enabled": True}, "envs": {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}}
    path.write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    return path


def test_an_agentx_test_names_its_warmup_run_and_splits_its_measured_run_at_its_clients_phases(
    monkeypatch, clock, tmp_path
) -> None:
    call = _call(tmp_path, _agentx_point(tmp_path))

    outcome, _ = _run(monkeypatch, clock, call, Rounds(clock).agentx())

    assert _stamped(outcome.phases) == [
        ("round_setup", 10.0, 0),
        ("server_boot", 80.0, 0),
        ("warmup", 40.0, 0),
        ("round_collect", 30.0, 0),
        ("client_start", 15.0, 0),
        ("warmup", 30.0, 0),
        ("measured", 200.0, 0),
        ("round_collect", 15.0, 0),
        ("teardown", 30.0, 0),
    ]


# -- a test that failed at boot -----------------------------------------------------------------------------------


def test_a_test_whose_server_never_got_ready_has_the_phases_up_to_the_failure(monkeypatch, clock, tmp_path) -> None:
    outcome, _ = _run(monkeypatch, clock, _call(tmp_path), Rounds(clock).failing_boot())

    assert outcome.returncode != 0
    assert _stamped(outcome.phases) == [("server_boot", 1710.0, 0)], "no boot stamp: the whole round was the boot"


# -- a sweep arm --------------------------------------------------------------------------------------------------


def _sweep_scripts(clock: FakeClock) -> dict[str, Script]:
    def measure(slot: Path, kwargs: dict[str, Any]):
        clock.advance(1.0)
        Run(clock, slot).found_ready().end(20.0)
        clock.advance(2.0)
        return result(slot.name.removeprefix("variant_00_"), slot)

    def boot_then_measure(slot: Path, kwargs: dict[str, Any]):
        if not kwargs["launch"].lifecycle_boot_only:
            return measure(slot, kwargs)
        clock.advance(5.0)
        Run(clock, slot).boot(70.0, log=Rounds(clock).vllm_log()).end(2.0)
        clock.advance(1.0)
        return result("c8", slot)

    return {"c8": boot_then_measure, "c4": measure, "c2": measure}


def test_a_sweep_arm_boots_once_and_every_level_is_a_measured_round_of_its_own_point(
    monkeypatch, clock, tmp_path
) -> None:
    outcome, _ = _run(monkeypatch, clock, _sweep_call(tmp_path, 2, 8, 4), _sweep_scripts(clock))

    assert _stamped(outcome.phases) == [
        ("round_setup", 5.0, 1),
        ("server_boot.other", 8.0, 1),
        ("server_boot.weights_load", 40.0, 1),
        ("server_boot.compile", 10.0, 1),
        ("server_boot.graph_capture", 12.0, 1),
        ("warmup", 2.0, 1),
        ("round_collect", 1.0, 1),
        ("round_setup", 1.0, 1),
        ("measured", 20.0, 1),
        ("round_collect", 2.0, 1),
        ("round_setup", 1.0, 2),
        ("measured", 20.0, 2),
        ("round_collect", 2.0, 2),
        ("round_setup", 1.0, 0),
        ("measured", 20.0, 0),
        ("teardown", 2.0, 0),
        ("teardown", 3.0, None),
    ], "the points are the call's own (c8, c4, c2 are 1, 2, 0) and the server is stopped once, after the last level"


# -- a test that timed out ----------------------------------------------------------------------------------------


def test_a_test_that_timed_out_says_the_phases_it_went_through(monkeypatch, clock, tmp_path) -> None:
    def timing_out(slot: Path, kwargs: dict[str, Any]):
        clock.advance(3.0)
        Run(clock, slot).boot(30.0, log=READY_ONLY_LOG).end(1500.0)
        return result("test", slot, status="failed", returncode=None, error="timed out", error_class="magpie_timeout")

    monkeypatch.setattr(local, "run_grid", StagedGrid({"test": timing_out}))

    with pytest.raises(MeasurementTimeout) as raised:
        LocalMeasurementBackend().run_test(replace(_call(tmp_path), phases=PhaseRecorder(clock)))

    assert _stamped(raised.value.phases) == [("round_setup", 3.0, 0), ("server_boot", 30.0, 0), ("measured", 1500.0, 0)]


# -- who owns the recorder ----------------------------------------------------------------------------------------


def test_a_call_with_no_recorder_gets_its_phases_from_a_recorder_of_the_backends_own(monkeypatch, tmp_path) -> None:
    async def instant(**kwargs: Any):
        return [result(kwargs["grid"][0].name, Path(kwargs["output_root"]))]

    monkeypatch.setattr(local, "run_grid", instant)

    first = LocalMeasurementBackend().run_test(_call(tmp_path))
    second = LocalMeasurementBackend().run_test(_call(tmp_path))

    assert first.phases and first.phases[0].name == "kernel_tables"
    assert not set(first.phases) & set(second.phases), "each test has a recorder of its own"
