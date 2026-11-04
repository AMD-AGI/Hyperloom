# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the runner signs of where its time went: the phases of the run on its own clock, from the start command that
began it to the result it signs, with whatever the backend stamped between them."""

from __future__ import annotations

import time

import pytest

from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.runner import run as runner
from hyperloom.measure.runner.tests.test_run import _Backend, _go, _ok, _sheet, env, key  # noqa: F401 - fixtures
from hyperloom.measure.socket import MeasurementTimeout, TestCall, TestOutcome

NAP = 0.02


def _stamping(call: TestCall) -> TestOutcome:
    """A backend that stamps the phases a test goes through, as the real one does through the call's recorder."""
    call.phases.begin("kernel_tables")
    time.sleep(NAP)
    call.phases.begin("server_boot", point=0)
    time.sleep(NAP)
    call.phases.begin("measured", point=0)
    time.sleep(NAP)
    return _ok(call)


def _timing(tmp_path):
    return ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).timing


def test_the_signed_result_states_the_runners_phases_in_order_with_the_backends_between_them(
    tmp_path, env, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_stamping))

    assert _go(tmp_path, env, _sheet()) == 0

    timing = _timing(tmp_path)
    names = [p.name for p in timing.phases]
    assert names == [
        "sheet_wait",
        "preflight",
        "env_rebuild",
        "kernel_tables",
        "server_boot",
        "measured",
        "result_signing",
    ]
    assert [p.point for p in timing.phases if p.name in ("server_boot", "measured")] == [0, 0]
    stated = sum(p.seconds for p in timing.phases)
    assert stated == pytest.approx(timing.finished_unix - timing.runner_started_unix, abs=0.05), "no second is dropped"
    assert all(a.started_at + a.seconds <= b.started_at + 1e-6 for a, b in zip(timing.phases, timing.phases[1:]))


def test_the_start_up_script_and_the_interpreters_start_are_the_first_phases_of_the_run(
    tmp_path, env, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))
    now = time.time()
    env["HYPERLOOM_BOOTSTRAP_STARTED_UNIX"] = f"{now - 30.0:.6f}"
    env["HYPERLOOM_BOOTSTRAP_ENDED_UNIX"] = f"{now - 2.0:.6f}"

    assert _go(tmp_path, env, _sheet()) == 0

    first, second, third, *_ = _timing(tmp_path).phases
    assert (first.name, second.name, third.name) == ("bootstrap", "runner_start", "sheet_wait")
    assert (first.seconds, second.seconds) == (pytest.approx(28.0, abs=1e-3), pytest.approx(2.0, abs=1.0))
    assert first.started_at + first.seconds == pytest.approx(second.started_at, abs=1e-3), (
        "the script ends where it said"
    )
    assert second.started_at + second.seconds == pytest.approx(third.started_at, abs=1e-3), "the runner starts the wait"


def test_a_script_that_said_when_it_began_but_not_when_it_ended_is_one_phase_up_to_the_runner(
    tmp_path, env, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))
    env["HYPERLOOM_BOOTSTRAP_STARTED_UNIX"] = f"{time.time() - 30.0:.6f}"
    env["HYPERLOOM_BOOTSTRAP_ENDED_UNIX"] = ""

    assert _go(tmp_path, env, _sheet()) == 0

    first, second, *_ = _timing(tmp_path).phases
    assert (first.name, second.name) == ("bootstrap", "sheet_wait") and first.seconds == pytest.approx(30.0, abs=1.0)


@pytest.mark.parametrize("said", [None, "", "not a time", "%N", "nan", "inf", "-5", "99999999999.0"])
def test_a_start_time_that_is_not_one_the_runner_could_have_started_after_is_no_phase(
    tmp_path, env, monkeypatch, said
) -> None:
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))
    if said is not None:
        env["HYPERLOOM_BOOTSTRAP_STARTED_UNIX"] = said
        env["HYPERLOOM_BOOTSTRAP_ENDED_UNIX"] = said

    assert _go(tmp_path, env, _sheet()) == 0

    assert not {"bootstrap", "runner_start"} & {p.name for p in _timing(tmp_path).phases}


def test_a_test_that_failed_is_a_signed_result_with_the_phases_up_to_its_failure(tmp_path, env, monkeypatch) -> None:
    def failing_boot(call: TestCall) -> TestOutcome:
        call.phases.begin("server_boot", point=0)
        time.sleep(NAP)
        return TestOutcome(1, "", "CUDA out of memory: HIP out of memory", (), SandboxFacts(sandbox_id="x"))

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(failing_boot))

    assert _go(tmp_path, env, _sheet()) == 0

    names = [p.name for p in _timing(tmp_path).phases]
    assert names[-2:] == ["server_boot", "result_signing"] and "measured" not in names


def test_a_test_that_timed_out_is_a_signed_timeout_with_the_phases_up_to_it(tmp_path, env, monkeypatch) -> None:
    def timing_out(call: TestCall) -> TestOutcome:
        call.phases.begin("server_boot", point=0)
        time.sleep(NAP)
        raise MeasurementTimeout(call.task_id, call.timeout_sec)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(timing_out))

    assert _go(tmp_path, env, _sheet()) == 0

    manifest = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes())
    assert manifest.outcome.timed_out
    assert [p.name for p in manifest.timing.phases][-2:] == ["server_boot", "result_signing"]
