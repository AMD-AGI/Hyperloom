# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A hosted test that goes silent before its server is ready ends as ``hung_before_ready``, an infrastructure failure."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.failures import Disposition, classify_test_failure
from hyperloom.measure.boot_watch import HUNG_BEFORE_READY, PRE_READY_SILENCE_SEC, SERVER_READY_KEY, PreReadySilence
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC
from hyperloom.measure.runner.service import Progress
from hyperloom.measure.tests.hosted_fixtures import make_call, make_rig, point_yaml

_LIMIT = 10.0
_RUNNING = {"phase": "running", "last_output_unix": 100.0, "updated_unix": 200.0, "last_log_age_sec": 5.0}


def _silence(**reading: object) -> PreReadySilence:
    silence = PreReadySilence(0.0, _LIMIT)
    silence.observe({**_RUNNING, **reading}, 1.0)
    return silence


def test_a_test_with_no_activity_for_the_limit_before_ready_is_silent_for_that_long() -> None:
    silence = _silence()
    assert silence.silent_for(1.0 + _LIMIT - 0.1) is None
    silence.observe(dict(_RUNNING), 5.0)  # the same reading again is no activity
    assert silence.silent_for(1.0 + _LIMIT) == pytest.approx(_LIMIT)


def test_progress_that_cannot_be_read_is_no_activity() -> None:
    silence = PreReadySilence(0.0, _LIMIT)
    silence.observe(None, 9.0)
    assert silence.silent_for(_LIMIT) == pytest.approx(_LIMIT)


@pytest.mark.parametrize(
    "change",
    [
        {"phase": "preparing"},
        {"last_output_unix": 150.0},
        {"last_line": "[aiter] start build module_moe"},
        {"updated_unix": 260.0, "last_log_age_sec": 3.0},  # the log was written again 62 s later
    ],
)
def test_each_new_sign_of_output_restarts_the_bound(change: dict) -> None:
    silence = _silence()
    silence.observe({**_RUNNING, **change}, 8.0)
    assert silence.silent_for(1.0 + _LIMIT) is None and silence.silent_for(8.0 + _LIMIT) is not None


def test_a_log_whose_write_time_only_jitters_by_the_clock_is_not_written_again() -> None:
    silence = _silence()
    silence.observe({**_RUNNING, "updated_unix": 220.0, "last_log_age_sec": 25.4}, 8.0)  # written at 194.6, not 195
    assert silence.silent_for(1.0 + _LIMIT) is not None


@pytest.mark.parametrize("reading", [{SERVER_READY_KEY: True}, {"phase": "collecting"}])
def test_once_the_server_is_ready_or_the_test_is_past_running_nothing_bounds_it(reading: dict) -> None:
    silence = _silence()
    silence.observe({**_RUNNING, **reading}, 2.0)
    silence.observe(dict(_RUNNING), 3.0)  # a later reading without the flag does not arm it again
    assert silence.silent_for(1.0 + 100 * _LIMIT) is None


def test_the_limit_is_above_the_longest_silence_measured_before_ready() -> None:
    """262 s: an aiter JIT build that reported 273 s, the longest silent stretch before ready in 1711 boots."""
    assert PRE_READY_SILENCE_SEC >= 3 * 262


def test_hung_before_ready_is_an_infrastructure_failure_a_new_sandbox_retries() -> None:
    verdict = classify_test_failure(returncode=0, error_class=HUNG_BEFORE_READY)
    assert (verdict.disposition, verdict.failure_class) == (Disposition.FAILED, FailureClass.INFRA)


def test_the_runner_reports_ready_once_a_round_stamped_its_server_ready(tmp_path: Path) -> None:
    files = tmp_path / "files"
    progress = Progress(tmp_path / "out", files_dir=files)
    (tmp_path / "out").mkdir()
    progress.flush()
    assert (tmp_path / "out" / "progress.json").read_text(encoding="utf-8").count('"server_ready": false') == 1
    (files / "variant_00_test" / "bench").mkdir(parents=True)
    (files / "variant_00_test" / "bench" / "server_ready_at").write_text(f"{time.time()} 80\n", encoding="utf-8")
    progress.flush()
    assert '"server_ready": true' in (tmp_path / "out" / "progress.json").read_text(encoding="utf-8")


@pytest.fixture
def rig(tmp_path: Path):
    made = make_rig(tmp_path, pre_ready_silence_sec=1.0)
    yield made
    made.backend.close()
    made.provider.close()


def test_a_test_silent_before_ready_ends_as_hung_before_ready_long_before_its_timeout(rig, tmp_path: Path) -> None:
    call = make_call(tmp_path, timeout_sec=60, point=point_yaml(tmp_path, fake_mode="hang"))
    started = time.monotonic()
    outcome = rig.backend.run_test(call)
    assert time.monotonic() - started < 30
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC and outcome.failure_class is FailureClass.INFRA
    assert outcome.failure_reason == HUNG_BEFORE_READY
    assert outcome.stderr.startswith(f"{HUNG_BEFORE_READY}: no output for ")
    assert rig.provider.find({}, role="measurement") == [], "the hung sandbox is closed"


#: The bound of the test below: longer than the stand-in runner takes to start and stamp its server ready (4.5 to
#: 7.7 s measured on a loaded dev pod, where a 1 s bound ended the test as hung before the runner wrote anything).
_STARTUP_BOUND_SEC = 15.0
#: Its timeout: past the bound counted from the ready stamp, so the bound would end the test first were it still armed.
_AFTER_READY_TIMEOUT_SEC = 25


def test_a_test_silent_after_its_server_is_ready_is_left_to_the_silence_gate_and_the_timeout(tmp_path: Path) -> None:
    made = make_rig(tmp_path, pre_ready_silence_sec=_STARTUP_BOUND_SEC)
    point = point_yaml(tmp_path, fake_mode="hang", fake_ready="true")
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            made.backend.run_test(make_call(tmp_path, timeout_sec=_AFTER_READY_TIMEOUT_SEC, point=point))
        assert made.backend.journal.latest("req-1").extra["verdict"] == "timeout"
    finally:
        made.backend.close()
        made.provider.close()
