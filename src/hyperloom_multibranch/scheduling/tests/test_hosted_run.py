# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted run reads from the host and writes back: the grant's bound on the session's time, the stop the
grant's end sends, and the files that tell the host whether to start the run again. The mapping of the spec
onto ``run`` is pinned through the command in ``test_cli``."""

from __future__ import annotations

import json
import signal
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hyperloom.hosted.host_info import HostInfo, read_host_info
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS
from hyperloom_multibranch.scheduling.hosted_run import (
    GRANT_MARGIN_SECONDS,
    GrantDeadline,
    HostedExit,
    HostedRefused,
    arm_grant_stop,
    grant_deadline,
    record_session,
    recorded_session,
    write_final_exit,
)

CREATED = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def _host(tmp_path: Path, **grant: int) -> HostInfo:
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    env = {
        "HYPERLOOM_INPUT_FILE": str(spec),
        "HYPERLOOM_RUN_ID": "run-1",
        "HYPERLOOM_RUN_CREATED_AT": CREATED.isoformat(),
        "HYPERLOOM_STATE_DIR": str(tmp_path),
    }
    if grant:
        env["HYPERLOOM_GRANT_LIMITS"] = json.dumps(grant)
    return read_host_info(env, require=())


def test_the_session_ends_a_close_grace_and_a_margin_before_the_grant_and_is_stopped_a_grace_before_it(
    tmp_path: Path,
) -> None:
    deadline = grant_deadline(_host(tmp_path, max_runtime_sec=7200), CREATED + timedelta(seconds=3600))

    assert deadline == GrantDeadline(3600.0)
    assert deadline.stop_after_sec == 3600 - MB_CLOSE_GRACE_CAP_SECONDS
    assert deadline.session_hours(8.0) == (3600 - MB_CLOSE_GRACE_CAP_SECONDS - GRANT_MARGIN_SECONDS) / 3600
    assert deadline.session_hours(0.5) == 0.5, "a spec asking less than the grant has left keeps its hours"
    assert deadline.session_hours(0.0) == deadline.session_hours(8.0), "no limit of its own is what the grant leaves"


def test_a_grant_that_states_no_runtime_leaves_the_specs_hours_and_stops_nothing(tmp_path: Path) -> None:
    deadline = grant_deadline(_host(tmp_path, max_gpus=8), CREATED + timedelta(days=30))

    assert (deadline.session_hours(8.0), deadline.stop_after_sec) == (8.0, None)


@pytest.mark.parametrize("left", [-10.0, 0.0, MB_CLOSE_GRACE_CAP_SECONDS + GRANT_MARGIN_SECONDS])
def test_a_grant_without_time_for_a_session_refuses_it(left: float) -> None:
    with pytest.raises(HostedRefused, match="CLOSE grace"):
        GrantDeadline(left).session_hours(8.0)


def test_the_grant_stop_sends_this_process_sigterm() -> None:
    received = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: received.set())
    try:
        timer = arm_grant_stop(0.01)
        timer.join(timeout=5)
        assert received.wait(timeout=5)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_the_host_reads_a_final_ending_and_the_session_a_restart_finishes(tmp_path: Path) -> None:
    assert recorded_session(tmp_path) is None

    record_session(tmp_path, tmp_path / "MB-session")
    write_final_exit(tmp_path, HostedExit(1, stop_reason="signal"))

    assert recorded_session(tmp_path) == tmp_path / "MB-session"
    exit_document = json.loads((tmp_path / "exit.json").read_text(encoding="utf-8"))
    assert exit_document == {"final": True, "returncode": 1, "stop_reason": "signal"}
    write_final_exit(tmp_path, HostedExit(2, error="multibranch.workload is missing"))
    exit_document = json.loads((tmp_path / "exit.json").read_text(encoding="utf-8"))
    assert exit_document == {"final": True, "returncode": 2, "error": "multibranch.workload is missing"}
