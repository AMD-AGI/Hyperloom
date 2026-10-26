# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A GEAK run launched the way the KERNEL phase launches it leaves nothing running once the launch returns."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from hyperloom.common.proctree import running
from hyperloom.orchestrator.actions.executors._subprocess_kill import REAP_MARK_ENV, RoundWatch, run_with_session_kill

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="procfs is Linux-only")

# A fake run_e2e that starts a long-lived "server" and then returns normally (GEAK's own timeout path) or keeps
# running. The server never holds run_e2e's stdio, so nothing but a reap can end it; ``detach`` starts it the way
# GEAK's agents start A/B legs, with ``setsid nohup ... &`` from a shell that exits, so it is in neither
# run_e2e's process group nor its process tree.
_RUNNER = """
    import json, subprocess, sys, time
    if {detach}:
        server_pid = subprocess.run(
            ["sh", "-c", 'setsid nohup "$0" -c "$1" </dev/null >/dev/null 2>&1 & echo $!', sys.executable,
             {server_code!r}],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    else:
        server_pid = str(subprocess.Popen(
            [sys.executable, "-c", {server_code!r}],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).pid)
    open(sys.argv[2] + ".server_pid", "w").write(server_pid)
    if {hang}:
        time.sleep(300)
    json.dump({{"status": "ok"}}, open(sys.argv[2], "w"))
"""
_SERVER = "import time; time.sleep(300)"
_STUBBORN_SERVER = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"

_DETACH = pytest.mark.parametrize("detach", [False, True], ids=["in_group", "setsid"])


def _wait_dead(pid: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not running(pid):
            return True
        time.sleep(0.2)
    return False


def _write_runner(tmp_path: Path, *, hang: bool, detach: bool, server_code: str = _SERVER) -> Path:
    path = tmp_path / "fake_run_e2e.py"
    path.write_text(textwrap.dedent(_RUNNER.format(hang=hang, detach=detach, server_code=server_code)), "utf-8")
    return path


def _launch(tmp_path: Path, runner: Path, *, timeout: float) -> subprocess.CompletedProcess:
    """Run geak_runner through the KERNEL phase's launcher, marked as the phase marks it."""
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({"schema_version": 1, "model_path": "/m", "exp_root": "/tmp/x"}), encoding="utf-8")
    env = dict(os.environ, GEAK_E2E_RUNNER=str(runner), GEAK_FLUSH_GRACE_S="1")
    env[REAP_MARK_ENV] = uuid.uuid4().hex
    cmd = [
        sys.executable,
        "-m",
        "hyperloom.orchestrator.kernel.geak_runner",
        str(handoff),
        str(tmp_path / "out"),
        "--timeout-s",
        "600",
    ]
    return run_with_session_kill(cmd, env=env, watch=RoundWatch(timeout=timeout))


def _server_pid(tmp_path: Path) -> int:
    return int((tmp_path / "out" / "result.json.server_pid").read_text())


@pytest.fixture
def kill_leftover():
    pids: list[int] = []
    yield pids.append
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@_DETACH
def test_servers_left_by_a_normally_returning_runner_are_reaped(tmp_path, kill_leftover, detach):
    started = time.monotonic()

    proc = _launch(tmp_path, _write_runner(tmp_path, hang=False, detach=detach), timeout=60)
    server_pid = _server_pid(tmp_path)
    kill_leftover(server_pid)

    assert proc.returncode == 0
    assert time.monotonic() - started < 30, "the launch waited on the server instead of on geak_runner"
    assert _wait_dead(server_pid), "a server run_e2e started outlived the launch"


def test_a_run_killed_at_its_timeout_leaves_nothing_running(tmp_path, kill_leftover):
    runner = _write_runner(tmp_path, hang=True, detach=True, server_code=_STUBBORN_SERVER)

    with pytest.raises(subprocess.TimeoutExpired):
        _launch(tmp_path, runner, timeout=3)
    server_pid = _server_pid(tmp_path)
    kill_leftover(server_pid)

    assert _wait_dead(server_pid), "a server run_e2e started outlived the launch's timeout"


def test_processes_without_the_run_mark_are_left_alone(tmp_path, kill_leftover):
    bystander = subprocess.Popen([sys.executable, "-c", _SERVER], start_new_session=True)
    kill_leftover(bystander.pid)

    _launch(tmp_path, _write_runner(tmp_path, hang=False, detach=True), timeout=60)
    kill_leftover(_server_pid(tmp_path))

    assert running(bystander.pid)
