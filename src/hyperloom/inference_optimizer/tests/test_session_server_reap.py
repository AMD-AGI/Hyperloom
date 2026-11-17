# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the session's orphan reap: every ``server.pid`` under ``runs/`` goes through ``inference_suite.stop``."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl
from hyperloom_common.proctree import proc_identity, running
from inference_suite.frameworks import Framework
from inference_suite.runtime.process import ServerPidfile, Spawned, spawn

pytestmark = pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs Linux procfs")

_DIGEST = "sha256:" + "a" * 64
# The leader starts one worker, which inherits the run mark, and records its pid.
_SERVER = "import subprocess, sys, time\nw = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\nopen(sys.argv[1], 'w').write(str(w.pid))\ntime.sleep(600)\n"


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _boot(tmp_path: Path, pid_dir: Path, start_time_shift: int = 0) -> tuple[Spawned, int]:
    """Start a fake server with one worker and record it in ``pid_dir``'s ``server.pid``; return it and the worker."""
    worker_file = tmp_path / f"worker-{pid_dir.name}"
    spawned = spawn((sys.executable, "-c", _SERVER, str(worker_file)), dict(os.environ), tmp_path / "server.log")
    deadline = time.monotonic() + 10
    while not worker_file.exists() or not worker_file.read_text(encoding="utf-8"):
        assert time.monotonic() < deadline, "the fake server never started its worker"
        time.sleep(0.05)
    pid_dir.mkdir(parents=True, exist_ok=True)
    ServerPidfile(
        pid=spawned.proc.pid,
        pgid=spawned.proc.pid,
        start_time=proc_identity(spawned.proc.pid)[1] + start_time_shift,
        run_mark=spawned.mark,
        port=30000,
        framework=Framework.SGLANG,
        launch_digest=_DIGEST,
    ).write(pid_dir / "server.pid")
    return spawned, int(worker_file.read_text(encoding="utf-8"))


@pytest.fixture
def servers():
    started: list[tuple[Spawned, int]] = []
    yield started
    for spawned, worker in started:
        spawned.reap()
        if running(worker):
            os.kill(worker, signal.SIGKILL)


def test_a_server_left_running_is_stopped_with_its_workers_and_its_pidfile_removed(tmp_path, servers):
    pid_dir = tmp_path / "session" / "runs" / "baseline" / "r1"
    servers.append(_boot(tmp_path, pid_dir))
    (spawned, worker) = servers[0]

    reaped = sl.reap_orphaned_servers(tmp_path / "session")

    assert {spawned.proc.pid, worker} <= set(reaped)
    spawned.proc.wait(timeout=10)
    assert _gone(worker)
    assert not (pid_dir / "server.pid").exists()


def test_workers_are_reaped_by_run_mark_once_the_leader_is_gone(tmp_path, servers):
    pid_dir = tmp_path / "session" / "runs" / "grid" / "r2"
    servers.append(_boot(tmp_path, pid_dir))
    (spawned, worker) = servers[0]
    spawned.proc.kill()
    spawned.proc.wait(timeout=10)

    reaped = sl.reap_orphaned_servers(tmp_path / "session")

    assert worker in reaped and spawned.proc.pid not in reaped
    assert _gone(worker)


def test_a_recycled_pid_is_not_signalled(tmp_path, servers):
    """A start time that does not match means the pid names another process now; only the mark may be followed."""
    pid_dir = tmp_path / "session" / "runs" / "r3"
    servers.append(_boot(tmp_path, pid_dir, start_time_shift=1))
    (spawned, worker) = servers[0]
    # Strip the mark's reach: the leader and its worker are someone else's now.
    record = ServerPidfile.read(pid_dir / "server.pid")
    ServerPidfile(**{**record.to_document(), "run_mark": "not-this-server"}).write(pid_dir / "server.pid")

    assert sl.reap_orphaned_servers(tmp_path / "session") == []
    assert spawned.proc.poll() is None and running(worker)


def test_only_this_session_runs_are_reaped(tmp_path, servers):
    elsewhere = tmp_path / "other-session" / "runs" / "r1"
    servers.append(_boot(tmp_path, elsewhere))
    (spawned, _worker) = servers[0]

    assert sl.reap_orphaned_servers(tmp_path / "session") == []
    assert spawned.proc.poll() is None
    assert (elsewhere / "server.pid").exists()


def test_a_session_without_runs_reaps_nothing(tmp_path):
    assert sl.reap_orphaned_servers(tmp_path) == []
