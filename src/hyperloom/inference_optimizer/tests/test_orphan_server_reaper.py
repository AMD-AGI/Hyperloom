# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the session-scoped orphaned serving-process reaper.

A monitor-process death (e.g. raylet crash taking the optimizer down) leaves the
setsid'd SGLang/vLLM server tree alive with its pidfile still on disk. The next
launch/resume must reap those orphans, scoped strictly to the current session's
own pidfiles and guarded by a cmdline match so a recycled pid is never killed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from hyperloom.orchestrator.actions.executors._server_lifecycle import (
    reap_orphaned_servers,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_cmdline_marker(pid: int, marker: str, timeout: float = 5.0) -> None:
    """Block until ``/proc/<pid>/cmdline`` contains ``marker``.

    ``subprocess.Popen`` returns as soon as the child is forked, but the reaper
    matches on ``/proc/<pid>/cmdline`` which stays empty until the child has
    finished ``exec``-ing the interpreter. Polling here removes that race so the
    reaper deterministically sees a server-looking cmdline.
    """
    deadline = time.time() + timeout
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    while time.time() < deadline:
        try:
            raw = proc_cmdline.read_bytes()
        except OSError:
            raw = b""
        if marker.encode() in raw:
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} cmdline never contained {marker!r}")


def _spawn_marker_process(marker: str) -> subprocess.Popen:
    """Spawn a long-lived child whose argv contains ``marker`` (cmdline match)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import time,sys; _={marker!r}; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _wait_for_cmdline_marker(proc.pid, marker)
    return proc


def _write_pidfile(session_dir: Path, tag: str, pid: int) -> Path:
    run_dir = session_dir / "runs" / "explore" / "v0"
    run_dir.mkdir(parents=True, exist_ok=True)
    pidfile = run_dir / f"{tag}.pid"
    pidfile.write_text(str(pid), encoding="utf-8")
    return pidfile


def _spawn_dead_leader_with_live_child(tmp_path: Path, marker: str) -> tuple[int, int]:
    """Return ``(leader_pid, child_pid)`` where child remains in leader's pgid."""
    info = tmp_path / "child.json"
    leader_script = tmp_path / "leader.py"
    leader_script.write_text(
        "\n".join(
            [
                "import json",
                "import subprocess",
                "import sys",
                f"p = subprocess.Popen([sys.executable, '-c', {('import time; _=' + repr(marker) + '; time.sleep(120)')!r}])",
                f"open({str(info)!r}, 'w').write(json.dumps({{'child_pid': p.pid}}))",
                "sys.exit(0)",
            ]
        ),
        encoding="utf-8",
    )
    leader = subprocess.Popen(
        [sys.executable, str(leader_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    leader_pid = leader.pid
    leader.wait(timeout=5)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if info.exists():
            import json

            child_pid = int(json.loads(info.read_text(encoding="utf-8"))["child_pid"])
            return leader_pid, child_pid
        time.sleep(0.05)
    raise AssertionError("leader did not write child pid")


def test_reap_kills_matching_orphan_and_clears_pidfile(tmp_path):
    """A live server whose cmdline matches is reaped and its pidfile removed."""
    proc = _spawn_marker_process("sglang.launch_server")
    pidfile = _write_pidfile(tmp_path, "sglang_8888", proc.pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        # Reap the zombie so the liveness probe reflects true termination
        # (the reaper is not this process's parent-waiter).
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

        assert proc.pid in reaped
        assert not _pid_alive(proc.pid)
        assert not pidfile.exists()
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait(timeout=5)


def test_reap_spares_pid_whose_cmdline_does_not_match(tmp_path):
    """A recycled pid running an unrelated process must NOT be killed."""
    proc = _spawn_marker_process("totally-unrelated-process")
    pidfile = _write_pidfile(tmp_path, "sglang_8888", proc.pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        assert proc.pid not in reaped
        assert _pid_alive(proc.pid), "unrelated pid must not be killed"
        # A spared pidfile is left so a later run can re-evaluate it.
        assert pidfile.exists()
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait(timeout=5)


def test_reap_no_pidfiles_is_noop(tmp_path):
    """Fresh session (no pidfiles) reaps nothing and never raises."""
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    assert reap_orphaned_servers(tmp_path) == []


def test_reap_dead_pid_clears_stale_pidfile(tmp_path):
    """A pidfile pointing at a dead pid is cleaned up without killing anything."""
    proc = _spawn_marker_process("sglang.launch_server")
    proc.kill()
    proc.wait(timeout=5)
    pidfile = _write_pidfile(tmp_path, "sglang_8888", proc.pid)

    reaped = reap_orphaned_servers(tmp_path)

    assert proc.pid not in reaped
    assert not pidfile.exists()


def test_reap_kills_group_when_recorded_leader_exited_but_child_survives(tmp_path):
    """The real leak shape: setsid leader exits, child remains in its pgid."""
    leader_pid, child_pid = _spawn_dead_leader_with_live_child(tmp_path, "sglang.launch_server")
    pidfile = _write_pidfile(tmp_path, "sglang_8888", leader_pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        deadline = time.time() + 5.0
        while time.time() < deadline and _pid_alive(child_pid):
            time.sleep(0.05)

        assert leader_pid in reaped
        assert not _pid_alive(child_pid)
        assert not pidfile.exists()
    finally:
        try:
            os.kill(child_pid, 9)
        except OSError:
            pass


def test_reap_spares_group_when_recorded_leader_exited_but_child_is_unrelated(tmp_path):
    """A recycled pgid must not be killed unless a group member still looks like a server."""
    leader_pid, child_pid = _spawn_dead_leader_with_live_child(tmp_path, "totally-unrelated-process")
    pidfile = _write_pidfile(tmp_path, "sglang_8888", leader_pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        assert leader_pid not in reaped
        assert _pid_alive(child_pid), "unrelated process group must not be killed"
        assert pidfile.exists()
    finally:
        try:
            os.kill(child_pid, 9)
        except OSError:
            pass
