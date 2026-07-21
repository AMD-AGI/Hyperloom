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


def _spawn_marker_process(marker: str) -> subprocess.Popen:
    """Spawn a long-lived child whose argv contains ``marker`` (cmdline match)."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time,sys; _={marker!r}; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _write_pidfile(session_dir: Path, tag: str, pid: int) -> Path:
    run_dir = session_dir / "runs" / "explore" / "v0"
    run_dir.mkdir(parents=True, exist_ok=True)
    pidfile = run_dir / f"{tag}.pid"
    pidfile.write_text(str(pid), encoding="utf-8")
    return pidfile


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
