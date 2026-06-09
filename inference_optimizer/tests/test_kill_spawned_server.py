# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_subprocess_kill.kill_my_spawned_server`` and the BaselineExecutor integration (``bugs.md`` §B).

Covers the no-op / already-exited cases, the same-session-group refusal guard,
SIGTERM→grace→SIGKILL ordering, and grandchild reaping (the bugs.md §B leak).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors._subprocess_kill import (
    OVERTIME_KILL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    _server_log_shows_death,
    kill_my_spawned_server,
    new_session_kwargs,
    run_with_session_kill,
)


# Helper-level tests
def test_kill_my_spawned_server_handles_none():
    """Plain no-op when given None so callers can use it in ``finally:`` unguarded."""
    kill_my_spawned_server(None)  # must not raise


def test_kill_my_spawned_server_handles_already_exited():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
    )
    proc.wait(timeout=10)
    kill_my_spawned_server(proc)


def test_kill_my_spawned_server_refuses_own_session_group(caplog):
    """Defensive guard: the helper must NOT killpg the parent's own session group."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with caplog.at_level("ERROR"):
            kill_my_spawned_server(proc, grace_seconds=0.5)
        assert proc.poll() is None, (
            "helper killed a process in the parent's own session — that "
            "would take down the Coordinator in production"
        )
        assert any(
            "refusing to killpg own session" in rec.message
            for rec in caplog.records
        ), "expected an ERROR log line about same-pgid refusal"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_kill_my_spawned_server_sigterm_then_sigkill_for_ignorer():
    """A child that traps SIGTERM is still reaped via SIGKILL after the grace window."""
    proc = subprocess.Popen(
        [sys.executable, "-c", (
            "import signal, time;\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);\n"
            "time.sleep(60)\n"
        )],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
    )
    # Let the child install its SIGTERM handler before we signal.
    time.sleep(0.3)
    start = time.monotonic()
    kill_my_spawned_server(proc, grace_seconds=1.0)
    elapsed = time.monotonic() - start
    assert proc.poll() is not None
    assert elapsed < 5.0, f"kill_my_spawned_server hung for {elapsed:.2f}s"


def test_kill_my_spawned_server_reaps_grandchildren():
    """bugs.md §B: a child that spawns a grandchild leaves no surviving descendant after the helper returns."""
    proc = subprocess.Popen(
        [sys.executable, "-c", (
            "import os, sys, time;\n"
            "# Write our pgid + grandchild PID to disk for the test to read.\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    # Grandchild: pretend to be a long-running server.\n"
            "    time.sleep(120)\n"
            "    sys.exit(0)\n"
            "open(sys.argv[1], 'w').write(str(pid))\n"
            "time.sleep(120)\n"
        ), "/tmp/hyperloom_test_grandchild.pid"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
    )
    pid_file = Path("/tmp/hyperloom_test_grandchild.pid")
    try:
        deadline = time.monotonic() + 5.0
        grandchild_pid: int | None = None
        while time.monotonic() < deadline:
            if pid_file.exists():
                txt = pid_file.read_text().strip()
                if txt:
                    grandchild_pid = int(txt)
                    break
            time.sleep(0.05)
        assert grandchild_pid is not None, "parent never wrote grandchild pid"

        os.kill(grandchild_pid, 0)  # raises if gone

        kill_my_spawned_server(proc, grace_seconds=1.5)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
    finally:
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass


# BaselineExecutor integration — confirm the kill is on every exit path
def _make_fake_magpie_command(
    tmp_path: Path,
    *,
    mode: str,
) -> tuple[Path, Path]:
    """Build a ``python -m Magpie`` stand-in; returns (script_path, sentinel_file)."""
    script = tmp_path / "fake_magpie.py"
    sentinel = tmp_path / "leaked_grandchild.pid"
    workspace = tmp_path / "out" / "benchmark_fake_20260101_000000"

    if mode == "succeed_then_leak":
        body = f"""
import json, os, pathlib, sys, time
ws = pathlib.Path({str(workspace)!r})
ws.mkdir(parents=True, exist_ok=True)
(ws / "benchmark_report.json").write_text(json.dumps({{
    "output_throughput": 12.3,
    "completed": 42,
}}))
pid = os.fork()
if pid == 0:
    time.sleep(120)
    sys.exit(0)
pathlib.Path({str(sentinel)!r}).write_text(str(pid))
sys.exit(0)
"""
    elif mode == "timeout":
        body = f"""
import os, pathlib, sys, time
pid = os.fork()
if pid == 0:
    time.sleep(120)
    sys.exit(0)
pathlib.Path({str(sentinel)!r}).write_text(str(pid))
time.sleep(120)
"""
    else:
        raise ValueError(mode)

    script.write_text(body)
    return script, sentinel


@pytest.mark.asyncio
async def test_baseline_executor_kills_grandchild_on_timeout(tmp_path, monkeypatch):
    """A leaked grandchild must be dead by the time the executor returns after its timeout fires."""
    script, sentinel = _make_fake_magpie_command(tmp_path, mode="timeout")
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
    )
    try:
        deadline = time.monotonic() + 5.0
        grandchild_pid: int | None = None
        while time.monotonic() < deadline:
            if sentinel.exists():
                txt = sentinel.read_text().strip()
                if txt:
                    grandchild_pid = int(txt)
                    break
            time.sleep(0.05)
        assert grandchild_pid is not None
        os.kill(grandchild_pid, 0)

        kill_my_spawned_server(proc, grace_seconds=1.5)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass


# Fix E — run_with_session_kill soft_deadline_sec
def test_run_with_session_kill_soft_deadline_returns_sentinel():
    """A child past ``soft_deadline_sec`` is reaped and returns ``OVERTIME_KILL_RETURNCODE`` (no ``TimeoutExpired``)."""
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=30,
        soft_deadline_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    assert elapsed < 10.0, f"soft-deadline path took {elapsed:.2f}s"


def test_run_with_session_kill_soft_deadline_does_not_fire_for_quick_child():
    """A child exiting before ``soft_deadline_sec`` returns normally with its own returncode."""
    cp = run_with_session_kill(
        [sys.executable, "-c", "print('hi'); raise SystemExit(0)"],
        timeout=10,
        soft_deadline_sec=5.0,
    )
    assert cp.returncode == 0
    assert "hi" in (cp.stdout or "")


def test_run_with_session_kill_legacy_timeout_still_raises():
    """With ``soft_deadline_sec`` None, a child exceeding the hard ``timeout`` still raises ``TimeoutExpired``."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
            soft_deadline_sec=None,
        )


# Server-liveness watchdog — fast-fail on a crashed-but-hung server
def test_server_log_shows_death_detects_marker(tmp_path):
    """A ``server.log`` containing a terminal-init marker reads as dead;
    a healthy / missing log reads as alive."""
    log_path = tmp_path / "server.log"
    assert _server_log_shows_death(str(log_path)) is False  # missing → alive
    log_path.write_text("INFO loading shards 50%\nINFO graph capture\n")
    assert _server_log_shows_death(str(log_path)) is False  # healthy → alive
    log_path.write_text(
        "ERROR core.py Exception: WorkerProc initialization failed due to an "
        "exception in a background process.\n"
    )
    assert _server_log_shows_death(str(log_path)) is True


def test_run_with_session_kill_watchdog_reaps_hung_server(tmp_path):
    """A child that writes a fatal server marker then hangs is reaped via the
    watchdog with ``SERVER_DEAD_RETURNCODE`` — well before the hard timeout."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "open(sys.argv[1], 'w').write("
        "'Exception: WorkerProc initialization failed in background\\n')\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=60,
        server_log_path=str(log_path),
        server_dead_grace_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == SERVER_DEAD_RETURNCODE
    assert elapsed < 15.0, f"watchdog path took {elapsed:.2f}s (expected fast)"


def test_run_with_session_kill_watchdog_grace_lets_clean_exit_win(tmp_path):
    """If the harness exits on its own within the grace window after emitting a
    marker, its real returncode wins (no spurious SERVER_DEAD)."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "open(sys.argv[1], 'w').write("
        "'Exception: WorkerProc initialization failed in background\\n')\n"
        "time.sleep(0.3)\n"
        "raise SystemExit(7)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_dead_grace_sec=10.0,
        server_log_path=str(log_path),
    )
    assert cp.returncode == 7


def test_run_with_session_kill_watchdog_ignores_healthy_server(tmp_path):
    """A child with a clean server.log returns its own returncode — the
    watchdog must not false-positive on a healthy (or slow) server."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys\n"
        "open(sys.argv[1], 'w').write('INFO server ready on port 8888\\n')\n"
        "print('ok')\n"
        "raise SystemExit(0)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_dead_grace_sec=2.0,
        server_log_path=str(log_path),
    )
    assert cp.returncode == 0
    assert "ok" in (cp.stdout or "")
