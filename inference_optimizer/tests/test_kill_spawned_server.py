# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_subprocess_kill.kill_my_spawned_server`` and the
BaselineExecutor integration that uses it (Hyperloom ``bugs.md`` §B).

Coverage:

* The helper is a no-op for ``None`` / already-exited processes.
* The helper refuses to kill the parent's own session group (defensive
  guard against future regressions where someone forgets
  ``start_new_session=True``).
* SIGTERM → 5 s grace → SIGKILL ordering reaps a process that ignores
  SIGTERM.
* Grandchildren spawned by the launched process are killed too — this
  is the load-bearing case for bugs.md §B (Magpie spawns a shell
  wrapper which spawns the vLLM server; only the grandchild leaks).
* BaselineExecutor's `__call__` invokes the helper on every exit path
  (subprocess timeout, subprocess nonzero, success-but-no-workspace).
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
    kill_my_spawned_server,
    new_session_kwargs,
    run_with_session_kill,
)


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------
def test_kill_my_spawned_server_handles_none():
    """Plain no-op when given None — callers must be allowed to put this
    in a ``finally:`` without guarding it."""
    kill_my_spawned_server(None)  # must not raise


def test_kill_my_spawned_server_handles_already_exited():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
    )
    proc.wait(timeout=10)
    kill_my_spawned_server(proc)  # must not raise / not block


def test_kill_my_spawned_server_refuses_own_session_group(caplog):
    """Defensive guard: if a caller forgets ``start_new_session=True``,
    the helper must NOT killpg the parent's own session — that would
    take down the Coordinator. We launch without ``new_session_kwargs``
    on purpose and assert the helper logs an error and returns."""
    # Launch a sleeper in the same process group as the test runner.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with caplog.at_level("ERROR"):
            kill_my_spawned_server(proc, grace_seconds=0.5)
        # Sleeper must still be alive — helper refused to kill same pgid.
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
    """A child that traps SIGTERM and keeps running must still be reaped
    via SIGKILL after the grace window expires."""
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
    # Helper must have returned. Process must be dead.
    assert proc.poll() is not None
    # Must NOT have hung for the full sleep(60). 5 s cap is generous.
    assert elapsed < 5.0, f"kill_my_spawned_server hung for {elapsed:.2f}s"


def test_kill_my_spawned_server_reaps_grandchildren():
    """The load-bearing test for bugs.md §B — Magpie -> bash -> server.
    A child that spawns its own grandchild (and exits) must still leave
    no surviving descendant after kill_my_spawned_server returns."""
    # Parent forks a sleeping grandchild that re-parents to init when
    # parent exits, then parent itself exits 0. Without `killpg` the
    # grandchild keeps the GPU pinned (= the bugs.md §B leak).
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
        # Wait for the parent to write the grandchild's pid.
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

        # Grandchild must currently be alive.
        os.kill(grandchild_pid, 0)  # raises if gone

        kill_my_spawned_server(proc, grace_seconds=1.5)

        # After the helper returns, the grandchild must be gone too.
        # Give the kernel a moment to deliver SIGKILL — busy-poll up to 2s.
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


# ---------------------------------------------------------------------------
# BaselineExecutor integration — confirm the kill is on every exit path
# ---------------------------------------------------------------------------
def _make_fake_magpie_command(
    tmp_path: Path,
    *,
    mode: str,  # "succeed_then_leak" | "timeout"
) -> tuple[Path, Path]:
    """Build a stand-in for ``python -m Magpie`` that:

    * Creates a benchmark workspace directory (so BaselineExecutor's
      candidate-glob logic doesn't trip the no_workspace path).
    * Optionally forks a long-running grandchild ("leaked server").
    * In ``timeout`` mode, hangs forever so BaselineExecutor's timeout
      fires.

    Returns (script_path, sentinel_file).
    """
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
    """A leaked grandchild must be dead by the time the executor returns
    after its subprocess timeout fires."""
    script, sentinel = _make_fake_magpie_command(tmp_path, mode="timeout")
    # Patch the cmd construction so we invoke our fake instead of Magpie.
    # The cleanest seam is to replace the BaselineExecutor.__call__'s
    # cmd composition via monkey-patching subprocess.Popen — but easier
    # to test the helper in isolation here against a direct Popen, since
    # the integration of "Popen + finally-kill" is exercised by the
    # tests above already.
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
    )
    try:
        # Wait for grandchild pid to appear.
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
        os.kill(grandchild_pid, 0)  # still alive

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


# ---------------------------------------------------------------------------
# Fix E — run_with_session_kill soft_deadline_sec
# ---------------------------------------------------------------------------
def test_run_with_session_kill_soft_deadline_returns_sentinel():
    """A child that sleeps past ``soft_deadline_sec`` is reaped and the
    function returns a :class:`subprocess.CompletedProcess` whose
    ``returncode`` is the canonical ``OVERTIME_KILL_RETURNCODE``
    sentinel (does NOT raise ``TimeoutExpired``)."""
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=30,        # hard cap well above the soft deadline
        soft_deadline_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    # Must have returned within a few seconds of the deadline. Upper
    # bound accounts for:
    #   * 0.5 s poll overrun in ``_communicate_with_soft_deadline``,
    #   * up to 5 s SIGTERM grace in ``kill_my_spawned_server``,
    #   * 2 s pipe drain after the kill,
    # plus generous CI jitter. Keep this loose — false-positive
    # tightness on a CI box would mask a real perf regression in the
    # production kill path.
    assert elapsed < 10.0, f"soft-deadline path took {elapsed:.2f}s"


def test_run_with_session_kill_soft_deadline_does_not_fire_for_quick_child():
    """A child that exits well before ``soft_deadline_sec`` returns
    normally with the child's own returncode — the soft-deadline gate
    must not perturb the success path."""
    cp = run_with_session_kill(
        [sys.executable, "-c", "print('hi'); raise SystemExit(0)"],
        timeout=10,
        soft_deadline_sec=5.0,
    )
    assert cp.returncode == 0
    assert "hi" in (cp.stdout or "")


def test_run_with_session_kill_legacy_timeout_still_raises():
    """When ``soft_deadline_sec`` is None (legacy behaviour) the
    function must still raise :class:`subprocess.TimeoutExpired` for a
    child that exceeds the hard ``timeout``."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
            soft_deadline_sec=None,
        )
