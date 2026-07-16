# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the from-ready soft-deadline caliber.

When a ``server.log`` is available the explore overtime soft deadline measures
only the post-ready phase: the clock starts at the server-ready marker,
excluding pre-ready boot / weight load / first-request recompile. Opt out via
``INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY=0``.
"""

from __future__ import annotations

import sys
import time

from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    OVERTIME_KILL_RETURNCODE,
    run_with_session_kill,
)

# Long stall grace so the detok-stall watchdog never interferes here.
_LONG_STALL_GRACE = 3600.0


def test_from_ready_excludes_pre_ready_phase(tmp_path):
    """Pre-ready time is NOT counted: a child that spends > deadline BEFORE the
    ready marker but only a little AFTER it finishes normally."""
    log_path = tmp_path / "server.log"
    # 3s pre-ready boot, then ready, then ~1s post-ready client.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('INFO loading weights\\n'); f.flush()\n"
        "time.sleep(3)\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "time.sleep(1)\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        soft_deadline_sec=2.0,
        server_log_path=str(log_path),
        detok_stall_grace_sec=_LONG_STALL_GRACE,
    )
    elapsed = time.monotonic() - start
    # Post-ready (~1s) < 2.0s deadline -> not killed despite ~4s wall-clock.
    assert cp.returncode == 0, f"unexpected returncode={cp.returncode}"
    assert elapsed >= 3.0, f"child should have run its full pre-ready sleep, got {elapsed:.2f}s"


def test_from_ready_fires_after_ready(tmp_path):
    """Post-ready overrun IS killed: once ready, exceeding the deadline in the
    client phase reaps the tree with the overtime sentinel."""
    log_path = tmp_path / "server.log"
    # Ready immediately, then a post-ready run that overruns the deadline.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "time.sleep(30)\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=60,
        soft_deadline_sec=1.0,
        server_log_path=str(log_path),
        detok_stall_grace_sec=_LONG_STALL_GRACE,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    # Killed shortly after ready, not at 30s.
    assert elapsed < 15.0, f"from-ready soft deadline took {elapsed:.2f}s"


def test_opt_out_reverts_to_from_spawn(tmp_path, monkeypatch):
    """With INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY=0 the legacy from-spawn
    clock applies even with a server.log: pre-ready time counts and trips."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY", "0")
    log_path = tmp_path / "server.log"
    # Long pre-ready phase; from-spawn overruns the 1s deadline.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('INFO loading weights\\n'); f.flush()\n"
        "time.sleep(30)\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=60,
        soft_deadline_sec=1.0,
        server_log_path=str(log_path),
        detok_stall_grace_sec=_LONG_STALL_GRACE,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    assert elapsed < 15.0, f"from-spawn opt-out took {elapsed:.2f}s"


def test_no_server_log_uses_from_spawn(tmp_path):
    """Without a server.log the soft deadline is the legacy from-spawn clock."""
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=60,
        soft_deadline_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    assert elapsed < 15.0, f"from-spawn (no server.log) took {elapsed:.2f}s"
