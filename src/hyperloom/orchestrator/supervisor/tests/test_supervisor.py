# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two failure shapes, and that an escalation is reachable in both.

A coordinator that died and a coordinator whose tick is wedged look identical
from inside the session -- nothing happens -- and they need different responses,
because in the second case there is still a process that can be asked to stop
itself and in the first there is not.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess  # nosec B404 - spawns a process purely so its pid can be signalled
import sys
import time

import pytest

from hyperloom.common.proctree import running
from hyperloom.inference_optimizer.session.session_paths import (
    optimizer_lock_path,
    reports_dir,
    supervisor_status_path,
)
from hyperloom.orchestrator.supervisor import store, tick_stall_sec
from hyperloom.orchestrator.supervisor.watch import (
    ALIVE,
    DEAD,
    DIED_STOP_REASON,
    UNKNOWN,
    WEDGED,
    WEDGED_STOP_REASON,
    Supervisor,
)

_NOW = 1_000_000.0


def _own_the_session(session_dir, pid: int) -> None:
    """Write an optimizer-lock owner document naming ``pid``."""
    path = optimizer_lock_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "hostname": socket.gethostname()}), encoding="utf-8")


def _a_pid_that_is_gone() -> int:
    """Return a pid that named a process and no longer does."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603
    child.wait(timeout=30)
    return child.pid


def _a_pid_that_is_running(request, *, deaf: bool = False) -> int:
    """Return the pid of a process that is already sleeping when this returns.

    Args:
        request: The pytest request, used to kill the child afterwards.
        deaf: Whether the child ignores the stop signal, standing in for a
            coordinator too wedged to run its own stop path.

    Returns:
        int: The child's pid, once it has announced it is ready to be signalled.
    """
    ignore = "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if deaf else ""
    program = f"import signal, sys, time; {ignore}sys.stdout.write('r'); sys.stdout.flush(); time.sleep(300)"
    child = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE)  # nosec B603
    request.addfinalizer(child.kill)
    assert child.stdout is not None
    assert child.stdout.read(1) == b"r"
    return child.pid


def _supervisor(session_dir, **kw) -> Supervisor:
    """A supervisor over ``session_dir`` with a frozen clock."""
    kw.setdefault("now", lambda: _NOW)
    return Supervisor(session_dir, **kw)


def test_the_stall_window_always_fits_inside_the_session_it_watches():
    """A window the session cannot outlast is a watch that never fires."""
    two_hours = 2 * 3600.0
    assert tick_stall_sec(two_hours) <= two_hours / 2
    assert tick_stall_sec(3600.0) <= 3600.0 / 2
    # And never so short that a slow tick reads as a stopped one.
    assert tick_stall_sec(60.0) == tick_stall_sec(3600.0)


def test_a_ticking_coordinator_is_left_alone(tmp_path):
    """A live pid plus an advancing tick is the only reading that means healthy."""
    _own_the_session(tmp_path, os.getpid())
    store.stamp_tick(tmp_path, tick=7, now_unix=_NOW - 5.0)

    observation = _supervisor(tmp_path).observe()

    assert observation.verdict == ALIVE
    assert observation.tick == 7


def test_a_live_pid_with_a_stopped_tick_is_wedged(tmp_path):
    """A pid says the process exists; only the tick says the loop is running."""
    _own_the_session(tmp_path, os.getpid())
    store.stamp_tick(tmp_path, tick=7, now_unix=_NOW - 9_999.0)

    observation = _supervisor(tmp_path, tick_stall_sec=100.0).observe()

    assert observation.verdict == WEDGED


def test_an_owner_on_another_host_is_never_diagnosed(tmp_path):
    """A pid means nothing off the machine that issued it."""
    path = optimizer_lock_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": 4242, "hostname": "some-other-pod"}), encoding="utf-8")

    assert _supervisor(tmp_path).observe().verdict == UNKNOWN


def test_a_gone_pid_is_dead(tmp_path):
    """The pid the lock names is the authoritative one."""
    dead = _a_pid_that_is_gone()
    _own_the_session(tmp_path, dead)

    assert _supervisor(tmp_path).observe().verdict == DEAD


@pytest.mark.asyncio
async def test_a_wedged_coordinator_is_asked_to_stop_on_the_one_channel_that_reaches_it(tmp_path, request):
    """A signal is recorded by the drain thread; nothing the loop reads is."""
    pid = _a_pid_that_is_running(request)
    _own_the_session(tmp_path, pid)
    store.stamp_tick(tmp_path, tick=1, now_unix=_NOW - 9_999.0)
    supervisor = _supervisor(tmp_path, tick_stall_sec=100.0)

    await supervisor.act(supervisor.observe())

    assert [r.split(":")[0] for r in supervisor.report.asked] == [WEDGED_STOP_REASON]
    assert "tick 1 last advanced" in supervisor.report.asked[0]
    _wait_for_exit(pid)


@pytest.mark.asyncio
async def test_the_stop_is_asked_for_once_and_then_waited_on(tmp_path, request):
    """A signal per poll would be a stop the coordinator never gets to run."""
    pid = _a_pid_that_is_running(request, deaf=True)
    _own_the_session(tmp_path, pid)
    store.stamp_tick(tmp_path, tick=1, now_unix=_NOW - 9_999.0)
    supervisor = _supervisor(tmp_path, tick_stall_sec=100.0, stop_grace_sec=1_000.0)

    await supervisor.act(supervisor.observe())
    await supervisor.act(supervisor.observe())

    assert len(supervisor.report.asked) == 1
    assert not supervisor.report.terminal_path


@pytest.mark.asyncio
async def test_a_coordinator_that_ignores_the_stop_is_left_diagnosed(tmp_path, request):
    """A stop that goes unanswered is reported, not escalated to a kill."""
    pid = _a_pid_that_is_running(request, deaf=True)
    _own_the_session(tmp_path, pid)
    store.stamp_tick(tmp_path, tick=1, now_unix=_NOW - 9_999.0)
    supervisor = _supervisor(tmp_path, tick_stall_sec=100.0, stop_grace_sec=0.0)

    await supervisor.act(supervisor.observe())
    await supervisor.act(supervisor.observe())

    assert not supervisor.report.terminal_path
    status = json.loads(supervisor_status_path(tmp_path).read_text(encoding="utf-8"))
    assert status["verdict"] == WEDGED


@pytest.mark.asyncio
async def test_a_dead_coordinator_gets_a_terminal_artifact(tmp_path):
    """Recording an already-dead session is the one thing left to do for it."""
    dead = _a_pid_that_is_gone()
    _own_the_session(tmp_path, dead)
    supervisor = _supervisor(tmp_path)

    done = await supervisor.act(supervisor.observe())

    assert done is True
    final = json.loads((reports_dir(tmp_path) / "final.json").read_text(encoding="utf-8"))
    assert final["producer"] == "supervisor"
    assert final["stop_reason"] == DIED_STOP_REASON
    assert final["supervisor"]["coordinator_pid"] == dead


@pytest.mark.asyncio
async def test_the_supervisors_terminal_never_replaces_a_real_report(tmp_path):
    """One terminal artifact, and the fuller writer outranks the fallback."""
    target = reports_dir(tmp_path) / "final.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"report_complete": True, "stop_reason": "target_reached"}), encoding="utf-8")
    dead = _a_pid_that_is_gone()
    _own_the_session(tmp_path, dead)

    await _supervisor(tmp_path).act(_supervisor(tmp_path).observe())

    assert json.loads(target.read_text(encoding="utf-8"))["stop_reason"] == "target_reached"


def test_the_coordinators_fallback_never_replaces_the_supervisors(tmp_path):
    """The supervisor watched the process end; the coordinator's fallback did not."""
    from hyperloom.inference_optimizer.breakdown import (
        FINAL_PRODUCER_COORDINATOR,
        FINAL_PRODUCER_SUPERVISOR,
        write_minimal_final_json,
    )

    write_minimal_final_json(tmp_path, producer=FINAL_PRODUCER_SUPERVISOR, extra={"stop_reason": DIED_STOP_REASON})
    write_minimal_final_json(tmp_path, producer=FINAL_PRODUCER_COORDINATOR)

    final = json.loads((reports_dir(tmp_path) / "final.json").read_text(encoding="utf-8"))
    assert final["producer"] == FINAL_PRODUCER_SUPERVISOR


def _wait_for_exit(pid: int, timeout: float = 30.0) -> None:
    """Block until ``pid`` is gone, failing the test if it never goes."""
    deadline = time.monotonic() + timeout
    while running(pid):
        if time.monotonic() > deadline:
            raise AssertionError(f"pid {pid} ignored the stop it was asked for")
        time.sleep(0.05)
