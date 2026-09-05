# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What each reap unit reaches, and what a success by it is allowed to claim.

The three units are not interchangeable, and the tests are written around the
difference rather than around the shared happy path: the weakest one reaches a
child that left its process group only because the tree was enumerated first,
and even then its success is not proof.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - starts and kills sleeps, to exercise the reaper
import sys
import time

import pytest

from hyperloom.common.proctree import running
from hyperloom.orchestrator.bringup.reap import (
    BACKEND_PROCESS_GROUP,
    CLAIM_REACHABLE,
    REAP_KILLED,
    REAP_UNOBSERVABLE,
    ProcessGroupReaper,
    pid_target,
    select_reaper,
)

_ESCAPING_PARENT = (
    "import subprocess, sys, time; "
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=True); "
    "print(p.pid, flush=True); "
    "time.sleep(300)"
)


def _alive(pid: int) -> bool:
    """Whether a pid names a live process.

    Zombie-aware, like the reaper's own reading: this test's parent process is
    a child of the test runner, so it lingers unreaped after it is killed and a
    bare signal-0 probe would call it alive forever.
    """
    return running(pid)


def _wait_gone(pids, timeout: float = 5.0) -> bool:
    """Wait for every pid to disappear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in pids):
            return True
        time.sleep(0.05)
    return False


def test_the_unit_never_claims_proof():
    """A process group can be left at will, so a success covers only what it listed."""
    unit = select_reaper()
    assert unit.name == BACKEND_PROCESS_GROUP
    assert unit.claim == CLAIM_REACHABLE


@pytest.mark.asyncio
async def test_the_process_group_unit_reaches_a_child_that_left_the_group():
    """The pre-collected tree is the only thing that reaches an escapee.

    An engine that puts its workers in their own session is the shape that once
    survived a bare group kill and kept holding a port.
    """
    parent = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", _ESCAPING_PARENT],
        stdout=subprocess.PIPE,
        text=True,
    )
    escapee = int(parent.stdout.readline().strip())
    try:
        assert os.getpgid(escapee) != os.getpgid(parent.pid), "the child must have left the group"

        reap = await ProcessGroupReaper(confirm_window_sec=5.0).reap(
            pid_target("holder", [parent.pid]),
            now_unix=1000.0,
        )

        assert reap.outcome == REAP_KILLED
        assert _wait_gone([parent.pid, escapee])
    finally:
        for pid in (parent.pid, escapee):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        parent.wait(timeout=5)


@pytest.mark.asyncio
async def test_a_process_group_success_is_never_proof():
    """It reached everything it could list, which is not the same as everything."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])  # nosec B603
    try:
        reap = await ProcessGroupReaper(confirm_window_sec=5.0).reap(
            pid_target("holder", [child.pid]),
            now_unix=1000.0,
        )
        assert reap.confirmed_unix is not None
        assert reap.claim == CLAIM_REACHABLE
    finally:
        child.kill()
        child.wait()


@pytest.mark.asyncio
async def test_a_target_with_no_recorded_process_is_unobservable():
    """Nothing recorded is not evidence of death; it is evidence of nothing."""
    reap = await ProcessGroupReaper().reap(pid_target("holder", []), now_unix=1000.0)
    assert reap.outcome == REAP_UNOBSERVABLE
    assert reap.confirmed_unix is None


@pytest.mark.asyncio
async def test_the_reaper_never_targets_itself():
    """Killing the process doing the reaping would end the session mid-repair."""
    target = pid_target("holder", [os.getpid()])
    assert target.pids == frozenset()
