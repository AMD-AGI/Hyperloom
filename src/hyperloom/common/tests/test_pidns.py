# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A PID namespace as the lifetime boundary for one task's processes."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from pathlib import Path

import pytest

from hyperloom.common import pidns


def _supported() -> bool:
    ok, _ = pidns.unshare_supported()
    return ok


def test_a_reused_pid_is_not_the_recorded_supervisor():
    """A bare pid cannot name an incarnation, and after a restart it often will not.

    Releasing a lane because "the pid is there" would be as wrong as releasing it
    because the pid is gone: the number may now belong to something unrelated.
    Start time settles it, and a mismatch is proof rather than suspicion.
    """
    live = pidns.identify_supervisor(os.getpid())
    assert live is not None

    impostor = pidns.SupervisorIdentity(pid=live.pid, start_time=live.start_time + 1, ns_inode=live.ns_inode)

    assert pidns.supervisor_state(live) is True
    assert pidns.supervisor_state(impostor) is False


def test_a_different_namespace_on_the_same_pid_is_not_it_either():
    """The namespace inode pins WHICH namespace that process is init of."""
    live = pidns.identify_supervisor(os.getpid())
    assert live is not None
    elsewhere = pidns.SupervisorIdentity(pid=live.pid, start_time=live.start_time, ns_inode=live.ns_inode + 1)

    assert pidns.supervisor_state(elsewhere) is False


def test_a_vanished_supervisor_reads_as_definitely_gone():
    """Its exit is what makes the lane releasable, so it must be decidable."""
    gone = pidns.SupervisorIdentity(pid=0x7FFFFFFF, start_time=1, ns_inode=1)

    assert pidns.supervisor_state(gone) is False


def test_an_unreadable_proc_is_not_evidence_of_absence(monkeypatch: pytest.MonkeyPatch):
    """None, not False. Retaining a lane costs time; releasing one early corrupts.

    This is the distinction the refuted designs kept collapsing: they treated
    "I could not find it" as "it is not there".
    """
    live = pidns.identify_supervisor(os.getpid())
    assert live is not None
    monkeypatch.setattr(pidns, "_start_time", lambda pid: None)

    assert pidns.supervisor_state(live) is None


def test_identity_of_a_dead_process_is_not_invented():
    """Nothing may be recorded for a process that is already gone."""
    assert pidns.identify_supervisor(0x7FFFFFFF) is None


@pytest.mark.skipif(not _supported(), reason="host refuses unprivileged PID namespaces")
def test_a_setsid_survivor_dies_with_its_namespace_init(tmp_path: Path):
    """The property the design rests on, against the real kernel.

    The survivor reproduces every escape that defeated the earlier proofs: its
    root exits, it is reparented, and it calls setsid twice, so it shares no
    process group or session with anything recorded at spawn. It cannot leave
    the PID namespace, and when init exits the kernel kills it.
    """
    mark = tmp_path / "survivor"
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        if libc.unshare(pidns.CLONE_NEWUSER | pidns.CLONE_NEWPID) != 0:
            os._exit(90)
        init = os.fork()
        if init == 0:
            kid = os.fork()
            if kid == 0:
                os.setsid()
                if os.fork() != 0:
                    os._exit(0)
                os.setsid()
                host = 0
                for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                    if line.startswith("NSpid:"):
                        host = int(line.split()[1])
                mark.write_text(str(host), encoding="utf-8")
                time.sleep(60)
                os._exit(0)
            os.waitpid(kid, 0)
            os._exit(0)  # init exits while the survivor sleeps
        os.waitpid(init, 0)
        os._exit(0)

    os.waitpid(pid, 0)
    for _ in range(30):
        if mark.exists():
            break
        time.sleep(0.1)
    assert mark.exists(), "the survivor never reported itself"
    survivor = int(mark.read_text())

    # Give the kernel its teardown; then the pid must be gone, not merely quiet.
    for _ in range(50):
        if not Path(f"/proc/{survivor}").exists():
            break
        time.sleep(0.1)
    assert not Path(f"/proc/{survivor}").exists(), "a setsid'd survivor outlived its namespace"
