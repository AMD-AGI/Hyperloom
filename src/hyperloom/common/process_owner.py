# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Linux subprocess owner; acknowledge cleanup only after reaping every child.

Executed by absolute path with only stdlib dependencies, including when a
specialist supplies an environment that cannot import Hyperloom. A dedicated
subreaper keeps double-forked and setsid children inside this ownership domain.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


TERM_GRACE_SEC = 5.0


def _ack(fd: int, message: bytes) -> None:
    try:
        os.write(fd, message)
    except BrokenPipeError:
        pass  # The owner still owes cleanup if its caller has died.


def supervise(cmd: list[str], ack_fd: int) -> int:
    """Run one command and reap its entire adoption domain before returning."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        _ack(ack_fd, b"C")
        raise OSError(ctypes.get_errno(), "cannot establish subprocess ownership")
    stopping = False
    force = False

    def stop(_sig: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    def escalate(_sig: int, _frame: object) -> None:
        nonlocal force, stopping
        force = stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGUSR1, escalate)
    os.set_inheritable(ack_fd, False)

    def parent_death_signal() -> None:
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "cannot arm parent-death cleanup")

    try:
        proc = subprocess.Popen(
            cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=parent_death_signal
        )
    except (OSError, subprocess.SubprocessError):
        _ack(ack_fd, b"C")
        raise
    _ack(ack_fd, b"R")
    root_code = None
    deadline = None
    signalled: dict[int, int] = {}
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                signalled.pop(pid, None)
                if pid == proc.pid:
                    root_code = os.waitstatus_to_exitcode(status)
                    proc.returncode = root_code
                    stopping = True
        except ChildProcessError:
            _ack(ack_fd, b"C")
            return root_code if root_code is not None else 1
        if stopping:
            if deadline is None:
                deadline = time.monotonic() + TERM_GRACE_SEC
            sig = signal.SIGKILL if force or time.monotonic() >= deadline else signal.SIGTERM
            # These are our unreaped children: their PIDs cannot be reused
            # between this read and kill. Their orphans are adopted here too.
            for child in children_file.read_text().split():
                pid = int(child)
                if signalled.get(pid) == sig:
                    continue
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    continue
                signalled[pid] = sig
        time.sleep(0.02)


if __name__ == "__main__":
    code = supervise(sys.argv[2:], int(sys.argv[1]))
    if code < 0:
        signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    sys.exit(code)
