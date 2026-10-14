# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-process launcher that keeps every orphan inside the sandbox's process tree.

Every top-level process of a sandbox runs under a launcher. The launcher marks
itself a child subreaper (``PR_SET_CHILD_SUBREAPER``), so a descendant that
detaches (double fork, ``setsid``, an emptied environment) and loses its parent
is re-parented to the launcher, never to init. The launcher outlives its command
until the sandbox's reaper kills it, so the reaper can always enumerate every
process the sandbox ever started by walking from the launcher.
"""

from __future__ import annotations

import os
import select
import subprocess
import time

__all__ = ["LAUNCHER_SOURCE", "StatusPipe", "launcher_argv"]

#: Self-contained (standard library only) so it runs under any python3 and
#: under another account.
LAUNCHER_SOURCE = r"""
import ctypes, os, signal, sys, time
status_fd = int(sys.argv[1])
argv = sys.argv[2:]
try:
    ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
except OSError:
    pass
signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # a closed status pipe must not kill the launcher
child = os.fork()
if child == 0:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    os.close(status_fd)
    try:
        os.execvp(argv[0], argv)
    except OSError:
        os._exit(127)
_, raw = os.waitpid(child, 0)
code = os.waitstatus_to_exitcode(raw)
try:
    os.write(status_fd, ("%d\n" % code).encode())
    os.close(status_fd)
except OSError:
    pass
null = os.open(os.devnull, os.O_RDWR)
for fd in (0, 1, 2):
    os.dup2(null, fd)  # let the readers of the command's output see EOF
while True:
    try:
        os.wait()  # reap re-parented orphans as they exit
    except ChildProcessError:
        time.sleep(0.2)
"""


def launcher_argv(python: str, status_fd: int, argv: list[str]) -> list[str]:
    """Wrap ``argv`` so it runs under a launcher that reports its exit code on ``status_fd``."""
    return [python, "-I", "-c", LAUNCHER_SOURCE, str(status_fd), *argv]


class StatusPipe:
    """The read end of a launcher's status pipe: its command's exit code, when it ends."""

    def __init__(self, read_fd: int) -> None:
        self._fd: int | None = read_fd
        self._buf = b""

    def wait_exit(self, proc: subprocess.Popen[bytes], timeout_sec: float | None) -> int | None:
        """Wait for the command to exit.

        Args:
            proc: The launcher process.
            timeout_sec: The longest to wait; None waits as long as the launcher lives.

        Returns:
            int | None: The exit code, or None if the timeout ran out (command still running)
            or the launcher died before reporting (see ``launcher_died``).

        Raises:
            TimeoutError: When ``timeout_sec`` ran out.
        """
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while self._fd is not None:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._fd], [], [], left)
            if not ready:
                raise TimeoutError
            chunk = os.read(self._fd, 64)
            if not chunk:  # EOF: the launcher is gone without a report
                self.close()
                return None
            self._buf += chunk
            if b"\n" in self._buf:
                self.close()
                return int(self._buf.split(b"\n", 1)[0])
        return None

    def close(self) -> None:
        """Release the read end."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
