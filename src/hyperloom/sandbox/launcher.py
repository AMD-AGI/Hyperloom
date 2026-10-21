# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-process launcher that keeps every orphan inside the sandbox's process tree.

Every top-level process of a sandbox runs under a launcher. The launcher marks
itself a child subreaper (``PR_SET_CHILD_SUBREAPER``), so a descendant that
detaches (double fork, ``setsid``, an emptied environment) and loses its parent
is re-parented to the launcher, never to init. The launcher outlives its command
until the sandbox's reaper kills it, so the reaper can always enumerate every
process the sandbox ever started by walking from the launcher.

A launcher can also give its command a private view of framework directories: it unshares its mount
namespace, mounts a size-capped tmpfs for the writes and puts an overlay on each root, so the command and everything
it starts read the root as it was and write only into the tmpfs, which goes away with the last process of the
namespace. A mount below a root (a volume) is put back on top of the overlay and stays shared. This is what keeps one
local test from changing the framework install the next test reads.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from dataclasses import asdict, dataclass

__all__ = ["LAUNCHER_SOURCE", "OverlaySetup", "SetupFailed", "StatusPipe", "launcher_argv", "read_setup_answer"]

SETUP_OK = b"ok\n"
_SETUP_TIMEOUT_SEC = 60.0

#: Self-contained (standard library only) so it runs under any python3 and
#: under another account.
LAUNCHER_SOURCE = r"""
import ctypes, json, os, signal, sys, time
status_fd = int(sys.argv[1])
setup = json.loads(sys.argv[2]) if sys.argv[2] else None
argv = sys.argv[3:]


def isolate(setup):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]

    def check(rc, what):
        if rc != 0:
            err = ctypes.get_errno()
            raise OSError("%s: %s" % (what, os.strerror(err)))

    roots = [os.path.realpath(root) for root in setup["roots"]]
    with open("/proc/self/mountinfo") as handle:
        mounted = [line.split()[4].replace("\\040", " ") for line in handle]
    kept = []
    for root in roots:
        if not os.path.isdir(root):
            raise OSError("framework root %s is not a directory" % root)
        below = [m for m in mounted if m.startswith(root.rstrip("/") + "/")]
        kept += [m for m in below if not any(m.startswith(o + "/") for o in below)]
    check(libc.unshare(0x00020000), "unshare(CLONE_NEWNS)")
    check(libc.mount(b"none", b"/", None, 0x4000 | 0x40000, None), "make the mounts private")
    scratch = setup["scratch"]
    options = ("size=%dm,mode=0700" % setup["upper_mib"]).encode()
    check(libc.mount(b"hl-framework", scratch.encode(), b"tmpfs", 0, options), "mount the tmpfs for the writes")
    held = []
    for index, point in enumerate(kept):
        hold = "%s/keep%d" % (scratch, index)
        os.mkdir(hold)
        check(libc.mount(point.encode(), hold.encode(), None, 0x1000 | 0x4000, None), "hold the mount at " + point)
        held.append((hold, point))
    for index, root in enumerate(roots):
        upper, work = "%s/upper%d" % (scratch, index), "%s/work%d" % (scratch, index)
        os.mkdir(upper)
        os.mkdir(work)
        info = os.stat(root)
        os.chown(upper, info.st_uid, info.st_gid)
        os.chmod(upper, info.st_mode & 0o7777)
        data = ("lowerdir=%s,upperdir=%s,workdir=%s" % (root, upper, work)).encode()
        check(libc.mount(b"hl-framework", root.encode(), b"overlay", 0, data), "mount an overlay on " + root)
    for hold, point in held:
        check(libc.mount(hold.encode(), point.encode(), None, 0x1000 | 0x4000, None), "put back the mount at " + point)
    if setup.get("drop"):
        uid, gid = setup["drop"]
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)


if setup is not None:
    ack = setup["ack_fd"]
    try:
        isolate(setup)
    except Exception as exc:
        os.write(ack, ("framework isolation failed: %s\n" % exc).encode()[:1000])
        os._exit(126)
    os.write(ack, b"ok\n")
    os.close(ack)
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


class SetupFailed(Exception):
    """The launcher could not give its command the private framework view and did not start it."""


@dataclass(frozen=True)
class OverlaySetup:
    """What the launcher does before it starts its command: overlay ``roots``, keeping the writes in a tmpfs.

    ``scratch`` is an existing directory the tmpfs is mounted on, ``upper_mib`` the size cap of that tmpfs, ``ack_fd``
    the write end of the pipe the launcher answers on and ``drop`` the ``(uid, gid)`` the launcher becomes once the
    mounts are made (none: it stays the account it was started as).
    """

    roots: tuple[str, ...]
    scratch: str
    upper_mib: int
    ack_fd: int
    drop: tuple[int, int] | None = None


def launcher_argv(python: str, status_fd: int, argv: list[str], *, setup: OverlaySetup | None = None) -> list[str]:
    """Wrap ``argv`` so it runs under a launcher that reports its exit code on ``status_fd``."""
    document = "" if setup is None else json.dumps(asdict(setup))
    return [python, "-I", "-c", LAUNCHER_SOURCE, str(status_fd), document, *argv]


def read_setup_answer(read_fd: int) -> None:
    """Read the launcher's answer to its setup from ``read_fd`` (which this closes); raise when it did not succeed.

    Raises:
        SetupFailed: With the launcher's own message, or when it ended without an answer.
    """
    with os.fdopen(read_fd, "rb") as answer:
        ready, _, _ = select.select([answer], [], [], _SETUP_TIMEOUT_SEC)
        text = answer.read(1000) if ready else b"the launcher did not answer in time"
    if text != SETUP_OK:
        raise SetupFailed(text.decode("utf-8", errors="replace").strip() or "the launcher ended before it answered")


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
