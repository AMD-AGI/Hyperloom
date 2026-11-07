# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A reap of a launcher that runs below the process that was started, as a confined sandbox's runs inside
bubblewrap: what its command left stays in the tree through the grace, and the grace does not wait for the launcher."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from hyperloom.common import proctree
from hyperloom.sandbox.launcher import launcher_argv
from hyperloom.sandbox.reaper import kill_launchers


def _wait(predicate: Callable[[], object], seconds: float = 30.0) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, "timed out waiting"
        time.sleep(0.05)


TRAPPED = "trap 'sleep 0.5; echo stopped > {marker}; exit 0' TERM; exec -a {name}-child sleep 1000 & wait"


def test_a_launcher_below_the_started_process_is_held_so_a_slow_stop_keeps_its_grace(tmp_path: Path) -> None:
    """The started process is not the launcher (bubblewrap's is not): the launcher below it is frozen, so a command
    that stops slowly stays in the tree through its grace, and the grace ends when it has stopped."""
    marker, name = tmp_path / "stopped", f"hl-test-{os.getpid()}-{time.monotonic_ns()}"
    read_fd, write_fd = os.pipe()
    command = ["/bin/bash", "-c", TRAPPED.format(marker=marker, name=name)]
    shell = ["/bin/sh", "-c", '"$@"; exit 3', "outer", *launcher_argv(sys.executable, write_fd, command)]
    started = subprocess.Popen(shell, pass_fds=(write_fd,), start_new_session=True)
    os.close(write_fd)
    try:
        _wait(lambda: _named(f"{name}-child"))
        took = time.monotonic()
        left = kill_launchers([started], grace_sec=10.0)
        took = time.monotonic() - took
    finally:
        os.close(read_fd)
        if started.poll() is None:
            os.killpg(started.pid, signal.SIGKILL)
            started.wait()

    assert left.clean and not _named(f"{name}-child")
    assert marker.read_text().strip() == "stopped", "the command's TERM handler ran to its end before the SIGKILL"
    assert took < 5.0, f"the grace waited for the held launcher ({took:.1f} s)"


def _named(name: str) -> list[int]:
    """The running processes of this host whose argv[0] is ``name``."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")[0] == name.encode():
                found.append(int(entry))
        except OSError:
            continue
    return [pid for pid in found if proctree.running(pid)]
