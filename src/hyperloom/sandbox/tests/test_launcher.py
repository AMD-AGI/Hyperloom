# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launcher: it reports its command's exit code, holds what the command left running, and then ends."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.common import proctree
from hyperloom.sandbox.launcher import StatusPipe, launcher_argv

#: How long a launcher with nothing left to hold may take to end.
_ENDS_WITHIN_SEC = 5.0


def _launch(argv: list[str], cwd: Path) -> tuple[subprocess.Popen[bytes], StatusPipe]:
    read_fd, write_fd = os.pipe()
    try:
        proc = subprocess.Popen(
            launcher_argv(sys.executable, write_fd, argv),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
    return proc, StatusPipe(read_fd)


def _parent(pid: int) -> int:
    identity = proctree.proc_identity(pid)
    assert identity is not None, f"pid {pid} is gone"
    return identity[0]


def test_a_launcher_whose_command_ended_and_left_nothing_ends_by_itself(tmp_path: Path) -> None:
    proc, status = _launch(["sh", "-c", "exit 3"], tmp_path)
    try:
        assert status.wait_exit(proc, _ENDS_WITHIN_SEC) == 3
        assert proc.wait(timeout=_ENDS_WITHIN_SEC) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_a_launcher_holds_an_orphan_its_command_left_until_it_ends_then_ends(tmp_path: Path) -> None:
    pid_file = tmp_path / "orphan.pid"
    proc, status = _launch(["sh", "-c", f"sleep 300 & echo $! > {pid_file}"], tmp_path)
    orphan = 0
    try:
        assert status.wait_exit(proc, _ENDS_WITHIN_SEC) == 0
        orphan = int(pid_file.read_text())
        assert _parent(orphan) == proc.pid, "the orphan is re-parented to the launcher, a child subreaper"
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=0.5)
        os.kill(orphan, signal.SIGTERM)
        assert proc.wait(timeout=_ENDS_WITHIN_SEC) == 0
        assert not proctree.running(orphan)
    finally:
        if orphan and proctree.running(orphan):
            os.kill(orphan, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
