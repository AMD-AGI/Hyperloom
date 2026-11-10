# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launcher: it reports its command's exit code, holds what the command left running, and then ends."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
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


#: How many tests the long-running command below stands for, each leaving an orphan behind.
_SESSION_TESTS = 5


def _zombies_under(pid: int) -> list[int]:
    zombies = []
    for descendant, _ in proctree.descendants(pid):
        try:
            stat = Path(f"/proc/{descendant}/stat").read_text(encoding="utf-8")
        except OSError:
            continue
        if stat.rsplit(")", 1)[-1].split()[0] == "Z":
            zombies.append(descendant)
    return zombies


def test_a_launcher_reaps_each_orphan_while_its_command_still_runs(tmp_path: Path) -> None:
    orphans_file = tmp_path / "orphans"
    command_file = tmp_path / "command.pid"
    # A session: each test runs a process whose child outlives it, then exits; the session itself goes on.
    script = (
        f'for i in $(seq {_SESSION_TESTS}); do sh -c "sleep 0.2 & echo \\$! >> {orphans_file}"; done; '
        f"echo $$ > {command_file}; exec sleep 300"
    )
    proc, status = _launch(["sh", "-c", script], tmp_path)
    try:
        deadline = time.monotonic() + _ENDS_WITHIN_SEC
        orphans: list[int] = []
        while time.monotonic() < deadline:
            if command_file.exists() and command_file.read_text().strip():
                orphans = [int(line) for line in orphans_file.read_text().split()]
                if not any(Path(f"/proc/{orphan}").exists() for orphan in orphans):
                    break
            time.sleep(0.05)
        assert len(orphans) == _SESSION_TESTS
        assert _zombies_under(proc.pid) == [], "every orphan that ended was reaped while the command still runs"
        assert not any(Path(f"/proc/{orphan}").exists() for orphan in orphans)
        assert proc.poll() is None, "the launcher holds its command, which still runs"
        os.kill(int(command_file.read_text()), signal.SIGTERM)
        assert status.wait_exit(proc, _ENDS_WITHIN_SEC) == -signal.SIGTERM
        assert proc.wait(timeout=_ENDS_WITHIN_SEC) == 0
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
