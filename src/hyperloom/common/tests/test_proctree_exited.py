# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A process that has exited is not running, whoever may signal it, and an exited child is waited only when named."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.common import proctree


def _state(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()[0]


@pytest.fixture
def zombie() -> Iterator[subprocess.Popen[bytes]]:
    """A real zombie: a child of this process that exited and is not waited for yet."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.monotonic() + 10
    while _state(child.pid) != "Z":
        assert time.monotonic() < deadline, "the child did not exit"
        time.sleep(0.02)
    yield child
    child.wait()


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    yield child
    child.kill()
    child.wait()


def _signal_refused(monkeypatch: pytest.MonkeyPatch, pid: int) -> None:
    """``pid`` answers every signal as another account's process does: it exists and may not be signalled."""
    real_kill = os.kill

    def _kill(target: int, sig: int) -> None:
        if target == pid:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        real_kill(target, sig)

    monkeypatch.setattr(os, "kill", _kill)


def test_a_zombie_is_not_running(zombie: subprocess.Popen[bytes]) -> None:
    assert proctree.running(zombie.pid) is False


def test_a_zombie_this_process_may_not_signal_is_not_running(
    zombie: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    _signal_refused(monkeypatch, zombie.pid)
    assert proctree.running(zombie.pid) is False


def test_a_live_process_this_process_may_not_signal_is_running(
    sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    _signal_refused(monkeypatch, sleeper.pid)
    assert proctree.running(sleeper.pid) is True


def test_a_process_hidden_from_proc_still_runs_while_a_signal_finds_it(
    sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``hidepid`` another account's ``/proc`` entry is missing while the process runs."""

    def _hidden(pid: int) -> list[str]:
        raise FileNotFoundError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr(proctree, "_stat_fields", _hidden)
    _signal_refused(monkeypatch, sleeper.pid)
    assert proctree.running(sleeper.pid) is True


def test_a_process_reaped_between_the_signal_and_the_read_is_not_running(
    sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _reaped_meanwhile(pid: int) -> list[str]:
        sleeper.kill()
        sleeper.wait()
        raise FileNotFoundError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr(proctree, "_stat_fields", _reaped_meanwhile)
    assert proctree.running(sleeper.pid) is False


def test_reap_exited_children_waits_a_named_exited_child(zombie: subprocess.Popen[bytes]) -> None:
    identity = proctree.proc_identity(zombie.pid)
    assert identity is not None and identity[0] == os.getpid()
    assert proctree.reap_exited_children([(zombie.pid, identity[1])]) == 1
    assert not Path(f"/proc/{zombie.pid}").exists()


def test_reap_exited_children_leaves_another_incarnation_and_a_live_child(
    zombie: subprocess.Popen[bytes], sleeper: subprocess.Popen[bytes]
) -> None:
    zombie_identity = proctree.proc_identity(zombie.pid)
    live_identity = proctree.proc_identity(sleeper.pid)
    assert zombie_identity is not None and live_identity is not None
    named = [(zombie.pid, zombie_identity[1] + 1), (sleeper.pid, live_identity[1])]
    assert proctree.reap_exited_children(named) == 0
    assert _state(zombie.pid) == "Z", "a zombie of another start time is not the one named"
    assert sleeper.poll() is None
