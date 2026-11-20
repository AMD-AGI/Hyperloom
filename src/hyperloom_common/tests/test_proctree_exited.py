# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A process that has exited is not running, whoever may signal it, and an exited child is waited only when named."""

from __future__ import annotations

import contextlib
import errno
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hyperloom_common import proctree


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


#: Ends the main thread, the thread-group leader, while another thread blocks reading the pipe named in ``argv[1]``.
_LEADER_EXITS_FIRST = (
    "import ctypes, os, sys, threading\n"
    "threading.Thread(target=os.read, args=(int(sys.argv[1]), 1)).start()\n"
    "ctypes.CDLL(None).pthread_exit(None)\n"
)


def _thread_states(pid: int) -> dict[int, str]:
    states: dict[int, str] = {}
    for task in Path(f"/proc/{pid}/task").iterdir():
        with contextlib.suppress(OSError):
            states[int(task.name)] = (task / "stat").read_text().rsplit(")", 1)[-1].split()[0]
    return states


def _wait_for(predicate: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, what
        time.sleep(0.02)


def test_a_process_whose_leader_exited_runs_until_its_last_thread_has() -> None:
    """``/proc/<pid>/stat`` shows a zombie for the leader alone; the process runs while another thread does."""
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen([sys.executable, "-c", _LEADER_EXITS_FIRST, str(read_fd)], pass_fds=(read_fd,))
    os.close(read_fd)
    try:

        def _leader_exited_thread_blocked() -> bool:
            states = _thread_states(child.pid)
            return states.pop(child.pid, "") == "Z" and list(states.values()) == ["S"]

        _wait_for(_leader_exited_thread_blocked, "the leader did not exit while its other thread blocked")
        assert _state(child.pid) == "Z"
        assert proctree.running(child.pid) is True, "a thread of it still runs"
        assert proctree.leader_exited(child.pid) is True, "its leader alone has exited"
        os.close(write_fd)
        write_fd = -1
        _wait_for(lambda: list(_thread_states(child.pid)) == [child.pid], "the other thread did not exit")
        assert proctree.running(child.pid) is False, "every thread of it has exited"
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        child.kill()
        child.wait()
    assert child.returncode == 0, "the process ended when its last thread did"


def test_leader_exited_reads_the_leader_and_leaves_a_gone_pid_to_running(
    zombie: subprocess.Popen[bytes], sleeper: subprocess.Popen[bytes]
) -> None:
    assert proctree.leader_exited(zombie.pid) is True
    assert proctree.leader_exited(sleeper.pid) is False
    sleeper.kill()
    sleeper.wait()
    assert proctree.leader_exited(sleeper.pid) is False


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


def test_tree_cpu_sec_sums_the_tree_and_skips_what_exited_or_was_replaced(monkeypatch):
    # utime, stime, cutime, cstime are the stat fields at 11-14 after the comm; the start time is at 19.
    def stat(ticks: int, start_time: int) -> list[str]:
        return ["S", *["0"] * 10, str(ticks), "0", "0", "0", *["0"] * 4, str(start_time)]

    fields = {1: ["S", *["0"] * 10, "1", "2", "3", "4", *["0"] * 4, "100"], 2: stat(10, 200), 4: stat(1000, 999)}

    def read(pid: int) -> list[str]:
        if pid not in fields:
            raise FileNotFoundError(pid)  # exited between the listing and its read
        return fields[pid]

    monkeypatch.setattr(proctree, "_stat_fields", read)
    monkeypatch.setattr(proctree, "descendants", lambda root: [(2, 200), (3, 300), (4, 400)])
    clock_ticks = os.sysconf("SC_CLK_TCK")
    # pid 4 is another process than the one listed (its start time differs): not the server's.
    assert proctree.tree_cpu_sec(1) == pytest.approx(20 / clock_ticks)
    assert proctree.tree_cpu_sec(5) is None, "a root that cannot be read is no reading"

    def unlisted(root: int) -> list[tuple[int, int]]:
        raise PermissionError(root)

    monkeypatch.setattr(proctree, "descendants", unlisted)
    assert proctree.tree_cpu_sec(1) is None, "a tree that cannot be listed is no reading"


def test_tree_cpu_sec_counts_a_child_collected_between_the_listing_and_its_read_through_the_root(monkeypatch):
    collected = False

    def read(pid: int) -> list[str]:
        nonlocal collected
        if pid == 2:
            collected = True  # it exits now, and the root waits for it before its own read
            raise FileNotFoundError(pid)
        cutime = 10 if collected else 0
        return ["S", *["0"] * 10, "5", "0", str(cutime), "0", *["0"] * 4, "100"]

    monkeypatch.setattr(proctree, "_stat_fields", read)
    monkeypatch.setattr(proctree, "descendants", lambda root: [(2, 200)])
    assert proctree.tree_cpu_sec(1) == pytest.approx(15 / os.sysconf("SC_CLK_TCK"))


def test_tree_cpu_sec_reads_this_process(sleeper):
    assert (proctree.tree_cpu_sec(os.getpid()) or 0.0) > 0.0
    assert proctree.tree_cpu_sec(sleeper.pid) is not None


def test_tree_rss_and_io_key_each_process_by_its_incarnation_and_skip_what_they_cannot_read(monkeypatch):
    identities = {1: (0, 100), 2: (1, 200), 3: (1, 300), 4: (1, 999)}
    rss = {1: 10, 2: 20, 4: 40}  # pid 3 is gone (or unreadable) by its read; pid 4 is another process than listed
    monkeypatch.setattr(proctree, "proc_identity", identities.get)
    monkeypatch.setattr(proctree, "descendants", lambda root: [(2, 200), (3, 300), (4, 400)])

    def status(pid: int) -> str:
        if pid not in rss:
            raise FileNotFoundError(pid)
        return f"Name:\tx\nVmRSS:\t   {rss[pid]} kB\n"

    monkeypatch.setattr(proctree, "_status_text", status)
    assert proctree.tree_rss_bytes(1) == {(1, 100): 10240, (2, 200): 20480}
    io = {1: "rchar: 7\nwchar: 1\nread_bytes: 3\n", 2: "rchar: 5\nread_bytes: 0\n"}

    def read_io(path: Path, encoding: str) -> str:
        pid = int(path.parent.name)
        if pid not in io:
            raise PermissionError(13, "Permission denied", str(path))  # EACCES: unknown, never zero
        return io[pid]

    monkeypatch.setattr(Path, "read_text", read_io)
    assert proctree.tree_io_bytes(1) == {(1, 100): 10, (2, 200): 5}
    del io[1]
    assert proctree.tree_io_bytes(1) is None, "a root whose counters cannot be read is no reading"
    assert proctree.tree_rss_bytes(5) is None, "a root that is gone is no reading"


def test_tree_rss_and_io_read_this_process():
    me = (os.getpid(), proctree.proc_identity(os.getpid())[1])
    assert (proctree.tree_rss_bytes(os.getpid()) or {}).get(me, 0) > 0
    io = proctree.tree_io_bytes(os.getpid())
    assert io is None or io.get(me, -1) >= 0
