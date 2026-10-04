"""The workspace reaper, run against real processes."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from kernelforge.llm import process_reaping
from kernelforge.agent_backends.claude import _reap_workspace_processes
from kernelforge.llm.process_reaping import (
    _read_proc,
    _Survey,
    install_child_subreaper,
    processes_under,
)
from kernelforge.loop.fanout import _reap_lane_processes

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc"),
    reason="the reaper reads process working directories from /proc",
)

# Announced after Popen has already chdir'd the child, so a child that reports itself ready is one whose /proc cwd is
# the directory under test.
_READY = "import sys; sys.stdout.write('ready\\n'); sys.stdout.flush(); "
_SLEEPS = _READY + "import time; time.sleep(120)"
_IGNORES_SIGTERM = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + _SLEEPS
# Announces and falls off the end, leaving a zombie until its parent reaps it.
_EXITS = _READY
# The same, with a status worth reading back: what an owner loses if something else collects its child first is the
# exit code, not the death.
_EXITS_WITH_7 = _READY + "raise SystemExit(7)"


# Runs in a process of its own: no event loop anywhere, and the flag installed from a worker thread, which is the one
# place ``signal.signal`` refuses.
_COLLECTS_AN_ORPHAN_OFF_THE_MAIN_THREAD = """
import os, signal, subprocess, sys, threading, time
from kernelforge.llm.process_reaping import _read_proc, install_child_subreaper

armed = []
worker = threading.Thread(target=lambda: armed.append(install_child_subreaper()))
worker.start()
worker.join()
if not armed[0]:
    print("unsupported")
    raise SystemExit(0)

# A parent that detaches a sleeper and exits, so the sleeper is reparented here
# exactly the way an agent's benchmark is when its shell goes.
parent = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import subprocess, sys\\n"
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(120)'],"
        " start_new_session=True)\\n"
        "sys.stdout.write('%d\\\\n' % child.pid)\\n"
        "sys.stdout.flush()\\n",
    ],
    stdout=subprocess.PIPE,
    text=True,
)
orphan = int(parent.stdout.readline())
parent.wait()
os.kill(orphan, signal.SIGKILL)

deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    entry = _read_proc(orphan)
    if entry is None:
        print("collected")
        break
    time.sleep(0.02)
else:
    entry = _read_proc(orphan)
    print("left in state", entry.state if entry is not None else "?")
"""


def _starts_a_child_in(directory: Path, *, then_exits: bool = False) -> str:
    """A script that starts a sleeper in ``directory`` and announces its pid."""
    tail = "" if then_exits else "import time; time.sleep(120)\n"
    return (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        + repr(_SLEEPS)
        + "], cwd="
        + repr(str(directory))
        + ", start_new_session="
        + repr(then_exits)
        + ")\n"
        "sys.stdout.write('ready %d\\n' % child.pid)\n"
        "sys.stdout.flush()\n" + tail
    )


def _holds_open(path: Path) -> str:
    """A script that keeps a file descriptor open on ``path``."""
    return (
        "import sys\n"
        "handle = open(" + repr(str(path)) + ")\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "import time; time.sleep(120)\n"
    )


def _disown(monkeypatch) -> None:
    """Make this process look like it started nothing."""
    monkeypatch.setattr(process_reaping, "_children_by_parent", lambda _: {})
    monkeypatch.setattr(process_reaping, "_owner_pid", None)


@pytest.fixture
def spawn():
    """Start children that are killed on the way out, assertion or not."""
    children: list[subprocess.Popen] = []
    strays: list[int] = []

    def _spawn(cwd: Path, script: str = _SLEEPS, *, own_group: bool = True):
        child = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=str(cwd),
            start_new_session=own_group,
            stdout=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        line = child.stdout.readline()
        if not line.startswith("ready"):
            raise AssertionError(f"child in {cwd} exited before it was ready")
        # A script that starts a process of its own announces that pid too, so the teardown can reach a grandchild
        # nothing here holds a handle to.
        child.announced = [int(part) for part in line.split()[1:]]
        strays.extend(child.announced)
        return child

    yield _spawn

    for pid in strays:
        with suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        child.stdout.close()


async def _wait_gone(pid: int) -> None:
    """Block until ``pid`` has stopped running."""
    deadline = time.monotonic() + 10
    while True:
        entry = _read_proc(pid)
        if entry is None or entry.state == "Z":
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"pid {pid} is still running")
        await asyncio.sleep(0.02)


async def _wait_collected(pid: int) -> None:
    """Block until ``pid`` has left the process table altogether."""
    deadline = time.monotonic() + 10
    while True:
        entry = _read_proc(pid)
        if entry is None:
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"pid {pid} is still listed, in state {entry.state}")
        await asyncio.sleep(0.02)


async def _wait_zombie(pid: int) -> None:
    """Block until ``pid`` is a zombie, so a reap pass has something to decide."""
    deadline = time.monotonic() + 10
    while True:
        entry = _read_proc(pid)
        if entry is not None and entry.state == "Z":
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"pid {pid} never became a zombie")
        await asyncio.sleep(0.01)


async def test_a_child_in_the_workspace_is_found_and_reaped(tmp_path, spawn):
    child = spawn(tmp_path)

    assert child.pid in processes_under(str(tmp_path))

    report = await _reap_workspace_processes(str(tmp_path))

    # The canonical measurement starts the moment this returns, so "reaped" has to mean nothing is left working in the
    # workspace by then, not merely signalled.
    assert processes_under(str(tmp_path)) == set()
    assert child.wait(timeout=10) != 0
    assert child.pid in report.reaped
    assert report.contended is False


async def test_a_child_in_a_subdirectory_is_reaped_too(tmp_path, spawn):
    """The agent builds and benches from subdirectories of its workspace."""
    nested = tmp_path / "workspace" / "build"
    nested.mkdir(parents=True)
    child = spawn(nested)

    assert child.pid in processes_under(str(tmp_path))

    await _reap_workspace_processes(str(tmp_path))

    assert processes_under(str(tmp_path)) == set()
    assert child.wait(timeout=10) != 0


async def test_a_child_outside_the_workspace_is_left_running(tmp_path, spawn):
    """A session's deadline is not a machine-wide kill switch: the sibling lanes benching from their own copies have to survive it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    bystander = spawn(elsewhere)

    assert bystander.pid not in processes_under(str(workspace))

    await _reap_workspace_processes(str(workspace))

    assert bystander.poll() is None


async def test_the_callers_own_process_group_is_never_signalled(tmp_path, spawn, monkeypatch):
    """The loop that awaits the reaper can itself be running in the workspace."""
    monkeypatch.chdir(tmp_path)
    attached = spawn(tmp_path, own_group=False)
    assert os.getpgid(attached.pid) == os.getpgrp()

    assert processes_under(str(tmp_path)) == set()

    report = await _reap_workspace_processes(str(tmp_path))

    assert attached.poll() is None
    assert report.contended is False


async def test_a_child_that_ignores_sigterm_is_killed_within_the_grace_window(tmp_path, spawn):
    """A hung driver never handles SIGTERM; SIGKILL is what frees the device."""
    child = spawn(tmp_path, _IGNORES_SIGTERM)

    started = time.monotonic()
    report = await _reap_workspace_processes(str(tmp_path))
    elapsed = time.monotonic() - started

    assert processes_under(str(tmp_path)) == set()
    assert child.wait(timeout=10) != 0
    assert report.contended is False
    # SIGTERM gets a 2s grace window and the SIGKILL that follows needs only scheduling: the caller is blocked for
    # that, and must not be held past it.
    assert elapsed < 5.0


async def test_the_lane_teardown_reaps_the_same_way(tmp_path, spawn):
    """Both callers share one implementation, so neither can regress alone."""
    lane_dir = tmp_path / "lane-1"
    lane_dir.mkdir()
    stubborn = spawn(lane_dir, _IGNORES_SIGTERM)
    bystander = spawn(tmp_path)

    report = await _reap_lane_processes(lane_dir)

    assert processes_under(lane_dir) == set()
    assert stubborn.wait(timeout=10) != 0
    assert bystander.poll() is None
    assert report.contended is False


async def test_an_orphaned_grandchild_is_still_this_campaign_s_to_reap(tmp_path, spawn):
    """The case that cwd-based ownership got right for the wrong reason."""
    if not install_child_subreaper():
        pytest.skip("this kernel does not support PR_SET_CHILD_SUBREAPER")
    parent = spawn(tmp_path, _starts_a_child_in(tmp_path, then_exits=True))
    orphan = parent.announced[0]
    assert parent.wait(timeout=10) == 0
    await _wait_gone(parent.pid)

    reparented = _read_proc(orphan)
    assert reparented is not None
    assert reparented.ppid == os.getpid()

    report = await _reap_workspace_processes(str(tmp_path))

    assert orphan in report.reaped
    assert processes_under(str(tmp_path)) == set()


async def test_a_process_of_ours_that_moved_out_of_the_workspace_is_reaped(tmp_path, spawn):
    """cwd is the scope, not the ownership -- and a process can leave the scope."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    parent = spawn(workspace, _starts_a_child_in(elsewhere))
    moved = parent.announced[0]
    # Its cwd says it has nothing to do with the workspace; its parent says otherwise, and the parent is the one that
    # started it there.
    assert os.path.realpath(f"/proc/{moved}/cwd") == str(elsewhere.resolve())
    assert moved in process_reaping.owned_processes_under(str(workspace))

    report = await _reap_workspace_processes(str(workspace))

    assert moved in report.reaped
    await _wait_gone(moved)


async def test_the_shell_above_a_process_in_the_workspace_is_reaped_too(tmp_path, spawn):
    """The detached shell is what will start the next command."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shell = spawn(elsewhere, _starts_a_child_in(workspace))
    working = shell.announced[0]
    assert os.getpgid(working) == os.getpgid(shell.pid)

    await _reap_workspace_processes(str(workspace))

    assert shell.wait(timeout=10) != 0
    await _wait_gone(working)


async def test_a_process_that_is_not_ours_is_reported_and_left_running(tmp_path, spawn, monkeypatch):
    """The reviewed bug, at the level it was actually wrong."""
    _disown(monkeypatch)
    bystander = spawn(tmp_path)

    report = await _reap_workspace_processes(str(tmp_path))

    assert bystander.poll() is None
    assert report.foreign == (bystander.pid,)
    assert report.reaped == ()
    # Present but idle is not a reason to refuse a measurement; it holds no device, so the loop is told about it and
    # carries on.
    assert report.contended is False


async def test_a_process_holding_a_device_makes_the_directory_contended(tmp_path, spawn, monkeypatch):
    """What separates \"something is here\" from \"do not measure\"."""
    device = tmp_path / "fake-device"
    device.write_text("")
    monkeypatch.setattr(process_reaping, "_DEVICE_PREFIXES", (str(device),))
    _disown(monkeypatch)
    holder = spawn(tmp_path, _holds_open(device))

    report = await _reap_workspace_processes(str(tmp_path))

    assert holder.poll() is None
    assert report.holding_device == (holder.pid,)
    assert report.contended is True
    assert str(holder.pid) in report.describe()


async def test_a_process_that_survives_sigkill_makes_the_directory_contended(tmp_path, spawn, monkeypatch):
    """SIGKILL cannot be declined, but it can be un-completable."""
    # Signalling is what is suppressed, not the process: the state under test is "asked to die, still here", which no
    # portable child can be made to reach on demand.
    monkeypatch.setattr(process_reaping, "_signal", lambda *_: None)
    monkeypatch.setattr(process_reaping, "_TERM_GRACE_SEC", 0.05)
    monkeypatch.setattr(process_reaping, "_KILL_CONFIRM_SEC", 0.05)
    survivor = spawn(tmp_path)

    report = await _reap_workspace_processes(str(tmp_path))

    assert survivor.poll() is None
    assert report.unkillable == (survivor.pid,)
    assert report.reaped == ()
    assert report.contended is True


async def test_a_process_that_appears_during_the_grace_window_is_asked_first(
    monkeypatch,
):
    """A shell being torn down starts its last command on the way out."""
    scans = [{11: 100}, {11: 100, 22: 200}, {22: 200}, {}]
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(process_reaping, "_survey", lambda _: _Survey(scans.pop(0), ()))
    monkeypatch.setattr(
        process_reaping,
        "_signal",
        lambda pid, _start, sig: sent.append((pid, sig)),
    )

    reaped, unkillable = await process_reaping._escalate("/nowhere")

    assert sent == [(11, signal.SIGTERM), (22, signal.SIGTERM)]
    assert reaped == (11, 22)
    assert unkillable == ()


async def test_a_zombie_is_not_mistaken_for_something_holding_the_device(tmp_path, spawn):
    """Being a subreaper means collecting orphans, and orphans become zombies."""
    child = spawn(tmp_path, _EXITS)
    await _wait_zombie(child.pid)

    assert processes_under(str(tmp_path)) == set()

    report = await _reap_workspace_processes(str(tmp_path))

    assert report.contended is False
    assert report.foreign == ()
    assert report.reaped == ()


def test_installing_the_reaper_tags_the_children_it_will_have(monkeypatch):
    """The tag is the half of ownership that works without the kernel's help."""
    monkeypatch.setenv(process_reaping._OWNER_ENV, "stale-value")
    monkeypatch.setattr(process_reaping, "_owner_pid", None)

    install_child_subreaper()
    tag = os.environ[process_reaping._OWNER_ENV]

    assert tag.startswith(f"{os.getpid()}:")
    # The start time is in the tag because pids are recycled: a later process reusing this pid must not inherit this
    # campaign's children.
    assert tag != f"{os.getpid()}:0"
    assert process_reaping._current_owner_tag() == tag
    # Re-arming per call would be wasted syscalls on every session.
    assert install_child_subreaper() is install_child_subreaper()


async def test_an_inherited_orphan_is_collected_rather_than_left_a_zombie(tmp_path, spawn):
    """The other half of asking for ``PR_SET_CHILD_SUBREAPER``."""
    if not install_child_subreaper():
        pytest.skip("this kernel does not support PR_SET_CHILD_SUBREAPER")
    parent = spawn(tmp_path, _starts_a_child_in(tmp_path, then_exits=True))
    orphan = parent.announced[0]
    assert parent.wait(timeout=10) == 0
    await _wait_gone(parent.pid)
    # Nothing forked it here; the kernel handed it over when its own parent exited, which is the only reason it is
    # this process's problem.
    reparented = _read_proc(orphan)
    assert reparented is not None
    assert reparented.ppid == os.getpid()

    os.kill(orphan, signal.SIGKILL)

    await _wait_collected(orphan)


async def test_collecting_orphans_leaves_this_process_s_own_children_alone(tmp_path, spawn):
    """The way a reaper like this goes wrong, pinned."""
    install_child_subreaper()
    child = spawn(tmp_path, _EXITS_WITH_7)
    await _wait_zombie(child.pid)

    assert process_reaping._reap_inherited_orphans() == ()

    assert child.wait(timeout=10) == 7


async def test_a_child_started_through_asyncio_is_recorded_before_it_can_die():
    """The transport waits on its own child, so the record has to cover it."""
    install_child_subreaper()
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "raise SystemExit(7)")

    assert proc.pid in process_reaping._spawned_children

    assert await proc.wait() == 7


def _exits_with_7() -> None:
    """A ``multiprocessing`` body whose only job is to have a status to lose."""
    raise SystemExit(7)


def test_a_child_forked_outside_subprocess_is_recorded_too():
    """``multiprocessing`` calls ``os.fork()`` and waits on the pid itself."""
    install_child_subreaper()
    child = multiprocessing.get_context("fork").Process(target=_exits_with_7)
    child.start()
    try:
        assert child.pid in process_reaping._spawned_children
    finally:
        child.join(10)

    assert child.exitcode == 7


def test_a_child_spawned_outside_subprocess_is_recorded_too():
    """``multiprocessing``'s spawn context never constructs a ``Popen``."""
    install_child_subreaper()
    child = multiprocessing.get_context("spawn").Process(target=_exits_with_7)
    child.start()
    try:
        assert child.pid in process_reaping._spawned_children
    finally:
        child.join(30)

    assert child.exitcode == 7


async def test_a_child_that_predates_the_flag_is_not_taken_for_an_orphan(tmp_path, spawn):
    """Nothing was reparented here before the flag was armed."""
    install_child_subreaper()
    child = spawn(tmp_path, _EXITS_WITH_7)
    # Undo what the constructor recorded, so this is the state the reaper would be in had the child been started
    # before the flag was armed.
    with process_reaping._reaper_lock:
        process_reaping._spawned_children.pop(child.pid, None)
        process_reaping._adopt_existing_children()
    await _wait_zombie(child.pid)

    assert process_reaping._reap_inherited_orphans() == ()

    assert child.wait(timeout=10) == 7


def test_orphans_are_collected_without_an_event_loop_or_the_main_thread():
    """Neither is available to install from, and neither may be assumed."""
    completed = subprocess.run(
        [sys.executable, "-c", _COLLECTS_AN_ORPHAN_OFF_THE_MAIN_THREAD],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.stdout.strip() == "unsupported":
        pytest.skip("this kernel does not support PR_SET_CHILD_SUBREAPER")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "collected", completed.stdout


def test_a_missing_proc_answers_empty_rather_than_raising(tmp_path, monkeypatch):
    """Reaping is best-effort on a host that cannot report process cwds."""
    real_isdir = os.path.isdir
    monkeypatch.setattr(os.path, "isdir", lambda path: False if path == "/proc" else real_isdir(path))

    assert processes_under(str(tmp_path)) == set()
    report = asyncio.run(_reap_workspace_processes(str(tmp_path)))
    assert report.contended is False
