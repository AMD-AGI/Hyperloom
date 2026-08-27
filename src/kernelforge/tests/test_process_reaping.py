"""The workspace reaper, run against real processes.

``tests/test_claude_timeout.py`` replaces the reaper with a recorder, so it pins
that the timeout path calls it and nothing about whether it works. This module
runs the real ``/proc`` scan and the real signalling against children it starts
itself, because what the reaper protects is not observable from a fake: a
benchmark child that outlives a session holds the GPU through the canonical
measurement that decides KEEP/REVERT for the whole iteration.

The question every test here circles is *whose process is this*. Killing too
little leaves the device busy; killing too much takes down a human's shell, a
sibling lane, or another campaign sharing the machine. Ownership is answered by
descent from this process -- which ``PR_SET_CHILD_SUBREAPER`` preserves across
the orphaning that detaching guarantees -- and by an inherited environment tag,
with the directory narrowing that set rather than defining it.

Both callers of the shared reaper are exercised -- the claude backend's timeout
path and the lane fan-out's teardown -- since they used to carry a copy each.

GPU-free and SDK-free, and every child is a ``sleep`` that dies in milliseconds.
Children announce themselves on stdout before they are asserted on, so nothing
here waits on a duration it guessed.
"""

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

# Announced after Popen has already chdir'd the child, so a child that reports
# itself ready is one whose /proc cwd is the directory under test.
_READY = "import sys; sys.stdout.write('ready\\n'); sys.stdout.flush(); "
_SLEEPS = _READY + "import time; time.sleep(120)"
_IGNORES_SIGTERM = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + _SLEEPS
# Announces and falls off the end, leaving a zombie until its parent reaps it.
_EXITS = _READY
# The same, with a status worth reading back: what an owner loses if something
# else collects its child first is the exit code, not the death.
_EXITS_WITH_7 = _READY + "raise SystemExit(7)"


# Runs in a process of its own: no event loop anywhere, and the flag installed
# from a worker thread, which is the one place ``signal.signal`` refuses. Prints
# a single word so the assertion is on what happened, not on a duration.
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
    """A script that starts a sleeper in ``directory`` and announces its pid.

    The announced pid is what lets a test reach a process it never held a
    handle to -- which is the whole point of the cases below, where the process
    that matters is not the one this fixture started.
    """
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
    """Make this process look like it started nothing.

    A subreaper cannot be escaped from below -- that is its purpose -- so a
    process this test suite starts can never really become someone else's. The
    two ownership signals are switched off instead, which is exactly the state
    the reaper sees when it meets a human's shell or another campaign's
    leftovers working in the same directory.
    """
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
        # A script that starts a process of its own announces that pid too, so
        # the teardown can reach a grandchild nothing here holds a handle to.
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
    """Block until ``pid`` has stopped running.

    A zombie counts: an orphan of ours is collected by the reaper thread and a
    child of ours by whoever started it, and neither has happened yet at the
    moment the process stops running. What the caller is asserting is that it
    stopped, which a zombie has.
    """
    deadline = time.monotonic() + 10
    while True:
        entry = _read_proc(pid)
        if entry is None or entry.state == "Z":
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"pid {pid} is still running")
        await asyncio.sleep(0.02)


async def _wait_collected(pid: int) -> None:
    """Block until ``pid`` has left the process table altogether.

    Stricter than :func:`_wait_gone` on the one point that matters here: a
    zombie is still an entry, still occupies its process group, and still
    answers ``killpg(pgid, 0)``.
    """
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

    # The canonical measurement starts the moment this returns, so "reaped" has
    # to mean nothing is left working in the workspace by then, not merely
    # signalled. Death itself is confirmed by waiting rather than by polling
    # once: a task drops its cwd while exiting, so it can leave the scan a
    # scheduling slice before its parent can reap it.
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
    """A session's deadline is not a machine-wide kill switch: the sibling lanes
    benching from their own copies have to survive it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    bystander = spawn(elsewhere)

    assert bystander.pid not in processes_under(str(workspace))

    await _reap_workspace_processes(str(workspace))

    assert bystander.poll() is None


async def test_the_callers_own_process_group_is_never_signalled(tmp_path, spawn, monkeypatch):
    """The loop that awaits the reaper can itself be running in the workspace.

    A child that did not detach shares this process's group, and the campaign
    runs plenty of those -- git, the build, the canonical driver -- so the
    reaper stays out of its own group entirely rather than reasoning about
    which member of it is a leftover.
    """
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
    # SIGTERM gets a 2s grace window and the SIGKILL that follows needs only
    # scheduling: the caller is blocked for that, and must not be held past it.
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
    """The case that cwd-based ownership got right for the wrong reason.

    Agent commands are started detached on purpose, so the shell above a
    benchmark exits first and routinely leaves it orphaned. Without
    ``PR_SET_CHILD_SUBREAPER`` that orphan reparents to init and nothing links
    it back to the session that caused it; with it, the campaign is still its
    parent and it is reaped as what it is.
    """
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
    """cwd is the scope, not the ownership -- and a process can leave the scope.

    A benchmark that chdir'd into ``/tmp`` still holds the device it opened
    from the workspace. It is reached through the process it belongs to rather
    than through where it happens to be standing.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    parent = spawn(workspace, _starts_a_child_in(elsewhere))
    moved = parent.announced[0]
    # Its cwd says it has nothing to do with the workspace; its parent says
    # otherwise, and the parent is the one that started it there.
    assert os.path.realpath(f"/proc/{moved}/cwd") == str(elsewhere.resolve())
    assert moved in process_reaping.owned_processes_under(str(workspace))

    report = await _reap_workspace_processes(str(workspace))

    assert moved in report.reaped
    await _wait_gone(moved)


async def test_the_shell_above_a_process_in_the_workspace_is_reaped_too(tmp_path, spawn):
    """The detached shell is what will start the next command.

    Killing only the benchmark leaves its shell free to launch another one into
    the middle of the canonical measurement, so the rest of the session's own
    process group goes with it.
    """
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
    """The reviewed bug, at the level it was actually wrong.

    A human's shell, a parallel campaign, or a leftover from a run that crashed
    weeks ago can be working in the same directory. The old scan matched on cwd
    alone and killed the whole process group each one belonged to. Now they are
    reported and left alone: not ours to kill is not a judgement call.
    """
    _disown(monkeypatch)
    bystander = spawn(tmp_path)

    report = await _reap_workspace_processes(str(tmp_path))

    assert bystander.poll() is None
    assert report.foreign == (bystander.pid,)
    assert report.reaped == ()
    # Present but idle is not a reason to refuse a measurement; it holds no
    # device, so the loop is told about it and carries on.
    assert report.contended is False


async def test_a_process_holding_a_device_makes_the_directory_contended(tmp_path, spawn, monkeypatch):
    """What separates "something is here" from "do not measure".

    The reaper cannot kill what is not this campaign's, and the loop cannot
    benchmark against a device someone else has open. Reporting it is the only
    move left, and it has to be a report the caller can act on.
    """
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
    """SIGKILL cannot be declined, but it can be un-completable.

    An uninterruptible-sleep process stuck in a driver ioctl stays on the
    device until the kernel lets it go. The old code logged a warning and
    returned, and the canonical benchmark then ran against a busy GPU and
    produced a number the loop acted on. Refusing to measure is the only honest
    answer.
    """
    # Signalling is what is suppressed, not the process: the state under test
    # is "asked to die, still here", which no portable child can be made to
    # reach on demand.
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
    """A shell being torn down starts its last command on the way out.

    The escalation rescans instead of working from the list it opened with, so
    a process that was not there when SIGTERM went out still gets one before
    the window closes -- a driver that is killed mid-ioctl leaves the device in
    the state the next measurement inherits.
    """
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
    """Being a subreaper means collecting orphans, and orphans become zombies.

    Nothing in the campaign waits on them, so they accumulate. A zombie holds
    no device and cannot be signalled, and counting one as contention would
    stall the loop over a process that has already exited.
    """
    child = spawn(tmp_path, _EXITS)
    await _wait_zombie(child.pid)

    assert processes_under(str(tmp_path)) == set()

    report = await _reap_workspace_processes(str(tmp_path))

    assert report.contended is False
    assert report.foreign == ()
    assert report.reaped == ()


def test_installing_the_reaper_tags_the_children_it_will_have(monkeypatch):
    """The tag is the half of ownership that works without the kernel's help.

    On a kernel that refuses ``PR_SET_CHILD_SUBREAPER`` an orphan is lost to
    init and no parent chain leads back here, so what a process was started
    with is the only thing left that still identifies it.
    """
    monkeypatch.setenv(process_reaping._OWNER_ENV, "stale-value")
    monkeypatch.setattr(process_reaping, "_owner_pid", None)

    install_child_subreaper()
    tag = os.environ[process_reaping._OWNER_ENV]

    assert tag.startswith(f"{os.getpid()}:")
    # The start time is in the tag because pids are recycled: a later process
    # reusing this pid must not inherit this campaign's children.
    assert tag != f"{os.getpid()}:0"
    assert process_reaping._current_owner_tag() == tag
    # Re-arming per call would be wasted syscalls on every session.
    assert install_child_subreaper() is install_child_subreaper()


async def test_an_inherited_orphan_is_collected_rather_than_left_a_zombie(tmp_path, spawn):
    """The other half of asking for ``PR_SET_CHILD_SUBREAPER``.

    The flag makes this process the parent of every orphaned descendant, and a
    parent that never waits turns each one into a zombie that lasts as long as
    the campaign. A zombie is not free: it holds its process group open, so
    anyone asking ``killpg(pgid, 0)`` whether a group is gone is told it is not,
    and it is deliberately invisible to the scan above -- so it is neither
    cleaned up nor reported. An 11-hour run starts enough detached benchmarks
    for that to matter.
    """
    if not install_child_subreaper():
        pytest.skip("this kernel does not support PR_SET_CHILD_SUBREAPER")
    parent = spawn(tmp_path, _starts_a_child_in(tmp_path, then_exits=True))
    orphan = parent.announced[0]
    assert parent.wait(timeout=10) == 0
    await _wait_gone(parent.pid)
    # Nothing forked it here; the kernel handed it over when its own parent
    # exited, which is the only reason it is this process's problem.
    reparented = _read_proc(orphan)
    assert reparented is not None
    assert reparented.ppid == os.getpid()

    os.kill(orphan, signal.SIGKILL)

    await _wait_collected(orphan)


async def test_collecting_orphans_leaves_this_process_s_own_children_alone(tmp_path, spawn):
    """The way a reaper like this goes wrong, pinned.

    ``waitpid(-1)`` would clear the zombies, and would also take the exit status
    of whichever child an ``asyncio`` transport, a ``Popen.wait()`` or the agent
    SDK asked for first -- ``Popen`` reports 0 for a child somebody else
    collected, so a failed build would come back as a passing one. The reap pass
    only ever waits on a pid that is not in the spawn record, and every child
    forked here is in it before its constructor returns.
    """
    install_child_subreaper()
    child = spawn(tmp_path, _EXITS_WITH_7)
    await _wait_zombie(child.pid)

    assert process_reaping._reap_inherited_orphans() == ()

    assert child.wait(timeout=10) == 7


async def test_a_child_started_through_asyncio_is_recorded_before_it_can_die():
    """The transport waits on its own child, so the record has to cover it.

    ``asyncio.create_subprocess_exec`` builds a ``subprocess.Popen`` under its
    transport, and so does the agent SDK through ``anyio.open_process``, which
    is why that constructor is one of the seams the record is hooked at. This
    asserts the seam still holds rather than trusting that it does.
    """
    install_child_subreaper()
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "raise SystemExit(7)")

    assert proc.pid in process_reaping._spawned_children

    assert await proc.wait() == 7


def _exits_with_7() -> None:
    """A ``multiprocessing`` body whose only job is to have a status to lose."""
    raise SystemExit(7)


def test_a_child_forked_outside_subprocess_is_recorded_too():
    """``multiprocessing`` calls ``os.fork()`` and waits on the pid itself.

    Nothing about that goes through ``subprocess``, and this suite forks that
    way, so a record built only from ``Popen`` would leave those children
    looking inherited -- and ``multiprocessing`` reports no exit code at all for
    a child something else collected, which is a hang rather than a wrong
    number. The at-fork handlers are what close that.
    """
    install_child_subreaper()
    child = multiprocessing.get_context("fork").Process(target=_exits_with_7)
    child.start()
    try:
        assert child.pid in process_reaping._spawned_children
    finally:
        child.join(10)

    assert child.exitcode == 7


def test_a_child_spawned_outside_subprocess_is_recorded_too():
    """``multiprocessing``'s spawn context never constructs a ``Popen``.

    ``util.spawnv_passfds`` calls ``_posixsubprocess.fork_exec`` itself, so a
    record hooked at ``subprocess.Popen`` -- an API rather than a primitive --
    misses every spawned worker, its forkserver and its resource tracker, and
    they all look inherited. That is not theoretical: it is what took
    ``test_tracker``'s concurrent workers away from ``Process.join()``, which
    then reported ``exitcode`` ``None`` rather than a wrong number.
    """
    install_child_subreaper()
    child = multiprocessing.get_context("spawn").Process(target=_exits_with_7)
    child.start()
    try:
        assert child.pid in process_reaping._spawned_children
    finally:
        child.join(30)

    assert child.exitcode == 7


async def test_a_child_that_predates_the_flag_is_not_taken_for_an_orphan(tmp_path, spawn):
    """Nothing was reparented here before the flag was armed.

    So every child that already existed at that moment was forked here and is
    being waited on here, and claiming them all is both safe and necessary: the
    campaign runs git and the build before its first agent session, and a test
    worker installs the flag with earlier tests' children still live.
    """
    install_child_subreaper()
    child = spawn(tmp_path, _EXITS_WITH_7)
    # Undo what the constructor recorded, so this is the state the reaper would
    # be in had the child been started before the flag was armed. Under the lock
    # because the reaper thread is running and would otherwise see the gap.
    with process_reaping._reaper_lock:
        process_reaping._spawned_children.pop(child.pid, None)
        process_reaping._adopt_existing_children()
    await _wait_zombie(child.pid)

    assert process_reaping._reap_inherited_orphans() == ()

    assert child.wait(timeout=10) == 7


def test_orphans_are_collected_without_an_event_loop_or_the_main_thread():
    """Neither is available to install from, and neither may be assumed.

    ``signal.signal`` is main-thread only, so a caller that installs from a
    worker gets the fallback wake-up instead of the SIGCHLD one. Both callers of
    ``install_child_subreaper`` are async today, but nothing about the flag is,
    and a campaign between sessions is running no loop at all. Run out of
    process because the flag and its thread are per-process and permanent: this
    worker has already installed the main-thread path.
    """
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
