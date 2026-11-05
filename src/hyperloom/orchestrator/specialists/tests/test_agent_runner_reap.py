# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local agent runner: what a CLI started, while it carries its launch's mark, does not outlive its session, however the
session ends."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from hyperloom.common.proctree import REAP_MARK_ENV, proc_identity, running
from hyperloom.orchestrator.specialists import agent_runner
from hyperloom.orchestrator.specialists.agent_runner import Liveness
from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.tests._agent_fixtures import make_runner, make_spec, wait_finished

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="procfs and sessions are POSIX-only")

#: How the stand-in CLI starts ``sleep 120``: in the CLI's own process group, in a session of its own, or in a session
#: of its own under a shell that has already exited, so no walk of the CLI's tree or group reaches it.
_START = {
    "group": "subprocess.Popen(['sleep', '120']).pid",
    "session": "subprocess.Popen(['sleep', '120'], start_new_session=True).pid",
    "detached": (
        "int(subprocess.run(['sh', '-c', 'setsid nohup sleep 120 </dev/null >/dev/null 2>&1 & echo $!'],"
        " capture_output=True, text=True, check=True).stdout)"
    ),
}
_STAY = "import time; time.sleep(60)\n"


def _cli(workspace: Path, how: str, then: str = "") -> str:
    """Stand-in CLI code: start ``sleep 120`` the way ``how`` names, write its pid to ``sleep.pid``, then run ``then``."""
    pid_file = workspace / "sleep.pid"
    return f"import pathlib, subprocess\npathlib.Path({str(pid_file)!r}).write_text(str({_START[how]}))\n{then}"


class _Started:
    """The pids of the sleeps the stand-in CLIs started, each pinned to its start time so teardown kills only them."""

    def __init__(self) -> None:
        self.pinned: list[tuple[int, int]] = []

    async def read(self, workspace: Path, timeout: float = 20.0) -> int:
        pid_file = workspace / "sleep.pid"
        end = time.monotonic() + timeout
        while not (pid_file.exists() and pid_file.read_text().strip()):
            assert time.monotonic() < end, "the stand-in CLI did not start its sleep"
            await asyncio.sleep(0.02)
        pid = int(pid_file.read_text())
        identity = proc_identity(pid)
        if identity is not None:
            self.pinned.append((pid, identity[1]))
        return pid

    def kill_survivors(self) -> None:
        for pid, start in self.pinned:
            identity = proc_identity(pid)
            if identity is not None and identity[1] == start:
                os.kill(pid, signal.SIGKILL)


@pytest.fixture
def started() -> Iterator[_Started]:
    sleeps = _Started()
    yield sleeps
    sleeps.kill_survivors()


class _SameUid:
    """A demoted launch whose account is the test's own uid, so no privilege is needed to run or scan as it."""

    user = "specialist"
    uid = os.getuid()

    def command(self, cmd: list[str]) -> list[str]:
        return list(cmd)

    def popen_kwargs(self) -> dict[str, Any]:
        return {}


async def _end(runner, handles: list) -> None:
    """Cancel and collect every session a test launched, whatever it left running."""
    for handle in handles:
        if not handle.session.collected:
            await runner.cancel(handle)
            await wait_finished(runner, handle)
            await runner.collect(handle)


def _kill_left_unconfirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tree kill kills, then cannot confirm it, as when a re-parented descendant is out of its reach."""
    kill = ss.SpecialistSubprocessDispatcher._kill

    def kill_then_doubt(proc: Any) -> bool:
        kill(proc)
        raise ss.ExecutionCleanupUnconfirmed(f"specialist pid={proc.pid}: tree cleanup unconfirmed")

    monkeypatch.setattr(ss.SpecialistSubprocessDispatcher, "_kill", staticmethod(kill_then_doubt))


def _fail_the_reaper_once_the_sleep_is_started(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    """The reap loop's look (not ``poll``'s) raises once the stand-in CLI has started its sleep."""
    look = ss.cli_alive

    def look_that_fails(*args: Any, **kwargs: Any) -> bool:
        pid_file = workspace / "sleep.pid"
        if pid_file.exists() and pid_file.read_text().strip():
            raise RuntimeError("the reaper's look failed")
        return look(*args, **kwargs)

    monkeypatch.setattr(ss, "cli_alive", look_that_fails)


@pytest.mark.parametrize("how", ["group", "session", "detached"])
async def test_what_a_cli_started_does_not_outlive_its_exit(tmp_path: Path, started: _Started, how: str) -> None:
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=_cli(ws, how)))
    try:
        sleep_pid = await started.read(ws)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await _end(runner, [handle])
    assert (status.state, status.returncode) == ("exited", 0)
    assert not running(sleep_pid)


async def test_a_demoted_cli_is_swept_as_its_account(
    tmp_path: Path, started: _Started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The app's uid cannot read a demoted child's environment, so the mark is looked for as the account."""
    monkeypatch.setattr(agent_runner, "_launch_user", _SameUid)
    monkeypatch.setattr(agent_runner, "prepare_demote", lambda *args: {})
    monkeypatch.setattr(ss, "marked_pids", lambda key, value: [])
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=_cli(ws, "detached"), mcp=False))
    try:
        sleep_pid = await started.read(ws)
        await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await _end(runner, [handle])
    assert not running(sleep_pid)


async def test_what_a_cli_detached_does_not_outlive_its_done_file(
    tmp_path: Path, started: _Started, monkeypatch: pytest.MonkeyPatch
) -> None:
    graces: list[int] = []
    grace = ss.SpecialistSubprocessDispatcher._reap_after_done_file

    async def counted(self, proc, now, scope) -> None:
        graces.append(proc.pid)
        await grace(self, proc, now, scope)

    monkeypatch.setattr(ss.SpecialistSubprocessDispatcher, "_reap_after_done_file", counted)
    ws = tmp_path / "w"
    then = "pathlib.Path('specialist_done.json').write_text('{}')\nimport time; time.sleep(0.5)\n"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=_cli(ws, "detached", then)))
    try:
        sleep_pid = await started.read(ws)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await _end(runner, [handle])
    assert graces == [handle.pid]  # the reap loop ended the session on its done file
    assert (status.state, status.returncode) == ("exited", 0)
    assert not running(sleep_pid)


async def test_what_a_cli_detached_does_not_outlive_its_deadline(tmp_path: Path, started: _Started) -> None:
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=_cli(ws, "detached", _STAY), deadline_unix=time.time() + 3))
    try:
        sleep_pid = await started.read(ws)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await _end(runner, [handle])
    assert status.state == "exited" and status.timed_out
    assert not running(sleep_pid)


async def test_cancel_ends_what_its_cli_detached_and_spares_another_session(tmp_path: Path, started: _Started) -> None:
    """Both launches get the same caller env, as the sessions of one campaign do, mark included; each sweeps its own."""
    runner = make_runner(tmp_path / "session")
    handles = []
    sleeps: list[int] = []
    try:
        for name in ("a", "b"):
            ws = tmp_path / name
            spec = make_spec(ws, task_id=name, code=_cli(ws, "detached", _STAY), env={REAP_MARK_ENV: "shared"})
            handles.append(await runner.launch(spec))
            sleeps.append(await started.read(ws))
        await runner.cancel(handles[0])
        await wait_finished(runner, handles[0])
        await runner.collect(handles[0])
        assert not running(sleeps[0])
        assert running(sleeps[1])
    finally:
        await _end(runner, handles)
    assert not running(sleeps[1])


async def test_what_a_cli_detached_does_not_outlive_its_loss(tmp_path: Path, started: _Started) -> None:
    """The reap loop ends a CLI that writes nothing and waits on no tool as lost; what it detached goes with it."""
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session", liveness=Liveness(lost_after_seconds=2.0))
    handle = await runner.launch(make_spec(ws, code=_cli(ws, "detached", _STAY)))
    try:
        sleep_pid = await started.read(ws)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await _end(runner, [handle])
    assert status.state == "lost" and handle.session.outcome["stale_heartbeat"]
    assert not running(sleep_pid)


async def test_what_a_cli_detached_does_not_outlive_a_failed_reap_loop(
    tmp_path: Path, started: _Started, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "w"
    _fail_the_reaper_once_the_sleep_is_started(monkeypatch, ws)
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=_cli(ws, "detached", _STAY)))
    try:
        sleep_pid = await started.read(ws)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await _end(runner, [handle])
    assert status.state == "exited" and "reap loop failed" in status.error
    assert not running(sleep_pid)


async def test_what_a_cli_detached_does_not_outlive_its_cancelled_supervisor(tmp_path: Path, started: _Started) -> None:
    """The supervisor is cancelled while it runs, as a shutting-down event loop does: it still sweeps on its way out."""
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=_cli(ws, "detached", _STAY)))
    proc = handle.session.proc
    try:
        sleep_pid = await started.read(ws)
        handle.session.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handle.session.task
        status = await runner.poll(handle)
    finally:
        if not handle.session.task.done():
            await _end(runner, [handle])
        if proc.poll() is None:  # only when the supervisor left its CLI running
            proc.kill()
            proc.wait()
    assert status.state == "exited" and status.returncode is not None
    assert not running(sleep_pid)


async def test_a_tree_kill_left_unconfirmed_at_teardown_is_logged_with_its_traceback(
    tmp_path: Path, started: _Started, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The reap loop fails with the CLI's root still up, so teardown kills its tree, and that kill cannot confirm."""
    ws = tmp_path / "w"
    _fail_the_reaper_once_the_sleep_is_started(monkeypatch, ws)
    _kill_left_unconfirmed(monkeypatch)
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=_cli(ws, "detached", _STAY)))
    with caplog.at_level(logging.ERROR, logger=agent_runner.__name__):
        try:
            sleep_pid = await started.read(ws)
            status = await wait_finished(runner, handle)
            await runner.collect(handle)
        finally:
            await _end(runner, [handle])
    unconfirmed = [r for r in caplog.records if "tree cleanup unconfirmed" in r.getMessage()]
    assert "tree cleanup unconfirmed" in status.error
    assert unconfirmed and all(r.exc_info and r.exc_info[0] is ss.ExecutionCleanupUnconfirmed for r in unconfirmed)
    assert not running(sleep_pid)


@pytest.mark.parametrize(
    ("ending", "state", "ended_by"), [("deadline", "exited", "timeout"), ("silence", "lost", "lost")]
)
async def test_a_reap_loop_kill_left_unconfirmed_still_reports_why_it_killed(
    tmp_path: Path, started: _Started, monkeypatch: pytest.MonkeyPatch, ending: str, state: str, ended_by: str
) -> None:
    _kill_left_unconfirmed(monkeypatch)
    ws = tmp_path / "w"
    code = _cli(ws, "detached", _STAY)
    if ending == "deadline":
        runner = make_runner(tmp_path / "session")
        spec = make_spec(ws, code=code, deadline_unix=time.time() + 3)
    else:
        runner = make_runner(tmp_path / "session", liveness=Liveness(lost_after_seconds=2.0))
        spec = make_spec(ws, code=code)
    handle = await runner.launch(spec)
    try:
        sleep_pid = await started.read(ws)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await _end(runner, [handle])
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert (status.state, status.timed_out) == (state, ending == "deadline")
    assert "reap loop failed" in status.error and "tree cleanup unconfirmed" in status.error
    assert record["ended_by"] == ended_by
    assert not running(sleep_pid)


async def test_a_failed_heartbeat_sink_fails_the_launch_and_leaves_nothing_running(
    tmp_path: Path, started: _Started
) -> None:
    """The sink fails once the CLI has detached its sleep: the launch raises the sink's error, and the CLI, what it
    detached, its beat and its claim on the config dir end with it, as when the spawn itself fails."""
    ws = tmp_path / "w"
    pid_file = ws / "sleep.pid"
    clis: list[subprocess.Popen] = []
    beats: list = []

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen:
        clis.append(subprocess.Popen(*args, **kwargs))
        return clis[-1]

    def sink_down(beat: Any) -> None:
        beats.append(beat)
        end = time.monotonic() + 20.0
        while not (pid_file.exists() and pid_file.read_text().strip()):
            assert time.monotonic() < end, "the stand-in CLI did not start its sleep"
            time.sleep(0.02)
        raise RuntimeError("sink down")

    runner = make_runner(tmp_path / "session", popen=popen)
    try:
        with pytest.raises(RuntimeError, match="sink down"):
            await runner.launch(make_spec(ws, code=_cli(ws, "detached", _STAY), heartbeat_sink=sink_down))
        sleep_pid = await started.read(ws)
        assert clis[0].poll() is not None and not running(sleep_pid)
        beats[0]()
        assert not (ws / "heartbeat.json").exists()
        handle = await runner.launch(make_spec(ws, code=_STAY))  # the config dir is free, and nothing else is live
        await _end(runner, [handle])
        assert handle.session.concurrent_at_start == 1
    finally:
        for cli in clis:
            if cli.poll() is None:
                cli.kill()
                cli.wait()
