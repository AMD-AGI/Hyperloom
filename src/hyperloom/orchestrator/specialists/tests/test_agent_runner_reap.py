# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local agent runner: what a CLI started does not outlive its session, however the session ends."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from hyperloom.common.proctree import REAP_MARK_ENV, proc_identity, running
from hyperloom.orchestrator.specialists import agent_runner
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
