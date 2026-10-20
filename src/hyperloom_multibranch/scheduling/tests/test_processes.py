# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's processes, started or adopted: recorded in ``MB-pids.jsonl`` and killed with everything they started."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hyperloom.common.proctree import descendants, running
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.processes import (
    UNREGISTERED_BRANCH,
    ProcessRegistry,
    RegisteredProcess,
    adopting_orphans,
)

SPAWNED_AT = 1_700_000_000.0
CHILD_WITH_GRANDCHILD = ("sh", "-c", "sleep 60 & sleep 60")
ORPHANING = ("sh", "-c", "sleep 60 & exit 0")
LONG_SLEEP = ("sleep", "60")
#: A root that starts a server leading a session of its own, the way the main flow launches inference servers.
SESSION_LEADING_SERVER = (
    sys.executable,
    "-c",
    "import subprocess, time; subprocess.Popen(['sleep', '60'], start_new_session=True); time.sleep(60)",
)

#: A root that, on SIGTERM, starts a server leading a session of its own and exits, the way the main flow's benchmark
#: script outlived it: the server is started after any tree taken before the signal, and orphaned at once.
LEAKS_ON_TERM = """
import pathlib, signal, subprocess, sys, time
out = pathlib.Path(sys.argv[1])
def leak(*_):
    server = subprocess.Popen(['sleep', '60'], start_new_session=True)
    out.with_suffix('.pid').write_text(str(server.pid))
    sys.exit(0)
signal.signal(signal.SIGTERM, leak)
out.write_text('ready')
time.sleep(60)
"""


@pytest.fixture
def registry(tmp_path: Path) -> ProcessRegistry:
    return ProcessRegistry(tmp_path / "MB-pids.jsonl", Clock(now=lambda: SPAWNED_AT))


async def _spawn(registry: ProcessRegistry, tmp_path: Path, argv: tuple[str, ...], branch: str) -> RegisteredProcess:
    return await registry.spawn(argv, branch=branch, log=tmp_path / f"{branch}.log")


async def _wait_for_descendants(pid: int, count: int) -> list[int]:
    deadline = time.monotonic() + 5.0
    while True:
        found = [child for child, _ in descendants(pid)]
        if len(found) >= count or time.monotonic() > deadline:
            return found
        await asyncio.sleep(0.05)


async def test_spawn_records_the_process_in_its_own_session(registry: ProcessRegistry, tmp_path: Path) -> None:
    proc = await registry.spawn(("sh", "-c", "echo out; echo err >&2"), branch="b01", log=tmp_path / "b01.log")
    await proc.process.wait()

    row = json.loads((tmp_path / "MB-pids.jsonl").read_text(encoding="utf-8").strip())
    assert row["pid"] == proc.record.pid == row["pgid"]
    assert row["branch"] == "b01"
    assert row["argv"] == ["sh", "-c", "echo out; echo err >&2"]
    assert row["started_at"] == SPAWNED_AT
    assert "owner_scope" in row
    assert (tmp_path / "b01.log").read_text(encoding="utf-8").split() == ["out", "err"]
    assert os.getpgid(0) != proc.record.pgid


async def test_kill_branch_kills_its_grandchildren_and_spares_other_branches(
    registry: ProcessRegistry, tmp_path: Path
) -> None:
    doomed = await _spawn(registry, tmp_path, CHILD_WITH_GRANDCHILD, "b01")
    spared = await _spawn(registry, tmp_path, LONG_SLEEP, "b02")
    grandchildren = await _wait_for_descendants(doomed.record.pid, 2)
    assert grandchildren

    assert await registry.kill_branch("b01")

    assert not any(running(pid) for pid in (doomed.record.pid, *grandchildren))
    assert [record.branch for record in await registry.leftovers()] == ["b02"]
    assert await registry.kill_all()
    assert not running(spared.record.pid)
    assert await registry.leftovers() == ()


async def test_a_member_orphaned_by_its_root_is_still_a_leftover_and_killed(
    registry: ProcessRegistry, tmp_path: Path
) -> None:
    proc = await _spawn(registry, tmp_path, ORPHANING, "prelude")
    await proc.process.wait()

    assert [record.pid for record in await registry.leftovers()] == [proc.record.pid]
    assert await registry.kill_all()
    assert await registry.leftovers() == ()


async def test_kill_with_nothing_registered_or_alive_succeeds(registry: ProcessRegistry, tmp_path: Path) -> None:
    assert await registry.kill_all()
    proc = await _spawn(registry, tmp_path, ("true",), "b01")
    await proc.process.wait()
    assert await registry.kill_branch("b01")
    assert await registry.leftovers() == ()


async def test_terminate_gives_the_root_its_grace(registry: ProcessRegistry, tmp_path: Path) -> None:
    proc = await _spawn(registry, tmp_path, LONG_SLEEP, "prelude")

    assert await registry.terminate(proc, grace_seconds=5.0) == -signal.SIGTERM
    assert await registry.leftovers() == ()


async def test_a_root_that_outlives_its_grace_is_killed_with_its_tree(
    registry: ProcessRegistry, tmp_path: Path
) -> None:
    proc = await _spawn(registry, tmp_path, ("sh", "-c", "trap '' TERM; sleep 60"), "prelude")
    children = await _wait_for_descendants(proc.record.pid, 1)

    assert await registry.terminate(proc, grace_seconds=0.2) is None
    assert not any(running(pid) for pid in (proc.record.pid, *children))
    assert await registry.leftovers() == ()


async def test_terminate_kills_a_descendant_that_leads_its_own_session(
    registry: ProcessRegistry, tmp_path: Path
) -> None:
    proc = await _spawn(registry, tmp_path, SESSION_LEADING_SERVER, "prelude")
    [server] = await _wait_for_descendants(proc.record.pid, 1)
    assert os.getpgid(server) != proc.record.pgid

    await registry.terminate(proc, grace_seconds=5.0)

    assert not running(server)


async def test_an_adopted_process_is_recorded_and_killed_with_its_branch(
    registry: ProcessRegistry, tmp_path: Path
) -> None:
    agent = subprocess.Popen(CHILD_WITH_GRANDCHILD, start_new_session=True)
    try:
        grandchildren = await _wait_for_descendants(agent.pid, 2)
        record = registry.adopt(agent.pid, branch="b01", argv=CHILD_WITH_GRANDCHILD)

        row = json.loads((tmp_path / "MB-pids.jsonl").read_text(encoding="utf-8").strip())
        assert row["pid"] == row["pgid"] == record.pid == agent.pid and row["branch"] == "b01"
        assert [leftover.pid for leftover in await registry.leftovers()] == [agent.pid]
        assert await registry.kill_branch("b01")
        assert not any(running(pid) for pid in (agent.pid, *grandchildren))
        assert await registry.leftovers() == ()
    finally:
        agent.kill()
        agent.wait()


async def test_a_process_that_leads_no_group_of_its_own_is_not_adopted(registry: ProcessRegistry) -> None:
    shared = subprocess.Popen(LONG_SLEEP)
    try:
        with pytest.raises(ValueError, match="does not lead its own process group"):
            registry.adopt(shared.pid, branch="b01", argv=LONG_SLEEP)
    finally:
        shared.kill()
        shared.wait()
    with pytest.raises(ValueError, match="is not running"):
        registry.adopt(shared.pid, branch="b01", argv=LONG_SLEEP)
    assert await registry.leftovers() == ()


async def _read_when_written(path: Path) -> str:
    deadline = time.monotonic() + 5.0
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} was never written"
        await asyncio.sleep(0.05)
    return path.read_text(encoding="utf-8")


async def test_a_server_the_root_orphans_on_its_way_out_is_a_leftover_until_the_session_kill(
    registry: ProcessRegistry, tmp_path: Path
) -> None:
    ready = tmp_path / "root"
    with adopting_orphans():
        proc = await _spawn(registry, tmp_path, (sys.executable, "-c", LEAKS_ON_TERM, str(ready)), "prelude")
        await _read_when_written(ready)

        assert await registry.terminate(proc, grace_seconds=5.0) == 0
        server = int(await _read_when_written(ready.with_suffix(".pid")))
        assert await registry.kill_branch("prelude")

        [leftover] = await registry.leftovers()
        assert (leftover.pid, leftover.pgid, leftover.branch) == (server, server, UNREGISTERED_BRANCH)
        assert leftover.argv == ("sleep", "60")
        assert await registry.kill_all()
        assert not running(server)
        assert await registry.leftovers() == ()
