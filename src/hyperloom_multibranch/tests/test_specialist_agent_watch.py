# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.specialist.agent_sandbox import LocalAgentSandbox
from hyperloom.common.proctree import collect_tree, kill_tree
from hyperloom_multibranch.specialist.agent_watch import REGISTRY_PREFIX, beyond, marked_pids


@pytest.fixture(autouse=True)
def _local_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "local-tmp"))
    (tmp_path / "local-tmp").mkdir()


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state not in ("Z", "X")


def _until(predicate: Callable[[], object], seconds: float = 10.0) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, "timed out waiting"
        time.sleep(0.05)


def _pids(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().split()] if path.exists() else []


@pytest.mark.parametrize(
    ("environ", "reaches"),
    [
        ({"ROCR_VISIBLE_DEVICES": "4,5", "HIP_VISIBLE_DEVICES": "0,1"}, ""),
        # A HIP-level mask indexes into what ROCr exposes, so card 5 of the host is out of its reach.
        ({"ROCR_VISIBLE_DEVICES": "4", "HIP_VISIBLE_DEVICES": "5"}, ""),
        ({"ROCR_VISIBLE_DEVICES": "4", "HIP_VISIBLE_DEVICES": ""}, ""),
        ({"ROCR_VISIBLE_DEVICES": "1"}, "ROCR_VISIBLE_DEVICES=1"),
        ({"ROCR_VISIBLE_DEVICES": "4,6"}, "ROCR_VISIBLE_DEVICES=4,6"),
        ({"ROCR_VISIBLE_DEVICES": ""}, "ROCR_VISIBLE_DEVICES="),
        ({"ROCR_VISIBLE_DEVICES": "GPU-4b2c1a9f"}, "ROCR_VISIBLE_DEVICES=GPU-4b2c1a9f"),
        # ROCr reads HSA_VISIBLE_DEVICES only when ROCR_VISIBLE_DEVICES is unset.
        ({"ROCR_VISIBLE_DEVICES": "4", "HSA_VISIBLE_DEVICES": "6"}, ""),
        ({"HSA_VISIBLE_DEVICES": "6"}, "HSA_VISIBLE_DEVICES=6"),
        ({"HIP_VISIBLE_DEVICES": "4"}, "no ROCR_VISIBLE_DEVICES"),
    ],
)
def test_beyond_names_the_mask_that_reaches_another_card(environ: dict[str, str], reaches: str) -> None:
    assert beyond(environ, (4, 5)) == reaches


def test_a_command_that_sets_its_own_rocr_mask_is_killed_and_the_rest_left(tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    sandbox = LocalAgentSandbox(cards=(4,), python_path=(), root=tmp_path / "sandbox", base_env=dict(os.environ))
    command = (
        f"(sleep 100 & echo $! >> {pids}); (HIP_VISIBLE_DEVICES=5 sleep 100 & echo $! >> {pids}); "
        f"(ROCR_VISIBLE_DEVICES=1 sleep 100 2>> {tmp_path / 'said'} & echo $! >> {pids}); sleep 100"
    )

    async def scenario() -> tuple[bool, bool, bool]:
        process = await sandbox.start(command, tmp_path, {})
        try:
            _until(lambda: len(_pids(pids)) == 3)
            plain, hip, rocr = _pids(pids)
            _until(lambda: not _alive(rocr))
            return _alive(plain), _alive(hip), process.returncode is None
        finally:
            await process.kill()

    plain, hip, agent = asyncio.run(scenario())

    assert plain and hip and agent
    assert "ROCR_VISIBLE_DEVICES=1" in (tmp_path / "sandbox" / "watch.log").read_text()
    assert "killed by the sandbox: ROCR_VISIBLE_DEVICES=1" in (tmp_path / "said").read_text()


OWNER = """
import asyncio, os, sys, time
from pathlib import Path
from hyperloom_multibranch.specialist.agent_sandbox import LocalAgentSandbox

root, pids, command = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

async def main():
    sandbox = LocalAgentSandbox(cards=(0,), python_path=(), root=root, base_env=dict(os.environ))
    await sandbox.start(command.format(pids=pids), root, {})
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
"""
#: The dummy agent: a server in a session of its own, as a Claude CLI's shell command starts one, and itself.
AGENT = "setsid sleep 1000 & echo $! >> {pids}; echo $$ >> {pids}; sleep 1000"
#: The same agent, deaf to SIGTERM as a server stuck in its shutdown is.
DEAF_AGENT = "trap '' TERM; " + AGENT


def _owner(tmp_path: Path, agent: str = AGENT, started: int = 2) -> tuple[subprocess.Popen[bytes], Path]:
    pids = tmp_path / "pids"
    script = tmp_path / "owner.py"
    script.write_text(textwrap.dedent(OWNER))
    env = {**os.environ, "TMPDIR": str(tmp_path / "local-tmp")}
    argv = [sys.executable, str(script), str(tmp_path / "sandbox"), str(pids), agent]
    owner = subprocess.Popen(argv, env=env)
    _until(lambda: len(_pids(pids)) == started, 60.0)
    return owner, pids


def test_the_agent_s_whole_tree_ends_when_its_owner_is_killed(tmp_path: Path) -> None:
    owner, pids = _owner(tmp_path)
    agent = _pids(pids)
    assert all(_alive(pid) for pid in agent)

    owner.send_signal(signal.SIGKILL)
    owner.wait()

    _until(lambda: not any(_alive(pid) for pid in agent))
    # The launchers go last, once what they hold has ended.
    _until(lambda: marked_pids(str(tmp_path / "sandbox")) == [])


#: Leaves the agent's tree by a double fork and setsid() and drops the marker with its environment.
ESCAPE = (
    "import os, sys\n"
    "if os.fork(): os._exit(0)\n"
    "os.setsid()\n"
    "if os.fork(): os._exit(0)\n"
    "open(sys.argv[1], 'a').write(f'{os.getpid()}\\n')\n"
    "os.execve('/bin/sleep', ['sleep', '60'], {})\n"
)


def test_an_unmarked_process_that_left_the_agent_s_tree_ends_when_its_owner_is_killed(tmp_path: Path) -> None:
    script = tmp_path / "escape.py"
    script.write_text(ESCAPE)
    owner, pids = _owner(tmp_path, f"{shlex.quote(sys.executable)} {script} {{pids}}; " + AGENT, started=3)
    escaped = _pids(pids)
    try:
        assert all(_alive(pid) for pid in escaped)

        owner.send_signal(signal.SIGKILL)
        owner.wait()

        _until(lambda: not any(_alive(pid) for pid in escaped))
    finally:
        _kill(escaped)


def test_a_terminated_watch_ends_a_deaf_agent_within_the_kill_all_grace_and_removes_its_registry(
    tmp_path: Path,
) -> None:
    owner, pids = _owner(tmp_path, DEAF_AGENT)
    agent = _pids(pids)
    [watch] = _watches_of(owner.pid)
    try:
        _until(lambda: _takes_sigterm(watch))
        # What Processes.kill_all does to the session's children it did not register: SIGTERM, SIGKILL after a grace.
        kill_tree(collect_tree([watch]))
        _until(lambda: not any(_alive(pid) for pid in agent), 1.0)
        assert not (tmp_path / "local-tmp" / f"{REGISTRY_PREFIX}{owner.pid}").exists()
    finally:
        owner.kill()
        owner.wait()
        _kill(marked_pids(str(tmp_path / "sandbox")))


def test_the_watch_logs_a_registry_entry_it_cannot_read_and_keeps_confining_the_others(tmp_path: Path) -> None:
    late = "echo $$ >> {pids}; sleep 3; (ROCR_VISIBLE_DEVICES=1 sleep 100 & echo $! >> {pids}); sleep 1000"
    owner, pids = _owner(tmp_path, late, started=1)
    try:
        registry = tmp_path / "local-tmp" / f"{REGISTRY_PREFIX}{owner.pid}"
        (registry / "torn.json").write_text("{", encoding="utf-8")
        _until(lambda: len(_pids(pids)) == 2)
        _until(lambda: not _alive(_pids(pids)[1]))
        log = (tmp_path / "sandbox" / "watch.log").read_text()
        assert log.count("torn.json cannot be read") == 1
        assert "ROCR_VISIBLE_DEVICES=1 reaches beyond its cards" in log
    finally:
        owner.kill()
        owner.wait()
        _kill(marked_pids(str(tmp_path / "sandbox")))


def test_a_watch_that_died_is_said_and_started_again_over_the_sandboxes_it_watched(tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    sandbox = LocalAgentSandbox(cards=(4,), python_path=(), root=tmp_path / "sandbox", base_env=dict(os.environ))
    command = f"sleep 2; (ROCR_VISIBLE_DEVICES=1 sleep 100 & echo $! >> {pids}); sleep 100"

    async def scenario() -> tuple[str, str]:
        process = await sandbox.start(command, tmp_path, {})
        try:
            killed = _watches_of(os.getpid())
            for watch in killed:
                os.kill(watch, signal.SIGKILL)
            # A dying watch's cmdline reads empty before it is a zombie its parent can reap, so wait on its state.
            _until(lambda: not any(_alive(pid) for pid in killed))
            said, again = sandbox.watched(), sandbox.watched()
            _until(lambda: _pids(pids) and not _alive(_pids(pids)[0]))
            return said, again
        finally:
            await process.kill()

    said, again = asyncio.run(scenario())

    assert "the sandbox's watch had ended (exit -9) and was started again" in said
    assert again == ""


def _watches_of(owner: int) -> list[int]:
    return [pid for pid in _children(owner) if b"agent_watch" in Path(f"/proc/{pid}/cmdline").read_bytes()]


def _kill(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _takes_sigterm(pid: int) -> bool:
    """Whether ``pid`` handles SIGTERM or holds it to wait for, rather than dying of it. While a process sleeps in
    ``sigtimedwait`` the kernel unblocks what it waits for, so its status shows SIGTERM blocked only between waits."""

    status = Path(f"/proc/{pid}/status").read_text().splitlines()
    masks = [int(line.split()[1], 16) for line in status if line.startswith(("SigCgt", "SigBlk"))]
    waiting = "sigtimedwait" in Path(f"/proc/{pid}/wchan").read_text()
    return waiting or any(mask & (1 << (signal.SIGTERM - 1)) for mask in masks)


def _children(pid: int) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if parent == pid and _alive(int(entry.name)):
            found.append(int(entry.name))
    return found
