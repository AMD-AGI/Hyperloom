# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import os
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
from hyperloom_multibranch.specialist.agent_watch import beyond, marked_pids


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

root, pids = Path(sys.argv[1]), Path(sys.argv[2])

async def main():
    sandbox = LocalAgentSandbox(cards=(0,), python_path=(), root=root, base_env=dict(os.environ))
    # The dummy agent: a server in a session of its own, as a Claude CLI's shell command starts one, and itself.
    await sandbox.start(f"setsid sleep 1000 & echo $! >> {pids}; echo $$ >> {pids}; sleep 1000", root, {})
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
"""


def _owner(tmp_path: Path) -> tuple[subprocess.Popen[bytes], Path]:
    pids = tmp_path / "pids"
    script = tmp_path / "owner.py"
    script.write_text(textwrap.dedent(OWNER))
    env = {**os.environ, "TMPDIR": str(tmp_path / "local-tmp")}
    owner = subprocess.Popen([sys.executable, str(script), str(tmp_path / "sandbox"), str(pids)], env=env)
    _until(lambda: len(_pids(pids)) == 2, 60.0)
    return owner, pids


def test_the_agent_s_whole_tree_ends_when_its_owner_is_killed(tmp_path: Path) -> None:
    owner, pids = _owner(tmp_path)
    agent = _pids(pids)
    assert all(_alive(pid) for pid in agent)

    owner.send_signal(signal.SIGKILL)
    owner.wait()

    _until(lambda: not any(_alive(pid) for pid in agent))
    assert marked_pids(str(tmp_path / "sandbox")) == []


def test_the_agent_s_whole_tree_ends_when_the_watch_is_terminated(tmp_path: Path) -> None:
    owner, pids = _owner(tmp_path)
    agent = _pids(pids)
    watch = [pid for pid in _children(owner.pid) if pid not in agent]
    try:
        assert len(watch) == 1
        _until(lambda: _catches_sigterm(watch[0]))
        # The session's stop terminates every child it did not register, the watch among them.
        os.kill(watch[0], signal.SIGTERM)
        _until(lambda: not any(_alive(pid) for pid in agent))
    finally:
        owner.kill()
        owner.wait()


def _catches_sigterm(pid: int) -> bool:
    caught = next(line for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("SigCgt"))
    return bool(int(caught.split()[1], 16) & (1 << (signal.SIGTERM - 1)))


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
