# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from hyperloom_multibranch.specialist.agent_sandbox import TIMED_OUT, LocalAgentSandbox, marked_pids

BASE = {
    "PATH": os.environ["PATH"],
    "HOME": "/home/agent",
    "PYTHONPATH": "/existing",
    "HIP_VISIBLE_DEVICES": "0,1",
    "CUDA_VISIBLE_DEVICES": "0",
    "GPU_DEVICE_ORDINAL": "0",
    "HSA_VISIBLE_DEVICES": "0",
    "ROCR_VISIBLE_DEVICES": "0,1,2,3",
}


@pytest.fixture(autouse=True)
def _local_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "local-tmp"))
    (tmp_path / "local-tmp").mkdir()


class _Processes:
    def __init__(self) -> None:
        self.adopted: list[tuple[int, str, tuple[str, ...]]] = []

    def adopt(self, pid: int, *, branch: str, argv: Sequence[str]) -> object:
        self.adopted.append((pid, branch, tuple(argv)))
        return None


def _sandbox(tmp_path: Path, processes: _Processes | None = None, branch: str = "") -> LocalAgentSandbox:
    return LocalAgentSandbox(
        cards=(4, 5),
        python_path=(tmp_path / "worktree" / "python",),
        root=tmp_path / "sandbox",
        base_env=BASE,
        grace_seconds=1.0,
        poll_seconds=0.02,
        processes=processes,
        branch=branch,
    )


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state not in ("Z", "X")


async def _until(predicate: Callable[[], object], seconds: float = 3.0) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, "timed out waiting"
        await asyncio.sleep(0.02)


def test_env_drops_masks_and_pins_cards_paths_and_tmpdir(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    shim_path = f"/agent/bin{os.pathsep}{BASE['PATH']}"
    env = sandbox.env({"PATH": shim_path, "FOO": "bar", "HIP_VISIBLE_DEVICES": "7"})

    assert not {"CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "HSA_VISIBLE_DEVICES"} & set(env)
    assert env["HIP_VISIBLE_DEVICES"] == "4,5" and "ROCR_VISIBLE_DEVICES" not in env
    # HIP unbundles its code objects in TMPDIR, which fails on a network file system such as the branch's directory.
    assert Path(env["TMPDIR"]).parent == Path(tempfile.gettempdir()) and Path(env["TMPDIR"]).is_dir()
    other = LocalAgentSandbox(cards=(6,), python_path=(), root=tmp_path / "other", base_env=BASE)
    assert other.env({})["TMPDIR"] != env["TMPDIR"]
    assert _sandbox(tmp_path).env({})["TMPDIR"] == env["TMPDIR"]
    assert env["PYTHONPATH"] == f"{tmp_path / 'worktree' / 'python'}{os.pathsep}/existing"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PATH"] == f"{tmp_path / 'sandbox' / 'bin'}{os.pathsep}{shim_path}"
    assert env["FOO"] == "bar" and env["HOME"] == "/home/agent"


def test_run_sees_the_env_in_its_own_session(tmp_path: Path) -> None:
    done = asyncio.run(
        _sandbox(tmp_path).run(
            'echo "$HIP_VISIBLE_DEVICES $(command -v pkill)"; ps -o sess= -p $$; echo $$', tmp_path, {}, 5
        )
    )

    devices, sid, pid = done.output.split("\n")[:3]
    assert done.returncode == 0
    assert devices == f"4,5 {tmp_path / 'sandbox' / 'bin' / 'pkill'}"
    assert sid.strip() == pid


def test_run_timeout_kills_the_session(tmp_path: Path) -> None:
    pids = tmp_path / "pids"

    async def scenario() -> object:
        return await _sandbox(tmp_path).run(f"sleep 100 & echo $! > {pids}; echo started; sleep 100", tmp_path, {}, 0.5)

    done = asyncio.run(scenario())

    assert done.returncode == TIMED_OUT
    assert "started" in done.output and "timed out after 0.5 s" in done.output
    assert not _alive(int(pids.read_text()))


def test_kill_reaches_a_grandchild_that_left_the_group(tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    processes = _Processes()

    async def scenario() -> tuple[int, int, int | None]:
        sandbox = _sandbox(tmp_path, processes=processes, branch="b07")
        process = await sandbox.start(
            f"bash -c 'set -m; nohup sleep 100 >/dev/null 2>&1 & echo $! > {pids}; wait' & wait", tmp_path, {}
        )
        await _until(lambda: pids.exists() and pids.read_text().strip())
        grandchild = int(pids.read_text())
        assert os.getpgid(grandchild) != process.popen.pid
        await process.kill()
        return process.popen.pid, grandchild, process.returncode

    leader, grandchild, returncode = asyncio.run(scenario())

    assert not _alive(grandchild)
    assert marked_pids(str(tmp_path / "sandbox")) == []
    assert returncode is not None
    assert processes.adopted[0][:2] == (leader, "b07")


def test_kill_waits_the_grace_before_sigkill(tmp_path: Path) -> None:
    async def scenario() -> float:
        process = await _sandbox(tmp_path).start("trap '' TERM; echo ready; sleep 100 & wait", tmp_path, {})
        await _until(lambda: "ready" in process.log.read_text())
        began = time.monotonic()
        await process.kill()
        return time.monotonic() - began

    assert 1.0 <= asyncio.run(scenario()) < 3.0


def test_pkill_inside_spares_the_same_process_outside(tmp_path: Path) -> None:
    marker = f"marker{uuid.uuid4().hex}"
    marked_argv = [sys.executable, "-c", "import time; time.sleep(100)", marker]
    outside = subprocess.Popen(marked_argv, start_new_session=True)
    pids = tmp_path / "pids"
    try:

        async def scenario() -> tuple[str, int, int | None]:
            # The agent's own commands are what pkill confines: a run is a session of its own, not the agent's.
            process = await _sandbox(tmp_path).start(
                f"{shlex.join(marked_argv)} & echo $! > {pids}; sleep 0.3; pkill -TERM -f {marker}; echo rc=$?; wait",
                tmp_path,
                {},
            )
            try:
                await _until(lambda: process.returncode is not None)
            finally:
                await process.kill()
            return await process.output(), int(pids.read_text()), process.returncode

        output, marked, returncode = asyncio.run(scenario())

        assert "rc=0" in output and returncode == 0, output
        assert not _alive(marked)
        assert outside.poll() is None
    finally:
        outside.kill()
        outside.wait()


def test_pkill_spares_its_caller_whose_command_line_matches(tmp_path: Path) -> None:
    marker = f"marker{uuid.uuid4().hex}"
    command = f'{sys.executable} -c \'import subprocess, sys; sys.exit(subprocess.call(["pkill", "-f", "{marker}"]))\''
    done = asyncio.run(_sandbox(tmp_path).run(command, tmp_path, {}, 5))

    assert done.returncode == 1, done.output


def test_pgrep_lists_no_process_outside_the_sandbox_whatever_it_selects(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    done = asyncio.run(sandbox.run("pgrep -s 1 . ; echo exit=$?", tmp_path, {}, 5))

    assert done.output.strip() == "exit=1"


def test_killall_kills_by_name_within_the_session(tmp_path: Path) -> None:
    async def scenario() -> tuple[object, object]:
        sandbox = _sandbox(tmp_path)
        killed = await sandbox.run("sleep 100 & sleep 0.2; killall -TERM sleep; wait", tmp_path, {}, 5)
        missing = await sandbox.run("killall nosuchprocessname", tmp_path, {}, 5)
        return killed, missing

    killed, missing = asyncio.run(scenario())

    assert killed.returncode == 143 or killed.returncode == 0, killed.output
    assert missing.returncode == 1 and "no process found" in missing.output


def test_output_is_the_logs_tail_and_logs_are_numbered(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, int | None]:
        sandbox = _sandbox(tmp_path)
        await sandbox.run("true", tmp_path, {}, 5)
        process = await sandbox.start("head -c 100000 /dev/zero | tr '\\0' a; echo; echo last-line", tmp_path, {})
        await _until(lambda: process.returncode is not None)
        return await process.output(), process.returncode

    output, returncode = asyncio.run(scenario())

    assert returncode == 0
    assert len(output) == 64 * 1024 and output.endswith("last-line\n")
    assert sorted(log.name for log in (tmp_path / "sandbox" / "logs").iterdir()) == ["0001-run.log", "0002-start.log"]
    assert next(_sandbox(tmp_path)._numbers) == 3


@pytest.mark.parametrize("tool", ["pkill", "pgrep"])
def test_a_pattern_is_required(tmp_path: Path, tool: str) -> None:
    done = asyncio.run(_sandbox(tmp_path).run(f"{tool} -TERM" if tool == "pkill" else tool, tmp_path, {}, 5))

    assert done.returncode != 0 and "give a pattern or a selection" in done.output


class _Late(_Processes):
    def adopt(self, pid: int, *, branch: str, argv: Sequence[str]) -> object:
        raise ValueError(f"{pid} ended")


async def test_a_command_that_ends_before_the_registry_takes_it_still_runs(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path, _Late(), "b07")

    process = await sandbox.start("echo started", tmp_path, {})

    await _until(lambda: process.returncode is not None)
    assert "started" in await process.output()


def test_a_server_started_in_a_session_of_its_own_is_still_the_sandbox_s(tmp_path: Path) -> None:
    marker = f"server{uuid.uuid4().hex}"
    pids = tmp_path / "pids"
    # A Claude CLI runs each Bash command in a new session, so a server it starts leaves the agent's session.
    serve = f"setsid {shlex.quote(sys.executable)} -c 'import time; time.sleep(100)' {marker} & echo $! > {pids}"

    async def scenario() -> tuple[int, int, bool]:
        sandbox = _sandbox(tmp_path)
        process = await sandbox.start(f"{serve}; sleep 100", tmp_path, {})
        await _until(lambda: pids.exists() and pids.read_text().strip())
        server = int(pids.read_text())
        await _until(lambda: _alive(server))
        assert os.getsid(server) != process.popen.pid
        stopped = await sandbox.run(f"pkill -f {marker}", tmp_path, {}, 10.0)
        await _until(lambda: not _alive(server))
        again = await sandbox.start(f"{serve}; sleep 100", tmp_path, {})
        await _until(lambda: pids.exists() and int(pids.read_text()) != server)
        second = int(pids.read_text())
        await again.kill()
        return stopped.returncode, second, _alive(second)

    stopped, _second, alive = asyncio.run(scenario())

    assert stopped == 0
    assert not alive


def test_pgrep_does_not_find_itself_so_a_wait_for_a_process_to_end_ends(tmp_path: Path) -> None:
    marker = f"bench{uuid.uuid4().hex}"
    # Arbor waits for its bench to end with `until ! pgrep -f validator/bench.sh`, from a model's shell command.
    script = (
        f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(1)' {marker} & "
        f"until ! pgrep -f {marker} >/dev/null; do sleep 0.2; done; echo ended; "
        f"{shlex.quote(sys.executable)} -c \"import subprocess; print('rc', subprocess.run(['pgrep', '-f', 'none{marker}']).returncode)\""
    )

    done = asyncio.run(_sandbox(tmp_path).run(script, tmp_path, {}, 20))

    assert "ended" in done.output and "rc 1" in done.output
