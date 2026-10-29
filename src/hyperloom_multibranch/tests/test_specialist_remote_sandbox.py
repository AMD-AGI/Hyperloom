# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The remote agent sandbox against a sandbox whose commands run on this host, so its launcher and kill scripts run."""

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest

from hyperloom.sandbox import SandboxGone
from hyperloom_multibranch.specialist.agent_sandbox import TIMED_OUT
from hyperloom_multibranch.specialist.agent_watch import MARKER, marked_pids
from hyperloom_multibranch.specialist.remote_sandbox import (
    LOST,
    PROCESS_MARKER,
    RemoteAgentSandbox,
    claude_config_dir,
    forwarded_env,
)
from hyperloom_multibranch.tests.subprocess_sandbox import SubprocessSandbox


def _remote(tmp_path: Path, sandbox: SubprocessSandbox | None = None, **forwarded: str) -> RemoteAgentSandbox:
    return RemoteAgentSandbox(
        sandbox or SubprocessSandbox(),
        tmp_path / "sandbox",
        (tmp_path / "worktree" / "python",),
        cards=2,
        forwarded=forwarded,
        grace_seconds=1.0,
        poll_seconds=0.05,
        probe_seconds=0.0,
    )


async def _ended(process, seconds: float = 10.0) -> None:
    until = time.monotonic() + seconds
    while process.returncode is None and time.monotonic() < until:
        await process.output()
        time.sleep(0.05)
    assert process.returncode is not None


async def test_a_command_runs_on_the_sandboxs_environment_with_only_what_is_forwarded(tmp_path):
    remote = _remote(tmp_path, ANTHROPIC_API_KEY="forwarded")
    cwd = tmp_path / "work"
    cwd.mkdir()
    shim = tmp_path / "agent" / "bin"
    shown = "pwd; echo PATH=$PATH; env | sort"
    env = {"PATH": f"{shim}{os.pathsep}{os.environ['PATH']}", "OWN": "1", "HIP_VISIBLE_DEVICES": "5"}

    done = await remote.run(shown, cwd, env, timeout=10)

    lines = done.output.splitlines()
    assert done.returncode == 0 and lines[0] == str(cwd)
    assert lines[1] == f"PATH={shim}{os.pathsep}{os.environ['PATH']}"
    assert {"OWN=1", "IMAGE_ONLY=yes", "ANTHROPIC_API_KEY=forwarded", "TMPDIR=/tmp/hl", "IS_SANDBOX=1"} <= set(lines)
    assert f"PYTHONPATH={tmp_path / 'worktree' / 'python'}" in lines
    assert f"CLAUDE_CONFIG_DIR={claude_config_dir(remote.root)}" in lines and f"HOME={remote.root / 'home'}" in lines
    assert "HIP_VISIBLE_DEVICES=0,1" in lines and "ROCR_VISIBLE_DEVICES=6,7" in lines
    assert f"{MARKER}={remote.root}" in lines and f"{PROCESS_MARKER}={remote.root}/1" in lines
    assert not any(line.startswith("HL_") for line in lines)


async def test_a_command_that_exits_reports_its_code_and_output_through_the_shared_files(tmp_path):
    remote = _remote(tmp_path)

    process = await remote.start("echo started; exit 3", tmp_path, {})
    await _ended(process)

    assert process.returncode == 3 and "started" in await process.output()
    assert process.log.name == "0001-start.log"


async def test_a_run_past_its_timeout_is_killed_with_what_it_started(tmp_path):
    remote = _remote(tmp_path)

    done = await remote.run("setsid sleep 300 & sleep 300", tmp_path, {}, timeout=0.5)

    assert done.returncode == TIMED_OUT and "timed out after 0.5 s" in done.output
    assert marked_pids(str(remote.root)) == []


async def test_kill_ends_one_command_and_stop_every_command(tmp_path):
    remote = _remote(tmp_path)
    first = await remote.start("sleep 300", tmp_path, {})
    second = await remote.start("setsid sleep 300 & wait", tmp_path, {})

    await first.kill()

    assert first.returncode == 128 + signal.SIGTERM and second.returncode is None
    await remote.stop()
    await _ended(second)
    assert marked_pids(str(remote.root)) == []


async def test_a_command_whose_launcher_is_gone_without_an_exit_code_ends_lost(tmp_path):
    remote = _remote(tmp_path)
    process = await remote.start("sleep 300", tmp_path, {})
    launcher, command = map(int, (remote.root / "run" / "1.pid").read_text().split())

    os.kill(launcher, signal.SIGKILL)
    os.kill(command, signal.SIGKILL)
    await _ended(process)

    assert process.returncode == LOST and "left no exit code; exit 137" in await process.output()


async def test_a_command_of_a_sandbox_that_ended_ends_lost_and_stops_quietly(tmp_path):
    sandbox = SubprocessSandbox()
    remote = _remote(tmp_path, sandbox)
    process = await remote.start("sleep 300", tmp_path, {})
    sandbox.gone = True

    output = await process.output()
    await process.kill()
    await remote.stop()

    assert process.returncode == LOST and "the sandbox ended: sandbox sbx-1 ended; exit 137" in output
    with pytest.raises(SandboxGone):
        await remote.start("true", tmp_path, {})
    sandbox.gone = False
    await remote.stop()


async def test_alive_follows_the_sandboxs_processes(tmp_path):
    remote = _remote(tmp_path)
    assert remote.alive() is False

    process = await remote.start("sleep 300", tmp_path, {})
    assert remote.alive() is True
    await process.kill()
    until = time.monotonic() + 10
    while remote.alive() and time.monotonic() < until:
        await asyncio.sleep(0.05)

    assert remote.alive() is False


async def test_a_sandbox_at_the_same_root_numbers_its_commands_after_the_last(tmp_path):
    await _remote(tmp_path).run("true", tmp_path, {}, timeout=10)

    process = await _remote(tmp_path).start("true", tmp_path, {})

    assert process.log.name == "0002-start.log"


async def test_a_command_in_a_directory_the_sandbox_lacks_fails(tmp_path):
    done = await _remote(tmp_path).run("true", tmp_path / "missing", {}, timeout=10)

    assert done.returncode == 1 and "missing" in done.output


def test_only_the_gateway_its_credentials_and_the_ca_bundle_are_forwarded():
    environ = {
        "ANTHROPIC_API_KEY": "k",
        "OPENAI_API_KEY": "o",
        "ANTHROPIC_BASE_URL": "u",
        "CLAUDE_MODEL": "m",
        "GEAK_CLAUDE_MODEL": "g",
        "SSL_CERT_FILE": "/tmp/hyperloom-host/ca-bundle.pem",
        "HOME": "/root",
        "PATH": "/usr/bin",
        "USER_DATA_PATH": "/data",
        "AWS_SECRET_ACCESS_KEY": "s",
    }

    assert forwarded_env(environ) == {
        "ANTHROPIC_API_KEY": "k",
        "OPENAI_API_KEY": "o",
        "ANTHROPIC_BASE_URL": "u",
        "CLAUDE_MODEL": "m",
        "GEAK_CLAUDE_MODEL": "g",
        "SSL_CERT_FILE": "/tmp/hyperloom-host/ca-bundle.pem",
    }
