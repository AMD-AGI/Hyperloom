# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The liveness rule the reaper and ``poll`` share: the CLI's output within a window, or a tool it waits on."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.common.proctree import descendants
from hyperloom.orchestrator.specialists.dsh_launch import dsh_home
from hyperloom.orchestrator.specialists.reap_watch import cli_alive

WINDOW = 60.0
IDLE_CLI = "import time; time.sleep(60)"
WAITING_CLI = "import subprocess; subprocess.run(['sleep', '60'])"  # the CLI waits on a command it started


def start_cli(code: str) -> subprocess.Popen:
    """An agent CLI stand-in in its own session, as the runner launches one."""
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    if code == WAITING_CLI:
        end = time.time() + 20
        while not descendants(proc.pid):
            assert time.time() < end, "the tool call never started"
            time.sleep(0.02)
    return proc


@pytest.fixture
def cli() -> Iterator:
    procs: list[subprocess.Popen] = []

    def start(code: str) -> subprocess.Popen:
        procs.append(start_cli(code))
        return procs[-1]

    yield start
    for proc in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def write_output(workspace: Path, *, at: float) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    log = workspace / "process.log"
    log.write_text("event\n", encoding="utf-8")
    os.utime(log, (at, at))
    return log


def test_output_within_the_window_is_alive_without_a_tool(tmp_path: Path, cli) -> None:
    now = time.time()
    write_output(tmp_path, at=now - WINDOW + 5)
    assert cli_alive(tmp_path, cli(IDLE_CLI).pid, window_sec=WINDOW, now_unix=now)


def test_silence_past_the_window_with_no_tool_is_not_alive(tmp_path: Path, cli) -> None:
    now = time.time()
    write_output(tmp_path, at=now - WINDOW - 5)
    assert not cli_alive(tmp_path, cli(IDLE_CLI).pid, window_sec=WINDOW, now_unix=now)


def test_silence_past_the_window_is_alive_while_the_cli_waits_on_a_tool_it_started_after_its_output(
    tmp_path: Path, cli
) -> None:
    now = time.time()
    write_output(tmp_path, at=now - 10 * WINDOW)
    assert cli_alive(tmp_path, cli(WAITING_CLI).pid, window_sec=WINDOW, now_unix=now)


def test_a_child_older_than_the_last_output_is_not_a_tool_in_flight(tmp_path: Path, cli) -> None:
    proc = cli(WAITING_CLI)
    output = time.time() + 2.0  # the CLI wrote after the command it now has running had started
    write_output(tmp_path, at=output)
    assert not cli_alive(tmp_path, proc.pid, window_sec=WINDOW, now_unix=output + 10 * WINDOW)


def test_no_output_at_all_is_not_alive_even_with_a_tool_running(tmp_path: Path, cli) -> None:
    assert not cli_alive(tmp_path, cli(WAITING_CLI).pid, window_sec=WINDOW, now_unix=time.time())


def test_the_session_log_of_a_dsh_specialist_counts_as_output(tmp_path: Path, cli) -> None:
    now = time.time()
    write_output(tmp_path, at=now - 10 * WINDOW)
    session_log = dsh_home(tmp_path) / "sessions" / "step.jsonl"
    session_log.parent.mkdir(parents=True)
    session_log.write_text("{}\n", encoding="utf-8")
    os.utime(session_log, (now - 5, now - 5))
    assert cli_alive(tmp_path, cli(IDLE_CLI).pid, window_sec=WINDOW, now_unix=now)
