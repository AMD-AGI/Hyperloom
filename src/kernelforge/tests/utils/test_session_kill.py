# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session stops its own processes by pattern without reaching anyone else's, or the shell that asked."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest

from kernelforge.utils import process_reaping, session_kill


def _sleeper(tag: str, env: dict[str, str]) -> subprocess.Popen:
    """A process whose command line carries ``tag``, started with ``env``."""
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep(60)  # {tag}"], env=env)


def _wait_gone(process: subprocess.Popen) -> bool:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return True


@pytest.fixture()
def marked_session(monkeypatch):
    """This test process as one marked session, with the environment its children inherit."""
    token = uuid.uuid4().hex
    monkeypatch.setenv("FORGE_SESSION_OWNER", token)
    started: list[subprocess.Popen] = []
    yield {**os.environ, "FORGE_SESSION_OWNER": token}, started
    for process in started:
        process.kill()
        process.wait()


def test_only_this_sessions_matching_processes_are_signalled(marked_session, capsys):
    own_env, started = marked_session
    tag = f"sweep-{uuid.uuid4().hex}"
    mine = _sleeper(tag, own_env)
    other_session = _sleeper(tag, {**os.environ, "FORGE_SESSION_OWNER": uuid.uuid4().hex})
    unrelated_mine = _sleeper(f"other-{uuid.uuid4().hex}", own_env)
    started += [mine, other_session, unrelated_mine]
    time.sleep(0.2)

    assert session_kill.main([tag]) == 0

    assert _wait_gone(mine)
    assert other_session.poll() is None
    assert unrelated_mine.poll() is None
    assert f"sent SIGTERM to pid {mine.pid}" in capsys.readouterr().out


def test_the_caller_and_its_ancestors_are_never_candidates(marked_session):
    _own_env, _started = marked_session

    candidates = process_reaping.own_session_processes()

    assert os.getpid() not in candidates
    assert os.getppid() not in candidates


def test_nothing_matches_outside_a_marked_session(monkeypatch, capsys):
    monkeypatch.delenv("FORGE_SESSION_OWNER", raising=False)

    assert process_reaping.own_session_processes() == {}
    assert session_kill.main(["python"]) == 1
    assert "nothing was signalled" in capsys.readouterr().out


def test_the_command_a_session_is_told_to_run_names_this_interpreter():
    assert session_kill.session_kill_command() == f"{sys.executable} -m kernelforge.utils.session_kill"
