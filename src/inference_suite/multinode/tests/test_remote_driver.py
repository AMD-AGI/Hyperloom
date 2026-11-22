# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from inference_suite.multinode import remote_driver

_DRIVER = Path(remote_driver.__file__)


def _drive(tmp_path, action):
    env = {
        **os.environ,
        "HL_ACTION": action,
        "HL_PIDFILE": str(tmp_path / "server.pid"),
        "HL_LAUNCH_PATH": str(tmp_path / "launch.sh"),
        "HL_NEEDLE": f"sleep 37{os.getpid()}",
        "HL_ENV_SOURCE": "session",
        "HL_CONSOLE": str(tmp_path / "no-console"),
    }
    done = subprocess.run([sys.executable, "-"], input=_DRIVER.read_text(), env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False


def _wait_for(condition):
    """Poll ``condition`` for up to 10s; return its last value."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not condition():
        time.sleep(0.1)
    return condition()


def test_driver_spawns_then_kills_a_detached_server(tmp_path):
    (tmp_path / "launch.sh").write_text(f"#!/bin/bash\nexec sleep 37{os.getpid()}\n")
    spawned = _drive(tmp_path, "restart")
    assert spawned["action"] == "spawn" and (tmp_path / "server.pid").read_text() == str(spawned["pid"])
    assert _wait_for(lambda: _alive(spawned["pid"]))
    stopped = _drive(tmp_path, "stop")
    assert stopped == {"status": "ok", "action": "stop-kill", "pid": spawned["pid"]}
    assert _wait_for(lambda: not _alive(spawned["pid"])) and not (tmp_path / "server.pid").exists()
    assert _drive(tmp_path, "stop")["action"] == "none"


def test_a_pidfile_naming_an_unrelated_process_is_not_the_server(tmp_path):
    unrelated = subprocess.Popen(["sleep", "30"])
    pidfile = tmp_path / "decode.pid"
    pidfile.write_text(str(unrelated.pid), encoding="utf-8")
    try:
        assert _wait_for(lambda: _alive(unrelated.pid))
        assert remote_driver.find_server(str(pidfile), f"sglang.launch_server-{os.getpid()}") != unrelated.pid
    finally:
        unrelated.kill()
        unrelated.wait()


def test_a_missing_executable_is_refused_before_the_server_is_touched(tmp_path):
    server = subprocess.Popen(["/bin/sh", "-c", "/bin/sleep 30 & wait"])
    try:
        assert _wait_for(lambda: remote_driver.descendant_pids(server.pid))
        children = remote_driver.descendant_pids(server.pid)
        with pytest.raises(OSError, match="not an executable"):
            remote_driver.replace_process_image(server.pid, ["/nonexistent/hl-no-such-binary"], str(tmp_path / "env"))
        time.sleep(0.3)
        assert server.poll() is None and remote_driver.descendant_pids(server.pid) == children
    finally:
        for child in remote_driver.descendant_pids(server.pid):
            os.kill(child, 9)
        server.kill()
        server.wait()


@pytest.mark.parametrize(("failing", "killed"), [("_syscall_gadget", False), ("_stack_slot", False), ("", True)])
def test_descendants_are_killed_only_once_the_exec_starts(monkeypatch, tmp_path, failing, killed):
    kills = []

    def _fail(*_args):
        raise OSError(errno.ENOENT, "boom")

    monkeypatch.setattr(remote_driver, "_ptrace", lambda *_args: None)
    monkeypatch.setattr(remote_driver, "_registers", lambda *_args: None)
    monkeypatch.setattr(remote_driver, "_syscall_gadget", lambda _pid: 0x1000)
    monkeypatch.setattr(remote_driver, "_stack_slot", lambda _pid, _rsp, _size: 0x2000)
    monkeypatch.setattr(remote_driver, "_poke", lambda *_args: None)
    monkeypatch.setattr(remote_driver, "baseline_environ", lambda _pid, _snapshot: [])
    monkeypatch.setattr(remote_driver, "descendant_pids", lambda _pid: [4243])
    if failing:
        monkeypatch.setattr(remote_driver, failing, _fail)
    monkeypatch.setattr(os, "waitpid", lambda pid, _flags: (pid, 0))
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))

    with pytest.raises(OSError):
        remote_driver.replace_process_image(4242, ["/bin/true"], str(tmp_path / "env"))

    assert kills == ([(4243, 9)] if killed else [])
