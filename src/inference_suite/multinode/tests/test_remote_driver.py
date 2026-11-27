# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from inference_suite.multinode import remote_driver

_DRIVER = Path(remote_driver.__file__)


def _drive(tmp_path, action, mode="in_place"):
    env = {
        **os.environ,
        "HL_ACTION": action,
        "HL_PIDFILE": str(tmp_path / "server.pid"),
        "HL_LAUNCH_PATH": str(tmp_path / "launch.sh"),
        "HL_NEEDLE": f"sleep 37{os.getpid()}",
        "HL_LAUNCH_MODE": mode,
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


def test_driver_spawns_then_kills_a_detached_server(tmp_path):
    (tmp_path / "launch.sh").write_text(f"#!/bin/bash\nexec sleep 37{os.getpid()}\n")
    spawned = _drive(tmp_path, "restart")
    assert spawned["action"] == "spawn" and (tmp_path / "server.pid").read_text() == str(spawned["pid"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not _alive(spawned["pid"]):
        time.sleep(0.1)
    assert _alive(spawned["pid"])
    stopped = _drive(tmp_path, "stop")
    assert stopped == {"status": "ok", "action": "stop-kill", "pid": spawned["pid"]}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _alive(spawned["pid"]):
        time.sleep(0.1)
    assert not _alive(spawned["pid"]) and not (tmp_path / "server.pid").exists()
    assert _drive(tmp_path, "stop")["action"] == "none"


def test_direct_mode_stops_only_the_server_it_recorded(tmp_path):
    decoy = subprocess.Popen(["sleep", f"37{os.getpid()}"])
    try:
        (tmp_path / "launch.sh").write_text(f"#!/bin/bash\nexec sleep 37{os.getpid()}\n")
        assert _drive(tmp_path, "stop", mode="direct") == {"status": "ok", "action": "none", "pid": 0}
        assert decoy.poll() is None
        spawned = _drive(tmp_path, "restart", mode="direct")
        assert spawned["pid"] != decoy.pid
        assert _drive(tmp_path, "stop", mode="direct") == {"status": "ok", "action": "stop-kill", "pid": spawned["pid"]}
        assert decoy.poll() is None
    finally:
        decoy.kill()
        decoy.wait()
