# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Linux regressions using only isolated, short-lived CPU subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_PROCESS_SCENARIO = r"""
import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from hyperloom.common.proctree import group_alive, running
from hyperloom.orchestrator.actions.executors._ray_serving import ManagedServerProcess

# Only this isolated harness adopts and reaps its orphaned descendants.
libc = ctypes.CDLL("libc.so.6", use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0, "could not become a child subreaper"
root_exits = sys.argv[1] == "exited"
pid_file = Path(sys.argv[2])
root_script = '''
import subprocess
import sys
import time
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(15)"],
    start_new_session=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(child.pid, flush=True)
if sys.argv[1] == "live":
    time.sleep(15)
'''
mgr = ManagedServerProcess()
root = None
child_pid = None
try:
    root_pid = mgr.start(
        [sys.executable, "-c", root_script, sys.argv[1]],
        log_path=str(pid_file),
    )
    root = mgr._proc
    deadline = time.monotonic() + 5
    while not pid_file.read_text().strip():
        assert time.monotonic() < deadline, "root never reported its child"
        time.sleep(0.01)
    child_pid = int(pid_file.read_text().strip())
    assert os.getpgid(child_pid) == child_pid
    assert child_pid != root_pid
    assert running(child_pid)
    if root_exits:
        assert root.wait(timeout=5) == 0
        assert not group_alive(root_pid), "the old root group should be empty"
        assert mgr.stop(grace_seconds=0.1) is False, "empty old PGID is not a tree cleanup ACK"
        assert mgr._proc is root
        assert running(child_pid), "unowned detached descendants must not be signalled"
    else:
        assert root.poll() is None
        assert mgr.stop(grace_seconds=0.2) is True
        assert mgr._proc is None
        assert not running(child_pid), "live-root enumeration must reach detached descendants"
finally:
    if root is not None:
        if root.poll() is None:
            root.kill()
        root.wait(timeout=5)
    if child_pid is not None:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Children are ours, adopted here, and self-expire even on setup failures.
    deadline = time.monotonic() + 20
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        assert time.monotonic() < deadline, "test children did not exit"
        if pid == 0:
            time.sleep(0.01)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux sessions and child subreaping")
@pytest.mark.parametrize("root_state", ["exited", "live"])
def test_managed_detached_child_cleanup_evidence(tmp_path: Path, root_state: str):
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-c", _PROCESS_SCENARIO, root_state, str(tmp_path / "child.pid")],
        env=env,
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
