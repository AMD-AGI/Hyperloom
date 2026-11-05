# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kill -USR1`` on a running ``hyperloom optimize`` writes every thread's stack to its stderr and nothing else."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time

import pytest

# An optimize run whose session parks a worker thread on stdin, so the test decides when it ends. The worker says
# ``ready`` itself: by then both threads exist.
_PARKED_OPTIMIZE = """
import asyncio, signal, sys

from hyperloom.inference_optimizer import cli

if sys.argv[1] == "with-handler":
    signal.signal(signal.SIGUSR1, lambda *_: print("the earlier handler ran", flush=True))


def parked_reading_stdin():
    print("ready", flush=True)
    return sys.stdin.readline()


async def parked_session(args):
    await asyncio.to_thread(parked_reading_stdin)
    return 0


cli._run_optimize = parked_session
sys.exit(cli.main(["optimize"]))
"""

_DEADLINE_SEC = 60.0


def _read_until(stream, needle: str, deadline: float) -> str:
    """Read ``stream`` until ``needle`` appears, failing the test at ``deadline``."""
    seen = ""
    while needle not in seen:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"{needle!r} never appeared; read so far: {seen!r}"
        ready, _, _ = select.select([stream], [], [], remaining)
        chunk = os.read(stream.fileno(), 65536).decode() if ready else ""
        assert chunk or not ready, f"the stream closed before {needle!r}; read so far: {seen!r}"
        seen += chunk
    return seen


@pytest.mark.parametrize("handler", ["none", "with-handler"])
def test_sigusr1_dumps_every_thread_and_the_run_goes_on(tmp_path, handler: str):
    proc = subprocess.Popen(
        [sys.executable, "-c", _PARKED_OPTIMIZE, handler],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
    )
    try:
        deadline = time.monotonic() + _DEADLINE_SEC
        stdout = _read_until(proc.stdout, "ready", deadline)

        proc.send_signal(signal.SIGUSR1)
        stderr = _read_until(proc.stderr, "parked_reading_stdin", deadline)
        if handler == "with-handler":
            stdout += _read_until(proc.stdout, "the earlier handler ran", deadline)
        proc.stdin.write(b"\n")
        proc.stdin.flush()
        out, err = proc.communicate(timeout=deadline - time.monotonic())
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    stderr += err.decode()

    assert proc.returncode == 0, f"SIGUSR1 ended the run; stderr: {stderr}"
    assert stderr.count("(most recent call first)") >= 2, stderr
    assert f"inference_optimizer{os.sep}cli{os.sep}__init__.py" in stderr, "the main thread's stack is missing"
    assert ("the earlier handler ran" in stdout + out.decode()) is (handler == "with-handler")
