# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a command in a session of its own, so that a timeout ends everything it started.

``subprocess.run`` kills only the process it started when its timeout passes: a compiler driver's children (``hipcc``,
``clang``, ``ninja`` and their jobs) keep running, holding CPUs and the build directory, long after the build was given
up. :func:`run_in_group` starts the command as the leader of a new session, and so of a new process group, and kills
that whole group when the timeout passes or anything else ends the wait before the command did. A process that leaves
the group on its own (``setsid``) is not reached.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Sequence
from typing import Any

__all__ = ["run_in_group"]


def run_in_group(
    argv: Sequence[str], *, timeout: float | None = None, capture_output: bool = False, **popen: Any
) -> subprocess.CompletedProcess[Any]:
    """``subprocess.run(argv, timeout=..., capture_output=..., **popen)`` with the command in a session of its own.

    Raises:
        subprocess.TimeoutExpired: The command did not end within ``timeout``; its whole process group was killed and
            the command reaped first.
    """
    if capture_output:
        popen["stdout"] = popen["stderr"] = subprocess.PIPE
    with subprocess.Popen(argv, start_new_session=True, **popen) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            _kill_group(process.pid)
            process.wait()  # not communicate: a process that left the group may still hold the pipes open
            raise
    return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)


def _kill_group(leader: int) -> None:
    """Kill the process group ``leader`` leads (its id is the leader's pid, which is not reaped yet)."""
    try:
        os.killpg(leader, signal.SIGKILL)
    except ProcessLookupError:
        pass  # every process of the group has ended already
