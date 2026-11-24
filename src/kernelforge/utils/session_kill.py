# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stop processes of the agent session this command runs in, chosen by a pattern on their command line.

``pkill -f`` matches every process on the host, and a host that runs several campaigns has the same driver, profiler
and scratch-script names in each of them. This command matches only the processes that carry the calling session's
own mark, and never the shell or the agent that ran it.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from collections.abc import Sequence

from kernelforge.utils.process_reaping import own_session_processes

_SIGNALS = {"TERM": signal.SIGTERM, "INT": signal.SIGINT, "KILL": signal.SIGKILL}


def session_kill_command() -> str:
    """How a session invokes this command, on the interpreter this process runs on."""
    return f"{sys.executable} -m kernelforge.utils.session_kill"


def pattern_kill_denial(reason: str) -> str:
    """What a session is told when a command of its own would have signalled processes it did not name."""
    return (
        f"This command is not allowed: {reason}. This machine runs other campaigns with the same driver, profiler "
        "and script names, and an agent's own command line can match a pattern too, so a kill by name ends "
        "processes that are not this session's. To stop what this session started, run "
        f"`{session_kill_command()} '<pattern>'`: it matches only this session's own processes and never the shell "
        "or the agent that ran it. `kill <pid>` with a PID your own command reported is also fine. Whatever the "
        "session leaves running is cleared when it ends."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="session_kill", description=__doc__.splitlines()[0])
    parser.add_argument("pattern", help="regular expression searched for in each process's command line")
    parser.add_argument("--signal", choices=sorted(_SIGNALS), default="TERM", help="signal to send (default TERM)")
    args = parser.parse_args(argv)
    try:
        wanted = re.compile(args.pattern)
    except re.error as error:
        parser.error(f"not a regular expression: {error}")
    matched = {pid: line for pid, line in own_session_processes().items() if wanted.search(line)}
    for pid, line in sorted(matched.items()):
        try:
            os.kill(pid, _SIGNALS[args.signal])
        except OSError as error:
            print(f"pid {pid} not signalled ({error}): {line[:120]}")
            continue
        print(f"sent SIG{args.signal} to pid {pid}: {line[:120]}")
    if not matched:
        print("no process of this session matches; nothing was signalled")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
