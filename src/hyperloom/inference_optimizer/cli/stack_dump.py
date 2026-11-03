# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kill -USR1 <pid>`` on a running ``optimize``: every thread's Python stack, written to stderr."""

from __future__ import annotations

import faulthandler
import signal
import sys


def dump_thread_stacks_on_sigusr1() -> None:
    """Make SIGUSR1 write every thread's Python stack to the process's stderr, the session's log.

    The dump is written by the signal handler itself, without the GIL, so it also reads a process frozen in native
    code that holds the GIL -- such as a Ray driver connect that never returns -- where nothing in Python runs. It
    goes to the stderr descriptor the process started with, not to whatever object ``sys.stderr`` names, which may
    have none. A handler already installed still runs after the dump; SIGUSR1's default action, ending the process,
    does not.
    """
    previous = signal.getsignal(signal.SIGUSR1)
    faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, all_threads=True, chain=previous != signal.SIG_DFL)
