# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A CPU list as ``taskset -c`` takes it: ``0-31``, ``0-15,64-79``, ``3``."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable

__all__ = ["SERVER_CPUS_ENV", "cpu_list", "runnable_cpus"]

#: The CPUs every measured server is pinned to, apart from the ones builds compile on: a host's deployment fact, read
#: where a round starts its server.
SERVER_CPUS_ENV = "HYPERLOOM_SERVER_CPUS"
_LIST = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")


def cpu_list(text: str) -> tuple[int, ...]:
    """The CPUs ``text`` names, in its order, each once.

    Raises:
        ValueError: ``text`` is no CPU list, or one of its ranges runs backwards.
    """
    if not _LIST.fullmatch(text):
        raise ValueError(f"{text!r} is not a CPU list such as 0-31 or 0-15,64-79")
    cpus: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        low, high = int(first), int(last or first)
        if high < low:
            raise ValueError(f"{text!r}: the range {part} runs backwards")
        cpus.extend(range(low, high + 1))
    return tuple(dict.fromkeys(cpus))


def runnable_cpus(cpus: Iterable[int]) -> frozenset[int]:
    """Those of ``cpus`` a process started from this one may run on.

    The kernel is asked for each, by a thread that ends with the question: the affinity this process runs with
    does not say, because a child may be given a CPU of the host that its parent is not running on.
    """
    granted: set[int] = set()

    def ask() -> None:
        for cpu in cpus:
            try:
                os.sched_setaffinity(0, (cpu,))
            except (OSError, OverflowError):  # no such CPU here: it stays out of the answer
                continue
            granted.add(cpu)

    thread = threading.Thread(target=ask)
    thread.start()
    thread.join()
    return frozenset(granted)
