# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A CPU list as ``taskset -c`` takes it: ``0-31``, ``0-15,64-79``, ``3``."""

from __future__ import annotations

import re

__all__ = ["SERVER_CPUS_ENV", "cpu_list"]

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
