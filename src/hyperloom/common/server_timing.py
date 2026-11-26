# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The last lines a server's log times its requests with, taken by the rule pack's patterns of the server's component.

A specialist confined to its own trees cannot open the log of a measured round, so whoever records a round's evidence
reads these lines from the log and hands them on. Which lines they are is the rule pack's
(``NATIVE_TIMING_LINES``, by component); a server it states none of has none.
"""

from __future__ import annotations

import re
from collections import deque
from itertools import chain
from pathlib import Path

from hyperloom.common.outside_facts import NATIVE_TIMING_LINES
from hyperloom.common.prompt_safety import flatten_for_prompt

__all__ = ["TIMING_LINES_KEPT", "TIMING_LINE_CHARS", "read_timing_lines"]

#: How many lines of each kind are kept: a server logs one of a kind for each request that ended.
TIMING_LINES_KEPT = 3
TIMING_LINE_CHARS = 300


def read_timing_lines(log: Path, component: str | None) -> tuple[str, ...]:
    """The last :data:`TIMING_LINES_KEPT` lines of each kind ``component``'s server times a request with in ``log``,
    in the order it logged them, each as one prompt line with its runs of blanks folded; none, with ``log`` left
    unopened, for a component the rule pack states no such lines of (``None``: a framework built from no checkout).

    Raises:
        OSError: ``log`` cannot be read.
    """
    patterns = [re.compile(source) for source in NATIVE_TIMING_LINES.get(component or "", ())]
    if not patterns:
        return ()
    kept: list[deque[tuple[int, str]]] = [deque(maxlen=TIMING_LINES_KEPT) for _ in patterns]
    with log.open(encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle):
            for pattern, last in zip(patterns, kept, strict=True):
                if pattern.search(line):
                    last.append((number, line))
                    break
    return tuple(flatten_for_prompt(" ".join(line.split()))[:TIMING_LINE_CHARS] for _, line in sorted(chain(*kept)))
