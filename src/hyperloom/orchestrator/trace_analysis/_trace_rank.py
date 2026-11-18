# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared trace-rank parsing."""

from __future__ import annotations

import re
from pathlib import Path

_GLOBAL_RANK_RE = re.compile(r"(?:^|[-_.])rank[-_]?(\d+)(?=[-_.]|$)", re.IGNORECASE)
# SGLang trace names encode the rank as ``TP-<n>``.
_TP_RANK_RE = re.compile(r"(?:^|[-_.])tp[-_](\d+)(?=[-_.]|$)", re.IGNORECASE)
_RANK_RES = (_GLOBAL_RANK_RE, _TP_RANK_RE)


def trace_rank(path: str | Path) -> int | None:
    """Return the global or fallback TP rank encoded by a trace path."""
    trace_path = Path(path)
    for pattern in _RANK_RES:
        if match := pattern.search(trace_path.name):
            return int(match.group(1))
    for pattern in _RANK_RES:
        if match := pattern.fullmatch(trace_path.parent.name):
            return int(match.group(1))
    return None


__all__ = ["trace_rank"]
