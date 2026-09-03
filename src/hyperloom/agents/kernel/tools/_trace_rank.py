# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared trace-rank parsing and primary-trace selection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

_GLOBAL_RANK_RE = re.compile(r"(?:^|[-_.])rank[-_]?(\d+)(?=[-_.]|$)", re.IGNORECASE)
# SGLang trace names encode the rank as ``TP-<n>``. A compact token such as
# ``tp8`` describes a tensor-parallel world size in paths and is not a rank.
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


def select_primary_trace(
    candidates: list[Path],
    *,
    file_size: Callable[[Path], int],
    preferred_rank: int = 0,
    tensor_parallel_size: int | None = None,
) -> Path | None:
    """Select one single-rank trace without substituting another known rank."""
    preferred = [path for path in candidates if trace_rank(path) == preferred_rank]
    if preferred:
        return max(preferred, key=lambda path: (file_size(path), path.name))

    non_merged = [path for path in candidates if not path.name.startswith("merged-")]
    unranked = [path for path in non_merged if trace_rank(path) is None]
    if tensor_parallel_size == 1 and unranked:
        return max(unranked, key=lambda path: (file_size(path), path.name))
    if tensor_parallel_size is None and len(non_merged) == 1 and len(unranked) == 1:
        return unranked[0]
    if tensor_parallel_size == 1 and not non_merged:
        merged = [path for path in candidates if path.name.startswith("merged-")]
        if len(merged) == 1:
            return merged[0]
    return None


__all__ = ["select_primary_trace", "trace_rank"]
