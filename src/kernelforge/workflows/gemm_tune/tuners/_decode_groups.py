# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decode-band grouping of aiter tuning shapes: dispatch groups of a tuning CSV, and the groups already served."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

__all__ = ["DispatchGroups", "decode_served_groups", "dispatch_groups"]


def decode_served_groups(entry: dict[str, Any], decode_m: Sequence[int]) -> frozenset[tuple[int, int]]:
    """(N, K) groups that logged no decode-band miss while another group did, so their decode lookups hit."""
    from ..evidence import padded_m

    top = max(decode_m)
    groups: set[tuple[int, int]] = set()
    missed: set[tuple[int, int]] = set()
    for key in entry.get("keys") or []:
        try:
            m, n, k = int(key["M"]), int(key["N"]), int(key["K"])
        except (KeyError, TypeError, ValueError):
            continue
        groups.add((n, k))
        if padded_m(m, n) <= top:
            missed.add((n, k))
    return frozenset(groups - missed) if missed else frozenset()


class DispatchGroups(NamedTuple):
    """A tuning CSV's aiter dispatch keys in first-seen order, the M values of each, and its non-blank rows."""

    order: list[tuple[int, int, str]]
    m_values: dict[tuple[int, int, str], set[int]]
    body: list[str]


def dispatch_groups(rows: list[str], idx: dict[str, int], q_idx: int | None) -> DispatchGroups:
    """The CSV's non-blank rows and the M values of each aiter dispatch key ``(N, K, q_dtype_w)``, in first-seen order."""
    order: list[tuple[int, int, str]] = []
    m_values: dict[tuple[int, int, str], set[int]] = {}
    body: list[str] = []
    for line in rows:
        if not line.strip():
            continue
        body.append(line)
        parts = [p.strip() for p in line.split(",")]
        try:
            m, n, k = int(parts[idx["M"]]), int(parts[idx["N"]]), int(parts[idx["K"]])
        except (ValueError, IndexError):
            continue
        key = (n, k, parts[q_idx] if q_idx is not None and q_idx < len(parts) else "")
        if key not in m_values:
            m_values[key] = set()
            order.append(key)
        m_values[key].add(m)
    return DispatchGroups(order, m_values, body)
