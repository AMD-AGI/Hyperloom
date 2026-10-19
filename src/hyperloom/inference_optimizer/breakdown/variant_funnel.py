# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the framework phase benched, and how each bench run ended.

Every configuration variant and source patch the phase measured is an
``attempts`` row on a ``framework_agent`` timeline event. The export writes the
count as ``outcome.variants`` and the report reads it from there, counting a
breakdown exported before the block existed by this same rule, so the two
cannot disagree. Kernel candidates are not counted: the kernel lane reports its
own funnel.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .recorder.framework_event import EVENT_TYPE as FRAMEWORK_EVENT_TYPE

__all__ = ["VARIANT_COUNTS", "variants_from_timeline"]

#: The counts an ``outcome.variants`` block carries beside ``keep_rate_pct``.
VARIANT_COUNTS = ("bench_runs", "distinct", "kept", "reverted", "failed")

#: Upper-cased, since the arms spell a verdict differently: the configuration
#: arm reports ``KEEP`` / ``REVERT`` and the source arm ``kept`` / ``reverted``.
#: ``KEEP_UNSTABLE`` is a win its confirmation round did not reproduce, so the
#: variant was not kept.
_KEPT = frozenset({"KEEP", "KEPT"})
_REVERTED = frozenset({"REVERT", "REVERTED", "KEEP_UNSTABLE"})
#: Ran without producing a measurement: the server failed or the run overran.
_FAILED = frozenset({"FAILED", "KILLED_OVERTIME"})


def variants_from_timeline(timeline: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count the framework phase's bench runs by how each ended.

    An attempt that ended any other way, such as a dispatch cancelled before it
    ran, never reached the bench and is not counted. ``distinct`` keys on the
    arm and the fingerprint, which names the change rather than the stack it was
    measured on: a change benched again on a later stack is one distinct change
    and two bench runs.

    Args:
        timeline (Iterable[Mapping[str, Any]]): The assembled V6 timeline.

    Returns:
        dict[str, Any]: ``bench_runs`` (``kept + reverted + failed``),
            ``distinct``, ``kept``, ``reverted``, ``failed`` and
            ``keep_rate_pct``, kept per bench run and ``None`` when nothing was
            benched.
    """
    ended = {"kept": 0, "reverted": 0, "failed": 0}
    changes: set[tuple[str, str]] = set()
    for attempt in _framework_attempts(timeline):
        verdict = _verdict(attempt.get("outcome"))
        if not verdict:
            continue
        ended[verdict] += 1
        change = str(attempt.get("fingerprint") or attempt.get("attempt_id") or "")
        changes.add((str(attempt.get("arm") or ""), change))
    runs = sum(ended.values())
    return {
        "bench_runs": runs,
        "distinct": len(changes),
        **ended,
        "keep_rate_pct": round(ended["kept"] / runs * 100, 2) if runs else None,
    }


def _framework_attempts(timeline: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Yield every attempt row off the timeline's ``framework_agent`` events."""
    for event in timeline:
        if not isinstance(event, Mapping) or event.get("type") != FRAMEWORK_EVENT_TYPE:
            continue
        ext = event.get("ext")
        rows = ext.get("attempts") if isinstance(ext, Mapping) else None
        for row in rows if isinstance(rows, list) else ():
            if isinstance(row, Mapping):
                yield row


def _verdict(outcome: Any) -> str:
    """The funnel bucket an attempt's outcome falls in, or ``""`` for none."""
    label = str(outcome or "").strip().upper()
    if label in _KEPT:
        return "kept"
    if label in _REVERTED:
        return "reverted"
    return "failed" if label in _FAILED else ""
