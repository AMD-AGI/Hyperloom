# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the framework phase benched, and how each bench run ended.

Every configuration variant and source patch the phase measured is an
``attempts`` row on a ``framework_agent`` timeline event. The export writes the
count as ``outcome.variants`` and the report reads it from there, counting a
breakdown exported before the block existed by this same rule, so the two
cannot disagree. Kernel candidates are not counted: the kernel lane reports its
own funnel.

A row is read by the facts its recorder settled on it rather than by its
``outcome``, which each arm's executor spells in its own vocabulary: ``benched``
says whether the attempt reached the bench, ``adopted`` is the adoption verdict,
``failure.attribution`` is set only for an outcome that produced no
measurement, and ``measurement.after_tput`` is what the bench read.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .recorder.framework_event import EVENT_TYPE as FRAMEWORK_EVENT_TYPE

__all__ = ["VARIANT_COUNTS", "variants_from_timeline"]

#: The counts an ``outcome.variants`` block carries beside ``keep_rate_pct``.
VARIANT_COUNTS = ("bench_runs", "distinct", "kept", "reverted", "failed")


def variants_from_timeline(timeline: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count the framework phase's bench runs by how each ended.

    ``kept`` is a run adopted into the stack and ``reverted`` one measured and
    not adopted: a run that lost, a KEEP the adoption rule refused, or a patch
    left dormant behind its switches. ``failed`` is a run that reached the bench
    and produced no measurement, which is also what a failure the recorder
    attributed reads as when it carries a throughput. An attempt that never
    reached the bench, such as a patch that did not apply, is not counted; a row
    recorded before ``benched`` existed is read by the other facts alone.
    ``distinct`` keys on the arm and the fingerprint, which names the change
    rather than the stack it was measured on: a change benched again on a later
    stack is one distinct change and two bench runs.

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
        verdict = _verdict(attempt)
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


def _verdict(attempt: Mapping[str, Any]) -> str:
    """The bucket an attempt falls in, or ``""`` for one that never reached the bench.

    The attribution is read before the throughput: a failed run can still carry
    the throughput of a measurement it was not judged on.
    """
    benched = attempt.get("benched")
    if benched is False:
        return ""
    if attempt.get("adopted") is True:
        return "kept"
    attributed = bool(_settled(attempt, "failure", "attribution"))
    if not attributed and isinstance(_settled(attempt, "measurement", "after_tput"), (int, float)):
        return "reverted"
    return "failed" if attributed or benched is True else ""


def _settled(attempt: Mapping[str, Any], block: str, field: str) -> Any:
    """One field of a block the recorder writes on the row, or ``None`` when the block is absent."""
    value = attempt.get(block)
    return value.get(field) if isinstance(value, Mapping) else None
