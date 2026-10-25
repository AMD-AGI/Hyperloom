# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How long each branch's lead over integration stayed stranded in a session, for CLOSE's breakdown.

A lead is stranded while a branch's head leads integration's by more than the merge bar on the session's metric and
one of the branch's commits has already failed to merge (``CommittedCommit.first_failed_at``), the view
:mod:`~hyperloom_multibranch.notebook.rsi.stranded` gives of one moment. Over the session,
each value holds from the commit that recorded it until the next one on its branch, so :func:`stranded_spans` adds up
the minutes such a lead held, until the snapshot was taken, and finds when it peaked. In session 4aef3318 three
branches held 7.8 to 11.1% over integration that way, for up to four hours (#126).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot

_MINUTE = 60.0
_PERCENT = 100.0


@dataclass(frozen=True)
class StrandedSpan:
    """A branch whose lead over integration on ``metric`` stayed over the merge bar ``bar_pct`` for ``minutes`` while
    ``failed_commits`` of its commits had failed to merge: from ``first_at``, at most ``peak_lead_pct``, at
    ``peak_at``. Times are the session's clock, as the records' ``at``."""

    branch: str
    metric: str
    bar_pct: float
    first_at: float
    minutes: float
    peak_lead_pct: float
    peak_at: float
    failed_commits: int


@dataclass(frozen=True)
class _Values:
    """Each branch's head value on the metric, by when the commit that recorded it was made, the earliest first."""

    by_branch: Mapping[str, Sequence[tuple[float, float]]]
    root: float

    def at(self, branch: str, t: float) -> float | None:
        """The value of ``branch``'s latest commit by ``t``: before its first, integration's root value, and none for
        another branch."""

        found = self.root if branch == INTEGRATION_BRANCH else None
        for when, value in self.by_branch.get(branch, ()):
            if when > t:
                break
            found = value
        return found

    def times(self, branch: str) -> set[float]:
        return {when for when, _ in self.by_branch.get(branch, ())}


def _values(snapshot: NotebookSnapshot, metric: str, root: float) -> _Values:
    found: dict[str, list[tuple[float, float]]] = {}
    for node in sorted(snapshot.lineage.commits, key=lambda item: item.at):
        value = snapshot.value_of(node.commit_id).get(metric)
        if value:
            found.setdefault(node.branch, []).append((node.at, value))
    return _Values(found, root)


def _span(
    values: _Values, branch: str, failed: Sequence[float], end: float, bar: float
) -> tuple[float, float, float, float] | None:
    """When the stranded lead of ``branch`` first held, how many minutes it held, and its peak with when; ``None`` when
    it never held. ``failed`` are when its commits first failed to merge, the earliest first."""

    events = sorted(when for when in values.times(branch) | values.times(INTEGRATION_BRANCH) | set(failed))
    events = [when for when in events if failed[0] <= when <= end] or [failed[0]]
    first, minutes, peak, peak_at = None, 0.0, 0.0, 0.0
    for when, following in zip(events, [*events[1:], end]):
        head, base = values.at(branch, when), values.at(INTEGRATION_BRANCH, when)
        lead = (head / base - 1.0) * _PERCENT if head and base else None
        if lead is None or lead <= bar:
            continue
        first = when if first is None else first
        minutes += max(following - when, 0.0) / _MINUTE
        if lead > peak:
            peak, peak_at = lead, when
    return None if first is None else (first, minutes, peak, peak_at)


def stranded_spans(snapshot: NotebookSnapshot, metric: str) -> tuple[StrandedSpan, ...]:
    """Each branch of ``snapshot`` whose lead was stranded on ``metric`` at some time, the longest first; none without
    the baseline's value and bar on ``metric``."""

    baseline = snapshot.baseline
    bar = baseline.threshold_pct.get(metric) if baseline is not None else None
    root = baseline.values.get(metric) if baseline is not None else None
    if bar is None or not root:
        return ()
    values = _values(snapshot, metric, root)
    failed: dict[str, list[float]] = {}
    for commit in snapshot.committed:
        if commit.first_failed_at is not None:
            failed.setdefault(commit.branch, []).append(commit.first_failed_at)
    spans = []
    for branch, times in failed.items():
        found = _span(values, branch, sorted(times), snapshot.taken_at, bar)
        if found is not None:
            first, minutes, peak, peak_at = found
            spans.append(StrandedSpan(branch, metric, bar, first, minutes, peak, peak_at, len(times)))
    return tuple(sorted(spans, key=lambda span: -span.minutes))


__all__ = ["StrandedSpan", "stranded_spans"]
