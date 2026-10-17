# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Events that change what the running branches are worth, detected on each EXPLORE tick for the judge to weigh.

The integration head moving by a large gain (``integration_moved``) leaves the branches started before it behind it,
and a measurement node quarantined (``environment_changed``) may make results on either side of it compare less well.
The scheduler records each event as a decision, shows it at the judge's next call and asks the judge at once; what to
stop and what to open is the judge's decision. A quarantine never stops the session: it only widens the nodes the
toolbox avoids, and on a single node the toolbox still measures there.

A large gain is read off the merge reports alone, against the bar each merge was judged by, so it follows the bar of
whatever metric the gate judges. Nothing here scores the frontier, which logs a round every time it is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.interfaces import NodeHealth
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, QuarantineEvent
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.integrator import MergeReport

#: A merge moves the integration head when its gain is at least this many times the bar it was judged against.
MB_BIG_GAIN_BARS = 2


class EventKind(str, Enum):
    INTEGRATION_MOVED = "integration_moved"
    ENVIRONMENT_CHANGED = "environment_changed"


@dataclass(frozen=True)
class SessionEvent:
    """One event as the decision log and the judge see it; ``detail`` holds the JSON-encodable facts behind it."""

    kind: EventKind
    reason: str
    detail: Mapping[str, object] = field(default_factory=dict)


def integration_moved(report: MergeReport) -> SessionEvent | None:
    """The event ``report`` makes when it merged a gain of at least :data:`MB_BIG_GAIN_BARS` bars; else ``None``."""

    if report.outcome is not MergeOutcome.MERGED or report.gain_pct is None or report.threshold_pct is None:
        return None
    if report.gain_pct < MB_BIG_GAIN_BARS * report.threshold_pct:
        return None
    reason = f"the integration head moved {report.gain_pct:+.2f}%, the bar was {report.threshold_pct:.2f}%"
    detail = {
        "candidate": report.candidate,
        "merged_commit": report.merged_commit,
        "gain_pct": report.gain_pct,
        "threshold_pct": report.threshold_pct,
    }
    return SessionEvent(EventKind.INTEGRATION_MOVED, reason, detail)


def environment_changed(quarantine: QuarantineEvent) -> SessionEvent:
    """The event of ``quarantine``; no ``gpu_ids`` is the whole node."""

    detail = {"node": quarantine.node, "gpu_ids": list(quarantine.gpu_ids), "quarantine_reason": quarantine.reason}
    return SessionEvent(EventKind.ENVIRONMENT_CHANGED, f"node {quarantine.node} quarantined", detail)


def behind_integration_pct(
    progress: Iterable[BranchProgress],
    integration_value: Mapping[str, float],
    root_value: Mapping[str, float],
    metric: str,
) -> dict[str, float]:
    """How far each running specialist branch's recorded head value lies below the integration head's, in percent of
    the root's value; a branch whose head has no value in ``metric`` is not in it, and neither is any when the
    integration head or the root has none."""

    lead, root = integration_value.get(metric), root_value.get(metric)
    if lead is None or not root:
        return {}
    return {
        row.branch: (lead - row.head_value[metric]) / root * 100.0
        for row in progress
        if row.running and row.branch != INTEGRATION_BRANCH and metric in row.head_value
    }


def _slot(quarantine: QuarantineEvent) -> tuple[str, tuple[int, ...]]:
    return quarantine.node, quarantine.gpu_ids


class SessionEvents:
    """Detects the events of one EXPLORE and holds them until the judge's next call.

    The integrator's task hands over each round's merge reports with :meth:`note_merges`; :meth:`detect` turns them,
    and the quarantines put in force since the last tick, into events. A quarantine lifted and put in force again is a
    new event.
    """

    def __init__(self, health: NodeHealth, decisions: DecisionLog) -> None:
        self._health = health
        self._decisions = decisions
        self._merges: list[MergeReport] = []
        self._quarantined: frozenset[tuple[str, tuple[int, ...]]] = frozenset()
        self._unshown: list[SessionEvent] = []

    def note_merges(self, reports: Iterable[MergeReport]) -> None:
        self._merges.extend(reports)

    async def detect(self) -> bool:
        """Record the events since the last tick as decisions and hold them for the judge; true when there was one."""

        moved = [event for event in map(integration_moved, self._merges) if event is not None]
        self._merges.clear()
        found = [*moved, *await self._newly_quarantined()]
        for event in found:
            detail = {"event": event.kind.value, **event.detail}
            self._decisions.record(Decision(DecisionKind.EVENT, event.reason, detail=detail))
        self._unshown.extend(found)
        return bool(found)

    def hand_over(self) -> tuple[SessionEvent, ...]:
        """The events detected since the last hand-over, for the judge call about to be made."""

        events, self._unshown = tuple(self._unshown), []
        return events

    async def _newly_quarantined(self) -> list[SessionEvent]:
        current = await self._health.quarantined()
        new = [quarantine for quarantine in current if _slot(quarantine) not in self._quarantined]
        self._quarantined = frozenset(map(_slot, current))
        return [environment_changed(quarantine) for quarantine in new]


__all__ = [
    "MB_BIG_GAIN_BARS",
    "EventKind",
    "SessionEvent",
    "SessionEvents",
    "behind_integration_pct",
    "environment_changed",
    "integration_moved",
]
