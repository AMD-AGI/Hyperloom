# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Events that change what the running branches are worth, detected on each EXPLORE tick for the judge to weigh.

The integration head moving by a large gain or by a foundation fix (``integration_moved``) leaves the branches started
before it behind it, its bottleneck moving (``bottleneck_moved``) leaves the branches attacking the old one with less to
gain, and a measurement node quarantined or a node's measurement environment moving between two of its results
(``environment_changed``: its image, GPU model, power cap or partition mode, as the notebook reads them off the
results' sandbox facts) may make results on either side of it compare less well. The scheduler records each event as a decision, shows it at the judge's next call and asks the judge at once;
what to stop and what to open is the judge's decision. A quarantine never stops the session: it only widens the nodes
the toolbox avoids, and on a single node the toolbox still measures there. What does stop it is the toolbox having
fewer usable cards than one test needs (``no_measurement_cards``): cards it lost for good are never leased again, so
no test, FINAL's included, could run. A toolbox that does not know its cards never stops the session.

A large gain is read off the merge reports alone, against the bar each merge was judged by, so it follows the bar of
whatever metric the gate judges. A merged foundation fix moves the head whatever its gain: it changes what every
branch builds on. Nothing here scores the frontier, which logs a round every time it is read.

After the integration head moved, :class:`IntegrationProfiler` profiles it, one profile at a time, in a task of its own:
a PROFILE test of the head, then a TraceLens analysis of that test's trace, whose op shares the bottleneck is read from.
Both wait for the measurement to end, so the tick only starts the task and looks whether it is done. The PROFILE takes
one of the integration branch's K slots. A backend that cannot profile, by refusing the PROFILE, failing it or running
no analysis, is not asked again in this session: on a backend that refuses after its sandbox is built, every try costs
a cold start and holds the integration branch's slot. The toolbox's own names are imported where the profile uses
them: the toolbox loads the main flow's runtime, which importing the session must not.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from hyperloom.contracts.measurement import Purpose, TestKind, Validity
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch, SnapshotQuery
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    EnvironmentChange,
    OpShares,
    QuarantineEvent,
)
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, RequestRejected, Toolbox
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.integrator import MergeReport
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: A merge moves the integration head when its gain is at least this many times the bar it was judged against.
MB_BIG_GAIN_BARS = 2
#: The bottleneck moves when the largest op changes, or its share falls by more than this many percentage points.
MB_BOTTLENECK_SHIFT_PP = 10
#: Clock seconds between two looks at the analysis of a profile.
ANALYSIS_POLL_SECONDS = 5.0
PROFILE_UNAVAILABLE = "profile not available on this backend"
HEAD_MOVED_DURING_PROFILE = "integration head moved during profiling"
PROFILED = "integration head profiled"
#: The PROFILE test ran, and its analysis left no op shares of that commit behind: nothing a refill can be focused on
#: and nothing to compare a bottleneck against, so it is not a profile the head has.
PROFILE_NOT_ANALYSED = "integration head profiled, its trace not analysed"
PROFILE_EXISTS = "integration head already profiled"
#: The toolbox's refusal of a test kind its backend does not run, a PROFILE on a backend that cannot profile.
UNSUPPORTED_KIND = "unsupported_kind"
#: The toolbox's refusal of a second profile of a commit and recipe that already has a live one.
PROFILE_EXISTS_CODE = "profile_exists"


class EventKind(str, Enum):
    INTEGRATION_MOVED = "integration_moved"
    BOTTLENECK_MOVED = "bottleneck_moved"
    ENVIRONMENT_CHANGED = "environment_changed"
    NO_MEASUREMENT_CARDS = "no_measurement_cards"


@dataclass(frozen=True)
class SessionEvent:
    """One event as the decision log and the judge see it; ``detail`` holds the JSON-encodable facts behind it."""

    kind: EventKind
    reason: str
    detail: Mapping[str, object] = field(default_factory=dict)


def integration_moved(report: MergeReport) -> SessionEvent | None:
    """The event ``report`` makes when it merged a foundation fix or a gain of at least :data:`MB_BIG_GAIN_BARS`
    bars; else ``None``."""

    if report.outcome is not MergeOutcome.MERGED or report.gain_pct is None or report.threshold_pct is None:
        return None
    if not report.foundation_fix and report.gain_pct < MB_BIG_GAIN_BARS * report.threshold_pct:
        return None
    moved_by = "a foundation fix moved the integration head" if report.foundation_fix else "the integration head moved"
    reason = f"{moved_by} {report.gain_pct:+.2f}%, the bar was {report.threshold_pct:.2f}%"
    detail = {
        "candidate": report.candidate,
        "merged_commit": report.merged_commit,
        "gain_pct": report.gain_pct,
        "threshold_pct": report.threshold_pct,
        "foundation_fix": report.foundation_fix,
    }
    return SessionEvent(EventKind.INTEGRATION_MOVED, reason, detail)


def bottleneck_moved(before: OpShares | None, after: OpShares | None) -> SessionEvent | None:
    """The event of the integration head's op shares going from ``before`` to ``after``: its largest op changed, or
    that op's share fell by more than :data:`MB_BOTTLENECK_SHIFT_PP` points; ``None`` without both to compare."""

    if before is None or after is None or not before.shares or not after.shares:
        return None
    [(old_op, old_share)], [(new_op, new_share)] = before.largest(1), after.largest(1)
    if new_op == old_op and old_share - after.shares.get(old_op, 0.0) <= MB_BOTTLENECK_SHIFT_PP:
        return None
    reason = f"the bottleneck moved from {old_op} at {old_share:.1f}% to {new_op} at {new_share:.1f}%"
    detail = {
        "commit": after.commit_id,
        "largest_op": new_op,
        "share_pct": new_share,
        "previous_commit": before.commit_id,
        "previous_largest_op": old_op,
        "previous_share_pct": old_share,
    }
    return SessionEvent(EventKind.BOTTLENECK_MOVED, reason, detail)


def environment_changed(quarantine: QuarantineEvent) -> SessionEvent:
    """The event of ``quarantine``; no ``gpu_ids`` is the whole node."""

    detail = {"node": quarantine.node, "gpu_ids": list(quarantine.gpu_ids), "quarantine_reason": quarantine.reason}
    return SessionEvent(EventKind.ENVIRONMENT_CHANGED, f"node {quarantine.node} quarantined", detail)


def environment_moved(change: EnvironmentChange) -> SessionEvent:
    """The event of ``change``: each field that moved on the node, from its last known value to its value under the
    result that moved it."""

    moved = ", ".join(f"{name} {before} -> {after}" for name, (before, after) in sorted(change.changed.items()))
    detail = {
        "node": change.node,
        "request_id": change.request_id,
        "changed": {name: {"before": before, "after": after} for name, (before, after) in change.changed.items()},
    }
    return SessionEvent(EventKind.ENVIRONMENT_CHANGED, f"node {change.node} moved: {moved}", detail)


def no_measurement_cards(capacity: MeasurementCapacity, cards_per_test: int) -> SessionEvent | None:
    """The event of ``capacity`` holding fewer usable cards than the ``cards_per_test`` one test needs; ``None`` while
    a test fits, or when the toolbox does not know its cards."""

    if not capacity.known or capacity.usable_cards >= cards_per_test:
        return None
    reason = (
        f"the toolbox can measure on {capacity.usable_cards} of its {capacity.total_cards} cards, "
        f"a test needs {cards_per_test}"
    )
    detail = {
        "usable_cards": capacity.usable_cards,
        "total_cards": capacity.total_cards,
        "cards_per_test": cards_per_test,
    }
    return SessionEvent(EventKind.NO_MEASUREMENT_CARDS, reason, detail)


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


@dataclass(frozen=True)
class ProfileOutcome:
    """How one profile of the integration head ended: the reason and facts of its decision, the event it found, and
    whether the backend can profile at all."""

    reason: str
    detail: Mapping[str, object]
    event: SessionEvent | None = None
    available: bool = True


class IntegrationProfiler:
    """Profiles the integration head after it moved, at most one profile at a time, in a task of its own.

    The profile is a PROFILE test of the head through ``tests``, at the default priority of its purpose, then a
    TraceLens analysis of that test, whose op shares the ``records`` hold once the analysis has ended. Its bottleneck is
    compared with the shares of the previous profile, or else with those of the head the move started from. Once a
    profile shows the backend cannot profile, no profile starts again.
    """

    def __init__(self, toolbox: RealToolbox, tests: SessionTests, records: RecordSearch, clock: Clock) -> None:
        self._toolbox = toolbox
        self._tests = tests
        self._records = records
        self._clock = clock
        self._task: asyncio.Future[ProfileOutcome] | None = None
        self._waiting: tuple[str, str] | None = None
        self._shares: OpShares | None = None
        self._available = True

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, head: str, moved_from: str) -> None:
        """Start profiling ``head``, which the integration head moved to from ``moved_from``, unless the backend
        cannot profile.

        A head that moves while a profile runs waits for that profile to end rather than going unprofiled: the
        refills opened on it are focused on its own shares, and only the latest head is worth waiting for.
        """

        if not self._available:
            return
        if self.running:
            self._waiting = (head, moved_from)
        else:
            self._task = asyncio.ensure_future(self._profile(head, moved_from))

    def take_outcome(self) -> ProfileOutcome | None:
        """The outcome of the profile that ended since the last look, ``None`` while it runs or when none ran; a head
        that waited for it is started in its place."""

        if self._task is None or not self._task.done():
            return None
        task, self._task = self._task, None
        outcome = task.result()
        self._available = self._available and outcome.available
        waiting, self._waiting = self._waiting, None
        if waiting is not None:
            self.start(*waiting)
        return outcome

    async def stop(self) -> None:
        """Cancel the running profile, which takes its PROFILE test back; its analysis goes with the integration
        branch's withdrawal."""

        if self._task is None:
            return
        task, self._task = self._task, None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _profile(self, head: str, moved_from: str) -> ProfileOutcome:
        from hyperloom_multibranch.toolbox.analysis import (
            CODE_NO_EXECUTOR,
            CODE_PROFILE_UNVERIFIED,
            CODE_PROFILE_WITHOUT_TRACE,
            CODE_TOOL_UNAVAILABLE,
        )

        before = self._shares or await self._records.op_shares(moved_from)
        request = integration_test_request(TestKind.PROFILE, Purpose.PROFILE, head)
        try:
            return await self._tests.judged(request, lambda request_id: self._analysed(head, request_id, before))
        except RequestRejected as rejected:
            detail = {"commit": head, "code": rejected.code, "rejection": rejected.reason}
            if rejected.code == CODE_PROFILE_UNVERIFIED:
                return ProfileOutcome(HEAD_MOVED_DURING_PROFILE, detail)
            if rejected.code == PROFILE_EXISTS_CODE:  # the head was profiled already, maybe by a specialist's branch
                return ProfileOutcome(PROFILE_EXISTS, {**detail, "request_id": rejected.existing_request_id})
            unable = (UNSUPPORTED_KIND, CODE_PROFILE_WITHOUT_TRACE, CODE_NO_EXECUTOR, CODE_TOOL_UNAVAILABLE)
            if rejected.code in unable:
                return ProfileOutcome(PROFILE_UNAVAILABLE, detail, available=False)
            raise

    async def _analysed(self, head: str, request_id: str, before: OpShares | None) -> ProfileOutcome:
        """Analyse the trace of the PROFILE ``request_id`` once it ended; the toolbox reads the trace off that test's
        own record."""

        from hyperloom_multibranch.toolbox.tools.base import JobStatus
        from hyperloom_multibranch.toolbox.tools.roofline import TRACELENS_TOOL

        result = await self._toolbox.outcome(request_id)
        detail: dict[str, object] = {"commit": head, "request_id": request_id}
        if result.validity is not Validity.SUCCEEDED:
            failed = {**detail, "validity": result.validity.value, "failure_reason": result.failure_reason}
            return ProfileOutcome(PROFILE_UNAVAILABLE, failed, available=False)
        job = await self._toolbox.submit_analysis(INTEGRATION_BRANCH, TRACELENS_TOOL, {}, request_id)
        while self._toolbox.tool_status(job) is JobStatus.RUNNING:
            await self._clock.sleep(ANALYSIS_POLL_SECONDS)
        # A job's status turns final just before its result is pushed to the notebook, so the shares are read a poll
        # later, and only when they are this job's.
        await self._clock.sleep(ANALYSIS_POLL_SECONDS)
        found = await self._records.op_shares(head)
        shares = found if found is not None and found.job_id == job else None
        detail.update(job_id=job, analysis=self._toolbox.tool_status(job).value)
        if shares is None:
            return ProfileOutcome(PROFILE_NOT_ANALYSED, detail)
        self._shares = shares
        detail.update(largest_op=dict(shares.largest(1)))
        return ProfileOutcome(PROFILED, detail, bottleneck_moved(before, shares))


def _slot(quarantine: QuarantineEvent) -> tuple[str, tuple[int, ...]]:
    return quarantine.node, quarantine.gpu_ids


class SessionEvents:
    """Detects the events of one EXPLORE and holds them until the judge's next call.

    The integrator's task hands over each round's merge reports with :meth:`note_merges`; :meth:`detect` turns them,
    the quarantines put in force and the environment changes the ``notebook`` shows since the last tick and the
    profile ``profiler`` finished, into events, and has the head a large merge moved to profiled. A quarantine lifted
    and put in force again is a new event; an environment change is one event, however many ticks see it.
    :meth:`record_card_shortage` reads the cards ``toolbox`` can still measure on against the ``cards_per_test`` one
    test needs.
    """

    def __init__(
        self,
        notebook: SnapshotQuery,
        decisions: DecisionLog,
        profiler: IntegrationProfiler,
        toolbox: Toolbox,
        cards_per_test: int,
    ) -> None:
        self._notebook = notebook
        self._decisions = decisions
        self._profiler = profiler
        self._toolbox = toolbox
        self._cards_per_test = cards_per_test
        self._merges: list[MergeReport] = []
        self._quarantined: frozenset[tuple[str, tuple[int, ...]]] = frozenset()
        self._changes_seen: set[tuple[str, str]] = set()
        self._unshown: list[SessionEvent] = []

    def note_merges(self, reports: Iterable[MergeReport]) -> None:
        self._merges.extend(reports)

    @property
    def profiling(self) -> bool:
        """Whether a profile of the integration head runs, holding one of the integration branch's K slots."""

        return self._profiler.running

    async def detect(self) -> bool:
        """Record the events since the last tick as decisions and hold them for the judge; true when there was one.

        A large merge has the head the last merge of the tick moved to profiled, when no profile runs."""

        found = self._profiled()
        moved = [event for event in map(integration_moved, self._merges) if event is not None]
        merged = [report for report in self._merges if report.merged_commit]
        if moved:
            self._profiler.start(str(merged[-1].merged_commit), merged[0].onto)
        self._merges.clear()
        found = [*moved, *await self._environment(), *found]
        for event in found:
            self._record(event)
        self._unshown.extend(found)
        return bool(found)

    async def record_card_shortage(self) -> bool:
        """Whether the toolbox has fewer usable cards than one test needs, recording that as a decision when it has;
        it is never shown to the judge, because the session stops."""

        event = no_measurement_cards(await self._toolbox.measurement_capacity(), self._cards_per_test)
        if event is not None:
            self._record(event)
        return event is not None

    def _record(self, event: SessionEvent) -> None:
        detail = {"event": event.kind.value, **event.detail}
        self._decisions.record(Decision(DecisionKind.EVENT, event.reason, detail=detail))

    def hand_over(self) -> tuple[SessionEvent, ...]:
        """The events detected since the last hand-over, for the judge call about to be made."""

        events, self._unshown = tuple(self._unshown), []
        return events

    async def stop_profiling(self) -> None:
        await self._profiler.stop()

    def _profiled(self) -> list[SessionEvent]:
        outcome = self._profiler.take_outcome()
        if outcome is None:
            return []
        self._decisions.record(Decision(DecisionKind.PROFILE, outcome.reason, detail=outcome.detail))
        return [] if outcome.event is None else [outcome.event]

    async def _environment(self) -> list[SessionEvent]:
        """The quarantines put in force and the environment changes recorded since the last look, from one read."""

        snapshot = await self._notebook.snapshot()
        current = snapshot.quarantined
        new = [quarantine for quarantine in current if _slot(quarantine) not in self._quarantined]
        self._quarantined = frozenset(map(_slot, current))
        changes = [
            change
            for change in snapshot.environment_changes
            if (change.node, change.request_id) not in self._changes_seen
        ]
        self._changes_seen.update((change.node, change.request_id) for change in changes)
        return [*map(environment_changed, new), *map(environment_moved, changes)]


__all__ = [
    "ANALYSIS_POLL_SECONDS",
    "HEAD_MOVED_DURING_PROFILE",
    "MB_BIG_GAIN_BARS",
    "MB_BOTTLENECK_SHIFT_PP",
    "PROFILED",
    "PROFILE_NOT_ANALYSED",
    "PROFILE_EXISTS",
    "PROFILE_EXISTS_CODE",
    "PROFILE_UNAVAILABLE",
    "UNSUPPORTED_KIND",
    "EventKind",
    "IntegrationProfiler",
    "ProfileOutcome",
    "SessionEvent",
    "SessionEvents",
    "behind_integration_pct",
    "bottleneck_moved",
    "environment_changed",
    "environment_moved",
    "integration_moved",
    "no_measurement_cards",
]
