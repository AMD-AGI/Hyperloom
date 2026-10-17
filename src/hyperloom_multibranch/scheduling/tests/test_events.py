# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The events the judge is shown: a merge that moved the integration head by a large gain, a quarantined node, and how
far each branch lies behind the integration head."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, QuarantineEvent
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome, QuarantineAction
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.events import (
    MB_BIG_GAIN_BARS,
    EventKind,
    SessionEvents,
    behind_integration_pct,
    integration_moved,
)
from hyperloom_multibranch.scheduling.integrator import MergeReport

METRIC = "output_throughput"
BAR = 2.0


def _merged(gain_pct: float, outcome: MergeOutcome = MergeOutcome.MERGED) -> MergeReport:
    merged = "c-merged" if outcome is MergeOutcome.MERGED else None
    return MergeReport("c1", "c-root", outcome, "", "r1", merged, (), gain_pct, BAR)


@pytest.mark.parametrize(
    ("bars", "moved"), [(MB_BIG_GAIN_BARS, True), (1.5, False)], ids=["two_bars", "one_and_a_half"]
)
def test_a_merge_moves_the_integration_head_from_its_gain_in_bars(bars: float, moved: bool) -> None:
    event = integration_moved(_merged(bars * BAR))

    assert (event is not None) is moved
    if event is not None:
        assert event.kind is EventKind.INTEGRATION_MOVED
        assert event.detail == {
            "candidate": "c1",
            "merged_commit": "c-merged",
            "gain_pct": bars * BAR,
            "threshold_pct": BAR,
        }


def test_only_a_merged_report_moves_the_integration_head() -> None:
    assert integration_moved(_merged(10 * BAR, MergeOutcome.NOT_STACKABLE)) is None
    assert integration_moved(MergeReport("c1", "c-root", MergeOutcome.MERGED, "", "r1", "c-merged")) is None


def _row(branch: str, value: float | None, *, running: bool = True) -> BranchProgress:
    head_value = {} if value is None else {METRIC: value}
    return BranchProgress(branch, running, "c-root", f"c-{branch}", 3, 1, 10.0, head_value)


def test_each_running_specialist_branch_is_shown_how_far_it_lies_behind_the_integration_head() -> None:
    progress = (
        _row(INTEGRATION_BRANCH, 1200.0),
        _row("b01", 1100.0),
        _row("b02", 1250.0),
        _row("b03", 900.0, running=False),
        _row("b04", None),
    )

    behind = behind_integration_pct(progress, {METRIC: 1200.0}, {METRIC: 1000.0}, METRIC)

    assert behind == {"b01": pytest.approx(10.0), "b02": pytest.approx(-5.0)}
    assert behind_integration_pct(progress, {}, {METRIC: 1000.0}, METRIC) == {}


class _Health:
    """The notebook's quarantine list: the quarantines in force."""

    def __init__(self) -> None:
        self.events: list[QuarantineEvent] = []

    async def record_quarantine(self, event: QuarantineEvent) -> None:
        self.events.append(event)

    async def quarantined(self) -> tuple[QuarantineEvent, ...]:
        return tuple(event for event in self.events if event.action is QuarantineAction.QUARANTINE)


def _logged(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "MB-decisions.jsonl"


def _events(health: _Health, log: Path) -> SessionEvents:
    return SessionEvents(health, DecisionLog(log, Clock(now=lambda: 7.0)))


async def test_a_whole_node_quarantine_is_shown_once_and_recorded(log: Path) -> None:
    health = _Health()
    events = _events(health, log)
    await health.record_quarantine(QuarantineEvent("node-0", QuarantineAction.QUARANTINE, "sandbox lost"))

    assert await events.detect()
    [shown] = events.hand_over()

    assert (shown.kind, shown.detail) == (
        EventKind.ENVIRONMENT_CHANGED,
        {"node": "node-0", "gpu_ids": [], "quarantine_reason": "sandbox lost"},
    )
    assert not await events.detect() and events.hand_over() == ()
    [row] = _logged(log)
    assert (row["kind"], row["reason"], row["detail"]["event"]) == (
        "event",
        "node node-0 quarantined",
        "environment_changed",
    )


async def test_a_big_merge_is_an_event_at_the_next_detection_and_a_small_one_is_not(log: Path) -> None:
    events = _events(_Health(), log)
    events.note_merges((_merged(1.5 * BAR), _merged(MB_BIG_GAIN_BARS * BAR)))

    assert await events.detect()
    assert [event.kind for event in events.hand_over()] == [EventKind.INTEGRATION_MOVED]
    events.note_merges(())
    assert not await events.detect()
    assert [row["detail"]["event"] for row in _logged(log)] == ["integration_moved"]


async def test_events_wait_for_the_next_judge_call_however_many_ticks_pass(log: Path) -> None:
    health = _Health()
    events = _events(health, log)
    events.note_merges((_merged(MB_BIG_GAIN_BARS * BAR),))
    assert await events.detect()
    await health.record_quarantine(QuarantineEvent("node-0", QuarantineAction.QUARANTINE, "lost", (2,)))
    assert await events.detect()

    assert [event.kind for event in events.hand_over()] == [EventKind.INTEGRATION_MOVED, EventKind.ENVIRONMENT_CHANGED]
