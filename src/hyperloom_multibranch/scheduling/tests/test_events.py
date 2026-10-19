# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The events the judge is shown: a merge that moved the integration head by a large gain, the bottleneck of the
integration head's profile moving, a quarantined node, and how far each branch lies behind the integration head."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    EnvironmentChange,
    OpShares,
    QuarantineEvent,
)
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome, QuarantineAction
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, RequestRejected
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.events import (
    HEAD_MOVED_DURING_PROFILE,
    MB_BIG_GAIN_BARS,
    MB_BOTTLENECK_SHIFT_PP,
    PROFILE_EXISTS,
    PROFILE_UNAVAILABLE,
    PROFILED,
    EventKind,
    IntegrationProfiler,
    SessionEvents,
    behind_integration_pct,
    bottleneck_moved,
    integration_moved,
    no_measurement_cards,
)
from hyperloom_multibranch.scheduling.integrator import MergeReport
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request
from hyperloom_multibranch.toolbox.analysis import CODE_NO_EXECUTOR, CODE_PROFILE_UNVERIFIED
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import Fate
from hyperloom_multibranch.toolbox.tools.base import JobStatus

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
            "foundation_fix": False,
        }


@pytest.mark.parametrize("gain_pct", [-1.5, 0.5], ids=["regressed", "below_the_large_gain"])
def test_a_merged_foundation_fix_moves_the_integration_head_whatever_its_gain(gain_pct: float) -> None:
    event = integration_moved(replace(_merged(gain_pct), foundation_fix=True))

    assert event is not None and event.kind is EventKind.INTEGRATION_MOVED
    assert event.reason.startswith("a foundation fix moved the integration head")
    assert (event.detail["gain_pct"], event.detail["foundation_fix"]) == (gain_pct, True)


def test_only_a_merged_report_moves_the_integration_head() -> None:
    assert integration_moved(_merged(10 * BAR, MergeOutcome.NOT_STACKABLE)) is None
    assert integration_moved(replace(_merged(0.0, MergeOutcome.NOT_STACKABLE), foundation_fix=True)) is None
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


@dataclass(frozen=True)
class _Snapshot:
    """The two views of a notebook snapshot the events read."""

    quarantined: tuple[QuarantineEvent, ...]
    environment_changes: tuple[EnvironmentChange, ...]


class _Health:
    """The notebook's snapshot as the events read it: the quarantines in force and the environment changes."""

    def __init__(self) -> None:
        self.events: list[QuarantineEvent] = []
        self.changes: list[EnvironmentChange] = []

    async def record_quarantine(self, event: QuarantineEvent) -> None:
        self.events.append(event)

    async def snapshot(self) -> Any:
        quarantined = tuple(event for event in self.events if event.action is QuarantineAction.QUARANTINE)
        return _Snapshot(quarantined, tuple(self.changes))


def _logged(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "MB-decisions.jsonl"


async def _no_wait(seconds: float) -> None:
    del seconds
    await asyncio.sleep(0)


CLOCK = Clock(now=lambda: 7.0, sleep=_no_wait)


class _Shares:
    """The notebook's op shares by commit, as the TraceLens results put them there."""

    def __init__(self, shares: Mapping[str, OpShares] | None = None) -> None:
        self.shares = dict(shares or {})
        self.reads: list[str] = []

    async def op_shares(self, commit_id: str) -> OpShares | None:
        self.reads.append(commit_id)
        return self.shares.get(commit_id)


def _events(
    health: _Health,
    log: Path,
    toolbox: FakeToolbox | None = None,
    shares: _Shares | None = None,
    cards_per_test: int = 1,
) -> SessionEvents:
    toolbox = toolbox or FakeToolbox()
    records = shares or _Shares()
    profiler = IntegrationProfiler(toolbox, SessionTests(toolbox, CLOCK), records, CLOCK)  # type: ignore[arg-type]
    return SessionEvents(health, DecisionLog(log, CLOCK), profiler, toolbox, cards_per_test)


@pytest.mark.parametrize(
    ("capacity", "stops"),
    [
        (MeasurementCapacity(8, 3), True),
        (MeasurementCapacity(8, 4), False),
        (MeasurementCapacity.unknown(), False),
    ],
    ids=["fewer_than_a_test_needs", "as_many_as_a_test_needs", "unknown"],
)
def test_a_toolbox_runs_out_of_cards_below_what_one_test_needs_and_never_when_it_does_not_know(
    capacity: MeasurementCapacity, stops: bool
) -> None:
    event = no_measurement_cards(capacity, 4)

    assert (event is not None) is stops
    if event is not None:
        assert (event.kind, event.reason) == (
            EventKind.NO_MEASUREMENT_CARDS,
            "the toolbox can measure on 3 of its 8 cards, a test needs 4",
        )
        assert event.detail == {"usable_cards": 3, "total_cards": 8, "cards_per_test": 4}


async def test_a_card_shortage_is_recorded_and_never_shown_to_the_judge(log: Path) -> None:
    toolbox = FakeToolbox()
    enough = _events(_Health(), log, toolbox, cards_per_test=8)
    short = _events(_Health(), log, toolbox, cards_per_test=9)

    assert not await enough.record_card_shortage() and _logged(log) == []
    assert await short.record_card_shortage()
    assert short.hand_over() == ()
    [row] = _logged(log)
    assert (row["kind"], row["detail"]["event"]) == ("event", "no_measurement_cards")


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


async def test_a_node_whose_environment_moved_is_shown_once_with_each_field_before_and_after(log: Path) -> None:
    health = _Health()
    events = _events(health, log)
    health.changes.append(
        EnvironmentChange("node-0", "r7", 5.0, {"power_cap_w": ("750.0", "600.0"), "image_digest": ("sha:a", "sha:b")})
    )

    assert await events.detect()
    [shown] = events.hand_over()

    assert shown.kind is EventKind.ENVIRONMENT_CHANGED
    assert shown.reason == "node node-0 moved: image_digest sha:a -> sha:b, power_cap_w 750.0 -> 600.0"
    assert shown.detail == {
        "node": "node-0",
        "request_id": "r7",
        "changed": {
            "power_cap_w": {"before": "750.0", "after": "600.0"},
            "image_digest": {"before": "sha:a", "after": "sha:b"},
        },
    }
    assert not await events.detect()
    health.changes.append(EnvironmentChange("node-0", "r9", 6.0, {"gpu_model": ("MI300X", "MI355X")}))
    assert await events.detect()
    assert [event.detail["request_id"] for event in events.hand_over()] == ["r9"]
    assert [row["detail"]["event"] for row in _logged(log)] == ["environment_changed", "environment_changed"]


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


def _op_shares(commit: str, shares: Mapping[str, float], job: str = "job-0") -> OpShares:
    return OpShares(commit, job, "r-profile", shares)


@pytest.mark.parametrize(
    ("after", "moved"),
    [
        ({"attn": 55.0, "gemm": 30.0}, False),
        ({"attn": 60.0 - MB_BOTTLENECK_SHIFT_PP - 1, "gemm": 30.0}, True),
        ({"gemm": 50.0, "attn": 40.0}, True),
    ],
    ids=["same_op_small_drop", "same_op_large_drop", "new_largest_op"],
)
def test_the_bottleneck_moves_when_the_largest_op_changes_or_its_share_falls_far(
    after: dict[str, float], moved: bool
) -> None:
    event = bottleneck_moved(_op_shares("c-old", {"attn": 60.0, "gemm": 30.0}), _op_shares("c-new", after))

    assert (event is not None) is moved
    if event is not None:
        assert event.kind is EventKind.BOTTLENECK_MOVED
        assert (event.detail["previous_largest_op"], event.detail["commit"]) == ("attn", "c-new")


def test_a_bottleneck_without_shares_to_compare_has_not_moved() -> None:
    assert bottleneck_moved(None, _op_shares("c-new", {"gemm": 50.0})) is None
    assert bottleneck_moved(_op_shares("c-old", {"gemm": 50.0}), None) is None


class _AnalysingToolbox(FakeToolbox):
    """The toolbox module's fake with an analysis route: a TraceLens job that runs until ``finish`` puts its op shares
    in the notebook; not a placeholder, the fake has no tool jobs of its own."""

    def __init__(self, shares: _Shares, **options: Any) -> None:
        super().__init__(**options)
        self._shares = shares
        self.analyses: list[tuple[str, str, dict[str, Any], str]] = []
        self.status = JobStatus.RUNNING
        self.rejection: RequestRejected | None = None

    async def submit_analysis(
        self, branch: str, tool: str, params: Mapping[str, Any], profile_request_id: str = ""
    ) -> str:
        if self.rejection is not None:
            raise self.rejection
        self.analyses.append((branch, tool, dict(params), profile_request_id))
        return f"job-{len(self.analyses)}"

    def tool_status(self, job_id: str) -> JobStatus:
        return self.status

    def finish(self, commit: str, shares: Mapping[str, float]) -> None:
        self._shares.shares[commit] = _op_shares(commit, shares, f"job-{len(self.analyses)}")
        self.status = JobStatus.SUCCEEDED


BEFORE = {"attn": 60.0, "gemm": 30.0}


def _moved_onto(head: str) -> MergeReport:
    return MergeReport("c1", "c-root", MergeOutcome.MERGED, "", "r1", head, (), MB_BIG_GAIN_BARS * BAR, BAR)


async def _until(condition: Any) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


async def test_a_large_merge_profiles_the_integration_head_and_a_moved_bottleneck_is_an_event(log: Path) -> None:
    shares = _Shares({"c-root": _op_shares("c-root", BEFORE)})
    toolbox = _AnalysingToolbox(shares)
    events = _events(_Health(), log, toolbox, shares)
    events.note_merges((_moved_onto("c-merged"),))

    assert await events.detect()
    [moved] = events.hand_over()
    await _until(lambda: toolbox.analyses)

    [(request_id, request)] = toolbox.requests.items()
    assert (request.kind, request.purpose, request.branch_id, request.parent_commit) == (
        TestKind.PROFILE,
        Purpose.PROFILE,
        INTEGRATION_BRANCH,
        "c-merged",
    )
    assert toolbox.analyses == [(INTEGRATION_BRANCH, "tracelens", {}, request_id)]
    assert toolbox.results[request_id].profile_location
    assert events.profiling and not await events.detect() and "c-merged" not in shares.reads

    toolbox.finish("c-merged", {"gemm": 55.0, "attn": 30.0})
    await _until(lambda: not events.profiling)

    assert await events.detect()
    [shifted] = events.hand_over()
    assert (shifted.kind, shifted.detail["largest_op"], shifted.detail["previous_largest_op"]) == (
        EventKind.BOTTLENECK_MOVED,
        "gemm",
        "attn",
    )
    rows = _logged(log)
    assert [(row["kind"], row["reason"]) for row in rows[1:]] == [("profile", PROFILED), ("event", shifted.reason)]
    assert rows[1]["detail"]["request_id"] == request_id and rows[1]["detail"]["analysis"] == "succeeded"


async def test_a_head_a_branch_already_profiled_is_not_profiled_again_and_is_not_an_error(log: Path) -> None:
    shares = _Shares()
    toolbox = _AnalysingToolbox(shares, autorun=False)
    branch_profile = await toolbox.submit_test(
        replace(integration_test_request(TestKind.PROFILE, Purpose.PROFILE, "c-merged"), branch_id="b01")
    )
    events = _events(_Health(), log, toolbox, shares)
    events.note_merges((_moved_onto("c-merged"),))

    await events.detect()
    await _until(lambda: not events.profiling)
    assert not await events.detect()

    assert list(toolbox.requests) == [branch_profile] and toolbox.analyses == []
    [profile] = [row for row in _logged(log) if row["kind"] == "profile"]
    assert profile["reason"] == PROFILE_EXISTS
    assert profile["detail"]["commit"] == "c-merged" and profile["detail"]["request_id"] == branch_profile
    assert profile["detail"]["code"] == "profile_exists"
    events.note_merges((_moved_onto("c-next"),))
    await events.detect()
    assert events.profiling, "a head that has no profile yet is profiled as before"
    await events.stop_profiling()


async def test_one_profile_runs_at_a_time_and_does_not_hold_up_the_tick(log: Path) -> None:
    toolbox = FakeToolbox(autorun=False)
    events = _events(_Health(), log, toolbox)
    events.note_merges((_moved_onto("c-merged"),))
    assert await events.detect()
    await _until(lambda: toolbox.requests)

    events.note_merges((_moved_onto("c-next"),))
    assert await events.detect()

    assert events.profiling and [request.parent_commit for request in toolbox.requests.values()] == ["c-merged"]
    [request_id] = toolbox.requests
    await events.stop_profiling()
    assert request_id in toolbox.withdrawn and not events.profiling


@pytest.mark.parametrize(
    "rejection",
    [RequestRejected("no tool jobs", CODE_NO_EXECUTOR), None],
    ids=["no_tool_executor", "profile_failed"],
)
async def test_a_backend_that_cannot_profile_is_tried_once_per_session(
    log: Path, rejection: RequestRejected | None
) -> None:
    shares = _Shares({"c-root": _op_shares("c-root", BEFORE)})
    toolbox = _AnalysingToolbox(shares)
    toolbox.rejection = rejection
    if rejection is None:
        toolbox.script(lambda request: request.kind is TestKind.PROFILE, [Fate.MEASUREMENT])
    events = _events(_Health(), log, toolbox, shares)
    events.note_merges((_moved_onto("c-merged"),))
    await events.detect()
    events.hand_over()
    await _until(lambda: not events.profiling)

    assert not await events.detect() and events.hand_over() == ()
    events.note_merges((_moved_onto("c-next"),))
    await events.detect()

    assert not events.profiling and len(toolbox.requests) == 1 and toolbox.analyses == []
    profiles = [row for row in _logged(log) if row["kind"] == "profile"]
    assert [row["reason"] for row in profiles] == [PROFILE_UNAVAILABLE]
    assert not [row for row in _logged(log) if row["detail"].get("event") == "bottleneck_moved"]


async def test_a_profile_that_failed_unexpectedly_raises_once_from_stop(log: Path) -> None:
    shares = _Shares()
    toolbox = _AnalysingToolbox(shares)
    toolbox.rejection = RequestRejected("backend gone", "backend_gone")
    events = _events(_Health(), log, toolbox, shares)
    events.note_merges((_moved_onto("c-merged"),))
    await events.detect()
    await _until(lambda: not events.profiling)

    with pytest.raises(RequestRejected, match="backend gone"):
        await events.stop_profiling()
    await events.stop_profiling()


async def test_a_merge_during_the_profile_is_recorded_and_the_next_move_profiles_again(log: Path) -> None:
    shares = _Shares()
    toolbox = _AnalysingToolbox(shares)
    toolbox.rejection = RequestRejected("not on the head", CODE_PROFILE_UNVERIFIED)
    events = _events(_Health(), log, toolbox, shares)
    events.note_merges((_moved_onto("c-merged"),))
    await events.detect()
    await _until(lambda: not events.profiling)

    await events.detect()
    events.note_merges((_moved_onto("c-next"),))
    await events.detect()

    await _until(lambda: len(toolbox.requests) == 2)
    [profile] = [row for row in _logged(log) if row["kind"] == "profile"]
    assert (profile["reason"], profile["detail"]["code"]) == (HEAD_MOVED_DURING_PROFILE, CODE_PROFILE_UNVERIFIED)
