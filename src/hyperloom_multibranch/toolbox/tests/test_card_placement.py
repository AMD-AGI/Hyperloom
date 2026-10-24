# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests on a two-card pool: disjoint cards, no waiting while a card is free, and a withdrawn test frees its card.

Board #80, session MB-20261019T141640Z. FINAL's test waited 8.6 minutes on a card the toolbox had already given up on:
the judge had withdrawn the explore test that ran on it, the queue freed its slot, and the next tests started, but the
withdrawn test's sandbox kept measuring to the end because cancelling the task cannot reach the worker thread that runs
it. The signed manifests showed two sandboxes on card 0 one after the other, which read as a placement fault; the
journal shows the allocator was right and card 1 was held by the withdrawn test until 17:03.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest_asyncio

from hyperloom.contracts.measurement import Purpose, TestResult, Validity
from hyperloom.measure.hosted import STOPPED_RETURNCODE
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SUBMITTED
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.toolbox import CancelOutcome, WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import FakePlanner, Rig as ToolboxRig, ScriptedExecutor, build, req

POLICY = QueuePolicy(per_branch_k=2, global_concurrency=2)
SLOW, HANG, FAST, FOREIGN = "slow", "hang", "fast", "foreign"


class _PointPerBranch(FakePlanner):
    """Plans a request to the point file of its branch, so each test can ask the fake runner for its own behaviour."""

    def __init__(self, points: dict[str, Path]) -> None:
        super().__init__()
        self.points = points

    def plan(self, request, parent):  # type: ignore[no-untyped-def]
        super().plan(request, parent)
        return AcceptedMeasurement(None, (self.points[request.branch_id],))  # type: ignore[arg-type]


@pytest_asyncio.fixture
async def two_cards(tmp_path: Path):
    """A hosted backend over a local provider that owns two cards, and the toolbox over it, started."""
    made: list[tuple[Rig, ToolboxRig]] = []

    async def make(**points: str) -> tuple[Rig, ToolboxRig]:
        # An unmet wait for a card is short here, so a test that waits for a card it should have got fails in seconds.
        rig = make_rig(tmp_path, gpus=2, max_parallel_tests=2, runner_start_timeout_sec=2)
        files = {
            branch: point_yaml(tmp_path, f"{branch}.yaml", **_behaviour(behaviour))
            for branch, behaviour in points.items()
        }
        toolbox = build(
            tmp_path,
            executor=HostedExecutor(rig.backend, tmp_path / "session"),  # type: ignore[arg-type]
            planner=_PointPerBranch(files),
            policy=POLICY,
            workload_kind=WorkloadKind.AGENTX,
        )
        made.append((rig, toolbox))
        await toolbox.toolbox.start()
        return rig, toolbox

    yield make
    for rig, toolbox in made:  # the worker threads must end before the loop closes, so the backend closes here
        await toolbox.toolbox.close()
        await asyncio.to_thread(rig.backend.close)
        await asyncio.to_thread(rig.provider.close)


def _behaviour(name: str) -> dict[str, object]:
    return {
        SLOW: {"fake_delay": 1.5},
        HANG: {"fake_mode": "hang"},
        FAST: {"fake_delay": 0.2},
        FOREIGN: {
            "fake_foreign_cards": [0],
            "fake_delay": 0.1,
        },  # refused when the sandbox has card 0, as the runner does
    }[name]


async def _until(condition: Callable[[], bool], what: str, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.02)


def _live(rig: Rig) -> int:
    return len(rig.provider.find({}, role="measurement"))


def _result(tb: ToolboxRig, request_id: str) -> TestResult:
    entry = tb.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry and entry.outcome
    return TestResult.from_document(entry.outcome["result"])


def _cards(result: TestResult) -> tuple[int, ...]:
    assert result.sandbox is not None
    return result.sandbox.gpu_ids


@dataclass(frozen=True)
class _Window:
    start: float
    end: float


def _window(rig: Rig, request_id: str) -> _Window:
    records = [r for r in rig.backend.journal.records() if r.request_id == request_id]
    start = next(float(r.extra["submitted_unix"]) for r in records if r.state == STATE_SUBMITTED)
    end = next(float(r.extra["finished_unix"]) for r in records if r.state == STATE_FINISHED)
    return _Window(start, end)


async def test_two_concurrent_tests_take_the_two_cards_and_neither_waits_for_the_other(two_cards) -> None:
    rig, tb = await two_cards(a=SLOW, b=SLOW)
    first = await tb.toolbox.submit_test(req("a", purpose=Purpose.FINAL))
    second = await tb.toolbox.submit_test(req("b", purpose=Purpose.FINAL))
    for request_id in (first, second):
        await asyncio.wait_for(_settled(tb, request_id), 60)
    one, two = _result(tb, first), _result(tb, second)
    assert one.validity is Validity.SUCCEEDED and two.validity is Validity.SUCCEEDED
    assert {_cards(one), _cards(two)} == {(0,), (1,)}, "disjoint cards, both of them used"
    a, b = _window(rig, first), _window(rig, second)
    assert a.start < b.end and b.start < a.end, "they ran at the same time, not one behind the other"
    assert len(one.attempts) == 1 and len(two.attempts) == 1


async def test_a_withdrawn_running_test_gives_its_card_to_the_next_test_at_once(two_cards) -> None:
    rig, tb = await two_cards(a=HANG, b=HANG, c=FAST)
    held_a = await tb.toolbox.submit_test(req("a"))
    held_b = await tb.toolbox.submit_test(req("b"))
    await _until(lambda: {held_a, held_b} <= {r.request_id for r in rig.backend.journal.records()}, "both sent")
    assert rig.provider.pool.free_cards() == ()
    a_card = next(r for r in rig.backend.journal.records() if r.request_id == held_a).sandbox_id
    assert await tb.toolbox.cancel(held_a) is CancelOutcome.WITHDRAWN
    next_id = await tb.toolbox.submit_test(req("c"))
    await asyncio.wait_for(_settled(tb, next_id), 30)
    result = _result(tb, next_id)
    assert result.validity is Validity.SUCCEEDED and len(result.attempts) == 1, "it did not wait for a card"
    assert result.sandbox is not None and result.sandbox.sandbox_id != a_card
    # the withdrawn test's sandbox is gone, the test that was not withdrawn holds its own card, the new one's is free
    assert a_card not in {info.sandbox_id for info in rig.provider.find({}, role="measurement")}
    assert _live(rig) == 1 and rig.provider.pool.free_cards() == _cards(result)
    finished = rig.backend.journal.latest(held_a)
    assert finished is not None and finished.state == STATE_FINISHED
    assert finished.extra["returncode"] == STOPPED_RETURNCODE
    assert tb.toolbox._log.get(held_b).outcome is None, "the other test was not touched"  # type: ignore[union-attr]


async def test_a_stopped_branch_frees_both_cards_for_final(two_cards) -> None:
    """The session's path: the judge stops b03, its two explore tests are withdrawn, FINAL's two tests are queued."""
    rig, tb = await two_cards(b03=HANG, integration=FAST)
    explore = [await tb.toolbox.submit_test(req("b03")) for _ in range(2)]
    await _until(lambda: _live(rig) == 2 and rig.provider.pool.free_cards() == (), "both cards held")
    assert sorted(await tb.toolbox.withdraw_branch("b03")) == sorted(explore)
    final = [await tb.toolbox.submit_test(req("integration", purpose=Purpose.FINAL)) for _ in range(2)]
    for request_id in final:
        await asyncio.wait_for(_settled(tb, request_id), 30)
    results = [_result(tb, request_id) for request_id in final]
    assert all(r.validity is Validity.SUCCEEDED and len(r.attempts) == 1 for r in results), "no card was waited for"
    assert {_cards(r) for r in results} == {(0,), (1,)}
    assert _live(rig) == 0 and rig.provider.pool.free_cards() == (0, 1)


async def test_a_test_queued_behind_the_slots_starts_when_a_running_test_is_cancelled(two_cards) -> None:
    rig, tb = await two_cards(a=HANG, b=HANG, c=FAST)
    first = await tb.toolbox.submit_test(req("a"))
    await tb.toolbox.submit_test(req("b"))
    queued = await tb.toolbox.submit_test(req("c"))
    await _until(lambda: _live(rig) == 2, "two sandboxes")
    assert (await tb.toolbox.queue_state()).queued_by_branch == {"c": 1}
    await tb.toolbox.cancel(first)
    await asyncio.wait_for(_settled(tb, queued), 30)
    assert _result(tb, queued).validity is Validity.SUCCEEDED and len(_result(tb, queued).attempts) == 1


async def _settled(tb: ToolboxRig, request_id: str) -> None:
    entry = tb.toolbox._log.get(request_id)
    assert entry is not None
    while entry.outcome is None:
        await asyncio.sleep(0.02)


# -- what the toolbox tells an executor ------------------------------------------------------------------------------


async def test_only_a_test_withdrawn_while_it_runs_is_withdrawn_from_the_executor(tmp_path: Path) -> None:
    executor = ScriptedExecutor(gate=asyncio.Event())
    tb = build(tmp_path, executor=executor, policy=QueuePolicy(per_branch_k=1, global_concurrency=1))
    await tb.toolbox.start()
    running = await tb.toolbox.submit_test(req())
    queued = await tb.toolbox.submit_test(req())
    await asyncio.sleep(0.05)
    await tb.toolbox.cancel(queued)
    assert executor.withdrawn == [], "a queued test holds nothing"
    await tb.toolbox.cancel(running)
    assert executor.withdrawn == [running]
    last = await tb.toolbox.submit_test(req())
    await asyncio.sleep(0.05)
    assert last in executor.started
    await tb.toolbox.close()
    assert executor.withdrawn == [running], "closing leaves the running test for the next start to re-attach to"
