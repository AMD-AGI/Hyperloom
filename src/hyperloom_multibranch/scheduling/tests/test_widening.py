# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a branch opened now would still pay off: the least useful EXPLORE time and the toolbox's headroom."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import OpShares
from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.scheduling.branches import SPECIALIST_START_SECONDS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.widening import (
    MB_MIN_USEFUL_COLD_STARTS,
    Widening,
    WideningSignals,
    min_useful_seconds,
    refill_focus,
    toolbox_headroom,
)
from hyperloom_multibranch.toolbox.tests.helpers import ScriptedExecutor, build, ok, req

METRIC = "output_throughput"
THREE_HOURS = 3 * 3600.0
CAPS = RosterCaps(max_branches=4, max_k_per_branch=2, max_parallel_tests=8)


def _queue(*, running: int = 0, queued: int = 0, cold_start: float | None = None) -> QueueState:
    return QueueState(
        queued_by_priority={0: 0, 1: 0, 2: queued, 3: 0, 4: 0},
        running_by_branch={"b01": running} if running else {},
        cold_start_seconds=cold_start,
    )


def test_the_trend_window_bounds_the_useful_time_when_it_is_the_longer() -> None:
    frontier = FrontierPolicy.for_budget(METRIC, THREE_HOURS)

    assert min_useful_seconds(frontier, _queue(cold_start=60.0)) == frontier.window_seconds == 1350.0


def test_the_start_up_and_cold_starts_bound_it_when_they_take_longer() -> None:
    frontier = FrontierPolicy.for_budget(METRIC, 3600.0)

    useful = min_useful_seconds(frontier, _queue(cold_start=600.0))

    assert useful == SPECIALIST_START_SECONDS + MB_MIN_USEFUL_COLD_STARTS * 600.0 > frontier.window_seconds


def test_before_the_first_cold_start_the_cold_starts_are_left_out() -> None:
    short_window = FrontierPolicy(METRIC, window_seconds=60.0)

    assert min_useful_seconds(short_window, _queue()) == SPECIALIST_START_SECONDS


@pytest.mark.parametrize(("running", "queued", "headroom"), [(0, 0, 8), (5, 1, 2), (8, 3, -3)])
def test_headroom_is_the_global_cap_less_what_runs_and_waits(running: int, queued: int, headroom: int) -> None:
    assert toolbox_headroom(CAPS, _queue(running=running, queued=queued)) == headroom


class _Toolbox:
    def __init__(self, state: QueueState) -> None:
        self.state = state

    async def queue_state(self) -> QueueState:
        return self.state


def test_signals_read_the_queue_the_toolbox_reports_now(tmp_path: Path) -> None:
    frontier = FrontierPolicy(METRIC, window_seconds=60.0)
    toolbox = _Toolbox(_queue(running=2, queued=1))
    roster = Roster(CAPS, tmp_path / "MB-roster.json", Clock(now=lambda: 0.0))
    widening = Widening(toolbox, roster, frontier)

    assert asyncio.run(widening.signals()) == WideningSignals(SPECIALIST_START_SECONDS, None, 5)
    toolbox.state = _queue(running=2, cold_start=100.0)
    assert asyncio.run(widening.signals()) == WideningSignals(SPECIALIST_START_SECONDS + 200.0, 100.0, 6)


def test_the_cold_start_the_real_toolbox_measured_is_what_the_signals_carry_to_the_judge(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=ScriptedExecutor([lambda job: ok(job, cold_start_seconds=82.2)]))
    roster = Roster(CAPS, tmp_path / "MB-roster.json", Clock(now=lambda: 0.0))
    widening = Widening(rig.toolbox, roster, FrontierPolicy(METRIC, window_seconds=60.0))

    async def signals_after_a_test() -> tuple[WideningSignals, WideningSignals]:
        before = await widening.signals()
        await rig.toolbox.submit_test(req())
        await rig.toolbox.drain()
        return before, await widening.signals()

    before, after = asyncio.run(signals_after_a_test())

    assert (before.cold_start_seconds, before.min_useful_seconds) == (None, SPECIALIST_START_SECONDS)
    assert after.cold_start_seconds == 82.2
    assert after.min_useful_seconds == SPECIALIST_START_SECONDS + MB_MIN_USEFUL_COLD_STARTS * 82.2


SHARES = OpShares("c-head", "job-1", "profile-1", {"attention": 22.0, "moe_gemm": 41.5, "allreduce": 9.0})


def test_a_refill_takes_the_judges_focus_and_else_the_largest_ops_of_its_start_commit() -> None:
    assert refill_focus(("attention", ""), 3, SHARES) == ("attention", "moe_gemm", "attention")


def test_a_refill_the_judge_left_open_spreads_over_the_profile_and_wraps() -> None:
    assert refill_focus((), 4, SHARES) == ("moe_gemm", "attention", "allreduce", "moe_gemm")


def test_a_refill_from_a_commit_never_profiled_has_no_focus_unless_the_judge_names_one() -> None:
    assert refill_focus(("attention",), 2, None) == ("attention", "")
