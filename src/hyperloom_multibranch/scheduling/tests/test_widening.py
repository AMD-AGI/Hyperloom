# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a branch opened now would still pay off: the least useful EXPLORE time and the toolbox's headroom."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, QueueState
from hyperloom_multibranch.blackboard.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.scheduling.branches import SPECIALIST_START_SECONDS, BranchFocus
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


def _cards(usable: int, *, leased: int = 0) -> MeasurementCapacity:
    return MeasurementCapacity(usable, usable, leased_cards=leased)


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
    assert toolbox_headroom(CAPS, _queue(running=running, queued=queued), _cards(16), 2) == headroom


def test_the_cards_a_branch_sandbox_holds_bound_the_headroom_below_the_cap() -> None:
    """A lease takes cards out of measurement, and the judge is told how many tests are left, not the cap."""

    assert toolbox_headroom(CAPS, _queue(), _cards(16, leased=8), 2) == 4
    assert toolbox_headroom(CAPS, _queue(running=2), _cards(16, leased=8), 2) == 2
    assert toolbox_headroom(CAPS, _queue(), _cards(16, leased=12), 4) == 1


def test_a_toolbox_with_no_card_pool_reads_under_the_cap_alone() -> None:
    assert toolbox_headroom(CAPS, _queue(running=1), MeasurementCapacity.unknown(), 2) == 7


class _Toolbox:
    def __init__(self, state: QueueState) -> None:
        self.state = state

    async def queue_state(self) -> QueueState:
        return self.state

    async def measurement_capacity(self) -> MeasurementCapacity:
        return _cards(16)


def test_signals_read_the_queue_the_toolbox_reports_now(tmp_path: Path) -> None:
    frontier = FrontierPolicy(METRIC, window_seconds=60.0)
    toolbox = _Toolbox(_queue(running=2, queued=1))
    roster = Roster(CAPS, tmp_path / "MB-roster.json", Clock(now=lambda: 0.0))
    widening = Widening(toolbox, roster, frontier, cards_per_test=2)

    assert asyncio.run(widening.signals()) == WideningSignals(SPECIALIST_START_SECONDS, None, 5)
    toolbox.state = _queue(running=2, cold_start=100.0)
    assert asyncio.run(widening.signals()) == WideningSignals(SPECIALIST_START_SECONDS + 200.0, 100.0, 6)


def test_the_cold_start_the_real_toolbox_measured_is_what_the_signals_carry_to_the_judge(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=ScriptedExecutor([lambda job: ok(job, cold_start_seconds=82.2)]))
    roster = Roster(CAPS, tmp_path / "MB-roster.json", Clock(now=lambda: 0.0))
    widening = Widening(rig.toolbox, roster, FrontierPolicy(METRIC, window_seconds=60.0), cards_per_test=1)

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


def _texts(focus: tuple[BranchFocus, ...]) -> tuple[str, ...]:
    return tuple(aim.text for aim in focus)


def test_a_refill_takes_the_judges_focus_and_else_the_largest_ops_of_its_start_commit() -> None:
    assert _texts(refill_focus(("attention", ""), 3, SHARES)) == ("attention", "moe_gemm", "attention")


def test_a_refill_the_judge_left_open_spreads_over_the_profile_and_wraps() -> None:
    focus = refill_focus((), 4, SHARES)
    assert focus == tuple(BranchFocus(op, op) for op in ("moe_gemm", "attention", "allreduce", "moe_gemm"))


def test_a_refill_from_a_commit_never_profiled_has_no_focus_unless_the_judge_names_one() -> None:
    assert refill_focus(("attention",), 2, None) == (BranchFocus("attention", "attention"), BranchFocus("", ""))


PROFILED = OpShares(
    "c-head",
    "job-1",
    "profile-1",
    {
        "aiter::cross_device_reduce_1stage": 18.6,
        "_act_mul_and_dynamic_fp8_group_quant_kernel.kd": 3.1,
        "aiter::add_rmsnorm_quant_kernel": 2.0,
        "aiter::dynamic_per_token_scaled_quant_kernel": 1.5,
        "add_rmsnorm": 1.0,
    },
)


@pytest.mark.parametrize(
    ("asked", "op"),
    [
        ("aiter::cross_device_reduce_1stage (allreduce, 18.6% of GPU time)", "aiter::cross_device_reduce_1stage"),
        ("_act_mul_and_dynamic_fp8_group_quant_kernel", "_act_mul_and_dynamic_fp8_group_quant_kernel.kd"),
        (
            "_act_mul_and_dynamic_fp8_group_quant_kernel.kd (fused act-mul)",
            "_act_mul_and_dynamic_fp8_group_quant_kernel.kd",
        ),
        ("fuse aiter::add_rmsnorm_quant_kernel with the quant", "aiter::add_rmsnorm_quant_kernel"),
        (
            "aiter::add_rmsnorm_quant_kernel and aiter::dynamic_per_token_scaled_quant_kernel (norm and quant fusion)",
            "aiter::add_rmsnorm_quant_kernel",
        ),
        (
            "aiter::add_rmsnorm_quant_kernel (norm): fuse with the following quant step",
            "aiter::add_rmsnorm_quant_kernel",
        ),
        ("gdr_mtp_snapshot_bf16 kernel (8.6% of GPU time)", "gdr_mtp_snapshot_bf16 kernel (8.6% of GPU time)"),
        ("aiter::cross_device_reduce_1stage_v2", "aiter::cross_device_reduce_1stage_v2"),
    ],
)
def test_a_refill_keeps_the_judges_text_and_reads_its_op_as_the_profile_op_it_names_first(asked: str, op: str) -> None:
    assert refill_focus((asked,), 1, PROFILED) == (BranchFocus(asked, op),)
