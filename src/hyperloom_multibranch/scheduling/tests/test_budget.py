# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the session budget: PRELUDE gets everything, the rest splits 85/5/2, then the CLOSE grace; a GPU-hour
budget leaves FINAL its share the same way, counted on the tests with a result.

The re-declared MB_ constants are compared with their sources by reading the source text;
those modules are never imported.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

import hyperloom
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, Withdrawal
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import FakeClock, intent, request, result
from hyperloom_multibranch.scheduling.budget import (
    MB_CLOSE_GRACE_CAP_SECONDS,
    MB_CLOSE_GRACE_FRACTION,
    MB_CLOSE_SHARE,
    MB_DEFAULT_MAX_HOURS,
    MB_EXPLORE_SHARE,
    MB_FINAL_SHARE,
    MB_UNLIMITED_HOURS,
    BudgetExhausted,
    HardStop,
    SessionBudget,
    explore_gpu_seconds,
)
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.explore import spent_gpu_seconds

_HOUR = 3600.0
_SRC = Path(hyperloom.__file__).parent


def _source(relative: str) -> str:
    return (_SRC / relative).read_text(encoding="utf-8")


def test_constants_match_the_main_flow_sources() -> None:
    parser = _source("inference_optimizer/cli/parser.py")
    machine = _source("orchestrator/phases/machine_state.py")
    shared = _source("orchestrator/state/shared_state.py")

    assert re.search(rf"^DEFAULT_MAX_HOURS = {MB_DEFAULT_MAX_HOURS}$", parser, re.M)
    assert re.search(r"^DEFAULT_LONGRUN_MAX_MINUTES: int = 14 \* 24 \* 60$", machine, re.M)
    assert MB_UNLIMITED_HOURS == 14 * 24
    shares = {name: int(round(float(pct) * 100)) for name, pct in re.findall(r"PHASE_(\w+): (0\.\d+),", machine)}
    assert shares["FRAMEWORK_AGENT"] + shares["KERNEL_AGENT"] == MB_EXPLORE_SHARE
    assert shares["SWEEP"] == MB_FINAL_SHARE
    assert shares["CLOSE"] == MB_CLOSE_SHARE
    assert f"min({MB_CLOSE_GRACE_CAP_SECONDS}, (max_minutes or 0.0) * 60.0 * {MB_CLOSE_GRACE_FRACTION})" in shared


def test_zero_hours_means_the_main_flows_unlimited_ceiling() -> None:
    assert SessionBudget.from_hours(0, 0.0).total_seconds == MB_UNLIMITED_HOURS * _HOUR


def test_prelude_gets_all_the_remaining_time() -> None:
    start = 1000.0
    budget = SessionBudget.from_hours(2.0, started_at=start)

    assert budget.deadline == start + 2 * _HOUR
    assert budget.prelude_max_hours(start + 0.5 * _HOUR) == pytest.approx(1.5)


def test_prelude_below_one_minute_is_exhausted() -> None:
    budget = SessionBudget.from_hours(1.0, started_at=0.0)

    with pytest.raises(BudgetExhausted):
        budget.prelude_max_hours(_HOUR - 30)
    with pytest.raises(BudgetExhausted):
        budget.prelude_max_hours(_HOUR + 30)


def test_after_prelude_the_remainder_splits_by_the_main_flow_shares() -> None:
    budget = SessionBudget.from_hours(10.0, started_at=0.0)
    unit = 100.0
    now = 10 * _HOUR - 92 * unit

    phases = budget.phases_after_prelude(now)

    assert phases.explore_end == pytest.approx(now + 85 * unit)
    assert phases.final_end == pytest.approx(now + 90 * unit)
    assert budget.hard_stop_at == budget.deadline + 120.0


def test_close_grace_is_two_percent_capped_at_two_minutes() -> None:
    assert SessionBudget.from_hours(2.0, 0.0).close_grace_seconds == 120.0
    assert SessionBudget.from_hours(0.5, 0.0).close_grace_seconds == pytest.approx(36.0)


def test_phases_after_the_deadline_all_collapse_onto_it() -> None:
    budget = SessionBudget.from_hours(1.0, started_at=0.0)

    phases = budget.phases_after_prelude(2 * _HOUR)

    assert phases.explore_end == phases.final_end == budget.deadline
    assert budget.remaining(2 * _HOUR) == 0.0


def test_the_hard_stop_is_brought_forward_to_one_close_grace_once() -> None:
    budget = SessionBudget.from_hours(1.0, 0.0)
    times = [600.0]
    hard_stop = HardStop(budget, Clock(now=lambda: times[0]))
    assert hard_stop.at == budget.hard_stop_at

    assert hard_stop.bring_forward() == 600.0 + budget.close_grace_seconds
    times[0] = 620.0
    assert hard_stop.bring_forward() == hard_stop.at == 600.0 + budget.close_grace_seconds


def test_the_hard_stop_is_never_brought_later() -> None:
    budget = SessionBudget.from_hours(1.0, 0.0)
    hard_stop = HardStop(budget, Clock(now=lambda: budget.deadline))

    assert hard_stop.bring_forward() == budget.hard_stop_at


def test_a_gpu_hour_budget_ends_explore_where_only_finals_share_is_left() -> None:
    assert explore_gpu_seconds(9.0) == pytest.approx(9.0 * _HOUR * 85 / 90)
    assert explore_gpu_seconds(0.0) == math.inf


async def test_the_gpu_seconds_spent_count_every_result_and_no_withdrawn_test(tmp_path: Path) -> None:
    clock = FakeClock()

    def handle(role: Role) -> LocalNotebook:
        return LocalNotebook(tmp_path, Writer(role, role.value), clock=clock)

    scheduler, toolbox = handle(Role.SCHEDULER), handle(Role.TOOLBOX)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    await toolbox.write_request(request("baseline", intent(INTEGRATION_BRANCH, "c0")))
    await toolbox.write_result(result("baseline", gpu_seconds=600.0))
    await toolbox.write_request(request("measured", intent("s1", "c0")))
    await toolbox.write_result(result("measured", gpu_seconds=90.0))
    await toolbox.write_request(request("withdrawn", intent("s1", "c0", target="kv cache")))
    await toolbox.write_withdrawal(Withdrawal("withdrawn", "branch s1 stopped"))

    assert spent_gpu_seconds(await scheduler.progress()) == 690.0
