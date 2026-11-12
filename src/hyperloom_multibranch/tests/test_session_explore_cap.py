# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session given an EXPLORE cap, as a hosted run's grant gives one: EXPLORE ends at the cap the validation FINAL would
run asks for when that comes before the end its own budget gives, the cap is asked again while EXPLORE runs, FINAL still
runs, and a session without one is as it was."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from hyperloom_multibranch.scheduling import budget, lifecycle
from hyperloom_multibranch.scheduling.budget import ExploreCap, FinalWork, SessionBudget
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from hyperloom_multibranch.tests.test_skeleton import SETTINGS, START, Stage, _decisions, _stage


async def _run(stage: Stage, cap: ExploreCap | None) -> lifecycle.SessionOutcome:
    """:meth:`Stage.run` with ``cap`` given to the session."""

    with stage._polls_on_virtual_time() as patch:
        patch.setattr(lifecycle, "run_image_check", stage.image_check)
        return await stage._session().run(cap)


def _phase(stage: Stage, name: str) -> dict[str, object]:
    [row] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == name]
    return row


async def test_a_cap_before_the_budget_s_own_explore_end_ends_explore_at_it_and_final_still_runs(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    budget = SessionBudget.from_hours(SETTINGS.max_hours, START)
    asked: list[FinalWork] = []

    def own(longest: float) -> float:
        """Where EXPLORE ends without a cap: one longest root test and one CLOSE grace before the deadline."""
        return budget.deadline - longest - budget.close_grace_seconds

    def cap(work: FinalWork) -> float:
        asked.append(work)
        return own(work.test_seconds) - 300.0

    outcome = await _run(stage, cap)

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    longest = asked[0].test_seconds
    assert longest > 0.0, "asked with the longest root test PRELUDE measured"
    assert asked[0].parallel == SETTINGS.caps.max_parallel_tests, "on the slots FINAL's quota gives it"
    explore = _phase(stage, "explore")
    assert explore["detail"]["explore_end"] == own(longest) - 300.0  # type: ignore[index]
    assert explore["detail"]["explore_cap"] == own(longest) - 300.0  # type: ignore[index]
    final = _phase(stage, "final")
    assert own(longest) - 300.0 <= float(str(final["at"])) < own(longest), (
        "FINAL started at the cap, not at its own end"
    )
    assert outcome.final is not None and outcome.final.chosen is not None


async def test_a_cap_after_the_budget_s_own_explore_end_and_no_cap_leave_explore_as_it_was(tmp_path: Path) -> None:
    ends = []
    for directory, cap in (("late", lambda _work: START + 10 * 3600.0), ("none", None)):
        stage = _stage(tmp_path / directory)
        outcome = await _run(stage, cap)
        assert outcome.stop_reason is SessionStopReason.TIME_EXHAUSTED
        detail = _phase(stage, "explore")["detail"]
        assert "explore_cap" not in detail or directory == "late"  # type: ignore[operator]
        ends.append(detail["explore_end"])  # type: ignore[index]

    own = SessionBudget.from_hours(SETTINGS.max_hours, START).explore_end
    assert ends[0] == ends[1], "a cap later than the session's own end changes nothing"
    assert ends[1] <= own(START, 0.0)


async def test_a_cap_already_past_ends_explore_at_once_says_so_and_final_still_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stage = _stage(tmp_path)

    with caplog.at_level(logging.WARNING, logger=budget.__name__):
        outcome = await _run(stage, lambda _work: START)

    explore, final = _phase(stage, "explore"), _phase(stage, "final")
    assert explore["detail"]["explore_end"] == explore["at"]  # type: ignore[index]
    assert float(str(final["at"])) >= float(str(explore["at"]))
    assert outcome.final is not None and outcome.final.chosen is not None
    assert any("EXPLORE gets no time" in record.getMessage() for record in caplog.records)


async def test_explore_s_end_is_asked_again_while_it_runs_and_ends_explore_once_final_s_planned_work_needs_it(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    works: list[FinalWork] = []

    def cap(work: FinalWork) -> float:
        works.append(work)
        return math.inf if len(works) <= 3 else START

    outcome = await _run(stage, cap)

    assert outcome.stop_reason is SessionStopReason.TIME_EXHAUSTED
    phases = {str(row["reason"]): row for row in _decisions(stage) if row["kind"] == "phase"}
    explore_end = float(str(phases["explore"]["detail"]["explore_end"]))
    assert len(works) == 4, "asked as EXPLORE opened, then every tick until it ended EXPLORE"
    opened, final = float(str(phases["explore"]["at"])), float(str(phases["final"]["at"]))
    assert final < explore_end, "FINAL started before the end EXPLORE opened with"
    assert final < opened + SETTINGS.judge_interval_seconds, "the end is asked every tick, not every judge interval"
    assert works[0].parallel == SETTINGS.caps.max_parallel_tests and works[0].test_seconds > 0
    assert outcome.final is not None and outcome.final.chosen is not None
