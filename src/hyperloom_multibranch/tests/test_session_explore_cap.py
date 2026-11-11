# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session given an EXPLORE cap, as a hosted run's grant gives one: EXPLORE ends at the cap the session's longest
root test asks for when that comes before the end its own budget gives, FINAL still runs, and a session without one
is as it was."""

from __future__ import annotations

from pathlib import Path

from hyperloom_multibranch.scheduling import lifecycle
from hyperloom_multibranch.scheduling.budget import ExploreCap, SessionBudget
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
    asked: list[float] = []

    def own(longest: float) -> float:
        """Where EXPLORE ends without a cap: one longest root test and one CLOSE grace before the deadline."""
        return budget.deadline - longest - budget.close_grace_seconds

    def cap(longest: float) -> float:
        asked.append(longest)
        return own(longest) - 300.0

    outcome = await _run(stage, cap)

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    [longest] = asked
    assert longest > 0.0, "asked once, with the longest root test PRELUDE measured"
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
    for directory, cap in (("late", lambda _longest: START + 10 * 3600.0), ("none", None)):
        stage = _stage(tmp_path / directory)
        outcome = await _run(stage, cap)
        assert outcome.stop_reason is SessionStopReason.TIME_EXHAUSTED
        detail = _phase(stage, "explore")["detail"]
        assert "explore_cap" not in detail or directory == "late"  # type: ignore[operator]
        ends.append(detail["explore_end"])  # type: ignore[index]

    own = SessionBudget.from_hours(SETTINGS.max_hours, START).explore_end
    assert ends[0] == ends[1], "a cap later than the session's own end changes nothing"
    assert ends[1] <= own(START, 0.0)
