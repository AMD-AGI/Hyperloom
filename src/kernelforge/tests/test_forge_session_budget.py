"""One implementer session's wall-clock budget is sized from the campaign.

The turn cap never bounded time (it fired 2.2% of the time), so a session that
neither answered nor capped ran until something outside killed it. The budget
below is what the Claude backend now enforces as ``AgentRunSpec.timeout_sec``.
It is a function of the campaign because the worst runaways are early iterations:
sizing off the TOTAL budget (not what remains) caps a single session before it
can eat a whole short run, while a floor keeps even a 1h run's session long
enough to read+edit+build+bench and a ceiling keeps a long overnight campaign
admitting many sessions rather than a few marathons. These tests need no
LLM / GPU / gateway.
"""

from __future__ import annotations

import pytest

from kernelforge.cli import (
    FORGE_SESSION_BUDGET_MAX_MINUTES,
    FORGE_SESSION_BUDGET_MIN_MINUTES,
    _forge_session_timeout_sec,
)


@pytest.mark.parametrize(
    ("max_hours", "expected_sec"),
    [
        # The floor dominates every ordinary campaign (0.15 * budget < 90 min
        # until the campaign passes 10 hours).
        (1.0, FORGE_SESSION_BUDGET_MIN_MINUTES * 60),
        (8.0, FORGE_SESSION_BUDGET_MIN_MINUTES * 60),
        (10.0, FORGE_SESSION_BUDGET_MIN_MINUTES * 60),
        # Between the floor and the ceiling the fraction takes over.
        (12.0, int(round(0.15 * 12.0 * 60)) * 60),
        # A very long campaign is held at the ceiling so it still admits many
        # sessions instead of a handful of marathon ones.
        (24.0, FORGE_SESSION_BUDGET_MAX_MINUTES * 60),
        (48.0, FORGE_SESSION_BUDGET_MAX_MINUTES * 60),
    ],
)
def test_session_budget_scales_between_floor_and_ceiling(max_hours, expected_sec):
    assert _forge_session_timeout_sec(max_hours, None) == expected_sec


def test_explicit_override_takes_precedence_over_the_formula():
    # The operator's explicit value wins over the computed one, whatever the
    # campaign budget would have produced.
    assert _forge_session_timeout_sec(1.0, 999) == 999
    assert _forge_session_timeout_sec(24.0, 60) == 60


def test_floor_is_below_ceiling():
    # A degenerate ordering would make the min/max clamp collapse to a constant.
    assert FORGE_SESSION_BUDGET_MIN_MINUTES < FORGE_SESSION_BUDGET_MAX_MINUTES
