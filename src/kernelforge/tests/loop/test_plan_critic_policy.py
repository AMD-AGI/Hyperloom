# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for choosing which rounds' plans go to the plan critic."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kernelforge.loop.iteration.round_admission import RoundAdmissionMixin
from kernelforge.loop.plan_critic_policy import (
    PLAN_CRITIC_ENV,
    PlanCriticMode,
    PlanCriticSetting,
    PlanReview,
    decide_plan_review,
    review_seconds,
)
from kernelforge.loop.run_state import BestRecord, RoundCost, RunState
from kernelforge.loop.search_policy import SEARCH_MODE_DIVERSIFY, SEARCH_MODE_EXPLOIT
from kernelforge.orchestrator.contracts import CaseEvidence, OrchestrationContext

# The setting


@pytest.mark.parametrize(
    ("value", "mode"),
    [
        (None, PlanCriticMode.ALWAYS),
        ("", PlanCriticMode.ALWAYS),
        ("always", PlanCriticMode.ALWAYS),
        (" after-blocked-round ", PlanCriticMode.AFTER_BLOCKED_ROUND),
        ("off", PlanCriticMode.OFF),
    ],
)
def test_the_mode_is_read_from_the_environment(value, mode):
    environ = {} if value is None else {PLAN_CRITIC_ENV: value}

    assert PlanCriticSetting.from_env(long_horizon=True, environ=environ).mode is mode


def test_an_unknown_mode_refuses_to_start_naming_the_choices():
    with pytest.raises(
        ValueError, match="FORGE_PLAN_CRITIC='sometimes' is not one of: always, after-blocked-round, off"
    ):
        PlanCriticSetting.from_env(long_horizon=True, environ={PLAN_CRITIC_ENV: "sometimes"})


@pytest.mark.parametrize(
    ("mode", "long_horizon", "enabled"),
    [
        (PlanCriticMode.ALWAYS, True, True),
        (PlanCriticMode.AFTER_BLOCKED_ROUND, True, True),
        (PlanCriticMode.OFF, True, False),
        (PlanCriticMode.ALWAYS, False, False),
        (PlanCriticMode.AFTER_BLOCKED_ROUND, False, False),
    ],
)
def test_a_critic_is_built_only_for_a_long_campaign_that_did_not_turn_it_off(mode, long_horizon, enabled):
    setting = PlanCriticSetting(mode=mode, long_horizon=long_horizon)

    assert setting.enabled is enabled
    # A campaign without a critic has no round to decide about.
    assert setting.loop_mode is (mode if enabled else PlanCriticMode.OFF)


def test_the_default_mode_is_announced_as_it_always_was():
    assert PlanCriticSetting(PlanCriticMode.ALWAYS, long_horizon=True).describe() == (
        "enabled (long-horizon, same backend/model)"
    )
    assert PlanCriticSetting(PlanCriticMode.OFF, long_horizon=False).describe() == (
        "disabled (requires --max-hours > 2)"
    )


# The per-round decision


def _decide(mode=PlanCriticMode.AFTER_BLOCKED_ROUND, **overrides) -> PlanReview:
    inputs = {
        "previous_round_iteration": 4,
        "best_iteration": 5,
        "search_mode": SEARCH_MODE_EXPLOIT,
        "last_critic_verdict": "REVISE",
    } | overrides
    return decide_plan_review(mode, **inputs)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, PlanReview(review=False, reason="previous_round_kept")),
        ({"best_iteration": 4}, PlanReview(review=False, reason="previous_round_kept")),
        ({"best_iteration": 3}, PlanReview(review=True, reason="previous_round_blocked")),
        ({"best_iteration": 0}, PlanReview(review=True, reason="previous_round_blocked")),
        ({"previous_round_iteration": None}, PlanReview(review=True, reason="first_round")),
        ({"search_mode": SEARCH_MODE_DIVERSIFY}, PlanReview(review=True, reason="search_mode_diversify")),
        ({"last_critic_verdict": "REPLACE"}, PlanReview(review=True, reason="challenge_requested")),
    ],
)
def test_a_round_is_reviewed_unless_it_continues_a_route_just_kept(overrides, expected):
    assert _decide(**overrides) == expected


@pytest.mark.parametrize(
    ("mode", "review"),
    [(PlanCriticMode.ALWAYS, True), (PlanCriticMode.OFF, False)],
)
def test_the_fixed_modes_ignore_the_round(mode, review):
    assert _decide(mode, previous_round_iteration=None, best_iteration=0).review is review
    assert _decide(mode).review is review


def test_review_time_is_the_critic_and_the_revision_it_asked_for():
    assert review_seconds({"dispatch": 30.0, "plan_critic": 240.0, "plan_revision": 180.5, "total": 900.0}) == (
        pytest.approx(420.5)
    )
    assert review_seconds({"dispatch": 30.0}) == 0.0


# The loop's side of it


def _context(**overrides) -> OrchestrationContext:
    fields = {
        "analysis_commit": "abc123",
        "workspace": "/workspace",
        "gpu_target": "gfx942",
        "objective": "equal-weight mean case speedup",
        "program_context": "Optimize the operator.",
        "source_map_path": "analysis/abc123/source_map.json",
        "cases": (
            CaseEvidence(
                case_id="case-a",
                latency_ms=1.0,
                bottleneck="memory",
                profile_summary_path="analysis/abc123/profiles/case-a/summary.json",
            ),
        ),
    } | overrides
    return OrchestrationContext(**fields)


class _Events:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def append_event(self, event: dict) -> None:
        self.events.append(event)


class _Loop(RoundAdmissionMixin):
    def __init__(self, mode: PlanCriticMode, *, previous_round: int | None, best_iteration: int) -> None:
        self.ic = SimpleNamespace(plan_critic_mode=mode)
        self.run_state = RunState(best=BestRecord(iteration=best_iteration))
        if previous_round is not None:
            self.run_state.round_costs.recent = [
                RoundCost(iteration=previous_round, lanes=3, planning_sec=1800.0, total_sec=3600.0)
            ]
        self.state_store = _Events()
        self._round_started_at = None


def test_a_round_after_a_kept_one_is_planned_without_review_and_says_why(capsys):
    loop = _Loop(PlanCriticMode.AFTER_BLOCKED_ROUND, previous_round=7, best_iteration=8)

    context = loop._with_plan_review(10, _context())

    assert context.review_plan is False
    assert [(e["type"], e["iter"], e["review"], e["reason"]) for e in loop.state_store.events] == [
        ("plan_review", 10, False, "previous_round_kept")
    ]
    assert "[plan-critic] skipping this round: previous_round_kept" in capsys.readouterr().out


def test_a_round_after_a_blocked_one_is_reviewed():
    loop = _Loop(PlanCriticMode.AFTER_BLOCKED_ROUND, previous_round=7, best_iteration=6)

    assert loop._with_plan_review(10, _context()).review_plan is True


def test_the_default_mode_reviews_every_round_and_records_nothing_new():
    loop = _Loop(PlanCriticMode.ALWAYS, previous_round=7, best_iteration=8)

    assert loop._with_plan_review(10, _context()).review_plan is True
    assert loop.state_store.events == []


def test_whether_a_round_is_reviewed_is_not_part_of_any_prompt():
    reviewed, skipped = _context(), _context(review_plan=False)

    assert json.dumps(reviewed.to_prompt_dict(), sort_keys=True) == json.dumps(skipped.to_prompt_dict(), sort_keys=True)


def test_the_open_round_is_charged_for_its_review_and_records_it():
    loop = _Loop(PlanCriticMode.AFTER_BLOCKED_ROUND, previous_round=None, best_iteration=0)
    loop.start_time = 0.0
    loop._campaign_started_at = 0.0
    loop.state_store.save = lambda state: None
    result = SimpleNamespace(
        structured_output_diagnostics={"phase_durations_sec": {"plan_critic": 200.0, "plan_revision": 100.0}}
    )

    loop._charge_plan_review(result)
    assert loop._round_review_sec == 0.0

    loop._open_round(1, lanes=3)
    loop._charge_plan_review(result)
    loop._round_planning_sec = 1800.0
    loop._close_round()

    assert loop.run_state.round_costs.recent[-1].review_sec == pytest.approx(300.0)
    assert loop.state_store.events[-1]["type"] == "round_cost"
    assert loop.state_store.events[-1]["review_sec"] == pytest.approx(300.0)
    assert loop._round_review_sec == 0.0
