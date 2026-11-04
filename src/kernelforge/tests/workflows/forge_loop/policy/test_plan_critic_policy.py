# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for choosing which rounds' plans go to the plan critic."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.iteration.merge_candidates import MERGE_PLAN_PREFIX
from kernelforge.workflows.forge_loop.iteration.round_admission import RoundAdmissionMixin
from kernelforge.workflows.forge_loop.policy.plan_critic_policy import PlanReview, decide_plan_review, review_seconds
from kernelforge.workflows.forge_loop.campaign.run_state import BestRecord, RoundCost, RunState
from kernelforge.workflows.forge_loop.search.search_policy import SEARCH_MODE_DIVERSIFY, SEARCH_MODE_EXPLOIT
from kernelforge.workflows.forge_loop.roles.contracts import CaseEvidence, OrchestrationContext

# The per-round decision


def _decide(**overrides) -> PlanReview:
    inputs = {
        "previous_round_kept": True,
        "search_mode": SEARCH_MODE_EXPLOIT,
        "last_critic_verdict": "REVISE",
    } | overrides
    return decide_plan_review(**inputs)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, PlanReview(review=False, reason="previous_round_kept")),
        ({"previous_round_kept": False}, PlanReview(review=True, reason="previous_round_blocked")),
        ({"previous_round_kept": None}, PlanReview(review=True, reason="first_round")),
        ({"search_mode": SEARCH_MODE_DIVERSIFY}, PlanReview(review=True, reason="search_mode_diversify")),
        ({"last_critic_verdict": "REPLACE"}, PlanReview(review=True, reason="challenge_requested")),
    ],
)
def test_a_round_is_reviewed_unless_it_continues_a_route_just_kept(overrides, expected):
    assert _decide(**overrides) == expected


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
    def __init__(self, *, previous_round: int | None, best_iteration: int, best_plan: str = "lane plan") -> None:
        self.ic = SimpleNamespace(lanes=3)
        self.run_state = RunState(best=BestRecord(iteration=best_iteration, plan=best_plan))
        if previous_round is not None:
            self.run_state.round_costs.recent = [
                RoundCost(iteration=previous_round, lanes=3, planning_sec=1800.0, total_sec=3600.0)
            ]
        self.state_store = _Events()
        self._round_started_at = None
        self._round_planning_sec = 0.0
        self._round_measurement_sec = 0.0
        self._round_review_sec = 0.0


@pytest.mark.parametrize(
    ("previous_round", "best_iteration", "best_plan", "kept"),
    [
        (None, 0, "", None),
        (7, 8, "lane plan", True),
        (7, 7, "lane plan", True),
        (7, 6, "lane plan", False),
        (7, 0, "", False),
        (7, 8, f"{MERGE_PLAN_PREFIX} 3+5: lane plan | lane plan", False),
    ],
)
def test_the_previous_round_kept_only_a_candidate_it_planned(previous_round, best_iteration, best_plan, kept):
    loop = _Loop(previous_round=previous_round, best_iteration=best_iteration, best_plan=best_plan)

    assert loop._previous_round_kept() is kept


def test_a_round_after_a_kept_one_is_planned_without_review_and_says_why(capsys):
    loop = _Loop(previous_round=7, best_iteration=8)

    context = loop._with_plan_review(10, _context())

    assert context.review_plan is False
    assert [(e["type"], e["iter"], e["review"], e["reason"]) for e in loop.state_store.events] == [
        ("plan_review", 10, False, "previous_round_kept")
    ]
    assert "[plan-critic] skipping this round: previous_round_kept" in capsys.readouterr().out


def test_a_round_after_a_blocked_one_is_reviewed_and_says_why():
    loop = _Loop(previous_round=7, best_iteration=6)

    assert loop._with_plan_review(10, _context()).review_plan is True
    assert loop.state_store.events[-1]["reason"] == "previous_round_blocked"


def test_whether_a_round_is_reviewed_is_not_part_of_any_prompt():
    reviewed, skipped = _context(), _context(review_plan=False)

    assert json.dumps(reviewed.to_prompt_dict(), sort_keys=True) == json.dumps(skipped.to_prompt_dict(), sort_keys=True)


def test_the_open_round_is_charged_for_its_review_and_records_it():
    loop = _Loop(previous_round=None, best_iteration=0)
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
