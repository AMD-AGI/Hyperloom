# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign clock: whether the budget admits a round, how wide it may be, and what its planning cost."""

from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import replace

import pytest

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.reporting import _round_budget_lines
from kernelforge.workflows.forge_loop.policy.round_budget import (
    ADMISSION_SESSION_SEC,
    FIRST_ROUND_MEASUREMENT_SEC,
    admit_dispatch,
    admit_round,
)
from kernelforge.workflows.forge_loop.campaign.run_state import (
    RoundCost,
    LoopStateStore,
    RoundCostState,
    RunState,
    apply_round_cost,
)
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.workflows.forge_loop.roles.plan_critic import PLAN_CRITIC_TIMEOUT_SEC
from kernelforge.workflows.forge_loop.iteration import round_admission
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _counting_plan,
    _lane_agent_factory,
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


def _seed_round_costs(workspace, *, planning_sec, lanes=3, rounds=2):
    """A campaign resumed with a durable record of what its rounds have cost.

    Observed cost only exists on a campaign that has run, so these tests resume
    one -- which is also the shape the guard matters most in, since the rounds
    that were killed in production were the last ones of a long run.
    """
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = RunState(
        campaign_id="campaign",
        baseline_case_times={"case": 1.0},
        head_commit=head,
    )
    for iteration in range(1, rounds + 1):
        apply_round_cost(
            state,
            RoundCost(iteration=iteration, lanes=lanes, planning_sec=planning_sec, total_sec=planning_sec + 2400.0),
            campaign_sec=iteration * (planning_sec + 2400.0),
        )
    LoopStateStore(str(workspace)).save(state)


def _round_admission_loop(
    tmp_path,
    monkeypatch,
    rounds,
    *,
    remaining_sec,
    planning_sec=2400.0,
    lanes=3,
):
    """One campaign whose budget decides how wide -- or whether -- it plans."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    loop.ic = replace(loop.ic, lanes=lanes)
    _seed_round_costs(workspace, planning_sec=planning_sec)
    monkeypatch.setattr(
        loop,
        "_run_orchestration",
        _counting_plan(loop, workspace, rounds),
    )
    # One round is all these tests need to see admitted, narrowed or refused;
    # the second is starved so the campaign ends on the older reserve guard
    # rather than on the one under test.
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: remaining_sec if not loop.results else 0.0,
    )

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({}),
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )
    return loop, workspace


def test_a_round_the_budget_can_finish_is_admitted_unchanged(tmp_path, monkeypatch):
    rounds: list[int] = []

    loop, _ = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=12 * 3600.0,
    )

    assert rounds[0] == 3
    assert loop.termination_reason != "round_budget_exhausted"


def test_a_round_the_budget_cannot_finish_narrows_instead_of_starting(tmp_path, monkeypatch):
    """A narrower round is worth more than a wide one that is killed halfway."""
    rounds: list[int] = []
    # A minute more than the single-lane round -- whose planning bound is the
    # seeded three-lane round less the two plan reads the Critic is spared --
    # and well short of what two lanes would cost.
    remaining = 2400.0 - 2 * PLAN_CRITIC_TIMEOUT_SEC + ADMISSION_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC + 60.0

    loop, _ = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=remaining,
    )

    assert rounds[0] == 1
    assert loop.termination_reason != "round_budget_exhausted"


# One minute short of the cheapest round this campaign could plan.
_UNAFFORDABLE_SEC = 2400.0 - 2 * PLAN_CRITIC_TIMEOUT_SEC + ADMISSION_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC - 60.0


def test_a_round_no_width_can_pay_for_ends_the_campaign(tmp_path, monkeypatch):
    rounds: list[int] = []

    loop, workspace = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=_UNAFFORDABLE_SEC,
    )

    assert rounds == []
    assert loop.results == []
    assert loop.termination_reason == "round_budget_exhausted"
    assert LoopStateStore(str(workspace)).load().termination_reason == ("round_budget_exhausted")


def test_a_refused_round_is_reported_as_a_refusal_not_as_an_empty_round(tmp_path, monkeypatch, capsys):
    rounds: list[int] = []

    loop, _ = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=_UNAFFORDABLE_SEC,
    )

    assert "ROUND REFUSED FOR BUDGET" in capsys.readouterr().out
    assert loop._round_budget_summary()["refused"]


# What an earlier session of the campaign already banked: 45 minutes of
# planning inside 50 minutes of wall-clock. The session under test then runs
# for seconds, which is the whole point -- the numerator outlives the process,
# the process clock does not.
_BANKED_PLANNING_SEC = 45.0 * 60.0
_BANKED_CAMPAIGN_SEC = 50.0 * 60.0


def test_a_resumed_campaign_reports_a_planning_share_within_its_definition(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The reviewer's reproduction, in the case the feature was built for.

    ``round_costs.planning_total_sec`` is campaign-cumulative and survives
    across sessions; this process's wall-clock does not. Divided one by the
    other, a resumed session running minutes against 45 cumulative minutes of
    planning published a share of several hundred percent -- in the operator
    summary and in ``optimization_report.md``'s ``Round Budget`` section.

    The share must be a share: bounded by 100, and equal to the division of the
    two numbers published beside it, both of which measure the campaign.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    LoopStateStore(str(workspace)).save(
        RunState(
            campaign_id="campaign",
            baseline_case_times={"case": 1.0},
            head_commit=head,
            round_costs=RoundCostState(
                rounds=3,
                planning_total_sec=_BANKED_PLANNING_SEC,
                total_sec=_BANKED_PLANNING_SEC + 600.0,
                campaign_sec=_BANKED_CAMPAIGN_SEC,
            ),
        )
    )
    # One iteration, then out -- so this session's own clock stays far below
    # the planning it inherited.
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: 0.0 if loop.results else 12 * 3600.0,
    )

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    summary = loop._round_budget_summary()
    share = summary["planning_share_pct"]
    # The banked planning is still there, and the campaign clock now covers it
    # rather than being replaced by this session's few seconds.
    assert summary["planning_total_sec"] == pytest.approx(_BANKED_PLANNING_SEC)
    assert summary["campaign_sec"] >= _BANKED_CAMPAIGN_SEC
    assert 0 < share <= 100.0
    assert share == pytest.approx(
        100.0 * summary["planning_total_sec"] / summary["campaign_sec"],
        abs=0.05,
    )

    # The operator summary -- the line that printed "450% of the run" -- and
    # the published report both say it about the campaign now, and both stay
    # inside 100.
    printed = next(
        line for line in capsys.readouterr().out.splitlines() if "Rounds planned across the campaign" in line
    )
    matched = re.search(r"\((\d+(?:\.\d+)?)% of campaign wall-clock\)", printed)
    assert matched, printed
    assert 0 < float(matched.group(1)) <= 100.0
    report = "\n".join(_round_budget_lines(summary))
    assert f"- Planning share of campaign wall-clock: {share:.0f}%" in report
    assert f"- Campaign wall-clock: {summary['campaign_sec'] / 60:.1f} min" in report

    # And the campaign clock is durable, so the NEXT session inherits a span
    # that still covers the planning inside it rather than starting over.
    reloaded = LoopStateStore(str(workspace)).load().round_costs
    assert reloaded.campaign_sec >= reloaded.planning_total_sec
    assert 0 < reloaded.planning_share_pct() <= 100.0


def test_a_finished_round_records_what_its_planning_cost(tmp_path, monkeypatch):
    """The first round has no history to price itself from; it makes some."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    rounds: list[int] = []
    monkeypatch.setattr(
        loop,
        "_run_orchestration",
        _counting_plan(loop, workspace, rounds),
    )

    asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                agent_factory=_lane_agent_factory({}),
                orchestration_service=object(),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    costs = LoopStateStore(str(workspace)).load().round_costs
    assert rounds == [2]
    assert costs.rounds == 1
    assert costs.recent[0].lanes == 2
    assert costs.recent[0].planning_sec > 0
    assert costs.recent[0].total_sec >= costs.recent[0].planning_sec


def test_an_iteration_that_did_not_plan_records_no_round_cost(tmp_path, monkeypatch):
    """A campaign with no orchestration never learns that planning is free."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    assert loop.results
    assert LoopStateStore(str(workspace)).load().round_costs.rounds == 0


def test_session_admission_is_time_only_with_thirty_minute_reserve(
    tmp_path,
    monkeypatch,
):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop.results = [
        IterationResult(
            iteration=index,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary="test",
        )
        for index in range(1, 1001)
    ]

    assert not hasattr(loop.ic, "max_iterations")
    assert loop.ic.budget_reserve_sec == 30 * 60

    monkeypatch.setattr(loop, "_time_remaining", lambda: 30 * 60)
    assert loop._is_budget_exhausted() is False

    monkeypatch.setattr(loop, "_time_remaining", lambda: 30 * 60 - 1)
    assert loop._is_budget_exhausted() is True


def test_the_finalize_reserve_is_its_own_bound_not_a_term_inside_admission(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The relationship ``budget_reserve_sec``'s defining comment states.

    That comment once said a round is admitted only when what remains covers
    its estimated cost ON TOP of the reserve. The code has never done that:
    both admission checks are handed ``_time_remaining()`` with nothing
    subtracted, so the reserve and a round's own requirement are two
    independent lower bounds and the larger of them binds -- ``max``, not a
    sum. That is deliberate, because the reserve is already withheld once by
    ``_is_budget_exhausted()`` and charging it again inside a round's cost
    refused rounds that went on to produce a KEEP.

    Pinned here so the comment cannot drift away from the code again in either
    direction: subtracting the reserve at either call site, or stacking it into
    a requirement, fails this test.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    reserve = float(loop.ic.budget_reserve_sec)

    # A campaign with no round of its own, so both requirements are the
    # constants and this test reads them rather than restating them.
    no_history: list = []
    admission_required = admit_round(
        remaining_sec=0.0,
        requested_lanes=3,
        history=no_history,
        measurement_sec=FIRST_ROUND_MEASUREMENT_SEC,
    ).required_sec
    dispatch_required = admit_dispatch(
        remaining_sec=0.0,
        measurement_sec=FIRST_ROUND_MEASUREMENT_SEC,
    ).required_sec
    round_required = max(admission_required, dispatch_required)
    clears_both = max(reserve, round_required)
    stacked = reserve + round_required
    assert 0 < clears_both < stacked

    seen: list[float] = []

    def _spy(real):
        def _wrapped(*, remaining_sec, **kwargs):
            seen.append(remaining_sec)
            return real(remaining_sec=remaining_sec, **kwargs)

        return _wrapped

    monkeypatch.setattr(round_admission, "admit_round", _spy(admit_round))
    monkeypatch.setattr(round_admission, "admit_dispatch", _spy(admit_dispatch))

    # Halfway between "clears each bound on its own" and "covers their sum".
    between = 0.5 * (clears_both + stacked)
    monkeypatch.setattr(loop, "_time_remaining", lambda: between)

    assert loop._admit_next_round(1) == 3
    assert loop._admit_dispatch(1) is True
    assert loop._is_budget_exhausted() is False
    # Both checks priced the round against the UNRESERVED remaining time. A
    # reserve subtracted at either call site shows up here as a smaller number.
    assert seen == [between, between]

    # Each bound refuses on its own, and neither needs the other's help. Above
    # the round requirement but below the reserve: the round would be
    # affordable, and the loop still will not start a session.
    assert dispatch_required < reserve
    monkeypatch.setattr(loop, "_time_remaining", lambda: dispatch_required + 1.0)
    assert loop._admit_dispatch(2) is True
    assert loop._is_budget_exhausted() is True

    # And the other way round: above the reserve but below what the cheapest
    # round costs, the reserve is satisfied and the round is still refused.
    assert round_required > reserve
    monkeypatch.setattr(loop, "_time_remaining", lambda: reserve + 1.0)
    assert loop._is_budget_exhausted() is False
    assert loop._admit_next_round(3) is None
    assert "ROUND REFUSED FOR BUDGET" in capsys.readouterr().out


def test_is_force_stopped_detects_stop_file(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    assert loop._is_force_stopped() is False
    (workspace / ".stop").touch()
    assert loop._is_force_stopped() is True
