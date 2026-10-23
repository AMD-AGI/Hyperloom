# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay cuts only what a session recorded, under a stop threshold and a branch cap."""

from __future__ import annotations

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.vocab import BranchDriver
from hyperloom_multibranch.notebook.rsi.replay import ReplayPolicy, cut_times, evaluate, replay
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, LATE_BLOOMER, TASK, branch, tree

DEFAULT = ReplayPolicy(stop_pct=0.5, branch_cap=4)
PATIENT = ReplayPolicy(stop_pct=0.25, branch_cap=4)


def test_a_branch_stops_at_the_first_tick_its_recent_gain_falls_under_the_threshold() -> None:
    searched = tree("t", branch("s1", LATE_BLOOMER))

    assert (cut_times(searched, PATIENT)["s1"], cut_times(searched, DEFAULT)["s1"]) == (9000.0, 3600.0)
    assert replay(searched, PATIENT).gain_pct == 12.0
    outcome = replay(searched, DEFAULT)
    assert (outcome.gain_pct, outcome.gpu_hours, outcome.saved_gpu_hours) == (2.3, 1.0, 3.0)
    assert outcome.utility == pytest.approx(2.3 + 0.5 * 2.3 * 3.0)


def test_the_cap_prunes_whole_branches_and_a_cut_branch_takes_its_late_children() -> None:
    searched = tree(
        "t",
        branch("s1", [(1800.0, 5.0), (3600.0, 5.2)]),
        branch("s2", [(1800.0, 4.0)]),
        branch("late", [(9000.0, 3.0)], parent_branch="s1", opened_s=7200.0),
        branch("closed", [], opened_s=600.0, closed_s=600.0, gpu_seconds=0.0),
    )

    assert cut_times(searched, ReplayPolicy(0.0, 1))["s2"] == 0.0
    assert cut_times(searched, ReplayPolicy(1.0, 6))["late"] == 7200.0
    assert cut_times(searched, ReplayPolicy(0.0, 6))["late"] > 7200.0
    assert replay(searched, ReplayPolicy(1.0, 6)).gain_pct == 5.2
    assert replay(searched, ReplayPolicy(0.0, 6)).utility == pytest.approx(5.2)


def test_a_branch_an_autonomous_agent_drives_is_never_cut_but_takes_a_slot() -> None:
    searched = tree(
        "t",
        branch("agent", [(1800.0, 1.0)], driver=BranchDriver.AUTONOMOUS),
        branch("s1", [(3600.0, 2.0)], opened_s=600.0),
    )
    strict = ReplayPolicy(stop_pct=5.0, branch_cap=1)

    assert cut_times(searched, strict) == {INTEGRATION_BRANCH: 4 * HOUR, "agent": 4 * HOUR, "s1": 600.0}
    assert replay(searched, strict).gain_pct == 1.0


def test_evaluation_weighs_similar_sessions_and_counts_its_evidence() -> None:
    history = [
        tree("same", branch("s1", LATE_BLOOMER)),
        tree("unrelated", branch("s1", [(1800.0, 50.0)]), task=TaskConditions(labels={"region": "x"})),
    ]

    patient = evaluate(PATIENT, history, TASK)
    nothing = evaluate(PATIENT, [], TASK)

    assert (patient.utility, patient.sessions, patient.effective_sessions) == (
        pytest.approx(replay(history[0], PATIENT).utility),
        ("same",),
        pytest.approx(1.0),
    )
    assert (nothing.utility, nothing.effective_sessions, nothing.sessions) == (0.0, 0.0, ())
    assert evaluate(DEFAULT, history, TASK, tick_seconds=HOUR).utility > 0
