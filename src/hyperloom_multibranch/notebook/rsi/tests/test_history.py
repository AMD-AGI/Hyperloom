# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The history digest the scheduler's judge reads: how similar past sessions went, and where this one stands."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import SOURCE_NOTEBOOK, SearchTree, TreeBranch, TreeDirection
from hyperloom_multibranch.notebook.rsi.history import SoFar, Standing, history_digest, session_history
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions, task_conditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, branch, tree
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC

TASK = task_conditions(profile())
GEMM = TreeDirection("moe gemm", "tuned kernel", IntentLayer.KERNEL, 6, 2, 3.0)
SPLIT_KV = TreeDirection("attention", "split kv", IntentLayer.CONFIG, 8, 0, -1.0)


def _multi(session_id: str, *branches: TreeBranch) -> SearchTree:
    return dataclasses.replace(tree(session_id, *branches, task=TASK), source=SOURCE_NOTEBOOK)


# Three branches over four hours: s1 commits at its 4th test, s2 never does and is stopped, s3 commits after its 10th.
WIDE = _multi(
    "wide",
    branch(
        "s1",
        [(1800.0, 2.0), (5400.0, 5.0)],
        test_times=(0.0, 600.0, 1200.0, 1800.0, 3600.0, 5400.0),
        directions=(GEMM,),
    ),
    branch("s2", [], test_times=tuple(600.0 * n for n in range(8)), stop_reason="no progress", directions=(SPLIT_KV,)),
    branch("s3", [(6000.0, 3.0)], test_times=tuple(600.0 * n for n in range(10))),
)
NARROW = _multi("narrow", branch("s1", [(3600.0, 2.0)], test_times=(0.0, 1200.0, 2400.0, 3600.0), directions=(GEMM,)))
OLD_FLOW = tree("old-flow", branch("s1", [(7200.0, 4.0)], test_times=(0.0, 3600.0)), task=TASK)
# The judge's own words for a stop, as long as a real session's: each is a reason of its own.
LONG_STOP = (
    "judge: 4 tests, no commits, measured trend expects 0% gain while consuming the most GPU time; replace it with a "
    "fresh specialist from the integration head while headroom and explore time remain"
)
WORDY = _multi(
    "wordy",
    branch("s1", [(1800.0, 2.0)], test_times=(0.0, 1800.0), directions=(GEMM,)),
    branch("s2", [], test_times=(0.0, 600.0), stop_reason=f"{LONG_STOP}, first", directions=(SPLIT_KV,)),
    branch("s3", [], test_times=(0.0, 600.0), stop_reason=f"{LONG_STOP}, second"),
)
DIRECTIONS_HEADING = "Directions tried, most commits first:"


def test_too_little_similar_history_says_so_instead_of_drawing_from_it() -> None:
    far = TaskConditions(labels={"gpu": TASK.labels["gpu"], "framework": "sglang", "model": "m", "precision": "bf16"})

    alone = history_digest([NARROW], TASK)
    unlike = history_digest([dataclasses.replace(WIDE, task=far), dataclasses.replace(NARROW, task=far)], TASK)

    assert not alone.enough and (alone.sessions, alone.effective_sessions) == (1, pytest.approx(1.0))
    assert not unlike.enough and unlike.sessions == 0
    assert alone.render().startswith("History: too thin to go by (1 similar session, worth 1.0")


def test_the_digest_sums_up_pace_final_gain_branches_and_directions() -> None:
    digest = history_digest([WIDE, NARROW, OLD_FLOW], TASK, so_far=SoFar(0.25, 1.0))

    assert digest.enough and (digest.sessions, digest.multi_branch, digest.backfilled) == (3, 2, 1)
    # By a quarter of the budget: wide had 2 of 5, narrow 2 of 2, the old flow 0 of 4; by half, all of it.
    assert digest.pace == {0.25: pytest.approx(0.4), 0.5: pytest.approx(1.0), 0.75: pytest.approx(1.0)}
    assert (digest.median_final_gain_pct, digest.p90_final_gain_pct) == (pytest.approx(4.0), pytest.approx(5.0))
    assert digest.standing == Standing(0.25, 1.0, pytest.approx(2.0))
    branches = digest.branches
    assert branches is not None and (branches.sessions, branches.mean_opened) == (2, pytest.approx(2.0))
    assert branches.committed_share == pytest.approx((2 / 3 + 1) / 2)
    assert branches.median_first_commit_test == pytest.approx(4.0)
    assert branches.late_commits == {4: (1, 2), 8: (1, 2)}
    assert branches.stop_reasons == (("no progress", 1),)
    assert [(d.target, d.sessions, d.tests, d.commits, d.best_gain_pct) for d in digest.directions] == [
        ("moe gemm", 2, 12, 4, pytest.approx(3.0)),
        ("attention", 1, 8, 0, pytest.approx(-1.0)),
    ]


def test_the_text_keeps_whole_lines_within_its_budget() -> None:
    digest = history_digest([WIDE, NARROW, OLD_FLOW], TASK, so_far=SoFar(0.25, 1.0))

    full, short = digest.render(), digest.render(max_chars=300)

    assert "History: 3 similar sessions (worth 3.0" in full
    assert "the median similar session had +2.0% by then (behind it by 1.0 points)" in full
    assert "Branches, from 2 multi-branch sessions: 2.0 opened on average; 83% of them committed" in full
    assert "Branches without a commit after 4 tests: 1 of 2 committed later." in full
    assert "- attention / split kv [config]: 1 session, 8 tests, 0 commits, best -1.0%" in full
    assert "- moe gemm / tuned kernel [kernel]: 2 sessions, 12 tests, 4 commits, best +3.0%" in full
    assert len(short) <= 300 and set(short.splitlines()) <= set(full.splitlines())
    assert len(short.splitlines()) < len(full.splitlines())


def test_a_heading_is_left_out_when_none_of_its_lines_fit() -> None:
    digest = history_digest([WORDY, NARROW], TASK)
    full = digest.render(max_chars=100_000)
    assert DIRECTIONS_HEADING in full.splitlines()

    for budget in range(len(full) + 1):
        lines = digest.render(max_chars=budget).splitlines()
        if DIRECTIONS_HEADING in lines:
            under = lines[lines.index(DIRECTIONS_HEADING) + 1 :]
            assert under and under[0].startswith("- "), budget


def test_the_directions_go_in_before_the_reasons_branches_were_stopped_for() -> None:
    digest = history_digest([WORDY, NARROW], TASK)
    full = digest.render(max_chars=100_000)
    assert "Stopped for:" in full

    for budget in range(len(full) + 1):
        text = digest.render(max_chars=budget)
        if "Stopped for:" in text:
            assert "- moe gemm / tuned kernel [kernel]" in text and "- attention / split kv [config]" in text, budget


def test_a_running_session_is_digested_from_its_own_snapshot(snapshot: NotebookSnapshot) -> None:
    digest = session_history(snapshot, [WIDE, NARROW, OLD_FLOW], METRIC, budget_seconds=2 * HOUR)
    unprofiled = session_history(dataclasses.replace(snapshot, profile=None), [WIDE], METRIC, budget_seconds=HOUR)

    assert digest.enough and digest.standing is not None
    assert digest.standing.gain_pct == pytest.approx(4.1) and 0.0 <= digest.standing.fraction < 0.01
    assert not unprofiled.enough and unprofiled.render().startswith("History: no digest yet")
