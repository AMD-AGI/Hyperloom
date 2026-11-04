# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stall odds the judge's stop rule rests on: how often a stalled branch and a new one went on to commit."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.contracts.blackboard.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.blackboard.rsi.archive import SearchTree, TreeCommit
from hyperloom_multibranch.blackboard.rsi.history import (
    DEFAULT_STALL_TESTS,
    MAX_POSITION,
    MAX_STALL_TESTS,
    OPENING_SECONDS,
    OTHER_MODEL_WEIGHT,
    PRIOR_TESTS,
    Odds,
    StallOdds,
    hazard_observations,
    new_branch_observations,
    stall_bar,
    stall_odds,
    within,
)
from hyperloom_multibranch.blackboard.rsi.similarity import TaskConditions
from hyperloom_multibranch.blackboard.rsi.tests.trees import TASK, branch, tree

MOE = dataclasses.replace(TASK, labels={**TASK.labels, "model_type": "qwen3_next"})
DENSE = dataclasses.replace(TASK, labels={**TASK.labels, "model_type": "qwen3"})
AGENTX = dataclasses.replace(MOE, labels={**MOE.labels, "workload": "agentx"})
F, T = False, True


def _judged(count: int, *, every: float = 600.0) -> tuple[float, ...]:
    return tuple(every * (n + 1) for n in range(count))


def _many(session_id: str, branches: int, task: TaskConditions) -> SearchTree:
    """``branches`` branches of five judged tests each, every one committing at its second."""

    made = (branch(f"b{n}", [(1250.0, 1.0)], judged_times=_judged(5)) for n in range(branches))
    return tree(session_id, *made, task=task)


def _flat(rate: float, positions: range, seen: float = 10.0) -> dict[int, Odds]:
    return {position: Odds(rate * seen, seen) for position in positions}


def test_every_judged_test_is_one_observation_at_its_position_past_the_last_commit() -> None:
    # Ten judged tests, committing at the second and the seventh.
    worked = branch("b1", [(1250.0, 1.0), (4250.0, 2.0)], judged_times=_judged(10))

    assert list(hazard_observations(worked)) == [
        (1, F),
        (2, T),
        (1, F),
        (2, F),
        (3, F),
        (4, F),
        (5, T),
        (1, F),
        (2, F),
        (3, F),
    ]


def test_a_commit_whose_merge_failed_does_not_end_the_stall() -> None:
    failed = TreeCommit(1250.0, 1.0, CommitOrigin.COMMIT, merge_failed=True)
    stalled = dataclasses.replace(branch("b1", [], judged_times=_judged(4)), commits=(failed,))

    assert list(hazard_observations(stalled)) == [(1, F), (2, F), (3, F), (4, F)]


def test_a_new_branch_counts_its_tests_up_to_its_first_commit() -> None:
    early = branch("b1", [(1850.0, 1.0), (3050.0, 2.0)], judged_times=_judged(5))
    flat = branch("b2", [], judged_times=_judged(3))
    root = branch("b3", [], judged_times=_judged(5), origin=LineageKind.ROOT)

    assert list(new_branch_observations(early)) == [(1, F), (2, F), (3, T)]
    assert list(new_branch_observations(flat)) == [(1, F), (2, F), (3, F)]
    assert list(new_branch_observations(root)) == []


def test_a_tree_archived_before_judgements_were_kept_reads_its_test_times() -> None:
    old = branch("b1", [(1250.0, 1.0)], test_times=_judged(4))

    assert list(hazard_observations(old)) == [(1, F), (2, T), (1, F), (2, F)]


def test_the_odds_within_a_horizon_multiply_each_tests_and_need_every_one_read() -> None:
    curve = {**_flat(0.1, range(1, 5)), 5: Odds(1.0, 2.0)}

    assert within(curve, 1) == pytest.approx(1 - 0.9**4)
    assert within(curve, 2) is None
    assert within({MAX_POSITION: Odds(1.0, 10.0)}, MAX_POSITION + 3, tests=2) == pytest.approx(1 - 0.9**2)


def test_the_stop_bar_is_the_first_stall_less_likely_to_commit_than_a_new_branch() -> None:
    new = _flat(0.2, range(1, 5))
    fading = _flat(0.1, range(1, MAX_POSITION + 1))
    lasting = _flat(0.3, range(1, MAX_POSITION + 1))
    unread = _flat(0.3, range(1, 7))

    assert stall_bar(fading, new) == 4
    assert stall_bar(lasting, new) == MAX_STALL_TESTS
    assert stall_bar(unread, new) == DEFAULT_STALL_TESTS
    assert stall_bar(fading, {1: Odds(1.0, 2.0)}) == DEFAULT_STALL_TESTS


def test_the_odds_read_p_of_k_and_q_off_their_curves() -> None:
    odds = StallOdds(1, 1, hazard=_flat(0.1, range(1, 9)), new_branch=_flat(0.2, range(1, 5)), stall_tests=4)

    assert odds.after_stall(4) == pytest.approx(1 - 0.9**4)
    assert odds.after_stall(5) is None
    assert odds.new_branch_odds == pytest.approx(1 - 0.8**4)
    assert "k=4 34%; too few tests to tell at the other k." in odds.render()
    assert "less likely to commit than a new one" in odds.render()


def test_another_model_type_weighs_less() -> None:
    odds = stall_odds((_many("same", 1, MOE), _many("other", 1, DENSE)), MOE)

    both = 1.0 + OTHER_MODEL_WEIGHT
    assert (odds.sessions, odds.same_model) == (2, 1)
    assert odds.hazard == {1: Odds(0.0, 2 * both), 2: Odds(both, 2 * both), 3: Odds(0.0, both)}
    assert odds.opening_branch == {1: Odds(0.0, both), 2: Odds(both, both)}


def test_only_a_branch_opened_after_the_opening_batch_tells_what_a_new_branch_does() -> None:
    opened = OPENING_SECONDS + 60.0
    opening = branch("b1", [(1250.0, 1.0)], judged_times=_judged(5))
    later = branch("b2", [], opened_s=opened, judged_times=tuple(opened + 600.0 * n for n in (1, 2, 3)))

    odds = stall_odds((tree("s", opening, later, task=MOE),), MOE)

    assert odds.new_branch == {1: Odds(0.0, 1.0), 2: Odds(0.0, 1.0), 3: Odds(0.0, 1.0)}
    assert odds.opening_branch == {1: Odds(0.0, 1.0), 2: Odds(1.0, 1.0)}
    assert "A branch opened mid-session commits within its first 4 judged tests: too few." in odds.render()


def test_the_archive_cap_leaves_the_running_session_its_full_weight() -> None:
    live = tree("live", branch("b1", [], judged_times=_judged(5)), task=MOE)

    odds = stall_odds((_many("crowd", 30, MOE),), MOE, session=live)

    assert odds.opening_branch[1] == Odds(0.0, PRIOR_TESTS + 1.0)
    assert odds.opening_branch[2] == Odds(PRIOR_TESTS, PRIOR_TESTS + 1.0)
    assert odds.session_tests == 5
    assert "and this session's 5 judged tests" in odds.render()


def test_another_workload_and_the_session_itself_are_left_out() -> None:
    live = _many("live", 1, MOE)

    odds = stall_odds((live, _many("agent", 1, AGENTX)), MOE, session=live)

    assert odds.sessions == 0
    assert odds.opening_branch == {1: Odds(0.0, 1.0), 2: Odds(1.0, 1.0)}


def test_the_first_commit_time_is_the_median_over_branches_opened_mid_session() -> None:
    quick = branch("b1", [(2400.0, 1.0)], opened_s=1800.0, judged_times=(2100.0, 2400.0))
    slow = branch("b2", [(5400.0, 1.0)], opened_s=1800.0, judged_times=(2100.0, 5400.0))
    opening = branch("b3", [(60.0, 1.0)], judged_times=_judged(4))

    odds = stall_odds((tree("s", quick, slow, opening, task=MOE),), MOE)

    assert odds.first_commit_minutes == 10.0


def test_the_odds_read_as_prompt_text_whole_lines_within_the_limit() -> None:
    odds = stall_odds((_many("same", 2, MOE),), MOE)
    text = odds.render()

    assert text.startswith("Stall odds, from 1 archived sessions (1 of this model type). How often")
    assert f"Stop bar: {odds.stall_tests} judged tests" in text
    assert "where the archive cannot tell yet" in text
    assert odds.render(max_chars=40) == ""
    assert all(line in text.splitlines() for line in odds.render(max_chars=400).splitlines())
