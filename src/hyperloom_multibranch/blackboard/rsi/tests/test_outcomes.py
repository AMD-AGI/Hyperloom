# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialists' outcomes at CLOSE: the menu's branches beside its control, read against the menus logged."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.peers import IntentLayer
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot
from hyperloom_multibranch.blackboard.families import Family, Mechanism
from hyperloom_multibranch.blackboard.rsi.archive import TreeArchive, TreeDirection
from hyperloom_multibranch.blackboard.rsi.menu import CODE, SETTINGS, LoggedMenu, record_family
from hyperloom_multibranch.blackboard.rsi.outcomes import (
    CONTROL_GROUP,
    MENU_GROUP,
    NO_MENU_GROUP,
    GroupOutcome,
    OutcomePool,
    SpecialistOutcomes,
    pooled,
    specialist_outcomes,
)
from hyperloom_multibranch.blackboard.rsi.similarity import task_conditions
from hyperloom_multibranch.blackboard.rsi.studies import main
from hyperloom_multibranch.blackboard.rsi.tests.trees import branch, tree


def _menu_before_r4(snapshot: BlackboardSnapshot) -> LoggedMenu:
    """A menu drawn for s1 after r1 and before r4: a setting at odds 0.2, then the code line of r4's family, logged at
    the source and the kernel layers at odds 0.3."""

    records = {record.request_id: record for record in snapshot.records}
    r1, r4 = records["r1"], records["r4"]
    family = record_family(snapshot, r4)
    assert family is not None and family.layer is IntentLayer.KERNEL
    at = (r1.requested_at + r4.requested_at) / 2
    source = dataclasses.replace(family, layer=IntentLayer.SOURCE)
    families = ("attention · other · config", source.label, family.label)
    return LoggedMenu(at, "s1", "c1", False, families, odds=(0.2, 0.3, 0.3), sections=(SETTINGS, CODE, CODE))


def test_each_explore_test_counts_for_its_branchs_group_and_against_the_menu_drawn_before_it(
    snapshot: BlackboardSnapshot,
) -> None:
    found = specialist_outcomes(snapshot, (_menu_before_r4(snapshot),), ())

    total = sum(found.groups.values(), GroupOutcome())
    assert set(found.groups) <= {MENU_GROUP, CONTROL_GROUP} and found.session_id == "sess-1"
    # r1 commits, r2, r4 and r5 do not; r3 failed to launch and was never judged.
    assert (total.branches, total.tests, total.commits, total.merged_gain_pct) == (2, 4, 1, 4.0)
    assert (total.exploring_tests, total.exploring_commits) == (4, 1)
    # Only r4 came after a menu: its kernel layer is on the code line, listed at 0.3, and it did not commit.
    assert (total.after_menu, total.listed, total.listed_commits) == (1, 1, 0)
    assert (total.odds_sum, total.squared_error) == (pytest.approx(0.3), pytest.approx(0.09))
    assert found.render().startswith("# Specialist outcomes of sess-1\n\n| group | branches | explore tests |")
    assert "| on a menu line |" in found.render()


def test_an_open_line_of_an_op_family_lists_every_test_of_that_op(snapshot: BlackboardSnapshot) -> None:
    drawn = _menu_before_r4(snapshot)
    r4 = next(record for record in snapshot.records if record.request_id == "r4")
    family = record_family(snapshot, r4)
    assert family is not None
    open_line = Family(family.target, Mechanism.OTHER, None).label

    one_line = dataclasses.replace(drawn, families=(open_line,), odds=(0.2,), sections=(CODE,))
    found = specialist_outcomes(snapshot, (one_line,), ())

    total = sum(found.groups.values(), GroupOutcome())
    assert (total.after_menu, total.listed, total.listed_commits) == (1, 1, 0)
    assert total.odds_sum == pytest.approx(0.2)


def test_an_explore_test_of_a_family_the_archive_tried_and_never_committed_in_is_a_retry(
    snapshot: BlackboardSnapshot,
) -> None:
    records = {record.request_id: record for record in snapshot.records}
    family = record_family(snapshot, records["r4"])
    assert family is not None and snapshot.profile is not None
    assert record_family(snapshot, records["r1"]) == family
    tried = TreeDirection(
        family.target,
        family.mechanism.value,
        family.layer,
        tests=3,
        family_target=family.target,
        family_mechanism=family.mechanism.value,
    )
    past = tree("past", branch("b1", [], directions=(tried,)), task=task_conditions(snapshot.profile))

    found = specialist_outcomes(snapshot, (), (past,))

    total = sum(found.groups.values(), GroupOutcome())
    # r1 and r4 are tests of the family the past session tried three times; r1 committed.
    assert (total.retried_tests, total.retried_commits) == (2, 1)
    # No menu was given, so no branch is the menu's or its control.
    assert set(found.groups) == {NO_MENU_GROUP} and "| no menu |" in found.render()
    assert found.render().splitlines()[2].endswith("| retried barren share | commit rate retried / rest |")


def test_the_pool_keeps_each_sessions_outcomes_once_and_the_studies_add_them_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = OutcomePool.beside(TreeArchive(tmp_path))
    first = SpecialistOutcomes("a", {MENU_GROUP: GroupOutcome(branches=3, tests=10, commits=3)})
    second = SpecialistOutcomes(
        "b", {MENU_GROUP: GroupOutcome(branches=1, tests=5, commits=1), CONTROL_GROUP: GroupOutcome(1, 4, 0)}
    )

    assert pool.record(first) and pool.record(second) and not pool.record(first)
    both = pooled(pool.sessions())
    code = main(["outcomes", "--archive", str(tmp_path)])

    assert both.groups[MENU_GROUP] == GroupOutcome(branches=4, tests=15, commits=4)
    assert both.note() == (
        "commit rate 0.27 with the menu, 0.00 without; on a menu line n/a against n/a; retried barren 0.00 against 0.00"
    )
    assert code == 0 and "| menu | 4 | 15 | 0.27 |" in capsys.readouterr().out
    assert main(["outcomes", "--archive", str(tmp_path / "empty")]) == 2


def test_an_outcome_record_the_pool_cannot_read_is_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pool = OutcomePool.beside(TreeArchive(tmp_path))
    pool.record(SpecialistOutcomes("a", {MENU_GROUP: GroupOutcome(branches=1, tests=2, commits=1)}))
    (pool.root / "b.json").write_text("{", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        kept = pool.sessions()

    assert [item.session_id for item in kept] == ["a"] and "skipping unreadable outcome record" in caplog.text
