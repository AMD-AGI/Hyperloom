# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialists' outcomes at CLOSE: the menu's branches beside its control, read against the menus logged."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.families import Family, Mechanism
from hyperloom_multibranch.notebook.rsi.archive import TreeArchive
from hyperloom_multibranch.notebook.rsi.menu import LoggedMenu, record_family
from hyperloom_multibranch.notebook.rsi.outcomes import (
    CONTROL_GROUP,
    MENU_GROUP,
    GroupOutcome,
    OutcomePool,
    SpecialistOutcomes,
    pooled,
    specialist_outcomes,
)
from hyperloom_multibranch.notebook.rsi.studies import main


def _menu_before_r4(snapshot: NotebookSnapshot) -> LoggedMenu:
    """A menu drawn for s1 after r1 and before r4, its first family r4's at odds 0.3."""

    records = {record.request_id: record for record in snapshot.records}
    r1, r4 = records["r1"], records["r4"]
    family = record_family(snapshot, r4)
    assert family is not None
    at = (r1.requested_at + r4.requested_at) / 2
    return LoggedMenu(at, "s1", "c1", False, (family.label, "attention · other · source"), odds=(0.3, 0.2))


def test_each_explore_test_counts_for_its_branchs_group_and_against_the_menu_drawn_before_it(
    snapshot: NotebookSnapshot,
) -> None:
    found = specialist_outcomes(snapshot, (_menu_before_r4(snapshot),), ())

    total = sum(found.groups.values(), GroupOutcome())
    assert set(found.groups) <= {MENU_GROUP, CONTROL_GROUP} and found.session_id == "sess-1"
    # r1 commits, r2, r4 and r5 do not; r3 failed to launch and was never judged.
    assert (total.branches, total.tests, total.commits, total.merged_gain_pct) == (2, 4, 1, 4.0)
    assert (total.exploring_tests, total.exploring_commits) == (4, 1)
    # Only r4 came after a menu: on its first family, listed at 0.3, and it did not commit.
    assert (total.after_menu, total.on_top, total.listed, total.listed_commits) == (1, 1, 1, 0)
    assert (total.odds_sum, total.squared_error) == (pytest.approx(0.3), pytest.approx(0.09))
    assert found.render().startswith("# Specialist outcomes of sess-1\n\n| group | branches | explore tests |")


def test_an_open_line_of_an_op_family_lists_every_test_of_that_op(snapshot: NotebookSnapshot) -> None:
    drawn = _menu_before_r4(snapshot)
    r4 = next(record for record in snapshot.records if record.request_id == "r4")
    family = record_family(snapshot, r4)
    assert family is not None
    open_line = Family(family.target, Mechanism.OTHER, None).label

    found = specialist_outcomes(snapshot, (dataclasses.replace(drawn, families=(open_line,), odds=(0.2,)),), ())

    total = sum(found.groups.values(), GroupOutcome())
    assert (total.after_menu, total.on_top, total.listed, total.listed_commits) == (1, 1, 1, 0)
    assert total.odds_sum == pytest.approx(0.2)


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
    assert both.note().startswith("commit rate 0.27 with the menu, 0.00 without")
    assert code == 0 and "| menu | 4 | 15 | 0.27 |" in capsys.readouterr().out
    assert main(["outcomes", "--archive", str(tmp_path / "empty")]) == 2
