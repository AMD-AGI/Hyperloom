# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The direction menu: families ranked by a draw of their odds, the gain within reach and the cost of a test."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.notebook.peers import Intent, IntentLayer
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus
from hyperloom_multibranch.notebook.families import Family, Mechanism
from hyperloom_multibranch.notebook.rsi.briefing import CrossSessionBriefing
from hyperloom_multibranch.notebook.rsi.family_stats import FamilyStat
from hyperloom_multibranch.notebook.rsi.menu import (
    MENU_ITEMS,
    MENU_LOG,
    ON_INTEGRATION,
    TRIED_HERE,
    DirectionMenus,
    MenuEvidence,
    MenuLog,
    direction_menu,
    in_control,
)
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import METRIC, scripted_session
from hyperloom_multibranch.notebook.tools import MENU_TITLE, render_brief

CONFIG, KERNEL = IntentLayer.CONFIG, IntentLayer.KERNEL
#: The scripted session's families: r1 commits and r4 does not, on c0 and c1; r2 and r5 fail on c0.
SCHEDULER = Family("scheduling and batching", Mechanism.KERNEL_TUNING, KERNEL)
SPLIT = Family("attention", Mechanism.KERNEL_TUNING, CONFIG)
FP8_KV = Family("KV cache", Mechanism.PRECISION, CONFIG)
KV_INTENT = Intent("s1", "c0", "kv cache", "fp8 kv cache", CONFIG)
KV_CHANGE = RecipeDelta(config=ConfigDelta(server_args="--kv-cache-dtype fp8"))


def _shown(snapshot: NotebookSnapshot) -> str:
    """A branch of the session that is shown its menu."""

    session = snapshot.profile.session_id if snapshot.profile is not None else ""
    return next(name for name in (f"s{n}" for n in range(1, 20)) if not in_control(session, name))


def test_the_menu_ranks_the_sessions_families_and_marks_where_each_stands(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    claim = snapshot.claims[0]
    held = dataclasses.replace(claim, status=ClaimStatus.HELD, holder="zz", intent=KV_INTENT)
    branch = _shown(snapshot)
    menus = DirectionMenus(metric=METRIC, log=MenuLog(tmp_path / MENU_LOG))

    menu = menus.menu(dataclasses.replace(snapshot, claims=(held, *snapshot.claims[1:])), branch, "c1")

    found = {item.family: item for item in menu.items}
    assert set(found) == {SCHEDULER, SPLIT, FP8_KV} and not menu.control
    assert found[SCHEDULER].marks == (TRIED_HERE, ON_INTEGRATION) and found[FP8_KV].marks == ("held by zz",)
    # r4 failed on c1 itself and counts in full; r5 failed on c0, elsewhere, and counts half.
    assert found[SCHEDULER].odds == pytest.approx((1 + 0.28 * 4) / (2 + 4))
    assert (found[FP8_KV].tests, found[FP8_KV].odds) == (0.5, pytest.approx(0.12 * 4 / 4.5))
    assert len(menu.lines()) == 3 and all(" commit odds " in line for line in menu.lines())
    (logged,) = MenuLog(tmp_path / MENU_LOG).menus()
    assert (logged.branch, logged.commit, logged.control) == (branch, "c1", False)
    assert logged.families == tuple(item.family.label for item in menu.items)


def test_one_branch_in_four_is_the_menus_control_and_sees_none(snapshot: NotebookSnapshot) -> None:
    controls = [name for name in (f"b{n}" for n in range(400)) if in_control("sess-1", name)]

    menu = DirectionMenus(metric=METRIC).menu(snapshot, controls[0], "c1")

    assert 60 < len(controls) < 140
    assert menu.control and menu.items and menu.lines() == ()


def test_the_last_line_explores_the_least_tried_family_with_the_most_within_reach(snapshot: NotebookSnapshot) -> None:
    strong = {
        Family(f"strong {n}", Mechanism.TUNING, CONFIG): FamilyStat(10.0, 10.0, (5.0,)) for n in range(MENU_ITEMS)
    }
    far = Family("weak far", Mechanism.TUNING, CONFIG)
    near = Family("weak near", Mechanism.TUNING, CONFIG)
    archived = {**strong, far: FamilyStat(10.0, 0.2, (9.0,)), near: FamilyStat(10.0, 0.0)}

    menu = direction_menu(snapshot, "s1", "c1", MenuEvidence(archived, {}), metric=METRIC)

    assert len(menu.items) == MENU_ITEMS + 1 and {item.family for item in menu.items[:MENU_ITEMS]} == set(strong)
    assert menu.items[-1].family == far and menu.items[-1].exploring
    assert menu.items[-1].render().startswith("to explore, weak far · parameter tuning · config: commit odds 0.")


def test_a_claims_note_says_its_familys_odds_and_its_tests_on_the_claimed_commit(snapshot: NotebookSnapshot) -> None:
    note = DirectionMenus(metric=METRIC).note(snapshot, KV_INTENT, KV_CHANGE)

    assert note == (
        "your direction is KV cache · quantization precision · config: commit odds 0.10 over 1.0 tests, "
        "no commit yet; tests of it on this commit: 1"
    )


async def test_a_brief_shows_its_branchs_menu_and_a_claims_note_comes_from_the_briefing(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    branch = _shown(await notebook.snapshot())
    briefing = CrossSessionBriefing(notebook, metric=METRIC, branch=branch, menus=DirectionMenus(metric=METRIC))

    brief = await briefing.startup_brief("c1")

    assert len(brief.menu) == 3 and f"{MENU_TITLE}\n- " in render_brief(brief)
    assert (await briefing.direction_note(KV_INTENT, KV_CHANGE)).startswith("your direction is KV cache")
    assert await CrossSessionBriefing(notebook).direction_note(KV_INTENT, KV_CHANGE) == ""
    assert (await CrossSessionBriefing(notebook, branch=branch).startup_brief("c1")).menu == ()
