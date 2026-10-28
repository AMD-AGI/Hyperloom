# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The direction menu: families ranked by a draw of their odds, the gain within reach and the cost of a test."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.notebook.peers import Intent, IntentLayer
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, OpShares, amdahl_ceiling_pct
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, LearningKind
from hyperloom_multibranch.notebook.families import Component, Family, Mechanism
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.notebook.rsi.briefing import MENU_NOTE, CrossSessionBriefing
from hyperloom_multibranch.notebook.rsi.family_stats import FamilyStat, base_rate
from hyperloom_multibranch.notebook.rsi.lesson_recall import LessonRecall
from hyperloom_multibranch.notebook.rsi.lessons import LessonPool, LessonRevision, lesson_id_for, revision_id_for
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.rsi.menu import (
    GAIN_PRIOR_COMMITS,
    HOT_OP_SHARE_PCT,
    MENU_ITEMS,
    MENU_LOG,
    ON_INTEGRATION,
    TRIED_HERE,
    UNTOUCHED,
    DirectionMenus,
    MenuEvidence,
    MenuLog,
    direction_menu,
    in_control,
)
from hyperloom_multibranch.notebook.tests.factories import START, FakeClock, profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC, scripted_session
from hyperloom_multibranch.notebook.tools import MENU_TITLE, render_brief

CONFIG, SOURCE, KERNEL = IntentLayer.CONFIG, IntentLayer.SOURCE, IntentLayer.KERNEL
#: The scripted session's families: r1 commits and r4 does not, on c0 and c1; r2 and r5 fail on c0.
SCHEDULER = Family("scheduling and batching", Mechanism.KERNEL_TUNING, KERNEL)
SPLIT = Family("attention", Mechanism.KERNEL_TUNING, CONFIG)
FP8_KV = Family("KV cache", Mechanism.PRECISION, CONFIG)
_KV_SOURCE = Family("KV cache", Mechanism.TUNING, SOURCE)
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

    found = {item.family: item for item in menu.items if not item.untried}
    assert set(found) == {SCHEDULER, SPLIT, FP8_KV} and not menu.control and menu.items[-1].untried
    assert found[SCHEDULER].marks == (TRIED_HERE, ON_INTEGRATION) and found[FP8_KV].marks == ("held by zz",)
    # r4 failed on c1 itself and counts in full; r5 failed on c0, elsewhere, and counts half.
    assert found[SCHEDULER].odds == pytest.approx((1 + 0.28 * 4) / (2 + 4))
    assert (found[FP8_KV].tests, found[FP8_KV].odds) == (0.5, pytest.approx(0.12 * 4 / 4.5))
    assert len(menu.lines()) == 4 and all(" commit odds " in line for line in menu.lines())
    (logged,) = MenuLog(tmp_path / MENU_LOG).menus()
    assert (logged.branch, logged.commit, logged.control) == (branch, "c1", False)
    assert logged.families == tuple(item.family.label for item in menu.items)
    assert logged.untried == menu.items[-1].family.label


def test_one_branch_in_four_is_the_menus_control_and_sees_none(snapshot: NotebookSnapshot) -> None:
    controls = [name for name in (f"b{n}" for n in range(400)) if in_control("sess-1", name)]

    menu = DirectionMenus(metric=METRIC).menu(snapshot, controls[0], "c1")

    assert 60 < len(controls) < 140
    assert menu.control and menu.items and menu.lines() == ()


def test_after_the_ranked_families_a_line_explores_the_least_tried_with_the_most_within_reach(
    snapshot: NotebookSnapshot,
) -> None:
    strong = {
        Family(f"strong {n}", Mechanism.TUNING, CONFIG): FamilyStat(10.0, 10.0, (5.0,)) for n in range(MENU_ITEMS)
    }
    far = Family("weak far", Mechanism.TUNING, CONFIG)
    near = Family("weak near", Mechanism.TUNING, CONFIG)
    archived = {**strong, far: FamilyStat(10.0, 0.2, (9.0,)), near: FamilyStat(10.0, 0.0)}

    menu = direction_menu(snapshot, "s1", "c1", MenuEvidence(archived, {}), metric=METRIC)

    assert len(menu.items) == MENU_ITEMS + 2 and {item.family for item in menu.items[:MENU_ITEMS]} == set(strong)
    exploring = menu.items[MENU_ITEMS]
    assert exploring.family == far and exploring.exploring and menu.items[-1].untried
    assert exploring.render().startswith("to explore, weak far · parameter tuning · config: commit odds 0.")


def test_a_gain_is_drawn_toward_the_usual_gain_as_little_as_its_commits_weigh(snapshot: NotebookSnapshot) -> None:
    plain = {Family(f"plain {n}", Mechanism.TUNING, CONFIG): FamilyStat(10.0, 5.0, (4.0,)) for n in range(3)}
    far = Family("far gain", Mechanism.TUNING, CONFIG)
    assert snapshot.profile is not None
    quiet = dataclasses.replace(
        snapshot, records=(), profile=dataclasses.replace(snapshot.profile, roofline_ceiling={})
    )
    evidence = MenuEvidence({**plain, far: FamilyStat(0.75, 0.25, (80.0,))}, {})

    menu = direction_menu(quiet, "s1", "c1", evidence, metric=METRIC)

    found = {item.family: item for item in menu.items}
    # The usual gain is the median of every commit's, +4%; one commit a quarter as like this session's moves it little.
    assert found[far].gain_pct == pytest.approx((0.25 * 80.0 + GAIN_PRIOR_COMMITS * 4.0) / (0.25 + GAIN_PRIOR_COMMITS))
    assert found[far].gain_pct == pytest.approx(19.2)
    assert found[Family("plain 0", Mechanism.TUNING, CONFIG)].gain_pct == pytest.approx(4.0)


def _tested(*families: Family) -> MenuEvidence:
    """Archived evidence of every component at the config and source layers, and of ``families``."""

    every = [Family(component.value, Mechanism.TUNING, layer) for component in Component for layer in (CONFIG, SOURCE)]
    return MenuEvidence({family: FamilyStat(1.0, 0.0) for family in (*every, *families)}, {})


def test_the_untried_line_offers_a_hot_op_family_at_a_layer_no_test_acted_on_at_its_base_rate(
    snapshot: NotebookSnapshot,
) -> None:
    shares = OpShares("c1", "j1", "p1", {"fused_moe_kernel": 30.0, "paged_attention_ll4mi": 40.0, "rmsnorm": 2.0})
    profiled = dataclasses.replace(snapshot, op_shares={"c1": shares})
    # moe was tested at the config and kernel layers and attention at no one layer, so at every one.
    evidence = _tested(
        Family("moe", Mechanism.TUNING, CONFIG),
        Family("moe", Mechanism.KERNEL_TUNING, KERNEL),
        Family("attention", Mechanism.TUNING, None),
    )

    menu = direction_menu(profiled, "s1", "c1", evidence, metric=METRIC)

    untried = menu.items[-1]
    assert untried.untried and untried.family == Family("moe", Mechanism.OTHER, SOURCE)
    assert (untried.tests, untried.odds) == (0.0, pytest.approx(base_rate(SOURCE)))
    assert untried.gain_pct <= amdahl_ceiling_pct(30.0)
    assert untried.marks == (f"{UNTOUCHED}; its ops take 30% of this commit's profile",)
    assert untried.render().startswith("untried, moe · other · source: commit odds 0.24 over 0.0 tests")
    # norm's 2% is under the bar, so it is not offered at all.
    assert 2.0 < HOT_OP_SHARE_PCT and not any(item.untried for item in menu.items[:-1])


def test_a_component_tested_at_one_layer_is_still_untried_at_another_profile_or_not(
    snapshot: NotebookSnapshot,
) -> None:
    every = _tested().archived
    evidence = MenuEvidence({family: stat for family, stat in every.items() if family.label != _KV_SOURCE.label}, {})

    menu = direction_menu(snapshot, "s1", "c1", evidence, metric=METRIC)

    # The session's FP8_KV acted on the KV cache at the config layer; the source layer is left.
    assert menu.items[-1].family == Family("KV cache", Mechanism.OTHER, SOURCE)
    assert menu.items[-1].marks == (UNTOUCHED,) and menu.items[-1].odds == pytest.approx(base_rate(SOURCE))


def test_a_claims_note_says_its_familys_odds_and_its_tests_on_the_claimed_commit(snapshot: NotebookSnapshot) -> None:
    note = DirectionMenus(metric=METRIC).note(snapshot, KV_INTENT, KV_CHANGE)

    assert note == (
        "your direction is KV cache · quantization precision · config: commit odds 0.10 over 1.0 tests, "
        "no commit yet; tests of it on this commit: 1"
    )


async def test_a_menu_line_carries_its_familys_best_note_and_the_brief_logs_it_as_shown(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    snapshot = await notebook.snapshot()
    pool = LessonPool(tmp_path / "pool")
    text = "fp8 kv cache needs its scales computed or the accuracy gate fails"
    pool.write(
        LessonRevision(
            revision_id_for("kv"),
            lesson_id_for("kv"),
            LearningKind.PITFALL,
            text,
            task_conditions(profile()),
            "x",
            START - 60.0,
            session_id="past-1",
            target="kv cache",
            mechanism="fp8 kv cache",
            layer=CONFIG,
        )
    )
    log = tmp_path / LEARNINGS_LOG
    recall = LessonRecall(pool=pool, log=RetrievalLog(log))
    menus = DirectionMenus(metric=METRIC)
    briefing = CrossSessionBriefing(notebook, metric=METRIC, branch=_shown(snapshot), menus=menus, learnings=recall)

    brief = await briefing.startup_brief("c1")

    kv = next(index for index, line in enumerate(brief.menu) if line.startswith(FP8_KV.label))
    assert brief.menu[kv + 1] == f"{MENU_NOTE}[past-1, same conditions, unmeasured] {text} [{lesson_id_for('kv')}]"
    assert f"{MENU_TITLE}\n- {brief.menu[0]}" in render_brief(brief) and f"\n{brief.menu[kv + 1]}" in render_brief(
        brief
    )
    (logged,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert lesson_id_for("kv") in logged["past"]


async def test_a_brief_shows_its_branchs_menu_and_a_claims_note_comes_from_the_briefing(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    branch = _shown(await notebook.snapshot())
    briefing = CrossSessionBriefing(notebook, metric=METRIC, branch=branch, menus=DirectionMenus(metric=METRIC))

    brief = await briefing.startup_brief("c1")

    assert len(brief.menu) == 4 and f"{MENU_TITLE}\n- " in render_brief(brief)
    assert (await briefing.direction_note(KV_INTENT, KV_CHANGE)).startswith("your direction is KV cache")
    assert await CrossSessionBriefing(notebook).direction_note(KV_INTENT, KV_CHANGE) == ""
    assert (await CrossSessionBriefing(notebook, branch=branch).startup_brief("c1")).menu == ()
