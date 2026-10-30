# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The direction menu: the settings this session has not tried and the code changes on the profile's operators, each
list ranked by a draw of the commit odds, what a commit can reach and the cost of a test."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.blackboard.peers import Intent, IntentLayer, Verdict
from hyperloom_multibranch.contracts.blackboard.records import (
    BlackboardSnapshot,
    OpShares,
    TestRecord,
    amdahl_ceiling_pct,
)
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus, Conclusion, LearningKind
from hyperloom_multibranch.blackboard.families import Component, Family, Mechanism
from hyperloom_multibranch.blackboard.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.blackboard.rsi.briefing import MENU_NOTE, CrossSessionBriefing
from hyperloom_multibranch.blackboard.rsi.family_stats import PRIOR_TRIALS, FamilyStat, base_rate
from hyperloom_multibranch.blackboard.rsi.lesson_recall import LessonRecall
from hyperloom_multibranch.blackboard.rsi.lessons import LessonPool, LessonRevision, lesson_id_for, revision_id_for
from hyperloom_multibranch.blackboard.rsi.similarity import task_conditions
from hyperloom_multibranch.blackboard.rsi.menu import (
    CODE,
    GAIN_PRIOR_COMMITS,
    HOT_OP_SHARE_PCT,
    MENU_ITEMS,
    MENU_LOG,
    ON_INTEGRATION,
    OPS_LINE,
    SETTINGS,
    TRIED_HERE,
    UNTOUCHED,
    DirectionMenus,
    MenuEvidence,
    MenuLog,
    direction_menu,
    in_control,
)
from hyperloom_multibranch.blackboard.tests.factories import (
    START,
    FakeClock,
    intent,
    judgement,
    profile,
    request,
    result,
)
from hyperloom_multibranch.blackboard.tests.sessions import METRIC, scripted_session
from hyperloom_multibranch.blackboard.tools import CODE_TITLE, SETTINGS_TITLE, render_brief

CONFIG, SOURCE, KERNEL = IntentLayer.CONFIG, IntentLayer.SOURCE, IntentLayer.KERNEL
#: The scripted session's families: r1 commits and r4 does not, on c0 and c1; r2 and r5 fail on c0.
SCHEDULER = Family("scheduling and batching", Mechanism.KERNEL_TUNING, KERNEL)
SPLIT = Family("attention", Mechanism.KERNEL_TUNING, CONFIG)
FP8_KV = Family("KV cache", Mechanism.PRECISION, CONFIG)
STRONG = FamilyStat(10.0, 10.0, (5.0,))
CODE_RATE = (base_rate(SOURCE) + base_rate(KERNEL)) / 2
_REPLAY = Component.WARM_START.value
KV_INTENT = Intent("s1", "c0", "kv cache", "fp8 kv cache", CONFIG)
KV_CHANGE = RecipeDelta(config=ConfigDelta(server_args="--kv-cache-dtype fp8"))


def _shown(snapshot: BlackboardSnapshot) -> str:
    """A branch of the session that is shown its menu."""

    session = snapshot.profile.session_id if snapshot.profile is not None else ""
    return next(name for name in (f"s{n}" for n in range(1, 20)) if not in_control(session, name))


def _line(family: Family) -> Family:
    """The family of ``family``'s line on the code list, which folds the source and kernel layers."""

    return dataclasses.replace(family, layer=None)


def _profiled(snapshot: BlackboardSnapshot, **shares: float) -> BlackboardSnapshot:
    return dataclasses.replace(snapshot, op_shares={"c1": OpShares("c1", "j1", "p1", shares)})


def _missed_on_c1(snapshot: BlackboardSnapshot, *names: str, target: str, mechanism: str) -> BlackboardSnapshot:
    """``snapshot`` with, for each of ``names``, a judged kernel-layer explore test of (``target``, ``mechanism``) on
    c1 that did not commit, after the session's own tests."""

    of = intent("s1", "c1", target, mechanism)
    missed = tuple(
        TestRecord(
            request(name, of, RecipeDelta()),
            result(name),
            judgement(0.2, (name,), verdict=Verdict.FLAT),
            Conclusion.RECORDED_ONLY,
            requested_at=START + 3600.0 + n,
            measured_at=START + 3600.5 + n,
            judged_at=START + 3600.5 + n,
        )
        for n, name in enumerate(names)
    )
    intents = {**snapshot.intents, of.fingerprint: of}
    return dataclasses.replace(snapshot, records=(*snapshot.records, *missed), intents=intents)


def test_the_code_list_folds_a_familys_source_and_kernel_layers_into_one_line_and_the_log_keeps_both(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    claim = snapshot.claims[0]
    scheduler = Intent("s1", "c1", "scheduler", "tune max-num-seqs", KERNEL)
    held = dataclasses.replace(claim, status=ClaimStatus.HELD, holder="zz", intent=scheduler)
    branch = _shown(snapshot)
    menus = DirectionMenus(metric=METRIC, log=MenuLog(tmp_path / MENU_LOG))

    menu = menus.menu(dataclasses.replace(snapshot, claims=(held, *snapshot.claims[1:])), branch, "c1")

    (line,) = menu.code
    assert line.families == (dataclasses.replace(SCHEDULER, layer=SOURCE), SCHEDULER) and not menu.control
    assert line.label == "scheduling and batching · kernel tuning · source or kernel"
    assert line.marks == (TRIED_HERE, "held by zz", ON_INTEGRATION)
    # r1 committed and r4 failed on c1 itself, over a base rate between the source layer's and the kernel layer's.
    assert (line.tests, line.odds) == (2.0, pytest.approx((1 + CODE_RATE * PRIOR_TRIALS) / (2 + PRIOR_TRIALS)))
    assert len(menu.settings) == MENU_ITEMS and len(menu.lines(SETTINGS)) == MENU_ITEMS
    assert menu.lines(CODE) == line.lines() and line.lines()[0].startswith(f"{line.label}: odds 0.34 over 2.0 tests")
    (logged,) = MenuLog(tmp_path / MENU_LOG).menus()
    assert (logged.branch, logged.commit, logged.control) == (branch, "c1", False)
    assert logged.families[-2:] == tuple(family.label for family in line.families)
    assert logged.sections == (*(SETTINGS,) * MENU_ITEMS, CODE, CODE)
    assert logged.odds[-2:] == (round(line.odds, 4),) * 2


def test_one_branch_in_four_is_the_menus_control_and_sees_neither_list(snapshot: BlackboardSnapshot) -> None:
    controls = [name for name in (f"b{n}" for n in range(400)) if in_control("sess-1", name)]

    menu = DirectionMenus(metric=METRIC).menu(snapshot, controls[0], "c1")

    assert 60 < len(controls) < 140
    assert menu.control and menu.settings and menu.lines(SETTINGS) == () == menu.lines(CODE)


def test_a_setting_this_session_tried_is_on_no_list_while_another_way_to_set_it_is(
    snapshot: BlackboardSnapshot,
) -> None:
    memory = Family(Component.MEMORY.value, Mechanism.TUNING, CONFIG)
    kv_size = Family(Component.KV_CACHE.value, Mechanism.TUNING, CONFIG)
    evidence = MenuEvidence({FP8_KV: STRONG, SPLIT: STRONG, memory: STRONG, kv_size: STRONG})

    menu = direction_menu(snapshot, "s1", "c1", evidence, metric=METRIC)

    offered = {item.family for item in menu.settings}
    # r2 split the attention's KV and r5 cast the KV cache to fp8 in this session; no test of it sized the KV cache.
    assert {memory, kv_size} <= offered and not offered & {FP8_KV, SPLIT}
    assert all(item.family.layer is CONFIG for item in menu.settings)


def test_a_gain_is_drawn_toward_the_usual_gain_as_little_as_its_commits_weigh(snapshot: BlackboardSnapshot) -> None:
    plain = {Family(f"plain {n}", Mechanism.TUNING, CONFIG): FamilyStat(10.0, 10.0, (4.0,)) for n in range(2)}
    far = Family("far gain", Mechanism.TUNING, CONFIG)
    assert snapshot.profile is not None
    quiet = dataclasses.replace(
        snapshot, records=(), profile=dataclasses.replace(snapshot.profile, roofline_ceiling={})
    )
    evidence = MenuEvidence({**plain, far: FamilyStat(0.75, 0.25, (80.0,))})

    menu = direction_menu(quiet, "s1", "c1", evidence, metric=METRIC)

    found = {item.family: item for item in menu.settings}
    # The usual gain is the median of every commit's, +4%; one commit a quarter as like this session's moves it little.
    assert found[far].gain_pct == pytest.approx((0.25 * 80.0 + GAIN_PRIOR_COMMITS * 4.0) / (0.25 + GAIN_PRIOR_COMMITS))
    assert found[far].gain_pct == pytest.approx(19.2)
    assert found[Family("plain 0", Mechanism.TUNING, CONFIG)].gain_pct == pytest.approx(4.0)


def test_the_code_list_offers_a_hot_op_no_code_test_acted_on_and_names_the_ops_of_each_line(
    snapshot: BlackboardSnapshot,
) -> None:
    profiled = _profiled(snapshot, fused_moe_kernel=30.0, paged_attention_ll4mi=40.0, rmsnorm=2.0)
    # moe was tuned as a kernel in past sessions; r2 acted on the attention at the config layer only.
    evidence = MenuEvidence({Family("moe", Mechanism.KERNEL_TUNING, KERNEL): FamilyStat(4.0, 1.0, (3.0,))})

    menu = direction_menu(profiled, "s1", "c1", evidence, metric=METRIC)

    found = {item.family: item for item in menu.code}
    attention = found[Family("attention", Mechanism.OTHER, None)]
    assert attention.marks == (UNTOUCHED,) and attention.ops == (("paged_attention_ll4mi", 40.0),)
    assert (attention.tests, attention.odds) == (0.0, pytest.approx(CODE_RATE))
    assert (attention.gain_pct, attention.profile) == (pytest.approx(amdahl_ceiling_pct(40.0)), "c1")
    moe = found[Family("moe", Mechanism.KERNEL_TUNING, None)]
    assert moe.ops == (("fused_moe_kernel", 30.0),) and moe.gain_pct == pytest.approx(amdahl_ceiling_pct(30.0))
    assert moe.lines()[1] == f"{OPS_LINE}30.0% of the profile of c1: fused_moe_kernel 30.0%"
    assert Family("moe", Mechanism.OTHER, None) not in found
    # norm's 2% is under HOT_OP_SHARE_PCT: with no test of it, no list offers it.
    assert 2.0 < HOT_OP_SHARE_PCT and not any(item.family.target == "norm" for item in menu.items)


def test_a_code_line_of_a_large_operator_ranks_above_one_of_a_small_operator_with_better_odds(
    snapshot: BlackboardSnapshot,
) -> None:
    gemm = Family("gemm", Mechanism.KERNEL_TUNING, KERNEL)
    norm = Family("norm", Mechanism.FUSION, KERNEL)
    evidence = MenuEvidence({gemm: FamilyStat(10.0, 2.0, (2.0,)), norm: FamilyStat(10.0, 8.0, (2.0,))})

    menu = direction_menu(_profiled(snapshot, gemm_a8w8=30.0, rmsnorm=3.0), "s1", "c1", evidence, metric=METRIC)

    found = {item.family: item for item in menu.code}
    lines = [item.family for item in menu.code]
    assert found[_line(norm)].odds > found[_line(gemm)].odds
    assert lines.index(_line(gemm)) < lines.index(_line(norm))


def test_a_family_whose_ceiling_is_under_the_sessions_bar_is_on_no_list(snapshot: BlackboardSnapshot) -> None:
    norm_kernel = Family("norm", Mechanism.FUSION, KERNEL)
    norm_setting = Family("norm", Mechanism.TUNING, CONFIG)
    evidence = MenuEvidence({norm_kernel: STRONG, norm_setting: STRONG})
    profiled = _profiled(snapshot, rmsnorm=1.0, gemm_a8w8=30.0)

    barred = direction_menu(profiled, "s1", "c1", evidence, metric=METRIC)
    unbarred = direction_menu(dataclasses.replace(profiled, baselines=()), "s1", "c1", evidence, metric=METRIC)

    # A norm of 1% of the profile can bring 1.01% at most, under the session's bar of 1.6%.
    assert amdahl_ceiling_pct(1.0) < 1.6 and not any(item.family.target == "norm" for item in barred.items)
    assert _line(norm_kernel) in {item.family for item in unbarred.code}


def test_a_structural_change_that_missed_on_the_commit_goes_after_the_other_code_lines(
    snapshot: BlackboardSnapshot,
) -> None:
    fused = Family("norm", Mechanism.FUSION, KERNEL)
    evidence = MenuEvidence({fused: FamilyStat(10.0, 10.0, (20.0,))})
    profiled = _profiled(snapshot, rmsnorm=60.0, gemm_a8w8=30.0)
    missed = _missed_on_c1(profiled, "r9", target="rmsnorm", mechanism="fuse rmsnorm into the gemm")

    before = direction_menu(profiled, "s1", "c1", evidence, metric=METRIC)
    after = direction_menu(missed, "s1", "c1", evidence, metric=METRIC)

    assert before.code[0].family == _line(fused)
    assert after.code[-1].family == _line(fused) and len(after.code) > 1
    # r4's kernel tuning missed on c1 too, and a tuning line stays where its score puts it.
    assert _line(SCHEDULER) in {item.family for item in after.code}


def test_a_code_line_that_missed_three_times_running_on_the_commit_leaves_the_list_unless_it_committed_on_its_line(
    snapshot: BlackboardSnapshot,
) -> None:
    profiled = _profiled(snapshot, gemm_a8w8=30.0)
    gemm = Family("gemm", Mechanism.KERNEL_TUNING, None)

    twice = _missed_on_c1(profiled, "r9", "r10", target="gemm a8w8", mechanism="tuned rows")
    thrice = _missed_on_c1(profiled, "r9", "r10", "r11", target="gemm a8w8", mechanism="tuned rows")
    # With r4, the scheduler missed three times on c1 as well, but r1 committed c1 itself in its family.
    scheduler = _missed_on_c1(profiled, "r9", "r10", target="scheduler", mechanism="tune max-num-seqs")

    def offered(at: BlackboardSnapshot) -> set[Family]:
        return {item.family for item in direction_menu(at, "s1", "c1", MenuEvidence(), metric=METRIC).code}

    assert gemm in offered(twice) and gemm not in offered(thrice)
    assert _line(SCHEDULER) in offered(scheduler)


def test_a_server_on_one_gpu_is_offered_no_parallelism_or_communication_on_any_line(
    snapshot: BlackboardSnapshot,
) -> None:
    parallel = Family(Component.PARALLEL.value, Mechanism.KERNEL_TUNING, CONFIG)
    evidence = MenuEvidence({parallel: FamilyStat(10.0, 10.0, (50.0,))})
    assert snapshot.profile is not None
    single = dataclasses.replace(snapshot, profile=dataclasses.replace(snapshot.profile, tp=1))

    on_one = direction_menu(single, "s1", "c1", evidence, metric=METRIC)
    on_eight = direction_menu(snapshot, "s1", "c1", evidence, metric=METRIC)

    assert not any(item.family.target == Component.PARALLEL.value for item in on_one.items)
    assert parallel in {item.family for item in on_eight.settings}


def test_an_op_family_the_nearest_profile_shows_no_op_of_is_on_no_line(snapshot: BlackboardSnapshot) -> None:
    moe = Family("moe", Mechanism.KERNEL_TUNING, KERNEL)
    evidence = MenuEvidence({moe: FamilyStat(10.0, 10.0, (50.0,))})
    dense = OpShares("c1", "j1", "p1", {"paged_attention_ll4mi": 40.0, "gemm_a16w16": 30.0})
    sparse = OpShares("c1", "j1", "p1", {**dense.shares, "fused_moe_kernel": 20.0})
    root = dataclasses.replace(dense, commit_id="c0")

    def offered(shares: dict[str, OpShares]) -> set[Family]:
        menu = direction_menu(dataclasses.replace(snapshot, op_shares=shares), "s1", "c1", evidence, metric=METRIC)
        return {family for item in menu.items for family in item.families}

    # c1 builds on the root c0, so the root's profile stands for c1's until c1 has one of its own.
    assert not any(family.target == "moe" for family in offered({"c1": dense}) | offered({"c0": root}))
    assert moe in offered({"c1": sparse}) and moe in offered({"c0": root, "c1": sparse}) and moe in offered({})


def test_neither_list_offers_a_warm_start_replay_as_replays_come_from_the_kb(snapshot: BlackboardSnapshot) -> None:
    evidence = MenuEvidence({Family(_REPLAY, Mechanism.OTHER, layer): STRONG for layer in (CONFIG, SOURCE)})

    menu = direction_menu(snapshot, "s1", "c1", evidence, metric=METRIC)

    assert menu.settings and not any(item.family.target == _REPLAY for item in menu.items)


def test_a_claims_note_says_its_familys_odds_and_its_tests_on_the_claimed_commit(snapshot: BlackboardSnapshot) -> None:
    note = DirectionMenus(metric=METRIC).note(snapshot, KV_INTENT, KV_CHANGE)

    assert note == (
        "your direction is KV cache · quantization precision · config: commit odds 0.10 over 1.0 tests, "
        "no commit yet; tests of it on this commit: 1"
    )


async def test_a_code_line_carries_its_familys_best_note_at_either_layer_and_the_brief_logs_it_as_shown(
    tmp_path: Path,
) -> None:
    blackboard = await scripted_session(tmp_path / "blackboard", FakeClock())
    snapshot = await blackboard.snapshot()
    pool = LessonPool(tmp_path / "pool")
    text = "max-num-seqs past 256 starves the decode batches of KV cache"
    pool.write(
        LessonRevision(
            revision_id_for("seqs"),
            lesson_id_for("seqs"),
            LearningKind.PITFALL,
            text,
            task_conditions(profile()),
            "x",
            START - 60.0,
            session_id="past-1",
            target="scheduler",
            mechanism="tune max-num-seqs",
            layer=KERNEL,
        )
    )
    log = tmp_path / LEARNINGS_LOG
    recall = LessonRecall(pool=pool, log=RetrievalLog(log))
    menus = DirectionMenus(metric=METRIC)
    briefing = CrossSessionBriefing(blackboard, metric=METRIC, branch=_shown(snapshot), menus=menus, learnings=recall)

    brief = await briefing.startup_brief("c1")

    line, note = brief.code_menu
    assert line.startswith("scheduling and batching · kernel tuning · source or kernel: ")
    assert note == f"{MENU_NOTE}[past-1, same conditions, unmeasured] {text} [{lesson_id_for('seqs')}]"
    assert f"{CODE_TITLE}\n- {line}\n{note}" in render_brief(brief)
    (logged,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert lesson_id_for("seqs") in logged["past"]


async def test_a_brief_shows_its_branchs_two_lists_and_a_claims_note_comes_from_the_briefing(tmp_path: Path) -> None:
    blackboard = await scripted_session(tmp_path / "blackboard", FakeClock())
    branch = _shown(await blackboard.snapshot())
    briefing = CrossSessionBriefing(blackboard, metric=METRIC, branch=branch, menus=DirectionMenus(metric=METRIC))

    brief = await briefing.startup_brief("c1")

    rendered = render_brief(brief)
    assert (len(brief.settings_menu), len(brief.code_menu)) == (MENU_ITEMS, 1)
    assert f"{SETTINGS_TITLE}\n- {brief.settings_menu[0]}" in rendered and f"{CODE_TITLE}\n- " in rendered
    assert (await briefing.direction_note(KV_INTENT, KV_CHANGE)).startswith("your direction is KV cache")
    assert await CrossSessionBriefing(blackboard).direction_note(KV_INTENT, KV_CHANGE) == ""
    plain = await CrossSessionBriefing(blackboard, branch=branch).startup_brief("c1")
    assert plain.settings_menu == () == plain.code_menu
