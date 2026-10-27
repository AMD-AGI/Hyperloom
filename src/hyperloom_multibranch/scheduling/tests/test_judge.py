# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The judge's contract: what it shows the LLM, how it reads the reply, and how the answer is cut to the caps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.lifecycle import SpecialistStatus
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, Validity
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    FrontierScore,
    PendingCandidate,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import FrontierBasis, MergeOutcome
from hyperloom_multibranch.contracts.toolbox import BuildStatus
from hyperloom_multibranch.contracts.judgement import Verdict
from hyperloom_multibranch.notebook import STALL_TESTS, LocalNotebook
from hyperloom_multibranch.notebook.rsi.frontier import (
    NEAR_BAR_COMPONENT,
    PROBATION_COMPONENT,
    SETTLED_TESTS,
    STRANDED_COMPONENT,
)
from hyperloom_multibranch.notebook.rsi.playbook import H0
from hyperloom_multibranch.notebook.rsi.stranded import StrandedCommit, StrandedLead
from hyperloom_multibranch.notebook.tests.factories import intent, judgement, request, result
from hyperloom_multibranch.scheduling.branches import REPAIR_PROTECTED_SECONDS
from hyperloom_multibranch.scheduling.build_waits import waiting_builds
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.events import MB_BIG_GAIN_BARS, MB_BOTTLENECK_SHIFT_PP, EventKind, SessionEvent
from hyperloom_multibranch.scheduling.judge import (
    ENVELOPE_HOLDS,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    PLAYBOOK_JUDGE_PROMPT,
    PLAYBOOK_JUDGE_PROMPT_VERSION,
    HeldOrder,
    HoldReason,
    HoldRules,
    Judge,
    JudgeDecision,
    JudgeView,
    KOrder,
    QueuedIntent,
    StopOrder,
    cap_decision,
    envelope_holds,
    held_branches,
    hold_decision,
    parse_decision,
    queued_intents,
    render_prompt,
    system_prompt,
)
from hyperloom_multibranch.scheduling.forks import PendingBranch
from hyperloom_multibranch.contracts.llm import CompletionFailed, CompletionReply
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    BranchStop,
    RosterCaps,
    StopCause,
    specialist_open,
)
from hyperloom_multibranch.scheduling.similarity import SimilarBranch
from hyperloom_multibranch.scheduling.widening import WideningSignals
from hyperloom_multibranch.toolbox.fake import FakeToolbox, FakeWiring
from hyperloom_multibranch.toolbox.fake_builds import FakeBuildDriver

METRIC = "output_throughput"
NOW = 1000.0
STOP_REASON = "no gain left"


def _entry(
    branch: str,
    *,
    kind: BranchKind = BranchKind.SPECIALIST,
    stopped: bool = False,
    origin: BranchOrigin = BranchOrigin.FIRST_BATCH,
    k: int = 1,
) -> BranchEntry:
    return BranchEntry(
        branch=branch,
        kind=kind,
        origin=BranchOrigin.ROOT if kind is BranchKind.INTEGRATION else origin,
        start_commit="c-root",
        k=k,
        opened_at=0.0,
        stop=BranchStop(StopCause.JUDGE, "flat", 5.0) if stopped else None,
    )


ROSTER = (
    _entry(INTEGRATION_BRANCH, kind=BranchKind.INTEGRATION),
    _entry("b01"),
    _entry("b02"),
    _entry("b03", stopped=True),
)


OPENING_ROSTER = (_entry(INTEGRATION_BRANCH, kind=BranchKind.INTEGRATION),)
WIDENING = WideningSignals(min_useful_seconds=1350.0, cold_start_seconds=90.0, toolbox_headroom=3)


def _view(
    *,
    opening: bool = False,
    roster: tuple[BranchEntry, ...] = ROSTER,
    free_slots: int = 2,
    pending: tuple[PendingBranch, ...] = (),
    history: str = "",
    widening: WideningSignals = WIDENING,
    similar_to: dict[str, tuple[SimilarBranch, ...]] | None = None,
    events: tuple[SessionEvent, ...] = (),
    behind: dict[str, float] | None = None,
    stranded: tuple[StrandedLead, ...] = (),
    profile_running: bool = False,
    explore_gpu_hours: float | None = None,
    max_llm_usd: float | None = None,
    specialists: dict[str, SpecialistStatus] | None = None,
    held: dict[str, HoldReason] | None = None,
    candidates: tuple[PendingCandidate, ...] = (),
    queued: dict[str, tuple[QueuedIntent, ...]] | None = None,
    building: dict[str, tuple[BuildStatus, ...]] | None = None,
) -> JudgeView:
    return JudgeView(
        now=NOW,
        explore_remaining_seconds=3600.0,
        session_remaining_seconds=4000.0,
        caps=RosterCaps(max_branches=4, max_k_per_branch=2, max_parallel_tests=8),
        roster=roster,
        progress=(
            BranchProgress(
                branch="b01",
                running=True,
                start_commit="c-root",
                head_commit="c-b01",
                tests=7,
                commits=2,
                gpu_seconds=120.0,
                judged_since_commit=5,
                head_value={METRIC: 110.0},
                frontier=FrontierScore(3.25, {"trend_pct": 2.5, "observing": 1.0}),
                pending=candidates,
            ),
        ),
        integration_value={METRIC: 100.0},
        roofline_ceiling={METRIC: 400.0},
        metric=METRIC,
        opening=opening,
        free_slots=free_slots,
        pending=pending,
        history=history,
        widening=widening,
        similar_to=similar_to or {},
        events=events,
        behind_integration_pct=behind or {},
        stranded_leads=stranded,
        profile_running=profile_running,
        gpu_hours_used=1.23456,
        explore_gpu_hours=explore_gpu_hours,
        prelude_gpu_hours_upper_bound=0.41666,
        specialist_llm_usd=3.14159,
        max_llm_usd=max_llm_usd,
        scheduler_llm_usd=0.27183,
        specialists=specialists or {},
        held=held or {},
        queued=queued or {},
        building=building or {},
    )


def _text(reply: str) -> CompletionReply:
    return CompletionReply(reply, TokenCounts(), 0.0)


def _evidence(branch: str, **components: float) -> BranchProgress:
    return replace(_view().progress[0], branch=branch, frontier=FrontierScore(1.0, components))


def test_the_safety_envelope_holds_a_branch_on_probation_or_with_a_near_bar_or_a_stranded_lead() -> None:
    scored = (
        _evidence("b01", **{PROBATION_COMPONENT: 1.0, NEAR_BAR_COMPONENT: 1.0}),
        _evidence("b02", **{NEAR_BAR_COMPONENT: 2.0}),
        _evidence("b04", **{STRANDED_COMPONENT: 3.5}),
        _evidence("b05", **{NEAR_BAR_COMPONENT: 0.0}),
        _evidence("b06", **{PROBATION_COMPONENT: 1.0}),
        replace(_evidence("b07"), frontier=None),
    )
    gave_up = {"b06": SpecialistStatus(running=False, crashes=4, lost=1, gave_up=True, judge_failed=False)}

    assert envelope_holds(scored, gave_up) == {
        "b01": HoldReason.PROBATION,
        "b02": HoldReason.NEAR_BAR,
        "b04": HoldReason.STRANDED,
    }


@pytest.mark.parametrize("hold", sorted(ENVELOPE_HOLDS))
def test_a_branch_the_envelope_holds_keeps_its_stop_refused_while_its_k_is_the_judges_to_lower(
    hold: HoldReason,
) -> None:
    roster = (_entry(INTEGRATION_BRANCH, kind=BranchKind.INTEGRATION), _entry("b01", k=2), _entry("b02", k=2))
    decision = JudgeDecision(k=(KOrder("b02", 1),), stop=(StopOrder("b01", STOP_REASON),))

    cut = hold_decision(decision, {"b01": hold, "b02": hold}, roster)
    graced = hold_decision(decision, {"b01": hold, "b02": HoldReason.GRACE}, roster)

    assert (cut.stop, cut.k, cut.held) == ((), (KOrder("b02", 1),), (HeldOrder("b01", hold),))
    assert graced.k == () and set(graced.held) == {HeldOrder("b01", hold), HeldOrder("b02", HoldReason.GRACE)}


def test_the_prompt_shows_the_open_areas_and_the_playbook_before_the_reply_format_only_when_given() -> None:
    areas = "Open areas for a new branch, from the profile of c1:\n- gemm_a8w8 (gemm): 10.0% of GPU time"
    played = replace(_view(history="Stall odds, from 3 sessions."), open_areas=areas, playbook=H0.render())
    prompt = render_prompt(played)
    bare = render_prompt(_view())

    assert prompt.index("Stall odds") < prompt.index(areas) < prompt.index("Scheduling playbook h0:")
    assert prompt.index("Scheduling playbook h0:") < prompt.index("Reply with")
    assert "Scheduling playbook" not in bare and "Open areas" not in bare
    assert (system_prompt(played), system_prompt(_view())) == (PLAYBOOK_JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT)


def test_a_playbook_session_s_prompt_makes_its_stall_evidence_the_one_stop_rule_for_a_stall() -> None:
    playbook, today = (" ".join(text.split()) for text in (PLAYBOOK_JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT))

    assert "stop_candidate is 1 when its stall reached the bar and nothing holds it" in playbook
    assert "stop a branch for stalling only when it is 1" in playbook
    assert "is a candidate to stop" not in playbook and "is a candidate to stop" in today
    assert (
        f"When it reaches {STALL_TESTS}, its specialist is asked to read its brief again and reconsider its direction; "
        "that is no reason to stop it."
    ) in playbook
    assert "context for its stall evidence, and no reason to stop it" in playbook
    assert "A low expected gain is no reason to stop a branch: its stall evidence is." in playbook
    assert "the scheduler cuts the opens it does not allow" in playbook
    assert "as many of its explore tests as the playbook's grace says" in playbook
    assert "near_bar_lead, stranded_lead and probation" in playbook
    for name in ("stop_candidate", "stall_bar", "near_bar_lead", "playbook"):
        assert name not in today


def test_the_prompt_shows_and_explains_the_gpu_hours_and_llm_dollars_used() -> None:
    assert _state(_view())["spend"] == {
        "gpu_hours_used": 1.235,
        "explore_gpu_hours": None,
        "prelude_gpu_hours_upper_bound": 0.417,
        "specialist_llm_usd": 3.14,
        "max_llm_usd": None,
        "scheduler_llm_usd": 0.27,
    }
    assert _state(_view(explore_gpu_hours=7.08333))["spend"]["explore_gpu_hours"] == 7.083
    assert _state(_view(max_llm_usd=25.0))["spend"]["max_llm_usd"] == 25.0
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    for text in (
        "gpu_hours_used",
        "explore_gpu_hours",
        "prelude_gpu_hours_upper_bound",
        "specialist_llm_usd",
        "max_llm_usd",
        "scheduler_llm_usd",
    ):
        assert text in prompt
    assert "a launch cut short, lost or stopped included" in prompt
    assert "a test taken back before its result is not in it" in prompt
    assert "the baseline measured on the root commit included" in prompt
    assert "the measurements the PRELUDE run that built the root commit made on its own" in prompt
    assert (
        "scheduler_llm_usd is what your own calls, the patch resolver's and the rule gate's fix critic's cost" in prompt
    )
    assert "max_llm_usd is the specialist_llm_usd plus scheduler_llm_usd" in prompt
    assert "its wall time times the cards one test uses, which nothing limits and which gpu_hours_used leaves out" in (
        prompt
    )


def test_the_prompt_opens_the_first_branches_where_explore_started_and_not_at_the_root() -> None:
    """EXPLORE opens on the Recipe KB champion when PRELUDE's replay of it improved on the root."""

    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "the opening batch, which opens from the commit EXPLORE started on" in prompt
    assert "the root commit, or the Recipe KB champion PRELUDE replayed onto it" in prompt
    assert "such branches open from that same commit ahead of the opening batch" in prompt


def test_a_branch_row_shows_what_is_known_of_its_specialist_and_a_gave_up_one_is_left_to_the_scheduler() -> None:
    gave_up = SpecialistStatus(running=False, crashes=4, lost=1, gave_up=True, judge_failed=False)
    judging_failed = SpecialistStatus(running=False, crashes=0, lost=6, gave_up=False, judge_failed=True)

    rows = {
        row["branch"]: row for row in _state(_view(specialists={"b01": gave_up, "b02": judging_failed}))["branches"]
    }

    assert rows["b01"]["specialist"] == {
        "running": False,
        "crashes": 4,
        "lost": 1,
        "gave_up": True,
        "judge_failed": False,
    }
    assert (rows["b02"]["specialist"]["lost"], rows["b02"]["specialist"]["judge_failed"]) == (6, True)
    assert "specialist" not in rows[INTEGRATION_BRANCH] and "specialist" not in rows["b03"]
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    for field in ("crashes", "lost", "gave_up", "judge_failed"):
        assert f"({field})" in prompt
    assert "The scheduler stops a gave-up branch itself" in prompt
    assert "A branch whose judging failed makes no more progress" in prompt
    assert "gave up or whose judging failed" not in prompt
    assert "weigh a specialist's lost count against its progress" in prompt


def test_the_prompt_says_a_free_slot_may_be_opened_without_a_stop() -> None:
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())

    assert "free_branch_slots counts the branches you may open at this call having stopped nothing" in prose
    assert "Leaving a free slot unopened needs a reason; opening into it does not." in prose


def test_a_branch_whose_specialist_is_not_known_shows_none() -> None:
    assert not [row for row in _state(_view())["branches"] if "specialist" in row]


def test_a_branch_row_lists_its_similar_branches_and_the_prompt_says_which_twin_to_stop() -> None:
    twins = {"b01": (SimilarBranch("b02", 1.0, identical=True),), "b02": (SimilarBranch("b01", 1.0, identical=True),)}

    rows = {row["branch"]: row for row in _state(_view(similar_to=twins))["branches"]}

    assert rows["b01"]["similar_to"] == [{"branch": "b02", "score": 1.0, "identical": True}]
    assert rows[INTEGRATION_BRANCH]["similar_to"] == []
    assert "similar_to" in JUDGE_SYSTEM_PROMPT and "identical" in JUDGE_SYSTEM_PROMPT
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "stop the one of the pair with the lower expected_gain_pct" in prose
    assert "their start commit or the integration commit they were re-rooted onto" in prose


def test_the_prompt_shows_and_explains_the_events_and_how_far_each_branch_lies_behind() -> None:
    moved = SessionEvent(EventKind.INTEGRATION_MOVED, "moved", {"gain_pct": 4.0, "threshold_pct": 2.0})
    quarantined = SessionEvent(EventKind.ENVIRONMENT_CHANGED, "node n1 quarantined", {"node": "n1", "gpu_ids": []})

    state = _state(_view(events=(moved, quarantined), behind={"b01": 3.14159}))

    assert state["events"] == [
        {"event": "integration_moved", "reason": "moved", "gain_pct": 4.0, "threshold_pct": 2.0},
        {"event": "environment_changed", "reason": "node n1 quarantined", "node": "n1", "gpu_ids": []},
    ]
    rows = {row["branch"]: row for row in state["branches"]}
    assert rows["b01"]["behind_integration_pct"] == 3.14
    assert "behind_integration_pct" not in rows["b02"] and "behind_integration_pct" not in rows[INTEGRATION_BRANCH]
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    for field in ("events", "integration_moved", "environment_changed", "behind_integration_pct"):
        assert field in prompt
    assert f"at least {MB_BIG_GAIN_BARS} times the bar threshold_pct" in prompt
    assert "or merged a foundation fix (foundation_fix true), which merges whatever its gain" in prompt


def test_the_prompt_makes_a_branch_behind_integration_a_stop_candidate_only_once_it_stalls() -> None:
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())

    assert (
        f"A branch far behind whose judged_since_commit has reached {STALL_TESTS} gains little even once merged, so "
        "it is a candidate to stop."
    ) in prompt
    assert (
        f"A branch whose judged_since_commit is below {STALL_TESTS} is not stalling: lying behind is no reason to "
        "stop it"
    ) in prompt
    assert "a branch far behind gains little even once merged, so it is a candidate to stop" not in prompt


def _lead(branch: str, lead_pct: float, commit: StrandedCommit) -> StrandedLead:
    value = 100.0 * (1.0 + lead_pct / 100.0)
    return StrandedLead(branch, True, f"c-{branch}", METRIC, value, 100.0, lead_pct, 1.6, (commit,))


def test_the_prompt_shows_one_line_per_stranded_lead_the_largest_first_and_explains_them() -> None:
    conflicted = StrandedCommit("aaaaaaaa1111", 5.0, MergeOutcome.CONFLICT_SKIPPED, "unresolved patch conflict", 3)
    refused = StrandedCommit("bbbbbbbb2222", 8.0, MergeOutcome.NOT_STACKABLE, "-0.99% on the integration head", 1)
    leads = (_lead("b02", 11.1, refused), _lead("b01", 7.8, conflicted))

    assert _state(_view(stranded=leads))["stranded_leads"] == [
        "b02 leads integration by +11.1% on output_throughput (111 against 100, over the 1.6% bar), and 1 of its "
        "commits did not merge: bbbbbbbb (+8.0%): -0.99% on the integration head",
        "b01 leads integration by +7.8% on output_throughput (108 against 100, over the 1.6% bar), and 1 of its "
        "commits did not merge: aaaaaaaa (+5.0%): unresolved patch conflict, 3 tries",
    ]
    assert _state(_view())["stranded_leads"] == []
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "stranded_leads lists, the largest lead first, each branch whose head leads the integration head" in prompt
    assert "skipped as a conflict, or tested and not stackable" in prompt


@pytest.mark.parametrize("running", [True, False])
def test_the_integration_row_shows_whether_its_profile_runs_and_the_prompt_says_it_holds_a_k_slot(
    running: bool,
) -> None:
    rows = {row["branch"]: row for row in _state(_view(profile_running=running))["branches"]}

    assert rows[INTEGRATION_BRANCH]["profile_running"] is running
    assert "profile_running" not in rows["b01"]
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "profile_running is true while that profile runs" in prompt
    assert "holds one of the integration branch's K slots" in prompt


def test_the_prompt_explains_a_bottleneck_that_moved() -> None:
    moved = SessionEvent(EventKind.BOTTLENECK_MOVED, "moved", {"largest_op": "gemm", "previous_largest_op": "attn"})

    [shown] = _state(_view(events=(moved,)))["events"]

    assert shown == {
        "event": "bottleneck_moved",
        "reason": "moved",
        "largest_op": "gemm",
        "previous_largest_op": "attn",
    }
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "bottleneck_moved" in prompt
    assert f"fell by more than {MB_BOTTLENECK_SHIFT_PP} percentage points" in prompt


def test_the_prompt_shows_no_event_when_none_happened() -> None:
    assert _state(_view())["events"] == []


def _reply(payload: object) -> str:
    return f"Reasoning first.\n```json\n{json.dumps(payload)}\n```"


def test_prompt_shows_frontier_roster_caps_budget_and_headroom() -> None:
    prompt = render_prompt(_view())
    state = json.loads(prompt[prompt.index("{") : prompt.index("\n\nReply with")])

    rows = {row["branch"]: row for row in state["branches"]}
    assert rows["b01"]["expected_gain_pct"] == 3.25
    assert rows["b01"]["components"] == {"trend_pct": 2.5, "observing": 1.0}
    assert rows["b03"]["status"] == "stopped" and rows["b03"]["stopped"] == "judge: flat"
    assert rows[INTEGRATION_BRANCH]["kind"] == "integration"
    assert state["caps"] == {"max_branches": 4, "max_k_per_branch": 2, "max_parallel_tests": 8}
    assert state["free_branch_slots"] == 2
    assert state["explore_remaining_seconds"] == 3600
    assert state["session_remaining_seconds"] == 4000
    assert state["integration"]["headroom_pct"] == 75.0
    assert '"stop_session"' in prompt and "ONE JSON object" in JUDGE_SYSTEM_PROMPT


def test_a_branch_row_shows_its_tests_judged_since_its_last_commit_and_the_prompt_names_the_stall_count() -> None:
    rows = {row["branch"]: row for row in _state(_view())["branches"]}

    assert rows["b01"]["judged_since_commit"] == 5
    assert "judged_since_commit" not in rows["b02"]
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert f"When it reaches {STALL_TESTS}, its specialist is asked to read its brief again" in prose


def _state(view: JudgeView) -> dict[str, object]:
    prompt = render_prompt(view)
    return json.loads(prompt[prompt.index("{") : prompt.index("\n\nReply with")])


def test_prompt_shows_and_explains_whether_a_branch_opened_now_would_pay_off() -> None:
    state = _state(_view())

    assert (state["min_useful_seconds"], state["cold_start"], state["toolbox_headroom"]) == (1350, 90, 3)
    for field in ("min_useful_seconds", "cold_start", "toolbox_headroom"):
        assert field in JUDGE_SYSTEM_PROMPT


def test_prompt_says_the_cold_start_is_unknown_before_the_first_one() -> None:
    state = _state(_view(widening=replace(WIDENING, cold_start_seconds=None)))
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())

    assert state["cold_start"] == "unknown"
    assert '"unknown" until the session\'s first test has waited through one' in prose
    assert "min_useful_seconds is then the trend window alone" in prose


def test_prompt_weighs_an_early_stop_against_the_explore_time_left() -> None:
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())

    assert "refill the free slots rather than stop the session" in prose
    assert "an early stop is your judgement that the time left cannot be spent at all" in prose


def test_the_prompt_version_is_derived_from_the_prompt_text() -> None:
    prompt = render_prompt(_view())
    reply = prompt[prompt.index("Reply with") :]

    assert JUDGE_PROMPT_VERSION == hashlib.sha256(f"{JUDGE_SYSTEM_PROMPT}\n{reply}".encode()).hexdigest()[:12]
    assert (
        PLAYBOOK_JUDGE_PROMPT_VERSION == hashlib.sha256(f"{PLAYBOOK_JUDGE_PROMPT}\n{reply}".encode()).hexdigest()[:12]
    )


def test_a_branch_row_names_what_its_expected_gain_rests_on() -> None:
    view = _view()
    [progress] = view.progress
    scored = replace(progress, frontier=replace(progress.frontier, basis=FrontierBasis.NEW))

    def b01_row(rendered: JudgeView) -> dict[str, object]:
        prompt = render_prompt(rendered)
        state = json.loads(prompt[prompt.index("{") : prompt.index("\n\nReply with")])
        return next(row for row in state["branches"] if row["branch"] == "b01")

    assert b01_row(replace(view, progress=(scored,)))["basis"] == "new"
    assert "basis" not in b01_row(view)


def test_prompt_carries_the_archive_digest_only_when_there_is_one() -> None:
    digest = "3 similar sessions: kernel-layer branches paid off after an hour"

    with_history = render_prompt(_view(history=digest))

    assert f"How similar past sessions went:\n{digest}" in with_history
    assert with_history.index(digest) < with_history.index("Reply with")
    assert "similar past sessions" not in render_prompt(_view())


def test_parse_reads_the_last_decision_object() -> None:
    reply = _reply({"open": 9, "branches": [], "stop": [], "stop_session": None}) + _reply(
        {
            "open": 2,
            "branches": [{"branch": "b01", "k": 2}, {"branch": INTEGRATION_BRANCH, "k": 3}],
            "stop": [{"branch": "b02", "reason": "no gain left"}],
            "stop_session": {"reason": "converged"},
        }
    )

    assert parse_decision(reply) == JudgeDecision(
        open_count=2,
        k=(KOrder("b01", 2), KOrder(INTEGRATION_BRANCH, 3)),
        stop=(StopOrder("b02", "no gain left"),),
        session_stop="converged",
    )


@pytest.mark.parametrize(
    ("reply", "decision"),
    [
        (
            {
                "open": 1,
                "open_k": 2,
                "branches": [{"branch": "b01", "k": 2}],
                "stop": [{"branch": "b02", "reason": STOP_REASON}],
                "stop_session": None,
            },
            JudgeDecision(open_count=1, open_k=2, k=(KOrder("b01", 2),), stop=(StopOrder("b02", STOP_REASON),)),
        ),
        ({"open": 0}, JudgeDecision()),
    ],
)
def test_a_reply_in_the_mb1_format_still_parses(reply: dict[str, object], decision: JudgeDecision) -> None:
    assert parse_decision(_reply(reply)) == decision


def test_a_reply_that_only_stops_the_session_is_a_decision() -> None:
    assert parse_decision(_reply({"stop_session": {"reason": "converged"}})) == JudgeDecision(session_stop="converged")


def test_an_open_object_outranks_a_stop_only_object() -> None:
    reply = _reply({"open": 1, "stop_session": None}) + _reply({"stop_session": {"reason": "converged"}})

    assert parse_decision(reply) == JudgeDecision(open_count=1)


@pytest.mark.parametrize(
    "reply",
    [
        "no json at all",
        _reply({"branches": []}),
        _reply({"stop_session": None}),
        _reply({"branches": [], "stop_session": None}),
        _reply({"stop_session": {"reason": ""}}),
        _reply({"stop_session": "yes"}),
        _reply({"open": "two"}),
        _reply({"open": True}),
        _reply({"open": 1, "branches": [{"branch": "b01", "k": 1.5}]}),
        _reply({"open": 1, "stop": [{"branch": "b01"}]}),
        _reply({"open": 1, "stop": "b01"}),
        _reply({"open": 1, "stop_session": "yes"}),
        _reply({"open": 1, "focus": "attention"}),
        _reply({"open": 1, "focus": [3]}),
    ],
)
def test_malformed_reply_gives_no_decision(reply: str) -> None:
    assert parse_decision(reply) is None


def test_cap_drops_unknown_stopped_and_integration_orders_and_clamps_k() -> None:
    decision = JudgeDecision(
        open_count=3,
        k=(
            KOrder("b01", 9),
            KOrder(INTEGRATION_BRANCH, 0),
            KOrder("b03", 2),
            KOrder("b99", 2),
            KOrder("b02", 2),
        ),
        stop=(
            StopOrder(INTEGRATION_BRANCH, "no"),
            StopOrder("b03", "again"),
            StopOrder("b99", "ghost"),
            StopOrder("b02", "flat"),
        ),
        session_stop="done",
    )

    assert cap_decision(decision, _view()) == JudgeDecision(
        open_count=3,
        k=(KOrder("b01", 2), KOrder(INTEGRATION_BRANCH, 1)),
        stop=(StopOrder("b02", "flat"),),
        session_stop="done",
    )


@pytest.mark.parametrize(("asked", "granted"), [(0, 1), (2, 2), (7, 4)])
def test_opening_count_is_clamped_to_one_and_the_branch_cap(asked: int, granted: int) -> None:
    view = _view(opening=True, roster=OPENING_ROSTER, free_slots=1)

    assert cap_decision(JudgeDecision(open_count=asked), view).open_count == granted


@pytest.mark.parametrize(("asked", "granted"), [(0, 0), (1, 1), (7, 4)])
def test_a_later_call_opens_what_it_asks_up_to_the_branch_cap_and_never_one_by_default(
    asked: int, granted: int
) -> None:
    assert cap_decision(JudgeDecision(open_count=asked), _view()).open_count == granted


def test_a_full_roster_still_opens_the_branch_a_stop_in_the_same_answer_frees() -> None:
    full = _view(free_slots=0)
    parsed = parse_decision(_reply({"open": 1, "stop": [{"branch": "b02", "reason": STOP_REASON}]}))

    assert parsed is not None
    capped = cap_decision(parsed, full)
    assert (capped.open_count, capped.stop) == (1, (StopOrder("b02", STOP_REASON),))


@pytest.mark.parametrize(
    ("reply", "focus"),
    [
        ({"open": 2, "focus": ["attention", "moe_gemm"]}, ("attention", "moe_gemm")),
        ({"open": 2, "focus": [None, " moe_gemm "]}, ("", "moe_gemm")),
        ({"open": 2}, ()),
    ],
)
def test_a_reply_names_the_focus_of_the_branches_it_opens_or_leaves_it_out(
    reply: dict[str, object], focus: tuple[str, ...]
) -> None:
    parsed = parse_decision(_reply(reply))

    assert parsed is not None and (parsed.open_count, parsed.focus) == (2, focus)


def test_a_focus_past_the_branches_opened_is_dropped() -> None:
    capped = cap_decision(JudgeDecision(open_count=1, focus=("attention", "moe_gemm")), _view())

    assert capped.focus == ("attention",)


def test_the_prompt_explains_opening_at_any_call_its_focus_and_the_free_slots() -> None:
    assert _state(_view())["free_branch_slots"] == 2
    for text in ("at any call", "focus", "free_branch_slots", "integration head"):
        assert text in JUDGE_SYSTEM_PROMPT
    assert '"focus"' in render_prompt(_view())


def test_the_prompt_shows_and_explains_the_forks_and_repairs_waiting_for_a_slot() -> None:
    waiting = PendingBranch("b05", BranchOrigin.REPAIR, "b01", "c-b01", NOW - 90.0)

    state = _state(_view(pending=(waiting,)))

    assert state["pending"] == [{"kind": "repair", "branch": "b05", "from_branch": "b01", "waiting_seconds": 90}]
    assert _state(_view())["pending"] == []
    assert "pending" in JUDGE_SYSTEM_PROMPT and "goes to them first" in JUDGE_SYSTEM_PROMPT


async def test_decide_returns_the_capped_decision_and_logs_the_exchange(tmp_path: Path) -> None:
    seen: list[tuple[str, str]] = []

    async def completion(system: str, user: str) -> CompletionReply:
        seen.append((system, user))
        return _text(_reply({"open": 3, "branches": [{"branch": "b01", "k": 5}], "stop": [], "stop_session": None}))

    transcript = tmp_path / "MB-judge.jsonl"
    judge = Judge(completion, transcript, Clock(now=lambda: NOW))

    decision = await judge.decide(_view())
    played = replace(_view(), playbook=H0.render())
    await judge.decide(played)

    assert decision == JudgeDecision(open_count=3, k=(KOrder("b01", 2),))
    assert seen == [(JUDGE_SYSTEM_PROMPT, render_prompt(_view())), (PLAYBOOK_JUDGE_PROMPT, render_prompt(played))]
    row, _played = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert row["at"] == NOW and row["error"] == "" and row["prompt"] == seen[0][1]
    assert '"open": 3' in row["reply"]


async def test_decide_keeps_the_status_quo_when_the_call_fails(tmp_path: Path) -> None:
    async def completion(system: str, user: str) -> CompletionReply:
        raise CompletionFailed("judge call failed: RuntimeError: status=529")

    transcript = tmp_path / "MB-judge.jsonl"

    assert await Judge(completion, transcript, Clock(now=lambda: NOW)).decide(_view()) is None
    (row,) = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert row["reply"] == "" and "status=529" in row["error"]


@pytest.mark.parametrize(
    "reply",
    ["I would open two branches.", _reply({"open": 0, "stop": [{"branch": "b02"}], "stop_session": None})],
)
async def test_decide_keeps_the_status_quo_on_an_unreadable_reply(tmp_path: Path, reply: str) -> None:
    async def completion(system: str, user: str) -> CompletionReply:
        return _text(reply)

    transcript = tmp_path / "MB-judge.jsonl"

    assert await Judge(completion, transcript, Clock(now=lambda: NOW)).decide(_view()) is None
    (row,) = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert row["reply"] == reply and row["error"]


async def test_an_unexpected_error_is_not_swallowed(tmp_path: Path) -> None:
    async def completion(system: str, user: str) -> CompletionReply:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        await Judge(completion, tmp_path / "MB-judge.jsonl", Clock(now=lambda: NOW)).decide(_view())


def test_the_opening_k_is_read_and_clamped() -> None:
    view = _view(opening=True, roster=OPENING_ROSTER, free_slots=4)

    parsed = parse_decision(_reply({"open": 2, "open_k": 9}))

    assert parsed is not None and parsed.open_k == 9
    assert cap_decision(parsed, view).open_k == 2
    assert parse_decision(_reply({"open": 2})) == JudgeDecision(open_count=2)
    assert parse_decision(_reply({"open": 2, "open_k": "two"})) is None


USER_ROSTER = (
    _entry(INTEGRATION_BRANCH, kind=BranchKind.INTEGRATION),
    _entry("b01", origin=BranchOrigin.USER, k=2),
    _entry("b02", origin=BranchOrigin.USER, k=2),
)


def test_a_branch_in_its_grace_is_neither_stopped_nor_lowered_and_one_past_it_is() -> None:
    decision = JudgeDecision(
        k=(KOrder("b01", 1), KOrder("b02", 1)),
        stop=(StopOrder("b01", STOP_REASON),),
    )
    capped = cap_decision(decision, _view(roster=USER_ROSTER))

    assert hold_decision(capped, dict.fromkeys(("b01", "b02"), HoldReason.GRACE), USER_ROSTER) == JudgeDecision(
        held=(HeldOrder("b01", HoldReason.GRACE), HeldOrder("b02", HoldReason.GRACE))
    )
    assert hold_decision(capped, {"b02": HoldReason.GRACE}, USER_ROSTER) == JudgeDecision(
        stop=(StopOrder("b01", STOP_REASON),), held=(HeldOrder("b02", HoldReason.GRACE),)
    )


def test_a_branch_in_its_grace_takes_a_higher_k_and_the_last_order_for_it_counts() -> None:
    view = _view(roster=USER_ROSTER)
    held = {"b01": HoldReason.GRACE}

    assert hold_decision(cap_decision(JudgeDecision(k=(KOrder("b01", 3),)), view), held, USER_ROSTER).k == (
        KOrder("b01", 2),
    )
    lowered = cap_decision(JudgeDecision(k=(KOrder("b01", 2), KOrder("b01", 1))), view)
    assert hold_decision(lowered, held, USER_ROSTER).k == ()


def test_the_opening_call_opens_none_by_default_once_user_branches_opened() -> None:
    view = _view(opening=True, roster=USER_ROSTER, free_slots=2, held=dict.fromkeys(("b01", "b02"), HoldReason.GRACE))

    assert cap_decision(JudgeDecision(open_count=0), view).open_count == 0
    assert cap_decision(JudgeDecision(open_count=2), view).open_count == 2


def test_the_opening_call_opens_a_specialist_branch_when_only_an_agent_s_branch_opened() -> None:
    roster = (_entry("b01", origin=BranchOrigin.AGENT),)
    view = _view(opening=True, roster=roster, free_slots=1, held={"b01": HoldReason.UNSTEERED})

    assert cap_decision(JudgeDecision(open_count=0), view).open_count == 1
    assert not specialist_open(roster)
    assert "an agent's branch does not count" in " ".join(JUDGE_SYSTEM_PROMPT.split())


def _ended(
    branch: str,
    number: int,
    *,
    validity: Validity | None = Validity.SUCCEEDED,
    kind: TestKind = TestKind.TEST,
    purpose: Purpose = Purpose.EXPLORE,
    judged: bool = True,
) -> TestRecord:
    """A test of ``branch``, judged FLAT unless ``judged`` is false; ``validity`` ``None`` leaves it without a
    result, withdrawn."""

    request_id = f"{branch}-r{number}"
    test = request(request_id, intent(branch, "c-root"), kind=kind, purpose=purpose)
    if validity is None:
        return TestRecord(test, withdrawn_at=10.0, withdrawal_reason="branch stopped")
    if not judged:
        return TestRecord(test, result(request_id, validity=validity))
    verdict = judgement(0.0, (request_id,), verdict=Verdict.FLAT)
    return TestRecord(test, result(request_id, validity=validity), verdict, judged_at=11.0)


def test_a_specialist_branch_keeps_its_grace_until_enough_of_its_explore_tests_succeeded_and_were_judged() -> None:
    roster = (*ROSTER, _entry("b04", origin=BranchOrigin.USER), _entry("b05", origin=BranchOrigin.FORK))
    early = [_ended("b01", n) for n in range(SETTLED_TESTS - 1)]
    not_counted = [
        _ended("b01", 90, validity=None),
        _ended("b01", 91, kind=TestKind.PROFILE, purpose=Purpose.PROFILE),
        _ended("b01", 92, purpose=Purpose.INTEGRATION),
        TestRecord(request("b01-r93", intent("b01", "c-root"))),
        _ended("b01", 94, validity=Validity.CANDIDATE_FAILED),
        _ended("b01", 95, validity=Validity.MEASUREMENT_FAILED),
        _ended("b01", 96, judged=False),
    ]
    done = [_ended(branch, n) for branch in ("b02", "b04") for n in range(SETTLED_TESTS)]
    integration = [_ended(INTEGRATION_BRANCH, n) for n in range(SETTLED_TESTS)]

    assert held_branches(roster, (), [*early, *not_counted, *done, *integration], {}, HoldRules(NOW)) == {
        "b01": HoldReason.GRACE,
        "b05": HoldReason.GRACE,
    }


def test_a_playbook_s_grace_holds_a_specialist_branch_for_as_many_judged_tests_as_it_says() -> None:
    settled = [_ended("b01", n) for n in range(3)]

    assert held_branches(ROSTER, (), settled, {}, HoldRules(NOW, 3)) == {"b02": HoldReason.GRACE}
    assert held_branches(ROSTER, (), settled, {}, HoldRules(NOW, 4)) == {
        "b01": HoldReason.GRACE,
        "b02": HoldReason.GRACE,
    }


@pytest.mark.parametrize("state", [{"gave_up": True}, {"judge_failed": True}])
def test_a_branch_whose_specialist_is_out_of_service_has_no_grace(state: dict[str, bool]) -> None:
    status = replace(SpecialistStatus(running=False, crashes=3, lost=0, gave_up=False, judge_failed=False), **state)

    assert held_branches(ROSTER, (), [], {"b01": status}, HoldRules(NOW)) == {"b02": HoldReason.GRACE}


def test_a_held_branch_row_names_why_it_is_held_and_the_prompt_explains_each_reason() -> None:
    view = _view(
        roster=(*ROSTER, _entry("b04", origin=BranchOrigin.REPAIR)),
        held={"b01": HoldReason.GRACE, "b04": HoldReason.REPAIR},
    )
    rows = {row["branch"]: row.get("held") for row in _state(view)["branches"]}

    assert rows == {INTEGRATION_BRANCH: None, "b01": "grace", "b02": None, "b03": None, "b04": "repair"}
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert (
        f"grace: a specialist branch until {SETTLED_TESTS} of its explore tests measured its candidate and were "
        "judged; a failed test, and one whose verdict is still to come, does not count. repair: a branch opened to fix "
        f"a blocked candidate, for {REPAIR_PROTECTED_SECONDS / 60:g} minutes from its open"
    ) in prose


def test_an_autonomous_branch_is_never_steered_by_the_judge_but_a_pending_confirm_still_names_its_hold() -> None:
    roster = (*ROSTER, _entry("b04", origin=BranchOrigin.AGENT), _entry("b05", origin=BranchOrigin.AGENT))
    settled = [_ended("b04", n) for n in range(SETTLED_TESTS)]
    gave_up = SpecialistStatus(running=False, crashes=0, lost=0, gave_up=True, judge_failed=False)
    waiting = (BranchProgress("b05", True, "c-root", "c-b05", 3, 1, 60.0, {}, pending=(CANDIDATE,)),)

    held = held_branches(roster, waiting, settled, {"b04": gave_up}, HoldRules(0.0))

    assert (held["b04"], held["b05"]) == (HoldReason.UNSTEERED, HoldReason.PENDING_CONFIRM)
    decision = JudgeDecision(k=(KOrder("b04", 2), KOrder("b05", 3)), stop=(StopOrder("b04", STOP_REASON),))
    applied = hold_decision(decision, held, roster)
    assert (applied.stop, applied.k) == ((), ())
    assert [order.branch for order in applied.held] == ["b04", "b05"]
    judge_failed = hold_decision(JudgeDecision(k=(KOrder("b04", 3),)), {}, roster)
    assert (judge_failed.k, [order.reason for order in judge_failed.held]) == ((), [HoldReason.UNSTEERED])
    rows = {row["branch"]: row for row in _state(_view(roster=roster, held=held))["branches"]}
    assert rows["b04"]["steered"] is False and "steered" not in rows["b01"]
    prompt = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "steered is false" in prompt
    steered_line = prompt[prompt.index("steered is false") :].split(".")[0]
    assert "unless its judging failed" in steered_line


def test_a_repair_branch_is_held_for_its_protected_period_whatever_its_tests_read() -> None:
    """The judge's grace runs out after a few tests; a numerics fix is worth more than the floor it would score at."""

    roster = (_entry("b01", origin=BranchOrigin.REPAIR), _entry("b02"))
    settled = [_ended(branch, n) for branch in ("b01", "b02") for n in range(SETTLED_TESTS)]
    gave_up = SpecialistStatus(running=False, crashes=0, lost=0, gave_up=True, judge_failed=False)

    protected = HoldRules(REPAIR_PROTECTED_SECONDS - 1)
    assert held_branches(roster, (), settled, {}, protected) == {"b01": HoldReason.REPAIR}
    assert held_branches(roster, (), settled, {}, HoldRules(REPAIR_PROTECTED_SECONDS)) == {}
    assert held_branches(roster, (), settled, {"b01": gave_up}, HoldRules(0.0)) == {}


def test_a_repair_branch_still_reads_as_held_by_its_grace_while_it_has_one() -> None:
    roster = (_entry("b01", origin=BranchOrigin.REPAIR),)

    assert held_branches(roster, (), [], {}, HoldRules(0.0)) == {"b01": HoldReason.GRACE}


CANDIDATE = PendingCandidate("b01-r7", "b01-r7-confirm", "c-b01", METRIC, 116.6, 6.04321)


def _waiting(branch: str, *, pending: bool = True) -> BranchProgress:
    return BranchProgress(
        branch, True, "c-root", f"c-{branch}", 3, 1, 60.0, {}, pending=(CANDIDATE,) if pending else ()
    )


def test_a_branch_with_a_pending_candidate_is_held_until_its_confirm_is_judged_unless_its_judging_failed() -> None:
    roster = (*ROSTER, _entry("b04", origin=BranchOrigin.USER))
    progress = (_waiting("b01"), _waiting("b02", pending=False), _waiting("b03"), _waiting("b04"))
    failed = SpecialistStatus(running=False, crashes=0, lost=0, gave_up=False, judge_failed=True)
    gave_up = replace(failed, gave_up=True, judge_failed=False)
    settled = [_ended(branch, n) for branch in ("b01", "b02", "b03") for n in range(SETTLED_TESTS)]

    assert held_branches(roster, progress, settled, {}, HoldRules(NOW)) == {
        "b01": HoldReason.PENDING_CONFIRM,
        "b04": HoldReason.GRACE,
    }
    assert held_branches(roster, progress, settled, {"b01": failed, "b04": gave_up}, HoldRules(NOW)) == {
        "b04": HoldReason.PENDING_CONFIRM
    }


def test_a_judge_stop_or_lower_k_of_a_branch_with_a_pending_candidate_is_refused_and_named_until_it_clears() -> None:
    roster = (_entry(INTEGRATION_BRANCH, kind=BranchKind.INTEGRATION), _entry("b01", k=2), _entry("b02", k=2))
    decision = JudgeDecision(k=(KOrder("b01", 1), KOrder("b02", 1)), stop=(StopOrder("b01", STOP_REASON),))
    waiting = {"b01": HoldReason.PENDING_CONFIRM}
    capped = cap_decision(decision, _view(roster=roster))

    held = hold_decision(capped, waiting, roster)

    assert held == JudgeDecision(k=(KOrder("b02", 1),), held=(HeldOrder("b01", HoldReason.PENDING_CONFIRM),))
    assert hold_decision(held, waiting, roster) == held
    assert cap_decision(held, _view(roster=roster)) == held
    assert hold_decision(JudgeDecision(k=(KOrder("b01", 2),)), waiting, roster).k == (KOrder("b01", 2),)
    assert (
        hold_decision(capped, {}, roster)
        == capped
        == JudgeDecision(k=(KOrder("b02", 1),), stop=(StopOrder("b01", STOP_REASON),))
    )


def test_a_branch_row_shows_its_pending_candidates_and_the_prompt_says_it_cannot_be_stopped_or_narrowed() -> None:
    rows = {row["branch"]: row for row in _state(_view(candidates=(CANDIDATE,)))["branches"]}

    assert rows["b01"]["pending_candidates"] == [{"request_id": "b01-r7", "gain_over_head_pct": 6.04}]
    assert _state(_view())["branches"][1]["pending_candidates"] == []
    assert "pending_candidates" not in rows["b02"]
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "A branch with a pending candidate cannot be stopped or narrowed until its confirm is judged" in prose


def test_a_branch_row_shows_its_explore_tests_not_measured_yet_with_their_intent_and_wait() -> None:
    moe, attention = intent("b01", "c-b01"), intent("b01", "c-b01", "attention", "fused rope")
    late = replace(TestRecord(request("b01-r9", attention)), requested_at=NOW - 30.0)
    early = replace(TestRecord(request("b01-r8", moe)), requested_at=NOW - 180.4)
    records = [
        late,
        early,
        _ended("b01", 1),
        _ended("b01", 2, validity=None),
        _ended("b01", 3, validity=Validity.MEASUREMENT_FAILED),
        TestRecord(request("b01-r4", moe, purpose=Purpose.INTEGRATION)),
        TestRecord(request("b02-r1", intent("b02", "c-root", "sampler", "cuda graph"))),
    ]
    known = {moe.fingerprint: moe, attention.fingerprint: attention}

    queued = queued_intents(records, known)

    assert queued == {
        "b01": (
            QueuedIntent("b01-r8", "moe gemm: tuned kernel", NOW - 180.4),
            QueuedIntent("b01-r9", "attention: fused rope", NOW - 30.0),
        ),
        "b02": (QueuedIntent("b02-r1", "", 0.0),),
    }
    rows = {row["branch"]: row for row in _state(_view(queued=queued))["branches"]}
    assert rows["b01"]["queued_intents"] == [
        {"request_id": "b01-r8", "intent": "moe gemm: tuned kernel", "waiting_seconds": 180},
        {"request_id": "b01-r9", "intent": "attention: fused rope", "waiting_seconds": 30},
    ]
    assert rows[INTEGRATION_BRANCH]["queued_intents"] == []
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "A branch's queued_intents are its explore tests requested and not measured yet, queued or running" in prose
    assert "Stopping the branch does not withdraw them: they run to their end and their results are recorded" in prose
    assert "weigh a stop against the tests the branch is about to report" in prose


async def test_a_branch_waiting_on_a_native_build_shows_it_once_with_its_component_and_eta(tmp_path: Path) -> None:
    driver = FakeBuildDriver(eta=412.6)
    driver.gate.clear()
    driver.need(lambda test: test.branch_id == "b01", "aiter", "fused-moe")
    toolbox = FakeToolbox(
        notebook=LocalNotebook(tmp_path / "nb", Writer(Role.TOOLBOX, "toolbox")), wiring=FakeWiring(builds=driver)
    )
    for branch, args in (("b01", "--a"), ("b01", "--b"), ("b02", "--c")):
        change = RecipeDelta(config=ConfigDelta(server_args=args))
        await toolbox.submit_test(TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, "c-root", delta=change))
    snapshot = await LocalNotebook(tmp_path / "nb", Writer(Role.OBSERVER, "judge")).snapshot()

    building = await waiting_builds(toolbox, snapshot.records)

    assert {branch: [build.component for build in builds] for branch, builds in building.items()} == {"b01": ["aiter"]}
    rows = {row["branch"]: row for row in _state(_view(building=building))["branches"]}
    assert rows["b01"]["waiting_builds"] == [{"component": "aiter", "eta_seconds": 413}]
    assert rows["b02"]["waiting_builds"] == []
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert "a branch whose tests wait only on them is building, not stalled" in prose
    driver.gate.set()
    await toolbox.close()
