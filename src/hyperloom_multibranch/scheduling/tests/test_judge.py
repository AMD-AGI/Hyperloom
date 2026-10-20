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
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom.contracts.measurement import TestKind, Validity
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    FrontierScore,
    PendingCandidate,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import FrontierBasis
from hyperloom_multibranch.notebook import STALL_TESTS
from hyperloom_multibranch.notebook.rsi.calibration import PolicyChoice
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.notebook.tests.factories import intent, request, result
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.events import MB_BIG_GAIN_BARS, MB_BOTTLENECK_SHIFT_PP, EventKind, SessionEvent
from hyperloom_multibranch.scheduling.judge import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    MB_USER_BRANCH_GRACE_TESTS,
    HeldOrder,
    HoldReason,
    Judge,
    JudgeDecision,
    JudgeView,
    KOrder,
    StopOrder,
    cap_decision,
    held_branches,
    hold_decision,
    parse_decision,
    render_prompt,
)
from hyperloom_multibranch.scheduling.forks import PendingBranch
from hyperloom_multibranch.scheduling.llm import CompletionFailed, CompletionReply
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    BranchStop,
    RosterCaps,
    StopCause,
)
from hyperloom_multibranch.scheduling.similarity import SimilarBranch
from hyperloom_multibranch.scheduling.widening import WideningSignals

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
CHOSEN = PolicyChoice(
    FrontierPolicy(METRIC, window_seconds=1800.0),
    "window x2",
    "window x2: rank correlation 0.61 against 0.42 for the defaults, over 24 compared rounds in 3 similar sessions",
)
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
    profile_running: bool = False,
    explore_gpu_hours: float | None = None,
    max_llm_usd: float | None = None,
    specialists: dict[str, SpecialistStatus] | None = None,
    held: dict[str, HoldReason] | None = None,
    candidates: tuple[PendingCandidate, ...] = (),
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
        profile_running=profile_running,
        gpu_hours_used=1.23456,
        explore_gpu_hours=explore_gpu_hours,
        prelude_gpu_hours_upper_bound=0.41666,
        specialist_llm_usd=3.14159,
        max_llm_usd=max_llm_usd,
        scheduler_llm_usd=0.27183,
        specialists=specialists or {},
        held=held or {},
        frontier_policy=CHOSEN,
    )


def _text(reply: str) -> CompletionReply:
    return CompletionReply(reply, TokenCounts(), 0.0)


def test_the_prompt_names_the_frontier_policy_and_the_evidence_it_was_chosen_on() -> None:
    assert _state(_view())["frontier_policy"] == {"label": CHOSEN.label, "evidence": CHOSEN.evidence}
    assert "frontier_policy names the scoring's variation" in " ".join(JUDGE_SYSTEM_PROMPT.split())


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


def test_a_branch_row_shows_what_is_known_of_its_specialist_and_the_prompt_says_to_stop_one_that_gave_up() -> None:
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
    assert "A branch whose specialist gave up or whose judging failed makes no more progress" in prompt
    assert "weigh a specialist's lost count against its progress" in prompt


def test_a_branch_whose_specialist_is_not_known_shows_none() -> None:
    assert not [row for row in _state(_view())["branches"] if "specialist" in row]


def test_a_branch_row_lists_its_similar_branches_and_the_prompt_says_which_twin_to_stop() -> None:
    twins = {"b01": (SimilarBranch("b02", 1.0, identical=True),), "b02": (SimilarBranch("b01", 1.0, identical=True),)}

    rows = {row["branch"]: row for row in _state(_view(similar_to=twins))["branches"]}

    assert rows["b01"]["similar_to"] == [{"branch": "b02", "score": 1.0, "identical": True}]
    assert rows[INTEGRATION_BRANCH]["similar_to"] == []
    assert "similar_to" in JUDGE_SYSTEM_PROMPT and "identical" in JUDGE_SYSTEM_PROMPT
    assert "stop the one of the pair with the lower expected_gain_pct" in " ".join(JUDGE_SYSTEM_PROMPT.split())


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

    assert state["cold_start"] == "unknown"
    assert '"unknown"' in JUDGE_SYSTEM_PROMPT


def test_the_prompt_version_is_derived_from_the_prompt_text() -> None:
    prompt = render_prompt(_view())
    texts = f"{JUDGE_SYSTEM_PROMPT}\n{prompt[prompt.index('Reply with') :]}"

    assert JUDGE_PROMPT_VERSION == hashlib.sha256(texts.encode()).hexdigest()[:12]


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

    assert decision == JudgeDecision(open_count=3, k=(KOrder("b01", 2),))
    assert seen == [(JUDGE_SYSTEM_PROMPT, render_prompt(_view()))]
    (row,) = [json.loads(line) for line in transcript.read_text().splitlines()]
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


def test_a_user_branch_in_its_grace_is_neither_stopped_nor_lowered_and_one_past_it_is() -> None:
    decision = JudgeDecision(
        k=(KOrder("b01", 1), KOrder("b02", 1)),
        stop=(StopOrder("b01", STOP_REASON),),
    )
    capped = cap_decision(decision, _view(roster=USER_ROSTER))

    assert hold_decision(capped, dict.fromkeys(("b01", "b02"), HoldReason.USER_GRACE), USER_ROSTER) == JudgeDecision(
        held=(HeldOrder("b01", HoldReason.USER_GRACE), HeldOrder("b02", HoldReason.USER_GRACE))
    )
    assert hold_decision(capped, {"b02": HoldReason.USER_GRACE}, USER_ROSTER) == JudgeDecision(
        stop=(StopOrder("b01", STOP_REASON),), held=(HeldOrder("b02", HoldReason.USER_GRACE),)
    )


def test_a_user_branch_in_its_grace_takes_a_higher_k_and_the_last_order_for_it_counts() -> None:
    view = _view(roster=USER_ROSTER)
    held = {"b01": HoldReason.USER_GRACE}

    assert hold_decision(cap_decision(JudgeDecision(k=(KOrder("b01", 3),)), view), held, USER_ROSTER).k == (
        KOrder("b01", 2),
    )
    lowered = cap_decision(JudgeDecision(k=(KOrder("b01", 2), KOrder("b01", 1))), view)
    assert hold_decision(lowered, held, USER_ROSTER).k == ()


def test_the_opening_call_opens_none_by_default_once_user_branches_opened() -> None:
    view = _view(
        opening=True, roster=USER_ROSTER, free_slots=2, held=dict.fromkeys(("b01", "b02"), HoldReason.USER_GRACE)
    )

    assert cap_decision(JudgeDecision(open_count=0), view).open_count == 0
    assert cap_decision(JudgeDecision(open_count=2), view).open_count == 2


def _judged(branch: str, number: int, *, judged: bool = True, kind: TestKind = TestKind.TEST) -> TestRecord:
    request_id = f"{branch}-r{number}"
    return TestRecord(
        request(request_id, intent(branch, "c-root"), kind=kind),
        result(request_id),
        judged_at=10.0 if judged else None,
    )


def test_a_user_branch_keeps_its_grace_until_enough_of_its_tests_were_judged() -> None:
    roster = (*USER_ROSTER, _entry("b03"), _entry("b04", origin=BranchOrigin.USER, stopped=True))
    early = [_judged("b01", n) for n in range(MB_USER_BRANCH_GRACE_TESTS - 1)]
    unjudged = [_judged("b01", 90, judged=False), _judged("b01", 91, kind=TestKind.PROFILE)]
    failed = TestRecord(
        request("b01-r92", intent("b01", "c-root")),
        result("b01-r92", validity=Validity.MEASUREMENT_FAILED),
        judged_at=10.0,
    )
    done = [_judged("b02", n) for n in range(MB_USER_BRANCH_GRACE_TESTS)]

    assert held_branches(roster, (), [*early, *unjudged, failed, *done], {}) == {"b01": HoldReason.USER_GRACE}


@pytest.mark.parametrize("state", [{"gave_up": True}, {"judge_failed": True}])
def test_a_user_branch_whose_specialist_is_out_of_service_has_no_grace(state: dict[str, bool]) -> None:
    status = replace(SpecialistStatus(running=False, crashes=3, lost=0, gave_up=False, judge_failed=False), **state)

    assert held_branches(USER_ROSTER, (), [], {"b01": status}) == {"b02": HoldReason.USER_GRACE}


def test_a_user_branch_row_shows_its_grace_and_the_prompt_explains_it() -> None:
    view = _view(roster=USER_ROSTER, held={"b01": HoldReason.USER_GRACE})
    rows = {row["branch"]: row for row in _state(view)["branches"]}
    plain = {row["branch"]: row for row in _state(_view())["branches"]}

    assert (rows["b01"]["origin"], rows["b01"]["grace"], rows["b02"]["grace"]) == ("user", True, False)
    assert "grace" not in plain["b01"]
    prose = " ".join(JUDGE_SYSTEM_PROMPT.split())
    assert f"until {MB_USER_BRANCH_GRACE_TESTS} of its tests were judged, a stop or a lower K you give it" in prose


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

    assert held_branches(roster, progress, [], {}) == {
        "b01": HoldReason.PENDING_CONFIRM,
        "b04": HoldReason.USER_GRACE,
    }
    assert held_branches(roster, progress, [], {"b01": failed, "b04": gave_up}) == {"b04": HoldReason.PENDING_CONFIRM}


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
