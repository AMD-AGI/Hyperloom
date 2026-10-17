# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The judge's contract: what it shows the LLM, how it reads the reply, and how the answer is cut to the caps."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, FrontierScore
from hyperloom_multibranch.contracts.notebook.vocab import FrontierBasis
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.judge import (
    JUDGE_SYSTEM_PROMPT,
    Judge,
    JudgeDecision,
    JudgeView,
    KOrder,
    StopOrder,
    cap_decision,
    parse_decision,
    render_prompt,
)
from hyperloom_multibranch.scheduling.llm import CompletionFailed
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    BranchStop,
    RosterCaps,
    StopCause,
)

METRIC = "output_throughput"
NOW = 1000.0
STOP_REASON = "no gain left"


def _entry(branch: str, *, kind: BranchKind = BranchKind.SPECIALIST, stopped: bool = False) -> BranchEntry:
    return BranchEntry(
        branch=branch,
        kind=kind,
        origin=BranchOrigin.ROOT if kind is BranchKind.INTEGRATION else BranchOrigin.FIRST_BATCH,
        start_commit="c-root",
        k=1,
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


def _view(
    *, opening: bool = False, roster: tuple[BranchEntry, ...] = ROSTER, free_slots: int = 2, history: str = ""
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
                head_value={METRIC: 110.0},
                frontier=FrontierScore(3.25, {"trend_pct": 2.5, "observing": 1.0}),
            ),
        ),
        integration_value={METRIC: 100.0},
        roofline_ceiling={METRIC: 400.0},
        metric=METRIC,
        opening=opening,
        free_slots=free_slots,
        history=history,
    )


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


@pytest.mark.parametrize(
    "reply",
    [
        "no json at all",
        _reply({"branches": []}),
        _reply({"open": "two"}),
        _reply({"open": True}),
        _reply({"open": 1, "branches": [{"branch": "b01", "k": 1.5}]}),
        _reply({"open": 1, "stop": [{"branch": "b01"}]}),
        _reply({"open": 1, "stop": "b01"}),
        _reply({"open": 1, "stop_session": "yes"}),
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
        open_count=0,
        k=(KOrder("b01", 2), KOrder(INTEGRATION_BRANCH, 1)),
        stop=(StopOrder("b02", "flat"),),
        session_stop="done",
    )


@pytest.mark.parametrize(("asked", "granted"), [(0, 1), (2, 2), (7, 4)])
def test_opening_count_is_clamped_to_one_and_the_free_slots(asked: int, granted: int) -> None:
    view = _view(opening=True, roster=OPENING_ROSTER, free_slots=4)

    assert cap_decision(JudgeDecision(open_count=asked), view).open_count == granted


async def test_decide_returns_the_capped_decision_and_logs_the_exchange(tmp_path: Path) -> None:
    seen: list[tuple[str, str]] = []

    async def completion(system: str, user: str) -> str:
        seen.append((system, user))
        return _reply({"open": 3, "branches": [{"branch": "b01", "k": 5}], "stop": [], "stop_session": None})

    transcript = tmp_path / "MB-judge.jsonl"
    judge = Judge(completion, transcript, Clock(now=lambda: NOW))

    decision = await judge.decide(_view())

    assert decision == JudgeDecision(open_count=0, k=(KOrder("b01", 2),))
    assert seen == [(JUDGE_SYSTEM_PROMPT, render_prompt(_view()))]
    (row,) = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert row["at"] == NOW and row["error"] == "" and row["prompt"] == seen[0][1]
    assert '"open": 3' in row["reply"]


async def test_decide_keeps_the_status_quo_when_the_call_fails(tmp_path: Path) -> None:
    async def completion(system: str, user: str) -> str:
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
    async def completion(system: str, user: str) -> str:
        return reply

    transcript = tmp_path / "MB-judge.jsonl"

    assert await Judge(completion, transcript, Clock(now=lambda: NOW)).decide(_view()) is None
    (row,) = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert row["reply"] == reply and row["error"]


async def test_an_unexpected_error_is_not_swallowed(tmp_path: Path) -> None:
    async def completion(system: str, user: str) -> str:
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
