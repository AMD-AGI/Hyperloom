# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE's curator: the lessons offered, the steps the LLM may take checked one by one, and what the pool holds after."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind
from hyperloom_multibranch.blackboard.rsi.archive import NO_ARCHIVE, SKIPPED, TreeArchive
from hyperloom_multibranch.blackboard.rsi.lesson_curation import (
    CURATION_SYSTEM_PROMPT,
    NO_LESSONS,
    curate_lessons,
    curation_prompt,
    offer,
)
from hyperloom_multibranch.blackboard.rsi.lessons import (
    LessonPool,
    LessonRevision,
    LessonStanding,
    SessionStandings,
    curator_author,
    lesson_id_for,
    revision_id_for,
)
from hyperloom_multibranch.blackboard.rsi.session import FAILED, NO_PROFILE, WRITTEN, CloseStep
from hyperloom_multibranch.blackboard.rsi.similarity import TaskConditions, task_conditions
from hyperloom_multibranch.blackboard.tests.factories import START, profile

SAME = task_conditions(profile())
UNLIKE = task_conditions(
    profile(
        model="Qwen/Qwen3-8B",
        model_type="qwen3",
        architectures=("Qwen3ForCausalLM",),
        hardware="H100",
        framework="sglang",
        precision="bf16",
        tp=1,
        conc=4,
    )
)


def lesson(pool: LessonPool, name: str, text: str, session: str = "past-1", conditions: TaskConditions = SAME) -> str:
    pool.write(
        LessonRevision(
            revision_id_for(name), lesson_id_for(name), LearningKind.PITFALL, text, conditions, "x", START, session
        )
    )
    return lesson_id_for(name)


class _Asked:
    def __init__(self, steps: list[dict[str, object]]) -> None:
        self.steps = steps
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "Here you are:\n" + json.dumps({"actions": self.steps})


@pytest.fixture
def pool(tmp_path: Path) -> LessonPool:
    return LessonPool(tmp_path / "pool")


async def test_the_curator_merges_rewrites_and_retires_what_it_may_and_says_why_it_refused_the_rest(
    snapshot: BlackboardSnapshot, pool: LessonPool
) -> None:
    kept = lesson(pool, "kept", "the bare --cuda-graph-max-bs is ambiguous here; use the -decode form", "sess-1")
    twin = lesson(pool, "twin", "the bare --cuda-graph-max-bs is ambiguous here, so use its -decode form")
    elsewhere = lesson(pool, "elsewhere", "the bare --cuda-graph-max-bs is ambiguous; use -decode", conditions=UNLIKE)
    vague = lesson(pool, "vague", "fp8 kv cache is sometimes slower and sometimes faster here", "sess-1")
    wrong = lesson(pool, "wrong", "split kv always regresses attention on this model")
    kept_alive = lesson(pool, "alive", "chunked prefill 4096 costs ttft on this model", "sess-1")
    for session in ("p2", "p3"):
        pool.record_standings(SessionStandings(session, START, {wrong: LessonStanding(shown=1, contradicted=1)}))
    pool.record_standings(SessionStandings("sess-1", START + 1, {elsewhere: LessonStanding(shown=1)}))
    merged_text = "--cuda-graph-max-bs alone is ambiguous here (it matches -decode and -prefill); use the -decode form"
    asked = _Asked(
        [
            {"action": "merge", "lesson": kept, "merge": [twin], "content": merged_text, "why": "the same pitfall"},
            {"action": "merge", "lesson": vague, "merge": [elsewhere], "content": merged_text, "why": "x"},
            {"action": "rewrite", "lesson": vague, "content": "fp8 kv cache is within noise here", "why": "clearer"},
            {"action": "retire", "lesson": wrong, "why": "two sessions gained there"},
            {"action": "retire", "lesson": kept_alive, "why": "not needed"},
            {"action": "rewrite", "lesson": lesson_id_for("nowhere"), "content": merged_text},
            {"action": "rewrite", "lesson": kept_alive, "content": "too short"},
            {"action": "rewrite", "lesson": twin, "content": merged_text},
            {"action": "promote", "lesson": kept},
        ]
    )

    done = await curate_lessons(snapshot, asked, model="claude", pool=pool)
    lessons = pool.lessons()

    assert (done.status, done.counts) == (WRITTEN, {"refused": 6, "merged": 1, "rewritten": 1, "retired": 1})
    assert [head.content for head in lessons[kept].heads] == [merged_text]
    assert lessons[kept].heads[0].author == curator_author("sess-1", "claude")
    assert lessons[twin].retired is not None and lessons[twin].retired.reason == f"merged into {kept}: the same pitfall"
    assert [head.content for head in lessons[vague].heads] == ["fp8 kv cache is within noise here"]
    assert lessons[wrong].retired is not None and lessons[kept_alive].retired is None
    assert lessons[elsewhere].retired is None
    for why in (
        "too unlike",
        "neither proposed nor contradicted",
        "not offered",
        "a lesson takes one step",
        "a text of 9 characters",
        "not a step",
    ):
        assert why in done.reason
    assert asked.calls[0][0] == CURATION_SYSTEM_PROMPT
    assert f"- {wrong} [pitfall] shown to 2 branches" in asked.calls[0][1] and "retirable" in asked.calls[0][1]

    again = await curate_lessons(snapshot, asked, model="claude", pool=pool)
    assert again.status == SKIPPED and "merged" not in again.counts


async def test_the_curator_skips_without_a_profile_a_pool_or_a_lesson_and_fails_soft_when_the_llm_does(
    snapshot: BlackboardSnapshot, pool: LessonPool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def broken(_system: str, _user: str) -> str:
        raise RuntimeError("the gateway is down")

    monkeypatch.delenv("HYPERLOOM_RSI_ARCHIVE", raising=False)
    no_profile = await curate_lessons(dataclasses.replace(snapshot, profile=None), broken, model="m", pool=pool)
    no_pool = await curate_lessons(snapshot, broken, model="m")
    empty = await curate_lessons(snapshot, broken, model="m", archive=TreeArchive(tmp_path / "empty"))
    lesson(pool, "mine", "the bare --cuda-graph-max-bs is ambiguous here; use the -decode form", "sess-1")
    failed = await curate_lessons(snapshot, broken, model="m", pool=pool)

    assert (no_profile, no_pool, empty) == (
        CloseStep(SKIPPED, NO_PROFILE),
        CloseStep(SKIPPED, NO_ARCHIVE),
        CloseStep(SKIPPED, NO_LESSONS),
    )
    assert failed == CloseStep(FAILED, "RuntimeError: the gateway is down")


def test_the_most_pressing_lessons_are_offered_first_and_only_live_ones(
    snapshot: BlackboardSnapshot, pool: LessonPool
) -> None:
    shown = lesson(pool, "shown", "attention split kv regresses when attention is not the bottleneck")
    proposed = lesson(pool, "proposed", "the moe gemm never pays from a tuned kernel here")
    mine = lesson(pool, "mine", "max-num-seqs 256 lifts decode at conc 64 on this model", "sess-1")
    for session in ("p2", "p3"):
        pool.record_standings(SessionStandings(session, START, {proposed: LessonStanding(shown=1, contradicted=1)}))
    pool.record_standings(SessionStandings("sess-1", START + 1, {shown: LessonStanding(shown=1, contradicted=1)}))

    found = offer("sess-1", pool.lessons(), pool.standings())

    assert list(found.lessons) == [proposed, mine, shown]
    assert found.retirable == {proposed, shown}
    assert mine in curation_prompt(snapshot, found)
