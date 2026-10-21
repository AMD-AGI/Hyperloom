# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Past sessions' lessons for a branch's next decision: matched by words, weighed by how alike the sessions are and by
the lesson's standing, and never one this session already revised or retired."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import Learning, LearningRecord, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.notebook.rsi.briefing import CrossSessionBriefing
from hyperloom_multibranch.notebook.rsi.lesson_recall import CLAIM_LESSONS_HEADING, LessonRecall, edited_lessons
from hyperloom_multibranch.notebook.rsi.lessons import (
    LessonEvidence,
    LessonPool,
    LessonRetirement,
    LessonRevision,
    LessonStanding,
    SessionStandings,
    human_author,
    lesson_id_for,
    revision_id_for,
)
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions, task_conditions
from hyperloom_multibranch.notebook.tests.factories import START, FakeClock, profile
from hyperloom_multibranch.notebook.tests.sessions import scripted_session

SAME = task_conditions(profile())
CORRECTION = "the bare --cuda-graph-max-bs is ambiguous; --cuda-graph-max-bs-decode sets the decode graph size"
MEASURED = (LessonEvidence("past-1", "42cc8632", "the candidate failed (no_known_signature)", measured=True),)


def lesson(
    pool: LessonPool,
    name: str,
    text: str,
    conditions: TaskConditions = SAME,
    session: str = "past-1",
    evidence: tuple[LessonEvidence, ...] = (),
) -> str:
    revision = LessonRevision(
        revision_id_for(name),
        lesson_id_for(name),
        LearningKind.PITFALL,
        text,
        conditions,
        "x",
        START - 60.0,
        session_id=session,
        evidence=evidence,
    )
    pool.write(revision)
    return revision.lesson_id


@pytest.fixture
def pool(tmp_path: Path) -> LessonPool:
    return LessonPool(tmp_path / "pool")


def test_a_lesson_bears_on_a_query_by_its_words_from_a_session_alike_enough_to_teach_this_one(
    snapshot: NotebookSnapshot, pool: LessonPool
) -> None:
    graph = lesson(pool, "graph", CORRECTION, evidence=MEASURED)
    split = lesson(pool, "split", "attention split kv pays only past conc 128", task_conditions(profile(tp=4)))
    lesson(pool, "agentx", "cuda graph capture fails on agentx traces", task_conditions(profile(isl=None, osl=None)))
    lesson(pool, "mine", "cuda graph padding is free at conc 64", session="sess-1")
    gone = lesson(pool, "gone", "cuda graph capture needs eager prefill here")
    pool.retire(LessonRetirement(gone, human_author("siliang"), START, "fixed upstream"))
    recall = LessonRecall(pool=pool)

    found = recall.past_for(snapshot, "cuda graph padding", limit=5)
    split_kv = recall.past_for(snapshot, "attention split kv", limit=5)

    assert found is not None and [item.render() for item in found.items] == [
        f"- [past-1, same conditions, measured] {CORRECTION} [{graph}]"
    ]
    assert split_kv is not None and [item.render() for item in split_kv.items] == [
        f"- [past-1, tp differ, unmeasured] attention split kv pays only past conc 128 [{split}]"
    ]
    assert (recall.knows(graph), recall.knows(gone), recall.knows(lesson_id_for("nowhere"))) == (True, False, False)
    assert LessonRecall().past_for(snapshot, "cuda graph", limit=5) is None
    assert recall.past_for(dataclasses.replace(snapshot, profile=None), "cuda graph", limit=5) is None


def test_a_lesson_this_session_revised_or_retired_is_shown_no_more_and_the_standing_weighs_the_rest(
    snapshot: NotebookSnapshot, pool: LessonPool
) -> None:
    revised = lesson(pool, "revised", "cuda graph padding costs two percent at conc 64")
    retired = lesson(pool, "retired", "cuda graph capture fails with chunked prefill")
    doubted = lesson(pool, "doubted", "cuda graph capture fails above batch size 64")
    trusted = lesson(pool, "trusted", "cuda graph capture fails below batch size 8")
    pool.record_standings(SessionStandings("past-2", START, {doubted: LessonStanding(shown=2, contradicted=2)}))
    edits = (
        Learning(
            "s1", LearningKind.PITFALL, "padding costs nothing at conc 64", request_ids=("r1",), supersedes=revised
        ),
        Learning(
            "s1", LearningKind.OTHER, "it captures fine with chunked prefill", request_ids=("r1",), retires=retired
        ),
    )
    edited = dataclasses.replace(
        snapshot, learnings=tuple(LearningRecord(f"s1:{n}", START, item) for n, item in enumerate(edits, 1))
    )

    found = LessonRecall(pool=pool).past_for(edited, "cuda graph capture fails batch size", limit=5)

    assert edited_lessons(edited) == {revised, retired}
    assert found is not None and [item.experience_id for item in found.items] == [trusted, doubted]


async def test_a_claim_lists_this_sessions_learnings_then_past_lessons_and_logs_both_at_once(
    tmp_path: Path, pool: LessonPool
) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    snapshot = await notebook.snapshot()
    graph = lesson(pool, "graph", CORRECTION, evidence=MEASURED)
    log = tmp_path / LEARNINGS_LOG
    recall = LessonRecall(pool=pool, log=RetrievalLog(log))

    listed = await recall.about(snapshot, "cuda graph max batch size", branch="s1")
    told = await recall.lines(snapshot, "cuda graph", limit=5, branch="s1")
    (claimed, asked) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    assert listed == [CLAIM_LESSONS_HEADING, f"- [past-1, same conditions, measured] {CORRECTION} [{graph}]"]
    assert told == ["Past sessions' learnings:", f"- [past-1, same conditions, measured] {CORRECTION} [{graph}]"]
    assert (claimed["source"], claimed["here"], claimed["past"], claimed["past_status"]) == (
        "claim",
        [],
        [graph],
        "completed",
    )
    assert (asked["source"], asked["past"]) == ("tool", [graph])


async def test_the_brief_shows_past_lessons_as_learnt_before(tmp_path: Path, pool: LessonPool) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    graph = lesson(pool, "graph", CORRECTION, evidence=MEASURED)

    briefing = CrossSessionBriefing(notebook, focus="cuda graph", branch="s3", learnings=LessonRecall(pool=pool))
    brief = await briefing.startup_brief("c0")

    assert f"learnt before: [past-1, same conditions, measured] {CORRECTION} [{graph}]" in brief.lessons
