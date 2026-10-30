# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE's lesson steps: the session's learnings into the pool, and the standing of the pooled lessons it showed."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard import NotRecorded
from hyperloom_multibranch.contracts.blackboard.records import Learning, LearningRecord, BlackboardSnapshot
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind
from hyperloom_multibranch.blackboard.rsi.lesson_close import (
    CAPPED,
    DUPLICATE,
    MAX_NEW_LESSONS,
    NEW,
    RETIRED,
    REVISED,
    publish_lessons,
    session_standings,
)
from hyperloom_multibranch.blackboard.rsi.lessons import (
    LessonEvidence,
    LessonPool,
    LessonRetirement,
    LessonRevision,
    LessonStanding,
    human_author,
    lesson_id_for,
    revision_id_for,
    specialist_author,
)
from hyperloom_multibranch.blackboard.rsi.similarity import task_conditions
from hyperloom_multibranch.blackboard.tests.factories import START, profile

PAST = "past-1"
CONDITIONS = task_conditions(profile())


def pooled(pool: LessonPool, name: str, text: str, kind: LearningKind = LearningKind.PITFALL, **about: str) -> str:
    """A lesson a past session left in ``pool``; its id."""

    pool.write(
        LessonRevision(
            revision_id_for(PAST, name),
            lesson_id_for(PAST, name),
            kind,
            text,
            CONDITIONS,
            specialist_author(PAST, "b01"),
            START - 3600.0,
            session_id=PAST,
            **about,
        )
    )
    return lesson_id_for(PAST, name)


def learnt(entry_id: str, kind: LearningKind, content: str, **fields: object) -> LearningRecord:
    return LearningRecord(entry_id, START + 100.0, Learning(entry_id.split(":")[0], kind, content, **fields))


def heads(pool: LessonPool, lesson_id: str) -> list[str]:
    return [head.content for head in pool.lessons()[lesson_id].heads]


def test_close_publishes_new_lessons_revises_and_retires_the_ones_named_and_skips_what_is_said(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    pool = LessonPool(tmp_path)
    corrected = pooled(pool, "x", "page size 64 is unsupported by the aiter attention backend here")
    chained = pooled(pool, "y", "the decode graph pads every batch to the next captured size here")
    wrong = pooled(pool, "z", "chunked prefill always kills the server during warmup on this image")
    said = pooled(pool, "w", "attention split kv regresses when attention is not the bottleneck")
    session = dataclasses.replace(
        snapshot,
        learnings=(
            learnt("s1:1", LearningKind.INSIGHT, "max-num-seqs 256 lifts decode at conc 64 here", request_ids=("r1",)),
            learnt("s2:1", LearningKind.PITFALL, "the bare flag is ambiguous, not page size", supersedes=corrected),
            learnt(
                "s2:2", LearningKind.INSIGHT, "padding to captured sizes costs little at conc 64", supersedes=chained
            ),
            learnt("s2:3", LearningKind.INSIGHT, "padding costs nothing at conc 64: bs sits at 64", supersedes="s2:2"),
            learnt("s1:2", LearningKind.OTHER, "it only fails with the old warmup script", retires=wrong),
            learnt("s1:3", LearningKind.PITFALL, "chunked prefill fails with the old warmup script", supersedes="s1:2"),
            learnt("s2:4", LearningKind.PITFALL, "Attention split KV  regresses when attention is not the bottleneck"),
        ),
    )

    published = publish_lessons(session, pool)
    again = publish_lessons(session, pool)

    assert published.counts == {NEW: 2, REVISED: 2, RETIRED: 1, DUPLICATE: 1} and published.written == 5
    assert heads(pool, corrected) == ["the bare flag is ambiguous, not page size"]
    assert heads(pool, chained) == ["padding costs nothing at conc 64: bs sits at 64"]
    assert pool.lessons()[wrong].retired == LessonRetirement(
        wrong, specialist_author("sess-1", "s1"), START + 100.0, "it only fails with the old warmup script"
    )
    assert heads(pool, said) == ["attention split kv regresses when attention is not the bottleneck"]
    new = pool.lessons()[lesson_id_for("sess-1", "s1:1")].heads[0]
    assert (new.session_id, new.conditions, new.author) == ("sess-1", CONDITIONS, specialist_author("sess-1", "s1"))
    assert new.evidence == (LessonEvidence("sess-1", "r1", "improved +4.00%, committed", measured=True),)
    assert lesson_id_for("sess-1", "s1:3") in pool.lessons()
    assert again.written == 0 and again.counts == {DUPLICATE: 5}


def test_a_revision_of_a_lesson_no_longer_live_begins_a_lesson_of_its_own(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    pool = LessonPool(tmp_path)
    gone = pooled(pool, "gone", "fp8 kv cache always fails the accuracy gate on this model")
    pool.retire(LessonRetirement(gone, human_author("siliang"), START, "true only before the scales fix"))
    session = dataclasses.replace(
        snapshot,
        learnings=(
            learnt("s2:1", LearningKind.PITFALL, "fp8 kv cache passes once kv scales are computed", supersedes=gone),
        ),
    )

    published = publish_lessons(session, pool)

    assert published.counts == {NEW: 1}
    assert pool.lessons()[lesson_id_for("sess-1", "s2:1")].heads[0].supersedes == ()
    with pytest.raises(NotRecorded):
        publish_lessons(dataclasses.replace(snapshot, profile=None), pool)


def test_the_standing_of_a_shown_lesson_is_what_the_branchs_tests_of_its_family_did_next(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    pool = LessonPool(tmp_path)
    # s2's split kv test r2 failed; s1's --max-num-seqs tests are scheduling changes, r1 committed and r4 did not.
    borne = pooled(pool, "p1", "attention split kv regresses on this model", target="attention", mechanism="split kv")
    wrong = pooled(pool, "p2", "raising --max-num-seqs never pays here", target="scheduler", mechanism="--max-num-seqs")
    untested = pooled(pool, "p3", "fp8 kv cache breaks accuracy here", target="kv cache", mechanism="fp8 kv cache")
    paid = pooled(
        pool, "p4", "raise --max-num-seqs first", LearningKind.INSIGHT, target="scheduler", mechanism="--max-num-seqs"
    )
    undirected = pooled(pool, "p5", "the profiler needs record shapes on this image", LearningKind.TOOLING)
    found_again = pooled(pool, "p6", "chunked prefill makes the server exit during warmup on this image")
    session = dataclasses.replace(
        snapshot,
        learnings=(learnt("s2:1", LearningKind.PITFALL, "the server exits during warmup with chunked prefill"),),
    )
    log = [
        {"at": START - 1.0, "branch": "s2", "source": "brief", "query": "x", "here": [], "past": [borne, undirected]},
        {"at": START - 1.0, "branch": "s1", "source": "claim", "query": "y", "here": [], "past": [wrong, untested]},
        {"at": START - 1.0, "branch": "s1", "source": "tool", "query": "z", "here": [], "past": [paid, "exp-kb-1"]},
    ]

    standings = session_standings(session, pool, log)

    assert (standings.session_id, standings.written_at) == ("sess-1", snapshot.taken_at)
    assert standings.lessons == {
        borne: LessonStanding(shown=1, borne_out=1),
        undirected: LessonStanding(shown=1),
        wrong: LessonStanding(shown=1, contradicted=1),
        untested: LessonStanding(shown=1),
        paid: LessonStanding(shown=1, borne_out=1),
        found_again: LessonStanding(rediscovered=1),
    }


def test_a_session_starts_at_most_so_many_lessons_and_skips_what_one_already_says_in_other_words(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    pool = LessonPool(tmp_path)
    pooled(pool, "x", "fp8 kv cache breaks the accuracy gate on this model")
    near = learnt("s2:1", LearningKind.PITFALL, "on this model fp8 kv cache breaks the accuracy gate badly")
    many = [
        learnt(f"s1:{n}", LearningKind.INSIGHT, f"knob{n}a sets alpha{n}b beta{n}c gamma{n}d for decode")
        for n in range(1, MAX_NEW_LESSONS + 3)
    ]

    published = publish_lessons(dataclasses.replace(snapshot, learnings=(near, *many)), pool)

    assert published.counts == {DUPLICATE: 1, NEW: MAX_NEW_LESSONS, CAPPED: 2}
