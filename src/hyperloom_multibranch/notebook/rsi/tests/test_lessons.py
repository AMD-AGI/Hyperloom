# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The lesson pool: each record written once, a lesson revised by new revisions, retired, purged, and weighed by what
the sessions did after it was shown."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import LESSON_ID
from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV, TreeArchive
from hyperloom_multibranch.notebook.rsi.lessons import (
    REVISION_ID,
    LessonEvidence,
    LessonPool,
    LessonPurge,
    LessonRetirement,
    LessonRevision,
    LessonStanding,
    SessionStandings,
    human_author,
    lesson_id_for,
    main,
    retirement_proposals,
    revision_id_for,
    specialist_author,
    standing_factor,
    standing_of,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.tests.factories import START, profile

CONDITIONS = task_conditions(profile())
LESSON = lesson_id_for("sess-1", "b05:1")
TEXT = "--cuda-graph-max-bs is ambiguous here: argparse matches the -decode and -prefill forms; use those"
FIXED = "--cuda-graph-max-bs-decode takes the decode graph size; the bare flag is ambiguous and fails at startup"


def revision(name: str, at: float, *, supersedes: tuple[str, ...] = (), text: str = TEXT) -> LessonRevision:
    return LessonRevision(
        revision_id_for(name),
        LESSON,
        LearningKind.PITFALL,
        text,
        CONDITIONS,
        specialist_author("sess-1", "b05"),
        at,
        session_id="sess-1",
        target="cuda graph max bs",
        layer=IntentLayer.CONFIG,
        supersedes=supersedes,
        evidence=(LessonEvidence("sess-1", "42cc8632", "the candidate failed (no_known_signature)", measured=True),),
    )


def test_ids_are_stable_however_often_they_are_made_and_name_a_lesson_or_a_revision() -> None:
    assert lesson_id_for("sess-1", "b05:1") == LESSON != lesson_id_for("sess-1", "b05:2")
    assert LESSON_ID.fullmatch(LESSON) and REVISION_ID.fullmatch(revision_id_for("sess-1", "b05:1"))


def test_a_revision_is_written_once_reads_back_whole_and_a_later_one_takes_its_place(tmp_path: Path) -> None:
    pool = LessonPool(tmp_path)
    first = revision("first", START)
    second = revision("second", START + 60, supersedes=(first.revision_id,), text=FIXED)

    assert pool.write(first) and not pool.write(first)
    assert pool.revisions() == (first,) and pool.live() == (first,) and first.measured
    assert pool.write(second)
    assert pool.live() == (second,)
    assert pool.lessons()[LESSON].revisions == (first, second)


def test_two_revisions_of_one_head_fork_the_lesson_until_one_supersedes_both(tmp_path: Path) -> None:
    pool = LessonPool(tmp_path)
    first = revision("first", START)
    left = revision("left", START + 10, supersedes=(first.revision_id,), text=FIXED)
    right = revision("right", START + 20, supersedes=(first.revision_id,))
    merged = revision("merged", START + 30, supersedes=(left.revision_id, right.revision_id), text=FIXED)
    for item in (first, left, right):
        pool.write(item)

    forked = pool.live()
    pool.write(merged)

    assert forked == (left, right)
    assert pool.live() == (merged,)


def test_the_pool_refuses_a_malformed_revision_and_one_superseding_another_lessons(tmp_path: Path) -> None:
    pool = LessonPool(tmp_path)
    other = LessonRevision(
        revision_id_for("other"), lesson_id_for("other"), LearningKind.INSIGHT, TEXT, CONDITIONS, "human:x", START
    )
    pool.write(other)

    for bad in (
        revision("short", START, text="too short"),
        revision("stray", START, supersedes=(other.revision_id,)),
        revision("missing", START, supersedes=(revision_id_for("nowhere"),)),
        LessonRevision("rev-1", LESSON, LearningKind.PITFALL, TEXT, CONDITIONS, "human:x", START),
        LessonRevision(revision_id_for("x"), "lsn-1", LearningKind.PITFALL, TEXT, CONDITIONS, "human:x", START),
    ):
        with pytest.raises(ValueError):
            pool.write(bad)
    assert pool.revisions() == (other,)


def test_a_retired_lesson_is_live_no_more_keeps_its_revisions_and_retires_once(tmp_path: Path) -> None:
    pool = LessonPool(tmp_path)
    first = revision("first", START)
    pool.write(first)
    retirement = LessonRetirement(LESSON, human_author("siliang"), START + 5, "true only before the flag split")

    assert pool.retire(retirement) and not pool.retire(retirement)
    assert pool.live() == () and pool.retirements() == (retirement,)
    assert pool.lessons()[LESSON].retired == retirement and pool.lessons()[LESSON].revisions == (first,)


def test_a_purge_deletes_the_lessons_revisions_and_its_retirement_and_keeps_only_why(tmp_path: Path) -> None:
    pool = LessonPool(tmp_path)
    first, second = revision("first", START), revision("second", START + 1, supersedes=(revision_id_for("first"),))
    keep = LessonRevision(
        revision_id_for("keep"), lesson_id_for("keep"), LearningKind.INSIGHT, FIXED, CONDITIONS, "human:x", START
    )
    for item in (first, second, keep):
        pool.write(item)
    pool.retire(LessonRetirement(LESSON, human_author("siliang"), START + 2, "it quoted a token"))
    purge = LessonPurge(LESSON, human_author("siliang"), START + 3, "it quoted a token")

    deleted = pool.purge(purge)

    assert deleted == 2 and pool.revisions() == (keep,) and pool.retirements() == ()
    assert pool.purges() == (purge,) and LESSON not in pool.lessons()
    with pytest.raises(ValueError):
        pool.write(revision("again", START + 4))


def test_a_session_uses_the_pool_beside_the_archive_the_environment_names_by_default(tmp_path: Path) -> None:
    found = LessonPool.from_env({ARCHIVE_ENV: str(tmp_path / "archive")})

    assert found is not None and found.root == LessonPool.beside(TreeArchive(tmp_path / "archive")).root
    assert found.root == tmp_path / "archive" / "lessons"
    assert LessonPool.from_env({}) is None


def test_an_unreadable_record_is_skipped_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pool = LessonPool(tmp_path)
    pool.write(revision("first", START))
    (pool.root / "revisions" / "rev-0000000000000000.json").write_text("[1, 2]", encoding="utf-8")
    (pool.root / "revisions" / "rev-1111111111111111.json").write_text('{"format": "other"}', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        found = pool.revisions()

    assert [item.revision_id for item in found] == [revision_id_for("first")]
    assert caplog.text.count("skipping unreadable lesson record") == 2


def test_standings_are_written_once_a_session_and_weigh_a_lesson_by_what_bore_it_out(tmp_path: Path) -> None:
    pool = LessonPool(tmp_path)
    contradicted = lesson_id_for("contradicted")
    borne = lesson_id_for("borne")
    sessions = [
        SessionStandings(
            "s1", START, {contradicted: LessonStanding(shown=2, contradicted=1), borne: LessonStanding(1)}
        ),
        SessionStandings("s2", START + 1, {contradicted: LessonStanding(shown=1, contradicted=2)}),
        SessionStandings("s3", START + 2, {borne: LessonStanding(shown=3, heeded=1, ignored=1, contradicted=1)}),
        SessionStandings("s4", START + 3, {borne: LessonStanding(shown=1, contradicted=1)}),
    ]
    for item in sessions:
        assert pool.record_standings(item)
    for lesson_id in (contradicted, borne):
        pool.write(
            LessonRevision(revision_id_for(lesson_id), lesson_id, LearningKind.PITFALL, TEXT, CONDITIONS, "x", START)
        )

    total = standing_of(pool.standings(), borne)

    assert not pool.record_standings(sessions[0]) and pool.standings() == tuple(sessions)
    assert total == LessonStanding(shown=5, heeded=1, ignored=1, contradicted=2)
    assert standing_factor(LessonStanding()) == 1.0 and standing_factor(total) == pytest.approx(2 / 5)
    assert retirement_proposals(pool.lessons(), pool.standings()) == (contradicted,)
    with pytest.raises(ValueError):
        pool.record_standings(SessionStandings("../escape", START))


def test_a_person_lists_shows_revises_retires_and_purges_lessons_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ARCHIVE_ENV, raising=False)
    pool = LessonPool.beside(TreeArchive(tmp_path))
    first = revision("first", START)
    other = lesson_id_for("other")
    pool.write(first)
    pool.write(LessonRevision(revision_id_for("other"), other, LearningKind.TOOLING, FIXED, CONDITIONS, "x", START))
    archive = ["--archive", str(tmp_path)]

    listed = main([*archive, "list", "--query", "argparse"])
    printed = capsys.readouterr().out
    revised = main([*archive, "revise", LESSON, "--author", "siliang", "--content", f"  {FIXED}  "])
    shown = main([*archive, "show", LESSON])
    history = capsys.readouterr().out
    retired = main([*archive, "retire", other, "--author", "siliang", "--reason", "fixed upstream in 0.12"])
    unconfirmed = main([*archive, "purge", LESSON, "--author", "siliang", "--reason", "it quoted a token"])
    purged = main([*archive, "purge", LESSON, "--author", "siliang", "--reason", "it quoted a token", "--yes"])
    missing = main([*archive, "retire", LESSON, "--author", "siliang", "--reason", "gone already"])

    assert (listed, revised, shown, retired, unconfirmed, purged, missing) == (0, 0, 0, 0, 2, 0, 1)
    assert printed.splitlines() == [f"{LESSON} live [pitfall] sess-1: {TEXT}"]
    assert f"replaces {first.revision_id}" in history and f"  {FIXED}" in history and "standing: {" in history
    assert [head.author for head in pool.lessons()[other].heads] == ["x"] and pool.lessons()[other].retired is not None
    assert LESSON not in pool.lessons() and [item.author for item in pool.purges()] == [human_author("siliang")]
    assert main(["list"]) == 2
