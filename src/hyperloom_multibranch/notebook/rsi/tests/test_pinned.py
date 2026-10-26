# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An archive read as of a time, as repeats and paired sessions read it: the trees, lessons and playbooks of then."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV, AS_OF_ENV, TreeArchive, as_of_from_env
from hyperloom_multibranch.notebook.rsi.lessons import LessonPool, LessonRevision, lesson_id_for, revision_id_for
from hyperloom_multibranch.notebook.rsi.playbook import H0, Crowned, Playbook, PlaybookPool, replaced
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.tests.trees import branch, tree

TEXT = "fp8 kv cache needs its scales computed or the accuracy gate fails"


def _tree(name: str, started_at: float):
    return dataclasses.replace(tree(name, branch("b1", [(60.0, 1.0)])), started_at=started_at, budget_seconds=100.0)


def _lesson(name: str, at: float) -> LessonRevision:
    return LessonRevision(
        revision_id_for(name), lesson_id_for(name), LearningKind.PITFALL, TEXT, TaskConditions(), "x", at
    )


def test_an_archive_read_as_of_a_time_holds_only_what_was_written_by_then(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = TreeArchive(tmp_path)
    archive.put(_tree("early", 1000.0))
    archive.put(_tree("late", 2000.0))
    lessons = LessonPool.beside(archive)
    lessons.write(_lesson("before", 1200.0))
    lessons.write(_lesson("after", 1800.0))
    playbooks = PlaybookPool.beside(archive)
    first = playbooks.register(
        Playbook("first", replaced(H0, stall_tests=5).numbers, parent="h0", registered_at=1200.0)
    )
    playbooks.register(Playbook("second", replaced(H0, stall_tests=7).numbers, parent="h0", registered_at=1800.0))
    playbooks.crown(Crowned("first", 1900.0, "won its pairs"))
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path))
    monkeypatch.setenv(AS_OF_ENV, "1500")

    pinned = TreeArchive.from_env()
    pool, ledger = LessonPool.from_env(), PlaybookPool.from_env()

    assert pinned is not None and pool is not None and ledger is not None
    assert [item.session_id for item in pinned.load()] == ["early"]
    assert [item.revision_id for item in pool.revisions()] == [revision_id_for("before")]
    assert (ledger.playbooks(), ledger.champion()) == ((first,), H0)
    assert [item.session_id for item in archive.load()] == ["early", "late"] and playbooks.champion() == first
    monkeypatch.setenv(AS_OF_ENV, "1970-01-01T00:25:00+00:00")
    assert as_of_from_env() == 1500.0
    monkeypatch.setenv(AS_OF_ENV, "yesterday")
    with pytest.raises(ValueError):
        TreeArchive.from_env()
