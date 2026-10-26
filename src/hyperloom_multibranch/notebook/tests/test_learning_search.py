# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""This session's learnings matched to a query: shared words, the direction's counting double, newest first."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import Learning, LearningRecord, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook.learning_search import (
    LEARNINGS_LOG,
    LEARNINGS_LOG_FORMAT,
    Retrieval,
    RetrievalLog,
    RetrievalSource,
    SessionLearnings,
    fit_lines,
    render_session_learning,
    session_matches,
    words,
)

ABOUT_KV = Learning("s2", LearningKind.PITFALL, "the cache dtype breaks accuracy", "kv cache", "fp8 kv cache")
SAYS_KV = Learning("s1", LearningKind.INSIGHT, "a smaller kv cache block helps decode")
OLDER = Learning("s1", LearningKind.INSIGHT, "a larger kv cache block hurts decode")
STALE = Learning("s3", LearningKind.INSIGHT, "kv cache fp8 is free on this model")


def _with(snapshot: NotebookSnapshot) -> NotebookSnapshot:
    replaced = dataclasses.replace(STALE, content="kv cache fp8 costs accuracy here", supersedes="s3:1")
    return dataclasses.replace(
        snapshot,
        learnings=(
            LearningRecord("s3:1", 1.0, STALE),
            LearningRecord("s1:1", 2.0, OLDER),
            LearningRecord("s2:1", 3.0, ABOUT_KV),
            LearningRecord("s1:2", 4.0, SAYS_KV),
            LearningRecord("s3:2", 5.0, replaced),
        ),
    )


def test_the_direction_counts_double_newer_wins_ties_and_superseded_learnings_are_out(
    snapshot: NotebookSnapshot,
) -> None:
    found = session_matches(_with(snapshot), "fp8 kv cache", limit=10)

    # s2:1 shares fp8, kv and cache with its direction (6) and cache with its text (7); s3:2 shares all three in its
    # text (3); s1:2 and s1:1 share kv and cache (2), the newer first; s3:1 was superseded.
    assert [record.entry_id for record in found] == ["s2:1", "s3:2", "s1:2", "s1:1"]
    assert [record.entry_id for record in session_matches(_with(snapshot), "fp8 kv cache", limit=2)] == ["s2:1", "s3:2"]
    assert session_matches(_with(snapshot), "the of on", limit=10) == ()


def test_a_server_flag_matches_a_query_for_its_parts_as_well_as_itself(snapshot: NotebookSnapshot) -> None:
    # b05 corrected this at 0.5 h of 64f775fe; b44 asked about cuda graphs at 7.7 h and was never shown it.
    corrected = Learning(
        "b05",
        LearningKind.PITFALL,
        "Correction: the failures were argparse finding '--cuda-graph-max-bs' ambiguous (it matches "
        "--cuda-graph-max-bs-decode/-prefill), not page-size. Use --cuda-graph-max-bs-decode.",
    )
    learnt = dataclasses.replace(snapshot, learnings=(LearningRecord("b05:2", 1.0, corrected),))

    assert [record.entry_id for record in session_matches(learnt, "cuda graph padding", limit=5)] == ["b05:2"]
    assert session_matches(learnt, "--cuda-graph-bs-decode", limit=5) != ()
    assert {"cuda-graph-max-bs", "cuda", "graph", "max", "bs", "page-size", "page", "size"} <= words(corrected.content)
    assert words("M=64 decode at conc 64, gate_up N=34816") == {"decode", "conc", "gate_up", "gate", "up", "34816"}


def test_a_learning_renders_with_its_direction_and_the_tests_behind_it(snapshot: NotebookSnapshot) -> None:
    cited = LearningRecord("s1:9", 1.0, dataclasses.replace(ABOUT_KV, branch="s1", request_ids=("r1", "r9")))

    assert render_session_learning(cited, snapshot) == (
        "- [s1, pitfall about kv cache / fp8 kv cache] the cache dtype breaks accuracy "
        "(r1: improved +4.00%, committed) [s1:9]"
    )


def test_whole_lines_are_kept_within_the_budget_and_a_long_one_is_skipped_not_cut() -> None:
    assert fit_lines(["a" * 5, "b" * 50, "c" * 5], 13) == ["a" * 5, "c" * 5]


async def test_each_retrieval_is_one_line_of_the_log(snapshot: NotebookSnapshot, tmp_path: Path) -> None:
    path = tmp_path / "reports" / LEARNINGS_LOG
    finder = SessionLearnings(RetrievalLog(path))

    finder.here(_with(snapshot), Retrieval("s4", RetrievalSource.CLAIM, "fp8 kv cache"), limit=2)
    await finder.lines(_with(snapshot), "speculative draft", limit=5, branch="s4")
    logged = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    common = {"format": LEARNINGS_LOG_FORMAT, "at": snapshot.taken_at, "branch": "s4", "past": [], "past_status": ""}
    assert logged == [
        {**common, "source": "claim", "query": "fp8 kv cache", "here": ["s2:1", "s3:2"]},
        {**common, "source": "tool", "query": "speculative draft", "here": []},
    ]


def test_a_log_that_cannot_be_written_holds_up_no_retrieval(
    snapshot: NotebookSnapshot, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    finder = SessionLearnings(RetrievalLog(tmp_path))

    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.notebook.learning_search"):
        found = finder.here(_with(snapshot), Retrieval("s4", RetrievalSource.TOOL, "fp8 kv cache"), limit=1)

    assert [record.entry_id for record in found] == ["s2:1"]
    assert "could not append the learnings retrieval" in caplog.text
