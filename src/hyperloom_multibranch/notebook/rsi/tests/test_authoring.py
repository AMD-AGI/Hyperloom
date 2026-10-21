# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE's frontier scorers written by the scheduler's LLM: the prompt it reads, the answer it gives, and the step."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV, SKIPPED, TreeArchive, tree_from_snapshot
from hyperloom_multibranch.notebook.rsi.authoring import (
    AUTHORING_SYSTEM_PROMPT,
    NO_ARCHIVE,
    NOT_ARCHIVED,
    author_scorers,
    authoring_prompt,
    read_candidates,
)
from hyperloom_multibranch.notebook.rsi.calibration import calibrate
from hyperloom_multibranch.notebook.rsi.expression import FEATURES
from hyperloom_multibranch.notebook.rsi.scorers import MAX_PER_SESSION, PoolRecord, ScorerPool, ScorerSource
from hyperloom_multibranch.notebook.rsi.session import FAILED, NO_PROFILE, WRITTEN, CloseStep
from hyperloom_multibranch.notebook.rsi.tests.trees import branch, tree
from hyperloom_multibranch.notebook.tests.sessions import METRIC

PROVENANCE = {"model": "claude", "session_id": "sess"}
DOUBLE = {"label": "trend-v2", "expression": "trend_pct * 2", "rationale": "  recent gain counts double  "}


def _answer(*candidates: object) -> str:
    return "Here they are:\n" + json.dumps({"candidates": list(candidates)}) + "\nThat is all."


def test_an_answer_gives_pool_records_and_says_why_it_refused_the_rest() -> None:
    unlabelled = {"label": "Bad Label", "expression": "1", "rationale": ""}
    authored = read_candidates(_answer(DOUBLE, dict(DOUBLE), unlabelled, DOUBLE), PROVENANCE, 100.0)

    expected = PoolRecord(
        "trend-v2", ScorerSource.EXPRESSION, 100.0, "trend_pct * 2", "recent gain counts double", PROVENANCE
    )
    assert authored.accepted == (expected,)
    assert authored.refused[:2] == (f"1 past the first {MAX_PER_SESSION}", "candidate 2: the label trend-v2 twice")
    assert authored.refused[2].startswith("candidate 3: 'Bad Label' is no scorer label")
    assert read_candidates("no JSON here", PROVENANCE, 1.0).refused == (
        "the answer holds no JSON object with a list of candidates",
    )
    assert read_candidates(
        '{"candidates": [1, {"label": "x-v1", "expression": "speed", "rationale": ""}]}', PROVENANCE, 1.0
    ).refused == (
        "candidate 1: a candidate is an object with a label, an expression and a rationale",
        "candidate 2: an expression may not hold the name speed, which is no feature",
    )


def test_the_prompt_holds_the_syntax_the_features_the_calibration_the_session_and_the_pool() -> None:
    ended = branch("s2", [], closed_s=1200.0, stop_reason="no progress")
    searched = tree("sess", branch("s1", [(600.0, 1.0)], test_times=(0.0, 600.0)), ended)
    pooled = (PoolRecord("trend-v2", ScorerSource.EXPRESSION, 5.0, "trend_pct * 2", "recent gain counts double"),)
    report = calibrate((searched,))

    prompt = authoring_prompt(searched, report, pooled)

    assert prompt.startswith("Syntax:") and all(f"- {name}: " in prompt for name in FEATURES)
    assert report.render() in prompt
    assert "- s2: opened at 0 min from +0.00%, stopped at 20 min (no progress); 0 tests, 0 commits" in prompt
    assert "is none the pool holds: trend-v2." in prompt
    assert "- trend-v2: trend_pct * 2 (recent gain counts double)" in prompt
    assert '{"candidates": [{"label":' in AUTHORING_SYSTEM_PROMPT


async def test_close_writes_the_candidates_the_llm_gave_to_the_pool(snapshot: NotebookSnapshot, tmp_path: Path) -> None:
    archive = TreeArchive(tmp_path / "archive")
    archive.put(tree_from_snapshot(snapshot, METRIC))
    asked: list[tuple[str, str]] = []

    async def complete(system: str, prompt: str) -> str:
        asked.append((system, prompt))
        return _answer(DOUBLE, {"label": "x-v1", "expression": "speed", "rationale": "no"})

    step = await author_scorers(snapshot, complete, model="claude", archive=archive)
    (written,) = ScorerPool.beside(archive).records()

    assert snapshot.profile is not None
    assert (step.status, step.counts) == (WRITTEN, {"written": 1, "refused": 1})
    assert step.reason.startswith("candidate 2: an expression may not hold the name speed")
    assert (written.label, written.registered_at) == ("trend-v2", snapshot.taken_at)
    assert written.provenance == {"model": "claude", "session_id": snapshot.profile.session_id}
    assert asked[0][0] == AUTHORING_SYSTEM_PROMPT and "This session's branches:" in asked[0][1]


async def test_close_writes_no_scorer_without_a_profile_an_archived_tree_or_an_answer(
    snapshot: NotebookSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ARCHIVE_ENV, raising=False)
    archive = TreeArchive(tmp_path / "archive")

    async def complete(system: str, prompt: str) -> str:
        raise ConnectionError("the gateway is down")

    unprofiled = dataclasses.replace(snapshot, profile=None)
    assert await author_scorers(unprofiled, complete, model="m", archive=archive) == CloseStep(SKIPPED, NO_PROFILE)
    assert await author_scorers(snapshot, complete, model="m") == CloseStep(SKIPPED, NO_ARCHIVE)
    assert await author_scorers(snapshot, complete, model="m", archive=archive) == CloseStep(SKIPPED, NOT_ARCHIVED)
    archive.put(tree_from_snapshot(snapshot, METRIC))

    failed = await author_scorers(snapshot, complete, model="m", archive=archive)

    assert (failed.status, failed.reason) == (FAILED, "ConnectionError: the gateway is down")
