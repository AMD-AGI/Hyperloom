# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE's frontier scorers written by the scheduler's LLM, a generation a session: where the pool stands, the prompt
the LLM reads, the answer it gives, and the step."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV, SKIPPED, TreeArchive, tree_from_snapshot
from hyperloom_multibranch.notebook.rsi.authoring import (
    AUTHORING_SYSTEM_PROMPT,
    FRONTIER_V1_EXPRESSION,
    NO_ARCHIVE,
    NO_CANDIDATES,
    NOT_ARCHIVED,
    author_scorers,
    authoring_prompt,
    feedback_for,
    parents,
    read_candidates,
    survey,
)
from hyperloom_multibranch.notebook.rsi.calibration import ranked_alike, rankings
from hyperloom_multibranch.notebook.rsi.expression import FEATURES, ExpressionScorer
from hyperloom_multibranch.notebook.rsi.frontier import FRONTIER_V1, FRONTIER_V1_LABEL
from hyperloom_multibranch.notebook.rsi.scorers import (
    MAX_PER_SESSION,
    ChampionRecord,
    PoolRecord,
    RetiredRecord,
    ScorerPool,
    ScorerSource,
)
from hyperloom_multibranch.notebook.rsi.session import FAILED, NO_PROFILE, WRITTEN, CloseStep
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, LATE_BLOOMER, branch, race, tree
from hyperloom_multibranch.notebook.tests.factories import START, profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC

PROVENANCE = {"model": "claude", "session_id": "sess"}
DOUBLE = {"label": "trend-v2", "expression": "trend_pct * 2", "rationale": "  recent gain counts double  "}
WRITTEN_AT = START - 3 * HOUR
# Each started after WRITTEN_AT, so each is evidence for an expression written then.
RACES = tuple(
    race(f"race-{n}", task=task_conditions(profile()), started_at=START - 2 * HOUR + 600.0 * n) for n in range(4)
)


def _pooled(label: str, text: str, *, at: float = WRITTEN_AT, refines: tuple[str, ...] = ()) -> PoolRecord:
    return PoolRecord(label, ScorerSource.EXPRESSION, at, text, "why", {"model": "claude"}, refines)


# On the races: the older branch first leads frontier-v1 by 1.00, the younger first trails it by as much.
OLDER = _pooled("older-v1", "age_s")
YOUNGER = _pooled("younger-v1", "1000000 - age_s")
LATE = _pooled("late-v1", "since_last_commit_s", at=START + HOUR)


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


def test_a_candidate_refines_the_parents_it_names_that_exist_and_one_new_idea_at_most_has_none() -> None:
    refined = {**DOUBLE, "parents": [FRONTIER_V1_LABEL, "nowhere", "older-v1", FRONTIER_V1_LABEL]}
    idea = {"label": "idea-v1", "expression": "commits", "rationale": "commits foretell commits"}
    unlisted = {"label": "idea-v3", "expression": "tests", "rationale": "", "parents": FRONTIER_V1_LABEL}

    authored = read_candidates(
        _answer(refined, idea, {**idea, "label": "idea-v2"}), PROVENANCE, 1.0, {FRONTIER_V1_LABEL, "older-v1"}
    )

    assert [(item.label, item.parents) for item in authored.accepted] == [
        ("trend-v2", (FRONTIER_V1_LABEL, "older-v1")),
        ("idea-v1", ()),
    ]
    assert authored.refused == ("candidate 3: a second new idea, with no parents",)
    assert read_candidates(_answer(unlisted), PROVENANCE, 1.0).refused == (
        "candidate 1: a candidate's parents are a list of labels",
    )


def test_frontier_v1_written_as_an_expression_ranks_the_branches_as_frontier_v1_does() -> None:
    a = branch("a", LATE_BLOOMER, test_times=(0.0, 1800.0, 3600.0))
    # Alike sessions, so similar archived branches back a prior.
    blooms = tuple(
        tree(
            f"bloom-{n}",
            a,
            branch("b", [(1800.0, 1.0 + n)], test_times=(0.0, 1800.0)),
            branch("c", [], opened_s=3000.0),
        )
        for n in range(4)
    )
    written = ExpressionScorer.read("frontier-v1-again", FRONTIER_V1_EXPRESSION)

    for trees in (RACES, blooms):
        assert ranked_alike(rankings(trees, written), rankings(trees, FRONTIER_V1)) == 1.0


def test_the_survey_sets_each_expression_beside_the_champion_on_the_sessions_after_it_was_written(
    tmp_path: Path,
) -> None:
    pool = ScorerPool(tmp_path)
    for record in (OLDER, YOUNGER, LATE):
        pool.register(record)

    found = survey(RACES, pool)
    standings = {item.label: item for item in found.standings.comparisons}
    pool.crown(ChampionRecord("older-v1", START, FRONTIER_V1_LABEL, "led"))
    held = survey(RACES, pool)

    assert (found.champion, found.pooled, found.retired) == (FRONTIER_V1_LABEL, (OLDER, YOUNGER, LATE), ())
    older = standings["older-v1"]
    assert (older.difference_low, older.difference_high, older.rounds, older.sessions) == (1.0, 1.0, 32, 4)
    assert standings["younger-v1"].difference_high == pytest.approx(-1.0) and standings["late-v1"].rounds == 0
    assert any(row.label == "late-v1" and row.rank_correlation is not None for row in found.hindsight.rows)
    assert [row.label for row in found.hindsight.rows] == ["current", "older-v1", "younger-v1", "late-v1"]
    assert parents(found) == (FRONTIER_V1_LABEL, "older-v1")
    assert held.champion == "older-v1" and parents(held) == ("older-v1",)
    assert [item.label for item in held.standings.comparisons] == ["current", "younger-v1", "late-v1"]


def test_the_prompt_holds_the_syntax_the_features_the_standings_the_session_the_pool_and_the_parents(
    tmp_path: Path,
) -> None:
    pool = ScorerPool(tmp_path)
    pool.register(OLDER)
    pool.register(_pooled("older-v2", "age_s + 1", refines=("older-v1",)))
    pool.retire(RetiredRecord("younger-v1", START, "behind frontier-v1 by -1.00 to -1.00", "1000000 - age_s"))
    ended = branch("s2", [], closed_s=1200.0, stop_reason="no progress")
    searched = tree("sess", branch("s1", [(600.0, 1.0)], test_times=(0.0, 600.0)), ended)
    found = survey(RACES, pool)

    prompt = authoring_prompt(searched, found)

    assert prompt.startswith("Syntax:") and all(f"- {name}: " in prompt for name in FEATURES)
    assert f"{FRONTIER_V1_LABEL}, as an expression: {FRONTIER_V1_EXPRESSION};" in prompt
    assert found.standings.render() in prompt and found.hindsight.render() in prompt
    assert "- s2: opened at 0 min from +0.00%, stopped at 20 min (no progress); 0 tests, 0 commits" in prompt
    assert "is none the pool holds: older-v1, older-v2, younger-v1." in prompt
    assert "- older-v2: age_s + 1 (refines older-v1; why)" in prompt
    assert "- younger-v1: 1000000 - age_s (behind frontier-v1 by -1.00 to -1.00)" in prompt
    assert "Parents to refine: frontier-v1, older-v1, older-v2." in prompt
    assert '"parents": ["..."]' in AUTHORING_SYSTEM_PROMPT
    assert "Whether each change helped" not in prompt


def _lineage(tmp_path: Path) -> ScorerPool:
    """A pool whose champion ranks the races' branches backwards; one refinement of each parent ranks them better than
    it, one worse, and one is too new for any race."""

    pool = ScorerPool(tmp_path)
    for record in (
        OLDER,
        _pooled("older-v2", "age_s * 2", refines=(FRONTIER_V1_LABEL,)),
        _pooled("younger-v2", "1000000 - age_s", refines=("older-v1",)),
        _pooled("late-v2", "since_last_commit_s", at=START + HOUR, refines=("older-v1",)),
    ):
        pool.register(record)
    pool.crown(ChampionRecord("younger-v2", START, FRONTIER_V1_LABEL, "led"))
    return pool


def test_the_feedback_says_whether_each_refinement_helped_and_where_each_parent_misranked_worst(tmp_path: Path) -> None:
    pool = _lineage(tmp_path)
    found = survey(RACES, pool)

    feedback = feedback_for(RACES, pool, found)

    compared = {(item.compared.label, item.parent): item.compared for item in feedback.refinements}
    assert list(compared) == [("older-v2", FRONTIER_V1_LABEL), ("younger-v2", "older-v1"), ("late-v2", "older-v1")]
    better, worse, waiting = compared.values()
    assert (better.difference_low, better.difference_high, better.rounds, better.sessions) == (1.0, 1.0, 32, 4)
    assert worse.difference_high == pytest.approx(-2.0) and waiting.rounds == 0
    assert parents(found) == ("younger-v2", "older-v1", "older-v2")
    assert [
        (item.picked.session_id, item.picked.t_s, item.picked.branch.name, item.best.branch.name)
        for item in feedback.misranked["younger-v2"]
    ] == [("race-0", 600.0, "b", "a"), ("race-1", 600.0, "b", "a"), ("race-2", 600.0, "b", "a")]
    first = feedback.misranked["younger-v2"][0]
    assert (first.picked.gain_after_pct, first.best.gain_after_pct) == (0.0, 2.0)
    assert ("age_s", 0.0, 600.0) in first.differed and "open_branches" not in {name for name, *_ in first.differed}
    assert feedback.misranked["older-v1"] == feedback.misranked["older-v2"] == () and not feedback.failures


def test_the_prompt_holds_the_feedback_before_the_parents(tmp_path: Path) -> None:
    pool = _lineage(tmp_path)
    found = survey(RACES, pool)
    searched = tree("sess", branch("s1", [(600.0, 1.0)], test_times=(0.0, 600.0)))

    prompt = authoring_prompt(searched, found, feedback_for(RACES, pool, found))

    assert "| older-v2 | frontier-v1 | 1.00 | 0.00 | 1.00 to 1.00 | 32 | 4 | better |" in prompt
    assert "| younger-v2 | older-v1 | -1.00 | 1.00 | -2.00 to -2.00 | 32 | 4 | worse |" in prompt
    assert "Too new to tell, no session like this one started after them: late-v2." in prompt
    assert (
        "- younger-v2, session race-0 at 10 min: it scored b highest (1000000.00), which gained +0.00% after; "
        in prompt
    )
    assert "a gained +2.00% (scored 999400.00). Where they differed, b / a: age_s 0.00 / 600.00, " in prompt
    assert "- older-v1: nowhere, the branch it scored highest gained the most in every round" in prompt
    assert prompt.index("Whether each change helped") < prompt.index("Parents to refine: younger-v2, older-v1")
    assert "where each parent misranked worst" in AUTHORING_SYSTEM_PROMPT


def test_a_parent_whose_rounds_cannot_be_replayed_says_why(tmp_path: Path) -> None:
    pool = ScorerPool(tmp_path)
    pool.register(_pooled("ratio-v1", "1 / age_s"))
    pool.crown(ChampionRecord("ratio-v1", START, FRONTIER_V1_LABEL, "led"))
    found = survey(RACES, pool)

    feedback = feedback_for(RACES, pool, found)

    assert feedback.failures["ratio-v1"].startswith("ExpressionError: 1 / age_s: ")
    assert "ratio-v1" not in feedback.misranked and feedback.refinements == ()
    assert "- ratio-v1: its rounds could not be replayed: ExpressionError: " in authoring_prompt(
        RACES[0], found, feedback
    )


async def test_close_writes_the_candidates_the_llm_gave_to_the_pool(snapshot: NotebookSnapshot, tmp_path: Path) -> None:
    archive = TreeArchive(tmp_path / "archive")
    archive.put(tree_from_snapshot(snapshot, METRIC))
    asked: list[tuple[str, str]] = []
    refined = {"label": "older-v1", "expression": "age_s", "rationale": "older first", "parents": [FRONTIER_V1_LABEL]}

    async def complete(system: str, prompt: str) -> str:
        asked.append((system, prompt))
        return _answer(refined, {"label": "x-v1", "expression": "speed", "rationale": "no"})

    step = await author_scorers(snapshot, complete, model="claude", archive=archive)
    (written,) = ScorerPool.beside(archive).records()

    assert snapshot.profile is not None
    assert (step.status, step.counts) == (WRITTEN, {"written": 1, "refused": 1, "retired": 0})
    assert step.reason.startswith("candidate 2: an expression may not hold the name speed")
    assert (written.label, written.registered_at, written.parents) == ("older-v1", snapshot.taken_at, ("frontier-v1",))
    assert written.provenance == {"model": "claude", "session_id": snapshot.profile.session_id}
    assert asked[0][0] == AUTHORING_SYSTEM_PROMPT and "This session's branches:" in asked[0][1]
    assert "Whether each change helped" in asked[0][1] and "Parents to refine: frontier-v1." in asked[0][1]


async def test_close_refuses_a_duplicate_and_retires_what_is_clearly_worse_than_the_champion(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    archive = TreeArchive(tmp_path / "archive")
    for searched in (*RACES, tree_from_snapshot(snapshot, METRIC)):
        archive.put(searched)
    pool = ScorerPool.beside(archive)
    for record in (OLDER, YOUNGER):
        pool.register(record)
    twin = {"label": "older-v2", "expression": "age_s * 2 + 1", "rationale": "", "parents": ["older-v1"]}
    fresh = {"label": "idle-v1", "expression": "since_last_commit_s", "rationale": "the idle branch next"}

    async def complete(system: str, prompt: str) -> str:
        return _answer(twin, fresh)

    step = await author_scorers(snapshot, complete, model="claude", archive=archive)
    (retired,) = pool.retired()

    assert (step.status, step.counts) == (WRITTEN, {"written": 1, "refused": 1, "retired": 1})
    assert step.reason == "older-v2: ranks the branches like older-v1"
    assert [item.label for item in pool.expressions()] == ["older-v1", "idle-v1"]
    assert (retired.label, retired.retired_at, retired.expression) == (
        "younger-v1",
        snapshot.taken_at,
        YOUNGER.expression,
    )
    assert retired.evidence.startswith("behind frontier-v1 by -1.00 to ")


async def test_close_writes_no_scorer_without_a_profile_an_archived_tree_or_an_answer(
    snapshot: NotebookSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ARCHIVE_ENV, raising=False)
    archive = TreeArchive(tmp_path / "archive")

    async def complete(system: str, prompt: str) -> str:
        raise ConnectionError("the gateway is down")

    async def nothing(system: str, prompt: str) -> str:
        return '{"candidates": []}'

    unprofiled = dataclasses.replace(snapshot, profile=None)
    assert await author_scorers(unprofiled, complete, model="m", archive=archive) == CloseStep(SKIPPED, NO_PROFILE)
    assert await author_scorers(snapshot, complete, model="m") == CloseStep(SKIPPED, NO_ARCHIVE)
    assert await author_scorers(snapshot, complete, model="m", archive=archive) == CloseStep(SKIPPED, NOT_ARCHIVED)
    archive.put(tree_from_snapshot(snapshot, METRIC))

    failed = await author_scorers(snapshot, complete, model="m", archive=archive)
    empty = await author_scorers(snapshot, nothing, model="m", archive=archive)

    assert (failed.status, failed.reason) == (FAILED, "ConnectionError: the gateway is down")
    assert empty == CloseStep(SKIPPED, NO_CANDIDATES, {"written": 0, "refused": 0, "retired": 0})
