# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE's knowledge in one call, and one session's frontier scoring and history digest for the judge."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from hyperloom_kb import RemoteExperienceKB
from hyperloom_kb.collect import load_mapping
from hyperloom_kb.collect.engine import CollectReport, FailedUnit

from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import Learning, LearningRecord, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook.knowledge import experience, learnings
from hyperloom_multibranch.notebook.knowledge.breakdown import CloseOutcome, build_breakdown
from hyperloom_multibranch.notebook.knowledge.tests.kb_service import experience_service
from hyperloom_multibranch.notebook.rsi import session
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV, ARCHIVED, NO_ARCHIVE, SKIPPED, TreeArchive
from hyperloom_multibranch.notebook.rsi.authoring import session_challenger
from hyperloom_multibranch.notebook.rsi.bandit import BANDIT_LABEL, BanditScorer
from hyperloom_multibranch.notebook.rsi.calibration import DEFAULT_POLICY, PolicyChoice
from hyperloom_multibranch.notebook.rsi.expression import ExpressionScorer
from hyperloom_multibranch.notebook.rsi.frontier import FRONTIER_V1, FRONTIER_V1_LABEL, FrontierPolicy, FrontierScoring
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, Retrieval, RetrievalLog, RetrievalSource
from hyperloom_multibranch.notebook.rsi.history import MAX_CHARS, session_stall_odds
from hyperloom_multibranch.notebook.rsi.lessons import (
    LessonPool,
    LessonRevision,
    LessonStanding,
    lesson_id_for,
    revision_id_for,
)
from hyperloom_multibranch.notebook.rsi.session import (
    ALREADY_RECORDED,
    CALIBRATION_REPORT,
    DISABLED,
    EXPERIENCE_RECEIPT,
    FAILED,
    FRONTIER_LOG,
    LEARNING_RECEIPT,
    MULTIBRANCH_RECEIPT,
    NO_KB,
    NO_PROFILE,
    NO_REPORTS,
    NO_RETRIEVAL_LOG,
    WRITTEN,
    CloseStep,
    CloseTargets,
    KnowledgeClose,
    close_knowledge,
    judge_history,
    named_frontier_choice,
    session_frontier_policy,
    session_frontier_scorer,
)
from hyperloom_multibranch.notebook.rsi.scorers import (
    ChampionRecord,
    PoolRecord,
    ScorerPool,
    ScorerSource,
    current_champion,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, branch, race, tree
from hyperloom_multibranch.notebook.tests.factories import START, FakeClock, profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC, scripted_session

KB_URL = "HYPERLOOM_KB_URL"
BUDGET = 2 * 3600.0
PAST = tree("past", branch("s1", [(1800.0, 3.0)], test_times=(0.0, 900.0, 1800.0)), task=task_conditions(profile()))
LEARNT = (
    LearningRecord(
        "s1:1",
        START + 10,
        Learning("s1", LearningKind.INSIGHT, "max-num-seqs 256 lifts decode at conc 64", "scheduler", "max num seqs"),
    ),
    LearningRecord(
        "s2:1", START + 20, Learning("s2", LearningKind.PITFALL, "split kv regresses here", layer=IntentLayer.CONFIG)
    ),
)


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KB_URL, raising=False)
    monkeypatch.delenv(ARCHIVE_ENV, raising=False)


@pytest.mark.usefixtures("unconfigured")
def test_without_a_kb_or_an_archive_each_step_says_why_and_close_goes_on(snapshot: NotebookSnapshot) -> None:
    done = close_knowledge(snapshot, METRIC)

    assert done == KnowledgeClose(
        experiences=CloseStep(DISABLED, NO_KB),
        learnings=CloseStep(DISABLED, NO_KB),
        archive=CloseStep(SKIPPED, "no_archive_configured"),
        multibranch_experiences=CloseStep(DISABLED, NO_KB),
    )


def test_a_live_kb_and_an_archive_take_the_experiences_the_learnings_and_the_tree(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    learnt = dataclasses.replace(snapshot, learnings=LEARNT)
    recipes = load_mapping(experience.MAPPING_PATH).declaration
    schema = load_mapping(learnings.MAPPING_PATH).declaration
    reports, archive = tmp_path / "reports", TreeArchive(tmp_path / "archive")
    with experience_service(tmp_path, recipes) as client:
        done = close_knowledge(
            learnt,
            METRIC,
            budget_seconds=BUDGET,
            targets=CloseTargets(
                experience_kb=RemoteExperienceKB(client, recipes),
                learning_kb=RemoteExperienceKB(client, schema),
                archive=archive,
                receipts=reports,
            ),
        )

    dry = experience.publish_experiences(learnt, METRIC, dry_run=True)
    assert done.experiences == CloseStep(
        WRITTEN, counts={"created": len(dry.collected), "skipped": len(dry.skipped), "errors": 0}
    )
    assert done.learnings == CloseStep(WRITTEN, counts={"created": 2, "skipped": 0, "errors": 0})
    assert done.archive == CloseStep(ARCHIVED)
    assert [(item.session_id, item.budget_seconds) for item in archive.load()] == [("sess-1", BUDGET)]
    assert (reports / EXPERIENCE_RECEIPT).is_file() and (reports / LEARNING_RECEIPT).is_file()
    assert done.multibranch_experiences == CloseStep(DISABLED, NO_KB)


@pytest.mark.usefixtures("unconfigured")
def test_close_writes_the_frontier_calibration_over_the_archived_sessions_like_this_one(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    reports = tmp_path / "reports"

    done = close_knowledge(snapshot, METRIC, targets=CloseTargets(archive=TreeArchive(tmp_path), receipts=reports))

    assert done.archive.status == ARCHIVED
    assert done.calibration.status == WRITTEN and done.calibration.reason.startswith("Too little evidence")
    assert (reports / CALIBRATION_REPORT).read_text(encoding="utf-8").startswith("Frontier calibration")


@pytest.mark.usefixtures("unconfigured")
def test_an_agentx_sessions_experiences_go_to_the_notebooks_own_declaration(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    agentx = dataclasses.replace(snapshot, profile=profile(isl=None, osl=None))
    own = load_mapping(experience.MULTIBRANCH_MAPPING_PATH).declaration
    reports = tmp_path / "reports"
    with experience_service(tmp_path, own) as client:
        done = close_knowledge(
            agentx, METRIC, targets=CloseTargets(multibranch_kb=RemoteExperienceKB(client, own), receipts=reports)
        )

    dry = experience.publish_multibranch_experiences(agentx, METRIC, dry_run=True)
    assert done.multibranch_experiences == CloseStep(
        WRITTEN, counts={"created": len(dry.collected), "skipped": len(dry.skipped), "errors": 0}
    )
    assert dry.collected and (reports / MULTIBRANCH_RECEIPT).is_file()
    assert done.experiences == CloseStep(DISABLED, NO_KB)


@pytest.mark.usefixtures("unconfigured")
def test_close_pools_the_learnings_beside_the_archive_and_records_the_shown_lessons_standing_once(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    pool, reports = LessonPool(tmp_path), tmp_path / "reports"
    earlier = LessonRevision(
        revision_id_for("past-1", "b02:1"),
        lesson_id_for("past-1", "b02:1"),
        LearningKind.PITFALL,
        "attention split kv regresses when attention is not the bottleneck",
        task_conditions(profile()),
        "specialist:past-1/b02",
        START - HOUR,
        session_id="past-1",
        target="attention",
        mechanism="split kv",
    )
    pool.write(earlier)
    before_its_tests = dataclasses.replace(snapshot, taken_at=START - 1.0)
    RetrievalLog(reports / LEARNINGS_LOG).record(
        before_its_tests, Retrieval("s2", RetrievalSource.BRIEF, "attention"), (), past=[earlier.lesson_id]
    )
    learnt = dataclasses.replace(snapshot, learnings=LEARNT)
    targets = CloseTargets(archive=TreeArchive(tmp_path), receipts=reports)

    done = close_knowledge(learnt, METRIC, targets=targets)
    again = close_knowledge(learnt, METRIC, targets=targets)

    assert done.lessons == CloseStep(WRITTEN, counts={"new": 2})
    assert done.lesson_standings.status == WRITTEN and done.lesson_standings.counts["lessons"] == 1
    assert [standing.lessons for standing in pool.standings()] == [{earlier.lesson_id: LessonStanding(1, ignored=1)}]
    assert again.lessons == CloseStep(SKIPPED, counts={"duplicate": 2})
    assert again.lesson_standings.status == SKIPPED and again.lesson_standings.reason == ALREADY_RECORDED


@pytest.mark.usefixtures("unconfigured")
def test_a_step_that_raises_is_recorded_and_the_steps_after_it_still_run(
    snapshot: NotebookSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object, **_kwargs: object) -> CollectReport:
        raise ValueError("the mapping does not load")

    monkeypatch.setattr(session, "publish_experiences", broken)

    done = close_knowledge(snapshot, METRIC, targets=CloseTargets(archive=TreeArchive(tmp_path)))

    assert done.experiences == CloseStep(FAILED, "ValueError: the mapping does not load")
    assert (done.learnings.status, done.archive.status) == (DISABLED, ARCHIVED)


def test_without_a_profile_nothing_is_published_or_archived(snapshot: NotebookSnapshot, tmp_path: Path) -> None:
    done = close_knowledge(
        dataclasses.replace(snapshot, profile=None), METRIC, targets=CloseTargets(archive=TreeArchive(tmp_path))
    )

    assert done == KnowledgeClose(
        experiences=CloseStep(SKIPPED, NO_PROFILE),
        learnings=CloseStep(SKIPPED, NO_PROFILE),
        archive=CloseStep(SKIPPED, "no_profile_or_baseline"),
        multibranch_experiences=CloseStep(SKIPPED, NO_PROFILE),
        lessons=CloseStep(SKIPPED, NO_PROFILE),
        lesson_standings=CloseStep(SKIPPED, NO_RETRIEVAL_LOG),
    )


@pytest.mark.parametrize(
    ("report", "step"),
    [
        (CollectReport("m", "s", "<memory>", False, True, blocked_reason="no tests"), CloseStep(SKIPPED, "no tests")),
        (
            CollectReport("m", "s", "<memory>", False, True, errors=(FailedUnit("u1", "OSError: down"),)),
            CloseStep(FAILED, "OSError: down", {"skipped": 0, "errors": 1}),
        ),
    ],
)
@pytest.mark.usefixtures("unconfigured")
def test_a_kb_report_becomes_the_steps_status(
    snapshot: NotebookSnapshot, monkeypatch: pytest.MonkeyPatch, report: CollectReport, step: CloseStep
) -> None:
    monkeypatch.setattr(session, "publish_learnings", lambda *_args, **_kwargs: report)

    assert close_knowledge(snapshot, METRIC).learnings == step


def test_the_breakdown_records_the_knowledge_steps_only_when_given(snapshot: NotebookSnapshot) -> None:
    counts = {"created": 2, "skipped": 0, "errors": 0}
    done = KnowledgeClose(CloseStep(WRITTEN, counts=counts), CloseStep(DISABLED, NO_KB), CloseStep(ARCHIVED))

    close = build_breakdown(snapshot, CloseOutcome(knowledge=done.document()))["close"]

    assert close["knowledge"] == {
        "experiences": {"status": WRITTEN, "reason": "", "counts": counts},
        "multibranch_experiences": {"status": DISABLED, "reason": NO_KB, "counts": {}},
        "learnings": {"status": DISABLED, "reason": NO_KB, "counts": {}},
        "archive": {"status": ARCHIVED, "reason": "", "counts": {}},
        "calibration": {"status": SKIPPED, "reason": NO_REPORTS, "counts": {}},
        "lessons": {"status": SKIPPED, "reason": NO_ARCHIVE, "counts": {}},
        "lesson_standings": {"status": SKIPPED, "reason": NO_ARCHIVE, "counts": {}},
    }
    assert json.loads(json.dumps(close["knowledge"])) == close["knowledge"]
    assert "knowledge" not in build_breakdown(snapshot)["close"]


async def test_frontier_scoring_sizes_its_windows_to_the_budget_reads_the_archive_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    TreeArchive(tmp_path / "archive").put(PAST)
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path / "archive"))
    path = tmp_path / "reports" / FRONTIER_LOG
    history = session.archived_history()
    policy = FrontierPolicy.for_budget(METRIC, BUDGET)

    scoring = FrontierScoring(notebook, policy, history, log=path)
    await scoring.progress()

    assert scoring.policy == policy
    assert [item.session_id for item in scoring.history] == ["past"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.usefixtures("unconfigured")
async def test_without_a_budget_or_an_archive_the_defaults_stand(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    chosen = dataclasses.replace(FrontierPolicy(METRIC), window_seconds=3600.0)

    plain = FrontierScoring(notebook, FrontierPolicy(METRIC))
    given = FrontierScoring(notebook, FrontierPolicy(METRIC), (PAST,))

    assert (plain.policy, plain.history, plain.log) == (FrontierPolicy(METRIC), (), None)
    assert given.history == (PAST,)
    assert FrontierScoring(notebook, chosen).policy == chosen


def test_a_session_scores_with_the_defaults_before_its_profile_and_while_the_evidence_is_thin(
    snapshot: NotebookSnapshot,
) -> None:
    defaults = FrontierPolicy.for_budget(METRIC, BUDGET)

    early = session_frontier_policy(dataclasses.replace(snapshot, profile=None), [PAST], defaults)
    started = session_frontier_policy(snapshot, [PAST], defaults)

    assert (early.policy, early.label) == (defaults, DEFAULT_POLICY)
    assert early.evidence == f"{DEFAULT_POLICY}: the session has no profile yet"
    assert (started.policy, started.label) == (defaults, DEFAULT_POLICY)
    assert started.evidence.startswith(f"{DEFAULT_POLICY}: Too little evidence")


def test_a_session_chooses_among_the_registered_scorers_too_and_a_label_names_a_choice(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    defaults = FrontierPolicy.for_budget(METRIC, BUDGET)
    pool = ScorerPool(tmp_path)

    early = session_frontier_scorer(dataclasses.replace(snapshot, profile=None), [PAST], defaults, pool)
    started = session_frontier_scorer(snapshot, [PAST], defaults, pool)
    named = {
        label: named_frontier_choice(label, defaults, "recorded", pool)
        for label in (DEFAULT_POLICY, "window x2", BANDIT_LABEL, "nowhere")
    }

    assert (early.label, early.evidence) == (DEFAULT_POLICY, f"{DEFAULT_POLICY}: the session has no profile yet")
    assert (started.label, started.policy, started.scorer) == (DEFAULT_POLICY, defaults, FRONTIER_V1)
    assert started.evidence.startswith(f"{DEFAULT_POLICY}: Too little evidence")
    assert [record.label for record in pool.records()] == [BANDIT_LABEL]
    assert named[DEFAULT_POLICY] == PolicyChoice(defaults, DEFAULT_POLICY, "recorded")
    window = named["window x2"]
    assert window is not None and window.policy.window_seconds == 2 * defaults.window_seconds
    bandit = named[BANDIT_LABEL]
    assert bandit is not None and (bandit.policy, bandit.scorer) == (defaults, BanditScorer())
    assert named["nowhere"] is None


def _pooled(pool: ScorerPool, label: str, text: str) -> None:
    pool.register(PoolRecord(label, ScorerSource.EXPRESSION, START - 2 * HOUR, text, "why"))


def test_a_challenger_that_beats_the_champion_is_crowned_and_the_sessions_after_start_from_it(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    defaults = FrontierPolicy.for_budget(METRIC, BUDGET)
    # Sessions after the expressions were written: what ranks the older branch first leads frontier-v1 there.
    races = [race(f"race-{n}", task=task_conditions(profile()), started_at=START - HOUR + 600.0 * n) for n in range(4)]
    pool, worse = ScorerPool(tmp_path / "pool"), ScorerPool(tmp_path / "worse")
    _pooled(pool, "older-v1", "age_s")
    _pooled(worse, "younger-v1", "1000000 - age_s")
    worse.crown(ChampionRecord("younger-v1", START - HOUR, FRONTIER_V1_LABEL, "led"))

    crowned = session_frontier_scorer(snapshot, races, defaults, pool)
    champion = pool.champion()
    later = dataclasses.replace(snapshot, taken_at=snapshot.taken_at + 60.0)
    kept = session_frontier_scorer(later, races, defaults, pool)
    back = session_frontier_scorer(snapshot, races, defaults, worse)

    assert (crowned.label, crowned.scorer) == ("older-v1", ExpressionScorer.read("older-v1", "age_s"))
    assert champion == ChampionRecord("older-v1", snapshot.taken_at, FRONTIER_V1_LABEL, crowned.evidence)
    assert kept.label == "older-v1" and kept.evidence.startswith("older-v1: Keep older-v1")
    assert pool.champion() == champion
    assert (back.label, back.scorer) == (DEFAULT_POLICY, FRONTIER_V1)
    assert worse.champion() == ChampionRecord(FRONTIER_V1_LABEL, snapshot.taken_at, "younger-v1", back.evidence)
    assert current_champion(worse) is None


def test_a_challenger_session_picks_the_best_pooled_scorer_without_crowning(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    defaults = FrontierPolicy.for_budget(METRIC, BUDGET)
    races = [race(f"race-{n}", task=task_conditions(profile()), started_at=START - HOUR + 600.0 * n) for n in range(4)]
    pool = ScorerPool(tmp_path / "pool")
    empty = ScorerPool(tmp_path / "empty")
    _pooled(pool, "older-v1", "age_s")

    picked = session_challenger(snapshot, races, defaults, pool)
    none_yet = session_challenger(dataclasses.replace(snapshot, profile=None), races, defaults, pool)
    none_empty = session_challenger(snapshot, races, defaults, empty)

    assert picked is not None
    assert (picked.label, picked.scorer, picked.policy) == (
        "older-v1",
        ExpressionScorer.read("older-v1", "age_s"),
        defaults,
    )
    assert picked.evidence.startswith("older-v1: exploration challenger")
    assert pool.champion() is None
    assert none_yet is None and none_empty is None


def test_the_judge_reads_the_stall_odds_of_the_archive_and_of_the_session_itself(
    snapshot: NotebookSnapshot,
) -> None:
    odds = session_stall_odds(snapshot, (PAST,), METRIC)
    alone = session_stall_odds(snapshot, (), METRIC)

    assert odds is not None and alone is not None
    assert (odds.sessions, alone.sessions) == (1, 0)
    assert judge_history(snapshot, (PAST,), METRIC, budget_seconds=BUDGET) == odds.render(MAX_CHARS)
    assert judge_history(dataclasses.replace(snapshot, profile=None), (PAST,), METRIC, budget_seconds=BUDGET) == ""
    assert len(judge_history(snapshot, (PAST,), METRIC, budget_seconds=BUDGET, max_chars=80)) <= 80
