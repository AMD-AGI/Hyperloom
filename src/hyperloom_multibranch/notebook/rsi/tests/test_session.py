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
from hyperloom_multibranch.notebook.rsi.archive import (
    ARCHIVE_ENV,
    ARCHIVED,
    NO_ARCHIVE,
    SKIPPED,
    TreeArchive,
    archive_session,
)
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, FrontierScoring
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, Retrieval, RetrievalLog, RetrievalSource
from hyperloom_multibranch.notebook.rsi.history import MAX_CHARS, session_stall_odds
from hyperloom_multibranch.notebook.rsi.decisions import decision_calls
from hyperloom_multibranch.notebook.rsi.ladder import standings, trials
from hyperloom_multibranch.notebook.rsi.menu import MENU_LOG, MenuLog
from hyperloom_multibranch.notebook.rsi.outcomes import OUTCOMES_REPORT, OutcomePool
from hyperloom_multibranch.notebook.rsi.playbook import FALLBACK_KEY, H0, PLAYBOOK_KEY, PlaybookPool
from hyperloom_multibranch.notebook.rsi.lessons import (
    LessonPool,
    LessonRevision,
    LessonStanding,
    lesson_id_for,
    revision_id_for,
)
from hyperloom_multibranch.notebook.rsi.session import (
    ALREADY_RECORDED,
    DISABLED,
    EXPERIENCE_RECEIPT,
    FAILED,
    FRONTIER_LOG,
    LEARNING_RECEIPT,
    MULTIBRANCH_RECEIPT,
    NO_KB,
    NO_PROFILE,
    NO_REPORTS,
    NO_PLAYBOOK_PLAYED,
    NO_RETRIEVAL_LOG,
    NO_TRANSCRIPT,
    NO_TREES,
    SCHEDULING_REPLAY_REPORT,
    STOP_AUDIT_REPORT,
    WRITTEN,
    CloseStep,
    CloseTargets,
    KnowledgeClose,
    close_knowledge,
    judge_history,
    session_menus,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, branch, tree
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
def test_close_keeps_a_session_that_played_a_playbook_as_its_trial_once(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    archive = TreeArchive(tmp_path)
    played = dataclasses.replace(snapshot.fingerprints[-1], policy_versions={PLAYBOOK_KEY: H0.label})
    session = dataclasses.replace(snapshot, fingerprints=(*snapshot.fingerprints[:-1], played))

    done = close_knowledge(session, METRIC, targets=CloseTargets(archive=archive))
    again = close_knowledge(session, METRIC, targets=CloseTargets(archive=archive))

    (kept,) = trials(PlaybookPool.beside(archive))
    assert done.playbook_trial == CloseStep(WRITTEN, "", {"trials": 1, "settled": 0})
    assert again.playbook_trial == CloseStep(SKIPPED, ALREADY_RECORDED, {"trials": 0, "settled": 0})
    assert (kept.session_id, kept.played, kept.champion, kept.fell_back) == ("sess-1", H0.label, H0.label, False)
    assert kept.task.startswith("workload=inferencex model=") and "early_stops" in kept.measures
    assert close_knowledge(snapshot, METRIC, targets=CloseTargets(archive=archive)).playbook_trial == CloseStep(
        SKIPPED, NO_PLAYBOOK_PLAYED
    )


@pytest.mark.usefixtures("unconfigured")
def test_a_challenger_session_that_fell_back_to_the_champion_is_kept_as_such(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    archive = TreeArchive(tmp_path)
    breach = "stop of s1, held by its probation"
    played = dataclasses.replace(
        snapshot.fingerprints[-1], policy_versions={PLAYBOOK_KEY: "patient", FALLBACK_KEY: breach}
    )

    close_knowledge(
        dataclasses.replace(snapshot, fingerprints=(played,)), METRIC, targets=CloseTargets(archive=archive)
    )

    (kept,) = trials(PlaybookPool.beside(archive))
    assert (kept.played, kept.fell_back) == ("patient", True)


@pytest.mark.usefixtures("unconfigured")
def test_close_reruns_the_scheduling_replay_and_says_what_moved_since_the_last_close(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    archive, reports = TreeArchive(tmp_path / "archive"), tmp_path / "reports"
    assert snapshot.profile is not None
    later = dataclasses.replace(
        snapshot,
        profile=dataclasses.replace(snapshot.profile, session_id="sess-2"),
        taken_at=snapshot.taken_at + HOUR,
    )

    first = close_knowledge(snapshot, METRIC, targets=CloseTargets(archive=archive, receipts=reports / "1"))
    second = close_knowledge(later, METRIC, targets=CloseTargets(archive=archive, receipts=reports / "2"))

    assert first.scheduling_replay == CloseStep(WRITTEN, "", {"sessions": 1, "moved": 0})
    assert second.scheduling_replay.status == WRITTEN and second.scheduling_replay.counts["sessions"] == 2
    text = (reports / "1" / SCHEDULING_REPLAY_REPORT).read_text(encoding="utf-8")
    assert text.startswith("Scheduling replay at the CLOSE of sess-1, over 1 archived sessions")
    assert "No earlier CLOSE read the replay against h0." in text
    assert "Since the CLOSE of sess-1:" in (reports / "2" / SCHEDULING_REPLAY_REPORT).read_text(encoding="utf-8")
    assert [item.session_id for item in standings(PlaybookPool.beside(archive))] == ["sess-1", "sess-2"]
    assert close_knowledge(snapshot, METRIC).scheduling_replay == CloseStep(SKIPPED, NO_REPORTS)


@pytest.mark.usefixtures("unconfigured")
def test_close_keeps_the_judges_exchanges_beside_the_archive_once(snapshot: NotebookSnapshot, tmp_path: Path) -> None:
    transcript = tmp_path / "MB-judge.jsonl"
    transcript.write_text(json.dumps({"at": START, "prompt": "view", "reply": "{}", "error": ""}) + "\n")
    targets = CloseTargets(archive=TreeArchive(tmp_path / "archive"), judge_transcript=transcript)
    missing = CloseTargets(judge_transcript=tmp_path / "none.jsonl")

    done = close_knowledge(snapshot, METRIC, targets=targets)
    again = close_knowledge(snapshot, METRIC, targets=targets)

    assert (done.judge_decisions, again.judge_decisions) == (
        CloseStep(WRITTEN, counts={"calls": 1}),
        CloseStep(SKIPPED, ALREADY_RECORDED),
    )
    assert [call["prompt"] for call in decision_calls(TreeArchive(tmp_path / "archive"), "sess-1")] == ["view"]
    assert close_knowledge(snapshot, METRIC, targets=missing).judge_decisions == CloseStep(SKIPPED, NO_TRANSCRIPT)


@pytest.mark.usefixtures("unconfigured")
def test_close_counts_the_specialists_outcomes_beside_the_receipts_and_keeps_them_beside_the_archive(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    archive, reports = TreeArchive(tmp_path / "archive"), tmp_path / "reports"

    done = close_knowledge(snapshot, METRIC, targets=CloseTargets(archive=archive, receipts=reports))

    assert done.specialist_outcomes.status == WRITTEN and done.specialist_outcomes.counts["kept"] == 1
    assert done.specialist_outcomes.reason.startswith("commit rate ")
    assert (reports / OUTCOMES_REPORT).read_text(encoding="utf-8").startswith("# Specialist outcomes of sess-1")
    assert [item.session_id for item in OutcomePool.beside(archive).sessions()] == ["sess-1"]
    assert close_knowledge(snapshot, METRIC).specialist_outcomes == CloseStep(SKIPPED, NO_REPORTS)


@pytest.mark.usefixtures("unconfigured")
def test_each_rsi_step_of_close_says_why_it_kept_nothing_and_close_goes_on(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    played = dataclasses.replace(snapshot.fingerprints[-1], policy_versions={PLAYBOOK_KEY: H0.label})
    session = dataclasses.replace(snapshot, fingerprints=(*snapshot.fingerprints[:-1], played))
    transcript = tmp_path / "MB-judge.jsonl"
    transcript.write_text(json.dumps({"at": START, "prompt": "view", "reply": "{}", "error": ""}) + "\n")
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where a folder should be\n")

    def closed(state: NotebookSnapshot, targets: CloseTargets) -> KnowledgeClose:
        return close_knowledge(state, METRIC, targets=dataclasses.replace(targets, judge_transcript=transcript))

    bare = closed(session, CloseTargets())
    unprofiled = closed(
        dataclasses.replace(session, profile=None),
        CloseTargets(archive=TreeArchive(tmp_path / "archive"), receipts=tmp_path / "reports"),
    )
    no_archive = closed(session, CloseTargets(archive=TreeArchive(blocked), receipts=tmp_path / "reports"))
    no_receipts = closed(session, CloseTargets(archive=TreeArchive(tmp_path / "written"), receipts=blocked))

    assert (bare.playbook_trial, bare.judge_decisions) == (CloseStep(SKIPPED, NO_ARCHIVE),) * 2
    assert (unprofiled.playbook_trial, unprofiled.scheduling_replay, unprofiled.judge_decisions) == (
        CloseStep(SKIPPED, NO_PROFILE),
    ) * 3
    assert no_archive.scheduling_replay == CloseStep(SKIPPED, NO_TREES)
    assert [step.status for step in (no_archive.playbook_trial, no_archive.judge_decisions)] == [FAILED] * 2
    assert no_archive.specialist_outcomes.status == FAILED
    assert [step.status for step in (no_receipts.stop_audit, no_receipts.scheduling_replay)] == [FAILED] * 2
    assert no_receipts.specialist_outcomes.status == FAILED and no_receipts.playbook_trial.status == WRITTEN


@pytest.mark.usefixtures("unconfigured")
def test_the_sessions_menus_read_the_archive_and_log_each_menu_beside_the_reports(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    archive, reports = TreeArchive(tmp_path / "archive"), tmp_path / "reports"
    assert archive_session(snapshot, METRIC, archive).status == ARCHIVED

    menus = session_menus(METRIC, reports, archive)
    assert menus is not None
    menus.menu(snapshot, "s1", "c1")
    unlogged = session_menus(METRIC, None, archive)

    assert [tree.session_id for tree in menus.trees] == ["sess-1"]
    assert [(logged.branch, logged.commit) for logged in MenuLog(reports / MENU_LOG).menus()] == [("s1", "c1")]
    assert unlogged is not None and unlogged.log is None
    assert session_menus(METRIC, reports) is None


@pytest.mark.usefixtures("unconfigured")
def test_close_writes_the_stop_audit_of_the_sessions_stops_beside_the_receipts(
    snapshot: NotebookSnapshot, tmp_path: Path
) -> None:
    reports = tmp_path / "reports"

    done = close_knowledge(snapshot, METRIC, targets=CloseTargets(archive=TreeArchive(tmp_path), receipts=reports))

    counts = {
        "stops": 1,
        "judge_stops": 0,
        "early_stops": 0,
        "below_bar": 0,
        "probations": 0,
        "probations_committed": 0,
    }
    assert done.stop_audit == CloseStep(WRITTEN, "the judge stopped no branch", counts)
    assert (reports / STOP_AUDIT_REPORT).read_text(encoding="utf-8").startswith("# Stop audit of sess-1\n")
    assert close_knowledge(snapshot, METRIC).stop_audit == CloseStep(SKIPPED, NO_REPORTS)


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
    assert [standing.lessons for standing in pool.standings()] == [{earlier.lesson_id: LessonStanding(1, borne_out=1)}]
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
        "lessons": {"status": SKIPPED, "reason": NO_ARCHIVE, "counts": {}},
        "lesson_standings": {"status": SKIPPED, "reason": NO_ARCHIVE, "counts": {}},
        "stop_audit": {"status": SKIPPED, "reason": NO_REPORTS, "counts": {}},
        "playbook_trial": {"status": SKIPPED, "reason": NO_PLAYBOOK_PLAYED, "counts": {}},
        "scheduling_replay": {"status": SKIPPED, "reason": NO_REPORTS, "counts": {}},
        "specialist_outcomes": {"status": SKIPPED, "reason": NO_REPORTS, "counts": {}},
        "judge_decisions": {"status": SKIPPED, "reason": NO_TRANSCRIPT, "counts": {}},
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


def test_the_judge_reads_the_stall_odds_of_the_archive_and_of_the_session_itself(
    snapshot: NotebookSnapshot,
) -> None:
    odds = session_stall_odds(snapshot, (PAST,), METRIC)
    alone = session_stall_odds(snapshot, (), METRIC)

    assert odds is not None and alone is not None
    assert (odds.sessions, alone.sessions) == (1, 0)
    assert judge_history(snapshot, (PAST,), METRIC, budget_seconds=BUDGET) == odds.render(MAX_CHARS)
    assert judge_history(snapshot, (), METRIC, budget_seconds=BUDGET) == ""
    assert judge_history(dataclasses.replace(snapshot, profile=None), (PAST,), METRIC, budget_seconds=BUDGET) == ""
    assert len(judge_history(snapshot, (PAST,), METRIC, budget_seconds=BUDGET, max_chars=80)) <= 80
