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
from hyperloom_multibranch.notebook.knowledge.breakdown import build_breakdown
from hyperloom_multibranch.notebook.knowledge.tests.kb_service import experience_service
from hyperloom_multibranch.notebook.rsi import session
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV, ARCHIVED, SKIPPED, TreeArchive
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.notebook.rsi.history import MAX_CHARS, session_history
from hyperloom_multibranch.notebook.rsi.session import (
    DISABLED,
    EXPERIENCE_RECEIPT,
    FAILED,
    FRONTIER_LOG,
    LEARNING_RECEIPT,
    NO_KB,
    NO_PROFILE,
    WRITTEN,
    CloseStep,
    CloseTargets,
    KnowledgeClose,
    close_knowledge,
    judge_history,
    session_frontier,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.rsi.tests.trees import branch, tree
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

    close = build_breakdown(snapshot, knowledge=done.document())["close"]

    assert close["knowledge"] == {
        "experiences": {"status": WRITTEN, "reason": "", "counts": counts},
        "learnings": {"status": DISABLED, "reason": NO_KB, "counts": {}},
        "archive": {"status": ARCHIVED, "reason": "", "counts": {}},
    }
    assert json.loads(json.dumps(close["knowledge"])) == close["knowledge"]
    assert "knowledge" not in build_breakdown(snapshot)["close"]


async def test_the_session_frontier_sizes_its_windows_to_the_budget_reads_the_archive_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    TreeArchive(tmp_path / "archive").put(PAST)
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path / "archive"))
    path = tmp_path / "reports" / FRONTIER_LOG

    scoring = session_frontier(notebook, METRIC, budget_seconds=BUDGET, log_path=path)
    await scoring.progress()

    assert scoring.policy == FrontierPolicy.for_budget(METRIC, BUDGET)
    assert [item.session_id for item in scoring.history] == ["past"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.usefixtures("unconfigured")
async def test_without_a_budget_or_an_archive_the_defaults_stand(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())

    plain = session_frontier(notebook, METRIC)
    given = session_frontier(notebook, METRIC, history=[PAST])

    assert (plain.policy, plain.history, plain.log) == (FrontierPolicy(METRIC), (), None)
    assert given.history == (PAST,)


def test_the_judge_reads_the_digest_of_the_archived_sessions_and_nothing_without_any(
    snapshot: NotebookSnapshot,
) -> None:
    digest = session_history(snapshot, (PAST,), METRIC, budget_seconds=BUDGET)

    assert judge_history(snapshot, (), METRIC, budget_seconds=BUDGET) == ""
    assert judge_history(snapshot, (PAST,), METRIC, budget_seconds=BUDGET) == digest.render(MAX_CHARS)
    assert len(judge_history(snapshot, (PAST,), METRIC, budget_seconds=BUDGET, max_chars=80)) <= 80
