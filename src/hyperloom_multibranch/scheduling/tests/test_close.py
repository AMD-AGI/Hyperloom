# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: everything is stopped first, then the exports land in the MB- directories and the KB gets the final recipe.

Both are bounded by the hard stop, and the report is written whatever became of the exports. The report is also read
back: it says how a session ended and whether FINAL may still be run over its directory, together with the processes
the session recorded, which say whether anything of it is still running. A session that was killed left no report at
all, and only those records.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from hyperloom.common.io import append_jsonl, atomic_write_json
from hyperloom.common.owner_scope import local_owner_scope
from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.orchestrator.knowledge.remote_recipe import KBStoreError
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import FinalRecipe
from hyperloom_multibranch.notebook.knowledge.tests.fakes import FakeRecipeStore
from hyperloom_multibranch.notebook.knowledge.tests.test_recipe_writeback import final
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV, ARCHIVED, SKIPPED
from hyperloom_multibranch.notebook.rsi.authoring import AUTHORING_SYSTEM_PROMPT
from hyperloom_multibranch.notebook.rsi.lesson_curation import NO_LESSONS
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, FrontierScoring
from hyperloom_multibranch.notebook.rsi.session import DISABLED, WRITTEN
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import METRIC, scripted_session
from hyperloom_multibranch.scheduling.close import CloseReport, Closer, ended_before_final, report_document
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.exports import (
    ExportServices,
    ExportSources,
    ScorerAuthoring,
    SessionExports,
    step_lines,
)
from hyperloom_multibranch.scheduling.tests.peer_fakes import ScriptedCompletion
from hyperloom_multibranch.scheduling.final import FinalCandidate, FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.manifest import CodeVersion, SessionManifest
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, RunSettings, Workload
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_EXPERIENCE_KB
from hyperloom_multibranch.scheduling.processes import SCHEDULER_BRANCH, ProcessRecord
from hyperloom_multibranch.scheduling.stops import SessionStopReason, StopReport

__all__ = ["final"]

NOW = 1_800_000_000.0
CLOCK = Clock(now=lambda: NOW)
HARD_STOP = NOW + 60.0
#: Real seconds the clock in these tests lets the exports run before the hard stop passes.
SHORT_GRACE = 0.3
#: The commit the scripted session's integration head is at.
CHOSEN = "i1"
#: The reason the sessions these tests read back ended for.
REASON = SessionStopReason.TIME_EXHAUSTED
#: An ownership scope of a pod that is gone: another boot, so none of its pids is a process of this environment.
ANOTHER_POD = "0" * 32 + ":1:1"
#: A session that started long before this test process, so a pid it recorded is another process's by now.
LONG_AGO = 1_700_000_000.0
AGENT_ARGV = ("claude", "-p")
CHOICE = FinalChoice(
    FinalCandidate("integration", INTEGRATION_BRANCH, CHOSEN, 1041.0), (), "the higher value", METRIC, 64
)
LEFTOVER = ProcessRecord(4242, 4242, 0.0, "b01", ("sleep", "600"), "scope")
BUDGET_SECONDS = 2 * 3600.0
KB_URL = "HYPERLOOM_KB_URL"
#: The source revision the manifest of these sessions records, which the breakdown reports the session as having run on.
RAN_ON = "7c1d9a204"
SETTINGS = RunSettings(
    Workload("org/model", "vllm", 8, 16, "fp8", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 1024)),
    max_hours=BUDGET_SECONDS / 3600.0,
    max_gpu_hours=0.0,
    max_llm_usd=0.0,
    caps=RosterCaps(3, 2, 8),
    judge_interval_seconds=600.0,
)


class RecordingStopper:
    """A session stopper that records the stop and reports ``report``; a test double, not a placeholder."""

    def __init__(self, report: StopReport, events: list[str] | None = None) -> None:
        self.report = report
        self.reasons: list[SessionStopReason] = []
        self.events = [] if events is None else events

    async def stop_everything(self, reason: SessionStopReason, hard_stop_at: float) -> StopReport:
        self.reasons.append(reason)
        self.events.append(f"stop by {hard_stop_at}")
        return self.report


class BrokenNotebook:
    """A notebook whose disk fails while CLOSE reads it; a failure double, not a placeholder."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def snapshot(self) -> NotebookSnapshot:
        self.events.append("snapshot")
        raise OSError("disk gone")


class StuckNotebook:
    """A notebook whose disk hangs while CLOSE reads it; a failure double, not a placeholder."""

    async def snapshot(self) -> NotebookSnapshot:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class ChosenRepo:
    """The experiment repository as CLOSE reads it: the chosen commit's recipe only; a test double, not a
    placeholder."""

    def __init__(self, final: FinalRecipe) -> None:
        self.final = final

    async def read_recipe(self, commit: str) -> Recipe:
        assert commit == CHOSEN
        return self.final.recipe


class AfterStopToolbox:
    """A toolbox already closed by the stop; CLOSE must not ask it for its queue. A failure double."""

    async def queue_state(self) -> QueueState:
        raise AssertionError("CLOSE rebuilds the dashboard with the queue the stop reported")


class UnreachableStore(FakeRecipeStore):
    """A Recipe KB store that cannot be reached; a failure double, not a placeholder."""

    def get_rollup(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise KBStoreError("connection refused")


@pytest.fixture(autouse=True)
def no_knowledge_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLOSE's knowledge steps find no archive unless a test names one, and no Experience KB in the environment."""

    monkeypatch.delenv(KB_URL, raising=False)
    monkeypatch.delenv(ARCHIVE_ENV, raising=False)


@pytest.fixture
async def notebook(tmp_path: Path) -> LocalNotebook:
    return await scripted_session(tmp_path / "notebook", FakeClock())


@pytest.fixture
def layout(tmp_path: Path) -> SessionLayout:
    """A session directory as the run that reached CLOSE left it, the manifest its first act wrote included."""

    layout = create_session(tmp_path / "workspace", "model", CLOCK)
    SessionManifest(
        code_version=CodeVersion("0.0", RAN_ON),
        session_id=layout.session_id,
        started_at=NOW,
        deadline=NOW + BUDGET_SECONDS,
        settings=SETTINGS,
        metric=METRIC,
        policy_versions={},
    ).write(layout.manifest)
    return layout


AUTHOR_MODEL = "author-model"


def _closer(
    layout: SessionLayout,
    notebook: object,
    stopper: RecordingStopper,
    recipe: FinalRecipe,
    kb_client: RemoteRecipeClient | None,
    author: ScriptedCompletion | None = None,
    curator: ScriptedCompletion | None = None,
) -> Closer:
    """CLOSE with ``recipe`` committed as :data:`CHOSEN`, its files in the session's blob store and not in the
    repository; ``curator`` curates the lesson pool."""

    author = author or ScriptedCompletion()
    curator = curator or ScriptedCompletion()
    progress = FrontierScoring(notebook, FrontierPolicy(metric=METRIC))  # type: ignore[arg-type]
    blobs = LocalBlobStore(layout.blobs)
    for path in (
        *(patch.path for component in recipe.recipe.components for patch in component.patches),
        *(kernel.ref for kernel in recipe.recipe.kernels),
    ):
        blobs.put((recipe.files_root / path).read_bytes())
    dashboard = SessionDashboard(notebook, progress, AfterStopToolbox(), METRIC, layout.dashboard)  # type: ignore[arg-type]
    sources = ExportSources(notebook, ChosenRepo(recipe), blobs, dashboard)
    services = ExportServices(kb_client, ScorerAuthoring(author, AUTHOR_MODEL), ScorerAuthoring(curator, AUTHOR_MODEL))
    exports = SessionExports(layout, sources, METRIC, BUDGET_SECONDS, services)  # type: ignore[arg-type]
    return Closer(layout, stopper, exports, CLOCK)  # type: ignore[arg-type]


def _clean() -> RecordingStopper:
    return RecordingStopper(StopReport((), QueueState()))


async def test_close_publishes_the_final_recipe_exports_and_stops_everything(
    layout: SessionLayout,
    notebook: LocalNotebook,
    final: FinalRecipe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path / "archive"))
    monkeypatch.setenv(KB_URL, "http://127.0.0.1:9/kb")
    store = FakeRecipeStore(champion={"metric": "optimized_throughput", "session_id": "older", "value": 1000.0})
    stopper = _clean()
    author, curator = ScriptedCompletion(), ScriptedCompletion()

    report = await _closer(layout, notebook, stopper, final, RemoteRecipeClient(store), author, curator).close(
        SessionStopReason.TIME_EXHAUSTED, CHOICE, "", HARD_STOP
    )

    assert report.exports is not None and report.exports.kb_write_back.status == "written"
    assert store.promoted is not None and store.promoted[2] == 1041.0
    recipe_files = sorted(path.read_bytes() for path in final.files_root.rglob("*") if path.is_file())
    assert sorted(store.uploads.values()) == recipe_files and not any(layout.final.iterdir())
    assert (stopper.reasons, stopper.events) == ([SessionStopReason.TIME_EXHAUSTED], [f"stop by {HARD_STOP}"])
    assert report.exports.breakdown == layout.reports / "session_breakdown.json"
    breakdown = json.loads(report.exports.breakdown.read_text())
    assert breakdown["close"]["kb_write_back"]["status"] == "written"
    knowledge = breakdown["close"]["knowledge"]
    for step in ("experiences", "learnings"):
        assert (knowledge[step]["status"], knowledge[step]["reason"]) == (DISABLED, MB_FAKE_EXPERIENCE_KB)
    assert knowledge["archive"]["status"] == ARCHIVED and any((tmp_path / "archive").iterdir())
    assert [system for system, _user in author.calls] == [AUTHORING_SYSTEM_PROMPT]
    assert knowledge["scorers"]["status"] in (WRITTEN, SKIPPED)
    # The session wrote no learning, so the pool holds none to curate and the curator is not asked.
    assert (knowledge["lesson_curation"]["status"], knowledge["lesson_curation"]["reason"]) == (SKIPPED, NO_LESSONS)
    assert curator.calls == [] and knowledge["lessons"]["status"] == SKIPPED
    assert breakdown["metadata"]["session"]["code_revision"] == RAN_ON
    assert breakdown["close"]["final"]["commit_id"] == CHOSEN
    assert breakdown["close"]["selection"]["chosen"]["commit"] == CHOSEN
    assert (breakdown["close"]["selection"]["metric"], breakdown["close"]["selection"]["primary_conc"]) == (METRIC, 64)
    files = report.exports.dashboard
    assert (files.json.parent, files.html.parent) == (layout.dashboard, layout.dashboard)
    dashboard = json.loads(files.json.read_text())
    assert dashboard["queue"]["running_by_branch"] == {} and dashboard["queue"]["queued_by_branch"] == {}
    assert any(row.get("frontier") for row in dashboard["branches"])
    written = json.loads(layout.report.read_text())
    assert (written["stop_reason"], written["stop"]["clean"]) == ("time_exhausted", True)
    assert written["final"]["chosen"]["commit"] == CHOSEN and written["exports"]["knowledge"] == knowledge
    markdown = layout.report_markdown.read_text()
    assert "Chosen: integration `i1`" in markdown
    assert f"Experiences: {DISABLED} {MB_FAKE_EXPERIENCE_KB}" in markdown
    assert f"Scorers: {knowledge['scorers']['status']}" in markdown
    assert f"Lesson curation: {SKIPPED} {NO_LESSONS}" in markdown
    assert f"Lesson standings: {knowledge['lesson_standings']['status']}" in markdown
    assert all(line in markdown.splitlines() for line in step_lines(report.exports.knowledge_steps()))
    assert len(step_lines(report.exports.knowledge_steps())) == len(knowledge)


async def test_without_a_kb_client_or_a_final_choice_the_write_back_is_recorded_as_not_made(
    layout: SessionLayout, notebook: LocalNotebook, final: FinalRecipe
) -> None:
    disabled = await _closer(layout, notebook, _clean(), final, None).close(
        SessionStopReason.SIGNAL, CHOICE, "", HARD_STOP
    )
    assert disabled.exports is not None
    assert json.loads(disabled.exports.breakdown.read_text())["close"]["final"]["commit_id"] == CHOSEN
    no_choice = FinalChoice(None, (), "no candidate passed the final validation", METRIC, 64)
    skipped = await _closer(layout, notebook, _clean(), final, RemoteRecipeClient(FakeRecipeStore())).close(
        SessionStopReason.PRELUDE_FAILED, no_choice, "prelude failed", HARD_STOP
    )

    assert disabled.exports.kb_write_back.status == "disabled"
    assert skipped.exports is not None
    assert (skipped.exports.kb_write_back.status, skipped.exports.kb_write_back.reason) == (
        "skipped",
        "no final choice",
    )
    close = json.loads(skipped.exports.breakdown.read_text())["close"]
    assert close["final"] is None and close["selection"]["chosen"] is None
    assert json.loads(layout.report.read_text())["error"] == "prelude failed"


async def test_an_unreachable_kb_is_recorded_and_close_carries_on(
    layout: SessionLayout, notebook: LocalNotebook, final: FinalRecipe
) -> None:
    stopper = _clean()
    closer = _closer(layout, notebook, stopper, final, RemoteRecipeClient(UnreachableStore()))

    report = await closer.close(SessionStopReason.TIME_EXHAUSTED, CHOICE, "", HARD_STOP)

    assert report.exports is not None and report.exports.kb_write_back.status == "failed"
    assert "connection refused" in report.exports.kb_write_back.reason
    assert stopper.reasons == [SessionStopReason.TIME_EXHAUSTED]


async def test_everything_is_stopped_before_the_exports_and_a_failed_export_is_reported(
    layout: SessionLayout, final: FinalRecipe
) -> None:
    events: list[str] = []
    stopper = RecordingStopper(StopReport((), QueueState()), events)

    report = await _closer(layout, BrokenNotebook(events), stopper, final, None).close(
        SessionStopReason.SIGNAL, None, "", HARD_STOP
    )

    assert events == [f"stop by {HARD_STOP}", "snapshot"]
    assert report.exports is None and report.stop.clean
    written = json.loads(layout.report.read_text())
    assert (written["stop_reason"], written["exports"], written["stop"]["clean"]) == ("signal", None, True)


async def test_exports_that_overrun_the_hard_stop_are_cut_and_the_report_still_written(
    layout: SessionLayout, final: FinalRecipe
) -> None:
    report = await _closer(layout, StuckNotebook(), _clean(), final, None).close(
        SessionStopReason.TIME_EXHAUSTED, CHOICE, "", NOW + SHORT_GRACE
    )

    assert report.exports is None and report.stop.clean
    assert json.loads(layout.report.read_text())["exports"] is None
    assert "Exports: not finished" in layout.report_markdown.read_text()


async def test_processes_and_tests_that_outlive_the_stop_are_reported(
    layout: SessionLayout, notebook: LocalNotebook, final: FinalRecipe
) -> None:
    stopper = RecordingStopper(StopReport((LEFTOVER,), QueueState(running_by_branch={"b01": 1})))

    report = await _closer(layout, notebook, stopper, final, None).close(SessionStopReason.SIGNAL, None, "", HARD_STOP)

    assert not report.stop.clean
    written = json.loads(layout.report.read_text())
    assert written["stop"]["clean"] is False
    assert [row["pid"] for row in written["stop"]["leftovers"]] == [4242]
    assert written["stop"]["queue"]["running_by_branch"] == {"b01": 1}
    assert "processes left: [4242]" in layout.report_markdown.read_text()


def _written(layout: SessionLayout, stop_reason: SessionStopReason, final: FinalChoice | None) -> SessionLayout:
    """``MB-report.json`` as CLOSE writes it for a session that ended for ``stop_reason`` with ``final``."""

    report = CloseReport(stop_reason, final, None, StopReport((), QueueState()), "")
    atomic_write_json(layout.report, report_document(report, layout.session_id, NOW))
    return layout


def _recorded(layout: SessionLayout, record: ProcessRecord) -> SessionLayout:
    """``MB-pids.jsonl`` as a session leaves it, holding ``record`` alone."""

    append_jsonl(layout.pids, asdict(record))
    return layout


def _this_process(branch: str) -> ProcessRecord:
    """This test process as a running session recorded it: it started before the record was written, so the pid is
    the one the record names."""

    return ProcessRecord(os.getpid(), os.getpgid(0), time.time(), branch, AGENT_ARGV, local_owner_scope())


def test_a_report_of_a_session_that_closed_before_final_carries_its_stop_reason(layout: SessionLayout) -> None:
    assert ended_before_final(_written(layout, SessionStopReason.TIME_EXHAUSTED, None)) is REASON


def test_a_session_killed_before_it_could_close_ended_killed(layout: SessionLayout) -> None:
    dead = ProcessRecord(os.getpid(), os.getpgid(0), NOW, "b01", AGENT_ARGV, ANOTHER_POD)

    assert ended_before_final(_recorded(layout, dead)) is SessionStopReason.KILLED


def test_a_session_whose_processes_still_run_here_is_refused(layout: SessionLayout) -> None:
    with pytest.raises(ValueError, match=f"still running as .{os.getpid()}."):
        ended_before_final(_recorded(layout, _this_process(SCHEDULER_BRANCH)))


def test_a_closed_session_a_process_of_outlived_its_report_is_refused(layout: SessionLayout) -> None:
    """A CLOSE that could not kill everything reports leftovers, and FINAL may not run beside one of them."""

    closed = _written(layout, SessionStopReason.TIME_EXHAUSTED, None)

    with pytest.raises(ValueError, match=f"still running as .{os.getpid()}."):
        ended_before_final(_recorded(closed, _this_process("b01")))


def test_a_pid_of_a_killed_session_another_process_reuses_is_not_the_session_running(layout: SessionLayout) -> None:
    reused = ProcessRecord(os.getpid(), os.getpgid(0), LONG_AGO, SCHEDULER_BRANCH, AGENT_ARGV, local_owner_scope())

    assert ended_before_final(_recorded(layout, reused)) is SessionStopReason.KILLED


def test_a_report_that_records_a_final_choice_refuses_a_second_final(layout: SessionLayout) -> None:
    with pytest.raises(ValueError, match=f"already chose {CHOSEN}"):
        ended_before_final(_written(layout, SessionStopReason.TIME_EXHAUSTED, CHOICE))


def test_a_final_that_chose_nothing_may_be_run_again(layout: SessionLayout) -> None:
    nothing = FinalChoice(None, (), "no candidate passed the final validation", METRIC, 64)

    assert ended_before_final(_written(layout, SessionStopReason.TIME_EXHAUSTED, nothing)) is REASON


def test_a_failed_prelude_leaves_no_root_to_validate(layout: SessionLayout) -> None:
    with pytest.raises(ValueError, match="PRELUDE failed"):
        ended_before_final(_written(layout, SessionStopReason.PRELUDE_FAILED, None))
