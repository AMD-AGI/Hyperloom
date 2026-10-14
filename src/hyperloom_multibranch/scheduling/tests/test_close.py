# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: everything is stopped first, then the exports land in the MB- directories and the KB gets the final recipe.

Both are bounded by the hard stop, and the report is written whatever became of the exports.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

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
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, FrontierScoring
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import METRIC, scripted_session
from hyperloom_multibranch.scheduling.close import Closer
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.exports import SessionExports
from hyperloom_multibranch.scheduling.final import FinalCandidate, FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.processes import ProcessRecord
from hyperloom_multibranch.scheduling.stops import SessionStopReason, StopReport
from hyperloom_multibranch.scheduling.patches import PatchStore

__all__ = ["final"]

NOW = 1_800_000_000.0
CLOCK = Clock(now=lambda: NOW)
HARD_STOP = NOW + 60.0
#: Real seconds the clock in these tests lets the exports run before the hard stop passes.
SHORT_GRACE = 0.3
#: The commit the scripted session's integration head is at.
CHOSEN = "i1"
CHOICE = FinalChoice(
    FinalCandidate("integration", INTEGRATION_BRANCH, CHOSEN, 1041.0), (), "the higher value", METRIC, 64
)
LEFTOVER = ProcessRecord(4242, 4242, 0.0, "b01", ("sleep", "600"), "scope")


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
    """The experiment repository as CLOSE reads it: the chosen commit's recipe; a test double, not a placeholder."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe

    async def read_recipe(self, commit: str) -> Recipe:
        assert commit == CHOSEN
        return self.recipe


class UnreachableStore(FakeRecipeStore):
    """A Recipe KB store that cannot be reached; a failure double, not a placeholder."""

    def get_rollup(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise KBStoreError("connection refused")


@pytest.fixture
async def notebook(tmp_path: Path) -> LocalNotebook:
    return await scripted_session(tmp_path / "notebook", FakeClock())


@pytest.fixture
def layout(tmp_path: Path) -> SessionLayout:
    return create_session(tmp_path / "workspace", "model", CLOCK)


def _closer(
    layout: SessionLayout,
    notebook: object,
    stopper: RecordingStopper,
    recipe: FinalRecipe,
    kb_client: RemoteRecipeClient | None,
) -> Closer:
    """CLOSE with ``recipe`` committed as :data:`CHOSEN` and its files in the session's blob store."""

    progress = FrontierScoring(notebook, FrontierPolicy(metric=METRIC))  # type: ignore[arg-type]
    repo = ChosenRepo(recipe.recipe)
    blobs = LocalBlobStore(layout.blobs)
    for path in recipe.files_root.rglob("*"):
        if path.is_file():
            blobs.put(path.read_bytes())
    patches = PatchStore(blobs)
    exports = SessionExports(layout, notebook, repo, patches, progress, METRIC, kb_client=kb_client)  # type: ignore[arg-type]
    return Closer(layout, stopper, exports, CLOCK)  # type: ignore[arg-type]


def _clean() -> RecordingStopper:
    return RecordingStopper(StopReport((), QueueState()))


async def test_close_publishes_the_final_recipe_exports_and_stops_everything(
    layout: SessionLayout, notebook: LocalNotebook, final: FinalRecipe
) -> None:
    store = FakeRecipeStore(champion={"metric": "optimized_throughput", "session_id": "older", "value": 1000.0})
    stopper = _clean()

    report = await _closer(layout, notebook, stopper, final, RemoteRecipeClient(store)).close(
        SessionStopReason.TIME_EXHAUSTED, CHOICE, "", HARD_STOP
    )

    assert report.exports is not None and report.exports.kb_write_back.status == "written"
    assert store.promoted is not None and store.promoted[2] == 1041.0
    assert sorted(path.name for path in layout.final_checkout.iterdir()) == ["kernel", "patch"]
    assert (stopper.reasons, stopper.events) == ([SessionStopReason.TIME_EXHAUSTED], [f"stop by {HARD_STOP}"])
    assert report.exports.breakdown == layout.reports / "session_breakdown.json"
    breakdown = json.loads(report.exports.breakdown.read_text())
    assert breakdown["close"]["kb_write_back"]["status"] == "written"
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
    assert written["final"]["chosen"]["commit"] == CHOSEN
    assert "Chosen: integration `i1`" in layout.report_markdown.read_text()


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
