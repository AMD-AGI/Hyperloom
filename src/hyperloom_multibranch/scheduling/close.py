# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: stop everything, export the session, and report.

The stop comes first: from then on no fork is admitted, no test runs and no process the scheduler started is left,
and the exports (:mod:`~hyperloom_multibranch.scheduling.exports`) read a notebook that no longer changes. Stopping
and exporting are each bounded by the hard stop. ``MB-report.json`` and ``MB-report.md`` record the outcome whatever
became of the exports, including what outlived the stop; a CLOSE cut short by a second stop signal is reported by the
session through :meth:`Closer.write_report`.
``MB-report.json`` is read back by one caller, :func:`ended_before_final`, which says how a session ended and
whether FINAL may still be run over its directory; it lives here so the format has one reader beside its writer.
Whether FINAL may run is not the report's to answer alone: what the session recorded as its processes says whether
it is over, and a session that was killed never reached CLOSE and left no report at all.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from hyperloom.common.io import WriteOptions, atomic_write_json, atomic_write_text
from hyperloom.common.jsonio import read_json_strict
from hyperloom.contracts.recipe import RecipeError
from hyperloom_multibranch.contracts.notebook.interfaces import NotebookError
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.exports import CloseExports, SessionExports
from hyperloom_multibranch.scheduling.final import FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.processes import live_processes
from hyperloom_multibranch.scheduling.stops import SessionStopper, SessionStopReason, StopReport

logger = logging.getLogger(__name__)

#: What an export raises when it cannot finish: a timeout at the hard stop, the disk, a recipe the repository holds,
#: or the notebook. Anything else is a bug and propagates once the report is written.
EXPORT_FAILURES: tuple[type[BaseException], ...] = (asyncio.TimeoutError, OSError, RecipeError, NotebookError)


@dataclass(frozen=True)
class CloseReport:
    """``error`` is what ended the session abnormally, as the caller gave it; ``stop.clean`` says nothing was left.

    ``exports`` is ``None`` when an export failed with one of :data:`EXPORT_FAILURES` or overran the hard stop.
    """

    stop_reason: SessionStopReason
    final: FinalChoice | None
    exports: CloseExports | None
    stop: StopReport
    error: str


def report_document(report: CloseReport, session_id: str, closed_at: float) -> dict[str, object]:
    """``MB-report.json``: the session's ending, its final choice, the exports and what outlived the stop."""

    exports = report.exports
    return {
        "session_id": session_id,
        "closed_at": closed_at,
        "stop_reason": report.stop_reason.value,
        "error": report.error,
        "final": encode(report.final),
        "exports": None
        if exports is None
        else {
            "kb_write_back": encode(exports.kb_write_back),
            "knowledge": exports.knowledge_document(),
            "breakdown": str(exports.breakdown),
            "dashboard": {"json": str(exports.dashboard.json), "html": str(exports.dashboard.html)},
        },
        "stop": {
            "clean": report.stop.clean,
            "leftovers": encode(report.stop.leftovers),
            "queue": encode(report.stop.queue),
            "failures": list(report.stop.failures),
        },
    }


def ended_before_final(layout: SessionLayout) -> SessionStopReason:
    """How the session in ``layout`` ended, for FINAL and CLOSE to carry forward; raises ValueError naming what keeps
    FINAL from running over it.

    Nothing the session recorded may still run, whatever its report says: a process of it that is alive would race
    the FINAL run, and a report says how CLOSE went, not that the kill it ordered reached everything.
    A session with no report never closed at all: it was killed before CLOSE could write one, and ended for
    :data:`SessionStopReason.KILLED`.
    A FINAL that ran and chose nothing keeps nothing from running it again: the session still has no winner, which is
    what running FINAL over it a second time is for.
    """

    alive = live_processes(layout.pids)
    if alive:
        raise ValueError(
            f"the session in {layout.root} is still running as {[record.pid for record in alive]}; "
            "stop it before finishing it"
        )
    report_file = layout.report
    if not report_file.is_file():
        return SessionStopReason.KILLED
    try:
        document = read_json_strict(report_file, require_dict=True)
        final, ended = document["final"], SessionStopReason(document["stop_reason"])
        chosen = None if final is None else final["chosen"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{report_file} does not say how the session ended: {exc}") from exc
    if chosen is not None:
        raise ValueError(f"the session {report_file} reports already chose {chosen['commit']} in FINAL")
    if ended is SessionStopReason.PRELUDE_FAILED:
        raise ValueError(f"PRELUDE failed in the session {report_file} reports, so there is no root to validate")
    return ended


def _final_lines(final: FinalChoice | None) -> list[str]:
    if final is None:
        return ["No final selection ran."]
    chosen = final.chosen
    head = f"Chosen: {chosen.label} `{chosen.commit}` on {chosen.branch}" if chosen else "Chosen: none"
    lines = [head, f"Reason: {final.reason}", "", "| candidate | commit | eligible | degraded concs | error |"]
    lines.append("|---|---|---|---|---|")
    for check in final.checks:
        candidate = check.candidate
        degraded = ", ".join(str(conc) for conc in check.degraded_concs)
        lines.append(f"| {candidate.label} | `{candidate.commit}` | {check.eligible} | {degraded} | {check.error} |")
    return lines


def report_markdown(report: CloseReport, session_id: str) -> str:
    """``MB-report.md``: the same outcome for people."""

    lines = [
        f"# Multi-branch session {session_id}",
        "",
        f"Stop reason: {report.stop_reason.value}",
        f"Error: {report.error or 'none'}",
        "",
        "## Final selection",
        "",
        *_final_lines(report.final),
        "",
        "## Close",
        "",
        *_export_lines(report.exports),
        *_stop_lines(report.stop),
    ]
    return "\n".join(lines) + "\n"


def _export_lines(exports: CloseExports | None) -> list[str]:
    if exports is None:
        return ["Exports: not finished"]
    kb = exports.kb_write_back
    knowledge = exports.knowledge
    steps = {
        "Experiences": knowledge.experiences,
        "Learnings": knowledge.learnings,
        "Archive": knowledge.archive,
        "Scorers": exports.scorers,
    }
    return [
        f"Recipe KB write-back: {kb.status} {kb.reason}".rstrip(),
        *(f"{name}: {step.status} {step.reason}".rstrip() for name, step in steps.items()),
        f"Breakdown: {exports.breakdown}",
        f"Dashboard: {exports.dashboard.html}",
    ]


def _stop_lines(stop: StopReport) -> list[str]:
    queue = stop.queue
    return [
        f"Stopped clean: {stop.clean}; processes left: {[record.pid for record in stop.leftovers]}",
        "Queue after the stop: unknown"
        if queue is None
        else f"Queue after the stop: queued {dict(queue.queued_by_branch)}, running {dict(queue.running_by_branch)}",
        *(f"Stop failure: {failure}" for failure in stop.failures),
    ]


class Closer:
    """Runs CLOSE once."""

    def __init__(self, layout: SessionLayout, stopper: SessionStopper, exports: SessionExports, clock: Clock) -> None:
        self._layout = layout
        self._stopper = stopper
        self._exports = exports
        self._clock = clock

    async def close(
        self, stop_reason: SessionStopReason, final: FinalChoice | None, error: str, hard_stop_at: float
    ) -> CloseReport:
        """Stop everything, then export by ``hard_stop_at``; the report is written whatever became of the exports."""

        stop = await self._stopper.stop_everything(stop_reason, hard_stop_at)
        exports = None
        try:
            exports = await self._clock.wait_for(
                self._exports.export(final, stop.queue), hard_stop_at - self._clock.now()
            )
        except EXPORT_FAILURES:
            logger.exception("the CLOSE exports did not finish")
        finally:
            report = CloseReport(stop_reason, final, exports, stop, error)
            self.write_report(report)
        return report

    def write_report(self, report: CloseReport) -> None:
        """Write ``MB-report.json`` and ``MB-report.md``; the session writes a CLOSE it cut short through here."""

        session_id = self._layout.session_id
        atomic_write_json(self._layout.report, report_document(report, session_id, self._clock.now()))
        markdown = report_markdown(report, session_id)
        atomic_write_text(self._layout.report_markdown, markdown, options=WriteOptions(make_parents=True))


__all__ = ["EXPORT_FAILURES", "CloseReport", "Closer", "ended_before_final", "report_document", "report_markdown"]
