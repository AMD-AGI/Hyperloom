# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One multi-branch session, PRELUDE -> EXPLORE -> FINAL -> CLOSE, on the peers a factory builds for it.

PRELUDE gets the whole remaining budget; whatever it leaves is split into the EXPLORE, FINAL and CLOSE windows. The
FINAL candidates are fixed while every branch still stands, then exploration ends and FINAL validates them. A
failed PRELUDE, an EXPLORE that ran out of cards to measure on, a stop signal, the session deadline or an unhandled error skip straight to CLOSE, which always runs
and kills every process the session started. The deadline plus the CLOSE grace bounds CLOSE itself; a stop signal
brings that hard stop forward to the CLOSE grace from the signal, once, and the unwinding of the cancelled phases
and CLOSE share it. An unhandled error brings it forward the same way when CLOSE starts, and so does a first signal
that arrives during CLOSE; CLOSE is cut short if it still runs then. A second signal, whenever it arrives, cuts the
unwinding and CLOSE short at once. A CLOSE cut short releases what the session holds, closing the toolbox within
its own short bound and killing every registered process, and writes the report.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import logging
import math
import os
from dataclasses import dataclass, replace
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.common.provenance import detect_code_revision
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.versions import socket_versions
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import ManifestFingerprint
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.rsi.session import FRONTIER_LOG, session_frontier
from hyperloom_multibranch.rulegate.critic import FixReviewer
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.budget import HardStop, SessionBudget, budget_seconds, explore_gpu_seconds
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.close import CloseReport, Closer
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.events import IntegrationProfiler, SessionEvents
from hyperloom_multibranch.scheduling.explore import Explorer, ExploreWindow, Steering, integration_row
from hyperloom_multibranch.scheduling.exports import ExportSources, SessionExports
from hyperloom_multibranch.scheduling.final import FinalCandidates, FinalChoice, FinalSelector, choose_candidates
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.integrator import Integrator, LlmPatchResolver, MergeRecords
from hyperloom_multibranch.scheduling.judge import JUDGE_PROMPT_VERSION, Judge
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.llm import Completion, recorded
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeerSetup
from hyperloom_multibranch.scheduling.policy import SCHEDULING_POLICY, policy_version
from hyperloom_multibranch.scheduling.prelude import Prelude, PreludeFailed, PreludeResult, WarmStartSource
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.requests import SessionChecks, SessionTests
from hyperloom_multibranch.scheduling.roster import QUOTA_POLICY_VERSION, Roster
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.scheduling.stops import (
    TOOLBOX_CLOSE_SECONDS,
    SessionStopper,
    SessionStopReason,
    StopReport,
    StopSignals,
)
from hyperloom_multibranch.scheduling.similarity import BranchSimilarity
from hyperloom_multibranch.scheduling.widening import Widening
from hyperloom_multibranch.specialist.prompts import prompt_versions

logger = logging.getLogger(__name__)

SCHEDULER_WRITER = "scheduler"
INTEGRATOR_WRITER = "integrator"
POLICY_VERSIONS = {
    "quota": QUOTA_POLICY_VERSION,
    "judge_prompt": JUDGE_PROMPT_VERSION,
    "scheduling": policy_version(SCHEDULING_POLICY),
}
#: Stop reasons a session ends with when it ran its course, a budget spent as set included; any other one, or an
#: error, is a failed session.
COMPLETED_REASONS = frozenset(
    {
        SessionStopReason.TIME_EXHAUSTED,
        SessionStopReason.GPU_HOURS_EXHAUSTED,
        SessionStopReason.LLM_BUDGET_EXHAUSTED,
        SessionStopReason.GLOBAL_CONVERGED,
    }
)
#: The distribution this package ships in (pyproject.toml ``[project] name``).
MB_DISTRIBUTION = "hyperloom-inference_optimizer"
CUT_SHORT = "CLOSE was cut short after a stop signal"
NO_CARDS_LEFT = "the toolbox has fewer usable cards than one test needs, so FINAL could not run"


@dataclass(frozen=True)
class SessionOutcome:
    """How the session ended; ``close`` is ``None`` only while CLOSE has not run.

    The session completed (exit code 0) only when it ran its course without an error, FINAL chose a commit, and the
    stop left no process and no test behind.
    """

    stop_reason: SessionStopReason
    final: FinalChoice | None
    close: CloseReport | None
    error: str

    @property
    def completed(self) -> bool:
        ran_its_course = self.stop_reason in COMPLETED_REASONS and not self.error
        chose = self.final is not None and self.final.chosen is not None
        return ran_its_course and chose and self.close is not None and self.close.stop.clean

    @property
    def exit_code(self) -> int:
        return 0 if self.completed else 1


@dataclass(frozen=True)
class SessionHost:
    """Where and when the session runs: its directory, the registry of every process it starts itself (production
    passes a registry on ``layout.pids``) and its clock."""

    layout: SessionLayout
    processes: ProcessRegistry
    clock: Clock


@dataclass(frozen=True)
class SessionServices:
    """What the session calls out to besides its peers: the completions the judge and the patch resolver ask, the
    Recipe KB PRELUDE reads the warm-start leads from and CLOSE writes the final recipe back to (``None`` when none is
    configured), and the critic the rule gate asks whether a proven correctness fix matters under the workload."""

    judge_completion: Completion
    resolver_completion: Completion
    recipe_kb: RemoteRecipeClient | None
    critic: FixReviewer


class Session:
    """Wires one session's scheduler, integrator and notebook handles onto the peers ``peer_factory`` builds."""

    def __init__(
        self, settings: RunSettings, peer_factory: PeerFactory, host: SessionHost, services: SessionServices
    ) -> None:
        layout, processes, clock = host.layout, host.processes, host.clock
        self._settings = settings
        self._layout = layout
        self._processes = processes
        self._clock = clock
        self._notebook = LocalNotebook(layout.notebook, Writer(Role.SCHEDULER, SCHEDULER_WRITER), clock=clock.now)
        merger = LocalNotebook(layout.notebook, Writer(Role.INTEGRATOR, INTEGRATOR_WRITER), clock=clock.now)
        self._roster = Roster(settings.caps, layout.roster, clock)
        self._decisions = DecisionLog(layout.decisions, clock)
        desk = ForkDesk(self._roster, self._decisions, clock)
        blobs = LocalBlobStore(layout.blobs)
        peers = peer_factory(PeerSetup(layout, settings.workload, desk, processes, blobs, services.critic))
        context = SchedulerContext(
            layout, self._roster, self._notebook, self._decisions, processes, clock, desk, peers.reserved
        )
        self._toolbox = peers.toolbox
        self._briefs = peers.briefs
        self._base_image_digest = peers.base_image_digest
        tests = SessionTests(peers.toolbox, clock)
        checks = SessionChecks(tests, peers.gate, peers.regression)
        budget = budget_seconds(settings.max_hours)
        metric = settings.workload.metric
        self._branches = BranchControl(context, peers.lifecycle, peers.toolbox, peers.repo, peers.briefs)
        self._stopper = SessionStopper(context, self._branches, peers.toolbox, peers.mcp)
        progress = session_frontier(
            self._notebook, metric, budget_seconds=budget, log_path=layout.reports / FRONTIER_LOG
        )
        dashboard = SessionDashboard(self._notebook, progress, peers.toolbox, metric, layout.dashboard)
        resolver = LlmPatchResolver(
            recorded(services.resolver_completion, self._notebook, UsageSource.RESOLVER), peers.repo
        )
        warm_start = None if services.recipe_kb is None else WarmStartSource(services.recipe_kb, blobs)
        self._prelude = Prelude(context, peers.repo, peers.gate, tests, warm_start=warm_start)
        integrator = Integrator(MergeRecords(merger, DecisionLog(layout.merges, clock)), peers.repo, checks, resolver)
        judge = Judge(recorded(services.judge_completion, self._notebook, UsageSource.JUDGE), layout.judge, clock)
        widening = Widening(peers.toolbox, self._roster, progress.policy)
        similarity = BranchSimilarity(peers.repo, self._roster)
        profiler = IntegrationProfiler(peers.toolbox, tests, self._notebook, clock)
        events = SessionEvents(self._notebook, self._decisions, profiler, peers.toolbox, settings.workload.tp)
        steering = Steering(
            judge,
            settings.judge_interval_seconds,
            progress,
            dashboard,
            widening,
            similarity,
            events,
            peers.lifecycle,
        )
        self._explorer = Explorer(context, self._branches, integrator, steering)
        self._selector = FinalSelector(checks, settings.workload, clock)
        sources = ExportSources(self._notebook, peers.repo, blobs, dashboard)
        closing = SessionExports(layout, sources, metric, budget, kb_client=services.recipe_kb)
        self._closer = Closer(layout, self._stopper, closing, clock)
        self._signals = StopSignals()

    async def run(self) -> SessionOutcome:
        """Run every phase and CLOSE; an unhandled error propagates once CLOSE has run."""

        budget = SessionBudget.from_hours(self._settings.max_hours, self._clock.now())
        self._write_manifest(budget, await asyncio.to_thread(code_version))
        self._signals.install()
        try:
            return await self._run_and_close(budget, HardStop(budget, self._clock))
        finally:
            self._signals.uninstall()

    async def _run_and_close(self, budget: SessionBudget, hard_stop: HardStop) -> SessionOutcome:
        ending = SessionOutcome(SessionStopReason.ABORTED, None, None, "")
        try:
            ending = await self._until_stopped(budget, hard_stop)
        except BaseException as exc:
            ending = replace(ending, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            ending = await self._close(ending, hard_stop)
        return ending

    async def _until_stopped(self, budget: SessionBudget, hard_stop: HardStop) -> SessionOutcome:
        phases = asyncio.ensure_future(self._phases(budget, hard_stop))
        signal = asyncio.ensure_future(self._signals.wait())
        deadline = asyncio.ensure_future(self._clock.sleep(budget.remaining(self._clock.now())))
        try:
            await asyncio.wait({phases, signal, deadline}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            signal.cancel()
            deadline.cancel()
        if phases.done():
            return phases.result()
        if self._signals.received is not None:
            ending = SessionOutcome(SessionStopReason.SIGNAL, None, None, f"stopped by signal {self._signals.received}")
            hard_stop.bring_forward()
        else:
            ending = SessionOutcome(
                SessionStopReason.TIME_EXHAUSTED, None, None, "the session deadline passed before FINAL"
            )
        await self._unwind(phases, hard_stop.at)
        return ending

    async def _unwind(self, phases: asyncio.Future[SessionOutcome], until: float) -> None:
        """Cancel the phases and give them until ``until`` (clock seconds), or a second signal, to unwind.

        CLOSE withdraws every test and kills every registered process, so it covers whatever they did not finish.
        """

        phases.cancel()
        timer = asyncio.ensure_future(self._clock.sleep(until - self._clock.now()))
        cut = asyncio.ensure_future(self._signals.wait_second())
        racing: set[asyncio.Future[Any]] = {phases, timer, cut}
        try:
            await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
        finally:
            timer.cancel()
            cut.cancel()
        if not phases.done():
            logger.error("the cancelled phases did not unwind in time; CLOSE goes ahead without them")
        elif not phases.cancelled():
            phases.result()

    async def _phases(self, budget: SessionBudget, hard_stop: HardStop) -> SessionOutcome:
        await self._toolbox.start()
        await self._branches.push_quota()
        self._phase("prelude")
        self._decisions.record(
            Decision(DecisionKind.PRELUDE, "scheduling policy", detail={"values": dict(SCHEDULING_POLICY)})
        )
        try:
            prelude = await self._root_session(budget, hard_stop)
        except PreludeFailed as exc:
            return SessionOutcome(SessionStopReason.PRELUDE_FAILED, None, None, str(exc))
        sockets = await asyncio.to_thread(socket_versions)
        await self._notebook.write_manifest_fingerprint(
            ManifestFingerprint(
                activity=self._layout.session_id,
                sockets=sockets,
                prompt_versions=prompt_versions(),
                policy_versions=POLICY_VERSIONS,
            )
        )
        deadlines = budget.phases_after_prelude(self._clock.now())
        self._phase("explore", explore_end=deadlines.explore_end, final_end=deadlines.final_end)
        window = ExploreWindow(
            prelude.root_commit,
            prelude.profile.roofline_ceiling,
            deadlines.explore_end,
            budget.deadline,
            budget.total_seconds,
            explore_gpu_seconds(self._settings.max_gpu_hours),
            self._settings.max_llm_usd or math.inf,
            prelude.child_gpu_seconds_upper_bound,
        )
        reason = await self._explorer.run(window)
        if reason is SessionStopReason.NO_MEASUREMENT_CARDS:
            await self._stopper.end_exploration(reason)
            return SessionOutcome(reason, None, None, NO_CARDS_LEFT)
        candidates = await self._final_candidates()
        await self._stopper.end_exploration(reason)
        self._phase("final", candidates=encode(candidates.each))
        final = await self._selector.select(candidates, deadlines.final_end)
        self._decisions.record(Decision(DecisionKind.FINAL, final.reason, detail={"chosen": encode(final.chosen)}))
        return SessionOutcome(reason, final, None, "")

    async def _root_session(self, budget: SessionBudget, hard_stop: HardStop) -> PreludeResult:
        """PRELUDE: the root commit and the warm-start leads every specialist is briefed on, the integration branch
        with its K, then the baseline measured on the root."""

        prelude = await self._prelude.root(self._settings.workload, budget, hard_stop, self._base_image_digest)
        self._briefs.warm = prelude.warm_leads
        self._roster.add_integration(prelude.root_commit, k=1)
        await self._branches.push_quota()
        await self._prelude.measure_baseline(prelude.root_commit)
        return prelude

    async def _final_candidates(self) -> FinalCandidates:
        progress = await self._notebook.progress()
        integration = integration_row(progress)
        if integration is None:
            raise RuntimeError("the notebook has no integration branch after PRELUDE recorded its root")
        return choose_candidates(
            self._roster.entries(),
            progress,
            integration.head_commit,
            integration.head_value,
            self._settings.workload.metric,
        )

    async def _close(self, ending: SessionOutcome, hard_stop: HardStop) -> SessionOutcome:
        self._phase("close", stop_reason=ending.stop_reason.value, error=ending.error)
        if ending.stop_reason is SessionStopReason.ABORTED:
            hard_stop.bring_forward()
        close_by = hard_stop.at
        closing = asyncio.ensure_future(self._closer.close(ending.stop_reason, ending.final, ending.error, close_by))
        cut = asyncio.ensure_future(self._cut(hard_stop, close_by))
        try:
            await asyncio.wait({closing, cut}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            cut.cancel()
        if not closing.done():
            closing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await closing
            return await self._cut_short(ending)
        report = closing.result()
        if not report.stop.clean:
            logger.error(
                "the session stop was not clean: processes left %s, failures %s, queue %s",
                [record.pid for record in report.stop.leftovers],
                list(report.stop.failures),
                report.stop.queue,
            )
        return replace(ending, close=report)

    async def _cut(self, hard_stop: HardStop, close_by: float) -> None:
        """Return when CLOSE has to be cut short: at a second stop signal, or once the CLOSE grace has passed from a
        first one that brings the hard stop before ``close_by``."""

        await self._signals.wait()
        until = hard_stop.bring_forward()
        if until >= close_by:
            await self._signals.wait_second()
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await self._clock.wait_for(self._signals.wait_second(), until - self._clock.now())

    async def _cut_short(self, ending: SessionOutcome) -> SessionOutcome:
        """CLOSE abandoned after a stop signal: release the toolbox and every registered process, and report what is
        still alive."""

        released = await self._stopper.release(TOOLBOX_CLOSE_SECONDS)
        error = "; ".join(filter(None, (ending.error, CUT_SHORT)))
        stop = StopReport(await self._processes.leftovers(), None, (CUT_SHORT, *released))
        report = CloseReport(ending.stop_reason, ending.final, None, stop, error)
        self._closer.write_report(report)
        logger.error("%s; the toolbox was released and every registered process killed", CUT_SHORT)
        return replace(ending, close=report, error=error)

    def _phase(self, name: str, **detail: object) -> None:
        self._decisions.record(Decision(DecisionKind.PHASE, name, detail=detail))

    def _write_manifest(self, budget: SessionBudget, version: CodeVersion) -> None:
        document = {
            "code_version": encode(version),
            "session_id": self._layout.session_id,
            "started_at": budget.started_at,
            "deadline": budget.deadline,
            "settings": encode(self._settings),
            "metric": self._settings.workload.metric,
            "policy_versions": POLICY_VERSIONS,
        }
        atomic_write_json(self._layout.manifest, document)


@dataclass(frozen=True)
class CodeVersion:
    """The installed package version and the source revision :func:`detect_code_revision` reports: the source
    tree's commit, else the revision baked into the image; empty where there is none."""

    package: str
    git_sha: str


def code_version() -> CodeVersion:
    """This scheduler's :class:`CodeVersion`. Blocks on git."""

    try:
        package = importlib.metadata.version(MB_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        package = ""
    return CodeVersion(package, detect_code_revision(os.environ))


__all__ = [
    "COMPLETED_REASONS",
    "NO_CARDS_LEFT",
    "CodeVersion",
    "Session",
    "SessionHost",
    "SessionOutcome",
    "SessionServices",
    "code_version",
]
