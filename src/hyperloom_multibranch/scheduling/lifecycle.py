# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One multi-branch session, PRELUDE -> EXPLORE -> FINAL -> CLOSE, on the peers a factory builds for it.

PRELUDE gets the whole remaining budget; whatever it leaves is split into the EXPLORE, FINAL and CLOSE windows. The
FINAL candidates are fixed while every branch still stands, then exploration ends and FINAL validates them. A
failed PRELUDE, a stop signal, the session deadline or an unhandled error skip straight to CLOSE, which always runs
and kills every process the session started. The deadline plus the CLOSE grace bounds CLOSE itself; after a stop
signal the CLOSE grace bounds both the unwinding of the cancelled phases and CLOSE, and a second signal, whenever it
arrives, cuts both short: every registered process is killed and the report written at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import atomic_write_json
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import ManifestFingerprint
from hyperloom_multibranch.contracts.scheduling import QUOTA_POLICY_VERSION, QuotaPolicy
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, FrontierScoring
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.budget import SessionBudget
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.close import CloseReport, Closer
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.explore import Explorer, ExploreWindow, integration_row
from hyperloom_multibranch.scheduling.final import FinalCandidates, FinalChoice, FinalSelector, choose_candidates
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.git import GitFailed, git_output
from hyperloom_multibranch.scheduling.integrator import Integrator, LlmPatchResolver
from hyperloom_multibranch.scheduling.judge import JUDGE_PROMPT_VERSION, Judge
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.llm import Completion
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_PRIMARY_METRIC
from hyperloom_multibranch.scheduling.peers import PeerFactory
from hyperloom_multibranch.scheduling.prelude import Prelude, PreludeFailed, PreludeResult
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.scheduling.stops import SessionStopper, SessionStopReason, StopReport, StopSignals

if TYPE_CHECKING:
    from hyperloom_multibranch.scheduling.exports import ExportsFactory

logger = logging.getLogger(__name__)

SCHEDULER_WRITER = "scheduler"
INTEGRATOR_WRITER = "integrator"
POLICY_VERSIONS = {"quota": QUOTA_POLICY_VERSION, "judge_prompt": JUDGE_PROMPT_VERSION}
#: Stop reasons a session ends with when it ran its course; any other one, or an error, is a failed session.
COMPLETED_REASONS = frozenset({SessionStopReason.TIME_EXHAUSTED, SessionStopReason.GLOBAL_CONVERGED})
#: The distribution this package ships in (pyproject.toml ``[project] name``).
MB_DISTRIBUTION = "hyperloom-inference_optimizer"
CUT_SHORT = "CLOSE was cut short by another stop signal"


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


class Session:
    """Wires one session's scheduler, integrator and notebook handles onto the peers ``peer_factory`` builds.

    ``processes`` registers every process the session starts itself; production passes a registry on
    ``layout.pids``. ``exports`` builds the CLOSE exports, Recipe KB write-back included.
    """

    def __init__(
        self,
        settings: RunSettings,
        layout: SessionLayout,
        peer_factory: PeerFactory,
        processes: ProcessRegistry,
        *,
        judge_completion: Completion,
        resolver_completion: Completion,
        exports: ExportsFactory,
        clock: Clock,
    ) -> None:
        self._settings = settings
        self._layout = layout
        self._processes = processes
        self._clock = clock
        self._notebook = LocalNotebook(layout.notebook, Writer(Role.SCHEDULER, SCHEDULER_WRITER), clock=clock.now)
        merger = LocalNotebook(layout.notebook, Writer(Role.INTEGRATOR, INTEGRATOR_WRITER), clock=clock.now)
        self._roster = Roster(settings.caps, layout.roster, clock)
        self._decisions = DecisionLog(layout.decisions, clock)
        desk = ForkDesk(self._roster, self._decisions)
        merges = DecisionLog(layout.merges, clock)
        peers = peer_factory(layout, desk)
        self._toolbox = peers.toolbox
        self._branches = BranchControl(
            self._roster, self._notebook, peers.lifecycle, peers.toolbox, processes, self._decisions
        )
        self._stopper = SessionStopper(self._roster, self._branches, peers.toolbox, processes, self._decisions, clock)
        progress = FrontierScoring(self._notebook, FrontierPolicy(metric=MB_FAKE_PRIMARY_METRIC))
        resolver = LlmPatchResolver(resolver_completion, peers.repo)
        self._prelude = Prelude(
            layout, processes, self._notebook, peers.repo, peers.toolbox, peers.gate, self._decisions, clock
        )
        self._explorer = Explorer(
            self._roster,
            self._branches,
            desk,
            Judge(judge_completion, layout.judge, clock),
            Integrator(merger, peers.repo, peers.toolbox, peers.gate, resolver, merges),
            progress,
            self._decisions,
            settings.judge_interval_seconds,
            clock,
        )
        self._selector = FinalSelector(
            peers.toolbox, peers.gate, settings.workload, MB_FAKE_PRIMARY_METRIC, layout.final_record, clock
        )
        closing = exports(layout, self._notebook, peers.repo, progress, MB_FAKE_PRIMARY_METRIC)
        self._closer = Closer(layout, self._stopper, closing, clock)
        self._signals = StopSignals()

    async def run(self) -> SessionOutcome:
        """Run every phase and CLOSE; an unhandled error propagates once CLOSE has run."""

        budget = SessionBudget.from_hours(self._settings.max_hours, self._clock.now())
        self._write_manifest(budget, await asyncio.to_thread(code_version))
        self._signals.install()
        try:
            return await self._run_and_close(budget)
        finally:
            self._signals.uninstall()

    async def _run_and_close(self, budget: SessionBudget) -> SessionOutcome:
        ending = SessionOutcome(SessionStopReason.ABORTED, None, None, "")
        try:
            ending = await self._until_stopped(budget)
        except BaseException as exc:
            ending = replace(ending, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            ending = await self._close(ending, budget)
        return ending

    async def _until_stopped(self, budget: SessionBudget) -> SessionOutcome:
        phases = asyncio.ensure_future(self._phases(budget))
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
        else:
            ending = SessionOutcome(
                SessionStopReason.TIME_EXHAUSTED, None, None, "the session deadline passed before FINAL"
            )
        await self._unwind(phases, self._close_by(ending, budget))
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

    async def _phases(self, budget: SessionBudget) -> SessionOutcome:
        caps = self._settings.caps
        await self._toolbox.apply_quota(QuotaPolicy(caps.max_parallel_tests, ()))
        self._phase("prelude")
        try:
            prelude = await self._root_session(budget)
        except PreludeFailed as exc:
            return SessionOutcome(SessionStopReason.PRELUDE_FAILED, None, None, str(exc))
        await self._notebook.write_manifest_fingerprint(
            ManifestFingerprint(activity=self._layout.session_id, policy_versions=POLICY_VERSIONS)
        )
        deadlines = budget.phases_after_prelude(self._clock.now())
        self._phase("explore", explore_end=deadlines.explore_end, final_end=deadlines.final_end)
        window = ExploreWindow(
            prelude.root_commit, prelude.profile.roofline_ceiling, deadlines.explore_end, budget.deadline
        )
        reason = await self._explorer.run(window)
        candidates = await self._final_candidates()
        await self._stopper.end_exploration(reason)
        self._phase("final", candidates=encode(candidates.each))
        final = await self._selector.select(candidates, deadlines.final_end)
        self._decisions.record(Decision(DecisionKind.FINAL, final.reason, detail={"chosen": encode(final.chosen)}))
        return SessionOutcome(reason, final, None, "")

    async def _root_session(self, budget: SessionBudget) -> PreludeResult:
        """PRELUDE: the root commit, the integration branch with its quota, then the baseline measured on it."""

        prelude = await self._prelude.root(self._settings.workload, budget)
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
            self._roster.entries(), progress, integration.head_commit, integration.head_value, MB_FAKE_PRIMARY_METRIC
        )

    async def _close(self, ending: SessionOutcome, budget: SessionBudget) -> SessionOutcome:
        self._phase("close", stop_reason=ending.stop_reason.value, error=ending.error)
        closing = asyncio.ensure_future(
            self._closer.close(ending.stop_reason, ending.final, ending.error, self._close_by(ending, budget))
        )
        cut = asyncio.ensure_future(self._signals.wait_second())
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

    async def _cut_short(self, ending: SessionOutcome) -> SessionOutcome:
        """CLOSE abandoned for a second signal: kill every registered process and report what is still alive."""

        await self._processes.kill_all()
        error = "; ".join(filter(None, (ending.error, CUT_SHORT)))
        stop = StopReport(await self._processes.leftovers(), None, (CUT_SHORT,))
        report = CloseReport(ending.stop_reason, ending.final, None, stop, error)
        self._closer.write_report(report)
        logger.error("%s; every registered process was killed", CUT_SHORT)
        return replace(ending, close=report, error=error)

    def _close_by(self, ending: SessionOutcome, budget: SessionBudget) -> float:
        """The hard stop of CLOSE: after a stop signal the operator waits the CLOSE grace at most, not the budget."""

        if ending.stop_reason is SessionStopReason.SIGNAL:
            return min(budget.hard_stop_at, self._clock.now() + budget.close_grace_seconds)
        return budget.hard_stop_at

    def _phase(self, name: str, **detail: object) -> None:
        self._decisions.record(Decision(DecisionKind.PHASE, name, detail=detail))

    def _write_manifest(self, budget: SessionBudget, version: CodeVersion) -> None:
        document = {
            "code_version": encode(version),
            "session_id": self._layout.session_id,
            "started_at": budget.started_at,
            "deadline": budget.deadline,
            "settings": encode(self._settings),
            "metric": MB_FAKE_PRIMARY_METRIC,
            "policy_versions": POLICY_VERSIONS,
        }
        atomic_write_json(self._layout.manifest, document, indent=2, sort_keys=True)


@dataclass(frozen=True)
class CodeVersion:
    """The installed package version and the git commit of its source tree; empty where there is none."""

    package: str
    git_sha: str


def code_version() -> CodeVersion:
    """This scheduler's :class:`CodeVersion`. Blocks on git."""

    try:
        package = importlib.metadata.version(MB_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        package = ""
    try:
        sha = git_output(Path(__file__).parent, "rev-parse", "HEAD")
    except GitFailed:
        sha = ""
    return CodeVersion(package, sha)


__all__ = ["COMPLETED_REASONS", "CodeVersion", "Session", "SessionOutcome", "code_version"]
