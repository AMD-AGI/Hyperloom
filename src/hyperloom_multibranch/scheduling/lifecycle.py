# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One multi-branch session, PRELUDE -> EXPLORE -> FINAL -> CLOSE, on the peers a factory builds for it.

PRELUDE gets the whole remaining budget, and spends none of it on a child when it reuses an earlier session's PRELUDE;
whatever it leaves is split into the EXPLORE, FINAL and CLOSE windows, FINAL's no shorter than the longest baseline
test. Once the baseline is established, the session chooses the frontier policy EXPLORE and CLOSE score with: the scorer
or variation the calibration over sessions like it chooses against the scorer pool's champion, the scorer
``--frontier-scorer`` pins, or the defaults under ``--frontier-policy default``; a scorer that keeps failing falls back
to ``frontier-v1``, recorded as a decision. The FINAL candidates are fixed while every branch still stands, then
exploration ends and FINAL validates them. A failed PRELUDE, an EXPLORE that ran out of cards to measure on, a stop
signal, the session deadline before FINAL starts or an unhandled error skip straight to CLOSE, which always runs and
kills every process the session started. FINAL waits for the validation tests it started, past the deadline and the hard
stop if need be, and moves the hard stop to no sooner than one CLOSE grace from its end. The deadline plus the CLOSE
grace bounds CLOSE itself; a stop signal brings that hard stop forward to the CLOSE grace from the signal, once, and the
unwinding of the cancelled phases and CLOSE share it. An unhandled error brings it forward the same way when CLOSE
starts, and so does a first signal that arrives during CLOSE; CLOSE is cut short if it still runs then. A second signal,
whenever it arrives, cuts the unwinding and CLOSE short at once. A CLOSE cut short releases what the session holds,
closing the toolbox within its own short bound and killing every registered process, and writes the report.
A session that ended before FINAL, killed with its machine before CLOSE included, is read back by :func:`persisted_run`
and taken up again by :meth:`Session.finish`, which runs FINAL and CLOSE alone on a budget that is already spent,
carrying the stop reason that session ended with: the roster, the frontier policy and the branches are the ones that
session persisted, and only its report, dashboard and breakdown are written again.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom.common.rule_pack import rule_pack_identity
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.versions import socket_versions
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.interfaces import UsageSink
from hyperloom_multibranch.contracts.notebook.records import ManifestFingerprint
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.rsi.authoring import session_challenger
from hyperloom_multibranch.notebook.rsi.calibration import DEFAULT_POLICY, PolicyChoice
from hyperloom_multibranch.notebook.rsi.frontier import SETTLED_TESTS, FrontierPolicy, FrontierScoring, ScorerGuard
from hyperloom_multibranch.notebook.rsi.session import (
    FRONTIER_LOG,
    archived_history,
    named_frontier_choice,
    session_frontier_scorer,
)
from hyperloom_multibranch.rulegate.critic import FixReviewer
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.budget import HardStop, SessionBudget, budget_seconds, explore_gpu_seconds
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.close import CloseReport, Closer, ended_before_final
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.events import IntegrationProfiler, SessionEvents
from hyperloom_multibranch.scheduling.explore import Explorer, ExploreWindow, Steering, integration_row
from hyperloom_multibranch.scheduling.exports import ExportServices, ExportSources, ScorerAuthoring, SessionExports
from hyperloom_multibranch.scheduling.final import FinalCandidates, FinalChoice, FinalSelector, choose_candidates
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.integrator import Integrator, LlmPatchResolver, MergeRecords
from hyperloom_multibranch.scheduling.judge import JUDGE_PROMPT_VERSION, Judge
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.llm import Completion, recorded
from hyperloom_multibranch.scheduling.manifest import CodeVersion, SessionManifest, code_version, read_manifest
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeerSetup
from hyperloom_multibranch.scheduling.policy import SCHEDULING_POLICY, policy_version
from hyperloom_multibranch.scheduling.prelude import (
    Prelude,
    PreludeFailed,
    PreludeResult,
    PreludeServices,
    WarmStartSource,
)
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.requests import SessionChecks, SessionTests
from hyperloom_multibranch.scheduling.roster import QUOTA_POLICY_VERSION, Roster, final_quota
from hyperloom_multibranch.scheduling.settings import FrontierPolicyMode, RunSettings
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
#: The ``policy_versions`` key the frontier policy's label is recorded under: the defaults' until PRELUDE chose it.
FRONTIER_POLICY_KEY = "frontier"
#: The ``policy_versions`` key of the scorer the session scores with, which a variation's label does not name.
FRONTIER_SCORER_KEY = "frontier_scorer"
#: The ``policy_versions`` key that says whether the session scored with a challenger, ``"1"``, or not, ``"0"``.
CHALLENGER_KEY = "frontier_challenger"
#: One session in this many scores with a challenger instead of the champion: replaying the archive credits only the
#: scorer a session ran with, since a branch it stopped gains nothing afterwards, so challengers must run sometimes.
CHALLENGER_EVERY = 4
#: The ``policy_versions`` key of the grace, :data:`SETTLED_TESTS`: a scorer's record compares only within one grace.
GRACE_KEY = "grace"
#: The ``policy_versions`` key of the RSI lesson pool's digest as the specialists' recall read it, empty for no pool:
#: two sessions of one build that were shown different past lessons differ here.
LESSONS_KEY = "lessons"
SCORER_FALLBACK_REASON = "frontier scorer fell back"
FORCED_DEFAULT_EVIDENCE = f"{DEFAULT_POLICY}: --frontier-policy default"
PINNED_EVIDENCE = "pinned by --frontier-scorer"
UNCHOSEN_EVIDENCE = f"{DEFAULT_POLICY}: not chosen yet, PRELUDE has not established the baseline"
#: Stop reasons a session ends with when it ran its course, a budget spent as set included; any other one, or an
#: error, is a failed session.
COMPLETED_REASONS = frozenset(
    {
        SessionStopReason.TIME_EXHAUSTED,
        SessionStopReason.GPU_HOURS_EXHAUSTED,
        SessionStopReason.LLM_BUDGET_EXHAUSTED,
        SessionStopReason.JUDGE_STOPPED,
    }
)
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
    configured), what builds the critic the rule gate asks whether a proven correctness fix matters under the
    workload, on the scheduler's notebook handle its spend is recorded through, and the completion CLOSE asks for a
    generation of frontier scorers, with ``scorer_model`` the model it runs on."""

    judge_completion: Completion
    resolver_completion: Completion
    recipe_kb: RemoteRecipeClient | None
    critic: Callable[[UsageSink], FixReviewer]
    scorer_completion: Completion
    scorer_model: str


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
        peers = peer_factory(
            PeerSetup(layout, settings.workload, desk, processes, blobs, services.critic(self._notebook))
        )
        context = SchedulerContext(
            layout, self._roster, self._notebook, self._decisions, processes, clock, desk, peers.reserved
        )
        self._toolbox = peers.toolbox
        self._briefs = peers.briefs
        self._base_image_digest = peers.base_image_digest
        tests = SessionTests(peers.toolbox, clock)
        checks = SessionChecks(tests, peers.gate, peers.regression)
        self._metric = settings.workload.metric
        self._budget_seconds = budget_seconds(settings.max_hours)
        self._challenger = False
        # The defaults until PRELUDE's choice replaces it; EXPLORE and CLOSE build their scoring from it.
        self._choose(
            PolicyChoice(
                FrontierPolicy.for_budget(self._metric, self._budget_seconds), DEFAULT_POLICY, UNCHOSEN_EVIDENCE
            )
        )
        # A session is not evidence about itself: a run over a directory whose CLOSE already archived its tree would
        # otherwise calibrate and score against that tree.
        self._history = tuple(tree for tree in archived_history() if tree.session_id != layout.session_id)
        self._context = context
        self._repo = peers.repo
        self._blobs = blobs
        self._recipe_kb = services.recipe_kb
        self._lifecycle = peers.lifecycle
        self._branches = BranchControl(context, peers.lifecycle, peers.toolbox, peers.repo, peers.briefs)
        self._stopper = SessionStopper(context, self._branches, peers.toolbox, peers.mcp)
        resolver = LlmPatchResolver(
            recorded(services.resolver_completion, self._notebook, UsageSource.RESOLVER), peers.repo
        )
        warm_start = None if services.recipe_kb is None else WarmStartSource(services.recipe_kb, blobs)
        self._prelude = Prelude(
            context, PreludeServices(peers.repo, peers.gate, tests, peers.toolbox), warm_start=warm_start
        )
        self._integrator = Integrator(
            MergeRecords(merger, DecisionLog(layout.merges, clock)), peers.repo, checks, resolver, clock
        )
        self._judge = Judge(recorded(services.judge_completion, self._notebook, UsageSource.JUDGE), layout.judge, clock)
        scorer = recorded(services.scorer_completion, self._notebook, UsageSource.SCORER)
        self._scorers = ScorerAuthoring(scorer, services.scorer_model)
        curator = recorded(services.scorer_completion, self._notebook, UsageSource.CURATOR)
        self._curator = ScorerAuthoring(curator, services.scorer_model)
        self._similarity = BranchSimilarity(peers.repo, self._roster)
        profiler = IntegrationProfiler(peers.toolbox, tests, self._notebook, clock)
        self._events = SessionEvents(self._notebook, self._decisions, profiler, peers.toolbox, settings.workload.tp)
        self._selector = FinalSelector(checks, settings.workload)
        self._signals = StopSignals()
        self._in_final = False

    async def run(self) -> SessionOutcome:
        """Run every phase and CLOSE; an unhandled error propagates once CLOSE has run.

        The scheduler records itself before the first phase, so the directory says the session is running from the
        start and a recovery over it is refused while it is.
        """

        budget = SessionBudget.from_hours(self._settings.max_hours, self._clock.now())
        self._processes.record_scheduler()
        self._write_manifest(budget, await asyncio.to_thread(code_version))
        hard_stop = HardStop(budget, self._clock)
        aborted = SessionOutcome(SessionStopReason.ABORTED, None, None, "")
        return await self._run_and_close(self._phases(budget, hard_stop), aborted, budget, hard_stop)

    async def finish(self, ended: SessionStopReason) -> SessionOutcome:
        """Run FINAL and CLOSE over the directory a session that ended for ``ended`` before FINAL left behind, and
        report as that session would have; the manifest and the notebook that session wrote are this one's input, so
        this opens no session of its own and only its report, dashboard and breakdown are written again. It ends for
        ``ended`` however FINAL goes: the session did not run its course, and a FINAL run after the fact does not
        make it so.

        The candidates and the frontier policy are read before anything else happens, so a session FINAL cannot run
        over is refused without a stop or a report being written. The budget is already spent, which is the state the
        normal path is in when FINAL overruns the deadline: FINAL is unbounded and CLOSE gets one CLOSE grace from
        its end.
        """

        candidates = await self._final_candidates()
        self._choose(self._recorded_frontier_policy())
        self._challenger = read_manifest(self._layout.manifest).policy_versions.get(CHALLENGER_KEY) == "1"
        total = budget_seconds(self._settings.max_hours)
        budget = SessionBudget(self._clock.now() - total, total)
        self._in_final = True
        hard_stop = HardStop(budget, self._clock)
        ending = SessionOutcome(ended, None, None, "")
        return await self._run_and_close(self._final_phases(candidates, ended), ending, budget, hard_stop)

    async def _run_and_close(
        self, work: Awaitable[SessionOutcome], ending: SessionOutcome, budget: SessionBudget, hard_stop: HardStop
    ) -> SessionOutcome:
        """Run ``work``, then CLOSE whatever became of it; ``ending`` is how the session ends when ``work`` raises
        instead of reporting an outcome of its own."""

        self._signals.install()
        try:
            ending = await self._until_stopped(work, budget, hard_stop)
        except BaseException as exc:
            ending = replace(ending, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            ending = await self._close(ending, hard_stop)
            self._signals.uninstall()
        return ending

    async def _until_stopped(
        self, work: Awaitable[SessionOutcome], budget: SessionBudget, hard_stop: HardStop
    ) -> SessionOutcome:
        phases = asyncio.ensure_future(work)
        signal = asyncio.ensure_future(self._signals.wait())
        deadline = asyncio.ensure_future(self._clock.sleep(budget.remaining(self._clock.now())))
        try:
            await asyncio.wait({phases, signal, deadline}, return_when=asyncio.FIRST_COMPLETED)
            if self._in_final:
                await asyncio.wait({phases, signal}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            signal.cancel()
            deadline.cancel()
        if self._in_final:
            hard_stop.follow_final()
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
            longest_root_test = await self._prelude.measure_baseline(prelude.root_commit)
        except PreludeFailed as exc:
            return SessionOutcome(SessionStopReason.PRELUDE_FAILED, None, None, str(exc))
        start_commit = await self._replay_champion(prelude)
        await self._adopt_frontier_policy()
        sockets = await asyncio.to_thread(socket_versions)
        await self._notebook.write_manifest_fingerprint(
            ManifestFingerprint(
                activity=self._layout.session_id,
                sockets=sockets,
                prompt_versions=prompt_versions(),
                policy_versions=self._policy_versions(),
            )
        )
        deadlines = budget.phases_after_prelude(self._clock.now(), longest_root_test)
        self._phase("explore", explore_end=deadlines.explore_end, final_end=deadlines.final_end)
        window = ExploreWindow(
            prelude.root_commit,
            start_commit,
            prelude.profile.roofline_ceiling,
            deadlines.explore_end,
            budget.deadline,
            budget.total_seconds,
            explore_gpu_seconds(self._settings.max_gpu_hours),
            self._settings.max_llm_usd or math.inf,
            prelude.child_gpu_seconds_upper_bound,
            self._settings.user_directions,
            self._settings.agents,
        )
        reason = await self._explorer().run(window)
        if reason is SessionStopReason.NO_MEASUREMENT_CARDS:
            await self._stopper.end_exploration(reason)
            return SessionOutcome(reason, None, None, NO_CARDS_LEFT)
        candidates = await self._final_candidates()
        await self._stopper.end_exploration(reason)
        return await self._final(candidates, reason)

    async def _replay_champion(self, prelude: PreludeResult) -> str:
        """The commit EXPLORE opens on: the root with the Recipe KB's champion replayed onto it, where PRELUDE found
        one and the integrator measured it better than the root, and the root itself otherwise."""

        champion = prelude.champion
        if champion is None:
            return prelude.root_commit
        return await self._integrator.replay_champion(champion, prelude.root_commit)

    async def _final_phases(self, candidates: FinalCandidates, ended: SessionStopReason) -> SessionOutcome:
        """FINAL alone over ``candidates``, inside the phases coroutine, so a failure still goes through CLOSE and
        leaves a report.

        The toolbox takes the dead session's unfinished tests up again from its submit log when it starts, and they
        belong to branches that are over: they are withdrawn, and the quota then admits FINAL's validation alone.
        """

        await self._toolbox.start()
        await self._toolbox.withdraw_all(ended.value)
        await self._toolbox.set_policy(final_quota(self._settings.caps))
        return await self._final(candidates, ended)

    async def _final(self, candidates: FinalCandidates, reason: SessionStopReason) -> SessionOutcome:
        self._in_final = True
        self._phase("final", candidates=encode(candidates.each))
        final = await self._selector.select(candidates)
        self._decisions.record(Decision(DecisionKind.FINAL, final.reason, detail={"chosen": encode(final.chosen)}))
        return SessionOutcome(reason, final, None, "")

    async def _adopt_frontier_policy(self) -> None:
        """Replace the defaults with the frontier policy the session scores with from here on, and record it as a
        scheduler decision and in the manifest: the calibration's choice for the profile PRELUDE recorded, the scorer
        ``--frontier-scorer`` pins, or the defaults under ``--frontier-policy default``."""

        defaults = self._frontier.policy
        if self._settings.frontier_scorer:
            choice = pinned_frontier_choice(self._settings.frontier_scorer, defaults)
        elif self._settings.frontier_policy is FrontierPolicyMode.DEFAULT:
            choice = PolicyChoice(defaults, DEFAULT_POLICY, FORCED_DEFAULT_EVIDENCE)
        else:
            snapshot = await self._notebook.snapshot()
            challenger = None
            if challenger_session(self._layout.session_id):
                challenger = await asyncio.to_thread(session_challenger, snapshot, self._history, defaults)
            self._challenger = challenger is not None
            choice = challenger or await asyncio.to_thread(session_frontier_scorer, snapshot, self._history, defaults)
        self._choose(choice)
        detail = {
            "label": choice.label,
            "evidence": choice.evidence,
            "scorer": choice.scorer.label,
            "challenger": self._challenger,
        }
        self._decisions.record(Decision(DecisionKind.PRELUDE, "frontier policy", detail=detail))
        manifest = read_manifest(self._layout.manifest)
        replace(manifest, policy_versions=self._policy_versions()).write(self._layout.manifest)

    def _recorded_frontier_policy(self) -> PolicyChoice:
        """The frontier policy the manifest records, rebuilt from its label: a run over a session that already chose
        one scores with that choice instead of calibrating again, so the label the manifest holds stays true."""

        label = read_manifest(self._layout.manifest).policy_versions[FRONTIER_POLICY_KEY]
        choice = named_frontier_choice(label, self._frontier.policy, f"{label}: the policy the session recorded")
        if choice is None:
            raise ValueError(f"the frontier policy {label} the manifest records names no scorer any more")
        return choice

    def _policy_versions(self) -> dict[str, str]:
        learnings = self._briefs.learnings
        return {
            **POLICY_VERSIONS,
            FRONTIER_POLICY_KEY: self._frontier.label,
            FRONTIER_SCORER_KEY: self._frontier.scorer.label,
            CHALLENGER_KEY: "1" if self._challenger else "0",
            GRACE_KEY: str(SETTLED_TESTS),
            LESSONS_KEY: learnings.past_revision() if learnings is not None else "",
        }

    def _choose(self, choice: PolicyChoice) -> None:
        """Score with ``choice`` from here on, through one guard that counts its scorer's failures across EXPLORE and
        CLOSE."""

        self._frontier = choice
        self._guard = ScorerGuard(choice.scorer, on_fallback=self._fell_back)

    def _scoring(self) -> FrontierScoring:
        log = self._layout.reports / FRONTIER_LOG
        return FrontierScoring(self._notebook, self._frontier.policy, self._history, log, self._guard)

    def _fell_back(self, why: str) -> None:
        self._decisions.record(Decision(DecisionKind.EVENT, SCORER_FALLBACK_REASON, detail={"why": why}))

    def _dashboard(self, scoring: FrontierScoring) -> SessionDashboard:
        return SessionDashboard(self._notebook, scoring, self._toolbox, self._metric, self._layout.dashboard)

    def _explorer(self) -> Explorer:
        scoring = self._scoring()
        steering = Steering(
            self._judge,
            self._settings.judge_interval_seconds,
            scoring,
            self._dashboard(scoring),
            Widening(self._toolbox, self._roster, scoring.policy, self._settings.workload.tp),
            self._similarity,
            self._events,
            self._lifecycle,
            self._frontier,
        )
        return Explorer(self._context, self._branches, self._integrator, steering)

    def _closer(self) -> Closer:
        sources = ExportSources(self._notebook, self._repo, self._blobs, self._dashboard(self._scoring()))
        services = ExportServices(self._recipe_kb, self._scorers, self._curator)
        closing = SessionExports(self._layout, sources, self._metric, self._budget_seconds, services)
        return Closer(self._layout, self._stopper, closing, self._clock)

    async def _root_session(self, budget: SessionBudget, hard_stop: HardStop) -> PreludeResult:
        """PRELUDE's first step: the root commit and the warm-start leads every specialist is briefed on, and the
        integration branch with its K that the baseline is then measured under. A reused PRELUDE starts no child."""

        workload, reused = self._settings.workload, self._settings.reuse_prelude
        if reused is None:
            prelude = await self._prelude.root(workload, budget, hard_stop, self._base_image_digest)
        else:
            prelude = await self._prelude.reuse(reused, workload, self._base_image_digest)
        self._briefs.warm = prelude.warm_leads
        self._roster.add_integration(prelude.root_commit, k=1)
        await self._branches.push_quota()
        return prelude

    async def _final_candidates(self) -> FinalCandidates:
        progress = await self._notebook.progress()
        integration = integration_row(progress)
        if integration is None:
            raise RuntimeError("the notebook has no integration branch after PRELUDE recorded its root")
        return await choose_candidates(
            progress, integration.head_commit, integration.head_value, self._settings.workload.metric, self._repo
        )

    async def _close(self, ending: SessionOutcome, hard_stop: HardStop) -> SessionOutcome:
        self._phase("close", stop_reason=ending.stop_reason.value, error=ending.error)
        if ending.stop_reason is SessionStopReason.ABORTED:
            hard_stop.bring_forward()
        close_by = hard_stop.at
        closer = self._closer()
        closing = asyncio.ensure_future(closer.close(ending.stop_reason, ending.final, ending.error, close_by))
        cut = asyncio.ensure_future(self._cut(hard_stop, close_by))
        try:
            await asyncio.wait({closing, cut}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            cut.cancel()
        if not closing.done():
            closing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await closing
            return await self._cut_short(ending, closer)
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

    async def _cut_short(self, ending: SessionOutcome, closer: Closer) -> SessionOutcome:
        """CLOSE abandoned after a stop signal: release the toolbox and every registered process, and report what is
        still alive."""

        released = await self._stopper.release(TOOLBOX_CLOSE_SECONDS)
        error = "; ".join(filter(None, (ending.error, CUT_SHORT)))
        stop = StopReport(await self._processes.leftovers(), None, (CUT_SHORT, *released))
        report = CloseReport(ending.stop_reason, ending.final, None, stop, error)
        closer.write_report(report)
        logger.error("%s; the toolbox was released and every registered process killed", CUT_SHORT)
        return replace(ending, close=report, error=error)

    def _phase(self, name: str, **detail: object) -> None:
        self._decisions.record(Decision(DecisionKind.PHASE, name, detail=detail))

    def _write_manifest(self, budget: SessionBudget, version: CodeVersion) -> None:
        SessionManifest(
            code_version=version,
            rule_pack=dict(rule_pack_identity()),
            session_id=self._layout.session_id,
            started_at=budget.started_at,
            deadline=budget.deadline,
            settings=self._settings,
            metric=self._settings.workload.metric,
            policy_versions=self._policy_versions(),
        ).write(self._layout.manifest)


def challenger_session(session_id: str) -> bool:
    """Whether the session ``session_id`` is one of the :data:`CHALLENGER_EVERY` that scores with a challenger; the
    same answer for the same session every time it is asked."""

    return int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % CHALLENGER_EVERY == 0


def pinned_frontier_choice(label: str, defaults: FrontierPolicy) -> PolicyChoice:
    """The choice ``--frontier-scorer`` pins with ``label`` under ``defaults``; raise ValueError when no scorer in code
    or in the scorer pool beside the archive has that label."""

    choice = named_frontier_choice(label, defaults, f"{label}: {PINNED_EVIDENCE}")
    if choice is None:
        raise ValueError(f"--frontier-scorer {label} names no frontier scorer in code or in the scorer pool")
    return choice


@dataclass(frozen=True)
class PersistedRun:
    """A session directory that ended without FINAL, as :meth:`Session.finish` takes it again; ``ended`` is the stop
    reason that session reported, or :data:`SessionStopReason.KILLED` when it was killed before it could report one,
    and the second report carries it."""

    layout: SessionLayout
    settings: RunSettings
    ended: SessionStopReason


def persisted_run(root: Path) -> PersistedRun:
    """The session in ``root``, read back from what :meth:`Session._write_manifest` and CLOSE left there, or from what
    a session that was killed left without a CLOSE; raises ValueError naming what keeps FINAL from running over it.
    Creates nothing. Blocks."""

    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")
    layout = SessionLayout(root)
    try:
        manifest = read_manifest(layout.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{root} is no multi-branch session that ran: {exc}") from exc
    ran_on, here = manifest.code_version, code_version()
    if ran_on != here:
        raise ValueError(
            f"the session in {root} ran on {ran_on} and this build is {here}; FINAL and CLOSE read back what that "
            "build persisted, so run them from it"
        )
    if not layout.root_recipe.is_file():
        raise ValueError(f"the session in {root} committed no root, so FINAL has nothing to validate")
    return PersistedRun(layout, manifest.settings, ended_before_final(layout))


__all__ = [
    "COMPLETED_REASONS",
    "CHALLENGER_EVERY",
    "CHALLENGER_KEY",
    "FORCED_DEFAULT_EVIDENCE",
    "FRONTIER_SCORER_KEY",
    "GRACE_KEY",
    "LESSONS_KEY",
    "SCORER_FALLBACK_REASON",
    "PINNED_EVIDENCE",
    "FRONTIER_POLICY_KEY",
    "NO_CARDS_LEFT",
    "PersistedRun",
    "Session",
    "SessionHost",
    "SessionOutcome",
    "SessionServices",
    "persisted_run",
    "challenger_session",
    "pinned_frontier_choice",
]
