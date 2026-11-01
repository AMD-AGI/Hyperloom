# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One multi-branch session, PRELUDE -> EXPLORE -> FINAL -> CLOSE, on the peers a factory builds for it.

PRELUDE gets the whole remaining budget, and spends none of it on a child when it reuses an earlier session's PRELUDE;
EXPLORE then runs until the longest baseline test and one CLOSE grace are left before the deadline. EXPLORE and CLOSE
score the branches with the frontier's defaults for the session's budget. Once the baseline is established, a session
with an RSI archive plays the scheduling playbook ``--playbook`` asks for, the champion of the pool beside it or a
playbook on trial, recorded under ``playbook`` in ``policy_versions``. The FINAL candidates are fixed while every
branch still stands, then exploration ends and FINAL validates them. A failed PRELUDE, a run asked to stop once its
Arbor agents' directions are mapped, an EXPLORE that ran out of cards to measure on, a stop signal, the session deadline
before FINAL starts or an unhandled error skip straight to CLOSE, which always runs and kills every process the session started. FINAL waits for the validation tests it started, past the deadline and the hard
stop if need be, and moves the hard stop to no sooner than one CLOSE grace from its end. The deadline plus the CLOSE
grace bounds CLOSE itself; a stop signal brings that hard stop forward to the CLOSE grace from the signal, once, and the
unwinding of the cancelled phases and CLOSE share it. An unhandled error brings it forward the same way when CLOSE
starts, and so does a first signal that arrives during CLOSE; CLOSE is cut short if it still runs then. A second signal,
whenever it arrives, cuts the unwinding and CLOSE short at once. A CLOSE cut short releases what the session holds,
closing the toolbox within its own short bound and killing every registered process, and writes the report.
A session that ended before FINAL, killed with its machine before CLOSE included, is read back by :func:`persisted_run`
and taken up again by :meth:`Session.finish`, which runs FINAL and CLOSE alone on a budget that is already spent,
carrying the stop reason that session ended with: the roster and the branches are the ones that session persisted,
and only its report, dashboard and breakdown are written again.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.outside_facts.image_check import ImageCheckReport, run_image_check
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.versions import socket_versions
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.interfaces import UsageSink
from hyperloom_multibranch.contracts.blackboard.records import ManifestFingerprint, SessionProfile
from hyperloom_multibranch.contracts.blackboard.vocab import UsageSource
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.codec import encode
from hyperloom_multibranch.blackboard.knowledge.reference_leads import (
    INFERENCEX_REFERENCE,
    ReferenceLead,
    reference_leads,
)
from hyperloom_multibranch.blackboard.rsi.frontier import FrontierPolicy, FrontierScoring
from hyperloom_multibranch.blackboard.rsi.playbook import PlaybookChoice, PlaybookPlay, PlaybookPool, session_playbook
from hyperloom_multibranch.blackboard.rsi.session import FRONTIER_LOG, archived_history
from hyperloom_multibranch.rulegate.critic import FixReviewer
from hyperloom_multibranch.scheduling.agent_plans import (
    AGENT_SOURCES_UNREAD,
    DOSSIER_ENDED,
    DOSSIER_SKIPPED,
    DOSSIER_STARTED,
    GENERATED_FOR,
    MADE_BY_PASS,
    MADE_BY_SHARES,
    MAP_KEPT,
    ROOT_SHARES_READ,
    agent_plans,
    arbor_agents,
    generated_map,
    keep_map,
)
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.budget import HardStop, SessionBudget, budget_seconds, explore_gpu_seconds
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.close import CloseReport, Closer, ended_before_final
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.events import IntegrationProfiler, SessionEvents
from hyperloom_multibranch.scheduling.explore import Explorer, ExploreWindow, Steering, integration_row
from hyperloom_multibranch.scheduling.exports import CloseAuthoring, ExportServices, ExportSources, SessionExports
from hyperloom_multibranch.scheduling.final import FinalCandidates, FinalChoice, FinalSelector, choose_candidates
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.integrator import Integrator, LlmPatchResolver, MergeRecords
from hyperloom_multibranch.scheduling.judge import GRACE_KEY, Judge, session_grace
from hyperloom_multibranch.scheduling.judge_prompt import JUDGE_PROMPT_VERSION, PLAYBOOK_JUDGE_PROMPT_VERSION
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.contracts.llm import Completion
from hyperloom_multibranch.scheduling.llm import recorded
from hyperloom_multibranch.scheduling.manifest import (
    CodeVersion,
    ImageCheckSummary,
    SessionManifest,
    code_version,
    read_manifest,
)
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
from hyperloom_multibranch.scheduling.reference import copy_references
from hyperloom_multibranch.scheduling.root_shares import RootShares, root_shares
from hyperloom_multibranch.scheduling.requests import SessionChecks, SessionTests
from hyperloom_multibranch.scheduling.roster import QUOTA_POLICY_VERSION, Roster, final_quota
from hyperloom_multibranch.scheduling.settings import PlaybookMode, RunSettings
from hyperloom_multibranch.scheduling.stops import (
    TOOLBOX_CLOSE_SECONDS,
    SessionStopper,
    SessionStopReason,
    StopReport,
    StopSignals,
)
from hyperloom_multibranch.scheduling.similarity import BranchSimilarity
from hyperloom_multibranch.scheduling.widening import Widening
from hyperloom_multibranch.specialist.agent_onboarding import ROOT_PROFILE_SOURCE, AttemptFailed
from hyperloom_multibranch.specialist.arbor_adapter import SplitFailed
from hyperloom_multibranch.specialist.prompts import prompt_versions

if TYPE_CHECKING:
    from hyperloom_multibranch.specialist.dossier_pass import DossierOutcome

logger = logging.getLogger(__name__)

SCHEDULER_WRITER = "scheduler"
INTEGRATOR_WRITER = "integrator"
POLICY_VERSIONS = {
    "quota": QUOTA_POLICY_VERSION,
    "scheduling": policy_version(SCHEDULING_POLICY),
}
#: The ``policy_versions`` key of the judge prompt's version: :data:`PLAYBOOK_JUDGE_PROMPT_VERSION` in a session that
#: plays a scheduling playbook, :data:`JUDGE_PROMPT_VERSION` in one that plays none.
JUDGE_PROMPT_KEY = "judge_prompt"
#: The ``policy_versions`` key of the RSI lesson pool's digest as the specialists' recall read it, empty for no pool:
#: two sessions of one build that were shown different past lessons differ here.
LESSONS_KEY = "lessons"
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
#: Why a session that was to stop after its map kept none.
NO_ARBOR_TO_MAP = "no map: no agent named is an Arbor whose directions could be mapped"
NO_MAP = "no map: the dossier pass set none and none could be generated from the root's shares"


@dataclass(frozen=True)
class SessionOutcome:
    """How the session ended; ``close`` is ``None`` only while CLOSE has not run.

    The session completed (exit code 0) only when it ran its course without an error, FINAL chose a commit, and the
    stop left no process and no test behind; or when, asked to stop after its map, it kept one without an error and the
    stop left nothing behind.
    """

    stop_reason: SessionStopReason
    final: FinalChoice | None
    close: CloseReport | None
    error: str

    @property
    def completed(self) -> bool:
        clean = not self.error and self.close is not None and self.close.stop.clean
        if self.stop_reason is SessionStopReason.MAPPED:
            return clean
        chose = self.final is not None and self.final.chosen is not None
        return self.stop_reason in COMPLETED_REASONS and chose and clean

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
    workload, on the scheduler's blackboard handle its spend is recorded through, and the completion CLOSE asks to change
    the scheduling playbook and to curate the RSI lesson pool, with ``author_model`` the model it runs on;
    ``resolver_model`` is the model the patch resolver runs on, which every patch it writes names."""

    judge_completion: Completion
    resolver_completion: Completion
    recipe_kb: RemoteRecipeClient | None
    critic: Callable[[UsageSink], FixReviewer]
    author_completion: Completion
    author_model: str
    resolver_model: str


class Session:
    """Wires one session's scheduler, integrator and blackboard handles onto the peers ``peer_factory`` builds."""

    def __init__(
        self, settings: RunSettings, peer_factory: PeerFactory, host: SessionHost, services: SessionServices
    ) -> None:
        self._settings = settings
        self._layout, self._processes, self._clock = host.layout, host.processes, host.clock
        self._open_records()
        desk = ForkDesk(self._roster, self._decisions, self._clock)
        peers = peer_factory(self._peer_setup(desk, services))
        context = SchedulerContext(
            self._layout,
            self._roster,
            self._blackboard,
            self._decisions,
            self._processes,
            self._clock,
            desk,
            peers.reserved,
        )
        self._toolbox = peers.toolbox
        self._dossier_pass = peers.dossier_pass
        self._briefs = peers.briefs
        self._base_image_digest = peers.base_image_digest
        tests = SessionTests(peers.toolbox, self._clock)
        checks = SessionChecks(tests, peers.gate, peers.regression)
        self._metric = settings.workload.metric
        self._budget_seconds = budget_seconds(settings.max_hours)
        self._score_with_defaults()
        self._context = context
        self._repo = peers.repo
        self._recipe_kb = services.recipe_kb
        self._lifecycle = peers.lifecycle
        self._branches = BranchControl(context, peers.lifecycle, peers.toolbox, peers.repo, peers.briefs)
        self._stopper = SessionStopper(context, self._branches, peers.toolbox, peers.mcp)
        warm_start = None if services.recipe_kb is None else WarmStartSource(services.recipe_kb, self._blobs)
        self._integrator = self._build_integrator(peers.repo, checks, services)
        profiler = IntegrationProfiler(peers.toolbox, tests, self._blackboard, self._clock)
        prelude_services = PreludeServices(
            peers.repo,
            peers.gate,
            tests,
            peers.toolbox,
            peers.on_this_machine,
            self._open_integration,
            self._integrator.replay_champion,
            profiler.profile_root,
        )
        self._prelude = Prelude(context, prelude_services, warm_start=warm_start)
        self._open_llm_roles(services)
        self._similarity = BranchSimilarity(peers.repo, self._roster)
        self._events = SessionEvents(self._blackboard, self._decisions, profiler, peers.toolbox, settings.workload)
        self._selector = FinalSelector(checks, settings.workload)
        self._signals = StopSignals()
        self._in_final = False
        self._image_check: ImageCheckReport | None = None

    def _open_records(self) -> None:
        """The scheduler's blackboard handle, the roster, the decision log and the blob store, on the session's layout."""

        layout, clock = self._layout, self._clock
        self._blackboard = LocalBlackboard(layout.blackboard, Writer(Role.SCHEDULER, SCHEDULER_WRITER), clock=clock.now)
        self._roster = Roster(self._settings.caps, layout.roster, clock)
        self._decisions = DecisionLog(layout.decisions, clock)
        self._blobs = LocalBlobStore(layout.blobs)

    def _peer_setup(self, desk: ForkDesk, services: SessionServices) -> PeerSetup:
        layout, settings = self._layout, self._settings
        return PeerSetup(
            layout,
            settings.workload,
            desk,
            self._processes,
            self._blobs,
            services.critic(self._blackboard),
            self._reference_dirs(),
            settings.dev_leases,
            settings.agents,
            settings.agent_pack,
            settings.agent_launch,
            settings.agent_port,
            settings.agent_dossier,
        )

    def _score_with_defaults(self) -> None:
        """Score with the defaults for the session's budget, against the trees archived by every other session, and
        play no playbook until PRELUDE adopts one."""

        self._play: PlaybookPlay | None = None
        self._policy = FrontierPolicy.for_budget(self._metric, self._budget_seconds)
        # A session is not evidence about itself: a run over a directory whose CLOSE already archived its tree would
        # otherwise set its stall bar and history against that tree.
        self._history = tuple(tree for tree in archived_history() if tree.session_id != self._layout.session_id)

    def _build_integrator(self, repo: ExperimentRepo, checks: SessionChecks, services: SessionServices) -> Integrator:
        """The integrator, writing as itself to the blackboard and the merge log, with the LLM patch resolver whose
        spend the scheduler records."""

        merger = LocalBlackboard(
            self._layout.blackboard, Writer(Role.INTEGRATOR, INTEGRATOR_WRITER), clock=self._clock.now
        )
        resolver = LlmPatchResolver(
            recorded(services.resolver_completion, self._blackboard, UsageSource.RESOLVER),
            repo,
            services.resolver_model,
        )
        records = MergeRecords(merger, DecisionLog(self._layout.merges, self._clock))
        return Integrator(records, repo, checks, resolver, self._clock)

    def _open_llm_roles(self, services: SessionServices) -> None:
        """The judge, and the playbook author and the curator CLOSE asks, each recording its spend on the blackboard."""

        judge = recorded(services.judge_completion, self._blackboard, UsageSource.JUDGE)
        self._judge = Judge(judge, self._layout.judge, self._clock)
        author = recorded(services.author_completion, self._blackboard, UsageSource.AUTHOR)
        self._author = CloseAuthoring(author, services.author_model)
        curator = recorded(services.author_completion, self._blackboard, UsageSource.CURATOR)
        self._curator = CloseAuthoring(curator, services.author_model)

    async def run(self) -> SessionOutcome:
        """Run every phase and CLOSE; an unhandled error propagates once CLOSE has run.

        The scheduler records itself before the first phase, so the directory says the session is running from the
        start and a recovery over it is refused while it is. The image check then compares the rule pack's outside
        facts with this environment, off the event loop; its summary goes into the manifest and CLOSE attaches the
        whole report. Under ``HYPERLOOM_IMAGE_CHECK=strict`` a failing fact raises :class:`ImageCheckFailed` here,
        before the manifest is written and before any phase or CLOSE runs. The reference material the run was given
        is copied in before the manifest, so the manifest the session writes describes a directory that already holds
        it.
        """

        budget = SessionBudget.from_hours(self._settings.max_hours, self._clock.now())
        self._processes.record_scheduler()
        image_check = await asyncio.to_thread(run_image_check)
        self._image_check = image_check
        await asyncio.to_thread(copy_references, self._settings.references, self._layout.reference)
        self._write_manifest(budget, await asyncio.to_thread(code_version), image_check)
        hard_stop = HardStop(budget, self._clock)
        aborted = SessionOutcome(SessionStopReason.ABORTED, None, None, "")
        return await self._run_and_close(self._phases(budget, hard_stop), aborted, budget, hard_stop)

    async def finish(self, ended: SessionStopReason) -> SessionOutcome:
        """Run FINAL and CLOSE over the directory a session that ended for ``ended`` before FINAL left behind, and
        report as that session would have; the manifest and the blackboard that session wrote are this one's input, so
        this opens no session of its own and only its report, dashboard and breakdown are written again. It ends for
        ``ended`` however FINAL goes: the session did not run its course, and a FINAL run after the fact does not
        make it so.

        The candidates are read before anything else happens, so a session FINAL cannot run over is refused without a
        stop or a report being written. The budget is already spent, which is the state the normal path is in when
        FINAL overruns the deadline: FINAL is unbounded and CLOSE gets one CLOSE grace from its end.
        """

        candidates = await self._final_candidates()
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
        except PreludeFailed as exc:
            return SessionOutcome(SessionStopReason.PRELUDE_FAILED, None, None, str(exc))
        await self._adopt_playbook()
        manifest = read_manifest(self._layout.manifest)
        replace(manifest, policy_versions=self._policy_versions()).write(self._layout.manifest)
        sockets = await asyncio.to_thread(socket_versions)
        await self._blackboard.write_manifest_fingerprint(
            ManifestFingerprint(
                activity=self._layout.session_id,
                sockets=sockets,
                prompt_versions=prompt_versions(),
                policy_versions=self._policy_versions(),
            )
        )
        if self._settings.stop_after_map:
            return await self._stop_after_map(prelude)
        plans = await self._agent_plans(prelude)
        explore_end = budget.explore_end(self._clock.now(), prelude.longest_root_test)
        self._phase("explore", explore_end=explore_end)
        window = ExploreWindow(
            prelude.root_commit,
            prelude.start_commit,
            prelude.profile.roofline_ceiling,
            explore_end,
            budget.deadline,
            budget.total_seconds,
            explore_gpu_seconds(self._settings.max_gpu_hours),
            self._settings.max_llm_usd or math.inf,
            prelude.child_gpu_seconds_upper_bound,
            self._settings.user_directions,
            self._settings.agents,
            plans,
        )
        reason = await self._explorer().run(window)
        if reason is SessionStopReason.NO_MEASUREMENT_CARDS:
            await self._stopper.end_exploration(reason)
            return SessionOutcome(reason, None, None, NO_CARDS_LEFT)
        candidates = await self._final_candidates()
        await self._stopper.end_exploration(reason)
        return await self._final(candidates, reason)

    async def _agent_plans(self, prelude: PreludeResult) -> tuple[str, ...]:
        """Each agent branch's share of the directions (:mod:`agent_plans`): of the user's map; for two or more Arbors
        without one, of the dossier pass's map, else of one generated from the root's shares, waited for (its op
        shares, else its kernel shares); for fewer, none. The time this takes is EXPLORE's."""

        settings, work = self._settings, self._layout.agent_plans
        if not settings.agents or settings.agent_directions:
            return await agent_plans(settings.agents, settings.agent_directions, work)
        arbors = await self._arbors()
        if len(arbors) < GENERATED_FOR:
            return await agent_plans(settings.agents, "", work)
        shares = await self._root_shares(prelude)
        outcome = await self._dossier_map(settings.agents[min(arbors)], prelude.start_commit, shares)
        map_file = "" if outcome is None or outcome.map is None else str(outcome.map)
        if shares is None:
            return await agent_plans(settings.agents, map_file, work)
        return await agent_plans(settings.agents, map_file, work, shares.shares, shares.source)

    async def _stop_after_map(self, prelude: PreludeResult) -> SessionOutcome:
        """End the session before EXPLORE once its Arbor agents' directions are mapped, for one Arbor too: the map of
        the dossier pass of the first Arbor, else one generated from the root's shares, kept with the pass's dossier in
        ``MB-agent-map`` for a person to review (:func:`keep_map`). Without either, the session ends with why as its
        error."""

        arbors = await self._arbors()
        if not arbors:
            return SessionOutcome(SessionStopReason.MAPPED, None, None, NO_ARBOR_TO_MAP)
        first = min(arbors)
        source, start_commit = self._settings.agents[first], prelude.start_commit
        shares = await self._root_shares(prelude)
        outcome = await self._dossier_map(source, start_commit, shares)
        passed = "no dossier pass ran" if outcome is None else outcome.reason
        made = {"conc": self._settings.workload.conc, "source": source, "start_commit": start_commit, "pass": passed}
        if outcome is not None and outcome.map is not None:
            kept = keep_map(self._layout.agent_map, outcome.map, outcome.dossier, {**made, "made_by": MADE_BY_PASS})
        else:
            given = (None, ROOT_PROFILE_SOURCE) if shares is None else (shares.shares, shares.source)
            try:
                generated = await generated_map(arbors[first], *given, self._layout.agent_plans)
            except SplitFailed as exc:
                return SessionOutcome(SessionStopReason.MAPPED, None, None, f"{NO_MAP}: {passed}; {exc}")
            dossier = None if outcome is None else outcome.dossier
            kept = keep_map(self._layout.agent_map, generated, dossier, {**made, "made_by": MADE_BY_SHARES})
        self._decisions.record(Decision(DecisionKind.PRELUDE, MAP_KEPT, INTEGRATION_BRANCH, {"summary": str(kept)}))
        return SessionOutcome(SessionStopReason.MAPPED, None, None, "")

    async def _root_shares(self, prelude: PreludeResult) -> RootShares | None:
        """The root's shares, waited for (its op shares, else its kernel shares), recorded as a ``PRELUDE`` decision;
        ``None`` when it has none."""

        read = await root_shares(
            self._blackboard,
            self._toolbox,
            self._clock,
            self._layout.toolbox,
            prelude.root_commit,
            prelude.root_profile_job,
        )
        self._decisions.record(Decision(DecisionKind.PRELUDE, ROOT_SHARES_READ, INTEGRATION_BRANCH, read.detail))
        return read.shares

    async def _arbors(self) -> dict[int, Path]:
        """The Arbor agents of the run, by index; none, recorded, when a source cannot be fetched."""

        try:
            return await arbor_agents(self._settings.agents, self._layout.agent_plans)
        except (AttemptFailed, OSError) as exc:
            detail = {"error": str(exc)}
            self._decisions.record(Decision(DecisionKind.PRELUDE, AGENT_SOURCES_UNREAD, INTEGRATION_BRANCH, detail))
            return {}

    async def _dossier_map(self, source: str, start_commit: str, shares: RootShares | None) -> DossierOutcome | None:
        """How the dossier pass of the Arbor ``source`` on ``start_commit``, given ``shares`` as its profile, ended;
        ``None`` when none ran (no runner, or no minutes for it) or it crashed. Each step is a ``PRELUDE`` decision."""

        from hyperloom_multibranch.specialist.dossier_pass import DossierStart

        minutes = self._settings.agent_dossier_minutes
        if self._dossier_pass is None or minutes <= 0:
            detail = {"minutes": minutes, "runner": self._dossier_pass is not None}
            self._decisions.record(Decision(DecisionKind.PRELUDE, DOSSIER_SKIPPED, INTEGRATION_BRANCH, detail))
            return None
        given = (None, ROOT_PROFILE_SOURCE) if shares is None else (shares.shares, shares.source)
        start = DossierStart(source, start_commit, self._clock.now() + minutes * 60, *given)
        detail = {"source": source, "start_commit": start_commit, "minutes": minutes, "profile": given[1]}
        self._decisions.record(Decision(DecisionKind.PRELUDE, DOSSIER_STARTED, INTEGRATION_BRANCH, detail))
        try:
            outcome = await self._dossier_pass(start)
        except Exception as exc:  # an optional step: the generic map stands in for it
            logger.exception("the dossier pass crashed")
            failed = {"reason": f"it crashed: {exc}", "map": "", "dossier": ""}
            self._decisions.record(Decision(DecisionKind.PRELUDE, DOSSIER_ENDED, INTEGRATION_BRANCH, failed))
            return None
        ended = {"reason": outcome.reason, "map": str(outcome.map or ""), "dossier": str(outcome.dossier or "")}
        self._decisions.record(Decision(DecisionKind.PRELUDE, DOSSIER_ENDED, INTEGRATION_BRANCH, ended))
        return outcome

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

    async def _adopt_playbook(self) -> None:
        """Play the scheduling playbook ``--playbook`` asks for from the pool beside the RSI archive, read off the event
        loop, and record it as a scheduler decision: the champion, or in a challenger session the oldest playbook on
        trial. Without an archive, or under ``HYPERLOOM_RSI_WRITE_ONLY=1``, the session plays none, and runs and is
        judged as one without the RSI."""

        challenger = self._settings.playbook is PlaybookMode.CHALLENGER
        choice = await asyncio.to_thread(_pooled_choice, challenger)
        if choice is None:
            return
        self._play = PlaybookPlay(choice)
        detail = {"playbook": choice.playbook.label, "champion": choice.champion.label, "challenger": choice.challenger}
        self._decisions.record(Decision(DecisionKind.PRELUDE, "scheduling playbook", detail=detail))

    def _policy_versions(self) -> dict[str, str]:
        learnings, play = self._briefs.learnings, self._play
        return {
            **POLICY_VERSIONS,
            JUDGE_PROMPT_KEY: PLAYBOOK_JUDGE_PROMPT_VERSION if play is not None else JUDGE_PROMPT_VERSION,
            GRACE_KEY: str(session_grace(play)),
            LESSONS_KEY: learnings.past_revision() if learnings is not None else "",
            **(play.versions() if play is not None else {}),
        }

    def _scoring(self) -> FrontierScoring:
        log = self._layout.reports / FRONTIER_LOG
        rule = self._play.stop_rule if self._play is not None else None
        return FrontierScoring(self._blackboard, self._policy, self._history, log, rule)

    def _dashboard(self, scoring: FrontierScoring) -> SessionDashboard:
        return SessionDashboard(self._blackboard, scoring, self._toolbox, self._metric, self._layout.dashboard)

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
            self._toolbox,
            self._play,
            self._layout.manifest,
        )
        return Explorer(self._context, self._branches, self._integrator, steering)

    def _closer(self) -> Closer:
        dashboard = self._dashboard(self._scoring())
        sources = ExportSources(
            self._blackboard, self._repo, self._blobs, dashboard, self._image_check, self._briefs.warm
        )
        services = ExportServices(self._recipe_kb, self._author, self._curator)
        closing = SessionExports(self._layout, sources, self._metric, self._budget_seconds, services)
        return Closer(self._layout, self._stopper, closing, self._clock)

    async def _root_session(self, budget: SessionBudget, hard_stop: HardStop) -> PreludeResult:
        """PRELUDE: the root commit with its baseline, the commit EXPLORE opens on and the warm-start leads every
        specialist is briefed on. A reused PRELUDE starts no child."""

        workload, reused = self._settings.workload, self._settings.reuse_prelude
        if reused is None:
            prelude = await self._prelude.root(workload, budget, hard_stop, self._base_image_digest)
        else:
            prelude = await self._prelude.reuse(reused, workload, self._base_image_digest)
        self._briefs.warm = prelude.warm_leads
        self._briefs.references = await asyncio.to_thread(self._reference_leads, prelude.profile)
        return prelude

    async def _open_integration(self, root: str) -> None:
        """Put the integration branch at ``root`` on the roster with its K, which the toolbox is then given."""

        self._roster.add_integration(root, k=1)
        await self._branches.push_quota()

    def _reference_dirs(self) -> dict[str, Path]:
        """Where the session keeps each reference the run was given, by name."""

        return {source.name: self._layout.reference / source.name for source in self._settings.references}

    def _reference_leads(self, profile: SessionProfile) -> tuple[ReferenceLead, ...]:
        """InferenceX's reference launches for ``profile``, when the run was given its checkout. Blocks."""

        source = next((item for item in self._settings.references if item.name == INFERENCEX_REFERENCE), None)
        if source is None:
            return ()
        return reference_leads(self._reference_dirs()[source.name], profile, source.revision)

    async def _final_candidates(self) -> FinalCandidates:
        progress = await self._blackboard.progress()
        integration = integration_row(progress)
        if integration is None:
            raise RuntimeError("the blackboard has no integration branch after PRELUDE recorded its root")
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

    def _write_manifest(self, budget: SessionBudget, version: CodeVersion, image_check: ImageCheckReport) -> None:
        SessionManifest(
            code_version=version,
            rule_pack=dict(rule_pack_identity()),
            session_id=self._layout.session_id,
            started_at=budget.started_at,
            deadline=budget.deadline,
            settings=self._settings,
            metric=self._settings.workload.metric,
            policy_versions=self._policy_versions(),
            image_check=ImageCheckSummary(image_check.summary()),
        ).write(self._layout.manifest)


def _pooled_choice(challenger: bool) -> PlaybookChoice | None:
    """What a session plays from the pool beside the RSI archive the environment names; ``None`` without one."""

    pool = PlaybookPool.from_env()
    return session_playbook(pool, challenger=challenger) if pool is not None else None


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
    "JUDGE_PROMPT_KEY",
    "LESSONS_KEY",
    "NO_CARDS_LEFT",
    "PersistedRun",
    "Session",
    "SessionHost",
    "SessionOutcome",
    "SessionServices",
    "persisted_run",
]
