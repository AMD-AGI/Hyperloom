# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One multi-branch session, PRELUDE -> EXPLORE -> FINAL -> CLOSE, on the peers a factory builds for it.

PRELUDE roots the session and measures its baseline (:mod:`.prelude`); EXPLORE then runs until the longest baseline
test and one CLOSE grace are left before the deadline. EXPLORE and CLOSE score the branches with the frontier's
defaults for the session's budget. Once the baseline is established, a session with an RSI archive plays the
scheduling playbook ``--playbook`` asks for, the champion of the pool beside it or a playbook on trial, as the session's id draws under
``auto``, recorded under ``playbook`` in ``policy_versions``. The FINAL candidates are fixed while every branch still stands,
then exploration ends and FINAL validates them. With the review on, a session whose Arbor agents are given a map pauses before their branches open until a person approves it (:mod:`.review`); the pause moves the
deadline, and EXPLORE's end, by its length. A map a mapping branch makes is reviewed in EXPLORE, while the clock runs,
and its shares open only with ``--agent-share-minutes`` left before their cutoff. A failed PRELUDE, a run asked to stop once its Arbor agents' directions are mapped,
a map a person rejected, an EXPLORE that ran out of cards to measure on, a stop signal, the session deadline
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
and only its report, dashboard and breakdown are written again. One that ended before PRELUDE recorded its baseline
(:func:`prelude_landed`) has no candidate: the finish withdraws its unfinished tests and CLOSE reports why nothing was
chosen.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from hyperloom.common.outside_facts.image_check import ImageCheckReport, run_image_check
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.versions import socket_versions
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.contracts.fork import ShareRequest
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.interfaces import NotRecorded, UsageSink
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, ManifestFingerprint, SessionProfile
from hyperloom_multibranch.contracts.blackboard.vocab import UsageSource
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.codec import encode
from hyperloom_multibranch.blackboard.knowledge.reference_leads import (
    INFERENCEX_REFERENCE,
    ReferenceLead,
    reference_leads,
)
from hyperloom_multibranch.blackboard.rsi.frontier import FrontierPolicy, FrontierScoring
from hyperloom_multibranch.blackboard.rsi.playbook import (
    PlaybookChoice,
    PlaybookPlay,
    PlaybookPool,
    drawn_challenger,
    session_playbook,
)
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
    SHARES_TOO_LATE,
    agent_plans,
    arbor_agents,
    generated_map,
    keep_map,
)
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.budget import HardStop, SessionBudget, budget_seconds, explore_gpu_seconds
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.checkpoint_tally import CheckpointScope
from hyperloom_multibranch.scheduling.close import CloseReport, Closer, ended_before_final
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.dashboard import SessionDashboard
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.events import IntegrationProfiler, SessionEvents
from hyperloom_multibranch.scheduling.explore import Explorer, ExploreWindow, Steering, integration_row
from hyperloom_multibranch.scheduling.exports import CloseAuthoring, ExportServices, ExportSources, SessionExports
from hyperloom_multibranch.scheduling.final import FinalCandidates, FinalChoice, FinalSelector, choose_candidates
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.integrator import BundlingIntegrator, Integrator, LlmPatchResolver, MergeRecords
from hyperloom_multibranch.scheduling.judge import GRACE_KEY, Judge, session_grace
from hyperloom_multibranch.scheduling.judge_prompt import JUDGE_PROMPT_VERSION, PLAYBOOK_JUDGE_PROMPT_VERSION
from hyperloom_multibranch.scheduling.launch_isolation import LaunchIsolation
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
from hyperloom_multibranch.scheduling.peers import PeerFactory, Peers, PeerSetup
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
from hyperloom_multibranch.scheduling.review import (
    MADE_BY_USER,
    POLL_SECONDS,
    REJECTED,
    REVIEW_APPROVED,
    REVIEW_PAGE,
    REVIEW_REJECTED,
    REVIEW_REQUESTED,
    REVIEW_SKIPPED,
    Reviewed,
    ReviewRejected,
    request,
    wait,
)
from hyperloom_multibranch.scheduling.root_shares import RootShares, ShareSources, root_shares
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
from hyperloom_multibranch.specialist.arbor_adapter import REVIEW_HOST, REVIEW_OFF, SplitFailed
from hyperloom_multibranch.specialist.map_session import MapCalls
from hyperloom_multibranch.specialist.prompts import prompt_versions

if TYPE_CHECKING:
    from hyperloom_multibranch.specialist.dossier_pass import DossierOutcome, DossierStart

logger = logging.getLogger(__name__)


class AgentBranches(NamedTuple):
    """The agents EXPLORE opens a branch for at once, each one's share of the directions, and the Arbor agents a
    mapping branch maps for (empty for none)."""

    agents: tuple[str, ...]
    plans: tuple[str, ...]
    map_for: tuple[str, ...] = ()


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
#: Seconds in a minute, the unit of the agents' time options.
MINUTE = 60.0
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
#: Why FINAL over a session that ended before its baseline was recorded chose nothing.
NO_BASELINE = "no candidate: the session ended before PRELUDE recorded its baseline, so FINAL had nothing to validate"
#: The blackboard writer name a restart reads the baseline as.
RESTART_READER = "restart"
#: Why a session that was to stop after its map kept none.
NO_ARBOR_TO_MAP = "no map: no agent named is an Arbor whose directions could be mapped"
NO_MAP = "no map: the dossier pass set none and none could be generated from the root's shares"


@dataclass(frozen=True)
class SessionOutcome:
    """How the session ended; ``close`` is ``None`` only while CLOSE has not run.

    The session completed (exit code 0) only when it ran its course without an error, FINAL chose a commit, and the
    stop left no process and no test behind; or when, asked to stop after its map, it kept one without an error and the
    stop left nothing behind; or when a person rejected its agent map at its review and the stop left nothing behind.
    """

    stop_reason: SessionStopReason
    final: FinalChoice | None
    close: CloseReport | None
    error: str

    @property
    def completed(self) -> bool:
        clean = not self.error and self.close is not None and self.close.stop.clean
        if self.stop_reason in (SessionStopReason.MAPPED, SessionStopReason.REVIEW_REJECTED):
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
        self._desk = desk
        self._prelude: PreludeResult | None = None
        self._map_shares: tuple[RootShares | None] | None = None
        self._explore_end = math.inf
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
        self._integrator = self._build_integrator(peers.repo, checks, services)
        profiler = IntegrationProfiler(peers.toolbox, tests, self._blackboard, self._clock)
        self._prelude = self._build_prelude(peers, tests, profiler, services)
        self._open_llm_roles(services)
        self._similarity = BranchSimilarity(peers.repo, self._roster)
        self._events = SessionEvents(self._blackboard, self._decisions, profiler, peers.toolbox, settings.workload)
        self._selector = FinalSelector(checks, settings.workload)
        self._signals = StopSignals()
        self._in_final = False
        self._image_check: ImageCheckReport | None = None
        self._agent_isolation: LaunchIsolation | None = peers.isolation
        # The budget and hard stop the session's deadline is read from, which a person's review moves by its length
        # (_extend); while the review is pending the deadline does not pass.
        self._budget: SessionBudget | None = None
        self._hard_stop: HardStop | None = None
        self._reviewing = False

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
            settings.reference_script,
            settings.agent_pack,
            settings.agent_launch,
            settings.agent_port,
            settings.agent_dossier,
            # The session's review stands in for each Arbor's own; without one, nobody's does.
            REVIEW_HOST if settings.review_gate else REVIEW_OFF,
            MapCalls(self._map_start, self._map_split) if settings.agents else None,
            settings.agent_share_minutes,
            settings.inferencex_commit,
            settings.agentx_early_stop,
        )

    def _score_with_defaults(self) -> None:
        """Score with the defaults for the session's budget, against the trees archived by every other session, and
        play no playbook until PRELUDE adopts one."""

        self._play: PlaybookPlay | None = None
        self._policy = FrontierPolicy.for_budget(self._metric, self._budget_seconds)
        # A session is not evidence about itself: a run over a directory whose CLOSE already archived its tree would
        # otherwise set its stall bar and history against that tree.
        self._history = tuple(tree for tree in archived_history() if tree.session_id != self._layout.session_id)

    def _build_prelude(
        self, peers: Peers, tests: SessionTests, profiler: IntegrationProfiler, services: SessionServices
    ) -> Prelude:
        """PRELUDE on the peers' repository, gate and toolbox: it opens the integration branch, replays the champion
        recipe on it and profiles the root, warm-started from the Recipe KB when the session has one."""

        warm_start = None if services.recipe_kb is None else WarmStartSource(services.recipe_kb, self._blobs)
        prelude_services = PreludeServices(
            peers.repo,
            peers.gate,
            tests,
            peers.toolbox,
            self._open_integration,
            self._integrator.replay_champion,
            profiler.profile_root,
        )
        return Prelude(self._context, prelude_services, warm_start=warm_start)

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
        integrator = BundlingIntegrator if self._settings.workload.bundles_merges else Integrator
        return integrator(records, repo, checks, resolver, self._clock)

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
        before the manifest is written and before any phase or CLOSE runs. How the peers isolate the agents and tests
        they launch goes into the manifest and the report. The reference material the run was given is copied in before
        the manifest, so the manifest the session writes describes a directory that already holds it.
        """

        budget = SessionBudget.from_hours(self._settings.max_hours, self._clock.now())
        self._processes.record_scheduler()
        image_check = await asyncio.to_thread(run_image_check)
        self._image_check = image_check
        await asyncio.to_thread(copy_references, self._settings.references, self._layout.reference)
        self._write_manifest(budget, await asyncio.to_thread(code_version), image_check)
        hard_stop = HardStop(budget, self._clock)
        aborted = SessionOutcome(SessionStopReason.ABORTED, None, None, "")
        return await self._run_and_close(self._phases(budget), aborted, budget, hard_stop)

    async def finish(self, ended: SessionStopReason) -> SessionOutcome:
        """Run FINAL and CLOSE over the directory a session that ended for ``ended`` before FINAL left behind, and
        report as that session would have; the manifest and the blackboard that session wrote are this one's input, so
        this opens no session of its own and only its report, dashboard and breakdown are written again. It ends for
        ``ended`` however FINAL goes: the session did not run its course, and a FINAL run after the fact does not
        make it so.

        The candidates are read before anything else happens, so a session FINAL cannot run over is refused without a
        stop or a report being written. A session that ended before PRELUDE recorded its baseline has no candidate:
        its unfinished tests are withdrawn and CLOSE reports :data:`NO_BASELINE` (:meth:`_no_final`). The budget is already spent, which is the state the normal path is in when
        FINAL overruns the deadline: FINAL is unbounded and CLOSE gets one CLOSE grace from its end. The report says
        how that session's agents were isolated, as its manifest recorded.
        """

        landed = await recorded_baseline(self._blackboard) is not None
        candidates = await self._final_candidates() if landed else None
        self._agent_isolation = read_manifest(self._layout.manifest).agent_isolation
        total = budget_seconds(self._settings.max_hours)
        budget = SessionBudget(self._clock.now() - total, total)
        self._in_final = True
        hard_stop = HardStop(budget, self._clock)
        ending = SessionOutcome(ended, None, None, "")
        work = self._no_final(ended) if candidates is None else self._final_phases(candidates, ended)
        return await self._run_and_close(work, ending, budget, hard_stop)

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
            try:
                ending = await self._close(ending, hard_stop)
            finally:
                self._signals.uninstall()
        return ending

    async def _until_stopped(
        self, work: Awaitable[SessionOutcome], budget: SessionBudget, hard_stop: HardStop
    ) -> SessionOutcome:
        self._budget, self._hard_stop = budget, hard_stop
        phases = asyncio.ensure_future(work)
        signal = asyncio.ensure_future(self._signals.wait())
        deadline = asyncio.ensure_future(self._past_deadline())
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

    async def _past_deadline(self) -> None:
        """Return once the session's deadline passes: its budget's, never while a person's review is pending, and
        later once the review moved it (:meth:`_extend`)."""

        while self._budget is not None:
            left = self._budget.remaining(self._clock.now())
            if left <= 0 and not self._reviewing:
                return
            await self._clock.sleep(left if left > 0 else POLL_SECONDS)

    def _extend(self, seconds: float) -> None:
        """Move the session's deadline and its hard stop ``seconds`` later: the time it paused for a person's review,
        which does not count against ``--max-hours``."""

        if self._budget is None or seconds <= 0:
            return
        self._budget = replace(self._budget, total_seconds=self._budget.total_seconds + seconds)
        if self._hard_stop is not None:
            self._hard_stop.extend(seconds)

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
        await self._toolbox.start()
        await self._branches.push_quota()
        self._phase("prelude")
        self._decisions.record(
            Decision(DecisionKind.PRELUDE, "scheduling policy", detail={"values": dict(SCHEDULING_POLICY)})
        )
        try:
            prelude = await self._root_session()
        except PreludeFailed as exc:
            return SessionOutcome(SessionStopReason.PRELUDE_FAILED, None, None, str(exc))
        await self._adopt_playbook()
        await self._record_versions()
        if self._settings.stop_after_map:
            # Its stop is the review point: the map is kept and reviewed offline, so the session does not pause.
            return await self._stop_after_map(prelude)
        self._prelude = prelude
        try:
            opening = await self._agent_branches(prelude)
        except ReviewRejected as exc:
            logger.info("the agent map was rejected at its review: %s", exc)
            return SessionOutcome(SessionStopReason.REVIEW_REJECTED, None, None, "")
        budget = self._budget or budget  # a review's pause moved the deadline
        explore_end = self._explore_end = budget.explore_end(self._clock.now(), prelude.longest_root_test)
        self._phase("explore", explore_end=explore_end)
        window = ExploreWindow(
            prelude.root_commit,
            prelude.start_commit,
            explore_end,
            budget.deadline,
            budget.total_seconds,
            explore_gpu_seconds(self._settings.max_gpu_hours),
            self._settings.max_llm_usd or math.inf,
            self._settings.user_directions,
            opening.agents,
            opening.plans,
            opening.map_for,
        )
        reason = await self._explorer().run(window)
        if reason is SessionStopReason.NO_MEASUREMENT_CARDS:
            await self._stopper.end_exploration(reason)
            return SessionOutcome(reason, None, None, NO_CARDS_LEFT)
        candidates = await self._final_candidates()
        await self._stopper.end_exploration(reason)
        return await self._final(candidates, reason)

    async def _record_versions(self) -> None:
        """The policy versions the session runs, once PRELUDE settled its playbook: in the manifest, and with the
        sockets' and the prompts' versions in the session's fingerprint on the blackboard."""

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

    async def _agent_branches(self, prelude: PreludeResult) -> AgentBranches:
        """The agents whose branches EXPLORE opens at once, each one's share of the directions (:mod:`agent_plans`),
        and the Arbor agents a mapping branch maps for (:mod:`~hyperloom_multibranch.specialist.map_session`).

        With the user's map, every agent opens on its share of it, after a person reviewed it when the review is on.
        Without one, two or more Arbors, or one under the review, wait for a mapping branch, which EXPLORE opens beside
        the others: its map is the dossier pass's, else one generated from the root's shares, and is reviewed, split
        and handed out as share branches (:meth:`_map_split`), so no other branch waits for it. Fewer Arbors map their
        own directions and open at once. Raises :class:`~hyperloom_multibranch.scheduling.review.ReviewRejected` when
        the person rejects the user's map."""

        settings, work = self._settings, self._layout.agent_plans
        if settings.agents and not settings.review_gate:
            self._decisions.record(Decision(DecisionKind.PRELUDE, REVIEW_SKIPPED, INTEGRATION_BRANCH, {}))
        if not settings.agents or settings.agent_directions:
            if settings.agent_directions and settings.review_gate and await self._arbors():
                dossier = Path(settings.agent_dossier) if settings.agent_dossier else None
                await self._review(Path(settings.agent_directions), dossier, MADE_BY_USER)
            return AgentBranches(settings.agents, await agent_plans(settings.agents, settings.agent_directions, work))
        arbors = await self._arbors()
        if not arbors or (len(arbors) < GENERATED_FOR and not settings.review_gate):
            return AgentBranches(settings.agents, await agent_plans(settings.agents, "", work))
        others = tuple(agent for index, agent in enumerate(settings.agents) if index not in arbors)
        return AgentBranches(others, ("",) * len(others), tuple(settings.agents[index] for index in sorted(arbors)))

    async def _map_start(self, source: str, start_commit: str, deadline: float) -> DossierStart | None:
        """The dossier pass a mapping branch runs on the Arbor ``source`` at ``start_commit``, by ``deadline`` or
        sooner by ``--agent-dossier-minutes``, with the root's shares as its profile; ``None`` with no minutes for it."""

        from hyperloom_multibranch.specialist.dossier_pass import DossierStart

        minutes = self._settings.agent_dossier_minutes
        shares = None if self._prelude is None else await self._mapped_shares(self._prelude)
        if minutes <= 0:
            self._decisions.record(
                Decision(DecisionKind.SHARE, DOSSIER_SKIPPED, INTEGRATION_BRANCH, {"minutes": minutes})
            )
            return None
        given = (None, ROOT_PROFILE_SOURCE) if shares is None else (shares.shares, shares.source)
        ends = self._clock.now() + minutes * MINUTE
        detail = {"source": source, "start_commit": start_commit, "minutes": minutes, "profile": given[1]}
        self._decisions.record(Decision(DecisionKind.SHARE, DOSSIER_STARTED, INTEGRATION_BRANCH, detail))
        return DossierStart(source, start_commit, min(ends, deadline) if deadline else ends, *given)

    async def _mapped_shares(self, prelude: PreludeResult) -> RootShares | None:
        """The root's shares a mapping branch maps and splits by, read once."""

        if self._map_shares is None:
            self._map_shares = (await self._root_shares(prelude),)
        return self._map_shares[0]

    async def _map_split(
        self, branch: str, start_commit: str, map_for: tuple[str, ...], outcome: DossierOutcome | None
    ) -> str:
        """Hand the map of the mapping branch ``branch`` out to the Arbor agents ``map_for``: the pass's map, else one
        generated from the root's shares, reviewed by a person when the session's review is on, while the other
        branches go on; split into one share each; and asked of the fork desk as share branches cut at
        ``start_commit``. A rejected map opens none, and neither does a split too late for the shares to run an Arbor
        iteration (:meth:`_too_late`), checked before the review and after it. What was done, for the branch log."""

        if outcome is not None:
            ended = {"reason": outcome.reason, "map": str(outcome.map or ""), "dossier": str(outcome.dossier or "")}
            self._decisions.record(Decision(DecisionKind.SHARE, DOSSIER_ENDED, branch, ended))
        if late := await self._too_late(branch, map_for, "before the review"):
            return late
        shares = None if self._prelude is None else await self._mapped_shares(self._prelude)
        map_file = "" if outcome is None or outcome.map is None else str(outcome.map)
        if self._settings.review_gate:
            made_by = MADE_BY_PASS if map_file else MADE_BY_SHARES
            arbors = await self._arbors()
            map_file = map_file or (await self._generated(arbors, shares) if arbors else "")
            dossier = None if outcome is None else outcome.dossier
            try:
                await self._review(Path(map_file) if map_file else None, dossier, made_by, branch=branch)
            except ReviewRejected as exc:
                return f"the map was rejected at its review ({exc}); no share branch opens"
            if late := await self._too_late(branch, map_for, "after the review"):
                return late
        work = self._layout.agent_plans
        given = () if shares is None else (shares.shares, shares.source)
        plans = await agent_plans(map_for, map_file, work, *given)
        answers = []
        for agent, plan in zip(map_for, plans):
            admission = await self._desk.admit_share(ShareRequest(branch, start_commit, agent, plan))
            answers.append(f"{admission.branch or admission.reason} ({admission.status.value})")
        split = "split" if any(plans) else "not split, each Arbor maps its own"
        return f"{len(map_for)} share branches asked, the map {split}: {', '.join(answers)}"

    async def _too_late(self, branch: str, map_for: tuple[str, ...], step: str) -> str:
        """Why the shares of ``map_for`` do not open when each would now run less than ``--agent-share-minutes``
        before its cutoff, recorded as a ``SHARE`` decision of the mapping ``branch`` at ``step``; ``""`` when they
        may open."""

        need = self._settings.agent_share_minutes
        runway, _open_by = await self._share_time()
        minutes = runway / MINUTE
        if minutes >= need:
            return ""
        detail = {"step": step, "agents": len(map_for), "minutes": round(minutes, 1), "floor": need}
        self._decisions.record(Decision(DecisionKind.SHARE, SHARES_TOO_LATE, branch, detail))
        return (
            f"no share branch opens ({step}): each would run {minutes:.0f} min before its cutoff, under the"
            f" {need:g} an Arbor iteration needs (--agent-share-minutes)"
        )

    async def _share_time(self) -> tuple[float, float]:
        """How long a share branch opened now would run before its cutoff, as its agent reckons it by the session's
        tests so far, and about when that falls under ``--agent-share-minutes``."""

        from hyperloom_multibranch.specialist.autonomous_session import CUTOFF_SECONDS, collection_margin, run_seconds

        records = (await self._blackboard.snapshot()).records
        runway = run_seconds(self._explore_end - self._clock.now(), records, CUTOFF_SECONDS)
        margin = collection_margin(records, CUTOFF_SECONDS)
        return runway, self._explore_end - margin - self._settings.agent_share_minutes * MINUTE

    async def _generated(self, arbors: dict[int, Path], shares: RootShares | None) -> str:
        """The map the first Arbor generates from the root's ``shares``, for a person to review before it is split;
        ``""`` when there are none, or Arbor cannot."""

        given = (None, ROOT_PROFILE_SOURCE) if shares is None else (shares.shares, shares.source)
        try:
            return str(await generated_map(arbors[min(arbors)], *given, self._layout.agent_plans))
        except SplitFailed as exc:
            logger.warning("no map could be generated for the review: %s", exc)
            return ""

    async def _review(self, map_file: Path | None, dossier: Path | None, made_by: str, *, branch: str = "") -> None:
        """Wait, holding no lease, until a person decides on ``map_file`` and ``dossier`` (:mod:`.review`). Before
        EXPLORE the session pauses: each step is a ``PRELUDE`` decision, the deadline does not pass while it waits and
        moves by the time paused (:meth:`_extend`). For the mapping ``branch`` in EXPLORE, only that branch waits, each
        step a ``SHARE`` decision of it, and the deadline stays: its page shows the clock (:meth:`_share_time`). Raises
        :class:`~hyperloom_multibranch.scheduling.review.ReviewRejected` on a rejection."""

        directory, began = self._layout.review, self._clock.now()
        kind, holder = (DecisionKind.SHARE, branch) if branch else (DecisionKind.PRELUDE, INTEGRATION_BRANCH)
        shown = Reviewed(map_file, dossier, made_by)
        if branch and math.isfinite(self._explore_end):
            _runway, open_by = await self._share_time()
            timed = open_by if self._settings.agent_share_minutes > 0 else None
            shown = replace(shown, explore_end=self._explore_end, open_by=timed)
        state = request(directory, self._layout.root, shown, began)
        asked = {"page": str(directory / REVIEW_PAGE), "map": state["map"], "dossier": state["dossier"]}
        self._decisions.record(Decision(kind, REVIEW_REQUESTED, holder, asked))
        logger.warning("waiting for a person's review of the agent map: read %s", directory / REVIEW_PAGE)
        self._reviewing = not branch
        try:
            decided = await wait(directory, self._clock.sleep)
        finally:
            self._reviewing = False
            if not branch:
                self._extend(self._clock.now() - began)
        detail = {key: decided.get(key) for key in ("by", "note", "changed", "decided_at_iso")}
        detail["paused_seconds"] = round(self._clock.now() - began, 1)
        rejected = decided.get("status") == REJECTED
        reason = REVIEW_REJECTED if rejected else REVIEW_APPROVED
        self._decisions.record(Decision(kind, reason, holder, detail))
        if rejected:
            raise ReviewRejected(f"rejected by {decided.get('by') or 'a person'}: {decided.get('note') or ''}")

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

        sources = ShareSources(self._blackboard, self._toolbox, self._clock, self._layout.toolbox)
        read = await root_shares(sources, prelude.root_commit, prelude.root_profile_job)
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
        ``None`` when none ran (no runner, or no minutes for it). Each step is a ``PRELUDE`` decision."""

        from hyperloom_multibranch.specialist.dossier_pass import DossierStart

        minutes = self._settings.agent_dossier_minutes
        if self._dossier_pass is None or minutes <= 0:
            detail = {"minutes": minutes, "runner": self._dossier_pass is not None}
            self._decisions.record(Decision(DecisionKind.PRELUDE, DOSSIER_SKIPPED, INTEGRATION_BRANCH, detail))
            return None
        given = (None, ROOT_PROFILE_SOURCE) if shares is None else (shares.shares, shares.source)
        start = DossierStart(source, start_commit, self._clock.now() + minutes * MINUTE, *given)
        detail = {"source": source, "start_commit": start_commit, "minutes": minutes, "profile": given[1]}
        self._decisions.record(Decision(DecisionKind.PRELUDE, DOSSIER_STARTED, INTEGRATION_BRANCH, detail))
        outcome = await self._dossier_pass(start)
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

    async def _no_final(self, ended: SessionStopReason) -> SessionOutcome:
        """No FINAL over a session that recorded no baseline: the toolbox takes the dead session's unfinished tests up
        again from its submit log, which re-attaches a measurement still running, and withdraws them, so CLOSE ends
        their sandboxes instead of leaving them to hold the grant's slots until they time out."""

        await self._toolbox.start()
        await self._toolbox.withdraw_all(ended.value)
        self._decisions.record(Decision(DecisionKind.FINAL, NO_BASELINE, detail={"chosen": None}))
        return SessionOutcome(ended, None, None, NO_BASELINE)

    async def _final(self, candidates: FinalCandidates, reason: SessionStopReason) -> SessionOutcome:
        self._in_final = True
        self._phase("final", candidates=encode(candidates.each))
        final = await self._selector.select(candidates)
        self._decisions.record(Decision(DecisionKind.FINAL, final.reason, detail={"chosen": encode(final.chosen)}))
        return SessionOutcome(reason, final, None, "")

    async def _adopt_playbook(self) -> None:
        """Play the scheduling playbook ``--playbook`` asks for from the pool beside the RSI archive, read off the event
        loop, and record it as a scheduler decision: the champion, or in a challenger session the oldest playbook on
        trial; under ``auto`` a session is a challenger session when its id draws it. Without an archive, or under
        ``HYPERLOOM_RSI_WRITE_ONLY=1``, the session plays none, and runs and is judged as one without the RSI."""

        mode = self._settings.playbook
        challenger = mode is PlaybookMode.CHALLENGER or (
            mode is PlaybookMode.AUTO and drawn_challenger(self._layout.session_id)
        )
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
            self._blackboard,
            self._repo,
            self._blobs,
            dashboard,
            self._image_check,
            self._briefs.warm,
            _checkpoint_scope(self._settings),
        )
        services = ExportServices(self._recipe_kb, self._author, self._curator)
        closing = SessionExports(self._layout, sources, self._metric, self._budget_seconds, services)
        return Closer(self._layout, self._stopper, closing, self._clock, self._agent_isolation)

    async def _root_session(self) -> PreludeResult:
        """PRELUDE: the root commit with its baseline, the commit EXPLORE opens on and the warm-start leads every
        specialist is briefed on."""

        prelude = await self._prelude.root(self._settings.workload, self._base_image_digest)
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
        root = (await self._blackboard.baseline()).root_commit
        return await choose_candidates(progress, integration, self._settings.workload.metric, self._repo, root)

    async def _close(self, ending: SessionOutcome, hard_stop: HardStop) -> SessionOutcome:
        if ending.stop_reason is SessionStopReason.ABORTED:
            hard_stop.bring_forward()
        close_by = hard_stop.at
        try:
            self._phase("close", stop_reason=ending.stop_reason.value, error=ending.error)
            closer = self._closer()
        except BaseException:  # re-raised: nothing is stopped yet, so the sandboxes and processes go first
            await self._stopper.release(TOOLBOX_CLOSE_SECONDS)
            raise
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
            agent_isolation=self._agent_isolation,
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


async def recorded_baseline(blackboard: LocalBlackboard) -> BaselineRecord | None:
    """The baseline the rule gate wrote on ``blackboard``, ``None`` while it has written none."""

    try:
        return await blackboard.baseline()
    except NotRecorded:
        return None


def prelude_landed(root: Path) -> bool:
    """Whether the session in ``root`` got through PRELUDE far enough for FINAL to validate it: it committed its root
    and the rule gate wrote its baseline. A session that did not measured nothing a later start could carry on from.
    Creates nothing. Blocks, and runs an event loop of its own."""

    layout = SessionLayout(root)
    if not layout.root_recipe.is_file():
        return False
    reader = LocalBlackboard(layout.blackboard, Writer(Role.OBSERVER, RESTART_READER))
    return asyncio.run(recorded_baseline(reader)) is not None


def _checkpoint_scope(settings: RunSettings) -> CheckpointScope | None:
    """The AgentX checkpoint the session ran under; an InferenceX session has none."""
    workload = settings.workload
    if workload.kind is not WorkloadKind.AGENTX:
        return None
    return CheckpointScope(settings.agentx_early_stop, workload.agentx_objective)


def persisted_run(root: Path) -> PersistedRun:
    """The session in ``root``, read back from what :meth:`Session._write_manifest` and CLOSE left there, or from what
    a session that was killed left without a CLOSE; raises ValueError naming what keeps FINAL from running over it.
    Only the build it ran on may read back what it measured. Creates nothing. Blocks."""

    persisted = persisted_session(root)
    ran_on, here = read_manifest(persisted.layout.manifest).code_version, code_version()
    if ran_on != here:
        raise ValueError(
            f"the session in {root} ran on {ran_on} and this build is {here}; FINAL and CLOSE read back what that "
            "build persisted, so run them from it"
        )
    return persisted


def persisted_session(root: Path) -> PersistedRun:
    """:func:`persisted_run` on any build: for a session that recorded no baseline, which a restart only closes, so
    nothing it measured is read back."""

    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")
    layout = SessionLayout(root)
    try:
        manifest = read_manifest(layout.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{root} is no multi-branch session that ran: {exc}") from exc
    if not layout.root_recipe.is_file():
        raise ValueError(f"the session in {root} committed no root, so FINAL has nothing to validate")
    return PersistedRun(layout, manifest.settings, ended_before_final(layout))


__all__ = [
    "COMPLETED_REASONS",
    "JUDGE_PROMPT_KEY",
    "LESSONS_KEY",
    "NO_BASELINE",
    "NO_CARDS_LEFT",
    "PersistedRun",
    "Session",
    "SessionHost",
    "SessionOutcome",
    "SessionServices",
    "persisted_run",
    "persisted_session",
    "prelude_landed",
    "recorded_baseline",
]
