# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE: run the main flow once for the baseline and roofline, then root the session on its result.

The child is ``hyperloom optimize`` with every optimization lever and enablement off and its roofline trace analysed
without a model session, given the whole remaining session budget. It changes nothing of the framework checkout it is
launched on, so the root recipe -- that checkout at its ``HEAD`` on the toolbox's image -- is known before it starts:
the experiment repository stores it as the root commit and the integration branch is opened on it with its K at once.
The baseline every recorded value is anchored to is measured through the toolbox while the child runs: as many tests of
the root as the rule gate asks for, from which the gate writes the baseline and its bar; on a toolbox that measures on
this machine's cards, the child runs on cards leased for it. Either failing ends the other and PRELUDE. Where that lease
would leave fewer cards than one test needs, the child runs first and the root tests after it. The longest of those
tests, start to finish, is how long one test of this workload takes; it is FINAL's reserve. A gate that asks for none
(AgentX, whose test takes over an hour) is given the child's own run of the root instead, once the child has ended: its
report and per-request export are copied to ``MB-prelude/baseline/``, the toolbox reads the copy as it reads a test's
round, and that run's wall time is the reserve.

A baseline the child cannot establish ends it with the main flow's own baseline stop, and PRELUDE fails naming that
stop; it never spends the session repairing the combo. Its session is read by
:mod:`~hyperloom_multibranch.scheduling.prelude_reader` alone; a child that rooted on another recipe than the one
committed fails PRELUDE, and the session profile is written with the roofline ceiling it measured. The rest of the
profile is known before the child starts -- the framework version is the one the child is launched to record -- so the
Recipe KB's champions for it are read as warm-start leads for the specialists at once, the best of them prepared for
replay on the root's image and offered to the integrator as the session's champion, which replays it beside the child
and the root tests, judges it once the baseline is anchored, and EXPLORE opens on it when it beats the baseline; where
the child runs first, the replay waits for the baseline as well. Their files are in the session's blob store; no
Recipe KB, or none of its champions in this session's scope, gives none, and a KB that cannot be read is recorded and
gives none either. Each replay is checked against the toolbox's acceptance on the root, queueing nothing: one it
refuses stays a lead with the refusal named in place of the replay, unless the champion was built on another image,
whose launch config alone is checked in its place, and one acceptance has not answered for when
:data:`PREFLIGHT_SECONDS` run out stays offered, its lead saying it was not checked.

Once the child has ended, the toolbox is asked for the TraceLens report of the child's latest roofline
profile, so the root has operator shares in the blackboard; PRELUDE does not wait for it, and a child without a profile
or a refusal only leaves a PRELUDE decision naming why. The toolbox reads a trace only inside its own directory, and the
child's session lies outside the session, so the trace and its point config are copied there first.

A session given the PRELUDE of an earlier one (:func:`reused_prelude`) starts no child: it roots on the child session
that PRELUDE rooted on, which it only reads, the same way. The warm-start leads are read again, as their replays' files
go to this session's blob store, and the baseline is measured as ever.

Apart from that AgentX baseline the child's measurements never reach the blackboard, and nothing records the card time
they took, so the session keeps an upper bound of it under its own name: the child's wall time from spawn to exit
times the workload's TP, which counts the child's time off the cards as well.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import read_json_strict
from hyperloom.common.perf_metric import GRADED_INTVTY_P50
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom.contracts.recipe import Recipe
from hyperloom.inference_optimizer.model_config_utils import summarize_model_config
from hyperloom.measure.compile import BlobStore
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError
from hyperloom_multibranch.contracts.gate import NotMeasured, PreludeMeasurement, SessionGate
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent, SessionProfile
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind, LineageKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.blackboard.knowledge.warm_start import (
    LAUNCH_ONLY_REASON,
    PreparedReplay,
    RecipeDownloader,
    ReplayRefusal,
    ReplayTarget,
    WarmLead,
    replay_leads,
    warm_leads,
)
from hyperloom_multibranch.scheduling.budget import BudgetExhausted, HardStop, SessionBudget
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.manifest import read_workload
from hyperloom_multibranch.scheduling.prelude_reader import (
    ChildBaselineRun,
    ChildProfile,
    PreludeArtifacts,
    PreludeReadError,
    framework_pin,
    launched_framework_version,
    launched_root,
    read_child_session,
    read_prelude,
    root_recipe,
)
from hyperloom_multibranch.scheduling.processes import RegisteredProcess
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request, together
from hyperloom_multibranch.scheduling.settings import ReusedPrelude, Workload

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: The ``hyperloom`` console script, run as a module so the child runs on the same install as the scheduler.
HYPERLOOM_COMMAND = ("-m", "hyperloom", "optimize")
#: Only the baseline and roofline: no optimization lever, no Recipe KB read or write-back, and no enablement, whose
#: default in the main flow admits it, so a baseline that cannot be established ends the child.
PRELUDE_FLAGS = (
    "--no-framework-agent",
    "--no-kernel",
    "--no-enable-conc-sweep",
    "--enablement",
    "off",
    "--no-warm-replay",
    "--no-research-scout",
    "--no-static-recon",
    "--degraded-kb",
)
PRELUDE_BRANCH = "prelude"
#: The cards the child runs on: the main flow keeps the ones it is given, and otherwise takes the machine's first TP
#: (src/hyperloom/orchestrator/actions/executors/_workload_envs.py).
CHILD_CARDS_ENV = "ROCR_VISIBLE_DEVICES"
#: The child analyses its roofline trace on the main flow's route that starts no model session
#: (src/hyperloom/orchestrator/actions/executors/trace_analyze.py): the session analyses the trace through the toolbox
#: and never reads the child's own analysis. Set in the child's environment, it overrides a toolbox route of this
#: process's environment that the main flow does not accept.
TRACE_ANALYSIS_ROUTE_ENV = "HYPERLOOM_TRACE_ANALYSIS_ROUTE"
CHILD_TRACE_ANALYSIS_ROUTE = "bypass"
#: The child's lease is the lineage root's: its card time counts on the integration branch, not as a branch of its own.
CHILD_LEASE_CHARGE = LeaseCharge(LeaseKind.PRELUDE_CHILD, INTEGRATION_BRANCH)
#: The lease refusal that leaves the child to run before the root tests: its cards would leave fewer than one test's.
LEAVES_NO_MEASUREMENT = "lease_leaves_no_measurement"
CHILD_FIRST = "child runs before the root tests"
#: The child reaps the inference servers it started on its own way out, which takes far longer than a kill's grace.
#: A cancelled PRELUDE gives it at most this, and never more than half the time left before the hard stop, so CLOSE
#: keeps the other half.
PRELUDE_TERMINATE_GRACE_SECONDS = 120.0
#: How long PRELUDE spends checking the warm-start replays against acceptance, all of them together.
PREFLIGHT_SECONDS = 120.0
#: Why a lead was not checked: acceptance had not answered for its replay when :data:`PREFLIGHT_SECONDS` ran out.
PREFLIGHT_OUT_OF_TIME = "PRELUDE's time for the check ran out"
BASELINE_REFUSED = "another baseline test was refused"
#: ``--reuse-prelude`` finds an earlier session's root by this reason and reads its ``child_session`` back.
ROOT_COMMITTED = "root committed"
CHILD_REUSED = "child reused"
ROOT_PROFILE_STARTED = "root profile started"
ROOT_PROFILE_NOT_STARTED = "root profile not started"
BASELINE_ESTABLISHED = "baseline established"
#: Where the child's baseline run is copied, below ``MB-prelude/``, and the files of it the toolbox reads.
PRELUDE_BASELINE_DIR = "baseline"
CHILD_REPORT = "benchmark_report.json"
CHILD_EXPORT = Path("aiperf_artifacts") / "profile_export.jsonl"
#: Read by the prefix-cache check; a run without one is left to the check, which then reads the point config alone.
CHILD_SERVER_LOG = "server.log"
#: How far the P50 read from the copy may sit from the child's own record of it: the same report, so rounding only.
CHILD_P50_REL_TOLERANCE = 1e-6
_HASH_CHUNK = 1 << 20
#: The workload fields a reused PRELUDE must match: all of them but the kernel optimizer, which only scopes the
#: session's recipes, so a field added to :class:`Workload` has to match as well.
CHILD_WORKLOAD_FIELDS = tuple(field.name for field in fields(Workload) if field.name != "kernel_optimizer")


class PreludeFailed(RuntimeError):
    """The baseline prelude established no baseline the session can start from."""


@dataclass(frozen=True)
class WarmStartSource:
    """The Recipe KB the warm-start leads are read from and the session's blob store their replays' files go into."""

    recipe_kb: RecipeDownloader
    blobs: BlobStore


@dataclass(frozen=True)
class _DerivedRoot:
    """The root recipe a finished child session gives."""

    child: Path
    artifacts: PreludeArtifacts
    recipe: Recipe


@dataclass(frozen=True)
class _OpenedRoot:
    """The root recipe as committed, the integration branch opened on its commit, and the session profile as it is
    known before the child has run: without the roofline ceiling only the child measures."""

    recipe: Recipe
    commit: str
    profile: SessionProfile


@dataclass(frozen=True)
class _ChildCards:
    """Where the child runs: on the cards of ``lease`` (``None`` where it takes cards itself), and beside the root
    tests or, when ``alongside`` is false, before them."""

    lease: BranchLease | None
    alongside: bool

    def env(self) -> Mapping[str, str]:
        """The child's environment: this process's, with :data:`CHILD_TRACE_ANALYSIS_ROUTE` and the leased cards
        named in :data:`CHILD_CARDS_ENV`."""

        cards = {} if self.lease is None else {CHILD_CARDS_ENV: ",".join(str(card) for card in self.lease.gpu_ids)}
        return {**os.environ, TRACE_ANALYSIS_ROUTE_ENV: CHILD_TRACE_ANALYSIS_ROUTE, **cards}


@dataclass(frozen=True)
class _ChildRun:
    hours: float
    hard_stop: HardStop
    cards: _ChildCards


@dataclass(frozen=True)
class _SettledRoot:
    """What the finished child gives the root: the session profile and the GPU-time bound."""

    profile: SessionProfile
    child_gpu_seconds: float


@dataclass(frozen=True)
class _Replayed:
    """The Recipe KB's leads and the commit EXPLORE opens on: the root, or its champion replayed onto it."""

    warm_leads: tuple[WarmLead, ...]
    start_commit: str


@dataclass(frozen=True)
class PreludeResult:
    """The root commit the session is anchored on and the commit EXPLORE opens on, the profile it runs on, the Recipe
    KB's leads for it, the upper bound of the GPU seconds the main-flow child used outside a lease (0 where it ran on
    leased cards, whose card time the blackboard charges the integration branch) and the longest wall time, in
    seconds, of the root tests the baseline anchored on; the rest is in the blackboard and ``MB-prelude/``."""

    root_commit: str
    start_commit: str
    profile: SessionProfile
    warm_leads: tuple[WarmLead, ...]
    child_gpu_seconds_upper_bound: float
    longest_root_test: float


def prelude_argv(workload: Workload, max_hours: float, launch_info: Path) -> tuple[str, ...]:
    """The child's command line: ``hyperloom optimize`` on the interpreter this package runs on. An AgentX child is
    given no sequence lengths; the environment the peers export tells it the workload is AgentX."""

    shape = workload.inferencex
    lengths = () if shape is None else ("--isl", str(shape.isl), "--osl", str(shape.osl))
    return (
        sys.executable,
        *HYPERLOOM_COMMAND,
        *("--model", workload.model, "--framework", workload.framework, "--gpu-type", workload.gpu_type),
        *("--tp", str(workload.tp), *lengths),
        *("--conc", str(workload.conc), "--precision", workload.precision),
        *PRELUDE_FLAGS,
        *("--max-hours", f"{max_hours:.6f}", "--launch-info-file", str(launch_info)),
    )


def reused_prelude(session: Path, workload: Workload, image_digest: str) -> ReusedPrelude:
    """The PRELUDE of the earlier session in ``session``, for a session of ``workload`` on the image
    ``image_digest``; raises ValueError naming what keeps it from being reused: no such directory, no root committed,
    another workload, another image or a child session that is gone. Blocks."""

    if not session.is_dir():
        raise ValueError(f"{session} is not a directory")
    earlier = SessionLayout(session)
    try:
        child = _rooted_child(earlier.decisions)
        ran = read_workload(earlier.manifest)
        root = Recipe.from_document(read_json_strict(earlier.root_recipe))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{session} holds no completed PRELUDE: {exc}") from exc
    differing = [
        f"{name} {getattr(ran, name)!r}, not {getattr(workload, name)!r}"
        for name in CHILD_WORKLOAD_FIELDS
        if getattr(ran, name) != getattr(workload, name)
    ]
    if differing:
        raise ValueError(f"the PRELUDE of {session} ran another workload: {'; '.join(differing)}")
    if root.base_image_digest != image_digest:
        raise ValueError(
            f"the PRELUDE of {session} ran on image {root.base_image_digest}, this session on {image_digest!r}"
        )
    if not child.is_dir():
        raise ValueError(f"the child session the PRELUDE of {session} rooted on, {child}, is gone")
    return ReusedPrelude(str(session), str(child), root.digest())


def _rooted_child(decisions: Path) -> Path:
    for line in decisions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") == DecisionKind.PRELUDE.value and row["reason"] == ROOT_COMMITTED:
            return Path(row["detail"]["child_session"])
    raise ValueError(f"{decisions} records no {ROOT_COMMITTED!r}")


#: Puts the integration branch at the root commit on the roster with its K and gives the toolbox the quota.
OpenIntegration = Callable[[str], Awaitable[None]]
#: Replays the champion on the root commit and returns the commit EXPLORE opens on; the replay is judged once the
#: awaitable, the baseline's anchoring, has ended.
ReplayChampion = Callable[[PreparedReplay, str, Awaitable[object]], Awaitable[str]]


@dataclass(frozen=True)
class PreludeServices:
    """The experiment repository the root is committed to, the rule gate and the session's tests the baseline is
    measured with, the toolbox that analyses the child's profile and leases its cards, which it does only where it
    measures on this machine's cards (``on_this_machine``), what opens the integration branch on the root and what
    replays the Recipe KB's champion on it (the integrator)."""

    repo: ExperimentRepo
    gate: SessionGate
    tests: SessionTests
    toolbox: RealToolbox
    on_this_machine: bool
    open_integration: OpenIntegration
    replay_champion: ReplayChampion


class Prelude:
    """Runs PRELUDE for one session; ``warm_start`` is ``None`` when no Recipe KB is configured."""

    def __init__(
        self, context: SchedulerContext, services: PreludeServices, *, warm_start: WarmStartSource | None
    ) -> None:
        self._layout = context.layout
        self._warm_start_source = warm_start
        self._processes = context.processes
        self._blackboard = context.blackboard
        self._repo = services.repo
        self._gate = services.gate
        self._tests = services.tests
        self._toolbox = services.toolbox
        self._on_this_machine = services.on_this_machine
        self._open_integration = services.open_integration
        self._replay_champion = services.replay_champion
        self._decisions = context.decisions
        self._clock = context.clock
        self._child_baseline: tuple[ChildBaselineRun | None, str] = (None, "PRELUDE committed no root")

    async def root(
        self, workload: Workload, budget: SessionBudget, hard_stop: HardStop, base_image_digest: str
    ) -> PreludeResult:
        """Root the session on the main flow's result, on ``base_image_digest``, and measure its baseline; raises
        :class:`PreludeFailed` when there is none.

        The root is the framework checkout the child is launched on, which it does not change, and the session
        profile's identity is known before it too, so the root is committed and the integration branch opened on it
        before the child starts; the root tests, and the Recipe KB's champion replayed on the root, run beside the
        child, on the cards a lease holds for it, and the replay is judged once the baseline is anchored. Where that
        lease would leave fewer cards than one test needs, and for a gate that anchors on the child's own run, the
        child runs first, then the root tests, then the replay. Any of them failing ends the others; a child that
        rooted on another recipe fails PRELUDE.
        """

        hours = self._child_hours(budget)
        recipe = await self._launched_root(workload.framework, base_image_digest)
        opened = await self._open_root(recipe, workload)
        root_tests = await self._gate.baseline_test_count()
        cards = await self._child_cards(workload.tp, root_tests)
        child = asyncio.ensure_future(self._root_on_child(opened, workload, _ChildRun(hours, hard_stop, cards)))
        if not cards.alongside:
            await child
        baseline = asyncio.ensure_future(self._measure_baseline(opened.commit, root_tests))
        if not cards.alongside:
            await baseline
        replayed = asyncio.ensure_future(self._replay(opened, baseline))
        await together(child, baseline, replayed)
        settled = child.result()
        return _result(opened, settled, replayed.result(), baseline.result())

    async def reuse(self, earlier: ReusedPrelude, workload: Workload, base_image_digest: str) -> PreludeResult:
        """Root the session as :meth:`root` does, on the child session of ``earlier``, without starting a child: the
        GPU time that child used is the earlier session's. The child session is only read; raises
        :class:`PreludeFailed`, committing nothing, when the root it gives now is not the one ``earlier`` committed."""

        self._decisions.record(Decision(DecisionKind.PRELUDE, CHILD_REUSED, PRELUDE_BRANCH, asdict(earlier)))
        derived = await self._derive_root(Path(earlier.child_session), workload.framework, base_image_digest)
        if derived.recipe.digest() != earlier.root_recipe_digest:
            raise PreludeFailed(
                f"the child session {derived.child} now gives the root recipe {derived.recipe.digest()}, not "
                f"{earlier.root_recipe_digest} that {earlier.session} committed"
            )
        opened = await self._open_root(derived.recipe, workload)
        settled = await self._settle(opened, derived, workload, 0.0)
        baseline = asyncio.ensure_future(self._measure_baseline(opened.commit, await self._gate.baseline_test_count()))
        longest = await baseline
        return _result(opened, settled, await self._replay(opened, baseline), longest)

    def _child_hours(self, budget: SessionBudget) -> float:
        try:
            return budget.prelude_max_hours(self._clock.now())
        except BudgetExhausted as exc:
            raise PreludeFailed(str(exc)) from exc

    async def _launched_root(self, framework: str, base_image_digest: str) -> Recipe:
        try:
            return await asyncio.to_thread(launched_root, framework, os.environ, base_image_digest)
        except PreludeReadError as exc:
            raise PreludeFailed(f"cannot root the session on the framework checkout the child runs on: {exc}") from exc

    async def _open_root(self, recipe: Recipe, workload: Workload) -> _OpenedRoot:
        atomic_write_json(self._layout.root_recipe, recipe.to_document())
        root = await self._repo.init_root(recipe)
        await self._blackboard.record_lineage(
            LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root, recipe_digest=recipe.digest())
        )
        await self._open_integration(root)
        version = await asyncio.to_thread(launched_framework_version, workload.framework, os.environ)
        model_config = await asyncio.to_thread(summarize_model_config, workload.model)
        return _OpenedRoot(recipe, root, _session_profile(self._layout.session_id, workload, version, model_config))

    async def _child_cards(self, tp: int, root_tests: int) -> _ChildCards:
        """Lease the child's ``tp`` cards where it runs beside the root tests on cards they measure on."""

        if root_tests == 0:
            return _ChildCards(None, alongside=False)
        if not self._on_this_machine:
            return _ChildCards(None, alongside=True)
        try:
            lease = await self._toolbox.lease_branch_sandbox(PRELUDE_BRANCH, tp, charge=CHILD_LEASE_CHARGE)
        except RequestRejected as refused:
            if refused.code != LEAVES_NO_MEASUREMENT:
                raise
            self._decisions.record(
                Decision(DecisionKind.PRELUDE, CHILD_FIRST, PRELUDE_BRANCH, {"refusal": refused.reason})
            )
            return _ChildCards(None, alongside=False)
        return _ChildCards(lease, alongside=True)

    async def _root_on_child(self, opened: _OpenedRoot, workload: Workload, run: _ChildRun) -> _SettledRoot:
        """Run the child and settle the root on what it measured; the child's lease ends with it."""

        spawned = self._clock.now()
        try:
            child = await self._run_child(workload, run)
        finally:
            if run.cards.lease is not None:
                await asyncio.shield(self._toolbox.release_branch_sandbox(run.cards.lease.lease_id))
        # A leased child's card time is already the integration branch's, from the lease the blackboard records.
        gpu_seconds = 0.0 if run.cards.lease is not None else (self._clock.now() - spawned) * workload.tp
        derived = await self._derive_root(child, workload.framework, opened.recipe.base_image_digest)
        if derived.recipe.digest() != opened.recipe.digest():
            raise PreludeFailed(
                f"the child session {child} gives the root recipe {derived.recipe.digest()}, not "
                f"{opened.recipe.digest()} that the session committed before it started"
            )
        return await self._settle(opened, derived, workload, gpu_seconds)

    async def _derive_root(self, child: Path, framework: str, base_image_digest: str) -> _DerivedRoot:
        try:
            artifacts = await asyncio.to_thread(read_prelude, child)
            pin = await asyncio.to_thread(framework_pin, artifacts.framework_root)
            recipe = root_recipe(framework, pin, base_image_digest)
        except PreludeReadError as exc:
            raise PreludeFailed(f"cannot root the session on the baseline prelude in {child}: {exc}") from exc
        return _DerivedRoot(child, artifacts, recipe)

    async def _settle(
        self, opened: _OpenedRoot, derived: _DerivedRoot, workload: Workload, child_gpu_seconds: float
    ) -> _SettledRoot:
        """Write the session profile with the roofline ceiling the child measured, kept only on the metric the session
        is graded on, record the root as committed on the child's session and start the root profile."""

        artifacts, root = derived.artifacts, opened.commit
        ceiling = {metric: value for metric, value in artifacts.roofline_ceiling.items() if metric == workload.metric}
        profile = replace(opened.profile, roofline_ceiling=ceiling)
        await self._blackboard.write_session_profile(profile)
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE,
                ROOT_COMMITTED,
                INTEGRATION_BRANCH,
                {
                    "root_commit": root,
                    "child_session": str(derived.child),
                    "child_gpu_seconds_upper_bound": child_gpu_seconds,
                },
            )
        )
        self._child_baseline = (artifacts.baseline_run, artifacts.baseline_run_refusal)
        await self._start_root_profile(root, artifacts.profile)
        return _SettledRoot(profile, child_gpu_seconds)

    async def _replay(self, opened: _OpenedRoot, anchored: Awaitable[object]) -> _Replayed:
        """Read the warm-start leads and replay the best champion among them on the root, judged once ``anchored``
        has ended; with none, EXPLORE opens on the root."""

        leads = await self._warm_start(opened.profile, opened.recipe, opened.commit)
        champion = next((lead.replay for lead in leads if lead.replay is not None), None)
        if champion is None:
            return _Replayed(leads, opened.commit)
        return _Replayed(leads, await self._replay_champion(champion, opened.commit, anchored))

    async def _start_root_profile(self, root: str, profile: ChildProfile | None) -> None:
        """Start the toolbox's TraceLens report of the child's ``profile`` for the ``root`` commit, without waiting for
        it; a ``PRELUDE`` decision names the job, or why none was started."""

        if profile is None:
            self._root_profile_not_started({"reason": "the child left no roofline profile"})
            return
        detail = {"source_id": profile.run_id, "trace_dir": str(profile.trace_dir)}
        try:
            copy = await asyncio.to_thread(self._copy_profile, profile)
            job = await self._toolbox.analyze_root_profile(
                root, copy.trace_dir, copy.point_config, source_id=copy.run_id
            )
        except OSError as exc:
            self._root_profile_not_started({**detail, "reason": f"cannot copy the profile into the session: {exc}"})
            return
        except RequestRejected as exc:
            self._root_profile_not_started({**detail, "code": exc.code, "reason": exc.reason})
            return
        self._decisions.record(
            Decision(DecisionKind.PRELUDE, ROOT_PROFILE_STARTED, INTEGRATION_BRANCH, {**detail, "job_id": job})
        )

    def _root_profile_not_started(self, detail: dict[str, str]) -> None:
        self._decisions.record(Decision(DecisionKind.PRELUDE, ROOT_PROFILE_NOT_STARTED, INTEGRATION_BRANCH, detail))

    def _copy_profile(self, profile: ChildProfile) -> ChildProfile:
        """``profile`` copied under the toolbox's directory; symlinks are followed, as the toolbox resolves them. Blocks."""

        target = self._layout.root_profile / profile.run_id
        trace_dir = shutil.copytree(profile.trace_dir, target / profile.trace_dir.name, dirs_exist_ok=True)
        point_config = shutil.copy2(profile.point_config, target / profile.point_config.name)
        return replace(profile, trace_dir=Path(trace_dir), point_config=Path(point_config))

    async def _warm_start(self, profile: SessionProfile, root: Recipe, root_commit: str) -> tuple[WarmLead, ...]:
        """The Recipe KB's leads for ``profile``, the best prepared for replay on ``root`` and checked against the
        toolbox's acceptance on ``root_commit``; the ``PRELUDE`` decision names them, or why the KB was not read."""

        source = self._warm_start_source
        leads: tuple[WarmLead, ...] = ()
        unread = "no Recipe KB is configured" if source is None else ""
        if source is not None:
            target = ReplayTarget(root, source.blobs, self._layout.warm_start)
            try:
                found = await asyncio.to_thread(warm_leads, source.recipe_kb, profile)
                leads = await asyncio.to_thread(replay_leads, source.recipe_kb, profile, found, target)
            except KBStoreError as exc:
                unread = f"{type(exc).__name__}: {exc}"
        leads = await self._preflight(leads, root, root_commit)
        detail = {"leads": [lead.describe() for lead in leads], "unread": unread}
        self._decisions.record(Decision(DecisionKind.PRELUDE, "warm start read", INTEGRATION_BRANCH, detail))
        return leads

    async def _preflight(self, leads: tuple[WarmLead, ...], root: Recipe, root_commit: str) -> tuple[WarmLead, ...]:
        """``leads``, each replay checked against the toolbox's acceptance on ``root_commit`` within
        :data:`PREFLIGHT_SECONDS` in all (:meth:`_checked`). The check measures nothing, so it spends no GPU."""

        deadline = self._clock.now() + PREFLIGHT_SECONDS
        return tuple([await self._checked(lead, root, root_commit, deadline) for lead in leads])

    async def _checked(self, lead: WarmLead, root: Recipe, root_commit: str, deadline: float) -> WarmLead:
        """``lead`` as acceptance takes its replay on ``root_commit`` by ``deadline``. A refusal goes on the lead in
        place of the replay, except that a champion built on another image falls back to its launch config
        (:meth:`PreparedReplay.launch_only`), checked in turn. A replay acceptance has not answered for by then stays
        offered, the lead saying it was not checked."""

        replay = lead.replay.on(root, root_commit) if lead.replay is not None else None
        if lead.replay is None or replay is None:
            return lead
        request = integration_test_request(TestKind.TEST, Purpose.INTEGRATION, root_commit, replay.delta)
        try:
            await asyncio.wait_for(self._toolbox.check_test(request), max(0.0, deadline - self._clock.now()))
        except RequestRejected as refused:
            if lead.replay.source_image and lead.replay.recipe is not None:
                launch_only = lead.replay.launch_only(f"{LAUNCH_ONLY_REASON}: {refused.code}")
                return await self._checked(replace(lead, replay=launch_only), root, root_commit, deadline)
            refusal = ReplayRefusal(refused.code, refused.reason, rule_pack_identity()["version"])
            return replace(lead, replay=None, refusal=refusal)
        except asyncio.TimeoutError:
            return replace(lead, unchecked=PREFLIGHT_OUT_OF_TIME)
        return lead

    async def _run_child(self, workload: Workload, run: _ChildRun) -> Path:
        argv = prelude_argv(workload, run.hours, self._layout.prelude_launch)
        self._decisions.record(Decision(DecisionKind.PRELUDE, "child started", PRELUDE_BRANCH, {"argv": list(argv)}))
        spawned = await self._processes.spawn(
            argv, branch=PRELUDE_BRANCH, log=self._layout.prelude_log, env=run.cards.env()
        )
        exit_code = await self._wait(spawned, run.hard_stop)
        try:
            child = read_child_session(self._layout.prelude_launch)
        except PreludeReadError as exc:
            raise PreludeFailed(str(exc)) from exc
        atomic_write_json(
            self._layout.prelude_child,
            {"session_dir": None if child is None else str(child), "exit_code": exit_code, "argv": list(argv)},
        )
        if child is None:
            raise PreludeFailed(f"the baseline prelude exited with {exit_code} before writing its launch info")
        return child

    async def _wait(self, child: RegisteredProcess, hard_stop: HardStop) -> int:
        """The child's exit code; nothing it started is left once this returns or is cancelled."""

        try:
            exit_code = await child.process.wait()
        except asyncio.CancelledError:
            grace = min(PRELUDE_TERMINATE_GRACE_SECONDS, (hard_stop.at - self._clock.now()) / 2)
            await self._processes.terminate(child, grace)
            await self._processes.kill_branch(PRELUDE_BRANCH)
            raise
        await self._processes.kill_branch(PRELUDE_BRANCH)
        return exit_code

    async def _measure_baseline(self, root: str, count: int) -> float:
        """Measure the baseline on ``root`` with ``count`` root tests through the toolbox and have the rule gate anchor
        it; return the longest wall time, in seconds, of the root tests it anchored on. Raises :class:`PreludeFailed`
        when a baseline test is refused, which takes back the ones accepted before it, or the baseline anchors nothing.

        A gate that asks for no root test anchors on the child's own run of the root (:meth:`_anchor_on_child_run`).
        """

        if count == 0:
            return await self._anchor_on_child_run(root)
        request = integration_test_request(TestKind.TEST, Purpose.BASELINE, root)
        try:
            request_ids = [await self._tests.submit(request) for _ in range(count)]
        except RequestRejected as exc:
            await self._tests.withdraw_integration(BASELINE_REFUSED)
            raise PreludeFailed(f"the toolbox refused the baseline test: {exc}") from exc
        try:
            baseline = await self._gate.establish_baseline(root, request_ids)
        except NotMeasured as exc:
            raise PreludeFailed(f"the baseline anchors nothing: {exc}") from exc
        records = [await self._blackboard.record(request_id) for request_id in baseline.request_ids]
        longest = _longest_wall_seconds(record.result for record in records if record.result is not None)
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE,
                BASELINE_ESTABLISHED,
                INTEGRATION_BRANCH,
                {"request_ids": list(baseline.request_ids), "longest_test_seconds": longest},
            )
        )
        return longest

    async def _anchor_on_child_run(self, root: str) -> float:
        """Have the gate anchor the baseline on the child's run of the root; return that run's wall time.

        The run is read from copies in the session, hashed as they are read, so the gate can tell they still hold what
        it was read from. The P50 read must be the one the child recorded for that run, or the copy is not the anchor.
        """

        run, refusal = self._child_baseline
        if run is None:
            raise PreludeFailed(f"the child's baseline run cannot stand for the root: {refusal}")
        try:
            copy = await asyncio.to_thread(self._copy_child_run, run)
        except OSError as exc:
            raise PreludeFailed(f"cannot copy the child's baseline run {run.run_dir}: {exc}") from exc
        request_id = f"prelude-child:{run.run_dir.name}"
        result = await self._toolbox.read_prelude_baseline(copy.run_dir, copy.point_config, request_id=request_id)
        _check_child_p50(result, run)
        measured = PreludeMeasurement(
            result,
            str(run.run_dir),
            str(copy.report),
            copy.report_sha256,
            str(copy.export),
            copy.export_sha256,
            run.wall_seconds,
        )
        try:
            baseline = await self._gate.establish_prelude_baseline(root, measured)
        except NotMeasured as exc:
            raise PreludeFailed(f"the child's baseline run anchors nothing: {exc}") from exc
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE,
                BASELINE_ESTABLISHED,
                INTEGRATION_BRANCH,
                {"request_ids": [], "source": baseline.source, "longest_test_seconds": run.wall_seconds},
            )
        )
        return run.wall_seconds

    def _copy_child_run(self, run: ChildBaselineRun) -> _ChildRunCopy:
        """The report, export, server log and point config of ``run`` copied under ``MB-prelude/baseline/``, in the
        layout the toolbox reads a round from; raises OSError when the report or export is missing or changed while it
        was copied. Blocks."""

        target = self._layout.prelude / PRELUDE_BASELINE_DIR / run.run_dir.name
        benchmark = target / run.benchmark_dir.name
        (benchmark / CHILD_EXPORT).parent.mkdir(parents=True, exist_ok=True)
        point_config = Path(shutil.copy2(run.point_config, target / run.point_config.name))
        if (run.benchmark_dir / CHILD_SERVER_LOG).is_file():
            shutil.copy2(run.benchmark_dir / CHILD_SERVER_LOG, benchmark / CHILD_SERVER_LOG)
        report, report_sha = _copied(run.benchmark_dir / CHILD_REPORT, benchmark / CHILD_REPORT)
        export, export_sha = _copied(run.benchmark_dir / CHILD_EXPORT, benchmark / CHILD_EXPORT)
        return _ChildRunCopy(target, point_config, report, report_sha, export, export_sha)


@dataclass(frozen=True)
class _ChildRunCopy:
    run_dir: Path
    point_config: Path
    report: Path
    report_sha256: str
    export: Path
    export_sha256: str


def _copied(source: Path, target: Path) -> tuple[Path, str]:
    """``source`` copied to ``target``, and the sha256 both hold; OSError when they differ. Blocks."""

    before = _sha256(source)
    shutil.copyfile(source, target)
    after = _sha256(target)
    if after != before:
        raise OSError(f"{source} changed while it was copied (sha256 {before}, the copy {after})")
    return target, after


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _check_child_p50(result: TestResult, run: ChildBaselineRun) -> None:
    """A measured copy must read the P50 the child recorded for the run; one that did not measure is the gate's."""

    if result.validity is not Validity.SUCCEEDED:
        return
    read = [value.metrics.get(GRADED_INTVTY_P50) for value in result.rounds]
    if run.p50 is None or len(read) != 1 or read[0] is None:
        raise PreludeFailed(f"the child's baseline run {run.run_dir} has no P50 to check its copy against: {read}")
    if not math.isclose(read[0], run.p50, rel_tol=CHILD_P50_REL_TOLERANCE):
        raise PreludeFailed(
            f"the copy of the child's baseline run reads P50 {read[0]}, the child recorded {run.p50} for {run.run_dir}"
        )


def _longest_wall_seconds(results: Iterable[TestResult]) -> float:
    """The longest start-to-finish time among ``results``; 0 for none."""

    return max(
        (
            result.finished_at - result.started_at
            for result in results
            if result.started_at is not None and result.finished_at is not None
        ),
        default=0.0,
    )


def _session_profile(
    session_id: str, workload: Workload, framework_version: str, model_config: Mapping[str, Any]
) -> SessionProfile:
    """The profile before the child measured its roofline ceiling. ``model_config`` is :func:`summarize_model_config`
    of the model; its type and architectures are left empty when the model's ``config.json`` does not name them. An
    AgentX profile has no sequence lengths."""

    shape = workload.inferencex
    return SessionProfile(
        session_id=session_id,
        model=workload.model,
        hardware=workload.gpu_type,
        framework=workload.framework,
        framework_version=framework_version,
        precision=workload.precision,
        tp=workload.tp,
        conc=workload.conc,
        model_type=model_config.get("model_type", ""),
        architectures=tuple(model_config.get("architectures", ())),
        isl=None if shape is None else shape.isl,
        osl=None if shape is None else shape.osl,
        kernel_optimizer=workload.kernel_optimizer.value,
    )


def _result(opened: _OpenedRoot, settled: _SettledRoot, replayed: _Replayed, longest_root_test: float) -> PreludeResult:
    return PreludeResult(
        opened.commit,
        replayed.start_commit,
        settled.profile,
        replayed.warm_leads,
        settled.child_gpu_seconds,
        longest_root_test,
    )


__all__ = [
    "PreludeFailed",
    "Prelude",
    "PreludeResult",
    "PreludeServices",
    "WarmStartSource",
    "prelude_argv",
    "reused_prelude",
]
