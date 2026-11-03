# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE: root the session and measure its baseline, every test of it through the toolbox.

The root recipe is the framework checkout the session runs on, at its ``HEAD``, on the toolbox's image
(:func:`~hyperloom_multibranch.scheduling.prelude_reader.launched_root`): the experiment repository stores it as the
root commit, the integration branch is opened on it with its K at once, and the session profile is written. Three
things then run together, any of them failing ending the others: as many tests of the root as the rule gate asks for,
from which the gate writes the baseline and its bar; a PROFILE test of the root and its TraceLens analysis in the
toolbox (:meth:`~hyperloom_multibranch.scheduling.events.IntegrationProfiler.profile_root`), which PRELUDE waits for so
the opening judge call reads the root's op shares, and which a backend that cannot profile or analyse leaves as its
decision only; and the replay of the Recipe KB's champion. The longest of the root tests, start to finish, is how long
one test of this workload takes; it is FINAL's reserve.

The session profile is known once the root is committed, so the Recipe KB's champions for it are read as warm-start
leads for the specialists at once, the best of them prepared for replay on the root's image and offered to the
integrator as the session's champion, which replays it beside the root tests and the profile, judges it once the
baseline is anchored, and EXPLORE opens on it when it beats the baseline. Their files are in the session's blob store;
no Recipe KB, or none of its champions in this session's scope, gives none, and a KB that cannot be read is recorded
and gives none either. Each replay is checked against the toolbox's acceptance on the root, queueing nothing: one it
refuses stays a lead with the refusal named in place of the replay, unless the champion was built on another image,
whose launch config alone is checked in its place, and one acceptance has not answered for when
:data:`PREFLIGHT_SECONDS` run out stays offered, its lead saying it was not checked.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from hyperloom_common.io import atomic_write_json
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.contracts.measurement import Purpose, TestKind, TestResult
from hyperloom.contracts.recipe import Recipe
from hyperloom.inference_optimizer.model_config_utils import summarize_model_config
from hyperloom.measure.compile import BlobStore
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError
from hyperloom_multibranch.contracts.gate import NotMeasured, SessionGate
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent, SessionProfile
from hyperloom_multibranch.contracts.blackboard.vocab import LineageKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected
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
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.events import ProfileOutcome
from hyperloom_multibranch.scheduling.prelude_reader import (
    PreludeReadError,
    launched_framework_version,
    launched_root,
)
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request, together
from hyperloom_multibranch.scheduling.settings import Workload

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: How long PRELUDE spends checking the warm-start replays against acceptance, all of them together.
PREFLIGHT_SECONDS = 120.0
#: Why a lead was not checked: acceptance had not answered for its replay when :data:`PREFLIGHT_SECONDS` ran out.
PREFLIGHT_OUT_OF_TIME = "PRELUDE's time for the check ran out"
ROOT_COMMITTED = "root committed"
BASELINE_ESTABLISHED = "baseline established"


class PreludeFailed(RuntimeError):
    """The baseline prelude established no baseline the session can start from."""


@dataclass(frozen=True)
class WarmStartSource:
    """The Recipe KB the warm-start leads are read from and the session's blob store their replays' files go into."""

    recipe_kb: RecipeDownloader
    blobs: BlobStore


@dataclass(frozen=True)
class _OpenedRoot:
    """The root recipe as committed, the integration branch opened on its commit, and the session profile."""

    recipe: Recipe
    commit: str
    profile: SessionProfile


@dataclass(frozen=True)
class _Replayed:
    """The Recipe KB's leads and the commit EXPLORE opens on: the root, or its champion replayed onto it."""

    warm_leads: tuple[WarmLead, ...]
    start_commit: str


@dataclass(frozen=True)
class PreludeResult:
    """The root commit the session is anchored on and the commit EXPLORE opens on, the profile it runs on, the Recipe
    KB's leads for it, the longest wall time, in seconds, of the root tests the baseline anchored on, and the toolbox's
    job that analysed the root's profile (``""`` when none did); the rest is in the blackboard and ``MB-prelude/``."""

    root_commit: str
    start_commit: str
    profile: SessionProfile
    warm_leads: tuple[WarmLead, ...]
    longest_root_test: float
    root_profile_job: str


#: Puts the integration branch at the root commit on the roster with its K and gives the toolbox the quota.
OpenIntegration = Callable[[str], Awaitable[None]]
#: Replays the champion on the root commit and returns the commit EXPLORE opens on; the replay is judged once the
#: awaitable, the baseline's anchoring, has ended.
ReplayChampion = Callable[[PreparedReplay, str, Awaitable[object]], Awaitable[str]]
#: Profiles the root commit in the toolbox and analyses the profile, returning how that ended.
ProfileRoot = Callable[[str], Awaitable[ProfileOutcome]]


@dataclass(frozen=True)
class PreludeServices:
    """The experiment repository the root is committed to, the rule gate and the session's tests the baseline is
    measured with, the toolbox whose acceptance checks the warm-start replays, what opens the integration branch on
    the root, what replays the Recipe KB's champion on it (the integrator) and what profiles it."""

    repo: ExperimentRepo
    gate: SessionGate
    tests: SessionTests
    toolbox: RealToolbox
    open_integration: OpenIntegration
    replay_champion: ReplayChampion
    profile_root: ProfileRoot


class Prelude:
    """Runs PRELUDE for one session; ``warm_start`` is ``None`` when no Recipe KB is configured."""

    def __init__(
        self, context: SchedulerContext, services: PreludeServices, *, warm_start: WarmStartSource | None
    ) -> None:
        self._layout = context.layout
        self._warm_start_source = warm_start
        self._blackboard = context.blackboard
        self._repo = services.repo
        self._gate = services.gate
        self._tests = services.tests
        self._toolbox = services.toolbox
        self._open_integration = services.open_integration
        self._replay_champion = services.replay_champion
        self._profile_root = services.profile_root
        self._decisions = context.decisions
        self._clock = context.clock

    async def root(self, workload: Workload, base_image_digest: str) -> PreludeResult:
        """Root the session on ``base_image_digest`` and measure its baseline; raises :class:`PreludeFailed` when there
        is none. The root tests, the root's profile and the champion's replay run together once the root is committed,
        any of them failing ending the others."""

        recipe = await self._launched_root(workload.framework, base_image_digest)
        opened = await self._open_root(recipe, workload)
        await self._blackboard.write_session_profile(opened.profile)
        self._decisions.record(
            Decision(DecisionKind.PRELUDE, ROOT_COMMITTED, INTEGRATION_BRANCH, {"root_commit": opened.commit})
        )
        baseline = asyncio.ensure_future(self._measure_baseline(opened.commit))
        profiled = asyncio.ensure_future(self._profile_root_commit(opened.commit))
        replayed = asyncio.ensure_future(self._replay(opened, baseline))
        await together(baseline, profiled, replayed)
        leads, start = replayed.result().warm_leads, replayed.result().start_commit
        return PreludeResult(opened.commit, start, opened.profile, leads, baseline.result(), profiled.result())

    async def _profile_root_commit(self, root: str) -> str:
        """Profile ``root`` and record how; the job that analysed the profile, ``""`` when none did."""
        outcome = await self._profile_root(root)
        self._decisions.record(Decision(DecisionKind.PROFILE, outcome.reason, detail=outcome.detail))
        return str(outcome.detail.get("job_id") or "")

    async def _launched_root(self, framework: str, base_image_digest: str) -> Recipe:
        try:
            return await asyncio.to_thread(launched_root, framework, os.environ, base_image_digest)
        except PreludeReadError as exc:
            raise PreludeFailed(f"cannot root the session on the framework checkout it runs on: {exc}") from exc

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

    async def _replay(self, opened: _OpenedRoot, anchored: Awaitable[object]) -> _Replayed:
        """Read the warm-start leads and replay the best champion among them on the root, judged once ``anchored``
        has ended; with none, EXPLORE opens on the root."""

        leads = await self._warm_start(opened.profile, opened.recipe, opened.commit)
        champion = next((lead.replay for lead in leads if lead.replay is not None), None)
        if champion is None:
            return _Replayed(leads, opened.commit)
        return _Replayed(leads, await self._replay_champion(champion, opened.commit, anchored))

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

    async def _measure_baseline(self, root: str) -> float:
        """Measure the baseline on ``root`` with the root tests the rule gate asks for and have it anchor them; return
        the longest wall time, in seconds, of the root tests it anchored on. Raises :class:`PreludeFailed` when a
        baseline test is refused, which takes back the ones accepted before it, or the baseline anchors nothing; a
        cancelled measurement takes its root tests back. The other tests of the integration branch, the root's profile
        and the champion's replay, are their own callers' to take back."""

        request = integration_test_request(TestKind.TEST, Purpose.BASELINE, root)
        request_ids: list[str] = []
        try:
            for _ in range(await self._gate.baseline_test_count()):
                request_ids.append(await self._tests.submit(request))
            baseline = await self._gate.establish_baseline(root, request_ids)
        except RequestRejected as exc:
            await self._withdraw(request_ids)
            raise PreludeFailed(f"the toolbox refused the baseline test: {exc}") from exc
        except NotMeasured as exc:
            raise PreludeFailed(f"the baseline anchors nothing: {exc}") from exc
        except asyncio.CancelledError:
            await asyncio.shield(self._withdraw(request_ids))
            raise
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

    async def _withdraw(self, request_ids: list[str]) -> None:
        await asyncio.gather(*(self._tests.withdraw(request_id) for request_id in request_ids))


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
    """The session profile. ``model_config`` is :func:`summarize_model_config` of the model; its type and
    architectures are left empty when the model's ``config.json`` does not name them. An AgentX profile has no sequence
    lengths."""

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


__all__ = [
    "PreludeFailed",
    "Prelude",
    "PreludeResult",
    "PreludeServices",
    "WarmStartSource",
]
