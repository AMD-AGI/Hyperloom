# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE: run the main flow once for the baseline, enablement and roofline, then root the session on its result.

The child is ``hyperloom optimize`` with every optimization lever off, given the whole remaining session budget. Its
session is read by :mod:`~hyperloom_multibranch.scheduling.prelude_reader` alone and folded into the root recipe,
whose kept patches go to the session's blob store and which the experiment repository stores as the root commit, and
the session profile is written. What the child kept that a recipe cannot express does not fail PRELUDE: the root
recipe names it as ``unexpressed``, and kept patches outside the framework checkout are copied to ``MB-prelude/``. The
Recipe KB's champions for that profile are read as warm-start leads for the
specialists, the best of them prepared for replay on the root's image, their files in the session's blob store; no
Recipe KB, or none of its champions in this session's scope, gives none, and a KB that cannot be read is recorded and
gives none either. The baseline every recorded value is anchored to is measured through the toolbox in a
second step, once the caller has put the integration branch on the roster and given the toolbox its K: as many tests
of the root as the rule gate asks for, from which the gate writes the baseline and its bar. The longest of those
tests, start to finish, is how long one test of this workload takes; FINAL's reserve is never shorter.

A session given the PRELUDE of an earlier one (:func:`reused_prelude`) starts no child: it roots on the child session
that PRELUDE rooted on, which it only reads, the same way. The warm-start leads are read again, as their replays' files
go to this session's blob store, and the baseline is measured as ever.

The child's own measurements never reach the notebook, and nothing records the card time they took, so the session
keeps an upper bound of it under its own name: the child's wall time from spawn to exit times the workload's TP,
which counts the child's time off the cards as well.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import read_json_strict
from hyperloom.contracts.measurement import Purpose, TestKind, TestResult
from hyperloom.contracts.recipe import PatchRef, Recipe
from hyperloom.inference_optimizer.model_config_utils import summarize_model_config
from hyperloom.measure.compile import BlobError, BlobStore
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError
from hyperloom_multibranch.contracts.gate import NotMeasured, SessionGate
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, SessionProfile
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook.knowledge.warm_start import (
    RecipeDownloader,
    ReplayTarget,
    WarmLead,
    replay_leads,
    warm_leads,
)
from hyperloom_multibranch.notebook.codec import decode
from hyperloom_multibranch.scheduling.budget import BudgetExhausted, HardStop, SessionBudget
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.prelude_reader import (
    PreludeArtifacts,
    PreludeReadError,
    framework_pin,
    read_child_session,
    read_prelude,
    root_recipe,
)
from hyperloom_multibranch.scheduling.processes import RegisteredProcess
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request
from hyperloom_multibranch.scheduling.settings import ReusedPrelude, Workload

#: The ``hyperloom`` console script, run as a module so the child runs on the same install as the scheduler.
HYPERLOOM_COMMAND = ("-m", "hyperloom", "optimize")
#: Only the baseline, enablement and roofline: no optimization lever, no Recipe KB read or write-back.
PRELUDE_FLAGS = (
    "--no-framework-agent",
    "--no-kernel",
    "--no-enable-conc-sweep",
    "--enablement",
    "launch",
    "--no-warm-replay",
    "--no-research-scout",
    "--degraded-kb",
)
PRELUDE_BRANCH = "prelude"
#: The child reaps the inference servers it started on its own way out, which takes far longer than a kill's grace.
#: A cancelled PRELUDE gives it at most this, and never more than half the time left before the hard stop, so CLOSE
#: keeps the other half.
PRELUDE_TERMINATE_GRACE_SECONDS = 120.0
BASELINE_REFUSED = "another baseline test was refused"
#: ``--reuse-prelude`` finds an earlier session's root by this reason and reads its ``child_session`` back.
ROOT_COMMITTED = "root committed"
CHILD_REUSED = "child reused"
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
    """The root recipe a finished child session gives, before the session commits it."""

    child: Path
    artifacts: PreludeArtifacts
    recipe: Recipe


@dataclass(frozen=True)
class PreludeResult:
    """The root commit the session starts from, the profile it runs on, the Recipe KB's leads for it and the upper
    bound of the GPU seconds the main-flow child used; the rest is in the notebook and ``MB-prelude/``."""

    root_commit: str
    profile: SessionProfile
    warm_leads: tuple[WarmLead, ...]
    child_gpu_seconds_upper_bound: float


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
        ran = decode(Workload, read_json_strict(earlier.manifest)["settings"]["workload"])
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


class Prelude:
    """Runs PRELUDE for one session; ``warm_start`` is ``None`` when no Recipe KB is configured."""

    def __init__(
        self,
        context: SchedulerContext,
        repo: ExperimentRepo,
        gate: SessionGate,
        tests: SessionTests,
        *,
        warm_start: WarmStartSource | None,
    ) -> None:
        self._layout = context.layout
        self._warm_start_source = warm_start
        self._processes = context.processes
        self._notebook = context.notebook
        self._repo = repo
        self._gate = gate
        self._tests = tests
        self._decisions = context.decisions
        self._clock = context.clock

    async def root(
        self, workload: Workload, budget: SessionBudget, hard_stop: HardStop, base_image_digest: str
    ) -> PreludeResult:
        """Root the session on the main flow's result, on ``base_image_digest``; raises :class:`PreludeFailed` when
        there is none.

        The baseline is measured afterwards, by :meth:`measure_baseline`, once the integration branch has its K.
        """

        spawned = self._clock.now()
        child = await self._run_child(workload, budget, hard_stop)
        derived = await self._derive_root(child, workload.framework, base_image_digest)
        return await self._commit_root(derived, workload, (self._clock.now() - spawned) * workload.tp)

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
        return await self._commit_root(derived, workload, 0.0)

    async def _derive_root(self, child: Path, framework: str, base_image_digest: str) -> _DerivedRoot:
        try:
            artifacts = _succeeded(await asyncio.to_thread(read_prelude, child))
            recipe = await self._root_recipe(artifacts, framework, base_image_digest)
        except (PreludeReadError, OSError, BlobError) as exc:
            raise PreludeFailed(f"cannot root the session on the baseline prelude in {child}: {exc}") from exc
        return _DerivedRoot(child, artifacts, recipe)

    async def _commit_root(self, derived: _DerivedRoot, workload: Workload, child_gpu_seconds: float) -> PreludeResult:
        child, artifacts, recipe = derived.child, derived.artifacts, derived.recipe
        try:
            saved = await asyncio.to_thread(self._save_foreign_patches, artifacts)
        except OSError as exc:
            raise PreludeFailed(f"cannot root the session on the baseline prelude in {child}: {exc}") from exc
        atomic_write_json(self._layout.root_recipe, recipe.to_document())
        root = await self._repo.init_root(recipe)
        await self._notebook.record_lineage(
            LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root, recipe_digest=recipe.digest())
        )
        model_config = await asyncio.to_thread(summarize_model_config, workload.model)
        profile = _session_profile(self._layout.session_id, workload, artifacts, model_config)
        await self._notebook.write_session_profile(profile)
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE,
                ROOT_COMMITTED,
                INTEGRATION_BRANCH,
                {
                    "root_commit": root,
                    "child_session": str(child),
                    "unexpressed": list(recipe.unexpressed),
                    "unexpressed_patches": saved,
                    "setup_script": None if artifacts.setup_script is None else str(artifacts.setup_script),
                    "child_gpu_seconds_upper_bound": child_gpu_seconds,
                },
            )
        )
        return PreludeResult(root, profile, await self._warm_start(profile, recipe), child_gpu_seconds)

    async def _warm_start(self, profile: SessionProfile, root: Recipe) -> tuple[WarmLead, ...]:
        """The Recipe KB's leads for ``profile``, the best prepared for replay on ``root``; the ``PRELUDE`` decision
        names them, or why the KB was not read."""

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
        detail = {"leads": [lead.describe() for lead in leads], "unread": unread}
        self._decisions.record(Decision(DecisionKind.PRELUDE, "warm start read", INTEGRATION_BRANCH, detail))
        return leads

    async def _run_child(self, workload: Workload, budget: SessionBudget, hard_stop: HardStop) -> Path:
        try:
            hours = budget.prelude_max_hours(self._clock.now())
        except BudgetExhausted as exc:
            raise PreludeFailed(str(exc)) from exc
        argv = prelude_argv(workload, hours, self._layout.prelude_launch)
        self._decisions.record(Decision(DecisionKind.PRELUDE, "child started", PRELUDE_BRANCH, {"argv": list(argv)}))
        exit_code = await self._wait(
            await self._processes.spawn(argv, branch=PRELUDE_BRANCH, log=self._layout.prelude_log), hard_stop
        )
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

    async def _root_recipe(self, artifacts: PreludeArtifacts, framework: str, base_image_digest: str) -> Recipe:
        """The git calls and file reads run off the event loop, so a stop signal is not held up behind them."""

        pin = await asyncio.to_thread(framework_pin, artifacts)
        patches: list[PatchRef] = []
        for kept in artifacts.kept_patches:
            text = await asyncio.to_thread(kept.patch_file.read_text, encoding="utf-8")
            patches.append(await self._repo.store_patch(framework, text))
        return await asyncio.to_thread(root_recipe, artifacts, framework, pin, tuple(patches), base_image_digest)

    def _save_foreign_patches(self, artifacts: PreludeArtifacts) -> dict[str, str]:
        """Copy the patches no component carries to ``MB-prelude/``; ``{copy: the tree it applies to}``. Blocks."""

        saved: dict[str, str] = {}
        for index, patch in enumerate(artifacts.foreign_patches):
            copy = self._layout.prelude_patches / f"{index:04d}-{patch.patch_file.name}"
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(patch.patch_file, copy)
            saved[str(copy)] = str(patch.root)
        return saved

    async def measure_baseline(self, root: str) -> float:
        """Measure the baseline on ``root`` through the toolbox and have the rule gate anchor it; return the longest
        wall time, in seconds, of the root tests it anchored on. Raises :class:`PreludeFailed` when a baseline test is
        refused, which takes back the ones accepted before it, or the baseline anchors nothing."""

        request = integration_test_request(TestKind.TEST, Purpose.BASELINE, root)
        try:
            request_ids = [await self._tests.submit(request) for _ in range(await self._gate.baseline_test_count())]
        except RequestRejected as exc:
            await self._tests.withdraw_integration(BASELINE_REFUSED)
            raise PreludeFailed(f"the toolbox refused the baseline test: {exc}") from exc
        try:
            baseline = await self._gate.establish_baseline(root, request_ids)
        except NotMeasured as exc:
            raise PreludeFailed(f"the baseline anchors nothing: {exc}") from exc
        records = [await self._notebook.record(request_id) for request_id in baseline.request_ids]
        longest = _longest_wall_seconds(record.result for record in records if record.result is not None)
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE,
                "baseline established",
                INTEGRATION_BRANCH,
                {"request_ids": list(baseline.request_ids), "longest_test_seconds": longest},
            )
        )
        return longest


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


def _succeeded(artifacts: PreludeArtifacts) -> PreludeArtifacts:
    if not artifacts.succeeded:
        raise PreludeFailed(
            f"the baseline prelude ended with stop_reason={artifacts.stop_reason!r} "
            f"and baseline_tput={artifacts.baseline_tput}"
        )
    return artifacts


def _session_profile(
    session_id: str, workload: Workload, artifacts: PreludeArtifacts, model_config: Mapping[str, Any]
) -> SessionProfile:
    """``model_config`` is :func:`summarize_model_config` of the model; its type and architectures are left empty
    when the model's ``config.json`` does not name them. An AgentX profile has no sequence lengths, and the roofline
    ceiling is kept only on the metric the session is graded on."""

    shape = workload.inferencex
    return SessionProfile(
        session_id=session_id,
        model=workload.model,
        hardware=workload.gpu_type,
        framework=workload.framework,
        framework_version=artifacts.framework_version,
        precision=workload.precision,
        tp=workload.tp,
        conc=workload.conc,
        model_type=model_config.get("model_type", ""),
        architectures=tuple(model_config.get("architectures", ())),
        isl=None if shape is None else shape.isl,
        osl=None if shape is None else shape.osl,
        kernel_optimizer=workload.kernel_optimizer.value,
        roofline_ceiling={
            metric: ceiling for metric, ceiling in artifacts.roofline_ceiling.items() if metric == workload.metric
        },
    )


__all__ = ["PreludeFailed", "Prelude", "PreludeResult", "WarmStartSource", "prelude_argv", "reused_prelude"]
