# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE: run the main flow once for the baseline, enablement and roofline, then root the session on its result.

The child is ``hyperloom optimize`` with every optimization lever off, given the whole remaining session budget. Its
session is read by :mod:`~hyperloom_multibranch.scheduling.prelude_reader` alone and folded into the root recipe,
whose kept patches go to the session's blob store and which the experiment repository stores as the root commit, and
the session profile is written. What the child kept that a recipe cannot express does not fail PRELUDE: the root
recipe names it as ``unexpressed``, and kept patches outside the framework checkout are copied to ``MB-prelude/``. The
Recipe KB's champions for that profile are read as warm-start leads for the
specialists; no Recipe KB, or none of its champions in this session's scope, gives none, and a KB that cannot be read
is recorded and gives none either. The baseline every recorded value is anchored to is measured through the toolbox in a
second step, once the caller has put the integration branch on the roster and given the toolbox its K: as many tests
of the root as the rule gate asks for, from which the gate writes the baseline and its bar.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import PatchRef, Recipe
from hyperloom.inference_optimizer.model_config_utils import summarize_model_config
from hyperloom.measure.compile import BlobError
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError
from hyperloom_multibranch.contracts.gate import NotMeasured, SessionGate
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, SessionProfile
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook.knowledge.warm_start import RecipeReader, WarmLead, warm_leads
from hyperloom_multibranch.scheduling.budget import BudgetExhausted, HardStop, SessionBudget
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind
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
from hyperloom_multibranch.scheduling.settings import Workload

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


class PreludeFailed(RuntimeError):
    """The baseline prelude established no baseline the session can start from."""


@dataclass(frozen=True)
class PreludeResult:
    """The root commit the session starts from, the profile it runs on and the Recipe KB's leads for it; the rest is
    in the notebook and ``MB-prelude/``."""

    root_commit: str
    profile: SessionProfile
    warm_leads: tuple[WarmLead, ...]


def prelude_argv(workload: Workload, max_hours: float, launch_info: Path) -> tuple[str, ...]:
    """The child's command line: ``hyperloom optimize`` on the interpreter this package runs on."""

    return (
        sys.executable,
        *HYPERLOOM_COMMAND,
        *("--model", workload.model, "--framework", workload.framework, "--gpu-type", workload.gpu_type),
        *("--tp", str(workload.tp), "--isl", str(workload.isl), "--osl", str(workload.osl)),
        *("--conc", str(workload.conc), "--precision", workload.precision),
        *PRELUDE_FLAGS,
        *("--max-hours", f"{max_hours:.6f}", "--launch-info-file", str(launch_info)),
    )


class Prelude:
    """Runs PRELUDE for one session; ``recipe_kb`` is ``None`` when no Recipe KB is configured."""

    def __init__(
        self,
        context: SchedulerContext,
        repo: ExperimentRepo,
        gate: SessionGate,
        tests: SessionTests,
        *,
        recipe_kb: RecipeReader | None,
    ) -> None:
        self._layout = context.layout
        self._recipe_kb = recipe_kb
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

        child = await self._run_child(workload, budget, hard_stop)
        try:
            artifacts = _succeeded(await asyncio.to_thread(read_prelude, child))
            recipe = await self._root_recipe(artifacts, workload.framework, base_image_digest)
            saved = await asyncio.to_thread(self._save_foreign_patches, artifacts)
        except (PreludeReadError, OSError, BlobError) as exc:
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
                "root committed",
                INTEGRATION_BRANCH,
                {
                    "root_commit": root,
                    "child_session": str(child),
                    "unexpressed": list(recipe.unexpressed),
                    "unexpressed_patches": saved,
                    "setup_script": None if artifacts.setup_script is None else str(artifacts.setup_script),
                },
            )
        )
        return PreludeResult(root, profile, await self._warm_start(profile))

    async def _warm_start(self, profile: SessionProfile) -> tuple[WarmLead, ...]:
        """The Recipe KB's leads for ``profile``; the ``PRELUDE`` decision names them, or why the KB was not read."""

        leads: tuple[WarmLead, ...] = ()
        unread = "no Recipe KB is configured" if self._recipe_kb is None else ""
        if self._recipe_kb is not None:
            try:
                leads = await asyncio.to_thread(warm_leads, self._recipe_kb, profile)
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

    async def measure_baseline(self, root: str) -> None:
        """Measure the baseline on ``root`` through the toolbox and have the rule gate anchor it; raises
        :class:`PreludeFailed` when a baseline test is refused, which takes back the ones accepted before it, or the
        baseline anchors nothing."""

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
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE,
                "baseline established",
                INTEGRATION_BRANCH,
                {"request_ids": list(baseline.request_ids)},
            )
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
    when the model's ``config.json`` does not name them."""

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
        isl=workload.isl,
        osl=workload.osl,
        kernel_optimizer=workload.kernel_optimizer.value,
        roofline_ceiling=dict(artifacts.roofline_ceiling),
    )


__all__ = ["PreludeFailed", "Prelude", "PreludeResult", "prelude_argv"]
