# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE: run the main flow once for the baseline, enablement and roofline, then root the session on its result.

The child is ``hyperloom optimize`` with every optimization lever off, given the whole remaining session budget. Its
session is read by :mod:`~hyperloom_multibranch.scheduling.prelude_reader` alone and folded into the root recipe,
whose kept patches go to the session's blob store and which the experiment repository stores as the root commit, and
the session profile is written. The baseline every recorded value is anchored to is measured through the toolbox in a
second step, once the caller has put the integration branch on the roster and given the toolbox its K.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.io import atomic_write_json
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import PatchRef, Recipe
from hyperloom.measure.compile import BlobError
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Notebook
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, SessionProfile
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.rulegate.accuracy import BaselineAccuracyError
from hyperloom_multibranch.scheduling.budget import BudgetExhausted, HardStop, SessionBudget
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.patches import PatchStore
from hyperloom_multibranch.scheduling.peers import RootWriter
from hyperloom_multibranch.scheduling.prelude_reader import (
    PreludeArtifacts,
    PreludeReadError,
    framework_pin,
    read_child_session,
    read_prelude,
    root_recipe,
)
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.scheduling.requests import SessionTests, TestWithdrawn, integration_test_request, together
from hyperloom_multibranch.scheduling.rule_judge import MB_FAKE_BASELINE_TESTS, MbFakeRuleJudge, NotMeasured
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


class PreludeFailed(RuntimeError):
    """The baseline prelude established no baseline the session can start from."""


@dataclass(frozen=True)
class PreludeResult:
    """The root commit the session starts from and the profile it runs on; the rest is in the notebook and
    ``MB-prelude/``."""

    root_commit: str
    profile: SessionProfile


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
    """Runs PRELUDE for one session."""

    def __init__(
        self,
        layout: SessionLayout,
        processes: ProcessRegistry,
        notebook: Notebook,
        write_root: RootWriter,
        patches: PatchStore,
        tests: SessionTests,
        judge: MbFakeRuleJudge,
        decisions: DecisionLog,
        clock: Clock,
    ) -> None:
        self._layout = layout
        self._processes = processes
        self._notebook = notebook
        self._write_root = write_root
        self._patches = patches
        self._tests = tests
        self._judge = judge
        self._decisions = decisions
        self._clock = clock

    async def root(self, workload: Workload, budget: SessionBudget, hard_stop: HardStop) -> PreludeResult:
        """Root the session on the main flow's result; raises :class:`PreludeFailed` when there is none.

        The baseline is measured afterwards, by :meth:`measure_baseline`, once the integration branch has its K.
        """

        child = await self._run_child(workload, budget, hard_stop)
        try:
            artifacts = _succeeded(await asyncio.to_thread(read_prelude, child))
            recipe = await self._root_recipe(artifacts, workload.framework)
        except (PreludeReadError, OSError, BlobError) as exc:
            raise PreludeFailed(f"cannot root the session on the baseline prelude in {child}: {exc}") from exc
        atomic_write_json(self._layout.root_recipe, recipe.to_document())
        root = await self._write_root(recipe)
        await self._notebook.record_lineage(
            LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root, recipe_digest=recipe.digest())
        )
        profile = _session_profile(self._layout.session_id, workload, artifacts)
        await self._notebook.write_session_profile(profile)
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE,
                "root committed",
                INTEGRATION_BRANCH,
                {
                    "root_commit": root,
                    "child_session": str(child),
                    "dropped": list(artifacts.dropped),
                    "setup_script": None if artifacts.setup_script is None else str(artifacts.setup_script),
                },
            )
        )
        return PreludeResult(root, profile)

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

    async def _root_recipe(self, artifacts: PreludeArtifacts, framework: str) -> Recipe:
        """The git calls and file reads run off the event loop, so a stop signal is not held up behind them."""

        pin = await asyncio.to_thread(framework_pin, artifacts)
        patches: list[PatchRef] = []
        for kept in artifacts.kept_patches:
            text = await asyncio.to_thread(kept.patch_file.read_text, encoding="utf-8")
            patches.append(await asyncio.to_thread(self._patches.store, framework, text))
        return await asyncio.to_thread(root_recipe, artifacts, framework, pin, tuple(patches))

    async def measure_baseline(self, root: str) -> None:
        """Measure the baseline on ``root``, accuracy included, through the toolbox and anchor it; raises
        :class:`PreludeFailed` when a baseline test is refused or withdrawn, or the baseline anchors nothing."""

        request = integration_test_request(TestKind.TEST, Purpose.BASELINE, root, accuracy=True)
        runs = [asyncio.ensure_future(self._tests.run(request)) for _ in range(MB_FAKE_BASELINE_TESTS)]
        try:
            await together(*runs)
            records = tuple(run.result() for run in runs)
            await self._judge.establish_baseline(root, records)
        except RequestRejected as exc:
            raise PreludeFailed(f"the toolbox refused the baseline test: {exc}") from exc
        except TestWithdrawn as exc:
            raise PreludeFailed(f"a baseline test was withdrawn: {exc}") from exc
        except (NotMeasured, BaselineAccuracyError) as exc:
            raise PreludeFailed(f"the baseline anchors nothing: {exc}") from exc
        request_ids = tuple(record.request_id for record in records)
        self._decisions.record(
            Decision(
                DecisionKind.PRELUDE, "baseline established", INTEGRATION_BRANCH, {"request_ids": list(request_ids)}
            )
        )


def _succeeded(artifacts: PreludeArtifacts) -> PreludeArtifacts:
    if not artifacts.succeeded:
        raise PreludeFailed(
            f"the baseline prelude ended with stop_reason={artifacts.stop_reason!r} "
            f"and baseline_tput={artifacts.baseline_tput}"
        )
    return artifacts


def _session_profile(session_id: str, workload: Workload, artifacts: PreludeArtifacts) -> SessionProfile:
    return SessionProfile(
        session_id=session_id,
        model=workload.model,
        hardware=workload.gpu_type,
        framework=workload.framework,
        framework_version=artifacts.framework_version,
        precision=workload.precision,
        tp=workload.tp,
        conc=workload.conc,
        isl=workload.isl,
        osl=workload.osl,
        roofline_ceiling=dict(artifacts.roofline_ceiling),
    )


__all__ = ["PreludeFailed", "Prelude", "PreludeResult", "prelude_argv"]
