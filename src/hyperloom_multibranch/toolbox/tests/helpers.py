# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fakes and builders for the toolbox tests: a repository, a compiler, a surface guard and a scripted executor."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.contracts.measurement import (
    AccuracyValue,
    FailureClass,
    Purpose,
    RoundValue,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.measure.surface import SurfaceViolation
from hyperloom.contracts.recipe import ConfigDelta, Recipe, RecipeDelta, RecipeError
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.attempts import MeasurementRefused, TestJob
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.submission_log import SubmissionLog
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox, ToolboxStores, ToolboxTuning

IMAGE = "sha256:" + "a" * 64
OTHER_IMAGE = "sha256:" + "b" * 64
PARENT = "c0"
BAD_DELTA = RecipeDelta(config=ConfigDelta(server_args="--does-not-compile"))


def delta(server_args: str = "--max-num-seqs 256") -> RecipeDelta:
    return RecipeDelta(config=ConfigDelta(server_args=server_args))


def req(
    branch: str = "b1",
    *,
    purpose: Purpose = Purpose.EXPLORE,
    kind: TestKind = TestKind.TEST,
    change: RecipeDelta | None = None,
    server_args: str = "--max-num-seqs 256",
    intent: str | None = None,
    **fields: object,
) -> TestRequest:
    if purpose in (Purpose.BASELINE, Purpose.HEALTH):
        change = RecipeDelta()
    return TestRequest(
        kind=kind,
        purpose=purpose,
        branch_id=branch,
        parent_commit=PARENT,
        delta=change if change is not None else delta(server_args),
        intent_fingerprint=intent,
        **fields,  # type: ignore[arg-type]
    )


class FakeRepo:
    def __init__(self, recipe: Recipe | None = None) -> None:
        self.recipes = {PARENT: recipe or Recipe(base_image_digest=IMAGE)}
        self.heads: dict[str, str] = {}

    async def head(self, branch: str) -> str:
        return self.heads.get(branch, PARENT)

    async def read_recipe(self, commit: str) -> Recipe:
        return self.recipes[commit]


class FakePlanner:
    """Refuses the deltas that ask for ``--does-not-compile``, as a patch that does not apply would, and a request
    whose ``surface_error`` is set, as a changed workload would."""

    def __init__(self) -> None:
        self.calls: list[RecipeDelta] = []
        self.mismatch: str | None = None

    def plan(self, request: TestRequest, parent: Recipe) -> object:
        self.calls.append(request.delta)
        if "--does-not-compile" in request.delta.config.server_args:
            raise RecipeError("patch patch/vllm/0001 does not apply")
        if self.mismatch:
            raise SurfaceViolation(self.mismatch)
        return {"env": parent.digest()}


def ok(job: TestJob, node: str = "n1", **fields: object) -> TestResult:
    return TestResult(
        request_id=job.request.request_id or "",
        validity=Validity.SUCCEEDED,
        rounds=(RoundValue("measure", 64, {"output_tps": 1000.0}),),
        accuracy=(AccuracyValue("gsm8k", "exact_match", 0.9),),
        sandbox=SandboxFacts(f"sb-{job.attempt}", node_name=node),
        started_at=1.0,
        finished_at=2.0,
        **fields,  # type: ignore[arg-type]
    )


def failed(job: TestJob, failure: FailureClass, node: str = "n1", reason: str = "boom") -> TestResult:
    validity = Validity.MEASUREMENT_FAILED if failure is FailureClass.INFRA else Validity.CANDIDATE_FAILED
    return TestResult(
        request_id=job.request.request_id or "",
        validity=validity,
        failure_class=failure,
        failure_reason=reason,
        sandbox=SandboxFacts(f"sb-{job.attempt}", node_name=node),
        started_at=1.0,
        finished_at=2.0,
    )


Behaviour = Callable[[TestJob], TestResult]


@dataclass
class ScriptedExecutor:
    """Runs attempts as ``script`` says (one behaviour per attempt, the last one repeats); can block on a gate."""

    script: list[Behaviour] = field(default_factory=list)
    jobs: list[TestJob] = field(default_factory=list)
    started: list[str] = field(default_factory=list)
    gate: asyncio.Event | None = None
    refuse: str | None = None
    unverified: str | None = None
    verified: list[TestResult] = field(default_factory=list)

    async def reverify(self, result: TestResult) -> None:
        self.verified.append(result)
        if self.unverified:
            raise ResultUnverified(result.request_id, self.unverified)

    async def run(self, job: TestJob) -> TestResult:
        self.jobs.append(job)
        self.started.append(job.request.request_id or "")
        if self.gate is not None:
            await self.gate.wait()
        if self.refuse:
            raise MeasurementRefused(self.refuse)
        index = min(len([j for j in self.jobs if j.request is job.request]) - 1, len(self.script) - 1)
        return self.script[index](job) if self.script else ok(job)


@dataclass
class Rig:
    toolbox: RealToolbox
    executor: ScriptedExecutor
    planner: FakePlanner
    repo: FakeRepo
    notebook: LocalNotebook
    root: Path
    clock: Callable[[], float]

    def open_notebook(self, role: Role, branch: str = "", name: str = "") -> LocalNotebook:
        return LocalNotebook(
            self.root / "notebook", Writer(role, name or branch or role.value, branch), clock=self.clock
        )


def build(
    tmp_path: Path,
    *,
    executor: ScriptedExecutor | None = None,
    policy: QueuePolicy | None = None,
    repo: FakeRepo | None = None,
    active_image: str | None = IMAGE,
    with_claims: bool = True,
    clock: Callable[[], float] | None = None,
    wrap_sink: Callable[[LocalNotebook], object] | None = None,
    planner: object | None = None,
    tools: Callable[[LocalNotebook], object] | None = None,
    workload_kind: WorkloadKind = WorkloadKind.INFERENCEX,
) -> Rig:
    """A toolbox over a notebook directory and a submit log under ``tmp_path``; building again over the same
    ``tmp_path`` is a restart."""

    from hyperloom_multibranch.notebook.tests.factories import FakeClock

    clock = clock or FakeClock()
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"), clock=clock)
    executor = executor or ScriptedExecutor()
    fake, repo = FakePlanner(), repo or FakeRepo()
    planner = planner or fake
    stores = ToolboxStores(
        log=SubmissionLog(tmp_path / "toolbox" / "submissions.jsonl", "campaign-1"),
        sink=wrap_sink(notebook) if wrap_sink else notebook,  # type: ignore[arg-type]
        health=notebook,
        claims=notebook if with_claims else None,
        records=notebook,
    )
    toolbox = RealToolbox(
        stores,
        executor,
        AcceptanceSetup(repo, planner, active_image, workload_kind),
        tools=tools(notebook) if tools else None,  # type: ignore[arg-type]
        tuning=ToolboxTuning(policy=policy, clock=clock),
    )
    return Rig(toolbox, executor, fake, repo, notebook, tmp_path, clock)


async def settle() -> None:
    """Let the tasks the toolbox started run until they block."""

    for _ in range(20):
        await asyncio.sleep(0)
