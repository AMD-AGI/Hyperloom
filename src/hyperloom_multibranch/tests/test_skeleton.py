# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The end-to-end skeleton: one session driven in-process on the specialist runtime, peer stand-ins and a virtual
clock.

PRELUDE roots the session on a scripted child session in a git repository; the judge opens two branches, whose
scripted specialists the runtime runs and judges; on ``b01`` the runtime confirms and commits ``c1``, and a fix of it is
asked for as a fork, which the stand-in gate splits off as ``b03`` and the scheduler starts at the fix (a second fork
is refused at the cap); on ``b02`` it commits ``c2``, the best single-branch value, whose specialist holds a claim and a process the AI work socket started, and the judge
stops ``b02``; the integrator merges ``c1`` and refuses ``c2``; FINAL weighs the judge-stopped ``b02`` against the
integration head and passes over it, because its head does not reproduce its recorded value; CLOSE leaves nothing
running.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.common.proctree import running
from hyperloom.contracts.measurement import Purpose, TestRequest
from hyperloom.contracts.recipe import Recipe, RecipeDelta, apply
from hyperloom.hosted.mcp_server import MIN_TOKEN_LENGTH, McpServer, McpServerConfig
from hyperloom.measure.versions import socket_versions
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission, ForkRequest
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.notebook.rsi.briefing import SessionBriefs
from hyperloom_multibranch.notebook.tests.factories import delta, intent
from hyperloom_multibranch.scheduling.budget import SessionBudget
from hyperloom_multibranch.scheduling.final import INTEGRATION_LABEL
from hyperloom_multibranch.scheduling.judge import JUDGE_PROMPT_VERSION
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.lifecycle import CUT_SHORT, Session, SessionHost, SessionOutcome, SessionServices
from hyperloom_multibranch.scheduling.llm import CompletionFailed
from hyperloom_multibranch.scheduling.peers import Peers, PeerSetup
from hyperloom_multibranch.scheduling.roster import BranchStatus, RosterCaps, StopCause
from hyperloom_multibranch.scheduling.settings import KernelOptimizer, RunSettings, Workload
from hyperloom_multibranch.scheduling.stops import (
    STOP_OVERRAN,
    TOOLBOX_CLOSE_OVERRAN,
    TOOLBOX_CLOSE_SECONDS,
    SessionStopReason,
)
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    MAIN_POINT,
    SPECIALIST_POLL_SECONDS,
    HeldGitRepo,
    MbFakeHeldRegistry,
    OneChampionKB,
    RecordedSessions,
    RecordingToolbox,
    RefusingRuntime,
    ScriptedCompletion,
    SessionStep,
    StuckToolbox,
    VirtualClock,
    recipe_effect,
    recipe_tags,
    session_gate,
    specialist_runtime,
    tagged,
)
from hyperloom_multibranch.scheduling.tests.child_sessions import child_with_patch
from hyperloom_multibranch.specialist.prompts import prompt_versions

START = 1_700_000_000.0
CONC = 16
WORKLOAD = Workload("/models/Llama-3-8B", "sglang", 1, 1024, 1024, CONC, "bf16", "mi355x", KernelOptimizer.FORGE)
SETTINGS = RunSettings(WORKLOAD, max_hours=0.5, caps=RosterCaps(3, 2, 8), judge_interval_seconds=600.0)
STEP_SECONDS = 30.0
CLAIM_TTL_SECONDS = 1e6
STOP_REASON = "no gain left"
CONVERGED_REASON = "every branch is flat"
#: What the scheduler's own tests measure: the root, ``c1`` alone (the integration head once merged) and ``c2``, which
#: sweeps higher than the integration head but measures the root's value when reproduced; every other recipe measures
#: 1000.
CURVES = {
    frozenset(): {MAIN_POINT: 1000.0, CONC: 1000.0},
    frozenset({"c1"}): {MAIN_POINT: 1040.0, CONC: 1040.0},
    frozenset({"c2"}): {MAIN_POINT: 1000.0, CONC: 1065.0},
}
#: What the specialists' tests and their retests measure: ``c2`` at 1060, which FINAL does not reproduce.
EXPLORED = {**CURVES, frozenset({"c2"}): {MAIN_POINT: 1060.0}}
SCRIPTS = {"b01": (tagged("c1"),), "b02": (tagged("c2"),)}
FIX_REASON = "fix f1 regresses"
#: An agent CLI as the AI work socket starts one: leading a session of its own.
AGENT_ARGV = ("sleep", "600")
#: Real seconds a scripted specialist's change gets to be judged and committed while virtual time stands still.
COMMIT_SECONDS = 30.0


def _reply(document: dict[str, object]) -> str:
    return json.dumps(document)


OPENING = _reply({"open": 2, "open_k": 9, "branches": [{"branch": INTEGRATION_BRANCH, "k": 2}], "stop": []})
STEER = _reply({"open": 0, "branches": [{"branch": "b01", "k": 7}], "stop": [{"branch": "b02", "reason": STOP_REASON}]})


@pytest.fixture(autouse=True)
def no_knowledge_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLOSE's knowledge steps find no archive: a test session leaves no knowledge behind."""

    monkeypatch.delenv(ARCHIVE_ENV, raising=False)


def _measured(repo: Path) -> Callable[[TestRequest], float]:
    """What a test measures: :data:`EXPLORED` for the specialists' tests and retests, :data:`CURVES` otherwise."""

    explored, measured = recipe_effect(repo, EXPLORED), recipe_effect(repo, CURVES)

    def effect(request: TestRequest) -> float:
        return (explored if request.purpose in (Purpose.EXPLORE, Purpose.CONFIRM) else measured)(request)

    return effect


class InsistingRuntime(RefusingRuntime):
    """A runtime whose stop hangs after the operator signals again; a failure double, not a placeholder."""

    async def stop(self, branch: str) -> None:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.Event().wait()


class SignallingToolbox(StuckToolbox):
    """The stuck toolbox, signalling the scheduler once as the session stop starts to withdraw the integration
    branch's tests; a failure double, not a placeholder."""

    stuck = frozenset({INTEGRATION_BRANCH})

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        if branch_id in self.stuck:
            os.kill(os.getpid(), signal.SIGTERM)
        return await super().cancel_branch(branch_id, reason)


class NoObjectStore(HeldGitRepo):
    """A repository that fails on the root commit with an error the session does not handle; a failure double."""

    async def init_root(self, recipe: Recipe) -> str:
        raise LookupError(f"no object store at {self.time}")


class SilentModel(ScriptedCompletion):
    """A judge model that never answers; a failure double, not a placeholder."""

    async def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@dataclass
class Stage:
    """One session on the specialist runtime: ``scripts`` and ``steps`` are what each branch's specialist does,
    the runtime refuses to stop a branch in ``refusing`` and a run of a branch in ``hanging`` never confirms its stop."""

    layout: SessionLayout
    time: VirtualClock
    processes: MbFakeHeldRegistry
    judge: ScriptedCompletion
    pids: list[int]
    scripts: Mapping[str, Sequence[RecipeDelta]] = field(default_factory=dict)
    steps: Mapping[str, SessionStep] = field(default_factory=dict)
    refusing: frozenset[str] = frozenset()
    hanging: frozenset[str] = frozenset()
    runtime_type: type[RefusingRuntime] = RefusingRuntime
    recipe_kb: OneChampionKB | None = None
    stuck: frozenset[str] = frozenset()
    toolbox_type: type[RecordingToolbox] = RecordingToolbox
    repo_type: type[HeldGitRepo] = HeldGitRepo
    toolbox: RecordingToolbox = field(init=False)
    sessions: RecordedSessions = field(init=False)
    repo: HeldGitRepo = field(init=False)
    setup: PeerSetup = field(init=False)
    briefs: SessionBriefs = field(init=False)
    forks: list[ForkAdmission] = field(default_factory=list)
    watchers: list[asyncio.Future[None]] = field(default_factory=list)
    signalled_at: float = 0.0

    def handle(self, role: Role, branch: str = "") -> LocalNotebook:
        return LocalNotebook(self.layout.notebook, Writer(role, branch or role.value, branch), clock=self.time.now)

    def factory(self, setup: PeerSetup) -> Peers:
        toolbox = self.toolbox_type(self.handle(Role.TOOLBOX), _measured(setup.layout.repo), conc=CONC)
        if isinstance(toolbox, StuckToolbox) and self.stuck:
            toolbox.stuck = self.stuck
        repo = self.repo_type(setup.layout.repo, setup.blobs, self.time)
        self.sessions = RecordedSessions(
            toolbox, repo, self.handle(Role.OBSERVER), scripts=self.scripts, steps=self.steps, hanging=self.hanging
        )
        runtime = specialist_runtime(
            setup.layout.notebook,
            toolbox,
            repo,
            self.sessions,
            now=self.time.now,
            desk=setup.desk,
            runtime=self.runtime_type,
        )
        runtime.refusing = self.refusing
        self.toolbox, self.repo, self.setup = toolbox, repo, setup
        self.briefs = SessionBriefs(self.handle(Role.OBSERVER))
        mcp = McpServer(McpServerConfig(host_token="t" * MIN_TOKEN_LENGTH), campaign=setup.layout.session_id)
        gate = session_gate(self.handle(Role.GATE), toolbox)
        return Peers(runtime, toolbox, repo, gate, mcp, self.briefs, toolbox.config.image_digest)

    async def root(self) -> str:
        return (await self.handle(Role.OBSERVER).baseline()).root_commit

    async def start_agent(self, branch: str) -> None:
        """An agent CLI of ``branch``'s specialist, started the way the AI work socket starts one and adopted by the
        session's process registry, as the runtime adopts every process it has started."""

        async with self.time.held():
            agent = await asyncio.create_subprocess_exec(*AGENT_ARGV, start_new_session=True)
        self.setup.processes.adopt(agent.pid, branch=branch, argv=AGENT_ARGV)
        self.pids.append(agent.pid)

    def once_committed(self, then: Callable[[str, str], Awaitable[None]]) -> SessionStep:
        """A step that stops virtual time until the runtime has committed on the branch, then runs ``then`` with the
        branch and its new head."""

        async def step(branch: str) -> None:
            self.time.hold()
            self.watchers.append(asyncio.ensure_future(self._after_commit(branch, then)))

        return step

    async def _after_commit(self, branch: str, then: Callable[[str, str], Awaitable[None]]) -> None:
        try:
            start = await self.repo.head(branch)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + COMMIT_SECONDS
            while (head := await self.repo.head(branch)) == start:
                assert loop.time() < deadline, f"nothing was committed on {branch}"
                await asyncio.sleep(SPECIALIST_POLL_SECONDS)
            await then(branch, head)
        finally:
            self.time.release()

    async def run(self) -> SessionOutcome:
        self.time.hold_threads()
        host = SessionHost(self.layout, self.processes, self.time.clock)
        session = Session(
            SETTINGS, self.factory, host, SessionServices(self.judge, ScriptedCompletion(), self.recipe_kb)
        )
        return await session.run()


def _stage(tmp_path: Path, *, child: Path | None, exit_code: int = 0) -> Stage:
    time = VirtualClock(START)
    layout = create_session(tmp_path / "workspace", WORKLOAD.model_basename, time.clock)
    session = "" if child is None else str(child)
    processes = MbFakeHeldRegistry(layout.pids, time, session=session, exit_code=exit_code)
    judge = ScriptedCompletion([OPENING, STEER])
    return Stage(layout, time, processes, judge, [])


def _script(stage: Stage) -> None:
    """``b01`` commits ``c1`` and asks for two forks of a fix of it; ``b02`` commits ``c2``, then holds a claim on it
    and a process."""

    async def forks(branch: str, c1: str) -> None:
        accepted = await stage.setup.desk.admit(ForkRequest(branch, c1, FIX_REASON))
        stage.forks.append(accepted)
        await stage.repo.create_branch(accepted.branch, c1)
        fixed = apply(await stage.repo.read_recipe(c1), tagged("f1"))
        fix = await stage.repo.commit(accepted.branch, c1, fixed, FIX_REASON)
        split = LineageEvent(LineageKind.SPLIT, accepted.branch, commit_id=fix, from_commit=c1, from_branch=branch)
        await stage.handle(Role.GATE).record_lineage(split)
        stage.forks.append(await stage.setup.desk.admit(ForkRequest(branch, c1, "fix f2 regresses")))

    async def holds(branch: str, c2: str) -> None:
        runtime = stage.handle(Role.SPECIALIST_RUNTIME, branch)
        await runtime.acquire(intent(branch, c2), delta().fingerprint(), ttl_seconds=CLAIM_TTL_SECONDS)
        await stage.start_agent(branch)

    stage.scripts = SCRIPTS
    stage.steps = {"b01": stage.once_committed(forks), "b02": stage.once_committed(holds)}


async def test_a_session_runs_every_phase_and_leaves_nothing_behind(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert (stage.setup.workload, stage.setup.processes) == (WORKLOAD, stage.processes)
    observer = stage.handle(Role.OBSERVER)
    lineage = await observer.lineage()
    branches = {branch.branch: branch for branch in lineage.branches}
    root, c1, c2 = await stage.root(), await stage.repo.head("b01"), await stage.repo.head("b02")
    assert lineage.commit(root) is not None and (await stage.repo.read_recipe(root)).components
    assert (await observer.session_profile()).session_id == stage.layout.session_id
    [fingerprint] = await observer.manifest_fingerprints()
    assert fingerprint.policy_versions["judge_prompt"] == JUDGE_PROMPT_VERSION
    assert dict(fingerprint.sockets) == socket_versions()
    assert dict(fingerprint.prompt_versions) == prompt_versions()

    fix = await stage.repo.head("b03")
    assert fix != c1 and stage.sessions.started == {"b01": root, "b02": root, "b03": fix}
    assert recipe_tags(await stage.repo.read_recipe(c1)) == {"c1"}
    assert recipe_tags(await stage.repo.read_recipe(c2)) == {"c2"}
    assert [(admission.status, admission.branch) for admission in stage.forks] == [
        (AdmissionStatus.ACCEPTED, "b03"),
        (AdmissionStatus.REJECTED, ""),
    ]
    assert (branches["b03"].origin, branches["b03"].start_commit) == (LineageKind.SPLIT, c1)

    assert stage.sessions.k["b01"] == 2 and stage.sessions.running == set()
    specialist_ks = [
        k for policy in stage.toolbox.policies for b, k in policy.branch_k.items() if b != INTEGRATION_BRANCH
    ]
    assert max(specialist_ks) == SETTINGS.caps.max_k_per_branch
    steering = [policy for policy in stage.toolbox.policies if "b01" in policy.branch_k]
    assert steering[-1].k_for(INTEGRATION_BRANCH) == 2 and "b02" not in steering[-1].branch_k
    assert all(policy.listed_branches_only for policy in stage.toolbox.policies)
    assert stage.toolbox.policies[-1].branch_k == {}

    roster = json.loads(stage.layout.roster.read_text())
    b02 = next(entry for entry in roster["entries"] if entry["branch"] == "b02")
    assert b02["stop"] == {"cause": StopCause.JUDGE.value, "reason": STOP_REASON, "at": b02["stop"]["at"]}
    assert branches["b02"].stop_reason == f"judge: {STOP_REASON}"
    assert branches[INTEGRATION_BRANCH].stopped_at is not None
    assert branches[INTEGRATION_BRANCH].stop_reason == f"session_end: {SessionStopReason.TIME_EXHAUSTED.value}"
    assert "b02" in stage.sessions.stopped and "b02" in stage.toolbox.cancelled
    assert stage.toolbox.serving and stage.toolbox.closed == 1 and stage.toolbox.results
    assert not any(running(pid) for pid in stage.pids)
    claims = (await observer.snapshot()).claims
    assert [claim.status for claim in claims if claim.holder == "b02"] == [ClaimStatus.RELEASED]

    merged = {item.commit_id: item.last_merge.outcome.value for item in await observer.committed() if item.last_merge}
    assert merged == {c1: "merged", c2: "not_stackable"}
    head = await stage.repo.head(INTEGRATION_BRANCH)
    assert recipe_tags(await stage.repo.read_recipe(head)) == {"c1"}

    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.label == INTEGRATION_LABEL and outcome.final.chosen.commit == head
    single = next(check for check in outcome.final.checks if check.candidate.label != INTEGRATION_LABEL)
    assert (single.candidate.branch, single.candidate.commit) == ("b02", c2)
    assert single.curve[CONC] > outcome.final.checks[0].curve[CONC] and not single.reproduced
    assert {check.candidate.commit for check in outcome.final.checks} == {head, c2}
    breakdown = json.loads((stage.layout.reports / "session_breakdown.json").read_text())
    assert breakdown["close"]["final"]["commit_id"] == head
    assert breakdown["close"]["selection"]["chosen"]["commit"] == head

    assert all(entry["status"] == BranchStatus.STOPPED.value for entry in roster["entries"])
    assert outcome.close is not None and outcome.close.stop.clean and outcome.close.exports is not None
    manifest = json.loads(stage.layout.manifest.read_text())
    assert set(manifest["code_version"]) == {"package", "git_sha"} and manifest["metric"] == GRADED_OUTPUT
    judge_calls = len(stage.layout.judge.read_text().splitlines())
    # One dashboard rebuild after each judge call during EXPLORE, and the stop's own read of the queue.
    assert judge_calls >= 2 and stage.toolbox.queue_reads == judge_calls + 1
    assert await stage.processes.leftovers() == ()
    for path in (
        stage.layout.manifest,
        stage.layout.decisions,
        stage.layout.judge,
        stage.layout.prelude_child,
        stage.layout.root_recipe,
        stage.layout.report,
        stage.layout.reports / "session_breakdown.json",
        stage.layout.merges,
        outcome.close.exports.dashboard.html,
    ):
        assert path.is_file(), path
    assert all(entry.name.startswith("MB-") for entry in stage.layout.root.iterdir())


async def test_the_specialists_are_briefed_on_the_warm_start_leads_prelude_read(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.recipe_kb = OneChampionKB()

    await stage.run()

    [lead] = stage.briefs.warm
    assert (lead.canonical_id, lead.server_args) == ("inference:champion", "--enable-expert-parallel")
    assert stage.recipe_kb.searched


async def test_a_failed_prelude_skips_to_close(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=None, exit_code=1)

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.PRELUDE_FAILED and outcome.exit_code == 1
    assert "exited with 1" in outcome.error
    assert outcome.final is None and outcome.close is not None and outcome.close.stop.clean
    assert stage.judge.calls == []
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.PRELUDE_FAILED.value


async def test_a_stop_signal_skips_to_close_and_stops_every_branch(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    async def interrupted(branch: str) -> None:
        await stage.time.sleep(STEP_SECONDS)
        os.kill(os.getpid(), signal.SIGTERM)

    stage.steps = {"b01": interrupted}

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.exit_code == 1
    assert outcome.final is None and outcome.close is not None
    assert sorted(stage.sessions.stopped) == ["b01", "b02"]
    assert set(stage.toolbox.cancelled) == {"b01", "b02", INTEGRATION_BRANCH}
    assert stage.toolbox.closed == 1
    late = await stage.setup.desk.admit(ForkRequest("b01", await stage.root(), "late fix"))
    assert late.status is AdmissionStatus.REJECTED
    roster = json.loads(stage.layout.roster.read_text())
    assert roster["closed"] and all(
        entry["stop"]["cause"] == StopCause.SESSION_END.value for entry in roster["entries"]
    )


def _signalling(stage: Stage) -> dict[str, SessionStep]:
    """``b01``'s specialist has an agent CLI started, then the operator stops the session."""

    async def interrupted(branch: str) -> None:
        await stage.time.sleep(STEP_SECONDS)
        await stage.start_agent(branch)
        stage.signalled_at = stage.time.now()
        os.kill(os.getpid(), signal.SIGTERM)

    return {"b01": interrupted}


async def test_close_after_a_signal_waits_the_close_grace_not_the_budget(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.steps, stage.hanging = _signalling(stage), frozenset({"b01", "b02"})

    outcome = await stage.run()

    assert outcome.close is not None and STOP_OVERRAN in outcome.close.stop.failures
    assert stage.toolbox.closed == 1
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_a_toolbox_that_cannot_stop_after_a_signal_does_not_hold_close_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.stuck = StuckToolbox, frozenset({INTEGRATION_BRANCH})
    stage.steps = _signalling(stage)

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.close is not None
    assert STOP_OVERRAN in outcome.close.stop.failures
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.SIGNAL.value
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_after_a_signal_the_session_ends_within_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.stuck = StuckToolbox, frozenset({INTEGRATION_BRANCH})
    stage.steps, stage.hanging = _signalling(stage), frozenset({"b01", "b02"})

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.close is not None
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.signalled_at and stage.time.now() <= stage.signalled_at + grace
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_a_signal_during_a_hanging_close_cuts_it_short_and_leaves_no_process(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.steps, stage.runtime_type = _signalling(stage), InsistingRuntime

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.exit_code == 1
    assert "cut short" in outcome.error
    assert outcome.close is not None and outcome.close.exports is None
    assert outcome.close.stop.failures == (CUT_SHORT,) and outcome.close.stop.leftovers == ()
    written = json.loads(stage.layout.report.read_text())
    assert (written["stop_reason"], written["stop"]["failures"]) == ("signal", [CUT_SHORT])
    assert stage.pids and not any(running(pid) for pid in stage.pids)
    assert await stage.processes.leftovers() == ()
    assert stage.toolbox.closed == 1


async def test_a_judge_that_never_answers_gives_up_at_the_explore_deadline_and_opens_nothing(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = SilentModel()

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and stage.sessions.started == {}
    decisions = [json.loads(line) for line in stage.layout.decisions.read_text().splitlines()]
    assert [row["reason"] for row in decisions if row["kind"] == "judge_failed"] == [
        "no answer by the EXPLORE deadline"
    ]


def _decisions(stage: Stage) -> list[dict[str, object]]:
    return [json.loads(line) for line in stage.layout.decisions.read_text().splitlines()]


async def test_a_judge_soft_stop_ends_explore_early_and_still_runs_final(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.GLOBAL_CONVERGED, "", 0)
    explore_end = SessionBudget.from_hours(SETTINGS.max_hours, START).phases_after_prelude(START).explore_end
    [final_phase] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "final"]
    assert float(str(final_phase["at"])) < explore_end
    root = await stage.root()
    assert [row["commit"] for row in final_phase["detail"]["candidates"]] == [root]
    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.commit == root
    assert sorted(stage.sessions.stopped) == ["b01", "b02"]


async def test_a_specialist_that_never_stops_at_the_end_of_explore_does_not_hold_final_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])
    stage.hanging = frozenset({"b01"})

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.GLOBAL_CONVERGED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b01"]
    assert [failure.split(":")[0] for failure in stop["detail"]["failures"]] == ["TimeoutError"]
    assert stage.sessions.stopped.count("b01") == 1 and "b01" in stage.sessions.running
    assert "the specialist of b01 did not confirm its stop" in outcome.close.stop.failures
    assert not outcome.completed and outcome.exit_code == 1


async def test_a_judge_stop_that_never_returns_does_not_freeze_explore(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)
    stage.hanging = frozenset({"b02"})

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert stop["reason"] == f"judge: {STOP_REASON}" and "b02" in stage.toolbox.cancelled
    assert len(stage.judge.calls) > 2


async def test_a_first_signal_during_close_cuts_it_short_after_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])
    stage.toolbox_type = SignallingToolbox

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.GLOBAL_CONVERGED and outcome.final is not None
    assert CUT_SHORT in outcome.error and outcome.close is not None
    assert outcome.close.stop.failures == (CUT_SHORT, TOOLBOX_CLOSE_OVERRAN)
    [close] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "close"]
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.time.now() <= float(str(close["at"])) + grace + TOOLBOX_CLOSE_SECONDS


async def test_an_unhandled_error_bounds_close_by_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.repo_type = StuckToolbox, NoObjectStore

    with pytest.raises(LookupError, match="no object store"):
        await stage.run()

    report = json.loads(stage.layout.report.read_text())
    assert report["stop_reason"] == SessionStopReason.ABORTED.value
    assert report["stop"]["failures"] == [TOOLBOX_CLOSE_OVERRAN]
    [close] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "close"]
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.time.now() <= float(str(close["at"])) + grace


async def test_a_soft_stop_on_the_opening_call_opens_no_branch(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 2, "stop_session": {"reason": CONVERGED_REASON}})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.GLOBAL_CONVERGED, "")
    assert stage.sessions.started == {} and outcome.final is not None


async def test_a_failed_opening_judge_call_opens_one_branch_at_k_one(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion(error=CompletionFailed("judge call failed: RuntimeError: status=529"))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert stage.sessions.started == {"b01": await stage.root()}
    assert stage.sessions.k == {"b01": 1}
    failed = [row["reason"] for row in _decisions(stage) if row["kind"] == "judge_failed"]
    assert failed and set(failed) == {"kept the status quo"}


async def test_a_judge_stop_the_runtime_refuses_does_not_end_the_session(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)
    stage.refusing = frozenset({"b02"})

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert stop["detail"]["failures"] == ["RuntimeError: the runtime refused to stop b02"]
    assert stage.pids and not any(running(pid) for pid in stage.pids)
