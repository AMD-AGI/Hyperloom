# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The end-to-end skeleton: one session driven in-process on peer stand-ins and a virtual clock.

PRELUDE roots the session on a scripted child session in a git repository; the judge opens two branches; ``b01``
commits ``c1`` through the rule gate and asks for a fork of a fix, which opens ``b03`` at ``c1`` (a second fork is
refused at the cap); ``b02`` commits ``c2a`` and then ``c2``, the best single-branch value, holds a claim and a process,
and is stopped by the judge; the integrator merges ``c1`` and refuses ``c2`` and ``c2a``; FINAL weighs the
judge-stopped ``b02`` against the integration head and passes over it, because it degrades at another conc; CLOSE
leaves nothing running.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest

from hyperloom.common.proctree import running
from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.fork import Admission, AdmissionStatus, ForkRequest
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta, intent
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.scheduling.budget import SessionBudget
from hyperloom_multibranch.scheduling.exports import SessionExports
from hyperloom_multibranch.scheduling.final import INTEGRATION_LABEL
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.lifecycle import CUT_SHORT, Session, SessionOutcome
from hyperloom_multibranch.scheduling.llm import CompletionFailed
from hyperloom_multibranch.scheduling.peers import Peers, PeerSetup, RootWriter
from hyperloom_multibranch.scheduling.requests import SessionTests
from hyperloom_multibranch.scheduling.roster import BranchStatus, RosterCaps, StopCause
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload
from hyperloom_multibranch.scheduling.stops import (
    STOP_OVERRAN,
    TOOLBOX_CLOSE_OVERRAN,
    TOOLBOX_CLOSE_SECONDS,
    SessionStopReason,
)
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    MAIN_POINT,
    REPRO_CASE,
    MbFakeHeldRegistry,
    MbFakeLifecycle,
    MbFakeSpecialist,
    HeldGitRepo,
    RecordingToolbox,
    ScriptedCompletion,
    SpecialistScript,
    StuckToolbox,
    VirtualClock,
    held_root_writer,
    recipe_effect,
    recipe_tags,
    tagged,
)
from hyperloom_multibranch.scheduling.tests.child_sessions import child_with_patch

START = 1_700_000_000.0
CONC = 16
WORKLOAD = Workload("/models/Llama-3-8B", "sglang", 1, 1024, 1024, CONC, "bf16")
SETTINGS = RunSettings(WORKLOAD, max_hours=0.5, caps=RosterCaps(3, 2, 8), judge_interval_seconds=600.0)
STEP_SECONDS = 30.0
CLAIM_TTL_SECONDS = 1e6
STOP_REASON = "no gain left"
CONVERGED_REASON = "every branch is flat"
#: The root, ``c1`` alone (the integration head once merged) and ``b02``'s head; every other recipe measures 1000.
CURVES = {
    frozenset(): {MAIN_POINT: 1000.0, CONC: 1000.0, 32: 1400.0},
    frozenset({"c1"}): {MAIN_POINT: 1040.0, CONC: 1040.0, 32: 1500.0},
    frozenset({"c2a", "c2"}): {MAIN_POINT: 1065.0, CONC: 1065.0, 32: 1300.0},
}
FIX_REASON = "fix f1 regresses"


def _reply(document: dict[str, object]) -> str:
    return json.dumps(document)


OPENING = _reply({"open": 2, "open_k": 9, "branches": [{"branch": INTEGRATION_BRANCH, "k": 2}], "stop": []})
STEER = _reply({"open": 0, "branches": [{"branch": "b01", "k": 7}], "stop": [{"branch": "b02", "reason": STOP_REASON}]})


class InsistingLifecycle(MbFakeLifecycle):
    """A lifecycle whose stop hangs after the operator signals again; a failure double, not a placeholder."""

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


def no_object_store(path: Path, time: VirtualClock) -> RootWriter:
    """A repository that fails on the root commit with an error the session does not handle; a failure double."""

    async def write_root(baseline: Recipe) -> str:
        raise LookupError(f"no object store at {path}")

    return write_root


class SilentModel(ScriptedCompletion):
    """A judge model that never answers; a failure double, not a placeholder."""

    async def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@dataclass
class Stage:
    layout: SessionLayout
    time: VirtualClock
    processes: MbFakeHeldRegistry
    lifecycle: MbFakeLifecycle
    judge: ScriptedCompletion
    pids: list[int]
    stuck: frozenset[str] = frozenset()
    toolbox_type: type[RecordingToolbox] = RecordingToolbox
    root_writer: Callable[[Path, VirtualClock], RootWriter] = held_root_writer
    toolbox: RecordingToolbox = field(init=False)
    specialist: MbFakeSpecialist = field(init=False)
    repo: HeldGitRepo = field(init=False)
    setup: PeerSetup = field(init=False)
    forks: list[Admission] = field(default_factory=list)
    signalled_at: float = 0.0

    def handle(self, role: Role, branch: str = "") -> LocalNotebook:
        return LocalNotebook(self.layout.notebook, Writer(role, branch or role.value, branch), clock=self.time.now)

    def factory(self, setup: PeerSetup) -> Peers:
        toolbox = self.toolbox_type(self.handle(Role.TOOLBOX), recipe_effect(setup.layout.repo, CURVES))
        if isinstance(toolbox, StuckToolbox) and self.stuck:
            toolbox.stuck = self.stuck
        gate_notebook = self.handle(Role.GATE)
        repo = HeldGitRepo(setup.layout.repo, self.time)
        gate = RuleGate(gate_notebook, repo, setup.desk)
        tests = SessionTests(toolbox, gate_notebook, self.time.clock)
        self.specialist = MbFakeSpecialist(gate, tests, gate_notebook, repo)
        self.toolbox, self.repo, self.setup = toolbox, repo, setup
        return Peers(self.lifecycle, toolbox, repo, self.root_writer(setup.layout.repo, self.time))

    async def root(self) -> str:
        return (await self.handle(Role.OBSERVER).baseline()).root_commit

    async def spawn_specialist_process(self, branch: str) -> None:
        """A process of ``branch``'s specialist, started the way a lifecycle starts one: through the peer setup."""

        log = self.layout.branch_dir(branch).with_suffix(".log")
        held = await self.setup.processes.spawn(("sleep", "600"), branch=branch, log=log)
        self.pids.append(held.record.pid)

    async def run(self) -> SessionOutcome:
        self.time.hold_threads()
        session = Session(
            SETTINGS,
            self.layout,
            self.factory,
            self.processes,
            judge_completion=self.judge,
            resolver_completion=ScriptedCompletion(),
            exports=partial(SessionExports, kb_client=None),
            clock=self.time.clock,
        )
        return await session.run()


def _stage(tmp_path: Path, *, child: Path | None, exit_code: int = 0) -> Stage:
    time = VirtualClock(START)
    layout = create_session(tmp_path / "workspace", WORKLOAD.model_basename, time.clock)
    session = "" if child is None else str(child)
    processes = MbFakeHeldRegistry(layout.pids, time, session=session, exit_code=exit_code)
    judge = ScriptedCompletion([OPENING, STEER])
    return Stage(layout, time, processes, MbFakeLifecycle(), judge, [])


def _scripts(stage: Stage) -> dict[str, SpecialistScript]:
    async def first(branch: str, start: str) -> None:
        specialist = stage.specialist
        await stage.time.sleep(STEP_SECONDS)
        await specialist.commit(branch, tagged("c1"), 1045.0)
        await stage.time.sleep(STEP_SECONDS)
        stage.forks.append(await specialist.fork(branch, tagged("f1"), FIX_REASON))
        stage.forks.append(await specialist.fork(branch, tagged("f2"), "fix f2 regresses"))

    async def second(branch: str, start: str) -> None:
        specialist = stage.specialist
        await stage.time.sleep(STEP_SECONDS)
        await specialist.commit(branch, tagged("c2a"), 1010.0)
        head = await specialist.commit(branch, tagged("c2"), 1060.0)
        runtime = stage.handle(Role.SPECIALIST_RUNTIME, branch)
        await runtime.acquire(intent(branch, head), delta().fingerprint(), ttl_seconds=CLAIM_TTL_SECONDS)
        await stage.spawn_specialist_process(branch)
        await asyncio.Event().wait()

    return {"b01": first, "b02": second}


async def test_a_session_runs_every_phase_and_leaves_nothing_behind(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.lifecycle.scripts = _scripts(stage)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert (stage.setup.workload, stage.setup.processes) == (WORKLOAD, stage.processes)
    observer = stage.handle(Role.OBSERVER)
    lineage = await observer.lineage()
    branches = {branch.branch: branch for branch in lineage.branches}
    root, c1, c2 = await stage.root(), await stage.repo.head("b01"), await stage.repo.head("b02")
    assert lineage.commit(root) is not None and (await stage.repo.read_recipe(root)).components
    assert (await observer.session_profile()).session_id == stage.layout.session_id
    assert [fp.policy_versions["judge_prompt"] for fp in await observer.manifest_fingerprints()] == ["mb1-judge-1"]

    assert stage.lifecycle.started == {"b01": root, "b02": root, "b03": c1}
    assert [admission.status for admission in stage.forks] == [AdmissionStatus.ACCEPTED, AdmissionStatus.REJECTED]
    assert (branches["b03"].origin, branches["b03"].start_commit) == (LineageKind.OPEN, c1)
    assert await stage.repo.head("b03") == c1 and FIX_REASON in str(stage.lifecycle.briefs["b03"])

    assert stage.lifecycle.last_k["b01"] == 2 and stage.lifecycle.running == {}
    specialist_ks = [
        k for policy in stage.toolbox.policies for b, k in policy.branch_k.items() if b != INTEGRATION_BRANCH
    ]
    assert max(specialist_ks) == SETTINGS.caps.max_k_per_branch
    steering = [policy for policy in stage.toolbox.policies if "b01" in policy.branch_k]
    assert steering[-1].k_for(INTEGRATION_BRANCH) == 2 and "b02" not in steering[-1].branch_k
    assert stage.toolbox.policies[-2].k_for(INTEGRATION_BRANCH) == SETTINGS.caps.max_parallel_tests

    roster = json.loads(stage.layout.roster.read_text())
    b02 = next(entry for entry in roster["entries"] if entry["branch"] == "b02")
    assert b02["stop"] == {"cause": StopCause.JUDGE.value, "reason": STOP_REASON, "at": b02["stop"]["at"]}
    assert branches["b02"].stop_reason == f"judge: {STOP_REASON}"
    assert branches[INTEGRATION_BRANCH].stopped_at is not None
    assert branches[INTEGRATION_BRANCH].stop_reason == f"session_end: {SessionStopReason.TIME_EXHAUSTED.value}"
    assert "b02" in stage.lifecycle.stopped and "b02" in stage.toolbox.cancelled
    assert stage.toolbox.serving and stage.toolbox.closed == 1 and stage.toolbox.results
    assert not any(running(pid) for pid in stage.pids)
    claims = (await observer.snapshot()).claims
    assert [claim.status for claim in claims if claim.holder == "b02"] == [ClaimStatus.RELEASED]

    merged = {item.commit_id: item.last_merge.outcome.value for item in await observer.committed() if item.last_merge}
    assert merged[c1] == "merged" and merged[c2] == "not_stackable" and len(merged) == 3
    head = await stage.repo.head(INTEGRATION_BRANCH)
    assert recipe_tags(await stage.repo.read_recipe(head)) == {"c1"}

    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.label == INTEGRATION_LABEL and outcome.final.chosen.commit == head
    single = next(check for check in outcome.final.checks if check.candidate.label != INTEGRATION_LABEL)
    assert (single.candidate.branch, single.candidate.commit) == ("b02", c2)
    assert {check.candidate.commit for check in outcome.final.checks} == {head, c2}
    breakdown = json.loads((stage.layout.reports / "session_breakdown.json").read_text())
    assert breakdown["close"]["final"]["commit_id"] == head
    assert breakdown["close"]["selection"]["chosen"]["commit"] == head

    assert all(entry["status"] == BranchStatus.STOPPED.value for entry in roster["entries"])
    assert outcome.close is not None and outcome.close.stop.clean and outcome.close.exports is not None
    assert set(json.loads(stage.layout.manifest.read_text())["code_version"]) == {"package", "git_sha"}
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

    async def interrupted(branch: str, start: str) -> None:
        await stage.time.sleep(STEP_SECONDS)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.Event().wait()

    stage.lifecycle.scripts = {"b01": interrupted}

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.exit_code == 1
    assert outcome.final is None and outcome.close is not None
    assert sorted(stage.lifecycle.stopped) == ["b01", "b02"]
    assert sorted(stage.toolbox.cancelled) == sorted(("b01", "b02", INTEGRATION_BRANCH))
    assert stage.toolbox.closed == 1
    late = await stage.setup.desk.request_fork(ForkRequest(await stage.root(), tagged("late"), REPRO_CASE, "late fix"))
    assert late.status is AdmissionStatus.REJECTED
    roster = json.loads(stage.layout.roster.read_text())
    assert roster["closed"] and all(
        entry["stop"]["cause"] == StopCause.SESSION_END.value for entry in roster["entries"]
    )


def _signalling(stage: Stage) -> SpecialistScript:
    """A specialist that starts a process of its branch, then the operator stops the session."""

    async def interrupted(branch: str, start: str) -> None:
        await stage.time.sleep(STEP_SECONDS)
        await stage.spawn_specialist_process(branch)
        stage.signalled_at = stage.time.now()
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.Event().wait()

    return interrupted


async def test_close_after_a_signal_waits_the_close_grace_not_the_budget(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.lifecycle = MbFakeLifecycle(scripts={"b01": _signalling(stage)}, hanging=frozenset({"b01", "b02"}))

    outcome = await stage.run()

    assert outcome.close is not None and STOP_OVERRAN in outcome.close.stop.failures
    assert stage.toolbox.closed == 1
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_a_toolbox_that_cannot_stop_after_a_signal_does_not_hold_close_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.stuck = StuckToolbox, frozenset({INTEGRATION_BRANCH})
    stage.lifecycle.scripts = {"b01": _signalling(stage)}

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.close is not None
    assert STOP_OVERRAN in outcome.close.stop.failures
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.SIGNAL.value
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_after_a_signal_the_session_ends_within_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.stuck = StuckToolbox, frozenset({INTEGRATION_BRANCH})
    stage.lifecycle = MbFakeLifecycle(scripts={"b01": _signalling(stage)}, hanging=frozenset({"b01", "b02"}))

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.close is not None
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.signalled_at and stage.time.now() <= stage.signalled_at + grace
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_a_signal_during_a_hanging_close_cuts_it_short_and_leaves_no_process(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    stage.lifecycle = InsistingLifecycle(scripts={"b01": _signalling(stage)})

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
    assert outcome.final is not None and stage.lifecycle.started == {}
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
    assert sorted(stage.lifecycle.stopped) == ["b01", "b02"]


async def test_a_specialist_that_never_stops_at_the_end_of_explore_does_not_hold_final_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])
    stage.lifecycle = MbFakeLifecycle(hanging=frozenset({"b01"}))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.GLOBAL_CONVERGED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b01"]
    assert [failure.split(":")[0] for failure in stop["detail"]["failures"]] == ["TimeoutError"]


async def test_a_judge_stop_that_never_returns_does_not_freeze_explore(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.lifecycle = MbFakeLifecycle(scripts=_scripts(stage), hanging=frozenset({"b02"}))

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
    stage.toolbox_type, stage.root_writer = StuckToolbox, no_object_store

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
    assert stage.lifecycle.started == {} and outcome.final is not None


async def test_a_failed_opening_judge_call_opens_one_branch_at_k_one(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion(error=CompletionFailed("judge call failed: RuntimeError: status=529"))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert stage.lifecycle.started == {"b01": await stage.root()}
    assert stage.lifecycle.last_k == {"b01": 1}
    failed = [row["reason"] for row in _decisions(stage) if row["kind"] == "judge_failed"]
    assert failed and set(failed) == {"kept the status quo"}


async def test_a_judge_stop_the_runtime_refuses_does_not_end_the_session(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.lifecycle = MbFakeLifecycle(scripts=_scripts(stage), refusing=frozenset({"b02"}))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert stop["detail"]["failures"] == ["RuntimeError: the runtime refused to stop b02"]
    assert stage.pids and not any(running(pid) for pid in stage.pids)
