# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The end-to-end skeleton: one session driven in-process on peer stand-ins and a virtual clock.

PRELUDE roots the session on a scripted child session; the judge opens two branches; ``b01`` commits and forks a fix
onto ``b03`` (a second fork is refused at the cap); ``b02`` commits the best single-branch value, holds a claim and a
process, and is stopped by the judge; the integrator merges ``c1`` and refuses ``c2``; FINAL weighs the judge-stopped
``b02`` against the integration head and passes over it, because it degrades at another conc; CLOSE leaves nothing
running.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest

from hyperloom.common.proctree import running
from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, LineageKind
from hyperloom_multibranch.contracts.scheduling import AdmissionStatus, ForkRequest
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta, intent
from hyperloom_multibranch.contracts.notebook.peers import Verdict
from hyperloom_multibranch.scheduling.budget import SessionBudget
from hyperloom_multibranch.scheduling.exports import SessionExports
from hyperloom_multibranch.scheduling.final import INTEGRATION_LABEL
from hyperloom_multibranch.scheduling.integrator import merge_intent
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.lifecycle import CUT_SHORT, Session, SessionOutcome
from hyperloom_multibranch.scheduling.llm import CompletionFailed
from hyperloom_multibranch.scheduling.peers import Peers, PeerSetup
from hyperloom_multibranch.scheduling.roster import BranchStatus, RosterCaps, StopCause
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload
from hyperloom_multibranch.scheduling.stops import STOP_OVERRAN, SessionStopReason
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    ROOT_COMMIT,
    MbFakeGate,
    MbFakeHeldRegistry,
    MbFakeLifecycle,
    MbFakeRepo,
    MbFakeToolbox,
    ScriptedCompletion,
    SpecialistScript,
    VirtualClock,
    judged,
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
SWEEPS = {
    ROOT_COMMIT: {CONC: 1000.0, 32: 1400.0},
    "i1": {CONC: 1040.0, 32: 1500.0},
    "c2": {CONC: 1065.0, 32: 1300.0},
}


def _reply(document: dict[str, object]) -> str:
    return json.dumps(document)


MERGE_VERDICTS = {merge_intent("c1"): judged(Verdict.IMPROVED, 4.0, "", 1040.0)}
OPENING = _reply({"open": 2, "open_k": 9, "branches": [{"branch": INTEGRATION_BRANCH, "k": 2}], "stop": []})
STEER = _reply({"open": 0, "branches": [{"branch": "b01", "k": 7}], "stop": [{"branch": "b02", "reason": STOP_REASON}]})


class InsistingLifecycle(MbFakeLifecycle):
    """A lifecycle whose stop hangs after the operator signals again; a failure double, not a placeholder."""

    async def stop(self, branch: str) -> None:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.Event().wait()


class StuckToolbox(MbFakeToolbox):
    """A toolbox whose withdrawal of every test never returns; a failure double, not a placeholder."""

    async def withdraw_all(self) -> tuple[str, ...]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class SignallingToolbox(StuckToolbox):
    """The stuck toolbox, signalling the scheduler once as it starts to withdraw; a failure double."""

    async def withdraw_all(self) -> tuple[str, ...]:
        os.kill(os.getpid(), signal.SIGTERM)
        return await super().withdraw_all()


class BrokenRepo(MbFakeRepo):
    """A repository that fails on the root commit with an error the session does not handle; a failure double."""

    async def init_root(self, recipe: Recipe) -> str:
        raise LookupError("no object store")


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
    toolbox_type: type[MbFakeToolbox] = MbFakeToolbox
    repo_type: type[MbFakeRepo] = MbFakeRepo
    toolbox: MbFakeToolbox = field(init=False)
    gate: MbFakeGate = field(init=False)
    repo: MbFakeRepo = field(init=False)
    setup: PeerSetup = field(init=False)
    signalled_at: float = 0.0

    def handle(self, role: Role, branch: str = "") -> LocalNotebook:
        return LocalNotebook(self.layout.notebook, Writer(role, branch or role.value, branch), clock=self.time.now)

    def factory(self, setup: PeerSetup) -> Peers:
        toolbox = self.toolbox_type(self.handle(Role.TOOLBOX), sweeps=SWEEPS, stuck=self.stuck)
        gate = MbFakeGate(self.handle(Role.GATE), toolbox, setup.desk, verdicts=MERGE_VERDICTS)
        repo = self.repo_type()
        self.toolbox, self.gate, self.repo, self.setup = toolbox, gate, repo, setup
        return Peers(self.lifecycle, toolbox, gate, repo)

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
        gate = stage.gate
        await stage.time.sleep(STEP_SECONDS)
        await gate.commit(branch, start, "c1", 1045.0)
        await stage.time.sleep(STEP_SECONDS)
        await gate.fork(branch, "c1", "f1")
        await gate.fork(branch, "c1", "f2")

    async def second(branch: str, start: str) -> None:
        gate = stage.gate
        await stage.time.sleep(STEP_SECONDS)
        await gate.commit(branch, start, "c2", 1060.0)
        runtime = stage.handle(Role.SPECIALIST_RUNTIME, branch)
        await runtime.acquire(intent(branch, "c2"), delta().fingerprint(), ttl_seconds=CLAIM_TTL_SECONDS)
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
    assert lineage.commit(ROOT_COMMIT) is not None
    assert (await observer.baseline()).root_commit == ROOT_COMMIT
    assert (await observer.session_profile()).session_id == stage.layout.session_id
    assert [fp.policy_versions["judge_prompt"] for fp in await observer.manifest_fingerprints()] == ["mb1-judge-1"]

    assert stage.lifecycle.started == {"b01": ROOT_COMMIT, "b02": ROOT_COMMIT, "b03": "f1"}
    assert [admission.status for admission in stage.gate.forks] == [
        AdmissionStatus.ACCEPTED,
        AdmissionStatus.REJECTED,
    ]
    assert branches["b03"].origin is LineageKind.SPLIT and branches["b03"].from_branch == "b01"

    assert stage.lifecycle.k["b01"] == 2
    assert (
        max(quota.k for policy in stage.toolbox.quotas for quota in policy.branches) == SETTINGS.caps.max_k_per_branch
    )
    quota = stage.toolbox.quotas[-2]
    assert quota.k_of(INTEGRATION_BRANCH) == 2 and quota.k_of("b02") is None

    roster = json.loads(stage.layout.roster.read_text())
    b02 = next(entry for entry in roster["entries"] if entry["branch"] == "b02")
    assert b02["stop"] == {"cause": StopCause.JUDGE.value, "reason": STOP_REASON, "at": b02["stop"]["at"]}
    assert branches["b02"].stop_reason == f"judge: {STOP_REASON}"
    assert "b02" in stage.lifecycle.stopped and "b02" in stage.toolbox.withdrawn
    assert not any(running(pid) for pid in stage.pids)
    claims = (await observer.snapshot()).claims
    assert [claim.status for claim in claims if claim.holder == "b02"] == [ClaimStatus.RELEASED]

    merged = {item.commit_id: item.last_merge.outcome.value for item in await observer.committed() if item.last_merge}
    assert merged["c1"] == "merged" and merged["c2"] == "not_stackable"
    assert stage.repo.integrated[0][2] == ("c1",)

    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.label == INTEGRATION_LABEL and outcome.final.chosen.commit == "i1"
    single = next(check for check in outcome.final.checks if check.candidate.label != INTEGRATION_LABEL)
    assert (single.candidate.branch, single.candidate.commit) == ("b02", "c2")
    assert {check.candidate.commit for check in outcome.final.checks} == {"i1", "c2"}
    assert json.loads(stage.layout.final_record.read_text())["choice"]["chosen"]["commit"] == "i1"

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
        stage.layout.final_record,
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
    assert stage.toolbox.withdrawn[:2] == [INTEGRATION_BRANCH, "*"]
    late = await stage.gate.desk.admit(ForkRequest("b01", ROOT_COMMIT, "fix after the stop"))
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
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_phases_that_cannot_unwind_after_a_signal_do_not_hold_close_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.stuck = frozenset({INTEGRATION_BRANCH})
    stage.lifecycle.scripts = {"b01": _signalling(stage)}

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.close is not None
    assert STOP_OVERRAN in outcome.close.stop.failures
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.SIGNAL.value
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_after_a_signal_the_session_ends_within_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.stuck = frozenset({INTEGRATION_BRANCH})
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


async def test_a_judge_that_never_answers_gives_up_at_the_explore_deadline(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = SilentModel()

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and stage.lifecycle.started == {"b01": ROOT_COMMIT}
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
    assert [row["commit"] for row in final_phase["detail"]["candidates"]] == [ROOT_COMMIT]
    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.commit == ROOT_COMMIT
    assert sorted(stage.lifecycle.stopped) == ["b01", "b02"]


async def test_a_first_signal_during_close_cuts_it_short_after_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])
    stage.toolbox_type = SignallingToolbox

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.GLOBAL_CONVERGED and outcome.final is not None
    assert CUT_SHORT in outcome.error and outcome.close is not None and outcome.close.stop.failures == (CUT_SHORT,)
    [close] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "close"]
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.time.now() <= float(str(close["at"])) + grace


async def test_an_unhandled_error_bounds_close_by_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.repo_type = StuckToolbox, BrokenRepo

    with pytest.raises(LookupError, match="no object store"):
        await stage.run()

    report = json.loads(stage.layout.report.read_text())
    assert report["stop_reason"] == SessionStopReason.ABORTED.value and STOP_OVERRAN in report["stop"]["failures"]
    [close] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "close"]
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.time.now() <= float(str(close["at"])) + grace


async def test_a_failed_opening_judge_call_opens_one_branch_at_k_one(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion(error=CompletionFailed("judge call failed: RuntimeError: status=529"))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert stage.lifecycle.started == {"b01": ROOT_COMMIT}
    assert stage.lifecycle.k == {"b01": 1}
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
