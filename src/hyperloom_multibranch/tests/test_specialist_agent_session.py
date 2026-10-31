# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.hosted.mcp_server import McpPrincipal, ReplyCall
from hyperloom.contracts.recipe import ComponentPatches, ConfigDelta, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom.orchestrator.specialists.agent_runner import AgentEnded, AgentHandle, AgentStatus, LocalAgentRunner
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.judgement import Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.blackboard.records import LlmUsage
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, Conclusion, UsageSource
from hyperloom_multibranch.contracts.toolbox import BranchLease, QueueState, SpecialistGrant
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import intent, judgement, request, result
from hyperloom_multibranch.specialist.agent_session import (
    CLI_SESSION_FILE,
    IDLE_TURNS,
    NUDGE_WAITS_S,
    RESUME_CONTEXT_MAX,
    SPECIALIST_SESSION_S,
    WAITED_END,
    AgentCommand,
    AgentSession,
    AgentSetup,
    BranchPlace,
)
from hyperloom_multibranch.specialist.dev_sandbox import DevLeases, DevLimits, LeaseTerm
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionIdle, SessionStart, restart_notice
from hyperloom_multibranch.tests.branch_rig import branch_rig
from hyperloom_multibranch.tests.following_rig import diverged
from hyperloom_multibranch.toolbox.fake import FakeToolbox, FakeWiring
from hyperloom_multibranch.toolbox.fake_builds import FakeBuild, FakeBuildDriver, FakeBuildEnd
from hyperloom_multibranch.toolbox.mcp_tools import REPLY_INBOX_HEADER, reply_inbox

BRANCH, TOKEN_ENV, FIRST_PID = "s1", "HYPERLOOM_SPECIALIST_MCP_TOKEN", 4000
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)
EXITED_WELL = AgentStatus("exited", 0, None)
NO_WAITS = (0.0,) * IDLE_TURNS
LOST = AgentStatus("lost", None, None, error="no heartbeat for more than 300s")


class _Clock:
    now = 1_800_000_000.0


def _text(line: str) -> str:
    """The text of a stream-json user message, or ``line`` itself when it is not one (codex reads plain text)."""
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return line
    return message["message"]["content"][0]["text"]


class _Runner:
    """Launches nothing: launch ``n`` runs until the test ends it with ``end(n, status, seconds)``, and keeps that
    status as a real runner does; ``states`` holds the conversation record each launch found; ``sent`` holds
    the text of every message sent to a running CLI, ``beats`` counts the beats its session gave."""

    def __init__(self, clock: _Clock) -> None:
        self.clock, self.specs, self.prompts, self.cancelled, self.collected = clock, [], [], [], []
        self.ends: dict[int, tuple[AgentStatus, float]] = {}
        self.sent: list[str] = []
        self.beats = 0
        self.exits_before_send: dict[int, tuple[AgentStatus, float]] = {}
        self.states: list[dict | None] = []
        self.done: dict[int, AgentStatus] = {}

    def end(self, n: int, status: AgentStatus, seconds: float) -> None:
        self.ends[n] = (status, seconds)

    async def launch(self, spec):
        self.specs.append(spec)
        self.prompts.append(_text((spec.workspace / "prompt.md").read_text()))
        state = spec.workspace / CLI_SESSION_FILE
        self.states.append(json.loads(state.read_text()) if state.exists() else None)
        spec.heartbeat_sink(self._beat)
        return AgentHandle(spec.task_id, FIRST_PID + len(self.specs), self.clock.now)

    def _beat(self) -> None:
        self.beats += 1

    async def send(self, handle, data):
        n = handle.pid - FIRST_PID
        if n in self.exits_before_send:
            self.ends[n] = self.exits_before_send.pop(n)
            raise AgentEnded("its CLI has ended")
        self.sent.append(_text(data.decode()))

    async def close_stdin(self, handle):
        pass

    async def poll(self, handle):
        n = handle.pid - FIRST_PID
        if n in self.done:
            return self.done[n]
        ended = self.ends.pop(n, None)
        if ended is None:
            return AgentStatus("running", None, None)
        self.clock.now += ended[1]
        self.done[n] = ended[0]
        return ended[0]

    async def cancel(self, handle):
        self.cancelled.append(handle.pid - FIRST_PID)
        if handle.pid - FIRST_PID not in self.done:
            self.ends.setdefault(handle.pid - FIRST_PID, (AgentStatus("exited", -9, None), 0.0))

    async def collect(self, handle):
        self.collected.append(handle.pid - FIRST_PID)


class _Tokens:
    def __init__(self) -> None:
        self.issued, self.revoked = [], []

    def issue(self, branch, scopes):
        self.issued.append(branch)
        return SpecialistGrant("http://127.0.0.1:9/mcp", TOKEN_ENV, f"token-of-{branch}")

    def revoke(self, branch):
        self.revoked.append(branch)


class _Processes:
    def __init__(self) -> None:
        self.adopted = []

    def adopt(self, pid, *, branch, argv):
        self.adopted.append((pid - FIRST_PID, branch, tuple(argv)))


@pytest.fixture
async def rig(tmp_path):
    built = await branch_rig(tmp_path, ROOT, BRANCH)
    base, repo, root, blackboard = built.base, built.repo, built.root, built.blackboard
    await blackboard.append_branch_log(BranchLogEntry(BRANCH, BranchLogType.PLAN, "try fp8 KV cache next"))
    clock = _Clock()
    rig = SimpleNamespace(
        root=root,
        repo=repo,
        clock=clock,
        runner=_Runner(clock),
        tokens=_Tokens(),
        processes=_Processes(),
        worktree=base.worktree(BRANCH, tmp_path / "s1" / "worktree"),
        workspace=tmp_path / "s1" / "session",
        blackboard=blackboard,
        toolbox_blackboard=LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox")),
    )

    def session(command=AgentCommand("claude", "claude"), runner=None, **overrides) -> AgentSession:
        setup = AgentSetup(
            command=command,
            role="You are a long-running optimization specialist.",
            env={"PATH": os.environ.get("PATH", "")},
            runner=runner or rig.runner,
            tokens=rig.tokens,
            processes=rig.processes,
            repo=repo,
            poll_seconds=0.01,
            clock=lambda: clock.now,
            **overrides,
        )
        place = BranchPlace(rig.worktree, blackboard, rig.workspace)
        return AgentSession(SessionStart(BRANCH, root, 2, BranchTask("leads: fp8 KV cache")), setup, place)

    rig.session = session
    return rig


async def _until(condition, what: str) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


async def test_a_session_that_ends_as_it_should_is_followed_by_a_fresh_one(rig):
    rig.workspace.mkdir(parents=True)
    (rig.workspace / "specialist_done.json").write_text("{}")
    session = rig.session()
    running = asyncio.create_task(session.run())

    await _until(lambda: len(rig.runner.specs) == 1, "the first launch")
    await session.set_k(4)
    rig.runner.end(1, EXITED_WELL, 3600.0)
    await _until(lambda: len(rig.runner.specs) == 2, "a fresh session")
    rig.runner.end(2, AgentStatus("exited", -9, None, timed_out=True), 6 * 3600.0)
    await _until(lambda: len(rig.runner.specs) == 3, "a session after the deadline")
    await session.stop()
    await asyncio.wait_for(running, 1)

    first, second, third = rig.runner.prompts
    assert first.startswith("Startup brief:\nleads: fp8 KV cache") and f"Its head: {rig.root}" in first
    assert "try fp8 KV cache next" in first and "at once: 2" in first
    assert "at once: 4" in second and "ended: its model finished its turn" in second
    assert "ended: it reached its deadline" in third
    spec = rig.runner.specs[0]
    assert (spec.cwd, spec.env[TOKEN_ENV], spec.gpus) == (rig.worktree.path, "token-of-s1", 0)
    assert spec.deadline_unix - 1_800_000_000.0 == SPECIALIST_SESSION_S >= 7 * 24 * 3600
    assert not (rig.workspace / "specialist_done.json").exists()
    assert [pid for pid, _, _ in rig.processes.adopted] == [1, 2, 3]
    assert rig.processes.adopted[0][1:] == (BRANCH, spec.argv)
    assert (rig.runner.cancelled, rig.runner.collected) == ([3], [1, 2, 3])
    assert (rig.tokens.issued, rig.tokens.revoked) == ([BRANCH], [BRANCH])
    assert rig.worktree.head() == rig.root


def _end_turn(rig, *rows) -> None:
    """Append ``rows`` and the ``result`` row that ends a turn to the CLI's log."""
    with (rig.workspace / "process.log").open("a") as log:
        for row in (*rows, {"type": "result", "subtype": "success"}):
            log.write(json.dumps(row) + "\n")


async def test_a_claude_cli_lives_on_after_its_turn_and_what_arrives_for_its_branch_wakes_it(rig):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    spec = rig.runner.specs[0]
    assert spec.stdin_open and spec.argv[spec.argv.index("--input-format") + 1] == "stream-json"
    assert "--include-partial-messages" in spec.argv

    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats, "an idle session to keep its heartbeat")
    assert rig.runner.sent == []
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.runner.sent) == 1, "the result to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert len(rig.runner.specs) == 1
    assert "this arrived for your branch" in rig.runner.sent[0] and "result r1" in rig.runner.sent[0]


async def test_a_cli_that_exits_as_it_is_woken_ends_its_launch_as_any_exit(rig):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    rig.runner.exits_before_send[1] = (EXITED_WELL, 3600.0)

    _end_turn(rig)
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.runner.specs) == 2, "a fresh launch after the CLI ended")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert rig.runner.sent == [] and "ended: its model finished its turn" in rig.runner.prompts[1]


async def test_a_profile_of_its_head_that_another_branch_runs_keeps_the_cli_waiting_and_wakes_it(rig):
    shared = request(
        "p1", intent(INTEGRATION_BRANCH, rig.root), RecipeDelta(), kind=TestKind.PROFILE, purpose=Purpose.PROFILE
    )
    await rig.toolbox_blackboard.write_request(shared)
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats + 2, "the turn's end to be waited out")
    assert rig.runner.sent == []
    await rig.toolbox_blackboard.write_result(result("p1"))
    await _until(lambda: len(rig.runner.sent) == 1, "the shared profile's result to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "p1" in rig.runner.sent[0] and "profile of your head" in rig.runner.sent[0]


async def test_a_runtime_notice_written_while_the_turn_is_over_reaches_the_cli(rig):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    _end_turn(rig)
    await rig.blackboard.append_branch_log(
        BranchLogEntry(BRANCH, BranchLogType.RESULT, f"{NOTICE} you may now have 3 tests running at once")
    )
    await _until(lambda: len(rig.runner.sent) == 1, "the notice to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "you may now have 3 tests running at once" in rig.runner.sent[0]


def _exhausted(why: str, detail: str) -> BranchLogEntry:
    record = {"direction": "Fusion", "head": "a" * 40, "tried": ["r0"], "why": why, "detail": detail, "reopen_if": "-"}
    return BranchLogEntry(BRANCH, BranchLogType.PLAN, f"Fusion is done. EXHAUSTED {json.dumps(record)}")


async def test_an_exhausted_record_its_turn_wrote_with_no_tried_path_is_refused_back_to_the_cli(rig):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    await rig.blackboard.append_branch_log(_exhausted("blocked_by_tool", "Memset needs an aiter C++ change."))
    _end_turn(rig)
    await _until(lambda: len(rig.runner.sent) == 1, "the refusal to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "EXHAUSTED record of 'Fusion' was refused" in rig.runner.sent[0]
    assert "names no file or call you tried and no error it gave" in rig.runner.sent[0]
    logged = [record.entry.content for record in await rig.blackboard.branch_log(BRANCH)]
    assert sum(content.startswith(NOTICE) and "was refused" in content for content in logged) == 1


async def test_an_exhausted_record_that_names_its_tried_path_is_not_sent_back(rig):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    detail = "aiter.jit.module_gemm_a8w8_bpreshuffle fails to import: ModuleNotFoundError"
    await rig.blackboard.append_branch_log(_exhausted("blocked_by_tool", detail))
    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats + 2, "the turn's end to be waited out")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert rig.runner.sent == []


async def test_idle_turns_in_a_row_with_nothing_running_end_the_session_idle_with_a_nudge_between(rig):
    await rig.toolbox_blackboard.write_request(request("r0", intent(BRANCH, rig.root)))
    await rig.toolbox_blackboard.write_result(result("r0"))
    rig.clock.now = time.time()
    session = rig.session(nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    for nudged in range(1, IDLE_TURNS):
        _end_turn(rig)
        await _until(lambda n=nudged: len(rig.runner.sent) == n, f"nudge {nudged}")
    _end_turn(rig)

    with pytest.raises(SessionIdle, match=f"{IDLE_TURNS} turns in a row"):
        await asyncio.wait_for(running, 1)
    assert IDLE_TURNS == len(NUDGE_WAITS_S) and len(rig.runner.sent) == IDLE_TURNS - 1
    assert all("nothing new for your branch" in text for text in rig.runner.sent)
    assert (rig.runner.cancelled, rig.runner.collected) == ([1], [1])


_WORKED = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "blackboard_brief", "input": {}}]}}


async def test_a_turn_that_called_tools_but_tested_nothing_counts_half_an_idle_turn(rig):
    session = rig.session(nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    for nudged in range(1, 2 * IDLE_TURNS):
        _end_turn(rig, _WORKED)
        await _until(lambda n=nudged: len(rig.runner.sent) == n, f"nudge {nudged}")
    _end_turn(rig, _WORKED)

    with pytest.raises(SessionIdle) as idle:
        await asyncio.wait_for(running, 1)
    assert len(rig.runner.sent) == 2 * IDLE_TURNS - 1
    assert not idle.value.tested


async def test_an_idle_session_says_whether_it_submitted_a_test(rig):
    session = rig.session(nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    await rig.toolbox_blackboard.write_result(result("r1"))

    for sent in range(1, IDLE_TURNS + 1):
        _end_turn(rig)
        await _until(lambda n=sent: len(rig.runner.sent) == n, f"message {sent}")
    _end_turn(rig)

    with pytest.raises(SessionIdle) as idle:
        await asyncio.wait_for(running, 1)
    assert idle.value.tested


async def test_a_profile_is_no_tested_candidate_for_an_idle_session(rig):
    session = rig.session(nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    await rig.toolbox_blackboard.write_request(
        request("p1", intent(BRANCH, rig.root), kind=TestKind.PROFILE, purpose=Purpose.PROFILE)
    )
    await rig.toolbox_blackboard.write_result(result("p1"))

    for sent in range(1, IDLE_TURNS + 1):
        _end_turn(rig)
        await _until(lambda n=sent: len(rig.runner.sent) == n, f"message {sent}")
    _end_turn(rig)

    with pytest.raises(SessionIdle) as idle:
        await asyncio.wait_for(running, 1)
    assert not idle.value.tested


async def test_what_arrives_within_the_patience_wakes_the_model_instead_of_a_nudge(rig):
    session = rig.session(nudge_waits_s=(30.0,) * IDLE_TURNS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats + 2, "the patience to be waited, beating")
    assert rig.runner.sent == []
    await rig.blackboard.append_branch_log(
        BranchLogEntry(BRANCH, BranchLogType.RESULT, f"{NOTICE} you may now have 3 tests running at once")
    )
    await _until(lambda: len(rig.runner.sent) == 1, "the notice to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "3 tests running at once" in rig.runner.sent[0] and "nothing new" not in rig.runner.sent[0]


def _building(rig) -> tuple[FakeToolbox, FakeBuildDriver]:
    """A fake toolbox writing to the rig's blackboard whose every test needs a held sgl_kernel build."""
    driver = FakeBuildDriver()
    driver.need(lambda request: True, "sgl_kernel")
    driver.gate.clear()
    box = FakeToolbox(blackboard=rig.toolbox_blackboard, autorun=False, wiring=FakeWiring(builds=driver))
    return box, driver


def _native_test(rig) -> TestRequest:
    return replace(request("n1", intent(BRANCH, rig.root)), request_id=None, intent_fingerprint=None)


async def test_a_test_waiting_on_its_build_keeps_the_turn_from_counting_idle_and_the_build_wakes_it(rig):
    box, driver = _building(rig)
    session = rig.session(nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    request_id = await box.submit_test(_native_test(rig))
    assert (await box.queue_state()).pending_build_by_branch == {BRANCH: 1}

    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats + 20, "the turn's wait, beating")
    assert rig.runner.sent == [] and not running.done()
    driver.gate.set()
    await _until(lambda: len(rig.runner.sent) == 1, "the build to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)
    await box.close()

    assert f"- build {request_id}: build sgl_kernel " in rig.runner.sent[0]
    assert f" ready for {request_id}\n" in rig.runner.sent[0]


async def test_a_build_the_machine_failed_wakes_the_cli_with_its_detail(rig):
    box, driver = _building(rig)
    driver.script("sgl_kernel", FakeBuild(FakeBuildEnd.INFRA, "the build runner was lost"))
    session = rig.session(nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    request_id = await box.submit_test(_native_test(rig))

    _end_turn(rig)
    driver.gate.set()
    await _until(lambda: len(rig.runner.sent) == 1, "the build's failure to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)
    await box.close()

    assert f" failed for {request_id}: BuildInfraError: the build" in rig.runner.sent[0]


async def test_a_build_item_reaches_the_next_tool_reply_unless_that_reply_carries_its_test(rig):
    box, driver = _building(rig)
    request_id = await box.submit_test(_native_test(rig))
    annotate = reply_inbox(lambda branch: rig.blackboard)
    principal = McpPrincipal("specialist", BRANCH, BRANCH)

    def call(tool: str, **arguments) -> ReplyCall:
        return ReplyCall(principal, tool, arguments, is_error=False)

    start = (await annotate(call("blackboard_brief"), None)).cursor
    driver.gate.set()
    await box.drain()
    shown = await annotate(call("blackboard_brief"), start)
    hidden = await annotate(call("wait_test", request_id=request_id), start)
    await box.close()

    assert shown.text.splitlines()[0] == REPLY_INBOX_HEADER
    assert shown.text.splitlines()[1].startswith(f"build {request_id}: build sgl_kernel ")
    assert hidden.text == "" and hidden.cursor == shown.cursor


async def test_each_turn_records_the_tokens_it_added(rig):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session(command=AgentCommand("claude", "claude", model="claude-opus-5-5"), nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    init = {"type": "system", "subtype": "init", "session_id": "sess-1"}

    def call(n: int, tokens: int) -> dict:
        usage = {"input_tokens": tokens, "output_tokens": 10}
        return {"type": "assistant", "message": {"id": f"m{n}", "model": "claude-opus-5-5", "usage": usage}}

    async def recorded() -> list:
        return [record.usage.tokens for record in (await rig.blackboard.snapshot()).llm_usage]

    _end_turn(rig, init, call(1, 1_000))
    await _until(lambda: rig.runner.beats > 0, "the first turn to end")
    first = await recorded()
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.runner.sent) == 1, "the result to wake the CLI")
    _end_turn(rig, call(2, 500))
    await _until(lambda: len(rig.runner.sent) == 2, "the second turn to end")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert first == [TokenCounts(1_000, 10)]
    assert await recorded() == [TokenCounts(1_000, 10), TokenCounts(500, 10)]


async def test_a_launch_ended_early_to_wait_for_its_test_is_no_crash_and_the_next_waits_for_the_result(rig):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the first launch")

    rig.runner.end(1, EXITED_WELL, 30.0)
    await asyncio.sleep(0.2)
    assert len(rig.runner.specs) == 1 and not running.done()
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.runner.specs) == 2, "a launch once the test has its result")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert f"ended: {WAITED_END}" in rig.runner.prompts[1]


async def test_each_launch_records_its_tokens_at_the_gateway_s_rates_however_it_ended(rig):
    session = rig.session(command=AgentCommand("claude", "claude", model="claude-opus-5-5"))
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the first launch")
    opus = {"id": "m1", "model": "claude-opus-5-5", "usage": {"input_tokens": 1_000, "output_tokens": 100}}
    rows = [{"type": "system", "subtype": "init", "session_id": "sess-1"}, {"type": "assistant", "message": opus}]
    (rig.workspace / "process.log").write_text("".join(json.dumps(row) + "\n" for row in rows))
    rig.runner.end(1, EXITED_WELL, 3600.0)
    await _until(lambda: len(rig.runner.specs) == 2, "a fresh session")
    odd = {"id": "m2", "model": "claude-unknown-1", "usage": {"input_tokens": 1_000_000}}
    (rig.workspace / "process.log").write_text(json.dumps({"type": "assistant", "message": odd}) + "\n")
    await session.stop()
    await asyncio.wait_for(running, 1)

    first, killed = (record.usage for record in (await rig.blackboard.snapshot()).llm_usage)
    assert first == LlmUsage(BRANCH, UsageSource.SPECIALIST, first.cost_usd, TokenCounts(1_000, 100), "sess-1")
    assert first.cost_usd == pytest.approx((1_000 * 4 + 100 * 20) / 1e6)
    assert (killed.cost_usd, killed.tokens) == (None, TokenCounts(input=1_000_000))


async def test_each_launch_keeps_the_stream_of_the_one_before_while_its_readers_see_only_its_own(rig):
    session = rig.session(command=AgentCommand("claude", "claude", model="claude-opus-5-5"), nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the first launch")

    def call(n: int, tokens: int) -> dict:
        usage = {"input_tokens": tokens, "output_tokens": 10}
        return {"type": "assistant", "message": {"id": f"m{n}", "model": "claude-opus-5-5", "usage": usage}}

    _end_turn(rig, call(1, 1_000))
    first = (rig.workspace / "process.log").read_text()
    await _until(lambda: rig.runner.beats > 0, "the first turn to end")
    rig.runner.end(1, EXITED_WELL, 3600.0)
    await _until(lambda: len(rig.runner.specs) == 2, "a fresh session")
    assert not (rig.workspace / "process.log").exists()
    beats = rig.runner.beats
    _end_turn(rig, call(2, 500))
    await _until(lambda: rig.runner.beats > beats, "the second launch's first turn to end")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert (rig.workspace / "process-1.log").read_text() == first
    current = (rig.workspace / "process.log").read_text()
    assert '"m2"' in current and '"m1"' not in current
    assert sorted(path.name for path in rig.workspace.glob("process*.log")) == ["process-1.log", "process.log"]
    tokens = [record.usage.tokens for record in (await rig.blackboard.snapshot()).llm_usage]
    assert tokens == [TokenCounts(1_000, 10), TokenCounts(500, 10)]


async def test_a_restarted_session_numbers_the_kept_streams_on_from_those_in_its_workspace(rig):
    rig.workspace.mkdir(parents=True)
    for name in ("process-1.log", "process-3.log", "process.log", "process-x.log"):
        (rig.workspace / name).write_text(name)
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert (rig.workspace / "process-4.log").read_text() == "process.log"
    assert not (rig.workspace / "process.log").exists()


async def test_each_launch_shows_the_brief_rebuilt_at_the_head_it_starts_at(rig):
    asked = []

    async def briefs(branch, head):
        asked.append((branch, head))
        return f"brief {len(asked)} for {branch} at {head}"

    session = rig.session(briefs=briefs)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the first launch")
    rig.runner.end(1, EXITED_WELL, 3600.0)
    await _until(lambda: len(rig.runner.specs) == 2, "a fresh session")
    await session.stop()
    await asyncio.wait_for(running, 1)

    first, second = rig.runner.prompts
    assert asked == [(BRANCH, rig.root), (BRANCH, rig.root)]
    assert first.startswith(f"Startup brief:\nbrief 1 for {BRANCH} at {rig.root}")
    assert second.startswith(f"Startup brief:\nbrief 2 for {BRANCH} at {rig.root}")


async def test_a_brief_that_cannot_be_rebuilt_leaves_the_first_one_and_a_warning(rig, caplog):
    async def briefs(branch, head):
        raise RuntimeError("the Experience KB is unreachable")

    session = rig.session(briefs=briefs)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert rig.runner.prompts[0].startswith("Startup brief:\nleads: fp8 KV cache")
    assert "could not be rebuilt" in caplog.text and "the Experience KB is unreachable" in caplog.text


async def test_a_lost_session_is_killed_collected_and_raised_for_the_runtime(rig):
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    rig.runner.end(1, LOST, 900.0)

    with pytest.raises(SessionCrashed, match="lost") as raised:
        await asyncio.wait_for(running, 1)
    assert (raised.value.lost, raised.value.worked) == (True, False)
    assert (rig.runner.cancelled, rig.runner.collected) == ([1], [1])


async def test_a_loss_after_a_session_that_worked_says_so(rig):
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    rig.runner.end(1, EXITED_WELL, 3600.0)
    await _until(lambda: len(rig.runner.specs) == 2, "a fresh session")

    rig.runner.end(2, LOST, 900.0)

    with pytest.raises(SessionCrashed, match="lost") as raised:
        await asyncio.wait_for(running, 1)
    assert (raised.value.lost, raised.value.worked) == (True, True)


@pytest.mark.parametrize(
    ("status", "seconds", "said"),
    [(AgentStatus("exited", 1, None, error="auth failed"), 3600.0, "exited 1"), (EXITED_WELL, 30.0, "too soon")],
    ids=["error-exit", "too-soon"],
)
async def test_a_session_that_errs_or_ends_within_minutes_is_raised(rig, status, seconds, said):
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    rig.runner.end(1, status, seconds)

    with pytest.raises(SessionCrashed, match=said) as raised:
        await asyncio.wait_for(running, 1)
    assert not raised.value.lost


def test_codex_reads_its_role_with_the_prompt(tmp_path):
    codex = AgentCommand("codex", "codex", model="gpt-5")

    assert codex.stdin("ROLE", "PROMPT") == "ROLE\n\nPROMPT"
    assert codex.argv(tmp_path / "role.md", tmp_path / "wt")[-3:] == ("-m", "gpt-5", "-")
    with pytest.raises(ValueError, match="claude or codex"):
        AgentCommand("gemini", "gemini")


_STAND_IN = """#!/usr/bin/env python3
import json, os, pathlib, sys
role = pathlib.Path(sys.argv[sys.argv.index("--system-prompt-file") + 1])
first = json.loads(sys.stdin.readline())
seen = {
    "prompt": first["message"]["content"][0]["text"],
    "mcp_config": "--mcp-config" in sys.argv,
    "token": os.environ.get("HYPERLOOM_SPECIALIST_MCP_TOKEN"),
}
(role.parent / "seen.json").write_text(json.dumps(seen))
sys.stdin.read()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the AI work socket starts the CLI as a POSIX process group")
async def test_a_real_session_feeds_its_cli_the_prompt_and_the_mcp_service(tmp_path, rig):
    cli = tmp_path / "fake-claude"
    cli.write_text(_STAND_IN)
    cli.chmod(0o755)
    runner = LocalAgentRunner(
        session_dir=tmp_path / "agents", poll_interval_seconds=0.05, version_probe=lambda a, e: "x"
    )
    session = rig.session(command=AgentCommand("claude", str(cli)), runner=runner, min_session_seconds=0.0)
    running = asyncio.create_task(session.run())

    await _until(lambda: (rig.workspace / "seen.json").exists(), "the stand-in CLI to run")
    await session.stop()
    await asyncio.wait_for(running, 10)

    seen = json.loads((rig.workspace / "seen.json").read_text())
    assert seen["prompt"].startswith("Startup brief:\nleads: fp8 KV cache")
    assert (seen["mcp_config"], seen["token"]) == (True, "token-of-s1")


_INIT = {"type": "system", "subtype": "init", "session_id": "from-the-cli"}
CRASHED = AgentStatus("exited", 1, None, error="API Error: 529 overloaded")


def _call(context: int) -> dict:
    usage = {"input_tokens": 10, "cache_read_input_tokens": context - 10, "output_tokens": 5}
    return {"type": "assistant", "message": {"id": f"m{context}", "model": "claude-sonnet-5-5", "usage": usage}}


def _flag(argv, flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


async def _crash_after_a_turn(rig, session, n: int, status=CRASHED, *rows) -> SessionCrashed:
    """Run ``session`` until its launch ``n`` ended a turn, then end that launch with ``status``."""
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == n, f"launch {n}")
    beats = rig.runner.beats
    _end_turn(rig, _INIT, *(rows or (_call(50_000),)))
    await _until(lambda: rig.runner.beats > beats, "the turn to end")
    rig.runner.end(n, status, 3600.0)
    with pytest.raises(SessionCrashed) as raised:
        await asyncio.wait_for(running, 1)
    return raised.value


async def _next_launch(rig, n: int, **overrides) -> tuple[AgentSession, asyncio.Task]:
    session = rig.session(**overrides)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == n, f"launch {n}")
    return session, running


async def test_a_fresh_launch_names_its_conversation_and_writes_it_down_before_the_cli_starts(rig):
    session, running = await _next_launch(rig, 1)
    await session.stop()
    await asyncio.wait_for(running, 1)

    argv, state = rig.runner.specs[0].argv, rig.runner.states[0]
    assert _flag(argv, "--session-id") == state["session_id"] and "--resume" not in argv
    assert state["resumable"] is False and state["turns_done"] == 0 and state["head"] == rig.root
    assert (rig.workspace / CLI_SESSION_FILE).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("planted", [f".{CLI_SESSION_FILE}.partial", "system_prompt.md", "prompt.md"])
async def test_a_link_the_cli_left_in_the_workspace_is_not_written_through(rig, tmp_path, planted):
    target = tmp_path / "apps-file"
    rig.workspace.mkdir(parents=True)
    (rig.workspace / planted).symlink_to(target)

    session, running = await _next_launch(rig, 1)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert not target.exists()
    assert rig.runner.states[0] is not None and not (rig.workspace / planted).is_symlink()


@pytest.mark.parametrize("status", [CRASHED, LOST], ids=["error-exit", "lost"])
async def test_a_crashed_cli_is_resumed_by_the_next_run_with_what_arrived_while_it_was_down(rig, status):
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    crashed = await _crash_after_a_turn(rig, rig.session(), 1, status)
    await rig.toolbox_blackboard.write_result(result("r1"))
    (rig.worktree.path / "half-done.txt").write_text("an edit the CLI made before it crashed")

    session, running = await _next_launch(rig, 2)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert crashed.resumable
    first, resumed = rig.runner.specs
    assert _flag(resumed.argv, "--resume") == _flag(first.argv, "--session-id")
    assert "--session-id" not in resumed.argv
    message = rig.runner.prompts[1]
    assert "Startup brief" not in message and "result r1" in message
    assert ("exited 1" if status is CRASHED else "lost") in message and "as you left it" in message
    assert (rig.worktree.path / "half-done.txt").exists()


async def test_a_resume_after_the_branch_head_moved_resets_the_worktree_and_says_so(rig, monkeypatch):
    await _crash_after_a_turn(rig, rig.session(), 1)
    monkeypatch.setattr(rig.worktree, "head", lambda: "an-older-head")

    session, running = await _next_launch(rig, 2)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "--resume" in rig.runner.specs[1].argv
    assert f"reset to your branch head {rig.root}" in rig.runner.prompts[1]


_TOO_LONG = {"type": "result", "subtype": "success", "is_error": True, "result": "Prompt is too long"}


_SUCCESS = {"type": "result", "subtype": "success"}


async def _idle_after_turns(rig, *last) -> SessionIdle:
    """Run a session whose first launch ends its turns with nothing to test, the last one logging ``last``."""
    session = rig.session(nudge_waits_s=NO_WAITS)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    for nudged in range(1, IDLE_TURNS):
        _end_turn(rig, _INIT, _call(50_000))
        await _until(lambda n=nudged: len(rig.runner.sent) == n, f"nudge {nudged}")
    with (rig.workspace / "process.log").open("a") as log:
        log.writelines(json.dumps(row) + "\n" for row in last)
    with pytest.raises(SessionIdle) as idle:
        await asyncio.wait_for(running, 1)
    return idle.value


async def _move_integration(rig) -> str:
    """Merge another branch's commit onto the integration head; the new head."""
    chunking = RecipeDelta(config=ConfigDelta(server_args="--enable-chunked-prefill"))
    await rig.repo.create_branch("s0", rig.root)
    tip = await rig.repo.commit("s0", rig.root, apply(ROOT, chunking), "enable chunked prefill")
    return await rig.repo.commit_integration(rig.root, chunking, (tip,))


async def test_an_idle_end_under_the_context_cap_is_resumed_with_what_changed_since(rig):
    idle = await _idle_after_turns(rig, _call(50_000), _SUCCESS)
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    await rig.toolbox_blackboard.write_result(result("r1"))
    moved = await _move_integration(rig)

    session, running = await _next_launch(rig, 2)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert idle.resumable
    first, resumed = rig.runner.specs
    assert _flag(resumed.argv, "--resume") == _flag(first.argv, "--session-id")
    assert "--session-id" not in resumed.argv
    message = rig.runner.prompts[1]
    assert "Startup brief" not in message and "result r1" in message
    assert "nothing to test" in message and "as you left it" in message
    assert f"integration head moved from {rig.root} to {moved}" in message
    assert f"Where the session's gains stack:\nThe integration head is {moved}" in message


async def test_an_idle_resume_with_nothing_new_says_so_and_names_the_unmoved_integration_head(rig):
    await _idle_after_turns(rig, _call(50_000), _SUCCESS)

    session, running = await _next_launch(rig, 2, idle_resume_wait_s=0.0)
    await session.stop()
    await asyncio.wait_for(running, 1)

    message = rig.runner.prompts[1]
    assert "--resume" in rig.runner.specs[1].argv
    assert "Nothing arrived for your branch" in message and f"integration head is still {rig.root}" in message
    assert "gains stack" not in message


@pytest.mark.parametrize(
    "rows",
    [(_call(RESUME_CONTEXT_MAX), _SUCCESS), (_TOO_LONG,)],
    ids=["past-the-context-cap", "prompt-too-long"],
)
async def test_an_idle_end_with_a_full_context_is_followed_by_a_fresh_conversation(rig, rows):
    idle = await _idle_after_turns(rig, *rows)

    session, running = await _next_launch(rig, 2)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert not idle.resumable
    first, second = (spec.argv for spec in rig.runner.specs)
    assert "--resume" not in second and _flag(second, "--session-id") != _flag(first, "--session-id")
    assert rig.runner.prompts[1].startswith("Startup brief")


async def test_a_crash_after_an_idle_resume_is_resumed_as_a_crash(rig):
    await _idle_after_turns(rig, _call(50_000), _SUCCESS)
    await _crash_after_a_turn(rig, rig.session(idle_resume_wait_s=0.0), 2)

    session, running = await _next_launch(rig, 3)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "--resume" in rig.runner.specs[2].argv
    assert "exited 1" in rig.runner.prompts[2] and "nothing to test" not in rig.runner.prompts[2]


async def test_a_session_that_ended_well_is_followed_by_a_fresh_conversation(rig):
    session, running = await _next_launch(rig, 1)
    _end_turn(rig, _INIT, _call(50_000))
    await _until(lambda: rig.runner.beats > 0, "the turn to end")
    rig.runner.end(1, EXITED_WELL, 3600.0)
    await _until(lambda: len(rig.runner.specs) == 2, "the next launch")
    await session.stop()
    await asyncio.wait_for(running, 1)

    first, second = (spec.argv for spec in rig.runner.specs)
    assert "--resume" not in second and _flag(second, "--session-id") != _flag(first, "--session-id")


@pytest.mark.parametrize(
    "rows",
    [(_call(RESUME_CONTEXT_MAX),), (_call(50_000), _TOO_LONG)],
    ids=["past-the-context-cap", "prompt-too-long"],
)
async def test_a_crash_with_a_full_context_starts_fresh(rig, rows):
    crashed = await _crash_after_a_turn(rig, rig.session(), 1, CRASHED, *rows)

    session, running = await _next_launch(rig, 2)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert not crashed.resumable
    assert "--resume" not in rig.runner.specs[1].argv and rig.runner.prompts[1].startswith("Startup brief")


async def test_a_crash_before_any_turn_ended_or_soon_after_a_clean_exit_starts_fresh(rig):
    session, running = await _next_launch(rig, 1)
    rig.runner.end(1, CRASHED, 3600.0)
    with pytest.raises(SessionCrashed) as before_a_turn:
        await asyncio.wait_for(running, 1)
    session, running = await _next_launch(rig, 2)
    _end_turn(rig, _INIT, _call(50_000))
    await _until(lambda: rig.runner.beats > 0, "the turn to end")
    rig.runner.end(2, EXITED_WELL, 30.0)
    with pytest.raises(SessionCrashed) as too_soon:
        await asyncio.wait_for(running, 1)

    assert not before_a_turn.value.resumable and not too_soon.value.resumable
    session, running = await _next_launch(rig, 3)
    await session.stop()
    await asyncio.wait_for(running, 1)
    assert "--resume" not in rig.runner.specs[2].argv


async def test_a_resume_the_cli_refuses_at_once_falls_back_to_a_fresh_session_in_the_same_run(rig):
    await _crash_after_a_turn(rig, rig.session(), 1)
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 2, "the resume")

    refused = {"type": "result", "subtype": "error_during_execution", "is_error": True, "num_turns": 0}
    (rig.workspace / "process.log").write_text(json.dumps(refused) + "\n")
    rig.runner.end(2, AgentStatus("exited", 1, None), 0.5)
    await _until(lambda: len(rig.runner.specs) == 3, "a fresh launch")
    await session.stop()
    await asyncio.wait_for(running, 1)

    first, resumed, fresh = (spec.argv for spec in rig.runner.specs)
    assert _flag(resumed, "--resume") == _flag(first, "--session-id") != _flag(fresh, "--session-id")
    assert "--resume" not in fresh and rig.runner.prompts[2].startswith("Startup brief")
    notices = [record.entry.content for record in await rig.blackboard.branch_log(BRANCH)]
    assert any(notice.startswith(NOTICE) and "a fresh session starts" in notice for notice in notices)


async def test_a_resumed_cli_that_crashes_before_finishing_a_turn_is_not_resumed_again(rig):
    await _crash_after_a_turn(rig, rig.session(), 1)
    session, running = await _next_launch(rig, 2)
    (rig.workspace / "process.log").write_text(json.dumps(_INIT) + "\n")
    rig.runner.end(2, CRASHED, 3600.0)
    with pytest.raises(SessionCrashed) as again:
        await asyncio.wait_for(running, 1)

    session, running = await _next_launch(rig, 3)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "--resume" in rig.runner.specs[1].argv and not again.value.resumable
    assert "--resume" not in rig.runner.specs[2].argv and "--session-id" in rig.runner.specs[2].argv


async def test_a_resumed_cli_that_finished_a_turn_may_be_resumed_again(rig):
    await _crash_after_a_turn(rig, rig.session(), 1)
    again = await _crash_after_a_turn(rig, rig.session(), 2)

    assert again.resumable and "--resume" in rig.runner.specs[1].argv


@pytest.mark.parametrize(
    "stored",
    ["{not json", json.dumps({"session_id": "a1"}), json.dumps({"session_id": 7, "resumable": True}), None],
    ids=["not-json", "missing-fields", "wrong-types", "missing-after-a-launch"],
)
async def test_a_conversation_state_that_cannot_be_read_starts_fresh_with_a_warning(rig, caplog, stored):
    rig.workspace.mkdir(parents=True)
    (rig.workspace / "process.log").write_text(json.dumps(_INIT) + "\n")
    if stored is not None:
        (rig.workspace / CLI_SESSION_FILE).write_text(stored)

    session, running = await _next_launch(rig, 1)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "--session-id" in rig.runner.specs[0].argv and "--resume" not in rig.runner.specs[0].argv
    assert "a fresh session starts" in caplog.text


async def test_codex_never_resumes(rig):
    session = rig.session(command=AgentCommand("codex", "codex"))
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    rig.runner.end(1, CRASHED, 3600.0)
    with pytest.raises(SessionCrashed) as crashed:
        await asyncio.wait_for(running, 1)

    assert not crashed.value.resumable and not (rig.workspace / CLI_SESSION_FILE).exists()
    assert "--session-id" not in rig.runner.specs[0].argv


_IDLE_RESTART = (
    f"{NOTICE} the specialist found nothing to test (its model ended 3 turns in a row with nothing to test); the "
    "branch resumes its conversation"
)
_CRASH_RESTART = (
    f"{NOTICE} the specialist crashed (SessionCrashed('the agent CLI exited 1')) and is restarted on this branch, "
    "resuming its conversation"
)
_MORE_ROOM = f"{NOTICE} you may now have 3 tests running at once"


async def _log(rig, *contents: str) -> None:
    for content in contents:
        await rig.blackboard.append_branch_log(BranchLogEntry(BRANCH, BranchLogType.RESULT, content))


async def test_an_idle_resume_leaves_the_runtime_s_notice_of_the_restart_out_of_what_arrived(rig):
    await _idle_after_turns(rig, _call(50_000), _SUCCESS)
    await _log(rig, _IDLE_RESTART)

    session, running = await _next_launch(rig, 2, idle_resume_wait_s=0.0)
    await session.stop()
    await asyncio.wait_for(running, 1)

    message = rig.runner.prompts[1]
    assert restart_notice(_IDLE_RESTART) and "--resume" in rig.runner.specs[1].argv
    assert "Nothing arrived for your branch" in message and "resumes its conversation" not in message


async def test_a_crash_resume_shows_the_notices_that_arrived_but_not_the_one_of_its_own_restart(rig):
    await _crash_after_a_turn(rig, rig.session(), 1)
    await _log(rig, _MORE_ROOM, _CRASH_RESTART)

    session, running = await _next_launch(rig, 2, idle_resume_wait_s=60.0)
    await session.stop()
    await asyncio.wait_for(running, 1)

    message = rig.runner.prompts[1]
    assert restart_notice(_CRASH_RESTART) and "--resume" in rig.runner.specs[1].argv
    assert "3 tests running at once" in message and "is restarted on this branch" not in message


@pytest.mark.parametrize("arrival", ["result", "notice", "integration"])
async def test_an_idle_resume_waits_until_something_arrives_for_its_branch(rig, arrival):
    await _idle_after_turns(rig, _call(50_000), _SUCCESS)
    await rig.toolbox_blackboard.write_request(request("r1", intent(BRANCH, rig.root)))
    await _log(rig, _IDLE_RESTART)
    session = rig.session(idle_resume_wait_s=60.0)
    running = asyncio.create_task(session.run())

    await asyncio.sleep(0.3)
    assert len(rig.runner.specs) == 1
    if arrival == "result":
        await rig.toolbox_blackboard.write_result(result("r1"))
        said = "result r1"
    elif arrival == "notice":
        await _log(rig, _MORE_ROOM)
        said = "3 tests running at once"
    else:
        said = f"integration head moved from {rig.root} to {await _move_integration(rig)}"
    await _until(lambda: len(rig.runner.specs) == 2, "the resume")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "--resume" in rig.runner.specs[1].argv and said in rig.runner.prompts[1]


async def test_an_idle_resume_nothing_reaches_starts_once_its_wait_is_over(rig):
    await _idle_after_turns(rig, _call(50_000), _SUCCESS)
    session = rig.session(idle_resume_wait_s=0.6)
    running = asyncio.create_task(session.run())

    await asyncio.sleep(0.2)
    assert len(rig.runner.specs) == 1
    await _until(lambda: len(rig.runner.specs) == 2, "the resume once the wait is over")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "--resume" in rig.runner.specs[1].argv and "Nothing arrived" in rig.runner.prompts[1]


async def test_a_branch_stopped_while_its_idle_resume_waits_launches_nothing(rig):
    await _idle_after_turns(rig, _call(50_000), _SUCCESS)
    session = rig.session(idle_resume_wait_s=60.0)
    running = asyncio.create_task(session.run())

    await asyncio.sleep(0.1)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert len(rig.runner.specs) == 1


async def test_a_fresh_launch_shows_what_the_integration_head_holds_and_what_the_head_lacks(rig, tmp_path):
    branched = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)

    session, running = await _next_launch(rig, 1)
    await session.stop()
    await asyncio.wait_for(running, 1)

    prompt = rig.runner.prompts[0]
    assert f"Where the session's gains stack:\nThe integration head is {branched.onto}" in prompt
    assert f"{branched.merged[:12]} of s0" in prompt and f"Your head {branched.own} lacks of it" in prompt


async def test_the_integration_head_moving_wakes_the_cli_with_what_it_now_holds(rig):
    session, running = await _next_launch(rig, 1)
    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats, "the turn's end to be waited out")
    moved = await _move_integration(rig)
    await _until(lambda: len(rig.runner.sent) == 1, "the move to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert rig.runner.sent[0].startswith("Since your last turn ended, the integration head moved.")
    assert f"The integration head is {moved}" in rig.runner.sent[0] and "--enable-chunked-prefill" in rig.runner.sent[0]


async def test_a_resume_after_the_integration_head_moved_mid_turn_shows_what_it_now_holds(rig):
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "launch 1")
    moved = await _move_integration(rig)
    beats = rig.runner.beats
    _end_turn(rig, _INIT, _call(50_000))
    await _until(lambda: rig.runner.beats > beats, "the turn to end")
    rig.runner.end(1, CRASHED, 3600.0)
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(running, 1)
    recorded = json.loads((rig.workspace / CLI_SESSION_FILE).read_text())["integration"]

    session, running = await _next_launch(rig, 2)
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert recorded == rig.root
    assert "--resume" in rig.runner.specs[1].argv
    assert f"Where the session's gains stack:\nThe integration head is {moved}" in rig.runner.prompts[1]


async def test_a_reroot_asked_for_waits_for_the_branch_s_test_then_moves_it_and_says_what_it_carried(rig, tmp_path):
    branched = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)
    await rig.toolbox_blackboard.write_request(request("r9", intent(BRANCH, branched.own)))
    session, running = await _next_launch(rig, 1)

    await rig.blackboard.append_branch_log(
        BranchLogEntry(BRANCH, BranchLogType.PLAN, "REROOT: it holds chunked prefill")
    )
    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats + 2, "the turn's end to be waited out")
    assert await rig.repo.head(BRANCH) == branched.own and rig.runner.sent == []
    await rig.toolbox_blackboard.write_result(result("r9"))
    await _until(lambda: len(rig.runner.sent) == 1, "the result to wake the CLI")
    _end_turn(rig)
    await _until(lambda: rig.runner.beats > beats + 6, "the judgement to be waited for")
    assert await rig.repo.head(BRANCH) == branched.own and len(rig.runner.sent) == 1
    flat = judgement(0.2, ("r9",), verdict=Verdict.FLAT)
    await LocalBlackboard(tmp_path / "blackboard", Writer(Role.GATE, "gate")).write_judgement(
        "r9", flat, Conclusion.RECORDED_ONLY
    )
    await _until(lambda: len(rig.runner.sent) == 2, "the judgement to wake the CLI")
    if "re-rooted" not in rig.runner.sent[1]:
        _end_turn(rig)
        await _until(lambda: len(rig.runner.sent) == 3, "the re-root to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert await rig.repo.head(BRANCH) == branched.onto
    rows = (await rig.blackboard.snapshot()).progress
    assert next(row.head_commit for row in rows if row.branch == BRANCH) == branched.onto
    woken = rig.runner.sent[-1]
    assert f"re-rooted from {branched.own} onto the integration head {branched.onto}" in woken
    assert (
        '"extra_envs": {"VLLM_ROCM_USE_AITER": "1"}' in woken and f"Your head {branched.onto} holds all of it" in woken
    )
    answers = [r.entry.content for r in await rig.blackboard.branch_log(BRANCH) if r.entry.content.startswith(NOTICE)]
    assert sum("re-rooted" in content for content in answers) == 1


async def test_a_reroot_waits_out_a_judge_notice_that_precedes_its_judgement(rig, tmp_path):
    """A fix judged on its performance alone is noticed before the gate commits it: moving the branch then would turn
    its retested gain into a record-only one."""
    branched = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)
    await rig.toolbox_blackboard.write_request(request("r9", intent(BRANCH, branched.own)))
    await rig.toolbox_blackboard.write_result(result("r9"))
    session, running = await _next_launch(rig, 1)

    await rig.blackboard.append_branch_log(
        BranchLogEntry(BRANCH, BranchLogType.PLAN, "REROOT: it holds chunked prefill")
    )
    alone = f"{NOTICE} r9 is judged on its performance alone: the case is not available"
    await rig.blackboard.append_branch_log(BranchLogEntry(BRANCH, BranchLogType.RESULT, alone, ("r9",)))
    _end_turn(rig)
    await _until(lambda: len(rig.runner.sent) == 1, "the notice to wake the CLI")
    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats + 2, "the notice to be waited past")
    assert await rig.repo.head(BRANCH) == branched.own

    gate = LocalBlackboard(tmp_path / "blackboard", Writer(Role.GATE, "gate"))
    await gate.write_judgement("r9", judgement(0.2, ("r9",), verdict=Verdict.FLAT), Conclusion.RECORDED_ONLY)
    await _until(lambda: any("re-rooted" in sent for sent in rig.runner.sent), "the re-root once judged")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert await rig.repo.head(BRANCH) == branched.onto


class _DevToolbox:
    def __init__(self, root) -> None:
        self.root, self.released = root, []

    async def queue_state(self) -> QueueState:
        return QueueState()

    async def lease_branch_sandbox(self, branch_id, cards, *, wait_sec=None, charge=None, lifetime_sec=None):
        return BranchLease("lease-1", branch_id, (0,), self.root, {}, 3)

    async def release_branch_sandbox(self, lease_id: str) -> None:
        self.released.append(lease_id)


class _Forge:
    """A dev sandbox whose one started process runs until the test ends it or the sandbox is stopped."""

    def __init__(self, root) -> None:
        self.started, self.returncode, self.root, self.lifetime_s = False, None, root, math.inf

    async def start(self, command, cwd, env):
        self.started = True
        return self

    async def output(self) -> str:
        return ""

    def alive(self) -> bool:
        return self.started and self.returncode is None

    async def stop(self) -> None:
        self.returncode = -15


async def _forge_running(rig, tmp_path) -> SimpleNamespace:
    """The branch's long dev lease, running Forge."""
    toolbox, forge = _DevToolbox(tmp_path / "lease-1"), _Forge(tmp_path / "lease-1")
    toolbox.root.mkdir()
    leases = DevLeases(toolbox, lambda branch, lease: forge, lambda branch: rig.blackboard, limits=DevLimits(cap=1))
    await leases.lease(BRANCH, 1, 180, LeaseTerm.LONG)
    await leases.start(BRANCH, "python -m kernelforge.cli forge-loop", None)
    return SimpleNamespace(leases=leases, toolbox=toolbox, forge=forge)


async def test_a_turn_waiting_on_its_long_dev_lease_s_process_is_not_idle_and_its_end_wakes_the_cli(rig, tmp_path):
    dev = await _forge_running(rig, tmp_path)
    session = rig.session(nudge_waits_s=NO_WAITS, dev_leases=dev.leases)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.sent or rig.runner.beats > beats + 20, "the turn's wait, beating")
    assert rig.runner.sent == [] and not running.done(), "Forge running in its long lease is the branch's work"
    dev.forge.returncode = 0
    await _until(lambda: len(rig.runner.sent) == 1, "a nudge once Forge ended")
    await session.stop()
    await asyncio.wait_for(running, 1)
    assert dev.toolbox.released == ["lease-1"], "with Forge ended, the run's stop gives the lease back"


async def test_a_cli_waiting_on_its_long_dev_lease_is_woken_before_the_reaper_takes_its_silence_for_a_loss(
    rig, tmp_path
):
    dev = await _forge_running(rig, tmp_path)
    session = rig.session(nudge_waits_s=NO_WAITS, dev_leases=dev.leases, lease_wake_s=0.05)
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    _end_turn(rig)
    await _until(lambda: rig.runner.sent, "a wake while Forge runs")
    assert "dev_sandbox_read" in rig.runner.sent[0] and dev.forge.alive()
    await session.stop()
    await asyncio.wait_for(running, 1)


async def test_a_cli_lost_while_its_long_dev_lease_s_process_runs_was_waiting_and_keeps_the_lease(rig, tmp_path):
    dev = await _forge_running(rig, tmp_path)
    first = rig.session(nudge_waits_s=NO_WAITS, dev_leases=dev.leases)

    crash = await _crash_after_a_turn(rig, first, 1, LOST)
    await first.stop()
    assert (crash.lost, crash.waiting) == (True, True), "a model waiting on Forge in silence is not hung"
    assert dev.toolbox.released == [] and dev.forge.alive()

    dev.forge.returncode = 0
    second = rig.session(nudge_waits_s=NO_WAITS, dev_leases=dev.leases)
    crash = await _crash_after_a_turn(rig, second, 2, LOST)
    await second.stop()
    assert (crash.lost, crash.waiting) == (True, False), "with Forge ended, a loss is a loss again"


async def test_a_run_that_ends_keeps_a_long_dev_lease_whose_process_runs_and_its_resume_says_so(rig, tmp_path):
    dev = await _forge_running(rig, tmp_path)
    first = rig.session(dev_leases=dev.leases)
    await _crash_after_a_turn(rig, first, 1)
    await first.stop()
    assert dev.toolbox.released == [] and dev.forge.alive(), "a run's end is no end of its branch's long lease"

    session, running = await _next_launch(rig, 2, dev_leases=dev.leases)
    await session.stop()
    await asyncio.wait_for(running, 1)
    message = rig.runner.prompts[1]
    assert "--resume" in rig.runner.specs[1].argv
    assert "dev sandbox kept" in message and "process 1" in message and "dev_sandbox_read" in message
    assert message.count("dev sandbox kept") == 1

    await session.release()
    assert dev.toolbox.released == ["lease-1"] and not dev.forge.alive(), "the branch's stop ends it"
