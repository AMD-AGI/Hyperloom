# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom.orchestrator.specialists.agent_runner import AgentEnded, AgentHandle, AgentStatus, LocalAgentRunner
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, LlmUsage
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, LineageKind, UsageSource
from hyperloom_multibranch.contracts.toolbox import SpecialistGrant
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import intent, request, result
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.specialist.agent_session import (
    IDLE_TURNS,
    WAITED_END,
    AgentCommand,
    AgentSession,
    AgentSetup,
    BranchPlace,
)
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionStart
from hyperloom_multibranch.specialist.worktree import SourceBase

BRANCH, TOKEN_ENV, FIRST_PID = "s1", "HYPERLOOM_SPECIALIST_MCP_TOKEN", 4000
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)
EXITED_WELL = AgentStatus("exited", 0, None)
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
    """Launches nothing: launch ``n`` runs until the test ends it with ``end(n, status, seconds)``; ``sent`` holds
    the text of every message sent to a running CLI, ``beats`` counts the beats its session gave."""

    def __init__(self, clock: _Clock) -> None:
        self.clock, self.specs, self.prompts, self.cancelled, self.collected = clock, [], [], [], []
        self.ends: dict[int, tuple[AgentStatus, float]] = {}
        self.sent: list[str] = []
        self.beats = 0
        self.exits_before_send: dict[int, tuple[AgentStatus, float]] = {}

    def end(self, n: int, status: AgentStatus, seconds: float) -> None:
        self.ends[n] = (status, seconds)

    async def launch(self, spec):
        self.specs.append(spec)
        self.prompts.append(_text((spec.workspace / "prompt.md").read_text()))
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
        ended = self.ends.pop(handle.pid - FIRST_PID, None)
        if ended is None:
            return AgentStatus("running", None, None)
        self.clock.now += ended[1]
        return ended[0]

    async def cancel(self, handle):
        self.cancelled.append(handle.pid - FIRST_PID)
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
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)
    base.build(["vllm"])
    repo = GitExperimentRepo(
        tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"), lambda _layout: installed
    )
    root = await repo.init_root(ROOT)
    await repo.create_branch(BRANCH, root)
    scheduler = LocalNotebook(tmp_path / "notebook", Writer(Role.SCHEDULER, "scheduler"))
    await scheduler.record_lineage(
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root, recipe_digest=ROOT.digest())
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, BRANCH, from_commit=root))
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH))
    await notebook.append_branch_log(BranchLogEntry(BRANCH, BranchLogType.PLAN, "try fp8 KV cache next"))
    clock = _Clock()
    rig = SimpleNamespace(
        root=root,
        clock=clock,
        runner=_Runner(clock),
        tokens=_Tokens(),
        processes=_Processes(),
        worktree=base.worktree(BRANCH, tmp_path / "s1" / "worktree"),
        workspace=tmp_path / "s1" / "session",
        notebook=notebook,
        toolbox_notebook=LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox")),
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
        place = BranchPlace(rig.worktree, notebook, rig.workspace)
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
    await rig.toolbox_notebook.write_request(request("r1", intent(BRANCH, rig.root)))
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
    await rig.toolbox_notebook.write_result(result("r1"))
    await _until(lambda: len(rig.runner.sent) == 1, "the result to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert len(rig.runner.specs) == 1
    assert "this arrived for your branch" in rig.runner.sent[0] and "result r1" in rig.runner.sent[0]


async def test_a_cli_that_exits_as_it_is_woken_ends_its_launch_as_any_exit(rig):
    await rig.toolbox_notebook.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    rig.runner.exits_before_send[1] = (EXITED_WELL, 3600.0)

    _end_turn(rig)
    await rig.toolbox_notebook.write_result(result("r1"))
    await _until(lambda: len(rig.runner.specs) == 2, "a fresh launch after the CLI ended")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert rig.runner.sent == [] and "ended: its model finished its turn" in rig.runner.prompts[1]


async def test_a_profile_of_its_head_that_another_branch_runs_keeps_the_cli_waiting_and_wakes_it(rig):
    shared = request(
        "p1", intent(INTEGRATION_BRANCH, rig.root), RecipeDelta(), kind=TestKind.PROFILE, purpose=Purpose.PROFILE
    )
    await rig.toolbox_notebook.write_request(shared)
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    _end_turn(rig)
    beats = rig.runner.beats
    await _until(lambda: rig.runner.beats > beats + 2, "the turn's end to be waited out")
    assert rig.runner.sent == []
    await rig.toolbox_notebook.write_result(result("p1"))
    await _until(lambda: len(rig.runner.sent) == 1, "the shared profile's result to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "p1" in rig.runner.sent[0] and "profile of your head" in rig.runner.sent[0]


async def test_a_runtime_notice_written_while_the_turn_is_over_reaches_the_cli(rig):
    await rig.toolbox_notebook.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    _end_turn(rig)
    await rig.notebook.append_branch_log(
        BranchLogEntry(BRANCH, BranchLogType.RESULT, f"{NOTICE} you may now have 3 tests running at once")
    )
    await _until(lambda: len(rig.runner.sent) == 1, "the notice to wake the CLI")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert "you may now have 3 tests running at once" in rig.runner.sent[0]


async def test_turns_that_end_with_nothing_running_are_nudged_until_too_many_in_a_row_raise(rig):
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")

    for nudged in range(1, IDLE_TURNS + 1):
        _end_turn(rig)
        await _until(lambda n=nudged: len(rig.runner.sent) == n, f"nudge {nudged}")
    _end_turn(rig)

    with pytest.raises(SessionCrashed, match=f"{IDLE_TURNS} turns in a row") as raised:
        await asyncio.wait_for(running, 1)
    assert not raised.value.lost
    assert all("nothing new for your branch" in text for text in rig.runner.sent)
    assert (rig.runner.cancelled, rig.runner.collected) == ([1], [1])


async def test_each_turn_records_the_tokens_it_added(rig):
    await rig.toolbox_notebook.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session(command=AgentCommand("claude", "claude", model="claude-opus-5-5"))
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the launch")
    init = {"type": "system", "subtype": "init", "session_id": "sess-1"}

    def call(n: int, tokens: int) -> dict:
        usage = {"input_tokens": tokens, "output_tokens": 10}
        return {"type": "assistant", "message": {"id": f"m{n}", "model": "claude-opus-5-5", "usage": usage}}

    async def recorded() -> list:
        return [record.usage.tokens for record in (await rig.notebook.snapshot()).llm_usage]

    _end_turn(rig, init, call(1, 1_000))
    await _until(lambda: rig.runner.beats > 0, "the first turn to end")
    first = await recorded()
    await rig.toolbox_notebook.write_result(result("r1"))
    await _until(lambda: len(rig.runner.sent) == 1, "the result to wake the CLI")
    _end_turn(rig, call(2, 500))
    await _until(lambda: len(rig.runner.sent) == 2, "the second turn to end")
    await session.stop()
    await asyncio.wait_for(running, 1)

    assert first == [TokenCounts(1_000, 10)]
    assert await recorded() == [TokenCounts(1_000, 10), TokenCounts(500, 10)]


async def test_a_launch_ended_early_to_wait_for_its_test_is_no_crash_and_the_next_waits_for_the_result(rig):
    await rig.toolbox_notebook.write_request(request("r1", intent(BRANCH, rig.root)))
    session = rig.session()
    running = asyncio.create_task(session.run())
    await _until(lambda: len(rig.runner.specs) == 1, "the first launch")

    rig.runner.end(1, EXITED_WELL, 30.0)
    await asyncio.sleep(0.2)
    assert len(rig.runner.specs) == 1 and not running.done()
    await rig.toolbox_notebook.write_result(result("r1"))
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

    first, killed = (record.usage for record in (await rig.notebook.snapshot()).llm_usage)
    assert first == LlmUsage(BRANCH, UsageSource.SPECIALIST, first.cost_usd, TokenCounts(1_000, 100), "sess-1")
    assert first.cost_usd == pytest.approx((1_000 * 4 + 100 * 20) / 1e6)
    assert (killed.cost_usd, killed.tokens) == (None, TokenCounts(input=1_000_000))


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
