# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom.orchestrator.specialists.agent_runner import AgentHandle, AgentStatus, LocalAgentRunner
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, LlmUsage
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, LineageKind, UsageSource
from hyperloom_multibranch.contracts.toolbox import SpecialistGrant
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.specialist.agent_session import (
    AgentCommand,
    AgentSession,
    AgentSetup,
    BranchPlace,
)
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


class _Runner:
    """Launches nothing: launch ``n`` runs until the test ends it with ``end(n, status, seconds)``."""

    def __init__(self, clock: _Clock) -> None:
        self.clock, self.specs, self.prompts, self.cancelled, self.collected = clock, [], [], [], []
        self.ends: dict[int, tuple[AgentStatus, float]] = {}

    def end(self, n: int, status: AgentStatus, seconds: float) -> None:
        self.ends[n] = (status, seconds)

    async def launch(self, spec):
        self.specs.append(spec)
        self.prompts.append((spec.workspace / "prompt.md").read_text())
        return AgentHandle(spec.task_id, FIRST_PID + len(self.specs), self.clock.now)

    async def poll(self, handle):
        ended = self.ends.pop(handle.pid - FIRST_PID, None)
        if ended is None:
            return AgentStatus("running", None, None)
        self.clock.now += ended[1]
        return ended[0]

    async def cancel(self, handle):
        self.cancelled.append(handle.pid - FIRST_PID)

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
        return AgentSession(SessionStart(BRANCH, root, "leads: fp8 KV cache", 2), setup, place)

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


async def test_each_launch_records_its_tokens_at_the_gateway_s_rates_however_it_ended(rig, caplog):
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
    assert (killed.cost_usd, killed.tokens.input) == (pytest.approx(4.0), 1_000_000)
    assert "'claude-unknown-1', which has no price" in caplog.text


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
seen = {
    "prompt": sys.stdin.read(),
    "mcp_config": "--mcp-config" in sys.argv,
    "token": os.environ.get("HYPERLOOM_SPECIALIST_MCP_TOKEN"),
}
(role.parent / "seen.json").write_text(json.dumps(seen))
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
