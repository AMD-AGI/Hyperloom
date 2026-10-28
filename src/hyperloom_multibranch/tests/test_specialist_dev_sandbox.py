# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe
from hyperloom.hosted.mcp_server import InvalidParamsError, ToolError
from hyperloom.measure.compile import PackageBase
from hyperloom.measure.ports import DEV_SANDBOX_PORTS
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.toolbox import BranchLease, QueueState, RequestRejected
from hyperloom_multibranch.specialist.agent_onboarding import CommandResult
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSession, AgentSetup, BranchPlace
from hyperloom_multibranch.specialist.dev_sandbox import (
    CACHE_DIRS,
    DevLeases,
    DevLimits,
    local_dev_sandboxes,
    register_dev_sandbox_tools,
)
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.prompts import role_prompt
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionStart
from hyperloom_multibranch.specialist.worktree import SourceBase
from hyperloom_multibranch.tests.branch_rig import branch_rig

B1, B2 = "s1", "s2"
MINUTE = 60.0
LIMITS = DevLimits(read_poll_s=0.01)
WORKLOAD = "qwen3-8b on sglang, TP 1, ISL 1024, OSL 1024, concurrency 64"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Toolbox:
    """Leases cards one after another; ``queued`` tests wait; ``refusal`` is raised by the next lease."""

    def __init__(self, root: Path) -> None:
        self.root, self.queued, self.refusal = root, 0, None
        self.leased: list[tuple[str, int]] = []
        self.released: list[str] = []

    async def queue_state(self) -> QueueState:
        return QueueState(queued_by_priority={2: self.queued})

    async def lease_branch_sandbox(self, branch_id: str, cards: int, *, wait_sec: float | None = None) -> BranchLease:
        if self.refusal is not None:
            raise self.refusal
        self.leased.append((branch_id, cards))
        lease_id = f"lease-{len(self.leased)}"
        root = self.root / lease_id
        root.mkdir(parents=True)
        return BranchLease(lease_id, branch_id, tuple(range(cards)), root, {"HIP_VISIBLE_DEVICES": "0"}, 7)

    async def release_branch_sandbox(self, lease_id: str) -> None:
        self.released.append(lease_id)


class _Process:
    def __init__(self, command: str) -> None:
        self.command, self.text, self.code = command, "", None

    @property
    def returncode(self) -> int | None:
        return self.code

    async def output(self) -> str:
        return self.text

    async def kill(self) -> None:
        self.code = -15


class _Sandbox:
    def __init__(self, lease: BranchLease) -> None:
        self.lease, self.runs, self.processes, self.stopped = lease, [], [], False

    async def run(self, command, cwd, env, timeout) -> CommandResult:
        self.runs.append((command, cwd, dict(env), timeout))
        return CommandResult(3, "\n".join(f"line {n}" for n in range(500)))

    async def start(self, command, cwd, env) -> _Process:
        process = _Process(command)
        self.processes.append(process)
        return process

    def alive(self) -> bool:
        return any(process.code is None for process in self.processes)

    async def stop(self) -> None:
        self.stopped = True
        for process in self.processes:
            await process.kill()


class _Logs:
    """A branch log and nothing else: the dev sandbox writes no test, claim or judgement."""

    def __init__(self) -> None:
        self.entries = []

    async def append_branch_log(self, entry):
        self.entries.append(entry)


class _Server:
    def __init__(self) -> None:
        self.handlers, self.scopes = {}, {}

    def register(self, name, *, description, schema, handler, scopes):
        self.handlers[name], self.scopes[name] = handler, scopes

    async def call(self, name: str, branch: str = B1, **arguments):
        return await self.handlers[name](arguments, SimpleNamespace(principal=SimpleNamespace(branch=branch)))


@pytest.fixture
def rig(tmp_path):
    clock, toolbox, logs, sandboxes = _Clock(), _Toolbox(tmp_path / "leases"), _Logs(), {}

    def sandbox_for(branch: str, lease: BranchLease) -> _Sandbox:
        sandboxes[branch] = _Sandbox(lease)
        return sandboxes[branch]

    def build(cap: int = 1, limits: DevLimits = LIMITS) -> SimpleNamespace:
        leases = DevLeases(toolbox, sandbox_for, lambda _branch: logs, limits=replace(limits, cap=cap), clock=clock)
        server = _Server()
        register_dev_sandbox_tools(server, leases=leases)
        return SimpleNamespace(leases=leases, server=server)

    return SimpleNamespace(clock=clock, toolbox=toolbox, logs=logs, sandboxes=sandboxes, build=build)


async def _refused(call, code: str) -> str:
    with pytest.raises(ToolError) as refused:
        await call
    assert str(refused.value).startswith(f"{code}:"), str(refused.value)
    return str(refused.value)


async def test_the_tools_are_the_specialists_own(rig):
    server = rig.build().server
    assert set(server.handlers) == {
        "dev_sandbox_lease",
        "dev_sandbox_run",
        "dev_sandbox_start",
        "dev_sandbox_read",
        "dev_sandbox_release",
    }
    assert all(scopes == frozenset({"specialist"}) for scopes in server.scopes.values())


async def test_a_session_without_dev_leases_refuses_every_lease(rig):
    server = rig.build(cap=0).server
    await _refused(server.call("dev_sandbox_lease"), "disabled")
    assert rig.toolbox.leased == []


async def test_a_lease_is_refused_while_tests_wait_for_cards(rig):
    server = rig.build().server
    rig.toolbox.queued = 1
    await _refused(server.call("dev_sandbox_lease"), "tests_waiting")
    assert rig.toolbox.leased == []


async def test_a_lease_is_granted_with_its_cards_ports_and_deadline(rig):
    server = rig.build(cap=2).server
    granted = await server.call("dev_sandbox_lease", cards=1, minutes=90)
    assert rig.toolbox.leased == [(B1, 1)]
    assert granted["cards"] == 1 and granted["expires_in_s"] == 30 * MINUTE
    assert granted["root"] == str(rig.toolbox.root / "lease-1")
    assert tuple(granted["ports"]) == (LIMITS.port, LIMITS.port + LIMITS.ports - 1)
    second = await server.call("dev_sandbox_lease", branch=B2, cards=1, minutes=5)
    assert second["ports"][0] == LIMITS.port + LIMITS.ports and second["expires_in_s"] == 5 * MINUTE
    assert [entry.content.startswith(NOTICE) for entry in rig.logs.entries] == [True, True]


async def test_one_lease_a_branch_and_cap_leases_a_session(rig):
    server = rig.build(cap=1).server
    await server.call("dev_sandbox_lease")
    await _refused(server.call("dev_sandbox_lease"), "lease_exists")
    await _refused(server.call("dev_sandbox_lease", branch=B2), "cap_reached")
    assert rig.toolbox.leased == [(B1, 1)]


async def test_the_toolboxs_refusal_reaches_the_specialist_and_frees_the_slot(rig):
    server = rig.build(cap=1).server
    rig.toolbox.refusal = RequestRejected("too few cards", "lease_leaves_no_measurement")
    await _refused(server.call("dev_sandbox_lease"), "lease_leaves_no_measurement")
    rig.toolbox.refusal = None
    await server.call("dev_sandbox_lease", branch=B2)
    assert rig.toolbox.leased == [(B2, 1)]


async def test_a_lease_asks_the_toolbox_not_to_wait_and_is_refused_when_no_card_is_free(rig):
    server = rig.build().server
    waits = []

    async def busy(branch_id, cards, *, wait_sec=None):
        waits.append(wait_sec)
        raise SandboxUnavailable("no card is free")

    rig.toolbox.lease_branch_sandbox = busy
    await _refused(server.call("dev_sandbox_lease"), "cards_busy")
    assert waits == [0]
    rig.toolbox.lease_branch_sandbox = _Toolbox.lease_branch_sandbox.__get__(rig.toolbox)
    await server.call("dev_sandbox_lease")


def test_the_dev_sandbox_ports_are_the_range_the_measurement_keeps_free():
    assert DevLimits().port == DEV_SANDBOX_PORTS.start


async def test_a_released_branch_cools_down_before_it_leases_again(rig):
    server = rig.build().server
    await server.call("dev_sandbox_lease")
    rig.clock.now += 2 * MINUTE
    released = await server.call("dev_sandbox_release")
    assert released["seconds"] == 2 * MINUTE
    assert rig.toolbox.released == ["lease-1"] and rig.sandboxes[B1].stopped
    rig.clock.now += 9 * MINUTE
    await _refused(server.call("dev_sandbox_lease"), "cooldown")
    rig.clock.now += MINUTE
    await server.call("dev_sandbox_lease")
    assert len(rig.toolbox.leased) == 2


async def test_the_runtime_takes_back_an_expired_lease_and_says_why(rig):
    built = rig.build()
    await built.server.call("dev_sandbox_lease", minutes=10)
    rig.clock.now += 9 * MINUTE
    await built.server.call("dev_sandbox_run", command="true")
    await built.leases.sweep()
    assert rig.toolbox.released == []
    rig.clock.now += MINUTE
    await built.leases.sweep()
    assert rig.toolbox.released == ["lease-1"] and rig.sandboxes[B1].stopped
    assert "expired" in rig.logs.entries[-1].content and rig.logs.entries[-1].content.startswith(NOTICE)
    refused = await _refused(built.server.call("dev_sandbox_run", command="true"), "no_lease")
    assert "expired" in refused


async def test_the_runtime_takes_back_an_idle_lease(rig):
    built = rig.build()
    await built.server.call("dev_sandbox_lease")
    rig.clock.now += LIMITS.idle_s
    await built.leases.sweep()
    assert rig.toolbox.released == ["lease-1"]


async def test_a_lease_running_a_process_is_idle_only_later(rig):
    built = rig.build()
    await built.server.call("dev_sandbox_lease")
    await built.server.call("dev_sandbox_start", command="python -m sglang.launch_server")
    rig.clock.now += LIMITS.idle_s
    await built.leases.sweep()
    assert rig.toolbox.released == []
    rig.clock.now += LIMITS.idle_running_s - LIMITS.idle_s
    await built.leases.sweep()
    assert rig.toolbox.released == ["lease-1"]
    assert rig.sandboxes[B1].processes[0].returncode is not None


async def test_the_runtime_takes_back_a_lease_once_tests_queued_long_enough(rig):
    built = rig.build()
    await built.server.call("dev_sandbox_lease")
    rig.toolbox.queued = 1
    await built.leases.sweep()
    rig.clock.now += 4 * MINUTE
    await built.server.call("dev_sandbox_run", command="true")
    await built.leases.sweep()
    assert rig.toolbox.released == []
    rig.toolbox.queued = 0
    await built.leases.sweep()
    rig.toolbox.queued = 1
    rig.clock.now += MINUTE
    await built.leases.sweep()
    assert rig.toolbox.released == [], "the queue emptied in between, so the wait starts again"
    rig.clock.now += 5 * MINUTE
    await built.leases.sweep()
    assert rig.toolbox.released == ["lease-1"]
    refused = await _refused(built.server.call("dev_sandbox_read", process=1), "no_lease")
    assert "tests" in refused


async def test_the_watch_reclaims_without_a_call(rig):
    built = rig.build(limits=replace(LIMITS, poll_s=0.01))
    await built.server.call("dev_sandbox_lease", minutes=1)
    rig.clock.now += MINUTE
    for _ in range(300):
        if rig.toolbox.released:
            break
        await asyncio.sleep(0.01)
    assert rig.toolbox.released == ["lease-1"]


async def test_run_start_and_read_work_in_the_branchs_own_sandbox(rig):
    server = rig.build(cap=2).server
    await server.call("dev_sandbox_lease")
    ran = await server.call("dev_sandbox_run", command="python bench.py", timeout_s=600, lines=5)
    command, cwd, env, timeout = rig.sandboxes[B1].runs[0]
    assert (command, cwd, timeout) == ("python bench.py", rig.toolbox.root / "lease-1", LIMITS.run_s)
    assert {name: env[name] for name in CACHE_DIRS} == {
        name: str(rig.toolbox.root / "lease-1" / "cache" / sub) for name, sub in CACHE_DIRS.items()
    }
    assert ran == {"returncode": 3, "output": "\n".join(f"line {n}" for n in range(495, 500))}

    started = await server.call("dev_sandbox_start", command="serve", cwd="/abs/dir")
    process = rig.sandboxes[B1].processes[0]
    assert started == {"process": 1} and rig.sandboxes[B1].processes[0].command == "serve"

    async def fired_up():
        await asyncio.sleep(0.05)
        process.text = "loading\nThe server is fired up and ready to roll!\n"

    task = asyncio.create_task(fired_up())
    read = await server.call("dev_sandbox_read", process=1, wait_s=5, until="fired up")
    await task
    assert read["running"] and read["matched"] and read["returncode"] is None
    process.code = 0
    read = await server.call("dev_sandbox_read", process=1, wait_s=5)
    assert not read["running"] and read["returncode"] == 0 and not read["matched"]
    await _refused(server.call("dev_sandbox_run", branch=B2, command="pkill -f sglang"), "no_lease")
    with pytest.raises(InvalidParamsError):
        await server.call("dev_sandbox_read", process=7)
    with pytest.raises(InvalidParamsError):
        await server.call("dev_sandbox_run", command="ls", cwd="relative")


async def test_the_dev_sandbox_writes_only_branch_log_notices(rig):
    server = rig.build().server
    await server.call("dev_sandbox_lease")
    await server.call("dev_sandbox_run", command="true")
    await server.call("dev_sandbox_release")
    assert all(entry.branch == B1 and entry.content.startswith(NOTICE) for entry in rig.logs.entries)
    assert len(rig.logs.entries) == 2


async def test_reclaim_by_the_runtime_releases_and_is_idempotent(rig):
    built = rig.build()
    await built.leases.reclaim(B1, "nothing held")
    await built.server.call("dev_sandbox_lease")
    await built.leases.reclaim(B1, "its specialist's session stopped")
    await built.leases.reclaim(B1, "again")
    assert rig.toolbox.released == ["lease-1"]


def test_the_role_prompt_is_unchanged_without_dev_leases():
    default = role_prompt(WORKLOAD)
    assert role_prompt(WORKLOAD, dev_leases=0) == default
    assert "You have no GPU of your own" in default and "dev_sandbox" not in default


def test_the_role_prompt_teaches_the_dev_sandbox_when_the_session_has_one():
    role = role_prompt(WORKLOAD, dev_leases=1)
    assert "You have no GPU of your own" not in role
    for said in ("dev_sandbox_lease", "never enter the gate", "dev_sandbox_release", "HIP_VISIBLE_DEVICES"):
        assert said in role


# -- the real sandbox against a real worktree ---------------------------------------------------------------------


@pytest.fixture
def _local_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "local-tmp"))
    (tmp_path / "local-tmp").mkdir()


# What a JIT or cache writes when its directory variable is unset: beside the package it compiles for.
CACHE_WRITER = """
import os, vllm
package = os.path.dirname(vllm.__file__)
for name in ("TRITON_CACHE_DIR", "AITER_JIT_DIR", "TORCH_EXTENSIONS_DIR"):
    where = os.environ.get(name) or os.path.join(package, name.lower())
    os.makedirs(os.path.join(where, "build"), exist_ok=True)
    with open(os.path.join(where, "build", "module.cpp"), "w") as out:
        out.write("generated")
print("imported", vllm.__file__)
"""


async def test_a_sandbox_run_leaves_the_worktrees_diff_unchanged(tmp_path, _local_temp):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    installed = PackageBase("vllm", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)
    base.build(["vllm"])
    worktree = base.worktree(B1, tmp_path / "worktrees" / B1)
    (worktree.path / "vllm" / "__init__.py").write_text("CHANGED = True\n")
    before = worktree.diff("vllm")
    script = tmp_path / "writer.py"
    script.write_text(CACHE_WRITER)

    sandboxes = local_dev_sandboxes(lambda branch: worktree, processes=None, limits=LIMITS)
    leases = DevLeases(_Toolbox(tmp_path / "leases"), sandboxes, lambda _branch: _Logs(), limits=replace(LIMITS, cap=1))
    await leases.lease(B1, 1, 5)
    ran = await leases.run(B1, f"python {script}", None, 30)
    await leases.release(B1)

    assert ran.returncode == 0, ran.output
    assert f"imported {worktree.path / 'vllm' / '__init__.py'}" in ran.output
    assert worktree.diff("vllm") == before
    assert (tmp_path / "leases" / "lease-1" / "cache" / "aiter-jit" / "build" / "module.cpp").exists()


# -- the specialist's session gives its lease back ---------------------------------------------------------------

ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)


class _Held:
    def __init__(self) -> None:
        self.reclaimed: list[tuple[str, str]] = []

    async def reclaim(self, branch: str, reason: str) -> None:
        self.reclaimed.append((branch, reason))


class _CrashingRunner:
    async def launch(self, spec):
        raise SessionCrashed("the CLI could not start")


class _Tokens:
    def issue(self, branch, scopes):
        from hyperloom_multibranch.contracts.toolbox import SpecialistGrant

        return SpecialistGrant("http://127.0.0.1:9/mcp", "TOKEN", "token")

    def revoke(self, branch):
        pass


@pytest.fixture
async def session_rig(tmp_path):
    built = await branch_rig(tmp_path, ROOT, B1)
    held = _Held()
    setup = AgentSetup(
        command=AgentCommand("claude", "claude"),
        role="role",
        env={},
        runner=_CrashingRunner(),
        tokens=_Tokens(),
        processes=SimpleNamespace(adopt=lambda *args, **kwargs: None),
        repo=built.repo,
        dev_leases=held,
    )
    place = BranchPlace(built.base.worktree(B1, tmp_path / "wt"), built.blackboard, tmp_path / "session")
    session = AgentSession(SessionStart(B1, built.root, 1, BranchTask("")), setup, place)
    return SimpleNamespace(session=session, held=held)


async def test_a_crashed_session_gives_its_dev_lease_back(session_rig):
    with pytest.raises(SessionCrashed):
        await session_rig.session.run()
    assert [branch for branch, _ in session_rig.held.reclaimed] == [B1]


async def test_a_stopped_and_a_released_session_give_their_dev_lease_back(session_rig):
    await session_rig.session.stop()
    await session_rig.session.release()
    assert [branch for branch, _ in session_rig.held.reclaimed] == [B1, B1]
