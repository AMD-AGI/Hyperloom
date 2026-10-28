# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
from types import SimpleNamespace
import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import replace, dataclass, field
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import (
    ComponentPatches,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.toolbox import Backpressure, BranchLease, RequestRejected
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.local.blackboard import LEARNING_MAX_CHARS
from hyperloom_multibranch.blackboard.tests.factories import result
from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec, CommandResult, OnboardingSetup
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest
from hyperloom_multibranch.specialist.autonomous_session import (
    AGENT_DIR,
    MANIFEST_FILE,
    STALL_SECONDS,
    AutonomousSession,
    AutonomousSetup,
    collection_margin,
    without_file,
)
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.specialist.agent_spend import AgentSpend, transcript_dir
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import PreflightFailed, SessionCrashed, SessionFinished, SessionStart
from hyperloom_multibranch.tests.branch_rig import branch_rig
from hyperloom_multibranch.toolbox.kernel_artifacts import KernelVariables

BRANCH = "a1"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)

#: An agent that, every ``gap`` seconds, writes each of ``keeps`` to its workspace's config.py and to the config it
#: runs with, commits it when ``commits`` is set, as Arbor does, and records and prints the keep; it ends with ``exit``
#: after ``tail`` more seconds of silence. With ``table`` it also writes that kernel table, for the keep, to its workspace's tuned.csv.
#: Before each keep it appends that keep's entries of ``ledger`` to its ledger, run/ledger.jsonl, as Arbor does.
AGENT = """
import json, pathlib, subprocess, sys, time
config = json.loads(pathlib.Path("campaign.json").read_text())
workspace = pathlib.Path(config["workspace"])
state = pathlib.Path("run/state.json")
state.parent.mkdir(exist_ok=True)
state.write_text(json.dumps({"keeps": [], "note": ""}))
for step, seqs in enumerate(config["keeps"], start=1):
    time.sleep(config["gap"])
    (workspace / "vllm" / "config.py").write_text(f"max_num_seqs = {seqs}\\n")
    (workspace / "agent-config.json").write_text(json.dumps({"args": config["args"].format(seqs=seqs), "envs": config["envs"]}))
    if config["table"]:
        (workspace / "tuned.csv").write_text(config["table"].format(seqs=seqs))
    if config["commits"]:
        subprocess.run("git add -A && git -c user.name=a -c user.email=a@a commit -qm k", shell=True, cwd=workspace)
    with pathlib.Path("run/ledger.jsonl").open("a") as ledger:
        for entry in config["ledger"][step - 1] if step <= len(config["ledger"]) else []:
            ledger.write(json.dumps(entry) + "\\n")
    state.write_text(json.dumps({"keeps": [{"decision": "KEEP"}] * step, "note": f"seqs {seqs}"}))
    print(f"kept {seqs}", flush=True)
end = time.time() + config["tail"]
while time.time() < end:
    time.sleep(0.1)
    if config.get("busy"):
        pathlib.Path("run/progress").write_text(str(time.time()))
sys.exit(config["exit"])
"""
KEEPS = {"path": "run/state.json", "field": "keeps", "where": {"decision": "KEEP"}}
LEDGER = {
    "path": "run/ledger.jsonl",
    "id": "id",
    "kind": "type",
    "text": "claim",
    "kinds": {"FACT": "insight", "DEAD_END": "pitfall"},
}
CONFIG_SOURCE = {"path": "agent-config.json", "server_args": "args", "envs": "envs"}


@dataclass(frozen=True)
class Agent:
    keeps: tuple[int, ...] = (256, 512)
    gap: float = 0.3
    tail: float = 0.3
    exit: int = 0
    commits: bool = False
    args: str = "--max-num-seqs {seqs}"
    envs: dict[str, str] = field(default_factory=dict)
    keep: object = field(default_factory=lambda: KEEPS)
    config_source: object = None
    resume: str = ""
    table: str = ""
    ledger: tuple[tuple[dict[str, str], ...], ...] = ()
    busy: bool = False


async def _until(condition, what: str, seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class LocalProcess:
    def __init__(self, process: asyncio.subprocess.Process, log: Path) -> None:
        self._process, self._log = process, log

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def output(self) -> str:
        return self._log.read_text(encoding="utf-8", errors="replace")

    async def kill(self) -> None:
        if self._process.returncode is None:
            os.killpg(self._process.pid, signal.SIGKILL)
            await self._process.wait()


class LocalSandbox:
    """Runs commands as local processes, each in a process group of its own."""

    def __init__(self, logs: Path) -> None:
        self._logs, self.started, self.envs, self.leases, self.prompts = logs, [], [], [], []
        self.usage = _Usage()
        self.watch_said: list[str] = []

    def watched(self) -> str:
        return self.watch_said.pop(0) if self.watch_said else ""

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult:
        done = subprocess.run(command, shell=True, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True)
        return CommandResult(done.returncode, done.stdout + done.stderr)

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> LocalProcess:
        self._logs.mkdir(parents=True, exist_ok=True)
        log = self._logs / f"agent-{len(self.started)}.log"
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env={**os.environ, **env},
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started.append(command)
        self.envs.append(dict(env))
        return LocalProcess(process, log)


class Toolbox:
    """Accepts each request and writes it to the blackboard, so it is in flight until the test settles it; ``refusals``
    are raised first, one a submit."""

    def __init__(self, blackboard: LocalBlackboard) -> None:
        self._blackboard, self.requests, self.refusals = blackboard, [], []
        self.refused_envs: set[str] = set()
        self.leases: list[tuple[str, int]] = []
        self.released: list[str] = []
        self.cards_left = 1
        self.tables: dict[str, bytes] = {}
        self._variables = KernelVariables()

    async def put_kernel_artifact(self, kind: KernelKind, target: str | None, env: str, data: bytes) -> KernelArtifact:
        artifact = self._variables.put(kind, target, env, data)
        self.tables[artifact.sha256] = data
        return artifact

    async def lease_branch_sandbox(self, branch_id: str, cards: int) -> BranchLease:
        if self.cards_left < cards:
            raise RequestRejected("leasing would leave no card to measure on", "lease_leaves_no_measurement")
        self.leases.append((branch_id, cards))
        root = self._blackboard.root.parent / "lease"
        return BranchLease(f"L{len(self.leases)}", branch_id, (3,), root, {"HIP_VISIBLE_DEVICES": "3"}, 1)

    async def release_branch_sandbox(self, lease_id: str) -> None:
        self.released.append(lease_id)

    async def check_test(self, request: TestRequest) -> None:
        if self.refused_envs & set(request.delta.config.envs):
            names = ", ".join(sorted(self.refused_envs))
            reason = f"the environment does not compile: the renderer dropped or changed launch variables {names}"
            raise RequestRejected(reason, "delta_not_applicable")

    async def submit_test(self, request: TestRequest) -> str:
        if self.refusals:
            raise self.refusals.pop(0)
        await self.check_test(request)
        accepted = TestRequest(**{**request.__dict__, "request_id": f"r{len(self.requests) + 1}"})
        self.requests.append(accepted)
        await self._blackboard.write_request(accepted)
        return accepted.request_id or ""


def _agent_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "agent-repo"
    repo.mkdir()
    (repo / "agent.py").write_text(AGENT, encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    return repo


def _manifest(workspace: str, agent: Agent) -> str:
    plan = {"keeps": list(agent.keeps), "gap": agent.gap, "tail": agent.tail, "exit": agent.exit, "busy": agent.busy}
    return json.dumps(
        {
            "source": "agent",
            "commit": "c",
            "install": [],
            "config_file": "campaign.json",
            "config": {
                "workspace": workspace,
                "commits": agent.commits,
                "args": agent.args,
                "envs": agent.envs,
                "table": agent.table,
                "ledger": [list(entries) for entries in agent.ledger],
                **plan,
            },
            "launch": "python3 agent.py",
            "resume": agent.resume,
            "keep": agent.keep,
            "config_source": agent.config_source,
            "summary_field": "note",
            "needs": [],
            "ledger": LEDGER if agent.ledger else None,
        }
    )


class _Usage:
    def __init__(self) -> None:
        self.recorded: list = []

    async def record_llm_usage(self, usage) -> None:
        self.recorded.append(usage)


@dataclass
class Rig:
    session: AutonomousSession
    sandbox: LocalSandbox
    toolbox: Toolbox
    toolbox_blackboard: LocalBlackboard
    repo: object
    blackboard: LocalBlackboard
    blobs: LocalBlobStore
    restart: object

    async def notices(self) -> list[str]:
        return [record.entry.content for record in await self.blackboard.branch_log(BRANCH)]

    def patch(self, request: TestRequest) -> str:
        [change] = request.delta.components
        return self.blobs.get(change.add_patches[0].sha256).decode()


async def _rig(
    tmp_path: Path,
    agent: Agent = Agent(),
    *,
    poll: float = 0.05,
    clock=time.time,
    start: Recipe | None = None,
    deadline: float = 0.0,
    stall: float = STALL_SECONDS,
    spend: type[AgentSpend] = AgentSpend,
) -> Rig:
    """A session of the agent on :data:`BRANCH`, opened at the root, or at ``start`` committed onto it the way PRELUDE
    replays a Recipe KB champion."""

    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    start_commit = rig.root if start is None else await rig.repo.commit(BRANCH, rig.root, start, "champion")
    toolbox_blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    worktree = rig.base.worktree(BRANCH, tmp_path / BRANCH / "agent-workspace")
    sandbox, toolbox = LocalSandbox(tmp_path / "logs"), Toolbox(toolbox_blackboard)

    async def onboard(checkout: Path, prompt: str) -> str:
        sandbox.prompts.append(prompt)
        return _manifest(str(worktree.path), agent)

    def onboarding(agent_sandbox) -> OnboardingSetup:
        return OnboardingSetup(agent_sandbox, onboard, start_seconds=5.0, poll_seconds=0.05)

    def sandbox_for(lease: BranchLease) -> LocalSandbox:
        sandbox.leases.append(lease)
        return sandbox

    spec = AgentSpec("/models/m", "vllm", 1, "fp8", "mi355x", {}, "output_throughput", "", {}, "", (), 60.0, 31100)

    def spend_of(roots: tuple[Path, ...], ledger: Path) -> AgentSpend:
        return spend(tmp_path / "claude", roots, ledger, BRANCH, "claude-sonnet-5-5", sandbox.usage)

    setup = AutonomousSetup(
        onboarding,
        sandbox_for,
        spec,
        toolbox,
        rig.repo,
        "vllm",
        spend_of,
        poll_seconds=poll,
        settle_seconds=0.3,
        cutoff_seconds=0.5,
        stall_seconds=stall,
        clock=clock,
    )
    place = BranchPlace(worktree, rig.blackboard, tmp_path / BRANCH / "session")
    start = SessionStart(BRANCH, start_commit, 1, BranchTask(agent=str(_agent_repo(tmp_path)), deadline=deadline))

    def restart() -> AutonomousSession:
        return AutonomousSession(start, setup, place)

    blobs = LocalBlobStore(tmp_path / "blobs")
    return Rig(restart(), sandbox, toolbox, toolbox_blackboard, rig.repo, rig.blackboard, blobs, restart)


@pytest.mark.parametrize("commits", [False, True], ids=["edits", "commits"])
async def test_an_agent_s_keeps_are_imported_one_at_a_time_against_the_branch_head(
    tmp_path: Path, commits: bool
) -> None:
    rig = await _rig(tmp_path, Agent(commits=commits))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await asyncio.sleep(1.0)
    assert len(rig.toolbox.requests) == 1
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    first, last = rig.toolbox.requests
    assert (first.branch_id, first.parent_commit, first.intent_fingerprint) == (
        BRANCH, await rig.repo.head(BRANCH), None,
    )  # fmt: skip
    assert "+max_num_seqs = 256" in rig.patch(first)
    assert "+max_num_seqs = 512" in rig.patch(last) and "-max_num_seqs = 128" in rig.patch(last)
    notices = await rig.notices()
    assert any(note.startswith(f"{NOTICE} keep 2") and "seqs 512" in note for note in notices)
    assert notices[-1] == f"{NOTICE} the agent finished its run"


async def test_a_keep_after_one_the_branch_committed_imports_only_its_own_change_on_top(tmp_path: Path) -> None:
    rig = await _rig(tmp_path)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    first = rig.toolbox.requests[0]
    recipe = await rig.repo.read_recipe(first.parent_commit)
    committed = await rig.repo.commit(BRANCH, first.parent_commit, apply(recipe, first.delta), "keep 1")
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    last = rig.toolbox.requests[1]
    assert last.parent_commit == committed
    assert "-max_num_seqs = 256" in rig.patch(last) and "+max_num_seqs = 512" in rig.patch(last)
    assert "max_num_seqs = 128" not in rig.patch(last)
    [change] = last.delta.components
    assert change.remove_patches == ()


async def test_an_import_waits_while_the_last_one_is_measured_or_judged(tmp_path: Path) -> None:
    later = [0.0]
    rig = await _rig(tmp_path, Agent(tail=30.0), clock=lambda: time.time() + later[0])

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await asyncio.sleep(0.6)
    await rig.toolbox_blackboard.write_result(result("r1"))
    await asyncio.sleep(0.2)
    assert len(rig.toolbox.requests) == 1
    later[0] = 1.0
    await _until(lambda: len(rig.toolbox.requests) == 2, "the import once the first one is judged")

    await rig.session.stop()
    await asyncio.wait_for(running, 10)


async def test_an_agent_that_exits_non_zero_is_a_crash_with_its_last_output(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(), exit=3))

    await rig.session.preflight()

    with pytest.raises(SessionCrashed, match="exited 3"):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()


async def test_an_agent_that_goes_silent_is_a_lost_session_once_it_wrote_nothing_for_the_stall_bound(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256, 512, 768), gap=0.5, tail=30.0), stall=1.0)
    await rig.session.preflight()
    started = time.monotonic()

    with pytest.raises(SessionCrashed, match="no output") as crashed:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert crashed.value.lost
    assert time.monotonic() - started > 2.0


async def test_an_agent_silent_on_its_console_while_it_writes_its_workspace_is_not_lost(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), gap=0.1, tail=30.0, busy=True), stall=1.0)
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())

    await asyncio.sleep(3.0)
    alive = not running.done()
    await rig.session.stop()
    await asyncio.gather(running, return_exceptions=True)

    assert alive, "the run ended while the agent kept writing its workspace"


class _HungSpend(AgentSpend):
    async def record(self) -> None:
        await asyncio.Event().wait()


async def test_a_run_stuck_in_one_of_its_steps_is_a_lost_session_while_the_agent_still_writes(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=tuple(range(1, 30)), gap=0.1), stall=0.5, spend=_HungSpend)
    await rig.session.preflight()

    with pytest.raises(SessionCrashed, match="polled") as crashed:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert crashed.value.lost


async def test_a_keep_waiting_at_a_crash_is_imported_by_the_resumed_run_with_the_docker_shim(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(exit=3, resume="sleep 30"))

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    assert len(rig.toolbox.requests) == 1
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the waiting keep")

    assert rig.sandbox.started[-1] == "sleep 30"
    assert rig.sandbox.envs[-1]["PATH"].startswith(str(tmp_path / BRANCH / "session" / AGENT_DIR / "bin"))
    assert "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[-1])
    await resumed.stop()
    await asyncio.wait_for(running, 10)


async def test_a_keep_file_with_no_counter_is_one_keep_at_each_change(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), keep={"path": "run/state.json"}, tail=30.0))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")

    assert "+max_num_seqs = 256" in rig.patch(rig.toolbox.requests[0])
    await rig.session.stop()
    await asyncio.wait_for(running, 10)


@pytest.mark.parametrize(
    ("args", "source", "why"),
    [
        ("--max-num-seqs {seqs}", {**CONFIG_SOURCE, "server_args": "envs"}, "cannot be read"),
        ("--max-num-seqs {seqs} /models/m", CONFIG_SOURCE, "is no recipe change"),
    ],
    ids=["unreadable", "positional"],
)
async def test_a_keep_that_cannot_be_imported_is_noted_once_and_the_run_goes_on(
    tmp_path: Path, args: str, source: dict[str, str], why: str
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), args=args, config_source=source, tail=0.6))

    await rig.session.preflight()
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(rig.session.run(), 10)

    assert not rig.toolbox.requests
    assert len([note for note in await rig.notices() if f"keep 1 {why}" in note]) == 1


async def test_a_keep_made_just_before_the_agent_exits_is_imported(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), gap=0.2, tail=0.0), poll=1.0)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the last keep", seconds=5.0)
    await rig.toolbox_blackboard.write_result(result("r1"))

    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)


def test_only_the_file_the_agent_keeps_its_config_in_leaves_the_source_patch() -> None:
    manifest = AgentManifest.read({**json.loads(_manifest("/ws", Agent())), "config_source": CONFIG_SOURCE})
    sections = [
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y\n"
        for path in ("agent-config.json", "vllm/agent-config.json.py")
    ]

    assert without_file("".join(sections), manifest) == sections[1]


async def test_an_agent_runs_on_a_lease_taken_once_for_its_branch_and_given_back_once(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(exit=3, resume="sleep 30"))

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await _until(lambda: len(rig.sandbox.started) == 2, "the resumed agent")
    await resumed.stop()
    await asyncio.wait_for(running, 10)
    await resumed.release()
    await resumed.release()

    assert rig.toolbox.leases == [(BRANCH, 1)]
    assert [lease.gpu_ids for lease in rig.sandbox.leases] == [(3,), (3,)]
    assert rig.toolbox.released == ["L1", "L1"]


async def test_an_agent_opened_on_a_recipe_kb_champion_starts_from_its_launch(tmp_path: Path) -> None:
    champion = replace(
        ROOT, launch=LaunchConfig("--max-num-seqs 256 --kv-cache-dtype fp8_e4m3", {"SGLANG_USE_AITER": "1"})
    )
    rig = await _rig(tmp_path, start=champion)

    await rig.session.preflight()
    await rig.session.stop()

    [prompt] = rig.sandbox.prompts
    assert '"server_args": "--max-num-seqs 256 --kv-cache-dtype fp8_e4m3"' in prompt
    assert '"SGLANG_USE_AITER": "1"' in prompt


def _spent(transcript: Path, message_id: str, output: int) -> None:
    usage = {"input_tokens": 3, "output_tokens": output}
    message = {"id": message_id, "model": "claude-opus-5-5", "usage": usage, "content": []}
    with transcript.open("a") as handle:
        handle.write(json.dumps({"type": "assistant", "sessionId": "s-1", "message": message}) + "\n")


async def test_what_the_agent_s_own_cli_spends_is_recorded_as_the_branch_s_once_a_minute_and_at_the_end(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(tail=30.0))
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    transcript = transcript_dir(tmp_path / "claude", tmp_path / BRANCH / "session" / "agent" / "src") / "s-1.jsonl"
    transcript.parent.mkdir(parents=True)

    _spent(transcript, "m1", 50)
    await _until(lambda: rig.sandbox.usage.recorded, "the agent's spend")
    _spent(transcript, "m2", 7)
    await asyncio.sleep(0.5)
    assert len(rig.sandbox.usage.recorded) == 1
    await rig.session.stop()
    await asyncio.wait_for(running, 10)

    assert [(used.branch, used.launch, used.tokens) for used in rig.sandbox.usage.recorded] == [
        (BRANCH, "s-1", TokenCounts(3, 50)),
        (BRANCH, "s-1", TokenCounts(3, 7)),
    ]


async def test_an_agent_with_no_keep_signal_has_its_workspace_collected_once_before_the_explore_end(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keep=None, tail=30.0), deadline=time.time() + 1.5)
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())

    await _until(lambda: rig.toolbox.requests, "the cut-off collection")
    await asyncio.sleep(1.5)
    await rig.session.stop()
    await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    assert "+max_num_seqs = 512" in rig.patch(request)
    assert any("EXPLORE ends" in note for note in await rig.notices())


async def test_a_passed_preflight_records_the_agent_s_manifest_in_the_blackboard(tmp_path: Path) -> None:
    rig = await _rig(tmp_path)

    await rig.session.preflight()
    await rig.session.release()

    stored = json.loads((tmp_path / BRANCH / "session" / MANIFEST_FILE).read_text(encoding="utf-8"))
    recorded = (await rig.blackboard.snapshot()).agent_manifests[BRANCH].manifest
    assert recorded["source"] == stored["source"] and recorded["commit"] == stored["commit"]


async def test_an_agent_whose_lease_would_leave_no_card_to_measure_fails_its_preflight(tmp_path: Path) -> None:
    rig = await _rig(tmp_path)
    rig.toolbox.cards_left = 0

    with pytest.raises(PreflightFailed, match="lease_leaves_no_measurement"):
        await rig.session.preflight()
    await rig.session.release()

    assert not rig.sandbox.started and rig.toolbox.released == []


async def test_a_keep_the_toolbox_refuses_is_noted_and_one_it_has_no_room_for_waits(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(tail=1.0))
    rig.toolbox.refusals = [
        RequestRejected("online tuning is not measured", code="online_tuning_unsupported"),
        Backpressure(BRANCH, 1, 1),
    ]

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the keep after the refused one")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    assert len([note for note in await rig.notices() if "keep 1 was refused: online tuning" in note]) == 1
    assert "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[0])


#: A resumed agent that starts its keep file over, as an agent relaunched from scratch does, then keeps once more.
RESTARTS_ITS_COUNT = (
    "python3 -c \"import json, pathlib, time; state = pathlib.Path('run/state.json'); "
    "state.write_text(json.dumps({'keeps': []})); time.sleep(0.3); "
    "state.write_text(json.dumps({'keeps': [{'decision': 'KEEP'}], 'note': 'after the restart'})); time.sleep(30)\""
)


async def test_a_keep_after_an_agent_restarted_its_count_is_a_new_keep(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), exit=3, resume=RESTARTS_ITS_COUNT))

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await asyncio.sleep(1.0)
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the keep after the restart")

    assert any(note.startswith(f"{NOTICE} keep 2 imported") for note in await rig.notices())
    await resumed.stop()
    await asyncio.wait_for(running, 10)


async def test_a_keep_file_whose_counter_never_resolves_is_noted_once(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keep={"path": "run/state.json", "field": "lessons"}, tail=0.6))

    await rig.session.preflight()
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(rig.session.run(), 10)

    unread = [note for note in await rig.notices() if "run/state.json" in note and "lessons" in note]
    assert len(unread) == 1 and not rig.toolbox.requests


async def test_a_refused_variable_is_left_out_of_later_keeps_while_their_other_launch_changes_go_in(
    tmp_path: Path,
) -> None:
    agent = Agent(envs={"SGLANG_USE_AITER": "1"}, config_source=CONFIG_SOURCE, tail=30.0)
    rig = await _rig(tmp_path, agent)
    rig.toolbox.refused_envs = {"SGLANG_USE_AITER"}

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first keep")
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second keep")

    configs = [request.delta.config for request in rig.toolbox.requests]
    assert [(config.server_args, dict(config.envs)) for config in configs] == [
        ("--max-num-seqs 256", {}),
        ("--max-num-seqs 512", {}),
    ]
    await rig.session.stop()
    await asyncio.wait_for(running, 10)


async def test_a_keep_whose_launch_the_toolbox_refuses_is_imported_without_it_and_so_are_later_keeps(
    tmp_path: Path,
) -> None:
    # The measurement's launcher sets SGLANG_USE_AITER already; an agent that sets it too keeps it in every keep after.
    agent = Agent(args="--max-num-seqs 128", envs={"SGLANG_USE_AITER": "1"}, config_source=CONFIG_SOURCE, tail=30.0)
    rig = await _rig(tmp_path, agent)
    rig.toolbox.refused_envs = {"SGLANG_USE_AITER"}

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first keep without its launch")
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second keep")

    assert all(not request.delta.config.envs for request in rig.toolbox.requests)
    assert "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[1])
    refused = [note for note in await rig.notices() if "SGLANG_USE_AITER" in note]
    assert len(refused) == 1 and "imported without SGLANG_USE_AITER" in refused[0]
    await rig.session.stop()
    await asyncio.wait_for(running, 10)


@pytest.mark.parametrize("bound", ["gemm_a8w8", None], ids=["bound", "unbound"])
async def test_a_kernel_table_the_agent_keeps_in_its_workspace_is_imported_as_the_session_s_artifact(
    tmp_path: Path, bound: str | None
) -> None:
    env = "AITER_CONFIG_GEMM_A8W8"
    table = str(tmp_path / BRANCH / "agent-workspace" / "tuned.csv")
    agent = Agent(keeps=(256,), envs={env: table}, table="M,N,K\n{seqs},5120,5120\n", config_source=CONFIG_SOURCE)
    rig = await _rig(tmp_path, agent)
    if bound:
        await rig.toolbox.put_kernel_artifact(KernelKind.GEMM, bound, env, b"M,N,K\n1,1,1\n")

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    Path(table).write_text("rewritten after the keep\n", encoding="utf-8")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    [artifact] = request.delta.add_kernels
    assert artifact.target == (bound or env.lower())
    assert rig.toolbox.tables[artifact.sha256] == b"M,N,K\n256,5120,5120\n"
    assert env not in request.delta.config.envs


def _entry(entry_id: str, kind: str, claim: str) -> dict[str, str]:
    return {"id": entry_id, "type": kind, "claim": claim, "evidence": "eval/x.json"}


async def test_the_agent_s_ledger_facts_and_dead_ends_become_the_branch_s_learnings_at_each_keep(
    tmp_path: Path,
) -> None:
    first = (
        _entry("gemm-untuned", "FACT", "dense fp8 GEMMs miss the tuned CSV and cost 64% of the decode step"),
        _entry("triton-slower", "DEAD_END", "aiter triton blockscale GEMM is slower than default CK at M=64"),
        _entry("guess", "ASSUMPTION", "the allreduce is not on the critical path at TP1"),
    )
    second = (
        _entry("gemm-untuned", "FACT", "dense fp8 GEMMs miss the tuned CSV; the asm table cuts them to 4.35 ms"),
        _entry("long", "FACT", "x" * 2000),
    )
    rig = await _rig(tmp_path, Agent(ledger=(first, second)))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    learnt = await rig.blackboard.learnings(branch=BRANCH)
    assert [(record.learning.kind.value, record.learning.content) for record in learnt] == [
        ("insight", f"FACT gemm-untuned: {first[0]['claim']}"),
        ("pitfall", f"DEAD_END triton-slower: {first[1]['claim']}"),
    ]
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    revised, long = (await rig.blackboard.learnings(branch=BRANCH))[2:]
    assert revised.learning.content == f"FACT gemm-untuned: {second[0]['claim']}"
    assert revised.learning.supersedes == learnt[0].entry_id
    assert long.learning.content.startswith("FACT long: xxx") and len(long.learning.content) == LEARNING_MAX_CHARS
    assert len(await rig.blackboard.learnings(branch=BRANCH)) == 4


KEPT = "the decode GEMMs miss the tuned CSV"
AFTER = "the fused qk-norm kernel is slower at batch 8"


async def test_ledger_findings_written_after_the_last_keep_are_learnt_when_the_branch_is_stopped(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), tail=30.0, ledger=((_entry("a", "FACT", KEPT),),)))
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    ledger = tmp_path / BRANCH / "session" / AGENT_DIR / "src" / "run" / "ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as lines:
        lines.write(json.dumps(_entry("b", "DEAD_END", AFTER)) + "\n")

    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    await rig.session.stop()

    learnt = [record.learning.content for record in await rig.blackboard.learnings(branch=BRANCH)]
    assert learnt == [f"FACT a: {KEPT}", f"DEAD_END b: {AFTER}"]


async def test_a_sandbox_watch_started_again_is_noted_in_the_branch_log(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0))
    await rig.session.preflight()
    rig.sandbox.watch_said.append("the sandbox's watch had ended (exit 1) and was started again")
    running = asyncio.create_task(rig.session.run())

    await _until(lambda: not rig.sandbox.watch_said, "the watch check")
    await rig.session.stop()
    await asyncio.wait_for(running, 10)

    assert f"{NOTICE} the sandbox's watch had ended (exit 1) and was started again" in await rig.notices()


async def test_a_finding_learnt_as_the_agent_finished_is_not_learnt_again_when_the_branch_is_stopped(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), tail=1.0, ledger=((_entry("a", "FACT", KEPT),),)))
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    ledger = tmp_path / BRANCH / "session" / AGENT_DIR / "src" / "run" / "ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as lines:
        lines.write(json.dumps(_entry("b", "DEAD_END", AFTER)) + "\n")
    await rig.toolbox_blackboard.write_result(result("r1"))

    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)
    await rig.session.stop()

    learnt = [record.learning.content for record in await rig.blackboard.learnings(branch=BRANCH)]
    assert learnt == [f"FACT a: {KEPT}", f"DEAD_END b: {AFTER}"]


def _record(*, measured: tuple[float, float] | None = None) -> SimpleNamespace:
    attempts = () if measured is None else (SimpleNamespace(started_at=measured[0], finished_at=measured[1]),)
    result = None if measured is None else SimpleNamespace(attempts=attempts)
    return SimpleNamespace(result=result, in_flight=measured is None)


def test_the_collection_margin_covers_the_tests_queued_and_the_import_s_test_and_confirm_at_the_longest_test() -> None:
    measured = [_record(measured=(0.0, 500.0)), _record(measured=(1000.0, 1540.0))]

    assert collection_margin([], 2400.0) == 2400.0
    assert collection_margin(measured, 600.0) == 2 * 540.0
    assert collection_margin([*measured, _record(), _record(), _record()], 600.0) == 5 * 540.0
    assert collection_margin([*measured, _record()], 2400.0) == 2400.0


async def test_the_agent_runs_until_its_imports_can_still_be_judged_against_the_gate_s_baseline(
    tmp_path: Path,
) -> None:
    now = time.time()
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0), clock=lambda: now, deadline=now + 3600.0)
    gate = LocalBlackboard(tmp_path / "blackboard", Writer(Role.GATE, "gate"))
    rates = {"output_throughput": 0.2}
    root = await rig.repo.head(INTEGRATION_BRANCH)
    await gate.write_baseline(BaselineRecord(root, {"output_throughput": 6290.7}, rates, rates, ("b1",)))

    await rig.session.preflight()

    prompt = rig.sandbox.prompts[0]
    spec = json.loads(prompt[prompt.index("{\n") : prompt.index("\n}\n") + 2])
    assert spec["seconds"] == pytest.approx(3600.0 - 0.5)
    assert spec["baseline"] == 6290.7
    assert spec["knowledge"] == str(tmp_path / BRANCH / "session" / AGENT_DIR / "mb-integration.md")
    await rig.session.stop()


async def test_the_agent_s_integration_notes_follow_the_integration_head(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0))
    notes = tmp_path / BRANCH / "session" / AGENT_DIR / "mb-integration.md"

    root = await rig.repo.head(INTEGRATION_BRANCH)
    await rig.session.preflight()
    assert root in notes.read_text() and "--max-num-seqs 512" not in notes.read_text()
    running = asyncio.create_task(rig.session.run())
    await rig.repo.create_branch("b2", root)
    other = await rig.repo.commit("b2", root, replace(ROOT, launch=LaunchConfig("--max-num-seqs 512")), "b2")
    head = await rig.repo.commit_integration(root, RecipeDelta(config=ConfigDelta("--max-num-seqs 512")), (other,))
    await _until(lambda: head in notes.read_text(), "the notes of the new integration head")

    assert "--max-num-seqs 512" in notes.read_text()
    await rig.session.stop()
    await asyncio.wait_for(running, 10)
