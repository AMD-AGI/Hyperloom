# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.toolbox import Backpressure, RequestRejected
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import result
from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec, CommandResult, OnboardingSetup
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest
from hyperloom_multibranch.specialist.autonomous_session import (
    AGENT_DIR,
    AutonomousSession,
    AutonomousSetup,
    without_file,
)
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import PreflightFailed, SessionCrashed, SessionFinished, SessionStart
from hyperloom_multibranch.tests.branch_rig import branch_rig

BRANCH = "a1"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)

#: An agent that, every ``gap`` seconds, writes each of ``keeps`` to its workspace's config.py and to the config it
#: runs with, commits it when ``commits`` is set, as Arbor does, and records the keep; it ends with ``exit`` after
#: ``tail`` more seconds.
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
    (workspace / "agent-config.json").write_text(json.dumps({"args": config["args"].format(seqs=seqs), "envs": {}}))
    if config["commits"]:
        subprocess.run("git add -A && git -c user.name=a -c user.email=a@a commit -qm k", shell=True, cwd=workspace)
    state.write_text(json.dumps({"keeps": [{"decision": "KEEP"}] * step, "note": f"seqs {seqs}"}))
time.sleep(config["tail"])
sys.exit(config["exit"])
"""
KEEPS = {"path": "run/state.json", "field": "keeps", "where": {"decision": "KEEP"}}
CONFIG_SOURCE = {"path": "agent-config.json", "server_args": "args", "envs": "envs"}


@dataclass(frozen=True)
class Agent:
    keeps: tuple[int, ...] = (256, 512)
    gap: float = 0.3
    tail: float = 0.3
    exit: int = 0
    commits: bool = False
    args: str = "--max-num-seqs {seqs}"
    keep: object = field(default_factory=lambda: KEEPS)
    config_source: object = None
    resume: str = ""


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
        self._logs, self.started, self.envs = logs, [], []

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
    """Accepts each request and writes it to the notebook, so it is in flight until the test settles it; ``refusals``
    are raised first, one a submit."""

    def __init__(self, notebook: LocalNotebook) -> None:
        self._notebook, self.requests, self.refusals = notebook, [], []

    async def submit_test(self, request: TestRequest) -> str:
        if self.refusals:
            raise self.refusals.pop(0)
        accepted = TestRequest(**{**request.__dict__, "request_id": f"r{len(self.requests) + 1}"})
        self.requests.append(accepted)
        await self._notebook.write_request(accepted)
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
    plan = {"keeps": list(agent.keeps), "gap": agent.gap, "tail": agent.tail, "exit": agent.exit}
    return json.dumps(
        {
            "source": "agent",
            "commit": "c",
            "install": [],
            "config_file": "campaign.json",
            "config": {"workspace": workspace, "commits": agent.commits, "args": agent.args, **plan},
            "launch": "python3 agent.py",
            "resume": agent.resume,
            "keep": agent.keep,
            "config_source": agent.config_source,
            "summary_field": "note",
            "needs": [],
        }
    )


@dataclass
class Rig:
    session: AutonomousSession
    sandbox: LocalSandbox
    toolbox: Toolbox
    toolbox_notebook: LocalNotebook
    repo: object
    notebook: LocalNotebook
    blobs: LocalBlobStore
    restart: object

    async def notices(self) -> list[str]:
        return [record.entry.content for record in await self.notebook.branch_log(BRANCH)]

    def patch(self, request: TestRequest) -> str:
        [change] = request.delta.components
        return self.blobs.get(change.add_patches[0].sha256).decode()


async def _rig(
    tmp_path: Path, agent: Agent = Agent(), *, poll: float = 0.05, clock=time.time, cards: tuple[int, ...] = (0,)
) -> Rig:
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    toolbox_notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    worktree = rig.base.worktree(BRANCH, tmp_path / BRANCH / "agent-workspace")
    sandbox, toolbox = LocalSandbox(tmp_path / "logs"), Toolbox(toolbox_notebook)

    async def onboard(checkout: Path, prompt: str) -> str:
        return _manifest(str(worktree.path), agent)

    onboarding = OnboardingSetup(sandbox, onboard, start_seconds=5.0, poll_seconds=0.05)
    spec = AgentSpec("/models/m", "vllm", 1, "fp8", "mi355x", {}, "output_throughput", "", {}, "", cards, 60.0, 31100)
    setup = AutonomousSetup(
        onboarding, spec, toolbox, rig.repo, "vllm", poll_seconds=poll, settle_seconds=0.3, clock=clock
    )
    place = BranchPlace(worktree, rig.notebook, tmp_path / BRANCH / "session")
    place.workspace.mkdir(parents=True)
    start = SessionStart(BRANCH, rig.root, 1, BranchTask(agent=str(_agent_repo(tmp_path))))

    def restart() -> AutonomousSession:
        return AutonomousSession(start, setup, place)

    blobs = LocalBlobStore(tmp_path / "blobs")
    return Rig(restart(), sandbox, toolbox, toolbox_notebook, rig.repo, rig.notebook, blobs, restart)


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
    await rig.toolbox_notebook.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second import")
    await rig.toolbox_notebook.write_result(result("r2"))
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


async def test_an_import_waits_while_the_last_one_is_measured_or_judged(tmp_path: Path) -> None:
    later = [0.0]
    rig = await _rig(tmp_path, Agent(tail=30.0), clock=lambda: time.time() + later[0])

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await asyncio.sleep(0.6)
    await rig.toolbox_notebook.write_result(result("r1"))
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


async def test_a_keep_waiting_at_a_crash_is_imported_by_the_resumed_run_with_the_docker_shim(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(exit=3, resume="sleep 30"))

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    assert len(rig.toolbox.requests) == 1
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await rig.toolbox_notebook.write_result(result("r1"))
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
    await rig.toolbox_notebook.write_result(result("r1"))

    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)


def test_only_the_file_the_agent_keeps_its_config_in_leaves_the_source_patch() -> None:
    manifest = AgentManifest.read({**json.loads(_manifest("/ws", Agent())), "config_source": CONFIG_SOURCE})
    sections = [
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y\n"
        for path in ("agent-config.json", "vllm/agent-config.json.py")
    ]

    assert without_file("".join(sections), manifest) == sections[1]


async def test_an_agent_with_no_card_left_fails_its_preflight_before_it_is_fetched(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, cards=())

    with pytest.raises(PreflightFailed, match="no GPU is left"):
        await rig.session.preflight()

    assert not rig.sandbox.started


async def test_a_keep_the_toolbox_refuses_is_noted_and_one_it_has_no_room_for_waits(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(tail=1.0))
    rig.toolbox.refusals = [
        RequestRejected("online tuning is not measured", code="online_tuning_unsupported"),
        Backpressure(BRANCH, 1, 1),
    ]

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the keep after the refused one")
    await rig.toolbox_notebook.write_result(result("r1"))
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
    await rig.toolbox_notebook.write_result(result("r1"))
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
