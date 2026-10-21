# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import result
from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec, CommandResult, OnboardingSetup
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.autonomous_session import AutonomousSession, AutonomousSetup
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionStart
from hyperloom_multibranch.tests.branch_rig import branch_rig

BRANCH = "a1"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)

#: An agent that keeps two changes to its workspace's config.py, a little apart, then ends with ``EXIT``.
AGENT = """
import json, pathlib, sys, time
config = json.loads(pathlib.Path("campaign.json").read_text())
target = pathlib.Path(config["workspace"]) / "vllm" / "config.py"
state = pathlib.Path("run/state.json")
state.parent.mkdir(exist_ok=True)
state.write_text(json.dumps({"keeps": [], "note": ""}))
for step, seqs in enumerate((256, 512), start=1):
    time.sleep(0.3)
    target.write_text(f"max_num_seqs = {seqs}\\n")
    state.write_text(json.dumps({"keeps": [{"decision": "KEEP"}] * step, "note": f"seqs {seqs}"}))
time.sleep(0.3)
sys.exit(EXIT)
"""


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
        self._logs, self.started = logs, []

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
        return LocalProcess(process, log)


class Toolbox:
    """Accepts each request and writes it to the notebook, so it is in flight until the test settles it."""

    def __init__(self, notebook: LocalNotebook) -> None:
        self._notebook, self.requests = notebook, []

    async def submit_test(self, request: TestRequest) -> str:
        accepted = TestRequest(**{**request.__dict__, "request_id": f"r{len(self.requests) + 1}"})
        self.requests.append(accepted)
        await self._notebook.write_request(accepted)
        return accepted.request_id or ""


def _agent_repo(tmp_path: Path, exit_code: int) -> Path:
    repo = tmp_path / "agent-repo"
    repo.mkdir()
    (repo / "agent.py").write_text(AGENT.replace("EXIT", str(exit_code)), encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    return repo


def _manifest(workspace: str) -> str:
    return json.dumps(
        {
            "source": "agent",
            "commit": "c",
            "install": [],
            "config_file": "campaign.json",
            "config": {"workspace": workspace},
            "launch": "python3 agent.py",
            "resume": "",
            "keep": {"path": "run/state.json", "field": "keeps", "where": {"decision": "KEEP"}},
            "config_source": None,
            "summary_field": "note",
            "result_dir": "",
            "needs": [],
        }
    )


async def _session(tmp_path: Path, exit_code: int):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    base, repo, root, notebook = rig.base, rig.repo, rig.root, rig.notebook
    toolbox_notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    worktree = base.worktree(BRANCH, tmp_path / BRANCH / "agent-workspace")
    sandbox, toolbox = LocalSandbox(tmp_path / "logs"), Toolbox(toolbox_notebook)

    async def onboard(checkout: Path, prompt: str) -> str:
        return _manifest(str(worktree.path))

    def spec(workspace: Path) -> AgentSpec:
        return AgentSpec(
            "/models/m", "vllm", 1, "fp8", "mi355x", {}, "output_throughput", ROOT.launch.server_args, {},
            str(workspace), (0,), 60.0,
        )  # fmt: skip

    onboarding = OnboardingSetup(sandbox, onboard, start_seconds=5.0, poll_seconds=0.05)
    setup = AutonomousSetup(onboarding, spec, toolbox, repo, "vllm", poll_seconds=0.05)
    place = BranchPlace(worktree, notebook, tmp_path / BRANCH / "session")
    place.workspace.mkdir(parents=True)
    source = str(_agent_repo(tmp_path, exit_code))
    start = SessionStart(BRANCH, root, 1, BranchTask(agent=source))
    return AutonomousSession(start, setup, place), toolbox, toolbox_notebook, repo, notebook


async def test_an_agent_s_keeps_are_imported_one_at_a_time_against_the_branch_head(tmp_path: Path) -> None:
    session, toolbox, toolbox_notebook, repo, notebook = await _session(tmp_path, exit_code=0)

    await session.preflight()
    running = asyncio.create_task(session.run())
    while not toolbox.requests:
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.0)
    assert len(toolbox.requests) == 1
    await toolbox_notebook.write_result(result("r1"))
    await asyncio.wait_for(running, 10)

    first, last = toolbox.requests[0], toolbox.requests[-1]
    assert (first.branch_id, first.parent_commit, first.intent_fingerprint) == (BRANCH, await repo.head(BRANCH), None)
    [change] = last.delta.components
    assert change.component == "vllm" and not change.remove_patches
    patch = LocalBlobStore(tmp_path / "blobs").get(change.add_patches[0].sha256).decode()
    assert "+max_num_seqs = 512" in patch and "-max_num_seqs = 128" in patch
    notices = [record.entry.content for record in await notebook.branch_log(BRANCH)]
    assert any(note.startswith(f"{NOTICE} keep") and "seqs 512" in note for note in notices)
    assert notices[-1] == f"{NOTICE} the agent finished its run"


async def test_an_agent_that_exits_non_zero_is_a_crash_with_its_last_output(tmp_path: Path) -> None:
    session, *_ = await _session(tmp_path, exit_code=3)

    await session.preflight()

    with pytest.raises(SessionCrashed, match="exited 3"):
        await asyncio.wait_for(session.run(), 10)
    await session.stop()
