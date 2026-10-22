# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.specialist.agent_onboarding import (
    AgentSpec,
    CommandResult,
    OnboardingSetup,
    make_ready,
    split_source,
)
from hyperloom_multibranch.specialist.runtime import PreflightFailed

SPEC = AgentSpec(
    model="/models/Qwen3-14B-FP8",
    framework="sglang",
    tp=1,
    precision="fp8",
    gpu_type="mi355x",
    workload={"kind": "inferencex", "isl": 1024, "osl": 1024, "conc": 64},
    metric="output_throughput",
    server_args="--tp 1",
    envs={},
    workspace="/ws/sglang",
    cards=(7,),
    seconds=3600.0,
)


def _agent_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "agent-repo"
    repo.mkdir()
    (repo / "README.md").write_text("run: python3 -m agent\n", encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    return repo


def _manifest(**overrides: Any) -> str:
    document: dict[str, Any] = {
        "source": "agent",
        "commit": "c",
        "install": ["pip install -e ."],
        "config_file": "campaign.json",
        "config": {"model": SPEC.model, "workspace_root": SPEC.workspace},
        "launch": "python3 -m agent run",
        "keep": {"path": "run/session.json", "field": "keeps"},
        "config_source": None,
        "summary_field": "",
        "result_dir": "",
        "needs": [],
    }
    document.update(overrides)
    return "Here it is:\n```json\n" + json.dumps(document) + "\n```"


class Process:
    def __init__(self, returncode: int | None, output: str = "") -> None:
        self.returncode, self._output, self.killed = returncode, output, False

    async def output(self) -> str:
        return self._output

    async def kill(self) -> None:
        self.killed = True


class Sandbox:
    """Runs nothing: each install exits with the next of ``installs`` (0 once they run out), each start returns the
    next of ``starts`` and, when ``keep`` is set, writes the agent's keep file the way a started agent would."""

    def __init__(self, starts: list[Process], installs: tuple[int, ...] = (), keep: str = "") -> None:
        self.starts, self.installs, self.keep = starts, list(installs), keep
        self.ran: list[tuple[str, Path, Mapping[str, str]]] = []

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult:
        self.ran.append((command, cwd, env))
        code = self.installs.pop(0) if self.installs else 0
        return CommandResult(code, "ERROR: no matching distribution\n" if code else "")

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> Process:
        self.ran.append((command, cwd, env))
        if self.keep:
            (cwd / self.keep).parent.mkdir(parents=True, exist_ok=True)
            (cwd / self.keep).write_text('{"keeps": 0}', encoding="utf-8")
        return self.starts.pop(0)


class Onboarder:
    def __init__(self, answers: list[str]) -> None:
        self.answers, self.prompts = answers, []

    async def __call__(self, checkout: Path, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


def _setup(sandbox: Sandbox, onboard: Onboarder) -> OnboardingSetup:
    async def no_wait(_seconds: float) -> None:
        return None

    ticks = iter(range(1000))
    return OnboardingSetup(
        sandbox, onboard, start_seconds=3.0, poll_seconds=1.0, sleep=no_wait, clock=lambda: float(next(ticks))
    )


async def test_an_agent_is_fetched_onboarded_configured_installed_and_started(tmp_path: Path) -> None:
    repo, process = _agent_repo(tmp_path), Process(None)
    sandbox, onboard = Sandbox([process], keep="run/session.json"), Onboarder([_manifest()])

    ready = await make_ready(str(repo), SPEC, tmp_path / "agent", _setup(sandbox, onboard))

    assert ready.process is process and ready.manifest.launch == "python3 -m agent run"
    assert json.loads((ready.source_dir / "campaign.json").read_text())["workspace_root"] == SPEC.workspace
    assert [command for command, _cwd, _env in sandbox.ran] == ["pip install -e .", "python3 -m agent run"]
    shim = Path(sandbox.ran[-1][2]["PATH"].split(os.pathsep)[0]) / "docker"
    assert os.access(shim, os.X_OK)
    assert '"workspace": "/ws/sglang"' in onboard.prompts[0] and str(repo) in onboard.prompts[0]


async def test_each_failed_attempt_reaches_the_next_prompt_and_three_give_up(tmp_path: Path) -> None:
    repo = _agent_repo(tmp_path)
    sandbox = Sandbox([Process(127, "agent: command not found\n")], installs=(1,))
    onboard = Onboarder(["no json here", _manifest(install=["pip install agent"]), _manifest(install=[])])

    with pytest.raises(PreflightFailed) as failed:
        await make_ready(str(repo), SPEC, tmp_path / "agent", _setup(sandbox, onboard))

    reasons = failed.value.reasons
    assert len(reasons) == 3
    assert reasons[0].startswith("attempt 1: the manifest")
    assert reasons[1].startswith("attempt 2: install `pip install agent` exited 1") and "no matching" in reasons[1]
    assert reasons[2].startswith("attempt 3: `python3 -m agent run` exited 127") and "not found" in reasons[2]
    assert reasons[0] in onboard.prompts[1] and reasons[1] in onboard.prompts[2]


async def test_an_agent_started_by_a_preflight_that_is_cancelled_is_killed(tmp_path: Path) -> None:
    repo, process = _agent_repo(tmp_path), Process(None)
    sandbox, onboard = Sandbox([process]), Onboarder([_manifest()])
    setup = OnboardingSetup(sandbox, onboard, start_seconds=30.0, poll_seconds=0.05)

    preflight = asyncio.create_task(make_ready(str(repo), SPEC, tmp_path / "agent", setup))
    await asyncio.sleep(0.5)
    preflight.cancel()

    with pytest.raises(asyncio.CancelledError):
        await preflight
    assert process.killed


def test_a_source_ref_follows_its_last_at_unless_that_names_a_host() -> None:
    assert split_source("https://github.com/acme/agent@v1.2") == ("https://github.com/acme/agent", "v1.2")
    assert split_source("git@github.com:acme/agent.git") == ("git@github.com:acme/agent.git", "")
    assert split_source("/shared/agent") == ("/shared/agent", "")
