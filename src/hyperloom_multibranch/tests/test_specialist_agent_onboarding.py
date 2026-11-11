# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.specialist import agent_onboarding
from hyperloom_multibranch.specialist.agent_onboard import cli_onboard
from hyperloom_multibranch.specialist.agent_onboarding import (
    ATTEMPTS,
    GATEWAY_PAUSE_SECONDS,
    RETRY_SECONDS,
    AgentSpec,
    AttemptFailed,
    CommandResult,
    ModelRefused,
    OnboardingSetup,
    agent_env,
    fetch_source,
    gateway_wait,
    make_ready,
    onboarding_prompt,
    split_source,
)
from hyperloom_multibranch.specialist.agent_session import AgentCommand
from hyperloom_multibranch.specialist.arbor_adapter import MARKERS, is_arbor
from hyperloom_multibranch.specialist.runtime import PreflightFailed
from hyperloom_multibranch.tests.agent_rig import Onboarder, Process, Sandbox, _setup

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
    port=31100,
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
        "needs": [],
    }
    document.update(overrides)
    return "Here it is:\n```json\n" + json.dumps(document) + "\n```"


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
    assert f"checkout, {ready.source_dir}, fetched from {repo}" in onboard.prompts[0]


async def test_an_onboarded_agent_is_never_told_why_no_source_change_can_be_kept(tmp_path: Path) -> None:
    spec = replace(SPEC, source_unkept="the image's sglang names no upstream commit")
    sandbox, onboard = Sandbox([Process(None)], keep="run/session.json"), Onboarder([_manifest()])

    ready = await make_ready(str(_agent_repo(tmp_path)), spec, tmp_path / "agent", _setup(sandbox, onboard))

    assert spec.source_unkept not in onboard.prompts[0]
    assert ready.source_unkept == ""


def test_the_prompt_has_every_a_b_measured_with_the_gate_s_client_and_the_integration_notes_read() -> None:
    client = "python3 /cache/InferenceX/utils/bench_serving/benchmark_serving.py --backend vllm --ignore-eos"
    spec = replace(SPEC, client=client, baseline=6290.7, knowledge="/b01/agent/mb-integration.md")

    prompt = onboarding_prompt(spec, "agent", Path("/b01/agent/src"), "c", ())

    document = json.loads(prompt[prompt.index("{\n") : prompt.index("\n}\n") + 2])
    assert (document["client"], document["baseline"], document["knowledge"]) == (
        client, 6290.7, "/b01/agent/mb-integration.md",
    )  # fmt: skip
    assert "every A/B" in prompt and "every iteration" in prompt


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


async def test_the_preflight_gives_up_after_the_attempts_the_module_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_onboarding, "ATTEMPTS", 1)
    onboard = Onboarder(["no json here", _manifest()])

    with pytest.raises(PreflightFailed) as failed:
        await make_ready(str(_agent_repo(tmp_path)), SPEC, tmp_path / "agent", _setup(Sandbox([]), onboard))

    assert len(failed.value.reasons) == 1 and len(onboard.prompts) == 1


class _TimedSandbox(Sandbox):
    def __init__(self, starts: list[Process]) -> None:
        super().__init__(starts, keep="run/session.json")
        self.timeouts: list[float] = []

    async def run(self, command: str, cwd: Path, env, timeout: float) -> CommandResult:
        self.timeouts.append(timeout)
        return await super().run(command, cwd, env, timeout)


async def test_each_install_command_runs_with_the_module_s_install_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_onboarding, "INSTALL_SECONDS", 12.5)
    sandbox = _TimedSandbox([Process(None)])

    await make_ready(str(_agent_repo(tmp_path)), SPEC, tmp_path / "agent", _setup(sandbox, Onboarder([_manifest()])))

    assert sandbox.timeouts == [12.5]


async def test_a_model_call_the_gateway_refuses_fails_the_preflight_at_once_naming_it(tmp_path: Path) -> None:
    refusal = "the onboarding model's call failed: API Error: 400 this CLI does not support this model"
    asked, waits = [], []

    async def refused(checkout: Path, prompt: str) -> str:
        asked.append(prompt)
        raise ModelRefused(refusal)

    with pytest.raises(PreflightFailed) as failed:
        await make_ready(str(_agent_repo(tmp_path)), SPEC, tmp_path / "agent", _setup(Sandbox([]), refused, waits))

    assert failed.value.reasons == (f"attempt 1: {refusal}",)
    assert len(asked) == 1 and waits == []


# Stands in for the onboarding Claude CLI: its first run ends in a gateway error, every later one answers the manifest.
FLAKY_CLI = """#!{python}
import json, os, sys
sys.stdin.read()
runs = os.environ["FAKE_RUNS"]
count = int(open(runs).read()) if os.path.exists(runs) else 0
open(runs, "w").write(str(count + 1))
usage = {{"input_tokens": 10, "output_tokens": 40}}
message = {{"id": f"m{{count}}", "model": "claude-sonnet-5-5", "usage": usage, "content": []}}
print(json.dumps({{"type": "assistant", "message": message, "session_id": f"s-{{count}}"}}))
error = count == 0
answer = "API Error: 503 upstream connect error" if error else {answer!r}
print(json.dumps({{"type": "result", "is_error": error, "result": answer, "session_id": f"s-{{count}}"}}))
"""


class _Usage:
    def __init__(self) -> None:
        self.recorded: list = []

    async def record_llm_usage(self, usage) -> None:
        self.recorded.append(usage)


async def test_the_onboarding_cli_failing_once_is_waited_out_and_the_next_attempt_onboards(tmp_path: Path) -> None:
    cli = tmp_path / "claude"
    cli.write_text(FLAKY_CLI.format(python=sys.executable, answer=_manifest()), encoding="utf-8")
    cli.chmod(0o755)
    env = {"PATH": os.environ.get("PATH", ""), "FAKE_RUNS": str(tmp_path / "runs")}
    usage, waits, process = _Usage(), [], Process(None)
    onboard = cli_onboard(AgentCommand("claude", str(cli), "claude-sonnet-5-5"), env, "b01", usage)

    ready = await make_ready(
        str(_agent_repo(tmp_path)), SPEC, tmp_path / "agent", _setup(Sandbox([process]), onboard, waits)
    )

    assert ready.process is process and ready.manifest.launch == "python3 -m agent run"
    assert RETRY_SECONDS in waits
    assert [used.launch for used in usage.recorded] == ["s-0", "s-1"]


# Stands in for the onboarding Claude CLI behind a gateway that cannot be reached: its first FAKE_DOWN runs retry their
# model call ten times and give up on the proxy tunnel, as the CLI does, every later one answers the manifest.
OUTAGE_CLI = """#!{python}
import json, os, sys
sys.stdin.read()
runs = os.environ["FAKE_RUNS"]
count = int(open(runs).read()) if os.path.exists(runs) else 0
open(runs, "w").write(str(count + 1))
print(json.dumps({{"type": "system", "subtype": "init", "session_id": f"s-{{count}}"}}))
if count < int(os.environ["FAKE_DOWN"]):
    for attempt in range(1, 11):
        retry = {{"type": "system", "subtype": "api_retry", "attempt": attempt, "max_retries": 10, "error_status": None}}
        print(json.dumps(retry))
    text = "API Error: Unable to connect to API (ERR_PROXY_TUNNEL)"
    message = {{"id": "m0", "model": "<synthetic>", "content": [{{"type": "text", "text": text}}]}}
    print(json.dumps({{"type": "assistant", "message": message, "is_api_error_message": True}}))
    print(json.dumps({{"type": "result", "is_error": True, "result": text, "session_id": f"s-{{count}}"}}))
    sys.exit(1)
message = {{"id": "m1", "model": "claude-sonnet-5-5", "content": []}}
print(json.dumps({{"type": "assistant", "message": message, "session_id": f"s-{{count}}"}}))
print(json.dumps({{"type": "result", "is_error": False, "result": {answer!r}, "session_id": f"s-{{count}}"}}))
"""


class _Clock:
    """A clock that moves only when the preflight sleeps."""

    def __init__(self) -> None:
        self.now, self.waits = 0.0, []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


def _outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, down: int
) -> tuple[OnboardingSetup, _Clock, Path, Process]:
    cli = tmp_path / "claude"
    cli.write_text(OUTAGE_CLI.format(python=sys.executable, answer=_manifest()), encoding="utf-8")
    cli.chmod(0o755)
    runs = tmp_path / "runs"
    env = {"PATH": os.environ.get("PATH", ""), "FAKE_RUNS": str(runs), "FAKE_DOWN": str(down)}
    onboard = cli_onboard(AgentCommand("claude", str(cli), "claude-sonnet-5-5"), env, "b01", _Usage())
    clock, process = _Clock(), Process(None)
    sandbox = Sandbox([process], keep="run/session.json")
    monkeypatch.setattr(agent_onboarding, "monotonic", clock)
    return OnboardingSetup(sandbox, onboard, sleep=clock.sleep), clock, runs, process


async def test_a_gateway_outage_longer_than_every_attempt_is_waited_out_and_the_agent_onboards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, clock, runs, process = _outage(tmp_path, monkeypatch, down=ATTEMPTS + 2)

    ready = await make_ready(str(_agent_repo(tmp_path)), SPEC, tmp_path / "agent", setup)

    assert ready.process is process
    assert runs.read_text() == str(ATTEMPTS + 3)
    assert clock.waits == sorted(clock.waits) and clock.waits[0] == RETRY_SECONDS


async def test_a_gateway_outage_is_waited_out_for_half_the_branch_s_time_then_spends_the_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, clock, _runs, _process = _outage(tmp_path, monkeypatch, down=1000)

    with pytest.raises(PreflightFailed) as failed:
        await make_ready(str(_agent_repo(tmp_path)), SPEC, tmp_path / "agent", setup)

    assert len(failed.value.reasons) == ATTEMPTS
    assert all("ERR_PROXY_TUNNEL" in reason for reason in failed.value.reasons)
    wait = gateway_wait(SPEC)
    assert wait == SPEC.seconds / 2
    assert wait - GATEWAY_PAUSE_SECONDS < clock.now <= wait + RETRY_SECONDS * (ATTEMPTS - 1)


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


def _git_out(cwd: Path, *argv: str) -> str:
    return subprocess.run(["git", *argv], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


async def test_a_source_tree_outside_git_is_copied_and_committed_once(tmp_path: Path) -> None:
    host = _agent_repo(tmp_path)
    tree = host / "src" / "vendored"
    for marker in (*MARKERS, "__pycache__/component.pyc"):
        (tree / marker).parent.mkdir(parents=True, exist_ok=True)
        (tree / marker).write_text("x", encoding="utf-8")
    checkout = tmp_path / "checkout"

    commit = await fetch_source(str(tree), checkout)

    assert is_arbor(checkout) and not (checkout / "__pycache__").exists()
    assert commit == _git_out(checkout, "rev-parse", "HEAD")
    assert _git_out(host, "rev-parse", "HEAD") in _git_out(checkout, "log", "-1", "--format=%B")
    (tree / MARKERS[0]).write_text("changed", encoding="utf-8")
    assert await fetch_source(str(tree), checkout) == commit
    assert (checkout / MARKERS[0]).read_text(encoding="utf-8") == "x"


async def test_a_bare_repository_source_is_cloned_not_copied(tmp_path: Path) -> None:
    repo = _agent_repo(tmp_path)
    bare = tmp_path / "agent.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(bare)], check=True)

    commit = await fetch_source(str(bare), tmp_path / "checkout")

    assert commit == _git_out(repo, "rev-parse", "HEAD")
    assert sorted(path.name for path in (tmp_path / "checkout").iterdir()) == [".git", "README.md"]


def test_a_source_ref_follows_its_last_at_unless_that_names_a_host() -> None:
    assert split_source("https://github.com/acme/agent@v1.2") == ("https://github.com/acme/agent", "v1.2")
    assert split_source("git@github.com:acme/agent.git") == ("git@github.com:acme/agent.git", "")
    assert split_source("/shared/agent") == ("/shared/agent", "")
    assert split_source("https://github.com/acme/agent.git@feature/x") == (
        "https://github.com/acme/agent.git",
        "feature/x",
    )
    assert split_source("ssh://git@github.com/acme/agent.git") == ("ssh://git@github.com/acme/agent.git", "")
    assert split_source("ssh://git@github.com/acme/agent@v1") == ("ssh://git@github.com/acme/agent", "v1")
    assert split_source("git@github.com:acme/agent.git@v1") == ("git@github.com:acme/agent.git", "v1")
    assert split_source("/shared/agent@feature/x") == ("/shared/agent", "feature/x")


async def test_a_ref_that_cannot_be_checked_out_fails_every_attempt(tmp_path: Path) -> None:
    repo = _agent_repo(tmp_path)
    sandbox, onboard = Sandbox([Process(None)] * 3, keep="run/session.json"), Onboarder([_manifest()] * 3)

    with pytest.raises(PreflightFailed) as failed:
        await make_ready(f"{repo}@no-such-ref", SPEC, tmp_path / "agent", _setup(sandbox, onboard))

    assert all("checking out no-such-ref" in reason for reason in failed.value.reasons)
    assert not sandbox.ran


def test_the_docker_shim_reads_a_device_mask_the_agent_gives_as_indices_into_its_own_cards(tmp_path: Path) -> None:
    docker = Path(agent_env(tmp_path)["PATH"].split(os.pathsep)[0]) / "docker"
    sandbox = {**os.environ, "ROCR_VISIBLE_DEVICES": "5,7", "HIP_VISIBLE_DEVICES": "0,1"}

    def visible(name: str, index: str) -> subprocess.CompletedProcess[str]:
        show = 'echo "$HIP_VISIBLE_DEVICES/$ROCR_VISIBLE_DEVICES"'
        argv = [str(docker), "exec", "-e", f"{name}={index}", "c", "sh", "-c", show]
        return subprocess.run(argv, env=sandbox, capture_output=True, text=True)

    assert visible("ROCR_VISIBLE_DEVICES", "0").stdout.strip() == "0/5,7"
    assert visible("HIP_VISIBLE_DEVICES", "1,0").stdout.strip() == "1,0/5,7"
    assert visible("CUDA_VISIBLE_DEVICES", "").stdout.strip() == "0,1/5,7"
    refused = visible("ROCR_VISIBLE_DEVICES", "2")
    assert refused.returncode != 0 and "card" in refused.stderr


# -- the agent's directory is the group's, and its checkout's config is not this host's ------------------------------
def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


async def test_a_clone_and_a_copy_are_the_group_s_to_write_and_the_shim_is_replaced_by_rename(tmp_path: Path) -> None:
    tmp_path.chmod(0o2775)  # storage set up to be shared: setgid, group-writable, closed to others
    repo = _agent_repo(tmp_path)
    tree = tmp_path / "vendored"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "run.py").write_text("x", encoding="utf-8")
    tree.chmod(0o755)

    await fetch_source(str(repo), tmp_path / "agent" / "cloned")
    await fetch_source(str(tree), tmp_path / "agent" / "copied")
    shim = Path(agent_env(tmp_path / "agent")["PATH"].split(os.pathsep)[0]) / "docker"
    (tmp_path / "earlier-shim").hardlink_to(shim)
    agent_env(tmp_path / "agent")

    for checkout in (tmp_path / "agent" / "cloned", tmp_path / "agent" / "copied"):
        assert _mode(checkout) & 0o070 == 0o070, checkout
        assert _mode(checkout / ".git" / "objects") & 0o070 == 0o070, checkout
    assert _mode(tmp_path / "agent" / "copied" / "pkg") & 0o2070 == 0o2070
    assert _mode(tmp_path / "agent" / "copied" / "pkg" / "run.py") & 0o060 == 0o060
    assert _mode(tmp_path / "agent") == 0o2775 and _mode(shim.parent) == 0o2775 and _mode(shim) == 0o755
    assert (tmp_path / "earlier-shim").stat().st_ino != shim.stat().st_ino, "replaced by rename, never written in place"


async def test_no_program_the_agent_writes_into_its_checkout_s_config_runs_as_this_host(tmp_path: Path) -> None:
    checkout = tmp_path / "agent" / "src"
    await fetch_source(str(_agent_repo(tmp_path)), checkout)
    ran = tmp_path / "ran"
    program = tmp_path / "recording-filter"
    program.write_text(f"#!/bin/sh\necho ran >> {ran}\ncat\n")
    program.chmod(0o755)
    hook = tmp_path / "hook"
    hook.write_text(f"#!/bin/sh\necho ran >> {ran}\n")
    hook.chmod(0o755)
    with (checkout / ".git" / "config").open("a") as config:
        config.write(
            f'[filter "own"]\n\tclean = {program}\n\tsmudge = {program}\n'
            f"[core]\n\tfsmonitor = {hook}\n\thooksPath = {tmp_path}\n"
        )
    (tmp_path / "post-checkout").write_text(f"#!/bin/sh\necho ran >> {ran}\n")
    (tmp_path / "post-checkout").chmod(0o755)
    (checkout / ".gitattributes").write_text("* filter=own\n")
    (checkout / "README.md").write_text("changed\n")

    for argv in (["git", "status", "--porcelain"], ["git", "add", "--all"], ["git", "checkout", "--", "README.md"]):
        done = await agent_onboarding._git(argv, checkout)
        assert done.returncode == 0, done.output

    assert not ran.exists(), ran.read_text()


async def test_a_copy_outside_shared_storage_keeps_the_modes_of_the_tree_it_copied(tmp_path: Path) -> None:
    tmp_path.chmod(0o755)
    tree = tmp_path / "vendored"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "run.py").write_text("x", encoding="utf-8")
    (tree / "pkg").chmod(0o755)
    (tree / "pkg" / "run.py").chmod(0o644)

    await fetch_source(str(tree), tmp_path / "agent" / "copied")

    assert _mode(tmp_path / "agent" / "copied" / "pkg") == 0o755
    assert _mode(tmp_path / "agent" / "copied" / "pkg" / "run.py") == 0o644


@pytest.mark.skipif(os.geteuid() != 0, reason="a checkout of another owner needs root to chown")
async def test_a_checkout_another_account_of_the_group_made_is_read_with_no_program_of_its_config_run(
    tmp_path: Path,
) -> None:
    checkout, source = tmp_path / "agent" / "src", str(_agent_repo(tmp_path))
    commit = await fetch_source(source, checkout)
    ran = tmp_path / "ran"
    program = tmp_path / "recording-filter"
    program.write_text(f"#!/bin/sh\necho ran >> {ran}\ncat\n")
    program.chmod(0o755)
    with (checkout / ".git" / "config").open("a") as config:
        config.write(f'[filter "own"]\n\tclean = {program}\n\tsmudge = {program}\n')
    (checkout / ".gitattributes").write_text("* filter=own\n")
    (checkout / "README.md").unlink()
    # An earlier attempt, or the run's earlier leg, was another account of the group.
    for path in (checkout, *checkout.rglob("*")):
        os.lchown(path, 65534, 65534)

    again = await fetch_source(source, checkout)
    checked = await agent_onboarding._checkout_git(["git", "checkout", "--", "README.md"], checkout)

    assert again == commit
    assert checked.returncode == 0, checked.output
    assert not ran.exists(), ran.read_text()


async def test_a_checkout_whose_commit_cannot_be_read_fails_the_attempt_naming_it(tmp_path: Path) -> None:
    checkout = tmp_path / "agent" / "src"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)  # an attempt cut short before its clone fetched

    with pytest.raises(AttemptFailed, match=f"reading the commit of {checkout}"):
        await fetch_source(str(_agent_repo(tmp_path)), checkout)


async def test_a_checkout_its_agent_made_a_partial_clone_fetches_nothing_behind_this_host_s_git(tmp_path: Path) -> None:
    checkout = tmp_path / "agent" / "src"
    commit = await fetch_source(str(_agent_repo(tmp_path)), checkout)
    ran, transport = tmp_path / "ran", tmp_path / "transport"
    transport.write_text(f"#!/bin/sh\necho ran >> {ran}\nexit 1\n")
    transport.chmod(0o755)
    with (checkout / ".git" / "config").open("a") as config:
        config.write(
            f"[core]\n\trepositoryformatversion = 1\n\tsshCommand = {transport}\n"
            '[extensions]\n\tpartialClone = promisor\n[remote "promisor"]\n\turl = ssh://example.invalid/r\n'
            "\tpromisor = true\n"
        )
    for packed in (checkout / ".git" / "objects" / "pack").glob("*"):
        packed.unlink()
    (checkout / ".git" / "objects" / commit[:2] / commit[2:]).unlink(missing_ok=True)

    done = await agent_onboarding._git(["git", "cat-file", "-p", commit], checkout)

    assert done.returncode != 0 and not ran.exists()
