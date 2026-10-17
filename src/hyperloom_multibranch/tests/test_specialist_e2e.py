# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A model-backed specialist end to end on the CPU: a stand-in agent CLI follows the role prompt through the session's
real MCP service, and the runtime, the judge, the rule gate and the experiment repository act on what it submits; only
the measurement is the fake toolbox's."""

import asyncio
import os
import secrets
import sys

import pytest

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe, RecipeDelta
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom.orchestrator.specialists.agent_runner import LocalAgentRunner
from hyperloom_multibranch.contracts.fork import AdmissionStatus
from hyperloom_multibranch.contracts.intent import IntentLayer, intent_fingerprint
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.baseline import write_inferencex_baseline
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSetup
from hyperloom_multibranch.specialist.judging import NOTICE, Judging
from hyperloom_multibranch.specialist.model_sessions import SESSION_DIR, model_sessions
from hyperloom_multibranch.specialist.prompts import role_prompt
from hyperloom_multibranch.specialist.runtime import RuntimePacing, SpecialistRuntime
from hyperloom_multibranch.specialist.tools import register_worktree_tools
from hyperloom_multibranch.specialist.worktree import SourceBase
from hyperloom_multibranch.tests.fakes import RecordingScheduler, RecordingVerifier
from hyperloom_multibranch.toolbox.fake import FakeExecutor
from hyperloom_multibranch.toolbox.fake_model import FakeToolboxConfig, NoiseModel
from hyperloom_multibranch.toolbox.mcp_tools import McpSpecialistTokens, register_specialist_tools
from hyperloom_multibranch.toolbox.submission_log import SubmissionLog
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox, ToolboxStores, ToolboxTuning

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the AI work socket starts the CLI as a POSIX process group"
)

BRANCH, CONC = "s1", 64
TARGET, MECHANISM = "scheduler batch", "raise the default max_num_seqs"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)
QUIET = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)

# The role prompt's work loop as a model follows it, through the MCP service --mcp-config names; a later launch waits,
# as a model waiting on its results does, until the session is stopped.
_STAND_IN = r"""#!/usr/bin/env python3
import json, os, pathlib, re, sys, time, urllib.request

argv = sys.argv
workspace = pathlib.Path(argv[argv.index("--system-prompt-file") + 1]).parent
prompt = sys.stdin.read()
launch = len(list(workspace.glob("seen-*.txt")))
(workspace / f"seen-{launch}.txt").write_text(prompt)
if launch:
    time.sleep(3600)
    sys.exit(0)
(server,) = json.loads(pathlib.Path(argv[argv.index("--mcp-config") + 1]).read_text())["mcpServers"].values()
token = os.environ[re.fullmatch(r"Bearer \$\{(\w+)\}", server["headers"]["Authorization"]).group(1)]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def call(name, **arguments):
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    request = urllib.request.Request(server["url"], json.dumps(message).encode(), headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.loads(response.read())
    with (workspace / "trace.jsonl").open("a") as trace:
        trace.write(json.dumps({"tool": name, "reply": reply}) + "\n")
    result = reply.get("result")
    if result is None or result["isError"]:
        sys.exit(f"{name} failed: {reply}")
    return result["content"][0]["text"]


head = re.search(r"Its head: ([0-9a-f]+)\.", prompt).group(1)
claimed = call("notebook_claim", parent_commit=head, target="@TARGET@", mechanism="@MECHANISM@", layer="source")
found = re.search(r"\b[0-9a-f]{64}\b", claimed)
call("worktree_sync")
pathlib.Path("vllm/config.py").write_text("max_num_seqs = 256\n")
call("worktree_check", component="vllm")
patch = json.loads(call("worktree_patch", component="vllm"))
submitted = call(
    "submit_test",
    kind="test",
    purpose="explore",
    parent_commit=patch["parent_commit"],
    delta=patch["delta"],
    intent_fingerprint=found.group(0) if found else None,
)
request_id = json.loads(submitted)["request_id"]
for _ in range(600):
    if json.loads(call("test_result", request_id=request_id))["state"] == "measured":
        break
    time.sleep(0.1)
call("notebook_append_branch_log", kind="result", content=f"max_num_seqs 256 on {head}: measured", request_ids=[request_id])
"""


class _Planner:
    """The fake measurer runs no compiled environment, so a request plans to nothing."""

    def plan(self, request, parent):
        return None


def _toolbox(root, notebook, repo) -> RealToolbox:
    """The real toolbox over the fake measurer, accepting each request against the session's experiment repository."""
    config = FakeToolboxConfig(
        noise=QUIET, accuracy_metric=ACCURACY_METRICS[0], effect_fn=lambda r: 0.10 if r.delta.components else 0.0
    )
    executor = FakeExecutor(config)
    log = SubmissionLog(root / "submissions.jsonl", str(config.campaign_id))
    stores = ToolboxStores(log, notebook, notebook, claims=notebook, records=notebook)
    acceptance = AcceptanceSetup(repo, _Planner(), None)
    return RealToolbox(stores, executor, acceptance, tuning=ToolboxTuning(clock=lambda: executor.now))


class _Processes:
    def __init__(self) -> None:
        self.adopted = []

    def adopt(self, pid, *, branch, argv):
        self.adopted.append(branch)


async def _until(condition, what: str, explain) -> None:
    for _ in range(1200):
        if await condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}\n{explain()}")


def _read(path) -> str:
    return path.read_text() if path.exists() else f"{path.name} was not written"


async def _prelude(repo, toolbox, notebook_root, gate_notebook) -> str:
    """PRELUDE as the scheduler and the gate leave it: the root committed, s1 open at it, the bar set."""
    root = await repo.init_root(ROOT)
    scheduler = LocalNotebook(notebook_root, Writer(Role.SCHEDULER, "scheduler"))
    root_event = LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root, recipe_digest=ROOT.digest())
    await scheduler.record_lineage(root_event)
    await repo.create_branch(BRANCH, root)
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, BRANCH, from_commit=root))
    baseline = TestRequest(TestKind.TEST, Purpose.BASELINE, INTEGRATION_BRANCH, root, RecipeDelta(), accuracy=True)
    ids = [await toolbox.submit_test(baseline) for _ in range(3)]
    await toolbox.drain()
    await write_inferencex_baseline(gate_notebook, root, [await gate_notebook.record(i) for i in ids], CONC)
    return root


async def test_a_model_backed_specialist_lands_a_source_change_through_the_session_s_tools(tmp_path):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    notebook_root, blobs = tmp_path / "notebook", LocalBlobStore(tmp_path / "blobs")
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs, lambda _layout: installed)
    toolbox = _toolbox(tmp_path, LocalNotebook(notebook_root, Writer(Role.TOOLBOX, "toolbox")), repo)
    gate_notebook = LocalNotebook(notebook_root, Writer(Role.GATE, "gate"))
    reader = LocalNotebook(notebook_root, Writer(Role.OBSERVER, "observer"))

    def notebook_for(branch: str) -> LocalNotebook:
        return LocalNotebook(notebook_root, Writer(Role.SPECIALIST_RUNTIME, branch, branch))

    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="e2e")
    cli = tmp_path / "stand-in-claude"
    cli.write_text(_STAND_IN.replace("@TARGET@", TARGET).replace("@MECHANISM@", MECHANISM))
    cli.chmod(0o755)
    setup = AgentSetup(
        command=AgentCommand("claude", str(cli)),
        role=role_prompt("qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64"),
        env={"PATH": os.environ.get("PATH", "")},
        runner=LocalAgentRunner(
            session_dir=tmp_path / "agents", poll_interval_seconds=0.05, version_probe=lambda a, e: "x"
        ),
        tokens=McpSpecialistTokens(server),
        processes=_Processes(),
        repo=repo,
        min_session_seconds=0.0,
        poll_seconds=0.05,
    )
    sessions = model_sessions(
        setup,
        base=SourceBase(tmp_path / "MB-source", locator=lambda _layout: installed),
        components=("vllm",),
        branch_dir=lambda branch: tmp_path / "MB-branches" / branch,
        notebook_for=notebook_for,
    )
    register_specialist_tools(server, toolbox=toolbox, notebooks=notebook_for)
    register_worktree_tools(server, repo=repo, worktrees=sessions.worktrees, locator=lambda _layout: installed)
    gate = RuleGate(gate_notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())
    judging = Judging(gate, toolbox, InferenceXRule(reader, CONC))
    runtime = SpecialistRuntime(notebook_for, judging, sessions.factory, pacing=RuntimePacing(poll_seconds=0.05))
    workspace = tmp_path / "MB-branches" / BRANCH / SESSION_DIR
    started = False
    try:
        await toolbox.start()
        root = await _prelude(repo, toolbox, notebook_root, gate_notebook)

        async def landed() -> bool:
            return await repo.head(BRANCH) != root

        server.start()
        await runtime.start(BRANCH, root, "leads: a larger max_num_seqs")
        started = True
        await _until(landed, "the candidate to land", lambda: _read(workspace / "trace.jsonl"))
    finally:
        if started:
            await runtime.stop(BRANCH)
        server.stop()
        await toolbox.close()

    (patch,) = (await repo.read_recipe(await repo.head(BRANCH))).component("vllm").patches
    assert b"+max_num_seqs = 256" in blobs.get(patch.sha256)
    records = (await reader.snapshot()).records
    (first,) = [r for r in records if r.request.branch_id == BRANCH and r.request.purpose is Purpose.EXPLORE]
    assert first.conclusion is Conclusion.COMMITTED
    logged = [record.entry.content for record in await reader.branch_log(BRANCH)]
    assert any("max_num_seqs 256" in content for content in logged)
    assert any(content.startswith(f"{NOTICE} {first.request_id} landed on your branch") for content in logged)
    seen = (workspace / "seen-0.txt").read_text()
    assert seen.startswith("Startup brief:\nleads: a larger max_num_seqs") and f"Your branch: {BRANCH}," in seen
    # The role prompt has the specialist submit with the intent fingerprint its claim returned.
    assert first.request.intent_fingerprint == intent_fingerprint(TARGET, MECHANISM, IntentLayer.SOURCE)
