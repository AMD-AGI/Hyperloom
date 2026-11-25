# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A model-backed specialist end to end on the CPU: a stand-in agent CLI follows the role prompt through the session's
real MCP service, and the runtime, the judge, the rule gate and the experiment repository act on what it submits; only
the measurement is the fake toolbox's. Where the DeepSeek Harness SDK is installed, its own runtime does the same
behind its bridge, on a scripted model that follows the role prompt one tool call at a time."""

import asyncio
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hyperloom.common.gain_math import ACCURACY_METRICS
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe, RecipeDelta
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.orchestrator.specialists import launch_env
from hyperloom.orchestrator.specialists.agent_runner import LocalAgentRunner
from hyperloom_common.llm.llm_config import deepseek_harness_sdk_installed
from hyperloom_common.tests._bwrap_host import usable_bwrap
from hyperloom_multibranch.contracts.fork import AdmissionStatus
from hyperloom_multibranch.contracts.intent import IntentLayer, intent_fingerprint
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, Conclusion, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.baseline import write_inferencex_baseline
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.specialist.agent_onboard import cli_onboard
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSetup
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.dsh_cli import MODULE as DSH_CLI
from hyperloom_multibranch.specialist.judging import NOTICE, Judging
from hyperloom_multibranch.specialist.model_sessions import SESSION_DIR, model_sessions
from hyperloom_multibranch.specialist.prompts import RoleWorkload, role_prompt
from hyperloom_multibranch.specialist.runtime import RuntimePacing, SpecialistRuntime
from hyperloom_multibranch.specialist import tools as tools_module
from hyperloom_multibranch.specialist.tools import register_worktree_tools
from hyperloom_multibranch.specialist import worktree as worktree_module
from hyperloom_multibranch.specialist.worktree import SourceBase
from hyperloom_multibranch.tests.fakes import RecordingScheduler, RecordingVerifier
from hyperloom_multibranch.toolbox.tests.fake import FakeExecutor
from hyperloom_multibranch.toolbox.tests.fake_model import FakeToolboxConfig, NoiseModel
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
prompt = json.loads(sys.stdin.readline())["message"]["content"][0]["text"]
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
claimed = call("blackboard_claim", parent_commit=head, target="@TARGET@", mechanism="@MECHANISM@", layer="source")
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
call("blackboard_append_branch_log", kind="result", content=f"max_num_seqs 256 on {head}: measured", request_ids=[request_id])
"""


#: What each stand-in leaves in the workspace of what it did, shown when the candidate never lands.
_TRACE = {"claude": "trace.jsonl", "deepseek": "process.log"}


class _Planner:
    """The fake measurer runs no compiled environment, so a request plans to nothing."""

    blobs = None

    def plan(self, request, parent):
        return None


def _toolbox(root, blackboard, repo) -> RealToolbox:
    """The real toolbox over the fake measurer, accepting each request against the session's experiment repository."""
    config = FakeToolboxConfig(
        noise=QUIET, accuracy_metric=ACCURACY_METRICS[0], effect_fn=lambda r: 0.10 if r.delta.components else 0.0
    )
    executor = FakeExecutor(config)
    log = SubmissionLog(root / "submissions.jsonl", str(config.campaign_id))
    stores = ToolboxStores(log, blackboard, blackboard, claims=blackboard, records=blackboard)
    acceptance = AcceptanceSetup(repo, _Planner(), None)
    return RealToolbox(stores, executor, acceptance, tuning=ToolboxTuning(clock=lambda: executor.now))


class _Processes:
    def __init__(self) -> None:
        self.adopted = []

    def adopt(self, pid, *, branch, argv):
        self.adopted.append(branch)

    def forget(self, pid):
        pass


async def _until(condition, what: str, explain) -> None:
    for _ in range(1200):
        if await condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}\n{explain()}")


def _read(path) -> str:
    return path.read_text() if path.exists() else f"{path.name} was not written"


async def _prelude(repo, toolbox, blackboard_root, gate_blackboard) -> str:
    """PRELUDE as the scheduler and the gate leave it: the root committed, s1 open at it, the bar set."""
    root = await repo.init_root(ROOT)
    scheduler = LocalBlackboard(blackboard_root, Writer(Role.SCHEDULER, "scheduler"))
    root_event = LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root, recipe_digest=ROOT.digest())
    await scheduler.record_lineage(root_event)
    await repo.create_branch(BRANCH, root)
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, BRANCH, from_commit=root))
    baseline = TestRequest(TestKind.TEST, Purpose.BASELINE, INTEGRATION_BRANCH, root, RecipeDelta(), accuracy=True)
    ids = [await toolbox.submit_test(baseline) for _ in range(3)]
    await toolbox.drain()
    await write_inferencex_baseline(gate_blackboard, root, [await gate_blackboard.record(i) for i in ids], CONC)
    return root


async def test_a_model_backed_specialist_lands_a_source_change_through_the_session_s_tools(tmp_path, monkeypatch):
    cli = tmp_path / "stand-in-claude"
    cli.write_text(_STAND_IN.replace("@TARGET@", TARGET).replace("@MECHANISM@", MECHANISM))
    cli.chmod(0o755)
    command = AgentCommand("claude", str(cli), model="claude-opus-5-5")

    workspace = await _lands_a_source_change(tmp_path, monkeypatch, command, {"PATH": os.environ.get("PATH", "")})

    seen = (workspace / "seen-0.txt").read_text()
    assert seen.startswith(f"Your branch's direction, to work on first:\n{DIRECTIONS[0]}")
    assert "Startup brief:\nleads: a larger max_num_seqs" in seen and f"Your branch: {BRANCH}," in seen


def _blocks(message: dict) -> list[dict]:
    content = message.get("content")
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])


def _result_text(block: dict) -> str:
    return "".join(part.get("text", "") for part in _blocks(block))


def _json_in(text: str) -> dict:
    return json.JSONDecoder().raw_decode(text[text.index("{") :])[0]


def _scripted_step(body: dict) -> tuple[str, dict] | str:
    """What the scripted model does next, read off the conversation the runtime sent: the next tool call of the role
    prompt's work loop, as the stand-in CLI makes them, or the text that ends its turn."""
    said = [block for message in body["messages"] for block in _blocks(message)]
    names = {block["id"]: block["name"].rsplit("__", 1)[-1] for block in said if block.get("type") == "tool_use"}
    results = [block for block in said if block.get("type") == "tool_result"]
    done = {names[block["tool_use_id"]]: _result_text(block) for block in results}
    if not any(block.get("type") == "tool_result" for block in _blocks(body["messages"][-1])):
        if done:
            return "Nothing more to do until something arrives."
        prompt = " ".join(block.get("text", "") for block in said if block.get("type") == "text")
        head = re.search(r"Its head: ([0-9a-f]+)\.", prompt).group(1)
        return "blackboard_claim", {"parent_commit": head, "target": TARGET, "mechanism": MECHANISM, "layer": "source"}
    last = names[results[-1]["tool_use_id"]]
    if results[-1].get("is_error"):
        return f"{last} failed: {done[last]}"
    if last == "blackboard_claim":
        return "worktree_sync", {}
    if last == "worktree_sync":
        write = "printf 'max_num_seqs = 256\\n' > vllm/config.py"
        return "bash", {"command": write, "description": "Raise the default max_num_seqs"}
    if last == "bash":
        return "worktree_check", {"component": "vllm"}
    if last == "worktree_check":
        return "worktree_patch", {"component": "vllm"}
    if last == "worktree_patch":
        patch, found = _json_in(done[last]), re.search(r"\b[0-9a-f]{64}\b", done["blackboard_claim"])
        return "submit_test", {
            "kind": "test",
            "purpose": "explore",
            "parent_commit": patch["parent_commit"],
            "delta": patch["delta"],
            "intent_fingerprint": found.group(0) if found else None,
        }
    request_id = _json_in(done["submit_test"])["request_id"]
    if last == "submit_test" or (last == "test_result" and _json_in(done[last])["state"] != "measured"):
        time.sleep(0.1)
        return "test_result", {"request_id": request_id}
    if last == "test_result":
        content = "max_num_seqs 256: measured"
        return "blackboard_append_branch_log", {"kind": "result", "content": content, "request_ids": [request_id]}
    return "The candidate is measured and logged."


class _ScriptedModel:
    """The endpoint the harness's runtime calls its model at (the Anthropic Messages wire, streamed), answering each
    request with what ``step`` makes of it: a tool call, or the text that ends the turn."""

    def __init__(self, step=_scripted_step) -> None:
        self.steps: list[tuple[str, dict] | str] = []
        self.bodies: list[dict] = []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                return

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                model.bodies.append(body)
                model.steps.append(step(body))
                step_made = model.steps[-1]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for name, payload in model.events(body, step_made, len(model.steps)):
                    self.wfile.write(f"event: {name}\ndata: {json.dumps({'type': name, **payload})}\n\n".encode())
                self.wfile.flush()
                self.close_connection = True

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def env(self, home) -> dict[str, str]:
        """The environment of a harness that calls this model, with ``home`` as its home. As a session's allowlisted
        environment, it does not say where this app's packages are, which a test may import from a source tree; what
        else this interpreter was pointed at, the harness SDK among it, it keeps."""
        home.mkdir()
        others = [entry for entry in sys.path if entry and Path(entry).resolve() != APP_PACKAGES]
        return {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "PYTHONPATH": os.pathsep.join(others),
            "OPENAI_BASE_URL": self.url,
            "OPENAI_API_KEY": "scripted",
        }

    @staticmethod
    def events(body: dict, step: tuple[str, dict] | str, number: int) -> list[tuple[str, dict]]:
        """The stream of one reply: a tool call by the name the runtime gave the tool, or text."""
        usage = {"input_tokens": 200, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}
        opened = {"id": f"msg_{number}", "type": "message", "role": "assistant", "model": body["model"], "content": []}
        if isinstance(step, str):
            block, delta, stop = {"type": "text", "text": ""}, {"type": "text_delta", "text": step}, "end_turn"
        else:
            raw, arguments = step
            tools = [tool["name"] for tool in body["tools"]]
            name = next(tool for tool in tools if tool == raw or tool.endswith(f"__{raw}"))
            block = {"type": "tool_use", "id": f"toolu_{number}", "name": name, "input": {}}
            delta, stop = {"type": "input_json_delta", "partial_json": json.dumps(arguments)}, "tool_use"
        return [
            ("message_start", {"message": {**opened, "usage": {**usage, "output_tokens": 1}}}),
            ("content_block_start", {"index": 0, "content_block": block}),
            ("content_block_delta", {"index": 0, "delta": delta}),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": stop}, "usage": {**usage, "output_tokens": 30}}),
            ("message_stop", {}),
        ]


needs_harness = pytest.mark.skipif(
    not deepseek_harness_sdk_installed(), reason="the DeepSeek Harness SDK and its runtime are an optional extra"
)
HARNESS = AgentCommand("deepseek", sys.executable, model="deepseek-v4-flash")
#: The directory this test imports the app's packages from.
APP_PACKAGES = Path(__file__).resolve().parents[2]


@needs_harness
async def test_the_deepseek_harness_lands_a_source_change_through_the_session_s_tools(tmp_path, monkeypatch):
    model = _ScriptedModel()
    try:
        workspace = await _lands_a_source_change(tmp_path, monkeypatch, HARNESS, model.env(tmp_path / "home"))
    finally:
        model.stop()

    called = [step[0] for step in model.steps if not isinstance(step, str)]
    assert called[:6] == [
        "blackboard_claim",
        "worktree_sync",
        "bash",
        "worktree_check",
        "worktree_patch",
        "submit_test",
    ]
    assert called[-1] == "blackboard_append_branch_log" and set(called[6:-1]) == {"test_result"}
    rows = [json.loads(line) for line in (workspace / "process.log").read_text().splitlines() if line.startswith("{")]
    init = rows[0]
    assert (init["type"], init["subtype"], init["harness"]) == ("system", "init", "dsh")
    assert init["mcp_servers"] == [{"name": "hyperloom"}]
    said = [block for row in rows if row["type"] == "assistant" for block in row["message"]["content"]]
    used = [block["name"] for block in said if block["type"] == "tool_use"]
    assert "mcp__hyperloom__submit_test" in used and "bash" in used
    # The conversation is kept in the branch's workspace, uncompressed, for a later launch to resume.
    assert list((workspace / ".dsh" / "sessions").glob("*/*/session.v*.jsonl"))
    # What the launch's model calls used is the branch's spend, at the harness model's rates.
    reader = LocalBlackboard(tmp_path / "blackboard", Writer(Role.OBSERVER, "observer"))
    spent = [record.usage for record in (await reader.snapshot()).llm_usage if record.usage.branch == BRANCH]
    assert spent and {usage.launch for usage in spent} == {init["session_id"]}
    assert sum(usage.tokens.input for usage in spent) >= 200 * len(called)
    assert all(usage.cost_usd for usage in spent)


@needs_harness
@pytest.mark.agent_isolation
async def test_the_deepseek_harness_lands_a_source_change_from_inside_a_bubblewrap_sandbox(tmp_path, monkeypatch):
    isolation = BwrapIsolation(usable_bwrap())
    monkeypatch.setattr(launch_env, "choose_agent_isolation", lambda **_kwargs: isolation)
    model = _ScriptedModel()
    try:
        workspace = await _lands_a_source_change(tmp_path, monkeypatch, HARNESS, model.env(tmp_path / "home"))
    finally:
        model.stop()

    # The launch was a sandbox's: its home was pointed into the workspace, the only tree besides the worktree it
    # writes. From there the runtime reached the session's tools and its model over a network of its own, and kept
    # its conversation in the workspace.
    assert (workspace / launch_env.SANDBOX_HOME).is_dir()
    assert "blackboard_append_branch_log" in [step[0] for step in model.steps if not isinstance(step, str)]
    assert list((workspace / ".dsh" / "sessions").glob("*/*/session.v*.jsonl"))


def _killed_after_a_turn(argv: list[str], message: str, env: dict[str, str], worktree) -> str:
    """Run the first turn of the CLI ``argv`` on ``message``, then SIGKILL it and its runtime once the runtime wrote
    the turn's end to its session log, as a crash between two turns leaves them; the turn's result row."""
    cli = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=worktree, text=True, start_new_session=True
    )
    try:
        cli.stdin.write(message)
        cli.stdin.flush()
        result = next(line for line in cli.stdout if '"result"' in line)
        logs = Path(env["DSH_HOME"], "sessions")
        for _ in range(600):
            if any('"turn/end"' in log.read_text() for log in logs.glob("*/*/session.v*.jsonl")):
                break
            time.sleep(0.05)
    finally:
        os.killpg(cli.pid, signal.SIGKILL)
        cli.wait()
    return result


@needs_harness
def test_the_deepseek_harness_goes_on_with_the_conversation_of_a_process_that_was_killed(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    model = _ScriptedModel(lambda body: f"heard {len(body['messages'])} messages")
    env = {**HARNESS.env(model.env(tmp_path / "home")), "DSH_HOME": str(tmp_path / "session" / ".dsh")}

    def command(*conversation: str) -> list[str]:
        return [sys.executable, "-m", DSH_CLI, "--model", HARNESS.model, *conversation]

    def message(text: str) -> str:
        return json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}})

    def launch(*conversation: str, says: str) -> subprocess.CompletedProcess[str]:
        asked = message(says) + "\n"
        return subprocess.run(
            command(*conversation), input=asked, env=env, cwd=worktree, capture_output=True, text=True, timeout=120
        )

    try:
        first = _killed_after_a_turn(
            command("--session-id", "c0ffee00-0000-4000-8000-000000000001"),
            message("remember the word heron") + "\n",
            env,
            worktree,
        )
        second = launch("--resume", "c0ffee00-0000-4000-8000-000000000001", says="which word was it?")
        unknown = launch("--resume", "c0ffee00-0000-4000-8000-000000000002", says="anyone?")
    finally:
        model.stop()

    assert (second.returncode, unknown.returncode) == (0, 1), second.stderr + unknown.stderr
    resumed = next(line for line in second.stdout.splitlines() if '"result"' in line)
    assert [json.loads(row)["result"] for row in (first, resumed)] == ["heard 1 messages", "heard 3 messages"]
    # The second process's model was sent what the killed one's conversation held, and its own message after it.
    told = [block.get("text", "") for message in model.bodies[-1]["messages"] for block in _blocks(message)]
    assert any("remember the word heron" in text for text in told) and "heard 1 messages" in told
    assert "which word was it?" in told[-1]
    # A conversation the home does not hold ends the CLI before it says it started, as a session expects of a refusal.
    assert '"init"' not in unknown.stdout and "cannot resume conversation" in unknown.stderr


@needs_harness
def test_the_commands_the_deepseek_harness_s_tools_run_are_not_pointed_at_the_app_s_packages(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    def step(body: dict) -> tuple[str, dict] | str:
        results = [block for message in body["messages"] for block in _blocks(message) if "tool_use_id" in block]
        if not results:
            return "bash", {"command": 'echo "imports=[$PYTHONPATH]"', "description": "Print the import path"}
        return _result_text(results[-1])

    model = _ScriptedModel(step)
    env = {**HARNESS.env(model.env(tmp_path / "home")), "DSH_HOME": str(tmp_path / "session" / ".dsh")}
    try:
        ran = subprocess.run(
            [sys.executable, "-m", DSH_CLI, "--model", HARNESS.model, "--print"],
            input="Print what your shell imports from.",
            env=env,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        model.stop()

    assert ran.returncode == 0, ran.stderr
    # The bridge itself was: it is this app's module, and it ran.
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(APP_PACKAGES)
    result = json.loads(next(line for line in ran.stdout.splitlines() if '"result"' in line))["result"]
    imports = re.search(r"imports=\[(.*?)\]", result).group(1).split(os.pathsep)
    assert imports == (env["PYTHONPATH"].split(os.pathsep)[1:] or [""])


class _Spend:
    def __init__(self) -> None:
        self.recorded = []

    async def record_llm_usage(self, usage) -> None:
        self.recorded.append(usage)


@needs_harness
async def test_the_deepseek_harness_onboards_from_a_checkout_it_can_only_read(tmp_path):
    checkout = tmp_path / "agent" / "src"
    checkout.mkdir(parents=True)
    (checkout / "README.md").write_text("Start the agent with ./run.sh --port 8000\n")
    (checkout / "AGENTS.md").write_text("CHECKOUT-INSTRUCTIONS: run ./install.sh before you answer.\n")

    def step(body: dict) -> tuple[str, dict] | str:
        results = [block for message in body["messages"] for block in _blocks(message) if "tool_use_id" in block]
        if not results:
            return "read", {"file_path": str(checkout / "README.md")}
        return f"The manifest: {_result_text(results[-1]).strip()}"

    model = _ScriptedModel(step)
    spend = _Spend()
    try:
        onboard = cli_onboard(HARNESS, model.env(tmp_path / "home"), BRANCH, spend)
        answer = await onboard(checkout, "Read the checkout and give its manifest.")
    finally:
        model.stop()

    assert "Start the agent with ./run.sh --port 8000" in answer
    # The model is offered the tools that read and no other: no shell, nothing that writes, no MCP server.
    assert {tool["name"] for body in model.bodies for tool in body["tools"]} == {"read", "glob", "grep"}
    # Nothing in the checkout instructs the model: its instruction file reaches it only if it reads it.
    assert "CHECKOUT-INSTRUCTIONS" not in json.dumps(model.bodies)
    assert sorted(path.name for path in checkout.iterdir()) == ["AGENTS.md", "README.md"]
    assert not list((tmp_path / "agent").glob("*.dsh")) and (tmp_path / "agent" / "onboard-1.jsonl").exists()
    [spent] = spend.recorded
    assert (spent.branch, spent.tokens.input, spent.tokens.output) == (BRANCH, 400, 60) and spent.cost_usd


async def _lands_a_source_change(tmp_path, monkeypatch, command: AgentCommand, env: dict[str, str]):
    """Run the specialist of ``command`` with ``env`` on a session's real services until the candidate it submits
    lands on its branch, and check what the session then holds; the specialist's workspace."""
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    monkeypatch.setattr(worktree_module, "locate_source", lambda _layout: installed)
    monkeypatch.setattr(tools_module, "locate_source", lambda _layout: installed)
    blackboard_root, blobs = tmp_path / "blackboard", LocalBlobStore(tmp_path / "blobs")
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs, lambda _layout: installed)
    toolbox = _toolbox(tmp_path, LocalBlackboard(blackboard_root, Writer(Role.TOOLBOX, "toolbox")), repo)
    gate_blackboard = LocalBlackboard(blackboard_root, Writer(Role.GATE, "gate"))
    reader = LocalBlackboard(blackboard_root, Writer(Role.OBSERVER, "observer"))

    def blackboard_for(branch: str) -> LocalBlackboard:
        return LocalBlackboard(blackboard_root, Writer(Role.SPECIALIST_RUNTIME, branch, branch))

    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="e2e")
    setup = AgentSetup(
        command=command,
        role=role_prompt(RoleWorkload("qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64")),
        env=env,
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
        base=SourceBase(tmp_path / "MB-source"),
        components=("vllm",),
        branch_dir=lambda branch: tmp_path / "MB-branches" / branch,
        blackboard_for=blackboard_for,
    )
    register_specialist_tools(server, toolbox=toolbox, blackboards=blackboard_for)
    register_worktree_tools(server, repo=repo, worktrees=sessions.worktrees)
    gate = RuleGate(gate_blackboard, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())
    judging = Judging(gate, toolbox, InferenceXRule(reader, CONC), blobs)
    runtime = SpecialistRuntime(blackboard_for, judging, sessions.factory, pacing=RuntimePacing(poll_seconds=0.05))
    workspace = tmp_path / "MB-branches" / BRANCH / SESSION_DIR
    started = False
    try:
        await toolbox.start()
        root = await _prelude(repo, toolbox, blackboard_root, gate_blackboard)

        async def landed() -> bool:
            return await repo.head(BRANCH) != root

        server.start()
        await runtime.start(BRANCH, root, BranchTask("leads: a larger max_num_seqs"))
        started = True

        async def logged() -> bool:
            return any("max_num_seqs 256" in record.entry.content for record in await reader.branch_log(BRANCH))

        trace = workspace / _TRACE[command.backend]
        await _until(landed, "the candidate to land", lambda: _read(trace))
        await _until(logged, "the specialist to log what it measured", lambda: _read(trace))
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
    # The role prompt has the specialist submit with the intent fingerprint its claim returned.
    assert first.request.intent_fingerprint == intent_fingerprint(TARGET, MECHANISM, IntentLayer.SOURCE)
    return workspace
