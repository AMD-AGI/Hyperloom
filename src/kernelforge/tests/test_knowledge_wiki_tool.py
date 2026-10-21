"""The read-only knowledge base reaches every forge role that has tools, and only when configured."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.config import Config
from kernelforge.knowledge.wiki_tool import LOG_NAME, SERVER_KEY, KnowledgeWikiTool, require_mcp_backend
from kernelforge.knowledge.wiki_tool import mcp_server as server
from kernelforge.orchestrator import agent as agent_module
from kernelforge.orchestrator import orchestration as orchestration_module
from kernelforge.orchestrator.contracts import (
    CaseEvidence,
    DispatchPlan,
    OrchestrationContext,
    SpecialistAssignment,
    SpecialistDefinition,
    SpecialistRound,
    SynthesizedPlan,
)
from kernelforge.orchestrator.analysis import AnalysisSources
from kernelforge.orchestrator.orchestration import OrchestrationAgent, make_orchestration_service
from kernelforge.orchestrator.plan_critic import PlanCriticAgent
from kernelforge.orchestrator.specialists import SpecialistAgent, SpecialistTools
from kernelforge.orchestrator.supervisor import make_supervisor_fn
from kernelforge.tests import test_analysis_agent as analysis_tests

TOOL_NAMES = (
    "mcp__knowledge_wiki__search",
    "mcp__knowledge_wiki__page",
    "mcp__knowledge_wiki__grep",
    "mcp__knowledge_wiki__guide",
)
# InstinctWiki's output shapes.
SEARCH_OUTPUT = "# 2 result(s)\n\n## MFMA\n- **id**: `technique-mfma-pipelining`\n\n## Fix\n- **id**: `pr-aiter-12`"
GREP_OUTPUT = "# 1 file(s) match\n\n## sources/docs/doc-llvm-amdgpu.md  (1 matches)\n```\n22: global_load_lds\n```"


def _script(output: str, *, fail_on: str = "") -> str:
    """A knowledge base script that records its argv and prints a fixed answer."""
    return (
        "import json, pathlib, sys\n"
        "with pathlib.Path(__file__).with_name('argv.jsonl').open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"if {fail_on!r} and sys.argv[-1] == {fail_on!r}:\n"
        "    sys.exit('no such page: ' + sys.argv[-1])\n"
        f"print({output!r})\n"
    )


def make_wiki(root: Path, *, guide: str | None = "# Guide\nSearch first, then read pages.\n") -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "query.py").write_text(_script(SEARCH_OUTPUT))
    (scripts / "get_page.py").write_text(_script("# page body", fail_on="missing"))
    (scripts / "grep_wiki.py").write_text(_script(GREP_OUTPUT))
    if guide is not None:
        (root / "SKILL.md").write_text(guide)
    return root


def _argv(wiki: Path) -> list[list[str]]:
    path = wiki / "scripts" / "argv.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def wiki(tmp_path) -> Path:
    return make_wiki(tmp_path / "wiki")


@pytest.fixture
def log(tmp_path) -> Path:
    return tmp_path / "logs" / LOG_NAME


def _server(wiki: Path, log: Path, role: str = "specialist memory") -> server.KnowledgeWikiServer:
    return server.KnowledgeWikiServer({server.ROOT_ENV: str(wiki), server.LOG_ENV: str(log), server.ROLE_ENV: role})


def _call(srv: server.KnowledgeWikiServer, name: str, arguments: dict) -> dict:
    return asyncio.run(srv.handle_tool_call(name, arguments))


# The server ------------------------------------------------------------------------------------------------------


def test_tool_definitions_match_the_declared_names():
    assert [d["name"] for d in server.TOOL_DEFINITIONS] == list(server.TOOL_NAMES)
    assert all(d["description"].strip() and d["inputSchema"]["type"] == "object" for d in server.TOOL_DEFINITIONS)
    # Every tool, guide included, requires a reason so the call log can be read as a trajectory.
    for definition in server.TOOL_DEFINITIONS:
        schema = definition["inputSchema"]
        assert "reason" in schema["properties"] and "reason" in schema["required"]


def test_search_runs_the_wiki_script_and_logs_the_role_and_returned_pages(wiki, log):
    reply = _call(
        _server(wiki, log),
        "search",
        {"query": "--help fp8", "tag": "fp8", "limit": 3, "reason": "picking the fp8 MFMA direction"},
    )

    assert reply["isError"] is False
    assert "technique-mfma-pipelining" in reply["content"][0]["text"]
    # Agent text follows "--", so a leading dash stays a query word; the reason never reaches the script.
    assert _argv(wiki) == [["--tag", "fp8", "--limit", "3", "--", "--help fp8"]]
    [record] = _log(log)
    assert record["role"] == "specialist memory"
    assert record["command"] == "search"
    assert record["reason"] == "picking the fp8 MFMA direction"
    assert record["args"] == {
        "query": "--help fp8",
        "tag": "fp8",
        "limit": 3,
        "reason": "picking the fp8 MFMA direction",
    }
    assert record["exit_code"] == 0
    assert record["results"] == ["technique-mfma-pipelining", "pr-aiter-12"]


def test_page_and_grep_pass_their_arguments_through(wiki, log):
    srv = _server(wiki, log)
    page = _call(srv, "page", {"page_id": "kernel-gemm-dense-matmul", "reason": "reuse the dense GEMM tiling"})
    grep = _call(srv, "grep", {"pattern": "global_load_lds", "only": "sources", "reason": "check direct-to-LDS loads"})

    assert page["isError"] is False and grep["isError"] is False
    assert _argv(wiki) == [["--", "kernel-gemm-dense-matmul"], ["--only", "sources", "--", "global_load_lds"]]
    assert [r["results"] for r in _log(log)] == [["kernel-gemm-dense-matmul"], ["sources/docs/doc-llvm-amdgpu.md"]]
    assert [r["reason"] for r in _log(log)] == ["reuse the dense GEMM tiling", "check direct-to-LDS loads"]


@pytest.mark.parametrize(
    ("name", "output", "expected"),
    [
        ("search", "  [wiki-technique] technique-mfma: MFMA  (wiki/techniques/mfma.md)", ["technique-mfma"]),
        ("search", SEARCH_OUTPUT, ["technique-mfma-pipelining", "pr-aiter-12"]),
        ("grep", "=== sources/docs/doc-llvm-amdgpu.md (1 match) ===\n>> 22: x", ["sources/docs/doc-llvm-amdgpu.md"]),
        ("grep", GREP_OUTPUT, ["sources/docs/doc-llvm-amdgpu.md"]),
    ],
)
def test_returned_pages_are_read_from_both_wiki_output_formats(name, output, expected):
    assert server.returned_items(name, {}, output) == expected


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search", {"reason": "r"}),
        ("search", {"query": "   ", "reason": "r"}),
        ("search", {"query": "two\nlines", "reason": "r"}),
        ("search", {"query": "x", "limit": 0, "reason": "r"}),
        ("search", {"query": "x", "limit": True, "reason": "r"}),
        ("search", {"query": "x", "tag": "two words", "reason": "r"}),
        ("search", {"query": "x", "architecture": "--all", "reason": "r"}),
        ("page", {"reason": "r"}),
        ("grep", {"pattern": "x", "only": "all", "reason": "r"}),
        ("nonexistent", {"reason": "r"}),
        # A call that omits or malforms its reason is refused before anything runs, on every tool.
        ("search", {"query": "x"}),
        ("page", {"page_id": "p"}),
        ("grep", {"pattern": "x"}),
        ("guide", {}),
        ("search", {"query": "x", "reason": "   "}),
        ("search", {"query": "x", "reason": "two\nlines"}),
    ],
)
def test_invalid_calls_are_refused_before_anything_runs(wiki, log, name, arguments):
    with pytest.raises(server.InvalidParamsError):
        _call(_server(wiki, log), name, arguments)
    assert _argv(wiki) == [] and _log(log) == []


def test_a_failed_lookup_is_an_error_that_carries_the_scripts_message(wiki, log):
    reply = _call(_server(wiki, log), "page", {"page_id": "missing", "reason": "look for a prior attempt"})

    assert reply["isError"] is True
    assert "no such page: missing" in reply["content"][0]["text"]
    [record] = _log(log)
    assert record["exit_code"] == 1 and record["results"] == []


def test_guide_returns_the_wikis_own_usage_guide(tmp_path, log):
    with_guide = make_wiki(tmp_path / "with")
    without = make_wiki(tmp_path / "without", guide=None)

    present = _call(_server(with_guide, log), "guide", {"reason": "learn how to cite pages"})
    absent = _call(_server(without, log), "guide", {"reason": "learn how to cite pages"})

    assert present["isError"] is False and "Search first" in present["content"][0]["text"]
    assert absent["isError"] is True and "no usage guide" in absent["content"][0]["text"]
    assert [r["results"] for r in _log(log)] == [["SKILL.md"], []]


def test_an_unusable_checkout_answers_every_call_with_an_error(tmp_path, log):
    srv = _server(tmp_path / "not-a-wiki", log)
    reply = _call(srv, "search", {"query": "mfma", "reason": "start the search"})

    assert reply["isError"] is True and "not a knowledge base checkout" in reply["content"][0]["text"]
    assert _log(log)[0]["exit_code"] == 2


def test_long_output_is_cut_and_says_so(wiki, log, monkeypatch):
    monkeypatch.setattr(server, "MAX_OUTPUT_CHARS", 40)
    reply = _call(_server(wiki, log), "grep", {"pattern": "x", "reason": "scan for a token"})

    assert "output truncated to 40 of" in reply["content"][0]["text"]
    assert _log(log)[0]["truncated"] is True


def test_the_session_server_entry_starts_and_serves_a_real_stdio_session(wiki, log):
    entry = KnowledgeWikiTool(root=wiki, log_path=log).mcp_servers("analysis")[SERVER_KEY]
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "mfma", "reason": "start the search"}},
        },
    ]
    # Only the entry's own environment, as an MCP client that forwards nothing else would start it.
    result = subprocess.run(
        [entry.command, *entry.args],
        input="".join(json.dumps(request) + "\n" for request in requests),
        env=entry.env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    replies = {reply["id"]: reply for reply in map(json.loads, result.stdout.splitlines())}
    assert replies[1]["result"]["serverInfo"]["name"] == server.SERVER_NAME
    assert [tool["name"] for tool in replies[2]["result"]["tools"]] == list(server.TOOL_NAMES)
    assert "technique-mfma-pipelining" in replies[3]["result"]["content"][0]["text"]
    assert _log(log)[0]["role"] == "analysis"


# Configuration -----------------------------------------------------------------------------------------------------


def test_the_knowledge_base_is_off_by_default(monkeypatch):
    monkeypatch.delenv("FORGE_KNOWLEDGE_WIKI_ROOT", raising=False)
    monkeypatch.delenv("FORGE_KNOWLEDGE_WIKI_LOG", raising=False)
    config = Config.from_env()

    assert config.knowledge_wiki_root == ""
    assert KnowledgeWikiTool.from_config(config) is None


def test_the_environment_names_the_checkout_and_the_log(monkeypatch, wiki, log):
    monkeypatch.setenv("FORGE_KNOWLEDGE_WIKI_ROOT", str(wiki))
    monkeypatch.setenv("FORGE_KNOWLEDGE_WIKI_LOG", str(log))
    tool = KnowledgeWikiTool.from_config(Config.from_env())

    assert tool == KnowledgeWikiTool(root=wiki, log_path=log)


def test_the_log_defaults_into_the_experiments_directory(wiki, tmp_path):
    config = Config(knowledge_wiki_root=str(wiki))
    config.experiments_dir = tmp_path / "experiments"

    assert KnowledgeWikiTool.from_config(config).log_path == tmp_path / "experiments" / LOG_NAME


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"knowledge_wiki_root": "relative/wiki"}, "must be an absolute path"),
        ({"knowledge_wiki_root": "/nonexistent/wiki"}, "missing scripts/query.py"),
        ({"knowledge_wiki_log": "relative.jsonl"}, "knowledge_wiki_log must be an absolute path"),
    ],
)
def test_unusable_settings_fail_before_any_session_starts(fields, message):
    with pytest.raises(ValueError, match=message):
        Config(**fields)


def test_each_session_entry_names_its_role_and_forwards_the_launch_environment(wiki, log, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/forge-engine")
    tool = KnowledgeWikiTool(root=wiki, log_path=log)
    entry = tool.mcp_servers("supervisor")[SERVER_KEY]

    assert tool.tool_names == TOOL_NAMES == entry.tools
    assert entry.args == ("-m", "kernelforge.knowledge.wiki_tool.mcp_server")
    assert entry.env[server.ROOT_ENV] == str(wiki)
    assert entry.env[server.LOG_ENV] == str(log)
    assert entry.env[server.ROLE_ENV] == "supervisor"
    assert entry.env["PYTHONPATH"] == "/opt/forge-engine"


def test_a_backend_without_mcp_refuses_a_configured_knowledge_base(wiki, log):
    tool = KnowledgeWikiTool(root=wiki, log_path=log)
    without = SimpleNamespace(name="plain", capabilities=AgentCapabilities(mcp=False))

    require_mcp_backend(None, without, "analysis")
    with pytest.raises(ValueError, match="MCP-capable backend"):
        require_mcp_backend(tool, without, "analysis")


# Every role that has tools ---------------------------------------------------------------------------------------


class _RecordingBackend:
    name = "fake"
    capabilities = AgentCapabilities(mcp=True)

    def __init__(self, text: str = "Widen the block.") -> None:
        self.text = text
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        return AgentRunResult(text=self.text, end_reason="agent_stopped")


def _context(workspace: Path) -> OrchestrationContext:
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace=str(workspace),
        gpu_target="gfx942",
        objective="equal-weight mean case speedup",
        program_context="Optimize the operator.",
        source_map_path="analysis/abc123/source_map.json",
        cases=(
            CaseEvidence(
                case_id="case-a",
                latency_ms=1.0,
                bottleneck="memory",
                profile_summary_path="analysis/abc123/profiles/case-a/summary.json",
            ),
        ),
    )


def _assert_offered(spec, tool: KnowledgeWikiTool, role: str) -> None:
    assert spec.mcp_servers[SERVER_KEY].env[server.ROLE_ENV] == role
    assert spec.mcp_servers[SERVER_KEY].env[server.ROOT_ENV] == str(tool.root)


def _specialist(backend, tool) -> SpecialistAgent:
    return SpecialistAgent(
        definition=SpecialistDefinition(
            role_id="memory",
            description="Memory optimization specialist",
            instructions="Analyze memory layout and cache behavior.",
            capabilities=("memory",),
        ),
        backend=backend,
        timeout_sec=60,
        max_turns=5,
        tools=SpecialistTools(knowledge_wiki=tool),
    )


@pytest.mark.parametrize("configured", [True, False])
async def test_a_specialist_without_a_shell_gets_the_knowledge_base_tools(tmp_path, wiki, log, configured):
    tool = KnowledgeWikiTool(root=wiki, log_path=log) if configured else None
    backend = _RecordingBackend()
    assignment = SpecialistAssignment(
        assignment_id="memory-1",
        role_id="memory",
        target_case_ids=("case-a",),
        evidence_refs=(),
        reason="Decide the block width.",
    )

    await _specialist(backend, tool).run(assignment, _context(tmp_path))

    [spec] = backend.specs
    assert spec.tool_policy.shell is False
    if configured:
        _assert_offered(spec, tool, "specialist memory")
        assert set(TOOL_NAMES) <= set(spec.tool_policy.extra_tools)
    else:
        assert SERVER_KEY not in spec.mcp_servers
        assert not set(TOOL_NAMES) & set(spec.tool_policy.extra_tools)


def test_planning_roles_refuse_a_backend_that_would_drop_the_tools(wiki, log):
    tool = KnowledgeWikiTool(root=wiki, log_path=log)
    backend = SimpleNamespace(name="plain", capabilities=AgentCapabilities(mcp=False))

    with pytest.raises(ValueError, match="specialist memory"):
        _specialist(backend, tool)
    with pytest.raises(ValueError, match="orchestration"):
        OrchestrationAgent(backend=backend, timeout_sec=60, max_turns=5, knowledge_wiki=tool)
    with pytest.raises(ValueError, match="plan critic"):
        PlanCriticAgent(backend=backend, timeout_sec=60, knowledge_wiki=tool)


def test_orchestration_turns_get_the_tools_except_tool_less_formatting_turns(tmp_path, wiki, log):
    tool = KnowledgeWikiTool(root=wiki, log_path=log)
    agent = OrchestrationAgent(backend=_RecordingBackend(), timeout_sec=60, max_turns=5, knowledge_wiki=tool)
    turn = dict(system_prompt="s", user_prompt="u", role="orchestration dispatch")

    with_tools = agent._run_spec(_context(tmp_path), orchestration_module._Turn(**turn), timeout_sec=60)
    tool_less = agent._run_spec(_context(tmp_path), orchestration_module._Turn(**turn, tools=False), timeout_sec=60)

    _assert_offered(with_tools, tool, "orchestration dispatch")
    assert with_tools.tool_policy.extra_tools == TOOL_NAMES
    assert tool_less.mcp_servers == {} and tool_less.tool_policy.extra_tools == ()


async def test_the_plan_critic_review_gets_the_tools(tmp_path, wiki, log):
    tool = KnowledgeWikiTool(root=wiki, log_path=log)
    backend = _RecordingBackend("VERDICT: ACCEPT\n\nThe plan is evidence-grounded.")
    critic = PlanCriticAgent(backend=backend, timeout_sec=60, max_turns=4, knowledge_wiki=tool)

    await critic.review(
        context=_context(tmp_path),
        drafts=[SynthesizedPlan(text="# Plan\nUse vector loads.")],
        specialists=SpecialistRound(DispatchPlan(analysis_commit="abc123", assignments=()), (), {}),
    )

    _assert_offered(backend.specs[0], tool, "plan critic")


def test_the_supervisor_gets_the_tools(tmp_path, wiki, log, monkeypatch):
    backend = _RecordingBackend("RULING: keep going")

    def factory(runtime, **_kwargs):
        backend.runtime = runtime
        return backend

    monkeypatch.setattr("kernelforge.agent_backends.registry.create_registered_backend", factory)
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-test",
        agent_precheck=False,
        knowledge_wiki_root=str(wiki),
        knowledge_wiki_log=str(log),
    )

    asyncio.run(make_supervisor_fn(config=config)(digest="stalled", reason="plateau", workspace=str(tmp_path)))

    _assert_offered(backend.specs[0], KnowledgeWikiTool.from_config(config), "supervisor")


def test_a_supervisor_backend_without_mcp_is_refused_on_every_pass(tmp_path, wiki, log, monkeypatch, capsys):
    backend = _RecordingBackend("RULING: keep going")
    backend.capabilities = AgentCapabilities(mcp=False)

    def factory(runtime, **_kwargs):
        backend.runtime = runtime
        return backend

    monkeypatch.setattr("kernelforge.agent_backends.registry.create_registered_backend", factory)
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-test",
        agent_precheck=False,
        knowledge_wiki_root=str(wiki),
        knowledge_wiki_log=str(log),
    )
    supervisor = make_supervisor_fn(config=config)

    for _ in range(2):
        asyncio.run(supervisor(digest="stalled", reason="plateau", workspace=str(tmp_path)))

    # That backend would drop the knowledge base, so no pass may run on it, the first or any later one.
    assert backend.specs == []
    assert capsys.readouterr().err.count("needs an MCP-capable backend") == 2


async def test_the_analysis_agent_gets_the_tools(tmp_path, wiki, log):
    class Backend(analysis_tests._BundleBackend):
        capabilities = AgentCapabilities(mcp=True)

    workspace, kernel, driver = analysis_tests._workspace(tmp_path)
    backend = Backend()
    service = analysis_tests._service(tmp_path, backend, knowledge_wiki_root=str(wiki), knowledge_wiki_log=str(log))

    await service.ensure_bundle(
        analysis_tests._context(workspace),
        sources=AnalysisSources(kernel_file=str(kernel), driver_script=str(driver), source_files=[str(kernel)]),
    )

    _assert_offered(backend.specs[0], service.knowledge_wiki, "analysis")
    assert backend.specs[0].tool_policy.extra_tools == TOOL_NAMES


def test_the_implementer_gets_the_tools_beside_its_own(tmp_path, wiki, log, monkeypatch):
    captured = {}

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities(mcp=True)

        async def run(self, spec, usage=None) -> AgentRunResult:
            captured["spec"] = spec.resolved(self.runtime)
            return AgentRunResult(text="PLAN: vectorize loads")

    def factory(runtime, **_kwargs):
        backend = Backend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(agent_module, "create_registered_backend", factory)
    kernel = tmp_path / "kernel.py"
    kernel.write_text("VALUE = 1\n")
    driver = tmp_path / "forge_driver.py"
    driver.write_text("print('allclose: True')\n")
    for args in (["init", "-q"], ["config", "user.name", "t"], ["config", "user.email", "t@example.invalid"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "base"], cwd=tmp_path, check=True)
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-test",
        agent_precheck=False,
        knowledge_wiki_root=str(wiki),
        knowledge_wiki_log=str(log),
    )

    agent_fn = agent_module.make_agent_fn(
        config,
        agent_module.ImplementerSetup(
            agent_module.ImplementerTask(program_md="Optimize VALUE."),
            measurement=agent_module.GateMeasurement(driver_script=str(driver)),
            session=agent_module.SessionOptions(agent_backend="codex", insession_gate=True),
        ),
    )
    asyncio.run(agent_fn(str(kernel), "", session_sink={}))

    spec = captured["spec"]
    _assert_offered(spec, KnowledgeWikiTool.from_config(config), "implementer")
    assert spec.tool_policy.shell is True
    assert "pr_monitor" not in spec.mcp_servers


def test_the_planning_factory_hands_every_role_the_same_knowledge_base(wiki, log, monkeypatch):
    runtime = SimpleNamespace(timeout_sec=1800)

    def factory(selected_runtime, **_kwargs):
        return SimpleNamespace(runtime=runtime, name="test", capabilities=AgentCapabilities(mcp=True))

    monkeypatch.setattr(orchestration_module, "create_registered_backend", factory)
    service = make_orchestration_service(
        config=SimpleNamespace(
            agent_runtime=lambda: runtime,
            workspace="/workspace",
            max_turns=500,
            specialist_probe=False,
            specialist_probe_max=6,
            specialist_probe_budget_sec=600.0,
            specialist_probe_scratch_root="",
            knowledge_wiki_root=str(wiki),
            knowledge_wiki_log=str(log),
        ),
        enable_plan_critic=True,
    )

    expected = KnowledgeWikiTool(root=wiki, log_path=log)
    assert service._agent.knowledge_wiki == expected
    assert service._plan_critic.knowledge_wiki == expected
    assert [agent.knowledge_wiki for agent in service._specialist_pool._agents.values()] == [expected] * len(
        service._definitions
    )
