# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What each turn loads: the built-in tools named for it, and a bounded specialist-findings block."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.roles import mcp_context_tools as ctx
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.claude import ClaudeBackend
from hyperloom.orchestrator.specialists.leaf import LEAF_AGENT_TOOLS
from hyperloom.orchestrator.specialists.subprocess_ import (
    SPECIALIST_BUILTIN_TOOLS,
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
)


class _Options:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


async def _no_query(*, prompt, options):
    return
    yield


def _backend(**kw: Any) -> ClaudeBackend:
    return ClaudeBackend(model="m", sdk_query_factory=_no_query, sdk_options_cls=_Options, **kw)


def _orchestration_tools() -> list[str]:
    return PolicyGate(role_registry=default_role_registry()).allowed_tools_for_agent("orchestration")


# --- orchestration / reactor built-ins -----------------------------------------
def test_orchestration_loads_only_the_built_ins_its_policy_names(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_REACTOR_TOOLS", raising=False)
    allowed = _orchestration_tools()
    kw = _backend(effort_role="orchestration")._build_options(tools=allowed, max_turns=4, system_prompt="sp").kwargs
    assert kw["tools"] == ["Read", "WebSearch", "WebFetch"]
    assert "Read" in kw["allowed_tools"]


def test_a_caller_naming_no_built_in_keeps_the_full_set(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_REACTOR_TOOLS", raising=False)
    kw = _backend()._build_options(tools=["emit_intent"], max_turns=4, system_prompt="sp").kwargs
    assert "tools" not in kw


def test_raw_completion_loads_no_built_in(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_REACTOR_TOOLS", raising=False)
    kw = _backend(raw_completion=True)._build_options(tools=[], max_turns=1, system_prompt="sp").kwargs
    assert kw["tools"] == []
    assert kw["allowed_tools"] == []


def test_reactor_tools_all_restores_the_full_set(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_REACTOR_TOOLS", "all")
    allowed = _orchestration_tools()
    assert "tools" not in _backend()._build_options(tools=allowed, max_turns=4, system_prompt="sp").kwargs
    assert "tools" not in _backend(raw_completion=True)._build_options(tools=[], max_turns=1, system_prompt="sp").kwargs


def test_the_pinned_sdk_accepts_a_tools_list(monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    monkeypatch.delenv("HYPERLOOM_REACTOR_TOOLS", raising=False)
    backend = ClaudeBackend(
        model="m", sdk_query_factory=_no_query, sdk_options_cls=sdk.ClaudeAgentOptions, enable_mcp_emit_intent=False
    )
    options = backend._build_options(tools=["Read", "WebSearch"], max_turns=4, system_prompt="sp")
    assert options.tools == ["Read", "WebSearch"]


# --- specialist built-ins ------------------------------------------------------
def _specialist_cmd(tmp_path: Path, **cfg: Any) -> list[str]:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return SpecialistSubprocessDispatcher(SpecialistSubprocessConfig(**cfg))._build_claude_cmd(
        system_prompt_file=ws / "system_prompt.md",
        system_prompt="SYS",
        workspace=ws,
        worktree=None,
        disallowed_tools=frozenset({"KillShell", "SlashCommand"}),
    )


def test_specialists_load_the_tools_they_call(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_TOOLS", raising=False)
    cmd = _specialist_cmd(tmp_path)
    assert cmd[cmd.index("--tools") + 1].split(",") == list(SPECIALIST_BUILTIN_TOOLS)
    assert set(LEAF_AGENT_TOOLS) <= set(SPECIALIST_BUILTIN_TOOLS), "a leaf only gets tools its parent loaded"


def test_specialist_tools_env_overrides_or_restores_the_full_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_TOOLS", "all")
    assert "--tools" not in _specialist_cmd(tmp_path)
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_TOOLS", "Bash, Read,KillShell")
    cmd = _specialist_cmd(tmp_path)
    assert cmd[cmd.index("--tools") + 1] == "Bash,Read", "a denied tool is never loaded"


def test_an_explicit_config_list_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_TOOLS", raising=False)
    assert "--tools" not in _specialist_cmd(tmp_path, builtin_tools=())
    cmd = _specialist_cmd(tmp_path, builtin_tools=("Read",))
    assert cmd[cmd.index("--tools") + 1] == "Read"


# --- specialist findings block --------------------------------------------------
def _finding(i: int, size: int = 900) -> dict[str, Any]:
    return {"what": f"finding-{i} " + "x" * size, "status": "proposed"}


def _collab(tmp_path: Path, rounds: list[dict[str, Any]]) -> SimpleNamespace:
    fake = SimpleNamespace(session_dir=tmp_path, shared_state=SimpleNamespace(specialist_rounds=rounds))
    fake._specialist_findings_parts = lambda: ConversationCollaborator._specialist_findings_parts(fake)
    return fake


def _rounds(n: int, per_round: int = 3, questions: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "domain": f"d{r}",
            "new_findings": [_finding(r * 10 + i) for i in range(per_round)],
            "residual_questions": [f"q{r}-{i} " + "y" * 200 for i in range(questions)],
        }
        for r in range(n)
    ]


def test_a_small_block_renders_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_FINDINGS_PROMPT_CHARS", raising=False)
    fake = _collab(tmp_path, _rounds(2, per_round=1, questions=1))
    bounded = ConversationCollaborator._specialist_findings_block(fake)
    monkeypatch.setenv("HYPERLOOM_FINDINGS_PROMPT_CHARS", "0")
    assert bounded == ConversationCollaborator._specialist_findings_block(fake)
    assert "not shown" not in bounded


def test_a_large_block_keeps_the_newest_within_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_FINDINGS_PROMPT_CHARS", "6000")
    fake = _collab(tmp_path, _rounds(12))
    block = ConversationCollaborator._specialist_findings_block(fake)
    assert block.startswith("=== Specialist findings ===")
    assert len(block) <= 6000 + 200
    assert "finding-110" in block, "the newest round comes first"
    assert "finding-0 " not in block
    assert "Residual questions:" in block
    footer = block.splitlines()[-1]
    assert "older findings" in footer and "get_specialist_findings" in footer
    monkeypatch.setenv("HYPERLOOM_FINDINGS_PROMPT_CHARS", "0")
    assert len(ConversationCollaborator._specialist_findings_block(fake)) > 30_000


def test_the_tool_pages_and_filters_every_finding_unclipped(tmp_path):
    fake = _collab(tmp_path, _rounds(12))
    first = ConversationCollaborator._context_findings_reader(fake)
    assert first.splitlines()[0] == "findings 1-10 of 36"
    assert "(more: offset=10)" in first and "Residual questions:" in first
    assert json.dumps(_finding(110), sort_keys=True) in first, "unclipped"
    last = ConversationCollaborator._context_findings_reader(fake, offset=30, limit=10)
    assert last.splitlines()[0] == "findings 31-36 of 36" and "more:" not in last
    one = ConversationCollaborator._context_findings_reader(fake, domain="d3")
    assert one.splitlines()[0] == "findings 1-3 of 3 in domains matching 'd3'"
    assert "[d3] q3-0" in one and "[d4]" not in one


async def test_get_specialist_findings_is_a_context_tool():
    assert "get_specialist_findings" in ctx.CONTEXT_TOOL_NAMES
    assert "get_specialist_findings" in _orchestration_tools()
    seen: list[tuple[str, int, int]] = []

    def reader(domain: str, offset: int, limit: int) -> str:
        seen.append((domain, offset, limit))
        return "page"

    provider = ctx.ContextProvider(shared_state=SimpleNamespace(), findings_reader=reader)
    handler = ctx._make_handler(provider, "specialist_findings")
    out = await handler({"domain": "serving", "offset": 10, "limit": 5})
    assert out["content"][0]["text"] == "page"
    assert seen == [("serving", 10, 5)]
    assert ctx.ContextProvider(shared_state=SimpleNamespace()).specialist_findings() == (
        "(specialist findings reader not wired)"
    )
