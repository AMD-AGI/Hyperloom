# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the in-process context-pull MCP tool surface."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.backends import mcp_context_tools as mct


def _shared_state():
    """Build a fake SharedState exposing the to_*_summary projections."""
    return SimpleNamespace(
        to_mission_summary=lambda: "mission",
        to_prompt_summary=lambda: "prompt",
        to_gaps_summary=lambda: "gaps",
        to_warm_start_summary=lambda: "warm",
        to_proposal_scores_summary=lambda: "scores",
        to_intervention_mix_summary=lambda: "mix",
        to_policy_denial_summary=lambda top_k=6: f"denials({top_k})",
    )


def test_qualified():
    assert mct._qualified("foo") == "mcp__inference_optimizer_context__foo"


def test_provider_projections():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert p.mission_status() == "mission"
    assert p.shared_state_summary() == "prompt"
    assert p.gaps() == "gaps"
    assert p.warm_start() == "warm"
    assert p.proposal_scores() == "scores"
    assert p.intervention_mix() == "mix"


def test_safe_handles_exception():
    def boom():
        raise RuntimeError("x")

    p = mct.ContextProvider(shared_state=SimpleNamespace())
    out = p._safe(boom, "lbl")
    assert "unavailable" in out


def test_safe_empty_marker():
    p = mct.ContextProvider(shared_state=SimpleNamespace())
    assert p._safe(lambda: "", "lbl") == "(lbl: empty)"
    assert p._safe(lambda: None, "lbl") == "(lbl: empty)"


def test_why_denied_via_shared_state():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert p.why_denied(top_k=3) == "denials(3)"


def test_why_denied_via_reader():
    p = mct.ContextProvider(
        shared_state=_shared_state(), denial_reader=lambda k: f"reader({k})",
    )
    assert p.why_denied(top_k=2) == "reader(2)"


def test_analysis_md_not_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.analysis_md()


def test_analysis_md_wired():
    p = mct.ContextProvider(shared_state=_shared_state(), analysis_reader=lambda: "md")
    assert p.analysis_md() == "md"


def test_inbox_not_wired_and_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.inbox()
    p2 = mct.ContextProvider(shared_state=_shared_state(), inbox_reader=lambda s: f"inbox({s})")
    assert p2.inbox(5) == "inbox(5)"


def test_recent_outcomes_not_wired_and_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.recent_outcomes()
    p2 = mct.ContextProvider(
        shared_state=_shared_state(), recent_outcomes_reader=lambda k: f"out({k})",
    )
    assert p2.recent_outcomes(4) == "out(4)"


def test_run_action_now_not_wired_and_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.run_action_now("a")
    p2 = mct.ContextProvider(
        shared_state=_shared_state(),
        action_runner=lambda name, params: f"ran {name} {params}",
    )
    assert p2.run_action_now("act", {"k": 1}) == "ran act {'k': 1}"


def test_tool_name_tuples():
    assert "get_mission_status" in mct.CONTEXT_TOOL_NAMES
    assert mct.CONTEXT_TOOL_QUALIFIED_NAMES[0].startswith("mcp__")
    assert len(mct.CONTEXT_TOOL_NAMES) == len(mct.CONTEXT_TOOL_SPECS)


# ---- _resolve_sdk ----

def test_resolve_sdk_explicit():
    sentinel = object()
    assert mct._resolve_sdk(sentinel) is sentinel


def test_resolve_sdk_import_error(monkeypatch):
    def boom(_name):
        raise ImportError("no sdk")

    monkeypatch.setattr(mct.importlib, "import_module", boom)
    assert mct._resolve_sdk(None) is None


# ---- _make_handler ----

async def test_make_handler_success():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "mission_status")
    out = await handler({})
    assert out["content"][0]["text"] == "mission"


async def test_make_handler_forwards_kwargs():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "why_denied")
    out = await handler({"top_k": 2})
    assert out["content"][0]["text"] == "denials(2)"


async def test_make_handler_forwards_since_seq():
    p = mct.ContextProvider(shared_state=_shared_state(), inbox_reader=lambda s: f"inbox({s})")
    handler = mct._make_handler(p, "inbox")
    out = await handler({"since_seq": 9})
    assert out["content"][0]["text"] == "inbox(9)"


async def test_make_handler_forwards_action_args():
    p = mct.ContextProvider(
        shared_state=_shared_state(),
        action_runner=lambda name, params: f"{name}:{params}",
    )
    handler = mct._make_handler(p, "run_action_now")
    out = await handler({"action_name": "act", "params": {"k": 1}})
    assert out["content"][0]["text"] == "act:{'k': 1}"


async def test_make_handler_non_str_result():
    provider = SimpleNamespace(foo=lambda **k: {"a": 1})
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert json.loads(out["content"][0]["text"]) == {"a": 1}


async def test_make_handler_exception():
    def boom(**k):
        raise RuntimeError("nope")

    provider = SimpleNamespace(foo=boom)
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert out["is_error"] is True


# ---- build_context_tools_server ----

def test_build_server_unavailable_without_factories():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert mct.build_context_tools_server(p, sdk_module=object()) is None


def test_build_server_with_fake_factories():
    p = mct.ContextProvider(shared_state=_shared_state())
    created = {}

    def tool_factory(name, desc, schema):
        def decorator(handler):
            return (name, handler)

        return decorator

    def server_factory(name, version, tools):
        created["name"] = name
        created["tools"] = tools
        return "SERVER"

    out = mct.build_context_tools_server(
        p, tool_factory=tool_factory, server_factory=server_factory,
    )
    assert out == "SERVER"
    assert created["name"] == mct.MCP_SERVER_NAME
    assert len(created["tools"]) == len(mct.CONTEXT_TOOL_SPECS)
