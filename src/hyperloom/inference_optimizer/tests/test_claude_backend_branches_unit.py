# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for ClaudeBackend: SDK import, __post_init__ wiring, option building (resume / context tools / raw
mode), timeout handling, the conversational session capture, and SDK-stream error tolerance.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.roles import claude as cl
from hyperloom.orchestrator.roles.base import BackendError


# ---- SDK fakes ------------------------------------------------------------
class ToolUseBlock:
    def __init__(self, name, input):
        self.name = name
        self.input = input


class TextBlock:
    def __init__(self, text):
        self.text = text


@dataclass
class _Msg:
    content: list = field(default_factory=list)
    result: str = ""
    usage: dict | None = None
    session_id: str | None = None


def _query(messages, *, raise_exc=None):
    async def _q(*, prompt, options):
        for m in messages:
            yield m
        if raise_exc is not None:
            raise raise_exc

    return _q


def _backend(claude_sdk, messages=None, **over):
    """A backend on the stand-in SDK, whose ``query`` streams ``messages`` until a test assigns another."""
    claude_sdk.query = _query(messages or [])
    return cl.ClaudeBackend(**over)


def _emit_tool_block():
    return ToolUseBlock(
        name=cl.EMIT_INTENT_TOOL_QUALIFIED,
        input={"intent_type": "send_message", "payload": {"topic": "t", "body_md": "ok"}},
    )


# ---- _import_sdk ----------------------------------------------------------
def test_import_sdk_missing(monkeypatch):
    def _raise(name):
        raise ImportError("nope")

    monkeypatch.setattr(cl.importlib, "import_module", _raise)
    with pytest.raises(BackendError, match="claude-agent-sdk not installed"):
        cl._import_sdk()


def test_import_sdk_incomplete(monkeypatch):
    monkeypatch.setattr(cl.importlib, "import_module", lambda name: SimpleNamespace(query=_query([])))
    with pytest.raises(BackendError, match="missing ClaudeAgentOptions / tool / create_sdk_mcp_server"):
        cl._import_sdk()


# ---- __post_init__ --------------------------------------------------------
async def test_backend_runs_the_turn_on_the_sdk_it_imports(claude_sdk):
    seen: list[tuple[str, object]] = []

    async def _recording_query(*, prompt, options):
        seen.append((prompt, options))
        yield _Msg(content=[_emit_tool_block()])

    claude_sdk.query = _recording_query
    b = cl.ClaudeBackend()
    result = await b.run("hi")

    assert len(result.intents) == 1
    assert [type(options) for _prompt, options in seen] == [claude_sdk.ClaudeAgentOptions]
    assert b.mcp_server_config["name"] == cl.MCP_SERVER_NAME
    assert [tool.name for tool in b.mcp_server_config["tools"]] == [cl.EMIT_INTENT_TOOL_NAME]


def test_post_init_emit_intent_setup_failure_fails_construction(claude_sdk, monkeypatch):
    monkeypatch.setattr(cl, "build_emit_intent_server", lambda sdk: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="^boom$"):
        _backend(claude_sdk)


# ---- set_context_provider -------------------------------------------------
def test_set_context_provider_failure_raises(claude_sdk, monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server", lambda sdk, provider: (_ for _ in ()).throw(ValueError("x")))
    b = _backend(claude_sdk)
    with pytest.raises(ValueError, match="^x$"):
        b.set_context_provider(SimpleNamespace())
    assert b._context_server_config is None


# ---- _build_options -------------------------------------------------------
def test_build_options_model_and_system_prompt(claude_sdk):
    b = _backend(claude_sdk, model="claude-x")
    opts = b._build_options(tools=["Read"], system_prompt="sys")
    assert opts.kwargs["model"] == "claude-x"
    assert Path(opts.kwargs["system_prompt"]["path"]).read_text(encoding="utf-8") == "sys"
    assert "resume" not in opts.kwargs


def test_build_options_with_context_tools(claude_sdk, monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server", lambda sdk, provider: SimpleNamespace(name="ctx"))
    b = _backend(claude_sdk)
    b.set_context_provider(SimpleNamespace())
    opts = b._build_options(tools=[], system_prompt=None)
    allowed = opts.kwargs.get("allowed_tools", [])
    for qn in cl.CONTEXT_TOOL_QUALIFIED_NAMES:
        assert qn in allowed
    assert cl.CONTEXT_MCP_SERVER_NAME in opts.kwargs["mcp_servers"]


# ---- CLI stderr -----------------------------------------------------------
async def test_cli_stderr_lines_go_to_the_debug_log(claude_sdk, caplog):
    """The SDK hands each CLI stderr line to the options' callback instead of the operator console."""

    async def _query_writing_stderr(*, prompt, options):
        options.kwargs["stderr"]("  error line  ")
        options.kwargs["stderr"]("   ")
        yield _Msg(content=[TextBlock("hi")], result="hi")

    b = _backend(claude_sdk)
    claude_sdk.query = _query_writing_stderr
    with caplog.at_level(logging.DEBUG, logger=cl.log.name):
        await b.run("hi")
    stderr_records = [r for r in caplog.records if r.getMessage().startswith("claude CLI stderr")]
    assert [(r.levelno, r.getMessage()) for r in stderr_records] == [(logging.DEBUG, "claude CLI stderr: error line")]


# ---- run(): timeout -------------------------------------------------------
async def test_run_timeout(claude_sdk):
    async def _slow_query(*, prompt, options):
        await asyncio.sleep(0.5)
        yield _Msg()

    b = _backend(claude_sdk)
    claude_sdk.query = _slow_query
    b.call_timeout_s = 0.01
    with pytest.raises(BackendError, match="timed out"):
        await b.run("hi")


# ---- run(): idle timeout tolerates a slow-but-live stream -----
async def test_run_idle_timeout_allows_slow_but_live_stream(claude_sdk):
    """A model that keeps streaming (gaps < idle budget) must NOT be killed, even when the TOTAL turn wall-clock
    exceeds ``call_timeout_s``.
    """

    async def _slow_live(*, prompt, options):
        for _ in range(4):
            # Per-message gap stays under the idle budget while cumulative time exceeds it, proving the guard is
            # idle-based, not a total wall-clock cap.
            await asyncio.sleep(0.03)
            yield _Msg(content=[_emit_tool_block()])

    b = _backend(claude_sdk)
    claude_sdk.query = _slow_live
    b.call_timeout_s = 0.05
    res = await b.run("hi")
    assert len(res.intents) == 4


# ---- run(): no intents ----------------------------------------------------
async def test_run_without_intents_returns_an_empty_result(claude_sdk):
    msg = _Msg(content=[TextBlock("just text")], result="hi")
    b = _backend(claude_sdk)
    claude_sdk.query = _query([msg])
    result = await b.run("hi")
    assert result.intents == []
    assert result.raw_text == "hi"


# ---- gateway endpoint identifier -----------------------------------------
def test_gateway_endpoint_drops_url_userinfo(claude_sdk, monkeypatch):
    """The diagnostic is appended to an on-disk trace, and a base URL of the form ``https://user:key@gw/...`` puts the key in netloc."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://user:s3cret@gw.example.com:8443/api/v1")
    assert _backend(claude_sdk)._gateway_endpoint_identifier() == "gw.example.com"


def test_gateway_endpoint_is_none_without_a_base_url(claude_sdk, monkeypatch):
    for var in ("ANTHROPIC_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert _backend(claude_sdk)._gateway_endpoint_identifier() is None


# ---- _invoke_and_collect: error-result-success tolerance ------------------
async def test_invoke_error_result_success_with_intents(claude_sdk):
    msg = _Msg(content=[_emit_tool_block()])
    b = _backend(claude_sdk)
    claude_sdk.query = _query([msg], raise_exc=Exception("error result: success"))
    res = await b.run("hi")
    assert len(res.intents) == 1


async def test_invoke_error_result_success_no_intents(claude_sdk):
    msg = _Msg(content=[TextBlock("hello")])
    b = _backend(claude_sdk)
    claude_sdk.query = _query([msg], raise_exc=Exception("error result: success"))
    res = await b.run("hi")
    assert res.intents == []


async def test_invoke_other_exception_reraises(claude_sdk):
    b = _backend(claude_sdk)
    claude_sdk.query = _query([], raise_exc=RuntimeError("real failure"))
    with pytest.raises(RuntimeError, match="real failure"):
        await b.run("hi")


# ---- per-request context water level -------------------------------------
@dataclass
class _ResultMsg(_Msg):
    num_turns: int = 0


async def test_context_peak_prefers_the_largest_single_request(claude_sdk):
    # Three internal turns, then the terminal cumulative summary.
    stream = [
        _Msg(usage={"input_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 90}),
        _Msg(usage={"input_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 45}),
        _Msg(
            content=[TextBlock("done")],
            usage={"input_tokens": 5, "cache_read_input_tokens": 150, "cache_creation_input_tokens": 20},
        ),
        _ResultMsg(
            result="done",
            num_turns=3,
            usage={"input_tokens": 20, "cache_read_input_tokens": 250, "cache_creation_input_tokens": 155},
        ),
    ]
    b = _backend(claude_sdk)
    claude_sdk.query = _query(stream)
    res = await b.run("hi")
    assert res.metadata["context_tokens_peak"] == 175
    # The cumulative counters are still reported verbatim for cost accounting.
    assert res.metadata["cache_read_input_tokens"] == 250


async def test_context_peak_falls_back_to_a_per_turn_estimate(claude_sdk):
    # Only the terminal cumulative usage is available: divide by the turn count.
    stream = [
        _ResultMsg(
            result="done",
            num_turns=4,
            usage={"input_tokens": 0, "cache_read_input_tokens": 300_000, "cache_creation_input_tokens": 100_000},
        ),
    ]
    b = _backend(claude_sdk)
    claude_sdk.query = _query(stream)
    res = await b.run("hi")
    assert res.metadata["context_tokens_peak"] == 100_000


async def test_context_peak_is_zero_without_any_usage(claude_sdk):
    b = _backend(claude_sdk)
    claude_sdk.query = _query([_Msg(content=[TextBlock("hi")], result="hi")])
    res = await b.run("hi")
    assert res.metadata["context_tokens_peak"] == 0


# ---- stop_reason passthrough ---------------------------------------------
@dataclass
class _StopMsg(_Msg):
    stop_reason: str | None = None


async def test_stop_reason_reaches_metadata(claude_sdk):
    """Without it a truncated reply is indistinguishable from a badly formatted one, so the SDK's own stop reason must survive to the caller."""
    stream = [_StopMsg(content=[TextBlock("half a rep")], result="half a rep", stop_reason="max_tokens")]
    b = _backend(claude_sdk)
    claude_sdk.query = _query(stream)
    res = await b.run("hi")
    assert res.metadata["stop_reason"] == "max_tokens"


async def test_stop_reason_is_none_when_the_sdk_omits_it(claude_sdk):
    b = _backend(claude_sdk)
    claude_sdk.query = _query([_Msg(content=[TextBlock("hi")], result="hi")])
    res = await b.run("hi")
    assert res.metadata["stop_reason"] is None


async def test_last_reported_stop_reason_wins(claude_sdk):
    """Per-message stop reasons stream in order; the terminal one describes the call."""
    stream = [
        _StopMsg(content=[TextBlock("a")], stop_reason="tool_use"),
        _StopMsg(content=[TextBlock("b")], result="ab", stop_reason="end_turn"),
    ]
    b = _backend(claude_sdk)
    claude_sdk.query = _query(stream)
    res = await b.run("hi")
    assert res.metadata["stop_reason"] == "end_turn"


# ---- mcp_emit_intent: handler + validation branches -----------------------

from hyperloom.orchestrator.roles import mcp_emit_intent as mei
from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
)


@pytest.mark.parametrize(
    ("tool_input", "error_match"),
    [
        pytest.param(
            {"intent_type": "send_message", "payload": "nope"}, "'payload' must be an object", id="non-dict-payload"
        ),
        pytest.param(["not", "a", "dict"], "must be an object", id="not-an-object"),
        pytest.param(
            {"intent_type": "send_message", "payload": {}, "extra": 1}, "unexpected keys", id="unexpected-keys"
        ),
        pytest.param({"intent_type": "send_message"}, "requires both", id="missing-top-level-key"),
        # send_message requires 'topic'; the message names the field, which is what tells the agent what to fix.
        pytest.param({"intent_type": "send_message", "payload": {}}, "topic", id="missing-required-payload-field"),
    ],
)
def test_validate_emit_intent_input_rejects(tool_input, error_match):
    with pytest.raises(IntentValidationError, match=error_match):
        mei.validate_emit_intent_input(tool_input)


@pytest.mark.asyncio
async def test_emit_intent_handler_validation_error_returns_is_error():
    res = await mei._emit_intent_handler({"intent_type": "bogus", "payload": {}})
    assert res["is_error"] is True
    assert "validation_error" in res["content"][0]["text"]
