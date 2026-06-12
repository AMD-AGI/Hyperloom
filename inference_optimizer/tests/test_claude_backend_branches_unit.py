# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch coverage for ClaudeBackend: SDK import, __post_init__ wiring,
option building (resume / context tools / raw mode), timeout handling, the
conversational session capture, and the SDK-stream error tolerance."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import claude as cl
from inference_optimizer.orchestrator.backends.base import BackendError
from inference_optimizer.protocol.intent import NoIntentEmitted


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


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _query(messages, *, raise_exc=None):
    async def _q(*, prompt, options):
        for m in messages:
            yield m
        if raise_exc is not None:
            raise raise_exc
    return _q


def _backend(messages=None, **over):
    kwargs = dict(
        sdk_query_factory=_query(messages or []),
        sdk_options_cls=_FakeOptions,
        api_key_env="UNSET_KEY_ENV_FOR_TEST",
    )
    kwargs.update(over)
    return cl.ClaudeBackend(**kwargs)


def _emit_tool_block():
    return ToolUseBlock(
        name=cl.EMIT_INTENT_TOOL_QUALIFIED,
        input={"intent_type": "send_message",
               "payload": {"topic": "t", "body_md": "ok"}},
    )


# ---- _import_sdk ----------------------------------------------------------
def test_import_sdk_missing(monkeypatch):
    def _raise(name):
        raise ImportError("nope")

    monkeypatch.setattr(cl.importlib, "import_module", _raise)
    with pytest.raises(BackendError, match="claude-agent-sdk not installed"):
        cl._import_sdk()


def test_import_sdk_incomplete(monkeypatch):
    monkeypatch.setattr(cl.importlib, "import_module",
                        lambda name: SimpleNamespace())  # no query/options
    with pytest.raises(BackendError, match="missing query"):
        cl._import_sdk()


def test_import_sdk_ok(monkeypatch):
    fake = SimpleNamespace(query=lambda: None, ClaudeAgentOptions=object)
    monkeypatch.setattr(cl.importlib, "import_module", lambda name: fake)
    q, opts, mod = cl._import_sdk()
    assert opts is object
    assert mod is fake


# ---- __post_init__ via real _import_sdk seam ------------------------------
def test_post_init_imports_sdk(monkeypatch):
    fake_q = _query([])
    fake = SimpleNamespace(query=fake_q, ClaudeAgentOptions=_FakeOptions)
    monkeypatch.setattr(cl, "_import_sdk", lambda: (fake_q, _FakeOptions, fake))
    b = cl.ClaudeBackend(api_key_env="UNSET_KEY_ENV_FOR_TEST")
    assert b.sdk_options_cls is _FakeOptions
    assert b.sdk_module is fake


def test_post_init_emit_intent_setup_failure(monkeypatch):
    monkeypatch.setattr(cl, "build_emit_intent_server",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
    b = _backend()
    assert b.has_emit_intent_tool is False
    assert any("emit_intent MCP setup failed" in c.get("warn", "")
               for c in b.calls)


def test_post_init_conversational_floors(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC", raising=False)
    b = _backend(conversational=True, max_turns_default=2)
    assert b.max_turns_default >= cl._CONVERSATIONAL_MIN_MAX_TURNS
    assert b.call_timeout_s >= cl._CONVERSATIONAL_DEFAULT_TIMEOUT_SEC


# ---- set_context_provider -------------------------------------------------
def test_set_context_provider_success(monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server",
                        lambda provider, **k: SimpleNamespace(name="ctx"))
    b = _backend()
    b.set_context_provider(SimpleNamespace())
    assert b.has_context_tools is True


def test_set_context_provider_failure(monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server",
                        lambda provider, **k: (_ for _ in ()).throw(ValueError("x")))
    b = _backend()
    b.set_context_provider(SimpleNamespace())
    assert b.has_context_tools is False
    assert any("context tools MCP setup failed" in c.get("warn", "")
               for c in b.calls)


# ---- _build_options -------------------------------------------------------
def test_build_options_model_and_resume():
    b = _backend(model="claude-x")
    opts = b._build_options(tools=["Read"], max_turns=5,
                            system_prompt="sys", resume_session_id="sess-1")
    assert opts.kwargs["model"] == "claude-x"
    assert opts.kwargs["system_prompt"] == "sys"
    assert opts.kwargs["resume"] == "sess-1"


def test_build_options_raw_completion():
    b = _backend(raw_completion=True)
    opts = b._build_options(tools=["Read"], max_turns=8, system_prompt=None)
    assert opts.kwargs["allowed_tools"] == []
    assert "Bash" in opts.kwargs["disallowed_tools"]


def test_build_options_with_context_tools(monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server",
                        lambda provider, **k: SimpleNamespace(name="ctx"))
    b = _backend()
    b.set_context_provider(SimpleNamespace())
    opts = b._build_options(tools=[], max_turns=3, system_prompt=None)
    allowed = opts.kwargs.get("allowed_tools", [])
    for qn in cl.CONTEXT_TOOL_QUALIFIED_NAMES:
        assert qn in allowed
    assert cl.CONTEXT_MCP_SERVER_NAME in opts.kwargs["mcp_servers"]


# ---- _instantiate_options resume fallback ---------------------------------
def test_instantiate_options_resume_fallback():
    class _PickyOptions:
        def __init__(self, **kwargs):
            if "resume" in kwargs:
                raise TypeError("unexpected kwarg resume")
            self.kwargs = kwargs

    b = _backend(sdk_options_cls=_PickyOptions)
    opts = b._instantiate_options({"max_turns": 4, "resume": "s"})
    assert "resume" not in opts.kwargs
    assert any("rejected resume" in c.get("warn", "") for c in b.calls)


def test_instantiate_options_typeerror_no_resume():
    class _Boom:
        def __init__(self, **kwargs):
            raise TypeError("other error")

    b = _backend(sdk_options_cls=_Boom)
    with pytest.raises(TypeError):
        b._instantiate_options({"max_turns": 4})


# ---- _stderr_sink ---------------------------------------------------------
def test_stderr_sink():
    b = _backend()
    b._stderr_sink("  error line  ")
    b._stderr_sink("   ")  # blank dropped
    assert any(c.get("stderr") == "error line" for c in b.calls)


# ---- run(): timeout -------------------------------------------------------
async def test_run_timeout():
    async def _slow_query(*, prompt, options):
        await asyncio.sleep(0.5)
        yield _Msg()

    b = _backend()
    b.sdk_query_factory = _slow_query
    b.call_timeout_s = 0.01
    with pytest.raises(BackendError, match="timed out"):
        await b.run("hi")


# ---- run(): conversational session capture --------------------------------
async def test_run_conversational_session_capture(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC", "60")
    msg = _Msg(content=[_emit_tool_block()], result="done",
               usage={"input_tokens": 5, "output_tokens": 2,
                      "cache_read_input_tokens": 1,
                      "cache_creation_input_tokens": 0},
               session_id="sess-9")
    b = _backend(messages=[msg], conversational=True)
    b.sdk_query_factory = _query([msg])
    res = await b.run("hi")
    assert b.conversation_session_id == "sess-9"
    assert res.metadata["input_tokens"] == 5
    assert len(res.intents) == 1
    # reset clears it
    b.reset_conversation()
    assert b.conversation_session_id is None


# ---- run(): no-intent raises ----------------------------------------------
async def test_run_no_intent_raises():
    msg = _Msg(content=[TextBlock("just text")], result="hi")
    b = _backend()
    b.sdk_query_factory = _query([msg])
    with pytest.raises(NoIntentEmitted):
        await b.run("hi")


# ---- _invoke_and_collect: error-result-success tolerance ------------------
async def test_invoke_error_result_success_with_intents():
    msg = _Msg(content=[_emit_tool_block()])
    b = _backend()
    b.sdk_query_factory = _query([msg], raise_exc=Exception("error result: success"))
    res = await b.run("hi", allow_no_intent=True)
    assert len(res.intents) == 1


async def test_invoke_error_result_success_no_intents():
    msg = _Msg(content=[TextBlock("hello")])
    b = _backend()
    b.sdk_query_factory = _query([msg], raise_exc=Exception("error result: success"))
    res = await b.run("hi", allow_no_intent=True)
    assert res.intents == []


async def test_invoke_other_exception_reraises():
    b = _backend()
    b.sdk_query_factory = _query([], raise_exc=RuntimeError("real failure"))
    with pytest.raises(RuntimeError, match="real failure"):
        await b.run("hi", allow_no_intent=True)


async def test_run_raw_completion_headroom():
    msg = _Msg(content=[TextBlock("the answer")], result="the answer")
    b = _backend(raw_completion=True)
    b.sdk_query_factory = _query([msg])
    res = await b.run("hi", max_turns=1)
    # raw mode bumps max_turns to >= 8 and returns text without an intent
    assert res.raw_text == "the answer"
    assert b.calls[-1]["max_turns"] >= 8


def test_parse_tool_use_block_invalid_returns_none():
    b = _backend()
    bad = ToolUseBlock(name=cl.EMIT_INTENT_TOOL_QUALIFIED,
                       input={"intent_type": "not_a_real_type", "payload": {}})
    assert b._parse_tool_use_block(bad) is None
