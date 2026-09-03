# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for ClaudeBackend: SDK import, __post_init__ wiring,
option building (resume / context tools / raw mode), timeout handling, the
conversational session capture, and SDK-stream error tolerance."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.roles import claude as cl
from hyperloom.orchestrator.roles.base import BackendError
from hyperloom.inference_optimizer.protocol.intent import NoIntentEmitted


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
    monkeypatch.setattr(cl.importlib, "import_module", lambda name: SimpleNamespace())
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
    monkeypatch.setattr(cl, "build_emit_intent_server", lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
    b = _backend()
    assert b.mcp_server_config is None
    assert b.mcp_tool_name is None
    assert any("emit_intent MCP setup failed" in c.get("warn", "") for c in b.calls)


def test_post_init_conversational_floors(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC", raising=False)
    b = _backend(conversational=True, max_turns_default=2)
    assert b.max_turns_default >= cl._CONVERSATIONAL_MIN_MAX_TURNS
    assert b.call_timeout_s >= cl._CONVERSATIONAL_DEFAULT_TIMEOUT_SEC


# ---- set_context_provider -------------------------------------------------
def test_set_context_provider_success(monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server", lambda provider, **k: SimpleNamespace(name="ctx"))
    b = _backend()
    b.set_context_provider(SimpleNamespace())
    assert b._context_server_config is not None


def test_set_context_provider_failure(monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server", lambda provider, **k: (_ for _ in ()).throw(ValueError("x")))
    b = _backend()
    b.set_context_provider(SimpleNamespace())
    assert b._context_server_config is None
    assert any("context tools MCP setup failed" in c.get("warn", "") for c in b.calls)


# ---- _build_options -------------------------------------------------------
def test_build_options_model_and_resume():
    b = _backend(model="claude-x")
    opts = b._build_options(tools=["Read"], max_turns=5, system_prompt="sys", resume_session_id="sess-1")
    assert opts.kwargs["model"] == "claude-x"
    assert opts.kwargs["system_prompt"] == "sys"
    assert opts.kwargs["resume"] == "sess-1"


def test_build_options_raw_completion():
    b = _backend(raw_completion=True)
    opts = b._build_options(tools=["Read"], max_turns=8, system_prompt=None)
    assert opts.kwargs["allowed_tools"] == []
    assert "Bash" in opts.kwargs["disallowed_tools"]


def test_build_options_with_context_tools(monkeypatch):
    monkeypatch.setattr(cl, "build_context_tools_server", lambda provider, **k: SimpleNamespace(name="ctx"))
    b = _backend()
    b.set_context_provider(SimpleNamespace())
    opts = b._build_options(tools=[], max_turns=3, system_prompt=None)
    allowed = opts.kwargs.get("allowed_tools", [])
    for qn in cl.CONTEXT_TOOL_QUALIFIED_NAMES:
        assert qn in allowed
    assert cl.CONTEXT_MCP_SERVER_NAME in opts.kwargs["mcp_servers"]


# ---- _instantiate_options --------------------------------------------------
def test_instantiate_options_passthrough():
    class _Options:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    b = _backend(sdk_options_cls=_Options)
    opts = b._instantiate_options({"max_turns": 4, "resume": "s"})
    assert opts.kwargs == {"max_turns": 4, "resume": "s"}


def test_instantiate_options_typeerror_propagates():
    class _Boom:
        def __init__(self, **kwargs):
            raise TypeError("boom")

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


# ---- run(): idle timeout tolerates a slow-but-live stream -----
async def test_run_idle_timeout_allows_slow_but_live_stream():
    """A model that keeps streaming (gaps < idle budget) must NOT be killed,
    even when the TOTAL turn wall-clock exceeds ``call_timeout_s``."""

    async def _slow_live(*, prompt, options):
        for _ in range(4):
            # Per-message gap stays under the idle budget while cumulative time exceeds it,
            # proving the guard is idle-based, not a total wall-clock cap.
            await asyncio.sleep(0.03)
            yield _Msg(content=[_emit_tool_block()])

    b = _backend()
    b.sdk_query_factory = _slow_live
    b.call_timeout_s = 0.05
    res = await b.run("hi")
    assert len(res.intents) == 4


# ---- run(): conversational session capture --------------------------------
async def test_run_conversational_session_capture(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC", "60")
    msg = _Msg(
        content=[_emit_tool_block()],
        result="done",
        usage={"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 1, "cache_creation_input_tokens": 0},
        session_id="sess-9",
    )
    b = _backend(messages=[msg], conversational=True)
    b.sdk_query_factory = _query([msg])
    res = await b.run("hi")
    assert b._session_id == "sess-9"
    assert res.metadata["input_tokens"] == 5
    assert len(res.intents) == 1
    # reset clears it
    b.reset_conversation()
    assert b._session_id is None


# ---- run(): no-intent raises ----------------------------------------------
async def test_run_no_intent_raises():
    msg = _Msg(content=[TextBlock("just text")], result="hi")
    b = _backend(capture_turn_diagnostics=True)
    b.sdk_query_factory = _query([msg])
    with pytest.raises(NoIntentEmitted):
        await b.run("hi")
    diag = b.get_turn_diagnostic()
    assert diag["outcome"] == "no_intent"
    assert diag["raw_text"] == "hi"
    assert diag["messages"] == [{"type": "_Msg", "result": "hi"}]


async def test_run_skips_diagnostics_when_not_requested():
    msg = _Msg(content=[TextBlock("just text")], result="hi")
    b = _backend()
    b.sdk_query_factory = _query([msg])
    await b.run("hi", allow_no_intent=True)
    assert b.get_turn_diagnostic() == {}


# ---- gateway endpoint identifier -----------------------------------------
def test_gateway_endpoint_drops_url_userinfo(monkeypatch):
    """The diagnostic is appended to an on-disk trace, and a base URL of the
    form ``https://user:key@gw/...`` puts the key in netloc."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://user:s3cret@gw.example.com:8443/api/v1")
    assert _backend()._gateway_endpoint_identifier() == "gw.example.com"


def test_gateway_endpoint_is_none_without_a_base_url(monkeypatch):
    for var in ("ANTHROPIC_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert _backend()._gateway_endpoint_identifier() is None


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


# ---- per-request context water level -------------------------------------
@dataclass
class _ResultMsg(_Msg):
    num_turns: int = 0


async def test_context_peak_prefers_the_largest_single_request():
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
    b = _backend()
    b.sdk_query_factory = _query(stream)
    res = await b.run("hi", allow_no_intent=True)
    assert res.metadata["context_tokens_peak"] == 175
    # The cumulative counters are still reported verbatim for cost accounting.
    assert res.metadata["cache_read_input_tokens"] == 250


async def test_context_peak_falls_back_to_a_per_turn_estimate():
    # Only the terminal cumulative usage is available: divide by the turn count.
    stream = [
        _ResultMsg(
            result="done",
            num_turns=4,
            usage={"input_tokens": 0, "cache_read_input_tokens": 300_000, "cache_creation_input_tokens": 100_000},
        ),
    ]
    b = _backend()
    b.sdk_query_factory = _query(stream)
    res = await b.run("hi", allow_no_intent=True)
    assert res.metadata["context_tokens_peak"] == 100_000


async def test_context_peak_is_zero_without_any_usage():
    b = _backend()
    b.sdk_query_factory = _query([_Msg(content=[TextBlock("hi")], result="hi")])
    res = await b.run("hi", allow_no_intent=True)
    assert res.metadata["context_tokens_peak"] == 0


# ---- stop_reason passthrough ---------------------------------------------
@dataclass
class _StopMsg(_Msg):
    stop_reason: str | None = None


async def test_stop_reason_reaches_metadata():
    """Without it a truncated reply is indistinguishable from a badly formatted
    one, so the SDK's own stop reason must survive to the caller."""
    stream = [_StopMsg(content=[TextBlock("half a rep")], result="half a rep", stop_reason="max_tokens")]
    b = _backend()
    b.sdk_query_factory = _query(stream)
    res = await b.run("hi", allow_no_intent=True)
    assert res.metadata["stop_reason"] == "max_tokens"


async def test_stop_reason_is_none_when_the_sdk_omits_it():
    b = _backend()
    b.sdk_query_factory = _query([_Msg(content=[TextBlock("hi")], result="hi")])
    res = await b.run("hi", allow_no_intent=True)
    assert res.metadata["stop_reason"] is None


async def test_last_reported_stop_reason_wins():
    """Per-message stop reasons stream in order; the terminal one describes the call."""
    stream = [
        _StopMsg(content=[TextBlock("a")], stop_reason="tool_use"),
        _StopMsg(content=[TextBlock("b")], result="ab", stop_reason="end_turn"),
    ]
    b = _backend()
    b.sdk_query_factory = _query(stream)
    res = await b.run("hi", allow_no_intent=True)
    assert res.metadata["stop_reason"] == "end_turn"


def test_parse_tool_use_block_invalid_returns_none():
    b = _backend()
    bad = ToolUseBlock(name=cl.EMIT_INTENT_TOOL_QUALIFIED, input={"intent_type": "not_a_real_type", "payload": {}})
    assert b._parse_tool_use_block(bad) is None


# ---- mcp_emit_intent: handler + validation branches -----------------------

from hyperloom.orchestrator.roles import mcp_emit_intent as mei  # noqa: E402
from hyperloom.inference_optimizer.protocol.intent import (  # noqa: E402
    IntentValidationError,
)


def test_validate_emit_intent_input_non_dict_payload_raises():
    with pytest.raises(IntentValidationError, match="'payload' must be an object"):
        mei.validate_emit_intent_input({"intent_type": "send_message", "payload": "nope"})


def test_validate_emit_intent_input_not_an_object_raises():
    with pytest.raises(IntentValidationError, match="must be an object"):
        mei.validate_emit_intent_input(["not", "a", "dict"])  # type: ignore[arg-type]


def test_validate_emit_intent_input_unexpected_keys_raises():
    with pytest.raises(IntentValidationError, match="unexpected keys"):
        mei.validate_emit_intent_input({"intent_type": "send_message", "payload": {}, "extra": 1})


def test_validate_emit_intent_input_missing_top_level_key_raises():
    with pytest.raises(IntentValidationError, match="requires both"):
        mei.validate_emit_intent_input({"intent_type": "send_message"})


def test_validate_emit_intent_input_missing_required_payload_field_raises():
    # send_message requires 'topic'; omit it.
    with pytest.raises(IntentValidationError, match="missing required fields"):
        mei.validate_emit_intent_input({"intent_type": "send_message", "payload": {}})


@pytest.mark.asyncio
async def test_emit_intent_handler_ok():
    res = await mei._emit_intent_handler({"intent_type": "send_message", "payload": {"topic": "heartbeat"}})
    assert res["content"][0]["text"] == "ok"
    assert "is_error" not in res


@pytest.mark.asyncio
async def test_emit_intent_handler_accepts_unparsed_wrapper():
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
    res = await mei._emit_intent_handler({"__unparsedToolInput": {"raw": raw, "len": len(raw)}})
    assert res["content"][0]["text"] == "ok"
    assert "is_error" not in res


@pytest.mark.asyncio
async def test_emit_intent_handler_validation_error_returns_is_error():
    res = await mei._emit_intent_handler({"intent_type": "bogus", "payload": {}})
    assert res["is_error"] is True
    assert "validation_error" in res["content"][0]["text"]
