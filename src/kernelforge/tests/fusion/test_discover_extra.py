# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover trace edge-cases, salvage/extract branches, default_llm_fn, source errors."""

from __future__ import annotations

import importlib
import json
import sys
import types

import anthropic as _REAL_ANTHROPIC
import pytest

# The httpx flavour the installed anthropic SDK was built against. anthropic 1.x
# moved to httpx2 and type-checks ``http_client`` against it, so a Response or
# MockTransport from the wrong module is rejected at client construction -- the
# same failure this file's production counterpart guards, arriving through the
# stub instead. Derived from the SDK rather than imported, so the tests follow
# it across the migration.
_REAL_HTTPX = importlib.import_module(_REAL_ANTHROPIC.DefaultHttpxClient.__mro__[1].__module__)

from kernelforge.fusion.diagnose import diagnose_from_shares
from kernelforge.fusion.llm_failure import (
    AUTH,
    CONTEXT_LENGTH,
    NOT_CONFIGURED,
    LlmUnavailableError,
)
from kernelforge.fusion.discover import (
    _extract_json_array,
    _salvage_objects,
    default_llm_fn,
    discover_recipes,
    hot_kernels_from_trace,
    parse_discovered_recipes,
)


def test_default_llm_fn_passes_apim_default_headers(monkeypatch):
    # Repro: on an APIM gateway the OpenAI client must send Ocp-Apim-Subscription-Key
    # via default_headers, else 401 "missing subscription key".
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm-api.amd.com/Unified/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub-key")
    captured = {}

    class _Resp:
        class _C:
            message = type("M", (), {"content": '[{"name":"x"}]'})()

        choices = [_C()]

    class _Completions:
        def create(self, **k):
            return _Resp()

    class _FakeClient:
        def __init__(self, **k):
            captured.update(k)
            self.chat = type("Chat", (), {"completions": _Completions()})()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    fake_openai.DefaultHttpxClient = lambda **_k: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    fn = default_llm_fn()
    fn("prompt")
    assert captured.get("default_headers") == {"Ocp-Apim-Subscription-Key": "sub-key"}, (
        "OpenAI client must carry the APIM subscription header"
    )


def _install_capturing_openai(monkeypatch):
    """Install a fake openai/httpx that records OpenAI(**kwargs); returns the dict."""
    captured: dict = {}

    class _Resp:
        class _C:
            message = type("M", (), {"content": "[]"})()

        choices = [_C()]

    class _FakeClient:
        def __init__(self, **k):
            captured.update(k)
            self.chat = type("Chat", (), {"completions": type("Cmp", (), {"create": lambda self, **k: _Resp()})()})()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    # discover.py asks the SDK for its own client class, so the fake SDK has to
    # offer one; a bare httpx stub would no longer be consulted.
    fake_openai.DefaultHttpxClient = lambda **k: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return captured


def test_default_llm_fn_no_custom_headers_omits_default_headers(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    captured = _install_capturing_openai(monkeypatch)
    default_llm_fn()("prompt")
    assert "default_headers" not in captured


def test_default_llm_fn_falls_back_to_anthropic_custom_headers(monkeypatch):
    # When OPENAI_CUSTOM_HEADERS is unset and both lines are the same gateway,
    # fusion discovery reuses the Anthropic line's headers so spend tags
    # injected there reach OpenAI-protocol calls.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gw")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "user: ntid-123")
    captured = _install_capturing_openai(monkeypatch)
    default_llm_fn()("prompt")
    assert captured["default_headers"] == {"user": "ntid-123"}


def test_default_llm_fn_does_not_send_anthropic_headers_to_another_host(monkeypatch):
    # These headers carry the operator's gateway secret; a different host is not
    # entitled to it just because this line left its own header slot empty.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://other/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gw")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "user: ntid-123")
    captured = _install_capturing_openai(monkeypatch)
    default_llm_fn()("prompt")
    assert "default_headers" not in captured


def _anthropic_only(monkeypatch, base="https://llm-api.amd.com/anthropic"):
    """An operator whose gateway serves Claude on the Anthropic line only."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")


def _messages_reply(blocks):
    """A Messages response body the SDK will accept and parse."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": blocks,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


_ANTHROPIC_OK = _messages_reply([{"type": "text", "text": "[]"}])


def _install_anthropic_transport(monkeypatch, *, payload=None, status=200, body=""):
    """Intercept at the HTTP layer so the real SDK builds the request.

    Faking the SDK itself would prove nothing about the URL it assembles, the
    header it derives from the credential kind, or how it raises a 4xx -- which
    is precisely what these tests are about.
    """
    import anthropic as real_anthropic

    # Bound at import time: a test may have replaced sys.modules["httpx"] with a
    # stub for the OpenAI leg, and this helper still needs the real transport.
    real_httpx = _REAL_HTTPX
    seen: dict = {"n": 0}
    real_cls = real_anthropic.Anthropic

    def _handler(request):
        seen["n"] += 1
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        if request.content:
            seen["json"] = json.loads(request.content)
        if status >= 400:
            return real_httpx.Response(status, text=body)
        return real_httpx.Response(200, json=payload if payload is not None else _ANTHROPIC_OK)

    def _factory(**kwargs):
        # Swap only the transport: the SDK itself stays real, so the assertions
        # below are about what it genuinely sends.
        kwargs["http_client"] = real_httpx.Client(transport=real_httpx.MockTransport(_handler))
        return real_cls(**kwargs)

    monkeypatch.setattr(real_anthropic, "Anthropic", _factory)
    return seen


def test_anthropic_only_gateway_is_usable_instead_of_not_configured(monkeypatch):
    # Repro: a gateway that serves Claude only over the Anthropic protocol (AMD's
    # APIM among them) left discovery dead -- the OpenAI line is unset, so every
    # run raised NOT_CONFIGURED and proposed nothing.
    _anthropic_only(monkeypatch)
    seen = _install_anthropic_transport(monkeypatch)

    assert default_llm_fn(model="claude-opus-5")("prompt") == "[]"
    assert seen["url"] == "https://llm-api.amd.com/anthropic/v1/messages"
    assert seen["json"]["model"] == "claude-opus-5"
    assert seen["headers"]["x-api-key"] == "ant-key"
    assert seen["headers"]["anthropic-version"]


@pytest.mark.parametrize(
    "configured",
    [
        # LiteLLM proxies publish a base that already ends in /v1, and a full
        # endpoint copied out of a curl command turns up too. The SDK appends
        # /v1/messages itself, so both must have their tail stripped first.
        "https://gw.example/api/v1/llm-proxy/v1",
        "https://gw.example/api/v1/llm-proxy/v1/messages",
        "https://gw.example/api/v1/llm-proxy/",
    ],
)
def test_anthropic_endpoint_is_never_doubled(monkeypatch, configured):
    _anthropic_only(monkeypatch, base=configured)
    seen = _install_anthropic_transport(monkeypatch)

    default_llm_fn()("prompt")
    assert seen["url"] == "https://gw.example/api/v1/llm-proxy/v1/messages"


def test_anthropic_reply_skips_thinking_blocks(monkeypatch):
    # A thinking-enabled deployment puts a thinking block first; reading
    # content[0] would hand discovery an empty string and lose the answer.
    _anthropic_only(monkeypatch)
    _install_anthropic_transport(
        monkeypatch,
        payload=_messages_reply(
            [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": '[{"name":"x"}]'},
            ]
        ),
    )

    assert default_llm_fn()("prompt") == '[{"name":"x"}]'


def test_anthropic_side_carries_its_own_custom_headers(monkeypatch):
    # APIM wants Ocp-Apim-Subscription-Key; per-side separation means the
    # OpenAI line's headers must not ride along.
    _anthropic_only(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Openai-Only: leak")
    seen = _install_anthropic_transport(monkeypatch)

    default_llm_fn()("prompt")
    assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "sub-key"
    assert "X-Openai-Only" not in seen["headers"]


def test_anthropic_auth_token_uses_bearer_not_x_api_key(monkeypatch):
    # ANTHROPIC_AUTH_TOKEN is a bearer token, not an API key. Sending it as
    # x-api-key fails auth on the gateways that only issue that form.
    _anthropic_only(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-123")
    seen = _install_anthropic_transport(monkeypatch)

    default_llm_fn()("prompt")
    assert seen["headers"]["Authorization"] == "Bearer tok-123"
    assert "x-api-key" not in seen["headers"]


@pytest.mark.parametrize(
    "status,body,expected_kind",
    [
        # The 403 body is a real one from the AMD gateway when the key lacks
        # access to a model. None of these phrases are recognizable to the
        # message scanner, so before the status was carried they all classified
        # as a retryable api_error and burned the whole retry budget on a
        # failure that would never have succeeded.
        (403, "Access to model [claude-opus-5] is not available.", AUTH),
        (401, "upstream said no", AUTH),
        (413, "that was a lot of tokens", CONTEXT_LENGTH),
    ],
)
def test_anthropic_http_status_drives_failure_classification(monkeypatch, status, body, expected_kind):
    _anthropic_only(monkeypatch)
    monkeypatch.setenv("FORGE_FUSION_LLM_ATTEMPTS", "4")
    calls = _install_anthropic_transport(monkeypatch, status=status, body=body)

    with pytest.raises(LlmUnavailableError) as excinfo:
        default_llm_fn()("prompt")
    assert excinfo.value.kind == expected_kind
    assert excinfo.value.retryable is False
    assert calls["n"] == 1


def test_anthropic_server_error_is_still_retried(monkeypatch):
    """Classifying on status must not turn every HTTP failure into a hard stop."""
    _anthropic_only(monkeypatch)
    monkeypatch.setenv("FORGE_FUSION_LLM_ATTEMPTS", "3")
    monkeypatch.setenv("FORGE_FUSION_LLM_RETRY_BASE_SEC", "0")
    calls = _install_anthropic_transport(monkeypatch, status=500, body="upstream exploded")

    with pytest.raises(LlmUnavailableError) as excinfo:
        default_llm_fn()("prompt")
    assert excinfo.value.retryable is True
    assert calls["n"] == 3


def test_openai_line_wins_on_the_direct_path(monkeypatch):
    # Scoped to the direct path: the conftest fixture has the harness off, and
    # in production the harness would take precedence over both lines.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm-api.amd.com/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    captured = _install_capturing_openai(monkeypatch)

    default_llm_fn()("prompt")
    assert captured["base_url"] == "http://gw"


def test_neither_line_configured_still_raises_not_configured(monkeypatch):
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(LlmUnavailableError) as excinfo:
        default_llm_fn()("prompt")
    assert excinfo.value.kind == NOT_CONFIGURED


def _candidate_diag():
    return diagnose_from_shares(
        {"gemm": 0.4, "add": 0.14, "elementwise": 0.14, "cast": 0.13, "mul": 0.08}, busy_fraction_of_wall=0.21
    )


# ── hot_kernels_from_trace edge cases ────────────────────────────────────────
def test_hot_kernels_events_not_list(tmp_path):
    p = tmp_path / "d.trace.json"
    p.write_text(json.dumps({"traceEvents": "oops"}))
    assert hot_kernels_from_trace(p) == []


def test_hot_kernels_skips_bad_dur_and_nonpositive(tmp_path):
    p = tmp_path / "d.trace.json"
    p.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "kernel", "name": "a", "dur": "notnum"},
                    {"cat": "kernel", "name": "b", "dur": 0},
                    {"cat": "kernel", "name": "c", "dur": -5},
                    {"cat": "cpu_op", "name": "skip", "dur": 100},
                    {"cat": "kernel", "name": "mul_kernel", "dur": 20},
                ]
            }
        )
    )
    hot = hot_kernels_from_trace(p)
    assert [h["name"] for h in hot] == ["mul_kernel"]


def test_hot_kernels_total_zero_returns_empty(tmp_path):
    p = tmp_path / "d.trace.json"
    p.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "kernel", "name": "a", "dur": 0},
                ]
            }
        )
    )
    assert hot_kernels_from_trace(p) == []


def test_hot_kernels_gz(tmp_path):
    import gzip

    p = tmp_path / "d.trace.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump({"traceEvents": [{"cat": "kernel", "name": "mul_k", "dur": 10}]}, fh)
    hot = hot_kernels_from_trace(p)
    assert hot and hot[0]["name"] == "mul_k"


# ── salvage / extract branches ───────────────────────────────────────────────
def test_salvage_ignores_braces_in_strings():
    text = '[{"name": "a {nested}", "v": 1}, {"name": "b", "esc": "x\\"y"}]'
    objs = _salvage_objects(text)
    assert len(objs) == 2
    assert objs[0]["name"] == "a {nested}"


def test_extract_empty_text():
    assert _extract_json_array("") == []


def test_extract_skips_invalid_array_then_salvages():
    # An unbalanced/invalid array falls through to object salvage.
    text = '[{"name": "a"}, {broken'
    objs = _extract_json_array(text)
    assert objs == [{"name": "a"}]


def test_parse_anchor_as_string_coerced():
    text = '[{"name":"x","env_flag":"F","source_anchors":"single_anchor"}]'
    rs = parse_discovered_recipes(text, model_type="m", framework="f", source_file="/x.py", shapes={})
    assert rs[0].source_hints == ["single_anchor"]


# ── default_llm_fn ───────────────────────────────────────────────────────────
def test_default_llm_fn_no_gateway_raises(monkeypatch):
    # An unconfigured gateway is an environment fault, not a finding about the
    # model; returning "" here used to land as verdict no_opportunity.
    for k in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    fn = default_llm_fn()
    with pytest.raises(LlmUnavailableError) as excinfo:
        fn("prompt")
    assert excinfo.value.kind == NOT_CONFIGURED
    assert excinfo.value.retryable is False


def test_default_llm_fn_success_writes_log(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")

    class _Msg:
        content = '[{"name":"x"}]'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **k):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        def __init__(self, **k):
            self.chat = _Chat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    fake_openai.DefaultHttpxClient = lambda **_k: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    log = tmp_path / "llm.log"
    fn = default_llm_fn(log_path=str(log))
    out = fn("prompt")
    assert out == '[{"name":"x"}]'
    assert log.read_text() == out


def test_default_llm_fn_uses_the_resolved_pair(monkeypatch):
    """The OpenAI client is built from the OpenAI line only, never the Anthropic one.

    ``create`` returns successfully so the last leg also proves the resolved
    credential drives a completed call, not just the constructor. An
    unconfigured line must raise rather than return "": an unreachable model is
    an environment fault, not the model reporting no opportunity.
    """
    captured: dict = {}

    class _Msg:
        content = '[{"name":"x"}]'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **k):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        def __init__(self, **k):
            captured.update(k)
            self.chat = _Chat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    fake_openai.DefaultHttpxClient = lambda **_k: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    for k in (
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "SAFE_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)

    # A retired key alongside a base URL is not a pair: no client, no call.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-gw/v1")
    monkeypatch.setenv("SAFE_API_KEY", "safe")
    with pytest.raises(LlmUnavailableError) as excinfo:
        default_llm_fn()("prompt")
    assert excinfo.value.kind == NOT_CONFIGURED
    assert captured == {}

    # A complete Anthropic line does serve discovery, but over its own protocol:
    # its endpoint and credential must never reach the OpenAI client.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://anthropic-gw")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    seen = _install_anthropic_transport(monkeypatch)
    assert default_llm_fn()("prompt") == "[]"
    assert seen["url"] == "http://anthropic-gw/v1/messages"
    assert captured == {}

    # Its own pair drives the client, endpoint exactly as configured.
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    assert default_llm_fn()("prompt") == '[{"name":"x"}]'
    assert captured["base_url"] == "http://openai-gw/v1"
    assert captured["api_key"] == "openai"


def test_default_llm_fn_retries_then_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.setenv("FORGE_FUSION_LLM_ATTEMPTS", "3")
    calls = {"n": 0}

    class _Completions:
        def create(self, **k):
            calls["n"] += 1
            raise RuntimeError("bad request 400")

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        def __init__(self, **k):
            self.chat = _Chat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    fake_openai.DefaultHttpxClient = lambda **_k: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    fn = default_llm_fn()
    with pytest.raises(LlmUnavailableError) as excinfo:
        fn("prompt")
    assert calls["n"] == 3
    assert excinfo.value.attempts == 3
    assert excinfo.value.retryable is True


def test_default_llm_fn_setup_failure_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")

    # Accessing OpenAI attribute raises inside the function.
    class _Broken(types.ModuleType):
        @property
        def OpenAI(self):
            raise ImportError("no openai")

    monkeypatch.setitem(sys.modules, "openai", _Broken("openai"))
    monkeypatch.setitem(sys.modules, "httpx", types.ModuleType("httpx"))
    fn = default_llm_fn()
    with pytest.raises(LlmUnavailableError):
        fn("prompt")


# ── discover_recipes source-read failures ────────────────────────────────────
def test_discover_recipes_empty_source_file(tmp_path):
    d = _candidate_diag()
    rs = discover_recipes(
        d, model_type="m", framework="f", source_file="", shapes={}, trace_path="/x", llm_fn=lambda p: "[]"
    )
    assert rs == []


def test_discover_recipes_unreadable_source(tmp_path):
    d = _candidate_diag()
    rs = discover_recipes(
        d,
        model_type="m",
        framework="f",
        source_file=str(tmp_path / "nope.py"),
        shapes={},
        trace_path="/x",
        llm_fn=lambda p: "[]",
    )
    assert rs == []


def _fake_anthropic_reply():
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="[]")])


def test_anthropic_adapter_sends_temperature_in_the_body_when_the_sdk_wont_name_it():
    """anthropic 1.x dropped ``temperature`` from ``Messages.create()``.

    The signature has no ``**kwargs``, so passing it named is a TypeError --
    which classify_llm_error reads as transient, so discovery burned its whole
    retry budget on a call that could never succeed. It is still a Messages API
    field, so it travels in ``extra_body`` instead.
    """
    from kernelforge.fusion.discover import _AnthropicChatCompletions

    seen: dict = {}

    class _OneXMessages:
        def create(self, *, model, max_tokens, messages, extra_body=None):
            seen.update(model=model, max_tokens=max_tokens, extra_body=extra_body)
            return _fake_anthropic_reply()

    client = types.SimpleNamespace(messages=_OneXMessages())
    out = _AnthropicChatCompletions(client).create(
        model="claude-opus-5", temperature=0, max_tokens=64, messages=[{"role": "user", "content": "p"}]
    )
    assert out.choices[0].message.content == "[]"
    assert seen["extra_body"] == {"temperature": 0}


def test_anthropic_adapter_still_names_temperature_when_the_sdk_declares_it():
    """The 0.x signature takes it by name; do not push it into the body there."""
    from kernelforge.fusion.discover import _AnthropicChatCompletions

    seen: dict = {}

    class _ZeroXMessages:
        def create(self, *, model, max_tokens, messages, temperature=None):
            seen.update(temperature=temperature)
            return _fake_anthropic_reply()

    client = types.SimpleNamespace(messages=_ZeroXMessages())
    _AnthropicChatCompletions(client).create(
        model="claude-opus-5", temperature=0, max_tokens=64, messages=[{"role": "user", "content": "p"}]
    )
    assert seen == {"temperature": 0}
