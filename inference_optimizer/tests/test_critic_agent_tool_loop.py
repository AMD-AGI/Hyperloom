# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``CriticAgentBackend._run_reasoning_loop`` — issue #170 web tools (web_search / web_fetch) integration.

Reuses the runtime-subprocess fake from ``test_critic_agent_backend`` so the
focus stays on the LLM tool-call loop. Web tool clients are injected via
``web_tool_clients_factory`` with fakes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import CriticAgentBackend, RuntimeCall

from inference_optimizer.tests.test_critic_agent_backend import (
    _make_fake_runtime,
)


# Resolve critic-agent's on-disk ``runtime`` package so its web_tools import succeeds at test time.
REAL_CRITIC_AGENT_ROOT = (
    Path(__file__).resolve().parents[2] / "critic-agent"
)


# Skip the module when critic-agent isn't checked out alongside the parent project.
pytestmark = pytest.mark.skipif(
    not (REAL_CRITIC_AGENT_ROOT / "runtime" / "web_tools" / "__init__.py").is_file(),
    reason="critic-agent runtime.web_tools not present on disk",
)


# OpenAI-shaped fakes that support tool_calls

@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction
    type: str = "function"


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclass
class _FakeResp:
    choices: list[_FakeChoice] = field(default_factory=list)


@dataclass
class _ScriptedReply:
    """One scripted assistant message — either text or tool_calls."""

    text: str | None = None
    tool_calls: list[_FakeToolCall] | None = None

    def to_choice(self) -> _FakeChoice:
        return _FakeChoice(
            message=_FakeMessage(
                content=self.text,
                tool_calls=self.tool_calls,
            ),
            finish_reason="tool_calls" if self.tool_calls else "stop",
        )


class _ScriptedChatCompletions:
    def __init__(self, replies: list[_ScriptedReply]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> _FakeResp:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if not self._replies:
            return _FakeResp(choices=[_FakeChoice(_FakeMessage(content=""))])
        reply = self._replies.pop(0)
        return _FakeResp(choices=[reply.to_choice()])


class _ScriptedOpenAIClient:
    def __init__(self, replies: list[_ScriptedReply]) -> None:
        self.completions = _ScriptedChatCompletions(replies)
        self.chat = type("_C", (), {"completions": self.completions})()


# Fake web tool clients

@dataclass
class _RecordingSearchClient:
    """In-process stand-in for :class:`WebSearchClient`."""

    output: str = "Web search results for query: ...\nLinks: []\n\nREMINDER: cite."
    raises: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def execute(self, payload: dict) -> str:
        self.calls.append(dict(payload))
        if self.raises is not None:
            raise self.raises
        return self.output


@dataclass
class _RecordingFetchClient:
    output: str = "URL: https://x/y\nStatus: 200\n---\nbody"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def execute(self, payload: dict) -> str:
        self.calls.append(dict(payload))
        return self.output


@dataclass
class _FakeWebClients:
    search: _RecordingSearchClient | None = None
    fetch: _RecordingFetchClient | None = None


# Helpers

def _bundle(session_id: str, *, proposals: list[dict] | None = None) -> dict[str, Any]:
    """Minimum judge-bundle shape that drives the LLM path (proposals!=[])."""
    return {
        "kind": "coordinator_inbox",
        "session_id": session_id,
        "merged_context": {"model": "Qwen3", "framework": "sglang"},
        "missing_context": {},
        "required_context": [],
        "proposals": proposals or [
            {
                "msg_id": "p1",
                "action_name": "kernel_opt",
                "predicted_gain_pct": 4.0,
            },
        ],
        "kb_priors_by_proposal": {},
        "kb_priors_for_decision": {},
        "review_constraints": {},
        "notes": [],
    }


def _build_review_reply() -> str:
    return (
        "```json\n"
        '{"review_verdicts": [{"target_proposal_msg_id": "p1", '
        '"verdict": "approve", "source": "critic", "reasoning": "ok", '
        '"confidence": "medium"}]}'
        "\n```"
    )


def _tool_call(
    call_id: str, name: str, arguments: dict[str, Any],
) -> _FakeToolCall:
    return _FakeToolCall(
        id=call_id,
        function=_FakeFunction(name=name, arguments=json.dumps(arguments)),
    )


def _import_web_tools_config():
    """Late import so REAL_CRITIC_AGENT_ROOT is on sys.path first; evicts a stale ``runtime`` namespace cache."""
    real = str(REAL_CRITIC_AGENT_ROOT)
    if real not in sys.path:
        sys.path.insert(0, real)
    expected = Path(real, "runtime").resolve()
    runtime_mod = sys.modules.get("runtime")
    cached_paths = []
    if runtime_mod is not None:
        try:
            cached_paths = [
                Path(p).resolve() for p in (getattr(runtime_mod, "__path__", []) or [])
            ]
        except (OSError, ValueError):
            cached_paths = []
    if runtime_mod is not None and expected not in cached_paths:
        for key in [k for k in sys.modules if k == "runtime" or k.startswith("runtime.")]:
            sys.modules.pop(key, None)
    from runtime.web_tools import WebToolsConfig
    return WebToolsConfig


def _make_backend(
    *,
    tmp_session: Path,
    scripted_replies: list[_ScriptedReply],
    search_client: _RecordingSearchClient | None = None,
    fetch_client: _RecordingFetchClient | None = None,
    web_enabled: bool = True,
    max_tool_turns: int = 4,
    judge_bundle: dict[str, Any] | None = None,
    runtime_calls: list[RuntimeCall] | None = None,
) -> tuple[CriticAgentBackend, _ScriptedOpenAIClient]:
    """Build a backend wired to scripted OpenAI + injected web clients."""
    Cfg = _import_web_tools_config()
    cfg = Cfg(
        critic_web_tools_enabled=web_enabled,
        critic_web_max_tool_turns=max_tool_turns,
        search_provider="tavily" if search_client is not None else "disabled",
        fetch_enabled=fetch_client is not None,
        tavily_api_key="dummy" if search_client is not None else "",
    )
    fake_client = _ScriptedOpenAIClient(scripted_replies)
    bundle = judge_bundle or _bundle(tmp_session.name)
    fake_caller = _make_fake_runtime(
        judge_bundle=bundle, capture=runtime_calls,
    )
    clients = _FakeWebClients(search=search_client, fetch=fetch_client)
    backend = CriticAgentBackend(
        critic_agent_root=REAL_CRITIC_AGENT_ROOT,
        session_dir=tmp_session,
        codex_model="gpt-test",
        codex_client_factory=lambda: fake_client,
        runtime_caller_factory=lambda: fake_caller,
        web_tools_config=cfg,
        web_tool_clients_factory=lambda _cfg: clients,
        static_context={"model": "Qwen3", "framework": "sglang"},
    )
    return backend, fake_client


@pytest.fixture
def tmp_session(tmp_path: Path) -> Path:
    sd = tmp_path / "session-xyz"
    sd.mkdir()
    return sd


# Tests: legacy single-call path (web tools off)

@pytest.mark.asyncio
async def test_disabled_web_tools_keeps_single_call_path(tmp_session: Path):
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[_ScriptedReply(text=_build_review_reply())],
        web_enabled=False,
    )

    await backend.run(prompt="inbox", system_prompt="You are critic.")

    assert len(fake.completions.calls) == 1
    assert "tools" not in fake.completions.calls[0]["kwargs"]


# Tests: enabled but no tool_calls on first turn

@pytest.mark.asyncio
async def test_enabled_with_immediate_text_makes_one_call(tmp_session: Path):
    search = _RecordingSearchClient()
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[_ScriptedReply(text=_build_review_reply())],
        search_client=search,
    )

    await backend.run(prompt="inbox", system_prompt="You are critic.")

    assert len(fake.completions.calls) == 1
    kwargs = fake.completions.calls[0]["kwargs"]
    tool_names = [t["function"]["name"] for t in kwargs["tools"]]
    assert tool_names == ["web_search"]
    assert kwargs["tool_choice"] == "auto"
    assert search.calls == []


# Tests: model issues a search tool call

@pytest.mark.asyncio
async def test_search_tool_call_is_dispatched_and_result_appended(tmp_session: Path):
    search = _RecordingSearchClient(output="SEARCH_RESULT_OK")
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _tool_call("c1", "web_search", {"query": "sglang fp8"}),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        search_client=search,
    )

    result = await backend.run(prompt="inbox", system_prompt="You are critic.")

    assert len(fake.completions.calls) == 2
    assert search.calls == [{"query": "sglang fp8"}]
    msgs = fake.completions.calls[1]["messages"]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs)
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "c1"
    assert tool_msgs[0]["content"] == "SEARCH_RESULT_OK"
    assert any(i.type.value == "review_verdict" for i in result.intents)


# Tests: fetch tool call is dispatched too

@pytest.mark.asyncio
async def test_fetch_tool_call_dispatched(tmp_session: Path):
    fetch = _RecordingFetchClient(output="PAGE_OK")
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _tool_call("c1", "web_fetch", {"url": "https://x/y"}),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        fetch_client=fetch,
    )

    await backend.run(prompt="inbox", system_prompt="You are critic.")

    assert fetch.calls == [{"url": "https://x/y"}]
    msgs = fake.completions.calls[1]["messages"]
    assert any(m.get("role") == "tool" and m["content"] == "PAGE_OK" for m in msgs)


# Tests: multiple parallel tool_calls in one assistant turn

@pytest.mark.asyncio
async def test_parallel_tool_calls_all_dispatched(tmp_session: Path):
    search = _RecordingSearchClient(output="S")
    fetch = _RecordingFetchClient(output="F")
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _tool_call("c1", "web_search", {"query": "a"}),
                _tool_call("c2", "web_fetch", {"url": "https://a/b"}),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        search_client=search,
        fetch_client=fetch,
    )

    await backend.run(prompt="inbox", system_prompt="You are critic.")
    assert search.calls and fetch.calls
    msgs = fake.completions.calls[1]["messages"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert [m["content"] for m in tool_msgs] == ["S", "F"]


# Tests: bad JSON arguments propagate as a tool error message

@pytest.mark.asyncio
async def test_invalid_tool_arguments_returns_error_message(tmp_session: Path):
    search = _RecordingSearchClient()
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _FakeToolCall(
                    id="c1",
                    function=_FakeFunction(name="web_search", arguments="not-json"),
                ),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        search_client=search,
    )
    await backend.run(prompt="inbox", system_prompt="You are critic.")
    assert search.calls == []
    tool_msgs = [
        m for m in fake.completions.calls[1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "valid JSON" in tool_msgs[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["[]", '"x"'])
async def test_non_object_tool_arguments_returns_error_message(
    tmp_session: Path, arguments: str,
):
    search = _RecordingSearchClient()
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _FakeToolCall(
                    id="c1",
                    function=_FakeFunction(name="web_search", arguments=arguments),
                ),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        search_client=search,
    )
    await backend.run(prompt="inbox", system_prompt="You are critic.")
    assert search.calls == []
    tool_msgs = [
        m for m in fake.completions.calls[1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "JSON object" in tool_msgs[0]["content"]


# Tests: unknown tool name yields an inline error

@pytest.mark.asyncio
async def test_unknown_tool_name_returns_error_message(tmp_session: Path):
    search = _RecordingSearchClient()
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _tool_call("c1", "evil_eval", {"code": "1+1"}),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        search_client=search,
    )
    await backend.run(prompt="inbox", system_prompt="You are critic.")
    tool_msgs = [
        m for m in fake.completions.calls[1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "unknown" in tool_msgs[0]["content"]


# Tests: max_tool_turns exhausted forces a final no-tool call

@pytest.mark.asyncio
async def test_max_tool_turns_forces_final_no_tool_call(tmp_session: Path):
    search = _RecordingSearchClient(output="S")
    # max_turns=1: turn 0 issues a tool call → turn 1 (final) is forced no-tool.
    backend, fake = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _tool_call("c1", "web_search", {"query": "x"}),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        search_client=search,
        max_tool_turns=1,
    )

    await backend.run(prompt="inbox", system_prompt="You are critic.")

    assert len(fake.completions.calls) == 2
    assert "tools" in fake.completions.calls[0]["kwargs"]
    assert "tools" not in fake.completions.calls[1]["kwargs"]


# Tests: web tools enabled by config but no clients usable

@pytest.mark.asyncio
async def test_enabled_config_but_empty_clients_falls_back_to_no_tools(
    tmp_session: Path,
):
    """When build_clients returns no usable clients the loop degrades to a single no-tool call."""
    Cfg = _import_web_tools_config()
    cfg = Cfg(critic_web_tools_enabled=True, search_provider="tavily")
    fake_client = _ScriptedOpenAIClient([_ScriptedReply(text=_build_review_reply())])
    bundle = _bundle(tmp_session.name)
    fake_caller = _make_fake_runtime(judge_bundle=bundle)

    backend = CriticAgentBackend(
        critic_agent_root=REAL_CRITIC_AGENT_ROOT,
        session_dir=tmp_session,
        codex_model="gpt-test",
        codex_client_factory=lambda: fake_client,
        runtime_caller_factory=lambda: fake_caller,
        web_tools_config=cfg,
        web_tool_clients_factory=lambda _c: _FakeWebClients(search=None, fetch=None),
        static_context={"model": "Qwen3", "framework": "sglang"},
    )
    await backend.run(prompt="inbox", system_prompt="You are critic.")
    assert "tools" not in fake_client.completions.calls[0]["kwargs"]


@pytest.mark.asyncio
async def test_fetch_only_config_exposes_fetch_schema_not_search(tmp_session: Path):
    """When the search provider has no API key, only ``web_fetch`` appears in the tool list."""
    Cfg = _import_web_tools_config()
    cfg = Cfg(
        critic_web_tools_enabled=True,
        search_provider="tavily",
        tavily_api_key="",
        fetch_enabled=True,
    )
    fetch = _RecordingFetchClient()
    fake_client = _ScriptedOpenAIClient([_ScriptedReply(text=_build_review_reply())])
    bundle = _bundle(tmp_session.name)
    fake_caller = _make_fake_runtime(judge_bundle=bundle)

    backend = CriticAgentBackend(
        critic_agent_root=REAL_CRITIC_AGENT_ROOT,
        session_dir=tmp_session,
        codex_model="gpt-test",
        codex_client_factory=lambda: fake_client,
        runtime_caller_factory=lambda: fake_caller,
        web_tools_config=cfg,
        web_tool_clients_factory=lambda _c: _FakeWebClients(
            search=None, fetch=fetch,
        ),
        static_context={"model": "Qwen3", "framework": "sglang"},
    )
    await backend.run(prompt="inbox", system_prompt="You are critic.")
    tool_names = [
        t["function"]["name"]
        for t in fake_client.completions.calls[0]["kwargs"]["tools"]
    ]
    assert tool_names == ["web_fetch"]


# Tests: web tool client raising propagates as BackendError context

@pytest.mark.asyncio
async def test_search_client_exception_is_not_swallowed_as_text(
    tmp_session: Path,
):
    """If the search client itself raises, the backend propagates rather than returning empty results."""
    search = _RecordingSearchClient(raises=RuntimeError("kaboom"))
    backend, _ = _make_backend(
        tmp_session=tmp_session,
        scripted_replies=[
            _ScriptedReply(tool_calls=[
                _tool_call("c1", "web_search", {"query": "x"}),
            ]),
            _ScriptedReply(text=_build_review_reply()),
        ],
        search_client=search,
    )
    with pytest.raises(RuntimeError, match="kaboom"):
        await backend.run(prompt="inbox", system_prompt="You are critic.")
