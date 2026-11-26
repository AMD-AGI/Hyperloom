# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests, part 5 (narrator and "Explain this"): the narrator calls Hyperloom's LLM configuration with the
arguments it declares, and "Explain this" names a journal change as Hyperloom's journal names it.

Calls into ``hyperloom_common.llm.llm_config`` are checked against the real functions' signatures (a stand-in only
replaces the network), and the OpenAI path runs through the real ``chat_completion``. A failure means a producer
changed an API or a format: update the Inspector code the failure message names, then this test. A test marked
``xfail(strict=True)`` pins a known break; it starts failing ("XPASS(strict)") once the Inspector is fixed, and then
the marker comes off.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inspector.tests.producers import contract_message

pytestmark = pytest.mark.inspector_contract

_NARRATOR = "narrator.py"


def _bound_stand_in(real: Any, result: Any, calls: list) -> Any:
    """A stand-in for *real* that refuses arguments *real*'s signature would refuse, records them, and returns
    *result*."""
    signature = inspect.signature(real)

    def stand_in(*args: Any, **kwargs: Any) -> Any:
        try:
            calls.append(signature.bind(*args, **kwargs).arguments)
        except TypeError as exc:
            raise AssertionError(contract_message(f"llm_config.{real.__name__}", _NARRATOR, str(exc))) from exc
        return result

    return stand_in


def _anthropic_call(monkeypatch, stop_reason: str) -> tuple[str, dict]:
    from hyperloom_common.llm import llm_config

    from hyperloom.inspector.narrator import _AnthropicClient

    calls: list = []
    result = llm_config.AnthropicMessageResult(text="the run is in EXPLORE", stop_reason=stop_reason, usage=None)
    monkeypatch.setattr(
        llm_config, "anthropic_completion", _bound_stand_in(llm_config.anthropic_completion, result, calls)
    )
    text = _AnthropicClient("claude-opus-4-1", 256).complete(system="s", user="u", operation="narrate")
    return text, calls[0]


def test_narrator_calls_anthropic_completion_as_llm_config_declares(monkeypatch) -> None:
    from hyperloom_common.llm import llm_config

    text, call = _anthropic_call(monkeypatch, "end_turn")
    request = call["request"]
    assert isinstance(request, llm_config.AnthropicCompletionRequest), contract_message(
        "llm_config.AnthropicCompletionRequest", _NARRATOR, repr(request)
    )
    assert (request.model, request.max_tokens, request.system) == ("claude-opus-4-1", 256, "s"), contract_message(
        "llm_config.AnthropicCompletionRequest", _NARRATOR, repr(request)
    )
    assert call["operation"] == "narrate" and text == "the run is in EXPLORE", contract_message(
        "llm_config.anthropic_completion / AnthropicMessageResult", _NARRATOR, f"{call} -> {text!r}"
    )
    assert "token limit" in _anthropic_call(monkeypatch, "max_tokens")[0], contract_message(
        "AnthropicMessageResult.stop_reason", _NARRATOR, "a reply cut at max_tokens is not marked"
    )


def _fake_openai_client(calls: list, finish_reason: str) -> Any:
    def create(**params: Any) -> Any:
        calls.append(params)
        choice = SimpleNamespace(message=SimpleNamespace(content="two variants failed"), finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], usage=None)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_narrator_calls_the_real_chat_completion() -> None:
    from hyperloom.inspector.narrator import _OpenAIClient

    calls: list = []
    text = _OpenAIClient(_fake_openai_client(calls, "stop"), "gpt-5", 256).complete(
        system="s", user="u", operation="ask"
    )
    assert text == "two variants failed", contract_message("llm_config.chat_completion", _NARRATOR, repr(text))
    sent = calls[0]
    assert sent["model"] == "gpt-5" and sent["messages"][-1] == {"role": "user", "content": "u"}, contract_message(
        "llm_config.chat_completion", _NARRATOR, repr(sent)
    )
    cut = _OpenAIClient(_fake_openai_client([], "length"), "gpt-5", 256).complete(system="s", user="u", operation="ask")
    assert "token limit" in cut, contract_message("ChatCompletionResult.finish_reason", _NARRATOR, repr(cut))


def test_narrator_builds_its_clients_with_llm_config(monkeypatch) -> None:
    from hyperloom_common.llm import llm_config

    from hyperloom.inspector.narrator import build_client

    calls: list = []
    for name, result in (("build_http_timeout", None), ("get_openai_client", object())):
        monkeypatch.setattr(llm_config, name, _bound_stand_in(getattr(llm_config, name), result, calls))
    client, info = build_client({"HYPERLOOM_INSPECTOR_LLM_BACKEND": "openai", "HYPERLOOM_INSPECTOR_LLM_MODEL": "m"})
    assert client is not None and info.backend == "openai" and len(calls) == 2, contract_message(
        "llm_config.build_http_timeout / get_openai_client", _NARRATOR, f"{info} {calls}"
    )
    _client, default = build_client({"HYPERLOOM_INSPECTOR_LLM_BACKEND": "openai"})
    assert default.model == llm_config.DEFAULT_CLAUDE_MODEL, contract_message(
        "llm_config.DEFAULT_CLAUDE_MODEL", _NARRATOR, default
    )
    assert issubclass(llm_config.LLMConfigError, Exception) and callable(llm_config.anthropic_transport_ready), (
        contract_message("llm_config.LLMConfigError / anthropic_transport_ready", _NARRATOR, "")
    )


def _openai_call(monkeypatch) -> dict:
    from hyperloom_common.llm import llm_config

    from hyperloom.inspector.narrator import _OpenAIClient

    calls: list = []
    result = llm_config.ChatCompletionResult("two variants failed", "stop", None)
    monkeypatch.setattr(llm_config, "chat_completion", _bound_stand_in(llm_config.chat_completion, result, calls))
    _OpenAIClient(object(), "gpt-5", 256).complete(system="s", user="u", operation="ask")
    return calls[0]


@pytest.mark.parametrize("trace", ["llm_trace", "trajectory_trace"])
def test_the_narrators_component_is_one_hyperloom_traces(trace: str) -> None:
    """Hyperloom's traces keep a closed set of LLM components, and a label outside it is not attributed.

    The trajectory ledger drops an unknown component from a request's row, so the row of a narrator request would
    carry whichever component happens to be ambient, or none; upstream's ``test_llm_attribution_coverage`` fails on
    it too. Add the label to ``VALID_COMPONENTS`` in ``hyperloom.inference_optimizer.trace``, or change
    ``narrator._COMPONENT`` to one that is there.
    """
    from importlib import import_module

    from hyperloom.inspector.narrator import _COMPONENT

    vocabulary = import_module(f"hyperloom.inference_optimizer.trace.{trace}").VALID_COMPONENTS
    assert _COMPONENT in vocabulary, contract_message(
        f"{trace}.VALID_COMPONENTS", _NARRATOR, f"_COMPONENT={_COMPONENT!r} not in {sorted(vocabulary)}"
    )


@pytest.mark.parametrize("backend", ["anthropic", "openai"])
def test_each_narrator_client_attributes_its_calls_to_that_component(backend: str, monkeypatch) -> None:
    """The component a client actually sends is the constant, so the test above covers what reaches the gateway.

    Hyperloom's own scan of call sites (``test_llm_attribution_coverage``) only reads a component written as a
    literal, so a call that names it through a constant, as these do, is checked here instead.
    """
    from hyperloom.inference_optimizer.trace.llm_trace import VALID_COMPONENTS

    from hyperloom.inspector.narrator import _COMPONENT

    call = _anthropic_call(monkeypatch, "end_turn")[1] if backend == "anthropic" else _openai_call(monkeypatch)
    assert call["component"] == _COMPONENT and _COMPONENT in VALID_COMPONENTS, contract_message(
        "llm_trace.VALID_COMPONENTS", _NARRATOR, f"component={call['component']!r}"
    )
    assert call["operation"] == ("narrate" if backend == "anthropic" else "ask"), contract_message(
        "llm_config attribution", _NARRATOR, f"operation={call['operation']!r}"
    )


# Variants as a grid entry carries them: server args, env vars, both, or a name alone.
_VARIANTS = [
    {"name": "mnbt", "extra_server_args": "--max-num-batched-tokens 8192"},
    {"name": "aiter", "extra_envs": {"VLLM_ROCM_USE_AITER": "1"}},
    {"name": "both", "extra_server_args": " --tp 2 ", "extra_envs": {"A": "1", "B": "2"}},
    {"name": "named", "extra_server_args": "", "extra_envs": {}},
]


@pytest.mark.parametrize("variant", _VARIANTS, ids=[v["name"] for v in _VARIANTS])
def test_recipe_change_matches_the_journal_s_change(variant: dict) -> None:
    from hyperloom.inference_optimizer.session.optimization_journal import summarize_change

    from hyperloom.inspector.explain import _recipe_change

    assert _recipe_change(variant) == summarize_change("explore", variant), contract_message(
        "optimization_journal.summarize_change", "explain.py (_recipe_change)", repr(variant)
    )
