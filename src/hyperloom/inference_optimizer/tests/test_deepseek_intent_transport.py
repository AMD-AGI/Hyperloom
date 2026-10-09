# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DeepSeek Harness intent transport: an envelope no schema is holding the model to."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.common import deepseek_session
from hyperloom.common.deepseek_session import (
    DeepSeekSession,
    DeepSeekSessionError,
    DeepSeekSessionResult,
    DeepSeekSessionUnavailableError,
    normalize_run_result,
    usage_from_events,
)
from hyperloom.inference_optimizer.protocol.intent import IntentType, NoIntentEmitted
from hyperloom.orchestrator.roles import deepseek as deepseek_module
from hyperloom.orchestrator.roles.agent_role import _ORCHESTRATION_INTENTS
from hyperloom.orchestrator.roles.base import LLMCallFailed
from hyperloom.orchestrator.roles.deepseek import DeepSeekBackend, decode_intent_envelope

_KERNEL_REQUEST_PAYLOAD = {
    "target_agent": "kernel_agent",
    "kind": "run_optimization",
    "kernel_id": "k001",
    "backend": "forge",
}


def _reply(intent_type: str, payload: dict[str, Any]) -> str:
    """Render the envelope the output contract asks the model for."""
    return json.dumps({"intents": [{"intent_type": intent_type, "payload": json.dumps(payload)}]})


def _backend(tmp_path: Path) -> DeepSeekBackend:
    return DeepSeekBackend(
        allowed_intents=_ORCHESTRATION_INTENTS,
        model="deepseek-v4-flash",
        cwd=tmp_path / "orchestration",
    )


def _stub_turn(
    monkeypatch: pytest.MonkeyPatch,
    result: DeepSeekSessionResult | BaseException,
) -> dict[str, Any]:
    """Replace the SDK session with a recorder returning ``result``."""
    captured: dict[str, Any] = {"resets": 0}

    async def _start(session: DeepSeekSession) -> None:
        captured["instructions"] = session.instructions

    async def _turn(session: DeepSeekSession, prompt: str, *, timeout_sec: float) -> DeepSeekSessionResult:
        captured["prompt"] = prompt
        captured["timeout_sec"] = timeout_sec
        captured["instructions"] = session.instructions
        if isinstance(result, BaseException):
            raise result
        return result

    async def _aclose(session: DeepSeekSession) -> None:
        captured["closed"] = True

    def _reset(session: DeepSeekSession) -> None:
        captured["resets"] += 1

    monkeypatch.setattr(DeepSeekSession, "start", _start)
    monkeypatch.setattr(DeepSeekSession, "turn", _turn)
    monkeypatch.setattr(DeepSeekSession, "aclose", _aclose)
    monkeypatch.setattr(DeepSeekSession, "reset_conversation", _reset)
    return captured


# The reply shapes a model can emit when nothing enforces a schema.


async def test_a_bare_envelope_round_trips_through_the_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path: the model answered with exactly the object asked for."""
    _stub_turn(
        monkeypatch,
        DeepSeekSessionResult(
            text=_reply("request", _KERNEL_REQUEST_PAYLOAD),
            usage={"input_tokens": 120, "output_tokens": 8},
            session_id="session-1",
        ),
    )
    result = await _backend(tmp_path).run("kernel k001 needs optimization")

    assert len(result.intents) == 1
    intent = result.intents[0]
    assert intent.type is IntentType.REQUEST
    assert intent.payload["target_agent"] == "kernel_agent"
    assert result.metadata["input_tokens"] == 120
    assert result.metadata["session_id"] == "session-1"


async def test_an_envelope_inside_a_code_fence_is_still_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No schema holds the reply, so the model may fence it as it would any JSON."""
    fenced = "```json\n" + _reply("request", _KERNEL_REQUEST_PAYLOAD) + "\n```"
    _stub_turn(monkeypatch, DeepSeekSessionResult(text=fenced))
    result = await _backend(tmp_path).run("kernel k001 needs optimization")

    assert [intent.type for intent in result.intents] == [IntentType.REQUEST]


async def test_an_envelope_wrapped_in_prose_is_still_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovering the object beats discarding a turn whose content was fine."""
    text = "Here is my decision:\n" + _reply("request", _KERNEL_REQUEST_PAYLOAD) + "\nLet me know if that works."
    _stub_turn(monkeypatch, DeepSeekSessionResult(text=text))
    result = await _backend(tmp_path).run("kernel k001 needs optimization")

    assert [intent.type for intent in result.intents] == [IntentType.REQUEST]


def test_braces_in_the_prose_before_the_envelope_do_not_hide_it() -> None:
    """A live run lost most intents: prose like `delegate:{action_name}` was taken for the envelope."""
    envelope = _reply("alert", {"severity": "high", "summary": "explore lane stalled"})
    text = (
        "The contract says delegate:{action_name} and propose_action:{action_name,predicted_gain_pct}; "
        "an escalation is warranted.\n\n" + envelope
    )
    decoded = decode_intent_envelope(text)
    assert decoded["intents"][0]["intent_type"] == "alert"


def test_a_json_object_in_the_prose_that_is_not_an_envelope_is_skipped() -> None:
    """Quoting a payload shape in the reasoning must not stand in for the reply."""
    text = 'An example payload is {"topic": "observation"}.\n' + _reply("send_message", {"topic": "heartbeat"})
    decoded = decode_intent_envelope(text)
    assert decoded["intents"][0]["payload"] == {"topic": "heartbeat"}


def test_an_unserialized_payload_object_is_accepted() -> None:
    """The contract asks for a serialized payload; an inlined object still decodes."""
    envelope = decode_intent_envelope(json.dumps({"intents": [{"intent_type": "request", "payload": {"a": 1}}]}))
    assert envelope["intents"][0]["payload"] == {"a": 1}


def test_a_reply_with_no_json_object_is_reported_as_no_intent() -> None:
    """Prose-only is the failure this route is most exposed to; it must be named."""
    with pytest.raises(NoIntentEmitted, match="no JSON object"):
        decode_intent_envelope("I think we should wait for the next profile.")


def test_an_empty_intents_list_is_rejected() -> None:
    """An empty list would otherwise read as a successful tick that did nothing."""
    with pytest.raises(NoIntentEmitted, match="empty 'intents' list"):
        decode_intent_envelope(json.dumps({"intents": []}))


def test_a_json_reply_without_an_intents_list_is_rejected() -> None:
    """Valid JSON is not yet an envelope."""
    with pytest.raises(NoIntentEmitted, match="no 'intents' list"):
        decode_intent_envelope(json.dumps({"decision": "wait"}))


def test_an_undecodable_payload_string_is_rejected() -> None:
    """A payload that is neither JSON nor an object cannot be validated."""
    envelope = json.dumps({"intents": [{"intent_type": "request", "payload": "not json"}]})
    with pytest.raises(NoIntentEmitted, match="not decodable JSON"):
        decode_intent_envelope(envelope)


async def test_the_specialist_runner_call_shape_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live run's specialists all crashed on TypeError: the runner passes disallowed_tools."""
    _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    result = await _backend(tmp_path).run(
        prompt="scout",
        system_prompt="You are a specialist.",
        disallowed_tools=["KillShell", "SlashCommand"],
        max_turns=1,
    )

    assert [intent.type for intent in result.intents] == [IntentType.SEND_MESSAGE]


# What the harness is told, given it has no field for a system prompt.


async def test_instructions_carry_the_runtime_mapping_the_contract_and_the_system_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness exposes no system-prompt field, so all three ride the session."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    await _backend(tmp_path).run("status?", system_prompt="You are the Coordinator.")

    instructions = captured["instructions"]
    assert "DeepSeek Harness runtime mapping" in instructions
    assert "You are the Coordinator." in instructions
    assert "OUTPUT FORMAT" in instructions


async def test_instructions_name_every_legal_intent_type_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No schema enum holds the model to the set, so the text has to; a live run emitted 'send_message:kernel'."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    await _backend(tmp_path).run("status?")

    legal = ", ".join(sorted(t.value for t in _ORCHESTRATION_INTENTS))
    assert f"`intent_type` MUST be exactly one of: {legal}." in captured["instructions"]


async def test_every_turn_restates_the_envelope_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instructions ride the first message only, so a later turn needs reminding."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    await _backend(tmp_path).run("status?")

    assert deepseek_module._ENVELOPE_REMINDER in captured["prompt"]


async def test_every_turn_spells_out_the_intent_object_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compaction folds the first message away; a live run then wrote 'intent_payload' for 'payload' on every tick."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    await _backend(tmp_path).run("status?")

    assert '{"intents": [{"intent_type": "<type>", "payload": {<JSON object>}}]}' in captured["prompt"]


async def test_the_payload_is_asked_for_as_an_inline_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No schema forces a string here; a live run broke the escaping of a serialized payload on 1 tick in 5."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    await _backend(tmp_path).run("status?")

    assert "Do not serialize it into a string" in captured["instructions"]
    assert "serialized JSON object" not in captured["instructions"]
    assert "not a string" in captured["prompt"]


def test_an_intent_without_a_payload_key_names_the_keys_it_carried() -> None:
    envelope = json.dumps({"intents": [{"intent_type": "alert", "intent_payload": {"severity": "low"}}]})
    with pytest.raises(NoIntentEmitted, match=r"no 'payload' key \(keys: intent_payload, intent_type\)"):
        decode_intent_envelope(envelope)


async def test_a_changed_system_prompt_reopens_the_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A running conversation cannot be told about a new set of instructions."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    backend = _backend(tmp_path)
    await backend.run("status?", system_prompt="You are the Coordinator.")
    assert captured["resets"] == 0
    await backend.run("status?", system_prompt="You are the Critic.")
    assert captured["resets"] == 1


async def test_a_checkpoint_turn_drops_the_contract_and_parses_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``allow_no_intent`` asks for a different shape, so the envelope must not be required."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text="a plain summary, no envelope"))
    result = await _backend(tmp_path).run("summarize", allow_no_intent=True)

    assert deepseek_module._PER_TURN_CONTRACT_OVERRIDE in captured["prompt"]
    assert result.intents == []
    assert result.raw_text == "a plain summary, no envelope"


# Failures the harness reports without failing the call.


async def test_a_session_error_becomes_an_llm_call_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reactor distinguishes a transport failure from a model that said nothing."""
    _stub_turn(monkeypatch, DeepSeekSessionError("runtime died"))
    with pytest.raises(LLMCallFailed, match="runtime died"):
        await _backend(tmp_path).run("status?")


async def test_a_reported_turn_error_becomes_an_llm_call_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness completes the run and reports the provider fault in the result."""
    _stub_turn(monkeypatch, DeepSeekSessionResult(text="", error="upstream_error: gateway refused"))
    with pytest.raises(LLMCallFailed, match="gateway refused"):
        await _backend(tmp_path).run("status?")


def test_a_spent_output_cap_is_surfaced_rather_than_read_as_silence() -> None:
    """``max-tokens`` completes the run, so an unexamined result looks like no reply."""
    result = normalize_run_result(SimpleNamespace(final_response="", finish_reason="max-tokens", events=[]), "s1")
    assert "output token cap" in result.error


def test_a_turn_end_error_is_quoted_in_the_result() -> None:
    """The reason the harness gives is more useful than the finish_reason alone."""
    events = [{"type": "turn/end", "data": {"reason": {"error": {"code": "E_AUTH", "message": "bad key"}}}}]
    result = normalize_run_result(SimpleNamespace(final_response="", finish_reason="error", events=events), "s1")
    assert result.error == "E_AUTH: bad key"


# What the runtime is started with.


def _pin_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure exactly one OpenAI-compatible gateway; a session cannot start without one."""
    for key in _GATEWAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "gateway-key")


def _capture_harness_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the SDK with one whose harness records its construction kwargs."""
    captured: dict[str, Any] = {}

    class _Harness:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(deepseek_session, "_load_harness_sdk", lambda: SimpleNamespace(DeepSeekHarness=_Harness))
    return captured


async def test_an_unset_effort_starts_the_runtime_at_a_level_it_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness refuses an empty effort at initialization, before any prompt runs."""
    _pin_gateway(monkeypatch)
    captured = _capture_harness_kwargs(monkeypatch)
    await DeepSeekSession(cwd=tmp_path, model="deepseek-v4-flash", dsh_home=tmp_path / "home").start()

    assert captured["reasoning_effort"] in {"low", "high", "max"}


async def test_an_effort_the_harness_lacks_is_rounded_up_to_one_it_has(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``medium`` is not a harness level; asking for more deliberation must not yield less."""
    _pin_gateway(monkeypatch)
    captured = _capture_harness_kwargs(monkeypatch)
    await DeepSeekSession(
        cwd=tmp_path, model="deepseek-v4-flash", dsh_home=tmp_path / "home", reasoning_effort="medium"
    ).start()

    assert captured["reasoning_effort"] == "high"


_GATEWAY_ENV_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "LLM_GATEWAY_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
)


async def test_the_gateway_reaches_the_harness_as_the_credential_its_adapter_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter reads DEEPSEEK_*; the gateway is configured on the OpenAI-compatible line."""
    _pin_gateway(monkeypatch)
    captured = _capture_harness_kwargs(monkeypatch)
    await DeepSeekSession(cwd=tmp_path, model="deepseek-v4-flash", dsh_home=tmp_path / "home").start()

    assert captured["env"]["DEEPSEEK_BASE_URL"] == "https://gateway.invalid/v1"
    assert captured["env"]["DEEPSEEK_API_KEY"] == "gateway-key"


async def test_an_unconfigured_gateway_fails_at_start_not_at_the_first_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise every turn completes with MISSING_CREDENTIAL and reads as a model that said nothing."""
    for key in _GATEWAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    _capture_harness_kwargs(monkeypatch)
    with pytest.raises(DeepSeekSessionUnavailableError, match="OPENAI_BASE_URL"):
        await DeepSeekSession(cwd=tmp_path, model="deepseek-v4-flash", dsh_home=tmp_path / "home").start()


async def test_a_timed_out_turn_is_stopped_and_the_next_one_runs_on_a_fresh_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK has no per-turn cancel, so a timed-out turn keeps spending until its runtime is closed."""
    import threading

    _pin_gateway(monkeypatch)
    release = threading.Event()
    harnesses: list[Any] = []

    class _Harness:
        def __init__(self, **_kwargs: Any) -> None:
            self.closed = False
            harnesses.append(self)

        def run(self, _body: str, *, session_id: str) -> Any:
            if len(harnesses) == 1:
                release.wait(timeout=5)
            return SimpleNamespace(final_response="ok", finish_reason="completed", events=[], session_id=session_id)

        def close(self) -> None:
            self.closed = True
            release.set()

    monkeypatch.setattr(deepseek_session, "_load_harness_sdk", lambda: SimpleNamespace(DeepSeekHarness=_Harness))
    session = DeepSeekSession(cwd=tmp_path, model="deepseek-v4-flash", dsh_home=tmp_path / "home")
    await session.start()

    with pytest.raises(deepseek_session.DeepSeekSessionTimeoutError):
        await session.turn("slow", timeout_sec=0.05)
    assert harnesses[0].closed

    result = await session.turn("fast", timeout_sec=5)
    assert len(harnesses) == 2
    assert result.text == "ok"
    await session.aclose()


# Token accounting.


def test_per_step_counts_are_summed_and_the_running_total_is_ignored() -> None:
    """``totalTokens`` is a turn-wide running total; summing it double-counts."""
    events = [
        {"type": "assistant/message", "data": {"usage": {"inputTokens": 100, "outputTokens": 5, "totalTokens": 105}}},
        {"type": "assistant/message", "data": {"usage": {"inputTokens": 110, "outputTokens": 7, "totalTokens": 222}}},
    ]
    usage = usage_from_events(events)
    assert usage["input_tokens"] == 210
    assert usage["output_tokens"] == 12
    assert "total_tokens" not in usage


def test_the_context_peak_is_the_largest_single_step_request() -> None:
    """Each step resends the whole context; summing steps overstates it, and inputTokens alone omits the cache."""
    events = [
        {"type": "assistant/message", "data": {"usage": {"inputTokens": 100, "cacheReadTokens": 900}}},
        {"type": "assistant/message", "data": {"usage": {"inputTokens": 50, "cacheReadTokens": 1000}}},
    ]
    usage = usage_from_events(events)
    assert usage["context_tokens_peak"] == 1050
    assert usage["input_tokens"] == 150


async def test_the_reported_context_peak_reaches_the_checkpoint_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = {"input_tokens": 150, "cache_read_input_tokens": 1900, "context_tokens_peak": 1050}
    _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "x"}), usage=usage))
    result = await _backend(tmp_path).run("status?")

    assert result.metadata["context_tokens_peak"] == 1050


async def test_an_open_trajectory_span_owns_the_call_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The token row and the trajectory row of one turn are joined on this id."""
    from hyperloom.inference_optimizer.trace.trajectory_trace import trajectory_scope

    _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "x"})))
    with trajectory_scope(call_id="span-call-1"):
        result = await _backend(tmp_path).run("status?")

    assert result.metadata["call_id"] == "span-call-1"


def test_a_turn_reporting_no_usage_reports_no_fields() -> None:
    """Zeroed counts would be indistinguishable from a turn that really cost nothing."""
    assert usage_from_events([{"type": "tool/call", "data": {"name": "read"}}]) == {}
