# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DeepSeek Harness intent transport: an envelope no schema is holding the model to."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from common import deepseek_session
from common.deepseek_session import (
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


def _stub_turns(monkeypatch: pytest.MonkeyPatch, results: list[DeepSeekSessionResult]) -> list[str]:
    """Replace the SDK session with one replying ``results`` in order; returns the prompts it was sent."""
    prompts: list[str] = []
    _stub_turn(monkeypatch, DeepSeekSessionResult(text=""))

    async def _turn(session: DeepSeekSession, prompt: str, *, timeout_sec: float) -> DeepSeekSessionResult:
        prompts.append(prompt)
        return results[len(prompts) - 1]

    monkeypatch.setattr(DeepSeekSession, "turn", _turn)
    return prompts


async def test_a_rejected_envelope_is_corrected_in_the_same_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-backed transport hands the validation error back to the model; this route must too."""
    rejected = _reply("alert", {"severity": "info", "summary": "holding"})
    corrected = _reply("alert", {"severity": "low", "summary": "holding"})
    prompts = _stub_turns(
        monkeypatch,
        [DeepSeekSessionResult(text=rejected, session_id="s1"), DeepSeekSessionResult(text=corrected, session_id="s1")],
    )

    result = await _backend(tmp_path).run("status?")

    assert [intent.type for intent in result.intents] == [IntentType.ALERT]
    assert len(prompts) == 2
    assert "severity must be one of" in prompts[1]


async def test_a_correction_that_is_still_rejected_fails_the_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = _stub_turns(
        monkeypatch,
        [DeepSeekSessionResult(text="no envelope", session_id="s1"), DeepSeekSessionResult(text="still none")],
    )

    with pytest.raises(NoIntentEmitted, match="no JSON object"):
        await _backend(tmp_path).run("status?")
    assert len(prompts) == 2


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


# What the harness is told, given it has no field for a system prompt.


async def test_instructions_and_every_turn_carry_the_whole_envelope_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness has no system-prompt field and no schema, so the wording is all that holds the model to the shape."""
    captured = _stub_turn(monkeypatch, DeepSeekSessionResult(text=_reply("send_message", {"topic": "heartbeat"})))
    await _backend(tmp_path).run("status?", system_prompt="You are the Coordinator.")

    instructions = captured["instructions"]
    legal = ", ".join(sorted(t.value for t in _ORCHESTRATION_INTENTS))
    # Carried on the first message: the runtime mapping, the system prompt, the output format, the exact legal names
    # (a live run emitted 'send_message:kernel'), and a payload written inline (a serialized one broke on 1 tick in 5).
    assert deepseek_session.DEEPSEEK_RUNTIME_MAPPING.strip() in instructions
    assert "You are the Coordinator." in instructions
    assert "OUTPUT FORMAT" in instructions
    assert f"`intent_type` MUST be exactly one of: {legal}." in instructions
    assert "Do not serialize it into a string" in instructions
    # Instructions ride the first message only and compaction folds it away, so each turn restates the contract.
    prompt = captured["prompt"]
    assert deepseek_module._ENVELOPE_REMINDER in prompt
    assert '{"intents": [{"intent_type": "<type>", "payload": {<JSON object>}}]}' in prompt
    assert "not a string" in prompt


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


class _FakeHarnessError(Exception):
    """Stands in for ``deepseek_harness.errors.HarnessError``."""


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, harness_cls: type) -> None:
    """Replace the SDK module with one whose harness class is ``harness_cls``."""
    sdk = SimpleNamespace(DeepSeekHarness=harness_cls, errors=SimpleNamespace(HarnessError=_FakeHarnessError))
    monkeypatch.setattr(deepseek_session, "load_harness_sdk", lambda: sdk)


def _capture_harness_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the SDK with one whose harness records its construction kwargs."""
    captured: dict[str, Any] = {}

    class _Harness:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def start(self) -> None:
            captured["started"] = True

    _install_fake_sdk(monkeypatch, _Harness)
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


class _RecordingHarness:
    """A harness that records its lifecycle; subclasses decide what ``run`` and ``start`` do."""

    instances: list[Any]

    def __init__(self, **_kwargs: Any) -> None:
        self.closed = False
        self.instances.append(self)

    def start(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def run(self, _body: str, *, session_id: str) -> Any:
        return SimpleNamespace(final_response="ok", finish_reason="completed", events=[], session_id=session_id)


def _recording_harness(monkeypatch: pytest.MonkeyPatch, **behaviour: Any) -> list[Any]:
    """Install a recording harness whose methods are overridden by ``behaviour``; return its instances."""
    instances: list[Any] = []
    namespace = {"instances": instances, **behaviour}
    _install_fake_sdk(monkeypatch, type("_Harness", (_RecordingHarness,), namespace))
    return instances


def _session(tmp_path: Path) -> DeepSeekSession:
    return DeepSeekSession(cwd=tmp_path, model="deepseek-v4-flash", dsh_home=tmp_path / "home")


async def test_a_timed_out_turn_is_stopped_and_the_next_one_runs_on_a_fresh_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK has no per-turn cancel, so a timed-out turn keeps spending until its runtime is closed."""
    import threading

    _pin_gateway(monkeypatch)
    release = threading.Event()

    def _run(self: Any, _body: str, *, session_id: str) -> Any:
        if len(instances) == 1:
            release.wait(timeout=5)
        return SimpleNamespace(final_response="ok", finish_reason="completed", events=[], session_id=session_id)

    def _close(self: Any) -> None:
        self.closed = True
        release.set()

    instances = _recording_harness(monkeypatch, run=_run, close=_close)
    session = _session(tmp_path)
    await session.start()

    with pytest.raises(deepseek_session.DeepSeekSessionTimeoutError):
        await session.turn("slow", timeout_sec=0.05)
    assert instances[0].closed

    result = await session.turn("fast", timeout_sec=5)
    assert len(instances) == 2
    assert result.text == "ok"
    await session.aclose()


async def test_a_dead_runtime_is_closed_so_the_next_turn_does_not_write_to_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the runtime process is gone every later turn fails on the same write; only a new runtime recovers."""
    _pin_gateway(monkeypatch)

    def _run(self: Any, _body: str, *, session_id: str) -> Any:
        if len(instances) == 1:
            raise _FakeHarnessError("Failed to write to DeepSeek Harness runtime")
        return SimpleNamespace(final_response="ok", finish_reason="completed", events=[], session_id=session_id)

    instances = _recording_harness(monkeypatch, run=_run)
    session = _session(tmp_path)

    with pytest.raises(DeepSeekSessionError, match="Failed to write"):
        await session.turn("first", timeout_sec=5)
    assert instances[0].closed

    assert (await session.turn("second", timeout_sec=5)).text == "ok"
    assert len(instances) == 2
    await session.aclose()


async def test_a_cancelled_turn_stops_its_runtime_instead_of_leaving_it_spending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coordinator cancels a turn with ``wait_for``; the abandoned thread would otherwise keep running and billing."""
    import asyncio
    import threading

    _pin_gateway(monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    def _run(self: Any, _body: str, *, session_id: str) -> Any:
        entered.set()
        release.wait(timeout=5)
        return SimpleNamespace(final_response="late", finish_reason="completed", events=[], session_id=session_id)

    def _close(self: Any) -> None:
        self.closed = True
        release.set()

    instances = _recording_harness(monkeypatch, run=_run, close=_close)
    session = _session(tmp_path)
    task = asyncio.create_task(session.turn("slow", timeout_sec=30))
    while not entered.is_set():
        await asyncio.sleep(0.01)

    task.cancel()
    await asyncio.wait([task])
    assert task.cancelled()
    assert instances[0].closed


async def test_start_runs_the_handshake_so_a_bad_configuration_is_not_retried_as_transient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK starts lazily, so constructing it proves nothing: a missing binary or a bogus model must fail here."""
    _pin_gateway(monkeypatch)

    def _start(self: Any) -> None:
        raise _FakeHarnessError("initialize refused model 'bogus'")

    instances = _recording_harness(monkeypatch, start=_start)
    with pytest.raises(DeepSeekSessionUnavailableError, match="initialize refused"):
        await _session(tmp_path).start()
    # The runtime the failed handshake spawned is stopped, not left running.
    assert instances[0].closed


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
