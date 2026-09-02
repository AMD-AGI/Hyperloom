# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codex intent transport: one schema, derived from the role's own intent set.

Regression cover for a measured OpenAI-only run in which the Coordinator
emitted 0 valid ``request`` intents over 256 turns. The Claude path derives the
legal ``intent_type`` list programmatically from :class:`IntentType`, while the
Codex path hand-wrote 5 of the 12 intents the orchestration role may emit, so
``request`` (the only way to reach the kernel agent) had no transport at all.
The model fell back to ``propose_action``, which PolicyGate denied 44 times
with ``rule=kernel_owned_by_kernel_agent``.

These tests pin the properties that make that drift impossible: the enum comes
from the role record, the schema satisfies the provider's strict structured
output rules, the shared transport text carries no other role's contract, and a
``request`` intent survives the whole round trip.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.common import codex_session
from hyperloom.common.codex_session import CodexSession, CodexSessionError, CodexSessionResult
from hyperloom.inference_optimizer.protocol.intent import IntentType, NoIntentEmitted
from hyperloom.orchestrator.roles import codex as codex_module
from hyperloom.orchestrator.roles.agent_role import (
    _CRITIC_INTENTS,
    _ORCHESTRATION_INTENTS,
    _ROBUSTNESS_INTENTS,
    default_role_registry,
)
from hyperloom.orchestrator.roles.base import LLMCallFailed
from hyperloom.orchestrator.roles.codex import CodexBackend
from hyperloom.orchestrator.roles.mcp_emit_intent import build_intent_envelope_schema


def _intent_type_enum(schema: dict[str, Any]) -> list[str]:
    """Pull the ``intent_type`` enum out of a generated envelope schema."""
    return schema["properties"]["intents"]["items"]["properties"]["intent_type"]["enum"]


def _object_nodes(node: Any) -> list[dict[str, Any]]:
    """Every ``{"type": "object"}` subschema in a generated schema, depth first."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_object_nodes(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_object_nodes(item))
    return found


# ---------------------------------------------------------------------------
# Drift guard: the enum is the role's intent set, never a hand-written literal.


def test_orchestration_schema_offers_every_intent_the_role_may_emit() -> None:
    """The Coordinator can only emit what the schema lists, so it must list all."""
    schema = build_intent_envelope_schema(_ORCHESTRATION_INTENTS)
    assert set(_intent_type_enum(schema)) == {t.value for t in _ORCHESTRATION_INTENTS}


def test_request_is_reachable_from_the_orchestration_schema() -> None:
    """``request`` is the only legal route to the kernel agent; it must be offered."""
    assert "request" in _intent_type_enum(build_intent_envelope_schema(_ORCHESTRATION_INTENTS))


@pytest.mark.parametrize(
    ("role_name", "expected"),
    [
        ("orchestration", _ORCHESTRATION_INTENTS),
        ("critic", _CRITIC_INTENTS),
        ("robustness", _ROBUSTNESS_INTENTS),
    ],
)
def test_every_role_schema_matches_its_registry_record(role_name: str, expected: frozenset[IntentType]) -> None:
    """Adding an IntentType or widening a role must regenerate, never diverge."""
    role = default_role_registry()[role_name]
    assert role.allowed_intents == expected
    schema = build_intent_envelope_schema(role.allowed_intents)
    assert set(_intent_type_enum(schema)) == {t.value for t in expected}


def test_schema_enum_is_ordered_and_deduplicated() -> None:
    """A frozenset has no order; the wire schema must still be stable."""
    first = _intent_type_enum(build_intent_envelope_schema(_ORCHESTRATION_INTENTS))
    second = _intent_type_enum(build_intent_envelope_schema(frozenset(reversed(list(_ORCHESTRATION_INTENTS)))))
    assert first == second
    assert len(first) == len(set(first))


def test_empty_intent_set_is_rejected() -> None:
    """An empty enum would be a schema the model can never satisfy."""
    with pytest.raises(ValueError, match="at least one"):
        build_intent_envelope_schema(frozenset())


# ---------------------------------------------------------------------------
# Provider constraint: OpenAI strict structured outputs.


def test_generated_schema_satisfies_strict_structured_outputs() -> None:
    """Azure rejects any object that omits additionalProperties or a required key."""
    schema = build_intent_envelope_schema(_ORCHESTRATION_INTENTS)
    nodes = _object_nodes(schema)
    assert nodes, "the envelope schema must contain object nodes"
    for node in nodes:
        assert node.get("additionalProperties") is False
        assert sorted(node.get("required", [])) == sorted(node.get("properties", {}))


def test_payload_is_carried_as_a_json_string() -> None:
    """A free-form ``{"type": "object"}`` payload is rejected in strict mode."""
    schema = build_intent_envelope_schema(_ORCHESTRATION_INTENTS)
    payload = schema["properties"]["intents"]["items"]["properties"]["payload"]
    assert payload["type"] == "string"


def test_schema_is_json_serializable() -> None:
    """It travels to the provider as JSON, so it must survive a round trip."""
    schema = build_intent_envelope_schema(_ORCHESTRATION_INTENTS)
    assert json.loads(json.dumps(schema)) == schema


# ---------------------------------------------------------------------------
# No cross-role contamination in the shared transport text.


def test_orchestration_transport_contract_carries_no_critic_text() -> None:
    """The Critic's review contract leaked into all 256 Coordinator turns."""
    instructions = codex_module.build_output_instructions(_ORCHESTRATION_INTENTS)
    blob = f"{instructions}\n{json.dumps(build_intent_envelope_schema(_ORCHESTRATION_INTENTS))}".lower()
    for token in ("critic", "review_verdict", "target_proposal_msg_id", "needs_review"):
        assert token not in blob


def test_transport_contract_names_every_allowed_intent() -> None:
    """The model needs the vocabulary in prose as well as in the enforced schema."""
    instructions = codex_module.build_output_instructions(_ORCHESTRATION_INTENTS)
    for intent_type in _ORCHESTRATION_INTENTS:
        assert intent_type.value in instructions


def test_critic_transport_contract_carries_no_orchestration_text() -> None:
    """Role scoping has to hold in both directions."""
    instructions = codex_module.build_output_instructions(_CRITIC_INTENTS).lower()
    for token in ("delegate", "propose_action", "prune_branch"):
        assert token not in instructions


# ---------------------------------------------------------------------------
# End-to-end round trip through the backend with a mocked Codex SDK.


_KERNEL_REQUEST_PAYLOAD = {
    "target_agent": "kernel_agent",
    "kind": "run_optimization",
    "kernel_id": "k001",
    "backend": "forge",
}


def _schema_enforced_reply(intent_type: str, payload: dict[str, Any]) -> str:
    """Render the exact wire shape ``output_schema`` forces the model to emit."""
    return json.dumps({"intents": [{"intent_type": intent_type, "payload": json.dumps(payload)}]})


def _backend(tmp_path: Path) -> CodexBackend:
    return CodexBackend(
        allowed_intents=_ORCHESTRATION_INTENTS,
        model="gpt-5.6-sol",
        cwd=tmp_path / "orchestration",
        sandbox_mode="workspace-write",
    )


def _stub_turn(monkeypatch: pytest.MonkeyPatch, result: CodexSessionResult | BaseException) -> dict[str, Any]:
    """Replace the SDK session with a recorder returning ``result``.

    Patches the real :class:`CodexSession` methods rather than substituting a
    fake class, so the recorded developer instructions and schema are the ones
    the backend actually configured.
    """
    captured: dict[str, Any] = {}

    async def _start(session: CodexSession) -> None:
        captured["developer_instructions"] = session.developer_instructions

    async def _turn(
        session: CodexSession,
        prompt: str,
        *,
        timeout_sec: float,
        output_schema: dict[str, Any] | None = None,
    ) -> CodexSessionResult:
        captured["prompt"] = prompt
        captured["timeout_sec"] = timeout_sec
        captured["output_schema"] = output_schema
        if isinstance(result, BaseException):
            raise result
        return result

    async def _aclose(session: CodexSession) -> None:
        captured["closed"] = True

    monkeypatch.setattr(CodexSession, "start", _start)
    monkeypatch.setattr(CodexSession, "turn", _turn)
    monkeypatch.setattr(CodexSession, "aclose", _aclose)
    return captured


async def test_request_intent_round_trips_through_the_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intent the Coordinator could never express now survives end to end."""
    captured = _stub_turn(
        monkeypatch,
        CodexSessionResult(
            text=_schema_enforced_reply("request", _KERNEL_REQUEST_PAYLOAD),
            usage={"input_tokens": 120, "output_tokens": 8},
            thread_id="thread-1",
        ),
    )
    result = await _backend(tmp_path).run("kernel k001 needs optimization", system_prompt="You are the Coordinator.")

    assert set(_intent_type_enum(captured["output_schema"])) == {t.value for t in _ORCHESTRATION_INTENTS}
    assert len(result.intents) == 1
    intent = result.intents[0]
    assert intent.type is IntentType.REQUEST
    assert intent.payload["target_agent"] == "kernel_agent"
    assert intent.payload["kind"] == "run_optimization"
    assert result.metadata["input_tokens"] == 120


async def test_system_prompt_and_contract_reach_the_developer_instructions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role separation is native on the SDK; the prompt stays the user turn."""
    captured = _stub_turn(
        monkeypatch,
        CodexSessionResult(text=_schema_enforced_reply("send_message", {"topic": "heartbeat"})),
    )
    await _backend(tmp_path).run("tick 7", system_prompt="SYSTEM_SENTINEL")

    assert "SYSTEM_SENTINEL" in captured["developer_instructions"]
    assert "OUTPUT FORMAT" in captured["developer_instructions"]
    assert captured["prompt"] == "tick 7"


async def test_reply_that_is_not_the_enforced_shape_raises_no_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded gateway must fail loudly, not silently drop the turn."""
    _stub_turn(monkeypatch, CodexSessionResult(text="I cannot decide right now."))
    with pytest.raises(NoIntentEmitted):
        await _backend(tmp_path).run("p")


async def test_payload_that_is_not_a_json_object_raises_no_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The payload string is model-authored, so decoding it is a validation step."""
    reply = json.dumps({"intents": [{"intent_type": "send_message", "payload": "not json"}]})
    _stub_turn(monkeypatch, CodexSessionResult(text=reply))
    with pytest.raises(NoIntentEmitted, match="payload"):
        await _backend(tmp_path).run("p")


async def test_payload_missing_a_required_field_raises_no_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload checking stays shared with the Claude path's validate_envelope."""
    _stub_turn(monkeypatch, CodexSessionResult(text=_schema_enforced_reply("request", {"target_agent": "critic"})))
    with pytest.raises(NoIntentEmitted, match="kind"):
        await _backend(tmp_path).run("p")


async def test_sdk_failure_is_reported_as_an_llm_call_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only provider-call failures belong in the LLM error rate."""
    _stub_turn(monkeypatch, CodexSessionError("gateway 500"))
    with pytest.raises(LLMCallFailed, match="gateway 500"):
        await _backend(tmp_path).run("p")


async def test_in_band_turn_error_is_reported_as_an_llm_call_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider-side failure can complete the turn without an answer."""
    _stub_turn(monkeypatch, CodexSessionResult(text="", error="model overloaded"))
    with pytest.raises(LLMCallFailed, match="model overloaded"):
        await _backend(tmp_path).run("p")


# ---------------------------------------------------------------------------
# The session wrapper actually forwards the schema to the SDK.


class _FakeTurnHandle:
    def __init__(self, recorder: dict[str, Any]) -> None:
        self._recorder = recorder

    async def run(self) -> SimpleNamespace:
        return SimpleNamespace(
            final_response=self._recorder["reply"],
            items=(),
            usage=None,
            error=None,
        )

    async def interrupt(self) -> None:
        return None


class _FakeThread:
    id = "thread-fake"

    def __init__(self, recorder: dict[str, Any]) -> None:
        self._recorder = recorder

    async def turn(self, prompt: str, **kwargs: Any) -> _FakeTurnHandle:
        self._recorder["turn_kwargs"] = kwargs
        self._recorder["turn_prompt"] = prompt
        return _FakeTurnHandle(self._recorder)


class _FakeAsyncCodex:
    def __init__(self, recorder: dict[str, Any]) -> None:
        self._recorder = recorder

    async def __aenter__(self) -> "_FakeAsyncCodex":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def thread_start(self, **kwargs: Any) -> _FakeThread:
        self._recorder["thread_kwargs"] = kwargs
        return _FakeThread(self._recorder)


def _fake_sdk(recorder: dict[str, Any]) -> SimpleNamespace:
    """Minimal stand-in for ``openai_codex`` — no child process, no network."""
    return SimpleNamespace(
        ApprovalMode=SimpleNamespace(deny_all="deny_all"),
        Sandbox=SimpleNamespace(full_access="full", read_only="ro", workspace_write="rw"),
        CodexConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        AsyncCodex=lambda _config: _FakeAsyncCodex(recorder),
    )


def test_run_codex_turn_forwards_the_output_schema_to_the_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``output_schema`` is what makes the enum binding enforceable."""
    recorder: dict[str, Any] = {"reply": '{"intents": []}'}
    monkeypatch.setattr(codex_session, "load_codex_sdk", lambda: _fake_sdk(recorder))
    schema = build_intent_envelope_schema(_ORCHESTRATION_INTENTS)

    result = asyncio.run(
        codex_session.run_codex_turn(
            prompt="p",
            developer_instructions="d",
            cwd=tmp_path,
            model="gpt-5.6-sol",
            timeout_sec=5.0,
            sandbox_mode="bypass",
            output_schema=schema,
            env={
                "OPENAI_API_KEY": "key",
                "OPENAI_BASE_URL": "https://gateway.invalid/v1",
                "HYPERLOOM_CODEX_SANDBOX_MODE": "bypass",
                "HYPERLOOM_RUNTIME_DIR": str(tmp_path / "runtime"),
            },
        )
    )

    assert recorder["turn_kwargs"]["output_schema"] == schema
    assert result.text == '{"intents": []}'


def test_run_codex_turn_defaults_to_no_output_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing callers (the specialist path) must be unchanged."""
    recorder: dict[str, Any] = {"reply": "done"}
    monkeypatch.setattr(codex_session, "load_codex_sdk", lambda: _fake_sdk(recorder))

    asyncio.run(
        codex_session.run_codex_turn(
            prompt="p",
            developer_instructions="d",
            cwd=tmp_path,
            model="gpt-5.6-sol",
            timeout_sec=5.0,
            sandbox_mode="bypass",
            env={
                "OPENAI_API_KEY": "key",
                "OPENAI_BASE_URL": "https://gateway.invalid/v1",
                "HYPERLOOM_CODEX_SANDBOX_MODE": "bypass",
                "HYPERLOOM_RUNTIME_DIR": str(tmp_path / "runtime"),
            },
        )
    )

    assert recorder["turn_kwargs"]["output_schema"] is None
