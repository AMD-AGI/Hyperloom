# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The answer shape a session is held to: how a provider is asked for it, and how it is read back."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import Field, ValidationError

from kernelforge.agent_backends import AgentRunResult, AgentRunSpec, AgentRuntimeConfig
from kernelforge.agent_backends.claude import ClaudeBackend, _session_result, _StreamTranscript
from kernelforge.agent_backends.structured_output import (
    AgentAnswer,
    StructuredOutputError,
    output_json_schema,
    schema_instruction,
    structured_answer,
)
from kernelforge.knowledge.kb_store.run_summary import RunSummary
from kernelforge.workflows.forge_loop.roles.plan_critic import PlanCriticAnswer
from kernelforge.workflows.forge_loop.roles.plan_requests import PartitionAnswer, dispatch_answer_model


class _Verdict(AgentAnswer):
    status: Literal["ready", "partial"] = Field(description="How far the work got.")
    count: int = Field(description="How many items were done.")


def test_a_provider_constrained_answer_is_validated_as_given() -> None:
    result = AgentRunResult(text="", structured_output={"status": "ready", "count": 2})

    assert structured_answer(result, _Verdict) == _Verdict(status="ready", count=2)


@pytest.mark.parametrize(
    "text",
    [
        '{"status": "ready", "count": 2}',
        'Here it is.\n```json\n{"status": "ready", "count": 2}\n```\nDone.',
        'prefix text\n{"status": "ready", "count": 2}\nsuffix text',
    ],
    ids=["whole-answer", "fenced-block", "embedded-object"],
)
def test_an_answer_returned_as_text_is_read_from_where_it_sits(text: str) -> None:
    assert structured_answer(AgentRunResult(text=text), _Verdict) == _Verdict(status="ready", count=2)


@pytest.mark.parametrize(
    "result",
    [
        AgentRunResult(text=""),
        AgentRunResult(text="no object here"),
        AgentRunResult(text='{"status": "finished", "count": 2}'),
        AgentRunResult(text='{"status": "ready", "count": 2, "extra": true}'),
        AgentRunResult(structured_output={"status": "ready"}),
    ],
    ids=["empty", "prose", "outside-vocabulary", "extra-key", "missing-field"],
)
def test_an_answer_in_no_usable_shape_names_the_shape_it_missed(result: AgentRunResult) -> None:
    with pytest.raises(StructuredOutputError, match="_Verdict"):
        structured_answer(result, _Verdict)


def test_a_provider_that_cannot_constrain_its_answer_is_shown_the_schema() -> None:
    instruction = schema_instruction(_Verdict)

    assert json.dumps(output_json_schema(_Verdict), indent=2, sort_keys=True) in instruction
    assert "exactly one JSON object" in instruction


def test_an_answer_shape_accepts_no_key_it_does_not_declare() -> None:
    with pytest.raises(ValidationError):
        _Verdict.model_validate({"status": "ready", "count": 1, "note": "x"})


def _strict_mode_violations(node: Any, where: str = "$") -> list[str]:
    """What a provider's strict schema mode would refuse in the schema ``node`` belongs to."""
    if isinstance(node, list):
        return [found for item in node for found in _strict_mode_violations(item, where)]
    if not isinstance(node, dict):
        return []
    found: list[str] = []
    if "$ref" in node and set(node) - {"$ref"}:
        found.append(f"{where}: $ref beside {sorted(set(node) - {'$ref'})}")
    if node.get("type") == "object":
        properties = node.get("properties", {})
        if node.get("additionalProperties") is not False:
            found.append(f"{where}: accepts undeclared keys")
        if set(node.get("required", [])) != set(properties):
            found.append(f"{where}: optional {sorted(set(properties) - set(node.get('required', [])))}")
    for key, value in node.items():
        found.extend(_strict_mode_violations(value, f"{where}.{key}"))
    return found


@pytest.mark.parametrize(
    "model",
    [
        PlanCriticAnswer,
        PartitionAnswer,
        dispatch_answer_model(("compute", "memory"), ("case-a",)),
        RunSummary,
    ],
    ids=lambda model: model.__name__,
)
def test_every_answer_shape_is_one_a_strict_schema_mode_accepts(model) -> None:
    """Codex holds an answer to a strict schema, which refuses optional properties, extra keys and $ref siblings."""
    assert _strict_mode_violations(output_json_schema(model)) == []


def _claude_backend() -> ClaudeBackend:
    backend = object.__new__(ClaudeBackend)
    backend.runtime = AgentRuntimeConfig(provider="claude", model="claude-test")
    return backend


def test_claude_is_given_the_answer_schema_to_hold_the_session_to(tmp_path) -> None:
    spec = AgentRunSpec(system_prompt="s", user_prompt="u", cwd=str(tmp_path), output_schema=_Verdict)

    options = _claude_backend()._provider_options(spec)

    assert options["output_format"] == {"type": "json_schema", "schema": output_json_schema(_Verdict)}


def test_claude_is_given_no_schema_for_a_free_form_session(tmp_path) -> None:
    spec = AgentRunSpec(system_prompt="s", user_prompt="u", cwd=str(tmp_path))

    assert "output_format" not in _claude_backend()._provider_options(spec)


class _Guard:
    def verify(self) -> list[str]:
        return []

    def count_target_edits(self) -> int:
        return 0


def test_claude_hands_over_the_answer_its_result_message_carries() -> None:
    transcript = _StreamTranscript()
    transcript.absorb(
        SimpleNamespace(
            session_id="s-1",
            total_cost_usd=0.01,
            subtype="success",
            num_turns=2,
            structured_output={"status": "partial", "count": 1},
        )
    )

    result = _session_result(transcript, _Guard(), timed_out=False, stream_error=None, contention="")

    assert structured_answer(result, _Verdict) == _Verdict(status="partial", count=1)
