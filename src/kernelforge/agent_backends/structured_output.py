# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shape a session's final answer is asked to take, and reading it back in that shape."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from kernelforge.agent_backends.base import AgentRunResult

AnswerT = TypeVar("AnswerT", bound=BaseModel)

_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


class AgentAnswer(BaseModel):
    """The base of every answer shape a session is asked for.

    A provider's strict schema mode requires every property of every object to be required and no other key to be
    accepted, so a field takes no default and the model forbids extra keys. A rule a strict schema cannot carry --
    a non-empty string, a positive number -- belongs in a validator rather than in a field constraint.
    """

    model_config = ConfigDict(extra="forbid")


class StructuredOutputError(ValueError):
    """Report a session whose answer is not in the shape its spec asked for."""


def output_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON Schema a provider constrains the answer to."""
    return model.model_json_schema()


def schema_instruction(model: type[BaseModel]) -> str:
    """Ask in the prompt for the answer a provider cannot constrain itself."""
    schema = json.dumps(output_json_schema(model), indent=2, sort_keys=True)
    return (
        "## Answer format\n"
        "End your answer with exactly one JSON object that conforms to this JSON Schema, and write nothing after it.\n"
        f"```json\n{schema}\n```"
    )


def _text_candidates(text: str) -> Iterator[str]:
    """The spans of a text answer that may hold the JSON object, in the order they are tried."""
    yield text
    for block in _FENCED_BLOCK.findall(text):
        yield block.strip()
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        yield text[start : end + 1]


def structured_answer(result: AgentRunResult, model: type[AnswerT]) -> AnswerT:
    """The session's answer as ``model``.

    A provider that constrained the answer itself hands it over as a JSON value; any other answer is text, and the
    object is read from the whole text, then from each fenced block, then from the first ``{`` to the last ``}``.
    """
    if result.structured_output is not None:
        try:
            return model.model_validate(result.structured_output)
        except ValidationError as error:
            raise StructuredOutputError(f"the answer is not a {model.__name__}: {error}") from error
    text = (result.text or "").strip()
    if not text:
        raise StructuredOutputError(f"the session returned no {model.__name__}")
    failure: ValidationError | None = None
    for candidate in _text_candidates(text):
        try:
            return model.model_validate_json(candidate)
        except ValidationError as error:
            failure = error
    raise StructuredOutputError(f"the answer is not a {model.__name__}: {failure}")


__all__ = [
    "AgentAnswer",
    "StructuredOutputError",
    "output_json_schema",
    "schema_instruction",
    "structured_answer",
]
