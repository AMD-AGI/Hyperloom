"""Tests for structured agent-output recovery helpers."""

from __future__ import annotations

import json

import pytest

from kernelforge.orchestrator.structured_output import (
    build_repair_prompt,
    extract_json_object,
)


def test_extract_json_object_accepts_fenced_payload():
    payload = extract_json_object(
        '```json\n{"status": "ready"}\n```',
        "analysis result",
    )

    assert payload == {"status": "ready"}


def test_extract_json_object_recovers_embedded_payload():
    payload = extract_json_object(
        'prefix text\n{"status": "partial", "count": 2}\nsuffix text',
        "analysis result",
    )

    assert payload == {"status": "partial", "count": 2}


def test_extract_json_object_rejects_response_without_object():
    with pytest.raises(
        ValueError,
        match="analysis result must contain one complete JSON object",
    ):
        extract_json_object("no structured payload", "analysis result")


def test_build_repair_prompt_preserves_recovery_context():
    prompt = json.loads(
        build_repair_prompt(
            label="analysis result",
            original_response='{"status": 1}',
            validation_error="status must be a string",
            output_schema={"status": "string"},
        )
    )

    assert prompt["original_response"] == '{"status": 1}'
    assert prompt["validation_error"] == "status must be a string"
    assert prompt["output_schema"] == {"status": "string"}
    assert "do not invent evidence" in prompt["task"]
