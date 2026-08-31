"""Shared helpers for bounded structured-agent output recovery."""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str, label: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a model response."""
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    for start, character in enumerate(raw):
        if character != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{label} must contain one complete JSON object")


def build_repair_prompt(
    *,
    label: str,
    original_response: str,
    validation_error: str,
    output_schema: dict[str, Any],
) -> str:
    """Build one deterministic follow-up request for schema repair."""
    payload = {
        "task": (
            f"Repair the invalid {label}. Return exactly one corrected JSON "
            "object and no other text. Preserve valid semantic content, but do "
            "not invent evidence or measurements."
        ),
        "validation_error": validation_error,
        "original_response": original_response,
        "output_schema": output_schema,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
