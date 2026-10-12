# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tool call / result rows for the trajectory ledger.

A row carries a redacted, clipped summary of the tool input and the size of its result, never either payload whole:
the ledger is uploaded to Langfuse, and ``conversations.jsonl`` already owns full-fidelity transcripts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_values

from .parse_usage import _summarize_tool_input
from .trajectory_trace import (
    EVENT_TOOL,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    record_event,
)

log = logging.getLogger(__name__)

_ERROR_PREVIEW_CHARS = 240

TIMING_NONE = "none"


def content_text(content: Any) -> str:
    """The text a tool result carries (``str`` or a list of content blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            parts.append(text if isinstance(text, str) else json.dumps(block, default=str, sort_keys=True))
        return "\n".join(parts)
    return json.dumps(content, default=str, sort_keys=True)


def _input_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, default=str, sort_keys=True)) if value is not None else 0


def tool_status(*, result_seen: bool, is_error: Any) -> str:
    """``completed`` / ``failed`` from the result's error flag; ``cancelled`` when no result ever arrived."""
    if not result_seen:
        return STATUS_CANCELLED
    return STATUS_FAILED if is_error is True else STATUS_COMPLETED


def tool_attributes(
    *,
    name: str,
    tool_use_id: str | None,
    tool_input: Any,
    result_content: Any = None,
    result_seen: bool = False,
    is_error: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Attributes of one ``tool`` row; the error preview is redacted and clipped like the input summary."""
    attributes: dict[str, Any] = {
        "name": name,
        "tool_use_id": tool_use_id,
        "input_summary": _summarize_tool_input(tool_input),
        "input_chars": _input_chars(tool_input),
        **extra,
    }
    if result_seen:
        text = content_text(result_content)
        attributes["result_chars"] = len(text)
        attributes["is_error"] = is_error is True
        if is_error is True:
            attributes["error_preview"] = redact_secret_values(text[: _ERROR_PREVIEW_CHARS * 4])[:_ERROR_PREVIEW_CHARS]
    return attributes


def stream_json_tool_events(log_path: str | Path) -> list[dict[str, Any]]:
    """Pair ``tool_use`` blocks with their ``tool_result`` blocks in a Claude CLI stream-json log, in call order."""
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    try:
        with Path(log_path).open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict) or obj.get("type") not in ("assistant", "user"):
                    continue
                message = obj.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use" and block.get("id") and block.get("name"):
                        tool_use_id = str(block["id"])
                        if tool_use_id not in calls:
                            order.append(tool_use_id)
                        calls[tool_use_id] = {
                            "name": str(block["name"]),
                            "tool_use_id": tool_use_id,
                            "tool_input": block.get("input"),
                            "message_id": message.get("id"),
                            "parent_tool_use_id": obj.get("parent_tool_use_id"),
                        }
                    elif block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                        call = calls[str(block["tool_use_id"])]
                        call["result_seen"] = True
                        call["result_content"] = block.get("content")
                        call["is_error"] = block.get("is_error")
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("tool_events: failed reading stream-json log %s: %r", log_path, exc)
        return []
    return [calls[tool_use_id] for tool_use_id in order]


def record_stream_json_tools(log_path: str | Path) -> int:
    """Put every tool call of a finished stream-json log on the trajectory; returns how many rows were written.

    The log has no per-line timestamps, so these rows are untimed (``timing_source="none"``).
    """
    written = 0
    for call in stream_json_tool_events(log_path):
        result_seen = bool(call.get("result_seen"))
        span_id = record_event(
            EVENT_TOOL,
            status=tool_status(result_seen=result_seen, is_error=call.get("is_error")),
            attributes=tool_attributes(
                name=call["name"],
                tool_use_id=call["tool_use_id"],
                tool_input=call.get("tool_input"),
                result_content=call.get("result_content"),
                result_seen=result_seen,
                is_error=call.get("is_error"),
                message_id=call.get("message_id"),
                parent_tool_use_id=call.get("parent_tool_use_id"),
                timing_source=TIMING_NONE,
            ),
        )
        written += span_id is not None
    return written


__all__ = [
    "TIMING_NONE",
    "content_text",
    "record_stream_json_tools",
    "stream_json_tool_events",
    "tool_attributes",
    "tool_status",
]
