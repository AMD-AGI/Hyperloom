# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read a DeepSeek Harness (``dsh --profile headless``) run's trace from its session logs."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.common.deepseek_session import step_usage, turn_end_failure
from hyperloom.common.env_safety import redact_secret_values

from ._row_utils import coerce_optional_int
from .agent_trace import AgentTrace
from .parse_usage import _iter_json_events, _summarize_tool_input, normalize_usage

log = logging.getLogger(__name__)

# Every session the run opened, the root and each subagent's, under the run's ``DSH_HOME``. The launch's patch turns
# off the log's compression, which is the harness's default and is not a format external readers may rely on.
_SESSION_LOG_GLOB = "sessions/*/*/session.v3.jsonl"

_REASONING_KEY = "reasoning_output_tokens"
_ASSISTANT_MESSAGE = "assistant/message"

# dsh tool name -> the Claude tool name the intel ledger and the web-tool policy already use.
_TOOL_NAMES: dict[str, str] = {
    "bash": "Bash",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "read": "Read",
    "subagent": "Task",
    "todo_write": "TodoWrite",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
    "write": "Write",
}


def _session_records(path: Path) -> list[dict[str, Any]]:
    """The JSON records of one session log, up to a cut-off final line."""
    return list(_iter_json_events(path, "dsh session"))


def _is_root(records: list[dict[str, Any]]) -> bool:
    """Whether a session is the run's own rather than a subagent's (``delegationDepth`` 0)."""
    header = records[0] if records and records[0].get("type") == "session" else {}
    return not coerce_optional_int(header.get("delegationDepth"))


def _data(record: dict[str, Any], kind: str) -> dict[str, Any]:
    data = record.get("data") if record.get("type") == kind else None
    return data if isinstance(data, dict) else {}


def _canonical(usage: dict[str, int]) -> dict[str, int | None] | None:
    """Project dsh token counts onto the canonical counters plus reasoning output."""
    normalized = normalize_usage(usage)
    if normalized is not None and _REASONING_KEY in usage:
        normalized[_REASONING_KEY] = usage[_REASONING_KEY]
    return normalized


def _reply_texts(record: dict[str, Any]) -> list[str]:
    message = _data(record, _ASSISTANT_MESSAGE).get("message")
    content = message.get("content") if isinstance(message, dict) else None
    blocks = content if isinstance(content, list) else []
    texts = (block.get("text") for block in blocks if isinstance(block, dict) and block.get("type") == "text")
    return [text for text in texts if isinstance(text, str) and text.strip()]


def _tool_call(record: dict[str, Any]) -> dict[str, Any] | None:
    data = _data(record, "tool/call")
    name = str(data.get("name") or "").strip()
    if not name:
        return None
    arguments: Any = data.get("arguments")
    if isinstance(arguments, str):
        # Arguments that are not JSON are summarized as the text they are.
        with contextlib.suppress(json.JSONDecodeError):
            arguments = json.loads(arguments)
    return {"tool": _TOOL_NAMES.get(name, name), "query": _summarize_tool_input(arguments)}


def parse_dsh_session_trace(dsh_home: str | Path) -> AgentTrace:
    """The trace of one ``dsh`` run: usage and tool calls over every session, reply and failure of the root's.

    A subagent's spend and calls are the task's; its reply is an input to the
    root's, not the answer.
    """
    steps: list[dict[str, int]] = []
    tool_calls: list[dict[str, Any]] = []
    replies: list[str] = []
    error = ""
    paths = sorted(Path(dsh_home).glob(_SESSION_LOG_GLOB))
    for path in paths:
        records = _session_records(path)
        steps += [usage for usage in map(step_usage, records) if usage]
        tool_calls += [call for call in map(_tool_call, records) if call is not None]
        if _is_root(records):
            replies += [text for record in records for text in _reply_texts(record)]
            error = turn_end_failure(records) or error
    totals: dict[str, int] = {}
    for step in steps:
        for key, count in step.items():
            totals[key] = totals.get(key, 0) + count
    if not paths:
        log.warning("parse_dsh_session: dsh left no session log under %s; its token spend is unknown", dsh_home)
        return AgentTrace(error=f"dsh: no session log was written under {dsh_home}")
    return AgentTrace(
        usage=_canonical(totals),
        response=replies[-1] if replies else None,
        tool_calls=tool_calls,
        turn_usages=[usage for usage in map(_canonical, steps) if usage is not None],
        error=f"dsh: {redact_secret_values(error)}" if error else "",
    )


__all__ = ["parse_dsh_session_trace"]
