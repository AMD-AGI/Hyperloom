# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recover token ``usage`` from out-of-process LLM-client output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_values

from ._row_utils import coerce_optional_int

log = logging.getLogger(__name__)


# The four canonical counters, in stable order.
_TOKEN_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


# Provider spellings for a reply's hidden reasoning output.
_REASONING_TOKEN_KEYS: tuple[str, ...] = ("reasoning_output_tokens", "reasoning_tokens")
_REASONING_DETAIL_KEYS: tuple[str, ...] = ("completion_tokens_details", "output_tokens_details")


def _field(usage: Any, key: str) -> Any:
    """Read ``key`` off a usage payload that may be a mapping or an SDK object."""
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def reasoning_output_tokens(usage: Any) -> int | None:
    """Recover the reasoning-output token count from any provider usage shape."""
    if usage is None:
        return None
    for key in _REASONING_TOKEN_KEYS:
        value = coerce_optional_int(_field(usage, key))
        if value is not None:
            return value
    for detail_key in _REASONING_DETAIL_KEYS:
        details = _field(usage, detail_key)
        if details is None:
            continue
        for key in _REASONING_TOKEN_KEYS:
            value = coerce_optional_int(_field(details, key))
            if value is not None:
                return value
    return None


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int | None] | None:
    """Project an arbitrary ``usage`` dict onto the canonical four keys."""
    if not isinstance(usage, dict) or not usage:
        return None
    projected: dict[str, int | None] = {k: coerce_optional_int(usage.get(k)) for k in _TOKEN_KEYS}
    if all(v is None for v in projected.values()):
        return None
    return projected


def parse_claude_stream_json_usage(
    log_path: str | Path,
) -> dict[str, int | None] | None:
    """Extract the final ``usage`` from a Claude CLI ``stream-json`` log."""
    path = Path(log_path)
    last_usage: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict):
                    continue
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage:
                    # A result-typed row is authoritative over earlier usage.
                    if obj.get("type") == "result" or last_usage is None:
                        last_usage = usage
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("parse_usage: failed reading stream-json log %s: %r", path, exc)
        return None
    return normalize_usage(last_usage)


def parse_claude_stream_json_response(
    log_path: str | Path,
) -> str | None:
    """Recover the assistant's full reply text from a Claude CLI stream-json log."""
    path = Path(log_path)
    result_text: str | None = None
    assistant_chunks: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict):
                    continue
                obj_type = obj.get("type")
                if obj_type == "result":
                    res = obj.get("result")
                    if isinstance(res, str) and res.strip():
                        result_text = res
                elif obj_type == "assistant":
                    message = obj.get("message")
                    if not isinstance(message, dict):
                        continue
                    for block in message.get("content") or []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text:
                                assistant_chunks.append(text)
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("parse_usage: failed reading stream-json log %s: %r", path, exc)
        return None
    if result_text is not None:
        return result_text
    if assistant_chunks:
        return "\n".join(assistant_chunks)
    return None


def _claude_result_output_tokens(result: dict[str, Any]) -> int | None:
    """Read the authoritative output-token count off a ``result`` row."""
    per_model = result.get("modelUsage")
    if isinstance(per_model, dict):
        total: int | None = None
        for entry in per_model.values():
            if not isinstance(entry, dict):
                continue
            count = coerce_optional_int(entry.get("outputTokens"))
            if count is not None:
                total = count if total is None else total + count
        if total is not None:
            return total
    usage = result.get("usage")
    if isinstance(usage, dict):
        return coerce_optional_int(usage.get("output_tokens"))
    return None


def _reattach_turn_output(
    usages: list[dict[str, int | None]],
    session_output: int | None,
) -> list[dict[str, int | None]]:
    """Swap placeholder per-turn ``output_tokens`` for the session's real count."""
    if not usages:
        return usages
    observed = sum(usage["output_tokens"] or 0 for usage in usages)
    if session_output is not None and observed == session_output:
        return usages
    for usage in usages[:-1]:
        usage["output_tokens"] = None
    usages[-1]["output_tokens"] = session_output
    return usages


def parse_claude_stream_json_turn_usages(
    log_path: str | Path,
) -> list[dict[str, int | None]]:
    """Recover *per-API-response* usage from a Claude CLI stream-json log."""
    path = Path(log_path)
    usages: list[dict[str, int | None]] = []
    seen_ids: set[str] = set()
    saw_message_id = False
    session_output: int | None = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "result":
                    recovered = _claude_result_output_tokens(obj)
                    if recovered is not None:
                        session_output = recovered
                    continue
                if obj.get("type") != "assistant":
                    continue
                message = obj.get("message")
                usage = message.get("usage") if isinstance(message, dict) else None
                normalized = normalize_usage(usage if isinstance(usage, dict) else None)
                if normalized is None:
                    continue
                message_id = message.get("id") if isinstance(message, dict) else None
                if isinstance(message_id, str) and message_id:
                    saw_message_id = True
                    if message_id in seen_ids:
                        continue
                    seen_ids.add(message_id)
                usages.append(normalized)
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("parse_usage: failed reading stream-json log %s: %r", path, exc)
        return []
    if len(usages) > 1 and not saw_message_id:
        log.warning(
            "parse_usage: stream-json log %s names no message ids; per-turn rows "
            "cannot be de-duplicated, deferring to the cumulative result row",
            path,
        )
        return []
    return _reattach_turn_output(usages, session_output)


def parse_claude_stream_json_tool_calls(
    log_path: str | Path,
) -> list[dict[str, Any]]:
    """Recover the intel/tool calls a specialist made from its stream-json log."""
    path = Path(log_path)
    calls: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "assistant":
                    continue
                message = obj.get("message")
                if not isinstance(message, dict):
                    continue
                for block in message.get("content") or []:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = str(block.get("name") or "").strip()
                    if not name:
                        continue
                    calls.append(
                        {
                            "tool": name,
                            "query": _summarize_tool_input(block.get("input")),
                        }
                    )
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("parse_usage: failed reading stream-json log %s: %r", path, exc)
        return []
    return calls


# Bound on the string fed to ``redact_secret_values`` before the 240-char clip.
_REDACT_SCAN_LIMIT = 4096


def _summarize_tool_input(value: Any, *, limit: int = 240) -> str:
    """Compact, clipped one-line summary of a tool_use ``input`` block."""
    if isinstance(value, dict):
        for key in ("query", "url", "pattern", "path", "prompt", "command"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                break
        else:
            try:
                s = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                s = str(value)
    else:
        s = "" if value is None else str(value)
    if len(s) > _REDACT_SCAN_LIMIT:
        s = s[:_REDACT_SCAN_LIMIT]
    redacted = redact_secret_values(s)
    return redacted if len(redacted) <= limit else (redacted[:limit] + "…")


# Codex CLI (``codex exec --json``)

# Codex spells its prompt-cache counter ``cached_input_tokens``.
_CODEX_TOKEN_ALIASES: dict[str, str] = {"cached_input_tokens": "cache_read_input_tokens"}

# Carried next to the canonical four rather than folded into ``output_tokens``: on a reasoning model these dominate
# the output budget while being invisible in the reply text, so summing them into the visible count would misreport
# both.
_CODEX_REASONING_TOKENS_KEY = "reasoning_output_tokens"

# ``item.type`` values that carry no tool call.
_CODEX_NON_TOOL_ITEM_TYPES: frozenset[str] = frozenset({"agent_message", "reasoning", "todo_list", "error"})

# Codex ``item.type`` -> the Claude tool name the intel ledger already uses, so ``specialist_intel.jsonl`` stays
# comparable across the two runtimes.
_CODEX_TOOL_NAMES: dict[str, str] = {
    "command_execution": "Bash",
    "file_change": "Edit",
    "mcp_tool_call": "mcp_tool_call",
    "web_search": "WebSearch",
}

# The two events that carry a thread item.
_CODEX_ITEM_EVENTS: frozenset[str] = frozenset({"item.started", "item.completed"})

# Error items are non-fatal warnings in the Codex schema, top-level ``error`` events are fatal stream errors, and
# ``turn.failed`` is the terminal outcome.
_CODEX_ERROR_AUTHORITY: dict[str, int] = {
    "item_error": 1,
    "error": 2,
    "turn.failed": 3,
}

# Adapters around the canonical exec schema sometimes preserve the app-server wrapper (``error.message``) or its
# additional-details spelling.
_CODEX_ERROR_MESSAGE_KEYS: tuple[str, ...] = (
    "message",
    "error",
    "reason",
    "detail",
    "details",
    "additional_details",
    "additionalDetails",
    "description",
    "text",
)
_CODEX_ERROR_MESSAGE_LIMIT = 2000


def _iter_codex_events(log_path: str | Path) -> "Any":
    """Yield each JSON object of a ``codex exec --json`` log, in stream order."""
    path = Path(log_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except FileNotFoundError:
        return
    except OSError as exc:
        log.warning("parse_usage: failed reading codex jsonl log %s: %r", path, exc)


def _codex_error_message(value: Any, *, depth: int = 0) -> str | None:
    """Extract and sanitize a scalar message from a known error payload."""
    if isinstance(value, str):
        message = value.strip()
        if not message:
            return None
        redacted = redact_secret_values(message)
        if len(redacted) > _CODEX_ERROR_MESSAGE_LIMIT:
            return redacted[: _CODEX_ERROR_MESSAGE_LIMIT - 1] + "…"
        return redacted
    if not isinstance(value, dict) or depth >= 6:
        return None
    for key in _CODEX_ERROR_MESSAGE_KEYS:
        if key not in value:
            continue
        message = _codex_error_message(value[key], depth=depth + 1)
        if message is not None:
            return message
    return None


def parse_codex_jsonl_error(log_path: str | Path) -> str | None:
    """Recover the most authoritative actionable Codex failure message."""
    best_authority = 0
    best_message: str | None = None
    for event in _iter_codex_events(log_path):
        event_type = event.get("type")
        authority = 0
        payload: Any = None
        if event_type == "turn.failed":
            authority = _CODEX_ERROR_AUTHORITY["turn.failed"]
            payload = event.get("error")
        elif event_type == "error":
            authority = _CODEX_ERROR_AUTHORITY["error"]
            payload = event
        elif event_type in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "error":
                authority = _CODEX_ERROR_AUTHORITY["item_error"]
                payload = item
        if authority < best_authority:
            continue
        message = _codex_error_message(payload)
        if message is not None:
            best_authority = authority
            best_message = message
    return best_message


def _codex_usage_to_canonical(usage: Any) -> dict[str, int | None] | None:
    """Project one Codex ``usage`` block onto the canonical counters."""
    if not isinstance(usage, dict) or not usage:
        return None
    renamed = {_CODEX_TOKEN_ALIASES.get(key, key): value for key, value in usage.items()}
    normalized = normalize_usage(renamed)
    if normalized is None:
        return None
    reasoning = coerce_optional_int(usage.get(_CODEX_REASONING_TOKENS_KEY))
    if reasoning is not None:
        normalized[_CODEX_REASONING_TOKENS_KEY] = reasoning
    return normalized


def parse_codex_jsonl_usage(
    log_path: str | Path,
) -> dict[str, int | None] | None:
    """Extract the session token usage from a ``codex exec --json`` log."""
    totals: dict[str, int] = {}
    for event in _iter_codex_events(log_path):
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            count = coerce_optional_int(value)
            if count is not None:
                totals[key] = totals.get(key, 0) + count
    return _codex_usage_to_canonical(totals)


def parse_codex_jsonl_response(
    log_path: str | Path,
) -> str | None:
    """Recover the agent's reply text from a ``codex exec --json`` log."""
    chunks: list[str] = []
    for event in _iter_codex_events(log_path):
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


def parse_codex_jsonl_turn_usages(
    log_path: str | Path,
) -> list[dict[str, int | None]]:
    """Recover per-turn usage from a ``codex exec --json`` log."""
    usages: list[dict[str, int | None]] = []
    for event in _iter_codex_events(log_path):
        if event.get("type") != "turn.completed":
            continue
        normalized = _codex_usage_to_canonical(event.get("usage"))
        if normalized is not None:
            usages.append(normalized)
    return usages


def _summarize_codex_item(kind: str, item: dict[str, Any]) -> str:
    """Summarize one Codex tool item as the intel ledger's ``query`` field."""
    if kind == "file_change":
        changes = item.get("changes")
        paths = [
            change["path"]
            for change in (changes if isinstance(changes, (list, tuple)) else ())
            if isinstance(change, dict) and isinstance(change.get("path"), str)
        ]
        return _summarize_tool_input(", ".join(paths))
    # The shared summarizer already prefers the query-ish keys Codex items use (``command`` for a shell call,
    # ``query`` for a web search).
    return _summarize_tool_input(item)


def parse_codex_jsonl_tool_calls(
    log_path: str | Path,
) -> list[dict[str, Any]]:
    """Recover the tool calls a specialist made from its ``codex exec --json`` log."""
    calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    unknown_types: set[str] = set()
    for event in _iter_codex_events(log_path):
        if event.get("type") not in _CODEX_ITEM_EVENTS:
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").strip()
        if not kind or kind in _CODEX_NON_TOOL_ITEM_TYPES:
            continue
        # ``item.started`` and ``item.completed`` describe one call; count it once.
        item_id = str(item.get("id") or "")
        if item_id:
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
        tool = _CODEX_TOOL_NAMES.get(kind)
        if tool is None:
            unknown_types.add(kind)
            tool = kind
        calls.append({"tool": tool, "query": _summarize_codex_item(kind, item)})
    if unknown_types:
        log.warning(
            "parse_usage: codex log %s carried unmodelled item types %s; recorded under their raw names",
            log_path,
            sorted(unknown_types),
        )
    return calls


__all__ = [
    "normalize_usage",
    "parse_claude_stream_json_response",
    "parse_claude_stream_json_tool_calls",
    "parse_claude_stream_json_turn_usages",
    "parse_claude_stream_json_usage",
    "parse_codex_jsonl_error",
    "parse_codex_jsonl_response",
    "parse_codex_jsonl_tool_calls",
    "parse_codex_jsonl_turn_usages",
    "parse_codex_jsonl_usage",
    "reasoning_output_tokens",
]
