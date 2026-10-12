# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recover token ``usage`` from out-of-process LLM-client output."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.token_usage import uncached_input_tokens

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


def _claude_stream_model(obj: dict[str, Any]) -> str | None:
    """The model a stream-json row names: an assistant message's, else the ``system/init`` row's."""
    message = obj.get("message")
    model = message.get("model") if obj.get("type") == "assistant" and isinstance(message, dict) else None
    if model is None and obj.get("type") == "system":
        model = obj.get("model")
    # The CLI stamps locally generated assistant messages (errors, interrupts) with ``<synthetic>``.
    return model if isinstance(model, str) and model and not model.startswith("<") else None


def _iter_json_events(log_path: str | Path, kind: str) -> Iterator[dict[str, Any]]:
    """Yield each JSON object of a line-delimited ``kind`` log, in stream order."""
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
        log.warning("parse_usage: failed reading %s log %s: %r", kind, path, exc)


def parse_claude_stream_json_usage(
    log_path: str | Path,
) -> dict[str, Any] | None:
    """Extract the final ``usage`` (plus the serving ``model``, when named) from a Claude CLI ``stream-json`` log."""
    last_usage: dict[str, Any] | None = None
    model: str | None = None
    for obj in _iter_json_events(log_path, "stream-json"):
        model = _claude_stream_model(obj) or model
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage:
            # A result-typed row is authoritative over earlier usage.
            if obj.get("type") == "result" or last_usage is None:
                last_usage = usage
    normalized: dict[str, Any] | None = normalize_usage(last_usage)
    if normalized is not None and model is not None:
        normalized["model"] = model
    return normalized


def parse_claude_stream_json_response(
    log_path: str | Path,
) -> str | None:
    """Recover the assistant's full reply text from a Claude CLI stream-json log."""
    result_text: str | None = None
    assistant_chunks: list[str] = []
    for obj in _iter_json_events(log_path, "stream-json"):
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


_INPUT_SIDE_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _input_side_total(usage: dict[str, Any] | None) -> int:
    """Sum the prompt-side buckets of one normalized usage block."""
    if not usage:
        return 0
    return sum(int(usage.get(key) or 0) for key in _INPUT_SIDE_KEYS)


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
    usages: list[dict[str, int | None]] = []
    seen_ids: set[str] = set()
    saw_message_id = False
    session_output: int | None = None
    session_input = 0
    for obj in _iter_json_events(log_path, "stream-json"):
        if obj.get("type") == "result":
            recovered = _claude_result_output_tokens(obj)
            if recovered is not None:
                session_output = recovered
            result_usage = obj.get("usage")
            session_input = _input_side_total(normalize_usage(result_usage if isinstance(result_usage, dict) else None))
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
    if len(usages) > 1 and not saw_message_id:
        log.warning(
            "parse_usage: stream-json log %s names no message ids; per-turn rows "
            "cannot be de-duplicated, deferring to the cumulative result row",
            log_path,
        )
        return []
    if usages and session_input and not any(_input_side_total(usage) for usage in usages):
        # Some gateways (seen with GLM behind LiteLLM) stream every assistant message with zeroed usage and report the
        # real counts only on the result row; per-turn rows would then book the whole session at zero.
        log.warning(
            "parse_usage: stream-json log %s carries zeroed per-turn usage; deferring to the cumulative result row",
            log_path,
        )
        return []
    return _reattach_turn_output(usages, session_output)


def parse_claude_stream_json_tool_calls(
    log_path: str | Path,
) -> list[dict[str, Any]]:
    """Recover the intel/tool calls a specialist made from its stream-json log."""
    calls: list[dict[str, Any]] = []
    for obj in _iter_json_events(log_path, "stream-json"):
        if obj.get("type") != "assistant":
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
            calls.append({"tool": name, "query": _summarize_tool_input(block.get("input"))})
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
    for event in _iter_json_events(log_path, "codex jsonl"):
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
    normalized["input_tokens"] = uncached_input_tokens(
        normalized["input_tokens"],
        normalized["cache_read_input_tokens"],
    )
    reasoning = coerce_optional_int(usage.get(_CODEX_REASONING_TOKENS_KEY))
    if reasoning is not None:
        normalized[_CODEX_REASONING_TOKENS_KEY] = reasoning
    return normalized


def parse_codex_jsonl_usage(
    log_path: str | Path,
) -> dict[str, int | None] | None:
    """Extract the session token usage from a ``codex exec --json`` log."""
    totals: dict[str, int] = {}
    for event in _iter_json_events(log_path, "codex jsonl"):
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
    for event in _iter_json_events(log_path, "codex jsonl"):
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
    for event in _iter_json_events(log_path, "codex jsonl"):
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
    for event in _iter_json_events(log_path, "codex jsonl"):
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


# DeepSeek Harness CLI (``dsh --profile headless``)

# Every session the run opened, the root and each subagent's, under the run's ``DSH_HOME``.
_DSH_SESSION_LOG_GLOB = "sessions/*/*/session.v3.jsonl.zstd"

# ``inputTokens`` already excludes the cached prefix, so it maps straight onto the uncached counter.
_DSH_TOKEN_NAMES: dict[str, str] = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cacheReadTokens": "cache_read_input_tokens",
    "reasoningTokens": _CODEX_REASONING_TOKENS_KEY,
}

# dsh tool name -> the Claude tool name the intel ledger and the web-tool policy already use.
_DSH_TOOL_NAMES: dict[str, str] = {
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


def _decode_dsh_session_log(path: Path) -> list[dict[str, Any]]:
    """Decode one session log, keeping every record before a cut-off final frame.

    The harness appends one zstd frame per flush, so a run reaped mid-write
    leaves only its last frame incomplete.
    """
    try:
        import zstandard
    except ImportError:
        log.warning("parse_usage: zstandard is not installed; dsh session log %s is unreadable", path)
        return []
    try:
        blob = path.read_bytes()
    except OSError as exc:
        log.warning("parse_usage: cannot read dsh session log %s: %s", path, exc)
        return []
    decoded = bytearray()
    while blob:
        decompressor = zstandard.ZstdDecompressor().decompressobj()
        try:
            decoded += decompressor.decompress(blob)
        except zstandard.ZstdError:
            break
        if not decompressor.eof:
            break
        blob = decompressor.unused_data
    rows: list[dict[str, Any]] = []
    for line in decoded.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _dsh_sessions(dsh_home: str | Path) -> list[tuple[int, list[dict[str, Any]]]]:
    """Every session log under ``dsh_home`` as ``(delegation depth, records)``."""
    sessions: list[tuple[int, list[dict[str, Any]]]] = []
    for path in sorted(Path(dsh_home).glob(_DSH_SESSION_LOG_GLOB)):
        rows = _decode_dsh_session_log(path)
        header = rows[0] if rows and rows[0].get("type") == "session" else {}
        depth = coerce_optional_int(header.get("delegationDepth"))
        sessions.append((depth if depth is not None else 0, rows))
    return sessions


def _dsh_step_usages(dsh_home: str | Path) -> list[dict[str, int]]:
    """The usage of every model step, across every session of the run."""
    usages: list[dict[str, int]] = []
    for _depth, rows in _dsh_sessions(dsh_home):
        for row in rows:
            data = row.get("data") if row.get("type") == "assistant/message" else None
            usage = data.get("usage") if isinstance(data, dict) else None
            if not isinstance(usage, dict):
                continue
            step = {
                name: count
                for wire, name in _DSH_TOKEN_NAMES.items()
                if (count := coerce_optional_int(usage.get(wire))) is not None
            }
            if step:
                usages.append(step)
    return usages


def _dsh_usage_to_canonical(usage: dict[str, int]) -> dict[str, int | None] | None:
    """Project one summed dsh usage onto the canonical counters plus reasoning output."""
    normalized = normalize_usage(usage)
    if normalized is None:
        return None
    if _CODEX_REASONING_TOKENS_KEY in usage:
        normalized[_CODEX_REASONING_TOKENS_KEY] = usage[_CODEX_REASONING_TOKENS_KEY]
    return normalized


def parse_dsh_session_usage(dsh_home: str | Path) -> dict[str, int | None] | None:
    """Total token usage of a ``dsh`` run, its subagents' included."""
    totals: dict[str, int] = {}
    for step in _dsh_step_usages(dsh_home):
        for key, count in step.items():
            totals[key] = totals.get(key, 0) + count
    return _dsh_usage_to_canonical(totals)


def parse_dsh_session_turn_usages(dsh_home: str | Path) -> list[dict[str, int | None]]:
    """Per-step usage of a ``dsh`` run, in session then step order."""
    return [usage for step in _dsh_step_usages(dsh_home) if (usage := _dsh_usage_to_canonical(step)) is not None]


def parse_dsh_session_response(dsh_home: str | Path) -> str | None:
    """The root session's reply text; a subagent's reply is an input to it, not the answer."""
    chunks: list[str] = []
    for depth, rows in _dsh_sessions(dsh_home):
        if depth != 0:
            continue
        for row in rows:
            data = row.get("data") if row.get("type") == "assistant/message" else None
            message = data.get("message") if isinstance(data, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            for block in content if isinstance(content, list) else ():
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
    return chunks[-1] if chunks else None


def parse_dsh_session_tool_calls(dsh_home: str | Path) -> list[dict[str, Any]]:
    """The tool calls a ``dsh`` run made, its subagents' included."""
    calls: list[dict[str, Any]] = []
    for _depth, rows in _dsh_sessions(dsh_home):
        for row in rows:
            data = row.get("data") if row.get("type") == "tool/call" else None
            if not isinstance(data, dict):
                continue
            name = str(data.get("name") or "").strip()
            if not name:
                continue
            arguments: Any = data.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append({"tool": _DSH_TOOL_NAMES.get(name, name), "query": _summarize_tool_input(arguments)})
    return calls


def parse_dsh_session_error(dsh_home: str | Path) -> str | None:
    """The failure the root session's last ``turn/end`` records, if it failed."""
    for depth, rows in _dsh_sessions(dsh_home):
        if depth != 0:
            continue
        for row in reversed(rows):
            if row.get("type") != "turn/end":
                continue
            data = row.get("data")
            reason = data.get("reason") if isinstance(data, dict) else None
            error = reason.get("error") if isinstance(reason, dict) else None
            if not isinstance(error, dict):
                break
            code = str(error.get("code") or "").strip()
            message = _codex_error_message(error.get("message")) or ""
            return f"{code}: {message}" if code and message else (message or code or None)
    return None


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
    "parse_dsh_session_error",
    "parse_dsh_session_response",
    "parse_dsh_session_tool_calls",
    "parse_dsh_session_turn_usages",
    "parse_dsh_session_usage",
    "reasoning_output_tokens",
]
