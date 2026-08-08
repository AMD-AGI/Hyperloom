# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recover token ``usage`` from out-of-process LLM-client output.

In-process backends hand us a ``BackendTurnResult.metadata`` dict directly,
but child-process paths (specialist subprocess, forge kernel candidates,
robustness RCA) leave their token counts only as text in a log/stdout/result
JSON. These parsers fold those counts into the same ledger.

All parsers are **tolerant**: a missing file, truncated JSON, or an absent
``usage`` block returns an empty / ``None`` result instead of raising.

Two agent CLIs are parsed, one per credential shape: the Claude CLI's
``--output-format stream-json`` and the Codex CLI's ``codex exec --json``. Each
has the same four recovery jobs (session usage, reply text, per-turn usage, tool
calls), so the parsers come in twins named after the log format they read.

Output shape: the token parsers (:func:`normalize_usage`,
:func:`parse_claude_stream_json_usage`, :func:`parse_codex_jsonl_usage`,
:func:`parse_forge_usage`) return the canonical four-key token dict, or ``None``
when nothing could be recovered:

    {"input_tokens", "output_tokens",
     "cache_creation_input_tokens", "cache_read_input_tokens"}

Backends with no prompt-cache concept (OpenAI / GEAK) leave the two ``cache_*``
values ``None`` so the collector can tell "no cache" from "zero cache hits".
The Codex parser adds one key beyond the canonical four,
``reasoning_output_tokens`` — see :data:`_CODEX_REASONING_TOKENS_KEY`.
The remaining parsers recover other shapes — reply text, per-turn usage lists,
tool-call lists, and the forge step timeline — see each parser's docstring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ._row_utils import coerce_optional_int

log = logging.getLogger(__name__)


# The four canonical counters, in stable order.
_TOKEN_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int | None] | None:
    """Project an arbitrary ``usage`` dict onto the canonical four keys.

    Returns ``None`` when ``usage`` is falsy or carries none of the four
    recognized counters. Unknown keys are dropped; absent counters become
    ``None``.

    Args:
        usage: Arbitrary usage dict, or ``None``.

    Returns:
        The canonical four-key token dict, or ``None`` when nothing usable.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    projected: dict[str, int | None] = {k: coerce_optional_int(usage.get(k)) for k in _TOKEN_KEYS}
    if all(v is None for v in projected.values()):
        return None
    return projected


def parse_claude_stream_json_usage(
    log_path: str | Path,
) -> dict[str, int | None] | None:
    """Extract the final ``usage`` from a Claude CLI ``stream-json`` log.

    ``claude --print --output-format stream-json --verbose`` writes one JSON
    object per line; the terminal ``{"type": "result", ..., "usage": {...}}``
    carries the cumulative session usage. Recovers token spend for the
    production-default specialist path (B1), otherwise invisible to the parent.

    Scans all lines, keeping the ``usage`` from the last object that carries one
    (``type=="result"`` preferred, but any line with a ``usage`` block is
    accepted). Malformed lines are skipped. Returns ``None`` if the file is
    missing or no ``usage`` is found.

    Args:
        log_path: Path to the Claude CLI ``stream-json`` log.

    Returns:
        The canonical token dict, or ``None`` when no usage was found.
    """
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
    """Recover the assistant's full reply text from a Claude CLI stream-json log.

    Sibling of :func:`parse_claude_stream_json_usage` (which reads token counts
    off the same log); this reads the conversation response so the
    production-default specialist path (B1) can land its completion in
    ``conversations.jsonl``. Only the response is recovered; the caller already
    holds the prompt.

    Reconstructs the reply from two sources, preferring the authoritative one:

    1. the terminal ``{"type": "result", ..., "result": "<text>"}`` row, whose
       ``result`` is the consolidated final answer; when non-empty it wins;
    2. otherwise concatenate the ``text`` blocks from every
       ``{"type": "assistant"}`` message in order — covering a truncated run
       that never emitted a ``result`` row. ``thinking`` and ``tool_use`` blocks
       are dropped.

    Tolerant by contract: a missing file, malformed lines, or no recoverable
    text returns ``None``.

    Args:
        log_path: Path to the Claude CLI ``stream-json`` log.

    Returns:
        The recovered response text, or ``None`` when none could be read.
    """
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


def parse_claude_stream_json_turn_usages(
    log_path: str | Path,
) -> list[dict[str, int | None]]:
    """Recover *per-assistant-turn* usage from a Claude CLI stream-json log.

    Unlike :func:`parse_claude_stream_json_usage` (which returns one cumulative
    row), this returns the per-message usage on each
    ``{"type":"assistant", "message": {..., "usage": {...}}}`` line, in order,
    so a multi-turn specialist subprocess can be traced as one row per turn.

    Each assistant message carries its OWN (non-cumulative) usage, so the rows
    sum to the session total. Lines without a usage block are skipped; the
    terminal cumulative ``result`` row is ignored to avoid double counting.

    Returns the normalized four-key dicts in stream order, or ``[]`` when the
    file is missing/truncated/carries no per-message usage.
    """
    path = Path(log_path)
    usages: list[dict[str, int | None]] = []
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
                usage = message.get("usage") if isinstance(message, dict) else None
                normalized = normalize_usage(usage if isinstance(usage, dict) else None)
                if normalized is not None:
                    usages.append(normalized)
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("parse_usage: failed reading stream-json log %s: %r", path, exc)
        return []
    return usages


def parse_claude_stream_json_tool_calls(
    log_path: str | Path,
) -> list[dict[str, Any]]:
    """Recover the intel/tool calls a specialist made from its stream-json log.

    Recovers every ``tool_use`` block the agent emitted, so the trace can
    surface what the specialist actually read (WebSearch / WebFetch / Grep /
    Read / ...). This is the data behind the per-call ``intel:<tool>`` spans.

    Each returned entry is ``{"tool": <name>, "query": <short input summary>}``,
    in call order. The input summary prefers common query-ish keys
    (``query`` / ``url`` / ``pattern`` / ``path`` / ``prompt``) and otherwise
    falls back to a compact clipped JSON of the input.

    Tolerant by contract: a missing file, malformed lines, or no tool calls
    returns ``[]``.
    """
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


def _summarize_tool_input(value: Any, *, limit: int = 240) -> str:
    """Compact, clipped one-line summary of a tool_use ``input`` block."""
    if isinstance(value, dict):
        for key in ("query", "url", "pattern", "path", "prompt", "command"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                return s if len(s) <= limit else (s[:limit] + "…")
        try:
            s = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            s = str(value)
    else:
        s = "" if value is None else str(value)
    return s if len(s) <= limit else (s[:limit] + "…")


# ---------------------------------------------------------------------------
# Codex CLI (``codex exec --json``)
# ---------------------------------------------------------------------------

# Codex spells its prompt-cache counter ``cached_input_tokens``. There is no
# cache-*write* counter, so ``cache_creation_input_tokens`` stays ``None`` and
# the collector still tells "no cache concept" from "zero cache hits".
_CODEX_TOKEN_ALIASES: dict[str, str] = {"cached_input_tokens": "cache_read_input_tokens"}

# Carried next to the canonical four rather than folded into ``output_tokens``:
# on a reasoning model these dominate the output budget while being invisible in
# the reply text, so summing them into the visible count would misreport both.
# ``LLMCallRecord.from_metadata`` reads named keys, so the extra one is inert
# there and survives only where the whole usage dict is kept (the specialist
# transcript). Mirrors ``common.codex_session.normalize_codex_usage``.
_CODEX_REASONING_TOKENS_KEY = "reasoning_output_tokens"

# ``item.type`` values that carry no tool call. Listed so an item type this
# parser has never seen can be reported without also warning about every
# message, reasoning summary and to-do update.
_CODEX_NON_TOOL_ITEM_TYPES: frozenset[str] = frozenset({"agent_message", "reasoning", "todo_list", "error"})

# Codex ``item.type`` -> the Claude tool name the intel ledger already uses, so
# ``specialist_intel.jsonl`` stays comparable across the two runtimes.
# ``mcp_tool_call`` keeps its Codex name: naming the server and tool would mean
# guessing field spellings no captured Codex stream has pinned yet.
_CODEX_TOOL_NAMES: dict[str, str] = {
    "command_execution": "Bash",
    "file_change": "Edit",
    "mcp_tool_call": "mcp_tool_call",
    "web_search": "WebSearch",
}

# The two events that carry a thread item. Both are read so a run killed
# mid-command still reports the call that was in flight; the item ``id``
# de-duplicates the pair.
_CODEX_ITEM_EVENTS: frozenset[str] = frozenset({"item.started", "item.completed"})


def _iter_codex_events(log_path: str | Path) -> "Any":
    """Yield each JSON object of a ``codex exec --json`` log, in stream order.

    Tolerant by contract (module docstring): a missing file yields nothing and
    an unparseable line is skipped, so a truncated log still reports the events
    it does hold.

    Args:
        log_path: Path to the Codex CLI JSONL log.

    Yields:
        Each decoded top-level JSON object that is a mapping.
    """
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


def _codex_usage_to_canonical(usage: Any) -> dict[str, int | None] | None:
    """Project one Codex ``usage`` block onto the canonical counters.

    Renames Codex's counters onto the canonical spellings and runs them through
    :func:`normalize_usage`, so there is exactly one token-dict shape in the
    ledger, then re-attaches :data:`_CODEX_REASONING_TOKENS_KEY`.

    Args:
        usage: A ``turn.completed`` usage mapping, or ``None``.

    Returns:
        The canonical token dict plus ``reasoning_output_tokens`` when reported,
        or ``None`` when nothing usable was present.
    """
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
    """Extract the session token usage from a ``codex exec --json`` log.

    The Codex twin of :func:`parse_claude_stream_json_usage`. ``codex exec
    --json`` writes one JSON event per line and reports token counts on
    ``{"type": "turn.completed", "usage": {...}}``. Unlike the Claude CLI's
    terminal ``result`` row, that usage covers only the turn that just ended, so
    the turns are summed here to give the caller the same session total its
    Claude counterpart returns.

    Args:
        log_path: Path to the Codex CLI JSONL log.

    Returns:
        The canonical token dict (plus ``reasoning_output_tokens``), or ``None``
        when the file is missing or reported no usage.
    """
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
    """Recover the agent's reply text from a ``codex exec --json`` log.

    The Codex twin of :func:`parse_claude_stream_json_response`, feeding the
    same ``conversations.jsonl`` row; only the response is recovered because the
    caller already holds the prompt.

    Codex has no consolidated final-answer row, so every
    ``{"type": "item.completed", "item": {"type": "agent_message"}}`` text is
    joined in order — the same reconstruction the Claude parser falls back to.
    For the common single-message turn that is just that message.

    Args:
        log_path: Path to the Codex CLI JSONL log.

    Returns:
        The recovered reply text, or ``None`` when none could be read.
    """
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
    """Recover per-turn usage from a ``codex exec --json`` log.

    The Codex twin of :func:`parse_claude_stream_json_turn_usages`. Each
    ``turn.completed`` event already carries its own turn's (non-cumulative)
    counts, so the rows sum to the total :func:`parse_codex_jsonl_usage`
    returns and can be traced as one ledger row per turn.

    Args:
        log_path: Path to the Codex CLI JSONL log.

    Returns:
        The normalized token dicts in stream order, or ``[]`` when the file is
        missing / truncated / carries no usage.
    """
    usages: list[dict[str, int | None]] = []
    for event in _iter_codex_events(log_path):
        if event.get("type") != "turn.completed":
            continue
        normalized = _codex_usage_to_canonical(event.get("usage"))
        if normalized is not None:
            usages.append(normalized)
    return usages


def _summarize_codex_item(kind: str, item: dict[str, Any]) -> str:
    """Summarize one Codex tool item as the intel ledger's ``query`` field.

    Args:
        kind: The item's ``type``.
        item: The item mapping.

    Returns:
        A compact, clipped one-line summary.
    """
    if kind == "file_change":
        changes = item.get("changes")
        paths = [
            change["path"]
            for change in (changes if isinstance(changes, (list, tuple)) else ())
            if isinstance(change, dict) and isinstance(change.get("path"), str)
        ]
        return _summarize_tool_input(", ".join(paths))
    # The shared summarizer already prefers the query-ish keys Codex items use
    # (``command`` for a shell call, ``query`` for a web search).
    return _summarize_tool_input(item)


def parse_codex_jsonl_tool_calls(
    log_path: str | Path,
) -> list[dict[str, Any]]:
    """Recover the tool calls a specialist made from its ``codex exec --json`` log.

    The Codex twin of :func:`parse_claude_stream_json_tool_calls`, producing the
    same ``{"tool", "query"}`` entries in call order that back the per-call
    ``intel:<tool>`` spans. Item types are mapped onto the Claude tool names via
    :data:`_CODEX_TOOL_NAMES` so the ledger reads the same on both runtimes.

    ``item.type`` is an open set. A type this parser does not know is still
    recorded, under its raw Codex name, and every such type is reported once in
    a warning — an unmodelled tool must not vanish from the trace, and must not
    crash the parse either.

    Args:
        log_path: Path to the Codex CLI JSONL log.

    Returns:
        One ``{"tool", "query"}`` entry per call, or ``[]`` when there were none.
    """
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


def parse_forge_usage(stdout: str) -> dict[str, int | None] | None:
    """Extract the run's LLM usage from a Kernel-Forge backend's stdout log.

    ``forge_submit`` aggregates the per-query ``ResultMessage`` token spend and
    prints one canonical marker line::

        FORGE_LLM_USAGE {"input_tokens": ..., "output_tokens": ...,
                         "cache_creation_input_tokens": ..., ...}

    Recovers the last such marker (the authoritative run total). Returns
    ``None`` when no marker is present.
    """
    if not stdout or "FORGE_LLM_USAGE" not in stdout:
        return None
    last_usage: dict[str, Any] | None = None
    for line in stdout.splitlines():
        marker = line.partition("FORGE_LLM_USAGE")
        if not marker[1]:
            continue
        blob = marker[2].strip()
        if not blob:
            continue
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and obj:
            last_usage = obj
    return normalize_usage(last_usage)


def parse_forge_steps(stdout: str) -> dict[str, Any] | None:
    """Extract the Kernel-Forge loop's key-step timeline from its stdout log.

    ``forge_submit`` prints one canonical marker carrying the per-iteration step
    timeline plus a run summary::

        FORGE_STEPS {"steps": [{"iteration": 1, "decision": "KEEP", ...}, ...],
                     "summary": {"iterations": ..., "termination_reason": ...}}

    Returns the parsed ``{"steps": [...], "summary": {...}}`` dict from the last
    marker, or ``None`` when no marker is present.
    """
    if not stdout or "FORGE_STEPS" not in stdout:
        return None
    last: dict[str, Any] | None = None
    for line in stdout.splitlines():
        marker = line.partition("FORGE_STEPS")
        if not marker[1]:
            continue
        blob = marker[2].strip()
        if not blob:
            continue
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
            last = obj
    return last


__all__ = [
    "normalize_usage",
    "parse_claude_stream_json_response",
    "parse_claude_stream_json_tool_calls",
    "parse_claude_stream_json_turn_usages",
    "parse_claude_stream_json_usage",
    "parse_codex_jsonl_response",
    "parse_codex_jsonl_tool_calls",
    "parse_codex_jsonl_turn_usages",
    "parse_codex_jsonl_usage",
    "parse_forge_steps",
    "parse_forge_usage",
]
