# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one agent CLI launch used, read from its stream-json log: the tokens of every model call, by model.

Claude Code ends each query with a result row, which counts it: the messages streamed before it carry placeholder
output counts, some gateways zero their usage, and a message without an id cannot be told apart from its streamed
repeats. The row's ``modelUsage`` holds the running totals of the CLI process per model, Task sub-agents included, so
the last one counts every finished query of a log, which holds one launch; a row without it counts its own ``usage``
under the model of the query's messages. The calls of a query the launch was killed in have no result row and are
read from their messages, each id once. Codex logs each turn's usage when the turn completes, without the model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of

#: The model of calls the log does not name (codex turns): the launch's own model.
UNNAMED = ""
#: How Claude Code marks the messages it makes up itself (API errors, interrupts), which are no model call.
_SYNTHETIC = "<"


@dataclass(frozen=True)
class LaunchUsage:
    """The tokens of a launch's model calls by the model the log names (``UNNAMED`` when it names none), and the CLI
    session or thread id it ran under."""

    tokens: Mapping[str, TokenCounts]
    session_id: str

    @property
    def total(self) -> TokenCounts:
        return sum(self.tokens.values(), TokenCounts())


def launch_usage(process_log: Path) -> LaunchUsage:
    """Every model call a Claude or codex stream-json log records; none when the log is missing."""
    running: dict[str, TokenCounts] = {}
    tokens: dict[str, TokenCounts] = {}
    unfinished: dict[str, tuple[str, TokenCounts]] = {}
    session_id, model = "", UNNAMED
    for row in _rows(process_log):
        session_id = session_id or str(row.get("session_id") or row.get("thread_id") or "")
        usage, per_model = row.get("usage"), row.get("modelUsage")
        message = _model_call(row)
        if message is not None:
            model = str(message.get("model") or UNNAMED)
            key = str(message.get("id") or f"#{len(unfinished)}")
            unfinished[key] = (model, TokenCounts.from_anthropic_usage(message["usage"]))
        elif row.get("type") == "result" and isinstance(per_model, Mapping) and per_model:
            running = {str(name): _model_usage_tokens(counts) for name, counts in per_model.items()}
            unfinished.clear()
        elif row.get("type") == "result" and isinstance(usage, Mapping):
            _add(tokens, model, TokenCounts.from_anthropic_usage(usage))
            unfinished.clear()
        elif row.get("type") == "turn.completed" and isinstance(usage, Mapping):
            _add(tokens, UNNAMED, _codex_tokens(usage))
    for named, counts in [*running.items(), *unfinished.values()]:
        _add(tokens, named, counts)
    return LaunchUsage(tokens, session_id)


def _model_call(row: Mapping[str, object]) -> Mapping[str, Any] | None:
    """The message of the Claude model call ``row`` logs; ``None`` for any other row."""
    message = row.get("message")
    if (
        row.get("type") != "assistant"
        or not isinstance(message, Mapping)
        or not isinstance(message.get("usage"), Mapping)
    ):
        return None
    return None if str(message.get("model") or "").startswith(_SYNTHETIC) else message


def _add(tokens: dict[str, TokenCounts], model: str, counts: TokenCounts) -> None:
    if counts != TokenCounts():
        tokens[model] = tokens.get(model, TokenCounts()) + counts


def launch_cost(tokens: Mapping[str, TokenCounts], model: str) -> float | None:
    """What a launch's calls of :attr:`LaunchUsage.tokens` cost at the gateway's prices, a call the log names no model
    of priced as ``model``'s; ``None`` when one of their models has no price."""

    prices = [(price_of(name or model), counts) for name, counts in tokens.items()]
    if any(price is None for price, _ in prices):
        return None
    return sum(price.cost_usd(counts) for price, counts in prices if price is not None)


def _rows(process_log: Path) -> Iterator[Mapping[str, object]]:
    try:
        handle = process_log.open(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return
    with handle:
        for line in handle:
            if '"usage"' not in line and '"session_id"' not in line and '"thread_id"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _codex_tokens(usage: Mapping[str, object]) -> TokenCounts:
    cached = _count(usage, "cached_input_tokens")
    return TokenCounts(max(0, _count(usage, "input_tokens") - cached), _count(usage, "output_tokens"), cached)


def _model_usage_tokens(counts: object) -> TokenCounts:
    counts = counts if isinstance(counts, Mapping) else {}
    return TokenCounts(
        _count(counts, "inputTokens"),
        _count(counts, "outputTokens"),
        _count(counts, "cacheReadInputTokens"),
        _count(counts, "cacheCreationInputTokens"),
    )


def _count(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


__all__ = ["UNNAMED", "LaunchUsage", "launch_cost", "launch_usage"]
