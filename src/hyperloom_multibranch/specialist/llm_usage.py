# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one agent CLI launch used, read from its stream-json log: the tokens of every model call, by model.

Claude Code logs each assistant message with its model and usage; a message streamed as several events repeats one
usage under one id, so each id counts once. Codex logs each turn's usage when the turn completes, without the model.
A launch killed mid-session has every call before the kill in its log, so what it used is known without the result
event the CLI writes only when its model finishes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of

#: The model of calls the log does not name (codex turns): the launch's own model.
UNNAMED = ""


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
    messages: dict[str, tuple[str, TokenCounts]] = {}
    turns = TokenCounts()
    session_id = ""
    for row in _rows(process_log):
        session_id = session_id or str(row.get("session_id") or row.get("thread_id") or "")
        message, usage = row.get("message"), row.get("usage")
        if (
            row.get("type") == "assistant"
            and isinstance(message, Mapping)
            and isinstance(message.get("usage"), Mapping)
        ):
            key = str(message.get("id") or f"#{len(messages)}")
            messages[key] = (str(message.get("model") or UNNAMED), TokenCounts.from_anthropic_usage(message["usage"]))
        elif row.get("type") == "turn.completed" and isinstance(usage, Mapping):
            turns += _codex_tokens(usage)
    tokens: dict[str, TokenCounts] = {}
    for model, counts in messages.values():
        tokens[model] = tokens.get(model, TokenCounts()) + counts
    if turns != TokenCounts():
        tokens[UNNAMED] = tokens.get(UNNAMED, TokenCounts()) + turns
    return LaunchUsage(tokens, session_id)


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


def _count(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


__all__ = ["UNNAMED", "LaunchUsage", "launch_cost", "launch_usage"]
