# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What each field and figure on the Tokens tab is called, and what it means: the page shows a definition on hover,
and the token timeline file carries them under ``terms``.
"""

from __future__ import annotations

# The window of the "Last 5 minutes" card, which its name states.
RECENT_SEC = 300.0
# A rollup's key for the events without one; in by_model, the calls with no model recorded.
NO_MODEL = "unknown"


def _term(group: str, label: str, definition: str) -> dict[str, str]:
    return {"group": group, "label": label, "definition": definition}


def _billed(group: str, label: str, what: str, arises: str, billed: str) -> dict[str, str]:
    """A cost category: its definition is three lines, what it is, how it arises and how it is billed."""
    return _term(group, label, f"What it is: {what}\nHow it arises: {arises}\nHow it is billed: {billed}")


_TYPES, _TOTALS, _CHARTS = "Token types", "Totals and rates", "Charts and tables"
_RECENT_MIN = f"{RECENT_SEC / 60:g}"
# The Tokens tab's names and definitions, which the page shows on hover and the timeline file carries. A definition
# is one line, but a billed category's (a token kind) is three. Keys are the field names (KINDS,
# totals) or, for what has no field, the figure's own.
TERMS: dict[str, dict[str, str]] = {
    "input": _billed(
        _TYPES,
        "Uncached input",
        "The part of an LLM call's input that the model processes in full, without the prompt cache.",
        "Every call sends input: the system prompt, tool definitions, the conversation so far and tool results. What "
        "the provider neither reads from nor writes to its prompt cache is uncached; without caching, all of it.",
        "Per token, at the model's input price. It is the base rate: writing to the cache usually costs more, reading "
        "from it far less.",
    ),
    "cache_write": _billed(
        _TYPES,
        "Input written to cache",
        "Input that the provider processes and also stores in its prompt cache, so that later calls can reuse it.",
        "A call that marks its input for caching (the Claude CLI does so on every turn) stores what the cache does not "
        "hold yet: what is new since the last call, or all of it again once the cache has expired. OpenAI reports no "
        "writes.",
        "Per token, at the model's cache-write price, which for Claude is above the input price; the built-in price "
        "sheet bills a model it lists none for at the input price.",
    ),
    "cache_read": _billed(
        _TYPES,
        "Input read from cache",
        "Input that the provider already holds in its prompt cache from an earlier call, reused instead of processed "
        "again.",
        "A call whose input starts with the same text as an earlier one reads that text from the cache while it lasts "
        "(Claude: 5 minutes, renewed on every read). An agent re-sends its whole conversation each turn, so this is "
        "usually the largest type.",
        "Per token, at the model's cache-read price, usually far below the input price. Every call that reads the text "
        "pays for it again.",
    ),
    "output": _billed(
        _TYPES,
        "Output",
        "The tokens the model generates: its reply text, its tool calls and, for Claude, its extended thinking.",
        "The model writes them one token at a time after reading the input. A Claude CLI specialist reports its "
        "output only when it finishes, so the output appears at its last turn.",
        "Per token, at the model's output price: several times the input price (Claude: 5 times, GPT-5: 8 times), so "
        "a small share of the tokens can be a large share of the cost.",
    ),
    "reasoning": _billed(
        _TYPES,
        "Reasoning output",
        "The hidden thinking of a reasoning model called over the OpenAI protocol (GPT-5, o-series, Codex, Kimi) "
        "before it answers; the reply does not show it.",
        "The OpenAI protocol reports these tokens inside the output count; the Inspector moves them here, so nothing "
        "is counted twice.",
        "Per token, at the model's output price, like Output, though the text is never returned.",
    ),
    "total": _term(
        _TOTALS,
        "Total tokens",
        "Uncached input + Input written to cache + Input read from cache + Output + Reasoning output, over all LLM "
        "calls.",
    ),
    "calls": _term(
        _TOTALS,
        "LLM calls",
        "Requests sent to a model's API; each turn of a Claude CLI specialist is one call.",
    ),
    "errors": _term(
        _TOTALS,
        "Failed calls",
        "LLM calls recorded with status error; any tokens they reported are still counted.",
    ),
    "cache_hit": _term(
        _TOTALS,
        "Input cache hit rate",
        "Input read from cache ÷ all input, by tokens: the share of input the provider did not process again.",
    ),
    "output_pending": _term(
        _TOTALS,
        "Output pending",
        "Turns of a running Claude CLI specialist whose output is not reported yet.",
    ),
    "output_unreported": _term(
        _TOTALS,
        "Output not reported",
        "Turns of a specialist that ended (cancelled, say) before reporting its output: that output is not counted.",
    ),
    "recent": _term(
        _TOTALS,
        f"Last {_RECENT_MIN} minutes",
        f"Tokens and LLM calls of the calls sent in the last {_RECENT_MIN} minutes.",
    ),
    "time": _term(
        _TOTALS,
        "Call time",
        "When a call's request was sent; the charts place each call there.",
    ),
    "agent": _term(
        _CHARTS,
        "Agent",
        "The harness graph node that made the call.",
    ),
    "model": _term(
        _CHARTS,
        "Model",
        "The model that answered the call, named by the specialist's log, else the call log; else no model recorded.",
    ),
    "share": _term(_CHARTS, "Share of tokens", "A row's total tokens ÷ the whole run's."),
    "running_total": _term(_CHARTS, "Running total", "Cumulative: at each time, everything since the run started."),
    "per_minute": _term(_CHARTS, "Per minute", "Rate: each bar is one time bucket's total ÷ its width in minutes."),
    "by_type": _term(
        _CHARTS,
        "By token type",
        "Bands stacked by token type.",
    ),
    "by_agent": _term(
        _CHARTS,
        "By agent",
        "Bands stacked by agent, the largest at the bottom.",
    ),
    "phases": _term(
        _CHARTS,
        "Phase bands",
        "The shaded spans behind the charts: the run's phases, from the Coordinator's phase history.",
    ),
    "now": _term(
        _CHARTS,
        "Now line",
        "The red line: the session's current time (the replay clock in a replay).",
    ),
}
