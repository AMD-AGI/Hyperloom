# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical input-token semantics shared by every provider adapter.

Hyperloom's ``input_tokens`` follows Anthropic: it counts only the uncached input, and the cached prefix is reported
separately as ``cache_read_input_tokens``. OpenAI-family counters (``prompt_tokens``, Codex ``input_tokens``) already
include the cached prefix, so their adapters subtract it here before the value reaches a ledger.
"""

from __future__ import annotations

from typing import overload

from hyperloom.common.coerce import to_int

TOKEN_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


@overload
def uncached_input_tokens(total_input_tokens: int, cached_input_tokens: int | None) -> int: ...


@overload
def uncached_input_tokens(total_input_tokens: None, cached_input_tokens: int | None) -> None: ...


def uncached_input_tokens(total_input_tokens: int | None, cached_input_tokens: int | None) -> int | None:
    """Return the uncached share of an input count that already includes its cached prefix."""
    if total_input_tokens is None:
        return None
    return max(0, total_input_tokens - (cached_input_tokens or 0))


def sdk_field(obj: object, key: str) -> object:
    """Read ``key`` off a dict or an attribute-carrying object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _token_count(usage: object, key: str) -> int:
    """Read one ``usage`` counter as a non-negative int; ``0`` when absent, not numeric, or negative."""
    return max(0, to_int(sdk_field(usage, key), default=0))  # type: ignore[type-var]


def openai_usage_counts(usage: object) -> dict[str, int]:
    """An OpenAI ``usage`` as Hyperloom's four counters; ``prompt_tokens`` already includes both cache shares."""
    if usage is None:
        return {}
    cached = _token_count(sdk_field(usage, "prompt_tokens_details") or {}, "cached_tokens")
    created = _token_count(usage, "cache_creation_input_tokens")
    return {
        "input_tokens": uncached_input_tokens(_token_count(usage, "prompt_tokens"), cached + created),
        "output_tokens": _token_count(usage, "completion_tokens"),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": created,
    }


def anthropic_usage_counts(usage: object) -> dict[str, int]:
    """An Anthropic ``usage``, whose ``input_tokens`` is already the uncached share."""
    if usage is None:
        return {}
    return {key: _token_count(usage, key) for key in TOKEN_KEYS}


__all__ = [
    "TOKEN_KEYS",
    "anthropic_usage_counts",
    "openai_usage_counts",
    "sdk_field",
    "uncached_input_tokens",
]
