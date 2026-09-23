# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical input-token semantics shared by every provider adapter.

Hyperloom's ``input_tokens`` follows Anthropic: it counts only the uncached input, and the cached prefix is reported
separately as ``cache_read_input_tokens``. OpenAI-family counters (``prompt_tokens``, Codex ``input_tokens``) already
include the cached prefix, so their adapters subtract it here before the value reaches a ledger.
"""

from __future__ import annotations

from typing import overload


@overload
def uncached_input_tokens(total_input_tokens: int, cached_input_tokens: int | None) -> int: ...


@overload
def uncached_input_tokens(total_input_tokens: None, cached_input_tokens: int | None) -> None: ...


def uncached_input_tokens(total_input_tokens: int | None, cached_input_tokens: int | None) -> int | None:
    """Return the uncached share of an input count that already includes its cached prefix."""
    if total_input_tokens is None:
        return None
    return max(0, total_input_tokens - (cached_input_tokens or 0))


__all__ = ["uncached_input_tokens"]
