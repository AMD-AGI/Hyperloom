# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one LLM call shape this package asks a model with: a system and a user prompt in, the reply text, its tokens,
what they cost and why the model stopped out.

It sits in the contracts so that a module building the call over a transport and a module taking one through its
constructor — the scheduler and the rule gate's fix critic — name the same type without importing each other.
A call that fails raises :class:`CompletionFailed` whatever the transport, so every caller names one error.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from hyperloom_multibranch.contracts.llm_prices import TokenCounts


#: The stop reason of a reply the output token cap cut off, as the Anthropic Messages API names it.
STOPPED_AT_MAX_TOKENS = "max_tokens"


@dataclass(frozen=True)
class CompletionReply:
    """The reply text, its tokens and what they cost, ``None`` for a model the price sheet does not list.
    ``stop_reason`` is why the model stopped, ``None`` where the transport does not say."""

    text: str
    tokens: TokenCounts
    cost_usd: float | None
    stop_reason: str | None = None


#: ``(system, user) -> reply``; raises :class:`CompletionFailed` when the call fails.
Completion = Callable[[str, str], Awaitable[CompletionReply]]


class CompletionFailed(RuntimeError):
    """The model could not be asked or gave no reply; the caller keeps what it had."""


__all__ = ["STOPPED_AT_MAX_TOKENS", "Completion", "CompletionFailed", "CompletionReply"]
