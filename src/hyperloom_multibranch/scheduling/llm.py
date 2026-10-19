# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one LLM call shape the scheduling side uses: a system and a user prompt in, the reply text and its usage out.

The judge and the integrator's patch resolver take a :data:`Completion` through their constructors, so tests hand
them a deterministic stand-in; production builds one with :func:`anthropic_completion` over ``llm_config``. A call
that fails raises :class:`CompletionFailed` whatever the transport, so the callers name one error. A
:class:`CompletionMeter` adds up the tokens of every reply the completions it meters return.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from hyperloom.common.llm_config import (
    AGENT_BACKEND_CLAUDE,
    AnthropicCompletionRequest,
    aanthropic_completion,
    build_http_timeout,
    provider_model_defaults,
    resolve_forge_llm_model,
)


@dataclass(frozen=True)
class CompletionUsage:
    """The tokens one call or several used; ``input_tokens`` counts every prompt token, the cached ones included."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: CompletionUsage) -> CompletionUsage:
        return CompletionUsage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass(frozen=True)
class CompletionReply:
    text: str
    usage: CompletionUsage


#: ``(system, user) -> reply``; raises :class:`CompletionFailed` when the call fails.
Completion = Callable[[str, str], Awaitable[CompletionReply]]

COMPLETION_MAX_TOKENS = 8192
# src/hyperloom/orchestrator/kernel/patch_conflict_merge.py _LLM_CONNECT_TIMEOUT_S / _LLM_TIMEOUT_S: httpx's
# default 5 s read timeout ends a long reply before it arrives.
MB_LLM_CONNECT_TIMEOUT_SECONDS = 15.0
MB_LLM_READ_TIMEOUT_SECONDS = 120.0


class CompletionFailed(RuntimeError):
    """The model could not be asked or gave no reply; the caller keeps what it had."""


def default_model() -> str:
    """``CLAUDE_MODEL``, else the model the configured Anthropic endpoint serves, else the default Claude model."""

    return resolve_forge_llm_model(AGENT_BACKEND_CLAUDE, env={**os.environ, **provider_model_defaults()})


def _transport_errors() -> tuple[type[BaseException], ...]:
    """What ``aanthropic_completion`` raises when a call fails.

    The HTTP transport raises RuntimeError on a bad status or body (LLMConfigError is one too) and httpx errors on
    the wire; the SDK transport raises the SDK's own errors and a TimeoutError. Both SDKs ship in the ``llm`` extra a
    session cannot run without, and are imported here so importing this module stays cheap.
    """

    import httpx
    from claude_agent_sdk import ClaudeSDKError

    return (RuntimeError, OSError, TimeoutError, asyncio.TimeoutError, httpx.HTTPError, ClaudeSDKError)


#: The prompt-token counters of an Anthropic ``usage``: its ``input_tokens`` is only the uncached share.
_ANTHROPIC_INPUT_COUNTERS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _token_count(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _anthropic_usage(usage: object) -> CompletionUsage:
    """The tokens of an Anthropic ``usage`` mapping, as both transports report it; none when it carries no mapping."""

    if not isinstance(usage, Mapping):
        return CompletionUsage()
    return CompletionUsage(
        input_tokens=sum(_token_count(usage, key) for key in _ANTHROPIC_INPUT_COUNTERS),
        output_tokens=_token_count(usage, "output_tokens"),
    )


class CompletionMeter:
    """Adds up the usage of every reply the completions it :meth:`meters <metered>` return; a failed call adds none."""

    def __init__(self) -> None:
        self._used = CompletionUsage()

    @property
    def used(self) -> CompletionUsage:
        return self._used

    def metered(self, completion: Completion) -> Completion:
        """``completion``, counting each reply's usage here before returning it."""

        async def complete(system: str, user: str) -> CompletionReply:
            reply = await completion(system, user)
            self._used += reply.usage
            return reply

        return complete


def anthropic_completion(model: str, *, component: str, operation: str) -> Completion:
    """A :data:`Completion` over ``llm_config``'s Anthropic transport, tagged with ``component`` and ``operation``."""

    errors = _transport_errors()
    timeout = build_http_timeout(connect=MB_LLM_CONNECT_TIMEOUT_SECONDS, read=MB_LLM_READ_TIMEOUT_SECONDS)

    async def complete(system: str, user: str) -> CompletionReply:
        request = AnthropicCompletionRequest(
            model=model,
            messages=[{"role": "user", "content": user}],
            max_tokens=COMPLETION_MAX_TOKENS,
            system=system,
            timeout=timeout,
            timeout_s=MB_LLM_READ_TIMEOUT_SECONDS,
        )
        try:
            result = await aanthropic_completion(request=request, component=component, operation=operation)
        except errors as exc:
            raise CompletionFailed(f"{operation} call failed: {type(exc).__name__}: {exc}") from exc
        return CompletionReply(result.text, _anthropic_usage(result.usage))

    return complete


__all__ = [
    "MB_LLM_CONNECT_TIMEOUT_SECONDS",
    "MB_LLM_READ_TIMEOUT_SECONDS",
    "Completion",
    "CompletionFailed",
    "CompletionMeter",
    "CompletionReply",
    "CompletionUsage",
    "anthropic_completion",
    "default_model",
]
