# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the scheduling side makes the :data:`~hyperloom_multibranch.contracts.llm.Completion` the contracts declare.

The judge and the integrator's patch resolver take a ``Completion`` through their constructors, so tests hand them a
deterministic stand-in; production builds one with :func:`anthropic_completion` over ``llm_config``, priced from the
price sheet when it lists the model. :func:`recorded` writes what each reply cost to the notebook as the scheduler's
own LLM usage.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping

from hyperloom.common.llm_config import (
    AGENT_BACKEND_CLAUDE,
    AnthropicCompletionRequest,
    aanthropic_completion,
    build_http_timeout,
    provider_model_defaults,
    resolve_forge_llm_model,
)
from hyperloom_multibranch.contracts.llm import Completion, CompletionFailed, CompletionReply
from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of
from hyperloom_multibranch.contracts.notebook.interfaces import UsageSink
from hyperloom_multibranch.contracts.notebook.records import LlmUsage
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource


logger = logging.getLogger(__name__)

COMPLETION_MAX_TOKENS = 8192
# src/hyperloom/orchestrator/kernel/patch_conflict_merge.py _LLM_CONNECT_TIMEOUT_S / _LLM_TIMEOUT_S: httpx's
# default 5 s read timeout ends a long reply before it arrives.
MB_LLM_CONNECT_TIMEOUT_SECONDS = 15.0
MB_LLM_READ_TIMEOUT_SECONDS = 120.0
# A patch resolver's reply carries a whole unified diff, which takes longer to write than a judgement: the
# ad4ab4d9 session lost two merge candidates to the 120 s read timeout rather than to a conflict it could not settle.
MB_RESOLVER_READ_TIMEOUT_SECONDS = 300.0


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


def recorded(completion: Completion, notebook: UsageSink, source: UsageSource) -> Completion:
    """``completion``, recording what each reply cost in ``notebook`` as the scheduler's ``source``; a failed call
    records nothing."""

    async def complete(system: str, user: str) -> CompletionReply:
        reply = await completion(system, user)
        usage = LlmUsage("", source, reply.cost_usd, reply.tokens)
        await notebook.record_llm_usage(usage)
        return reply

    return complete


def anthropic_completion(
    model: str, *, component: str, operation: str, read_seconds: float = MB_LLM_READ_TIMEOUT_SECONDS
) -> Completion:
    """A :data:`Completion` over ``llm_config``'s Anthropic transport, tagged with ``component`` and ``operation``,
    each reply priced at ``model``'s rates; a model the price sheet does not list gives replies with no cost, which
    no dollar cap counts, and says so once here. ``read_seconds`` bounds how long a reply may take to arrive."""

    price = price_of(model)
    if price is None:
        logger.warning("%s runs on %s, which the price sheet does not list: it records tokens only", operation, model)
    errors = _transport_errors()
    timeout = build_http_timeout(connect=MB_LLM_CONNECT_TIMEOUT_SECONDS, read=read_seconds)

    async def complete(system: str, user: str) -> CompletionReply:
        request = AnthropicCompletionRequest(
            model=model,
            messages=[{"role": "user", "content": user}],
            max_tokens=COMPLETION_MAX_TOKENS,
            system=system,
            timeout=timeout,
            timeout_s=read_seconds,
        )
        try:
            result = await aanthropic_completion(request=request, component=component, operation=operation)
        except errors as exc:
            raise CompletionFailed(f"{operation} call failed: {type(exc).__name__}: {exc}") from exc
        usage = result.usage if isinstance(result.usage, Mapping) else {}
        tokens = TokenCounts.from_anthropic_usage(usage)
        return CompletionReply(result.text, tokens, None if price is None else price.cost_usd(tokens))

    return complete


__all__ = [
    "MB_LLM_CONNECT_TIMEOUT_SECONDS",
    "MB_LLM_READ_TIMEOUT_SECONDS",
    "MB_RESOLVER_READ_TIMEOUT_SECONDS",
    "anthropic_completion",
    "default_model",
    "recorded",
]
