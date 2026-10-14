# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one LLM call shape the scheduling side uses: a system and a user prompt in, the reply text out.

The judge and the integrator's patch resolver take a :data:`Completion` through their constructors, so tests hand
them a deterministic stand-in; production builds one with :func:`anthropic_completion` over ``llm_config``. A call
that fails raises :class:`CompletionFailed` whatever the transport, so the callers name one error.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

from hyperloom.common.llm_config import (
    AGENT_BACKEND_CLAUDE,
    aanthropic_completion,
    build_http_timeout,
    provider_model_defaults,
    resolve_forge_llm_model,
)

#: ``(system, user) -> reply text``; raises :class:`CompletionFailed` when the call fails.
Completion = Callable[[str, str], Awaitable[str]]

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


def anthropic_completion(model: str, *, component: str, operation: str) -> Completion:
    """A :data:`Completion` over ``llm_config``'s Anthropic transport, tagged with ``component`` and ``operation``."""

    errors = _transport_errors()
    timeout = build_http_timeout(connect=MB_LLM_CONNECT_TIMEOUT_SECONDS, read=MB_LLM_READ_TIMEOUT_SECONDS)

    async def complete(system: str, user: str) -> str:
        try:
            result = await aanthropic_completion(
                model=model,
                messages=[{"role": "user", "content": user}],
                max_tokens=COMPLETION_MAX_TOKENS,
                system=system,
                timeout=timeout,
                timeout_s=MB_LLM_READ_TIMEOUT_SECONDS,
                component=component,
                operation=operation,
            )
        except errors as exc:
            raise CompletionFailed(f"{operation} call failed: {type(exc).__name__}: {exc}") from exc
        return result.text

    return complete


__all__ = [
    "MB_LLM_CONNECT_TIMEOUT_SECONDS",
    "MB_LLM_READ_TIMEOUT_SECONDS",
    "Completion",
    "CompletionFailed",
    "anthropic_completion",
    "default_model",
]
