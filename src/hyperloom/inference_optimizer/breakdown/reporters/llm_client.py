# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Thin LLM client adapters for the report narrative pass.

The report layer only needs ``(system, user) -> str`` (not the
orchestrator's MCP-coupled backend), so this exposes
:class:`OpenAIHttpClient`, :class:`AnthropicSdkClient`, and a no-op
:class:`NullClient`. :func:`build_client_from_env` picks one from
``HYPERLOOM_REPORT_LLM_BACKEND``, falling back to ``None``
(deterministic-only) when config is missing.

Provider credentials, client construction, and the request/response shape
belong to ``hyperloom.common.llm_config`` and, for the Anthropic side,
``hyperloom.common.claude_oneshot``; this module keeps only the
report-specific surface (``model``/``max_output_tokens`` defaults and the
``HYPERLOOM_REPORT_LLM_BACKEND``-driven env wiring).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from hyperloom.common import claude_oneshot, llm_config

log = logging.getLogger(__name__)

# Report narratives are a single long generation; the gateway is allowed a
# generous window because the alternative is a deterministic-only report.
REPORT_HTTP_TIMEOUT_SEC = 60.0

__all__ = [
    "REPORT_HTTP_TIMEOUT_SEC",
    "NullClient",
    "OpenAIHttpClient",
    "AnthropicSdkClient",
    "build_client_from_env",
]


@dataclass
class NullClient:
    """No-op client; compose treats this exactly like ``llm_client=None``."""

    def complete(self, *, system: str, user: str) -> str:  # noqa: D401
        """Return an empty string, disabling the narrative pass.

        Args:
            system (str): The system prompt (ignored).
            user (str): The user message (ignored).

        Returns:
            str: Always an empty string.
        """
        return ""


@dataclass
class OpenAIHttpClient:
    """OpenAI-compatible chat-completions client for the narrative pass."""

    client: Any
    model: str = "claude-opus-5"
    max_output_tokens: int = 1024

    def complete(self, *, system: str, user: str) -> str:
        """Issue a single chat completion and return the text.

        Args:
            system: System prompt content.
            user: User prompt content.

        Returns:
            The reply text.
        """
        return llm_config.chat_completion(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_output_tokens,
            temperature=0.2,
        ).text


@dataclass
class AnthropicSdkClient:
    """Anthropic client for the narrative pass, driven by the Claude CLI."""

    client: Any
    model: str = "claude-opus-5"
    max_output_tokens: int = 1024

    def complete(self, *, system: str, user: str) -> str:
        """Issue one single-shot completion and return the reply text.

        Args:
            system (str): The system prompt.
            user (str): The user message.

        Returns:
            str: The reply text.
        """
        return self.client.messages(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.max_output_tokens,
        ).text


def build_client_from_env() -> Any | None:
    """Construct an LLM client from environment.

    Reads ``HYPERLOOM_REPORT_LLM_BACKEND`` (default ``none``),
    ``HYPERLOOM_REPORT_MODEL`` and ``HYPERLOOM_REPORT_MAX_TOKENS``; the
    provider credentials come from ``hyperloom.common.llm_config``. Returns
    ``None`` (deterministic-only) when the backend is off or unconfigured.

    Returns:
        A configured backend client instance, or ``None`` when the backend
        is disabled or the provider credentials are missing.
    """
    backend = (os.environ.get("HYPERLOOM_REPORT_LLM_BACKEND") or "none").lower()
    if backend in ("", "none", "off", "disabled"):
        return None
    model = os.environ.get("HYPERLOOM_REPORT_MODEL") or "claude-opus-5"
    try:
        max_tokens = int(os.environ.get("HYPERLOOM_REPORT_MAX_TOKENS") or "1024")
    except ValueError:
        max_tokens = 1024
    timeout = llm_config.build_http_timeout(
        connect=REPORT_HTTP_TIMEOUT_SEC,
        read=REPORT_HTTP_TIMEOUT_SEC,
        write=REPORT_HTTP_TIMEOUT_SEC,
        pool=REPORT_HTTP_TIMEOUT_SEC,
    )

    if backend == "openai":
        try:
            client = llm_config.get_openai_client(timeout=timeout)
        except llm_config.LLMConfigError as exc:
            log.warning("HYPERLOOM_REPORT_LLM_BACKEND=openai but %s; falling back to deterministic-only report.", exc)
            return None
        return OpenAIHttpClient(client=client, model=model, max_output_tokens=max_tokens)
    if backend == "anthropic":
        try:
            claude_oneshot.ensure_available()
        except RuntimeError as exc:
            log.warning(
                "HYPERLOOM_REPORT_LLM_BACKEND=anthropic but %s; falling back to deterministic-only report.", exc
            )
            return None
        client = claude_oneshot.ClaudeOneShotClient(timeout_s=REPORT_HTTP_TIMEOUT_SEC)
        return AnthropicSdkClient(client=client, model=model, max_output_tokens=max_tokens)
    log.warning(
        "Unknown HYPERLOOM_REPORT_LLM_BACKEND=%r; falling back to deterministic-only report.",
        backend,
    )
    return None
