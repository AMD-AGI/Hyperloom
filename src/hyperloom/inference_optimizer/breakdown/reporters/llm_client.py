# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Thin LLM client adapters for the report narrative pass.

The report layer only needs ``(system, user) -> str`` (not the
orchestrator's MCP-coupled backend), so this exposes
:class:`OpenAIHttpClient`, :class:`AnthropicHttpClient`, and a no-op
:class:`NullClient`. :func:`build_client_from_env` picks one from
``HYPERLOOM_REPORT_LLM_BACKEND``, falling back to ``None``
(deterministic-only) when config is missing.

The underlying HTTP protocol skeleton (one POST, parse the response) is
shared via ``hyperloom.common.llm.http_client``; this module keeps the
report-specific surface (``model``/``max_output_tokens`` field defaults, the
``HYPERLOOM_REPORT_LLM_BACKEND``-driven env wiring) local.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from hyperloom.common.llm.http_client import (
    LLMClientError,
    call_anthropic_messages,
    call_openai_chat_completions,
)

log = logging.getLogger(__name__)

__all__ = [
    "LLMClientError",
    "NullClient",
    "OpenAIHttpClient",
    "AnthropicHttpClient",
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
    """Minimal OpenAI-compatible chat-completions client (one POST, returns ``choices[0].message.content``)."""

    base_url: str
    api_key: str
    model: str = "claude-sonnet-4-5"
    max_output_tokens: int = 1024
    timeout_sec: float = 60.0

    def complete(self, *, system: str, user: str) -> str:
        """Issue a single chat-completion request and return the text.

        Args:
            system: System prompt content.
            user: User prompt content.

        Returns:
            The content of the first choice's message.
        """
        return call_openai_chat_completions(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            system=system,
            user=user,
            max_output_tokens=self.max_output_tokens,
            timeout_sec=self.timeout_sec,
            temperature=0.2,
        )


@dataclass
class AnthropicHttpClient:
    """Minimal Anthropic Messages-API client (talks directly to ``ANTHROPIC_BASE_URL``)."""

    base_url: str
    api_key: str
    model: str = "claude-sonnet-4-5"
    max_output_tokens: int = 1024
    timeout_sec: float = 60.0

    def complete(self, *, system: str, user: str) -> str:
        """POST one Messages-API request and return the concatenated text.

        Args:
            system (str): The system prompt.
            user (str): The user message.

        Returns:
            str: The joined text of all ``text`` content blocks in the
                response.

        Raises:
            LLMClientError: If the HTTP request fails or the response shape is
                unexpected.
        """
        return call_anthropic_messages(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            system=system,
            user=user,
            max_output_tokens=self.max_output_tokens,
            timeout_sec=self.timeout_sec,
        )


def build_client_from_env() -> Any | None:
    """Construct an LLM client from environment.

    Reads ``HYPERLOOM_REPORT_LLM_BACKEND`` (default ``none``),
    ``HYPERLOOM_REPORT_MODEL``, ``HYPERLOOM_REPORT_MAX_TOKENS`` and the
    per-backend base-url/api-key vars; returns ``None``
    (deterministic-only) when required vars are missing.

    Returns:
        A configured backend client instance, or ``None`` when the backend
        is disabled or required environment variables are missing.
    """
    backend = (os.environ.get("HYPERLOOM_REPORT_LLM_BACKEND") or "none").lower()
    if backend in ("", "none", "off", "disabled"):
        return None
    model = os.environ.get("HYPERLOOM_REPORT_MODEL") or "claude-sonnet-4-5"
    try:
        max_tokens = int(os.environ.get("HYPERLOOM_REPORT_MAX_TOKENS") or "1024")
    except ValueError:
        max_tokens = 1024

    if backend == "openai":
        base = os.environ.get("OPENAI_BASE_URL")
        key = os.environ.get("OPENAI_API_KEY")
        if not (base and key):
            log.warning(
                "HYPERLOOM_REPORT_LLM_BACKEND=openai but OPENAI_BASE_URL or "
                "OPENAI_API_KEY is unset; falling back to deterministic-only "
                "report."
            )
            return None
        return OpenAIHttpClient(
            base_url=base,
            api_key=key,
            model=model,
            max_output_tokens=max_tokens,
        )
    if backend == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL")
        key = (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("CLAUDE_API_KEY")
            or os.environ.get("PRIMUS_SAFE_API_KEY")
        )
        if not (base and key):
            log.warning(
                "HYPERLOOM_REPORT_LLM_BACKEND=anthropic but ANTHROPIC_BASE_URL "
                "or ANTHROPIC_API_KEY is unset; falling back to "
                "deterministic-only report."
            )
            return None
        return AnthropicHttpClient(
            base_url=base,
            api_key=key,
            model=model,
            max_output_tokens=max_tokens,
        )
    log.warning(
        "Unknown HYPERLOOM_REPORT_LLM_BACKEND=%r; falling back to deterministic-only report.",
        backend,
    )
    return None
