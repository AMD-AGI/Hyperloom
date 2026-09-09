# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Thin LLM client adapters for the report narrative pass."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from hyperloom.common import llm_config

log = logging.getLogger(__name__)

# Report narratives are a single long generation; the gateway is allowed a generous window because the alternative is
# a deterministic-only report.
REPORT_HTTP_TIMEOUT_SEC = 60.0

__all__ = [
    "REPORT_HTTP_TIMEOUT_SEC",
    "NullClient",
    "OpenAIHttpClient",
    "AnthropicClient",
    "build_client_from_env",
]


@dataclass
class NullClient:
    """No-op client; compose treats this exactly like ``llm_client=None``."""

    def complete(self, *, system: str, user: str) -> str:  # noqa: D401
        """Return an empty string, disabling the narrative pass."""
        return ""


@dataclass
class OpenAIHttpClient:
    """OpenAI-compatible chat-completions client for the narrative pass."""

    client: Any
    model: str = "claude-opus-5"
    max_output_tokens: int = 1024

    def complete(self, *, system: str, user: str) -> str:
        """Issue a single chat completion and return the text."""
        return llm_config.chat_completion(
            self.client,
            component="breakdown",
            operation="compose_report",
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_output_tokens,
            temperature=0.2,
        ).text


@dataclass
class AnthropicClient:
    """Anthropic client for the narrative pass."""

    model: str = "claude-opus-5"
    max_output_tokens: int = 1024
    timeout: Any = None
    timeout_s: float = REPORT_HTTP_TIMEOUT_SEC

    def complete(self, *, system: str, user: str) -> str:
        """Issue one single-shot completion and return the reply text."""
        return llm_config.anthropic_completion(
            component="breakdown",
            operation="compose_report",
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.max_output_tokens,
            timeout=self.timeout,
            timeout_s=self.timeout_s,
        ).text


def build_client_from_env() -> Any | None:
    """Construct an LLM client from environment."""
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
        if not llm_config.anthropic_transport_ready():
            log.warning(
                "HYPERLOOM_REPORT_LLM_BACKEND=anthropic but no usable Anthropic transport "
                "(credential missing, or the Claude CLI SDK its credential requires is "
                "not installed); falling back to deterministic-only report."
            )
            return None
        return AnthropicClient(model=model, max_output_tokens=max_tokens, timeout=timeout)
    log.warning(
        "Unknown HYPERLOOM_REPORT_LLM_BACKEND=%r; falling back to deterministic-only report.",
        backend,
    )
    return None
