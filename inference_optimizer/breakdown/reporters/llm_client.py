# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Thin LLM client adapters for the report narrative pass.

The report layer only needs ``(system, user) -> str`` (not the
orchestrator's MCP-coupled backend), so this exposes
:class:`OpenAIHttpClient`, :class:`AnthropicHttpClient`, and a no-op
:class:`NullClient`. :func:`build_client_from_env` picks one from
``HYPERLOOM_REPORT_LLM_BACKEND``, falling back to ``None``
(deterministic-only) when config is missing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# httpx is imported lazily so offline users without it don't pay the cost.

__all__ = [
    "LLMClientError",
    "NullClient",
    "OpenAIHttpClient",
    "AnthropicHttpClient",
    "build_client_from_env",
]


class LLMClientError(RuntimeError):
    """Wraps any client-side error so compose can degrade gracefully."""


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
        import httpx  # local import (see module docstring)

        url = self.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_output_tokens,
            "temperature": 0.2,  # narrative pass — keep prose stable but not robotic
        }
        try:
            r = httpx.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                content=json.dumps(body),
                timeout=self.timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMClientError(f"openai chat-completions failed: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                f"unexpected openai response shape: {data!r}"
            ) from exc


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
        import httpx

        url = self.base_url.rstrip("/") + "/v1/messages"
        body = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        try:
            r = httpx.post(
                url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                content=json.dumps(body),
                timeout=self.timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMClientError(f"anthropic messages failed: {exc}") from exc
        try:
            blocks = data.get("content") or []
            text_parts = [b.get("text") or "" for b in blocks if b.get("type") == "text"]
            return "".join(text_parts)
        except Exception as exc:  # noqa: BLE001
            raise LLMClientError(
                f"unexpected anthropic response shape: {data!r}"
            ) from exc


def build_client_from_env() -> Any | None:
    """Construct an LLM client from environment.

    Reads ``HYPERLOOM_REPORT_LLM_BACKEND`` (default ``none``),
    ``HYPERLOOM_REPORT_MODEL``, ``HYPERLOOM_REPORT_MAX_TOKENS`` and the
    per-backend base-url/api-key vars; returns ``None``
    (deterministic-only) when required vars are missing.
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
            base_url=base, api_key=key, model=model,
            max_output_tokens=max_tokens,
        )
    if backend == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL")
        key = (os.environ.get("ANTHROPIC_API_KEY")
               or os.environ.get("CLAUDE_API_KEY")
               or os.environ.get("PRIMUS_SAFE_API_KEY"))
        if not (base and key):
            log.warning(
                "HYPERLOOM_REPORT_LLM_BACKEND=anthropic but ANTHROPIC_BASE_URL "
                "or ANTHROPIC_API_KEY is unset; falling back to "
                "deterministic-only report."
            )
            return None
        return AnthropicHttpClient(
            base_url=base, api_key=key, model=model,
            max_output_tokens=max_tokens,
        )
    log.warning(
        "Unknown HYPERLOOM_REPORT_LLM_BACKEND=%r; falling back to "
        "deterministic-only report.", backend,
    )
    return None
