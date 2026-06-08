# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Thin LLM client adapters for the report narrative pass.

We intentionally do NOT reuse :class:`ClaudeBackend` from
``inference_optimizer.orchestrator.backends.claude`` — that backend is
tightly coupled to the in-process MCP ``emit_intent`` tool contract,
which the report layer doesn't need. The report layer just needs
``(system, user) -> str``, so this module provides three minimal
adapters:

* :class:`OpenAIHttpClient` — POSTs to any OpenAI-compatible
  ``/chat/completions`` endpoint. Works for both the AMD primus-safe
  LiteLLM gateway and ``api.openai.com`` itself.
* :class:`AnthropicHttpClient` — POSTs to the Anthropic Messages API
  (used when only an ``ANTHROPIC_BASE_URL`` proxy is reachable).
* :class:`NullClient` — never used by compose (you'd just pass
  ``llm_client=None``); exposed so callers can branch cleanly.

:func:`build_client_from_env` reads ``HYPERLOOM_REPORT_LLM_BACKEND``
(``openai`` | ``anthropic`` | ``none``) and constructs the right one,
falling back to ``None`` when configuration is missing — the compose
layer then renders deterministic-only output, which is the desired
behavior for offline / batch runs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# httpx is already a Hyperloom runtime dependency (cli.py preflight
# installs it); we still import it lazily so users running offline
# without the package don't pay the import cost.

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
    """No-op client; compose treats this exactly like ``llm_client=None``.

    Kept as a real class so tests can pass a ``NullClient`` instance
    instead of remembering "pass None".
    """

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
    """Minimal OpenAI-compatible chat-completions client.

    No streaming, no tool calls, no function calling — just one POST
    and we return ``choices[0].message.content``. Tuned for use against
    the AMD primus-safe LiteLLM gateway (``OPENAI_BASE_URL``,
    ``OPENAI_API_KEY``) but works against ``api.openai.com`` directly
    too.
    """

    base_url: str
    api_key: str
    model: str = "claude-sonnet-4-5"
    max_output_tokens: int = 1024
    timeout_sec: float = 60.0

    def complete(self, *, system: str, user: str) -> str:
        """POST one chat-completion request and return the message text.

        Args:
            system (str): The system prompt.
            user (str): The user message.

        Returns:
            str: The content of ``choices[0].message.content`` (empty string
                if the model returned no content).

        Raises:
            LLMClientError: If the HTTP request fails or the response shape is
                unexpected.
        """
        import httpx  # local import (see module docstring rationale)

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
    """Minimal Anthropic Messages-API client.

    Talks directly to the upstream gateway pointed at by
    ``ANTHROPIC_BASE_URL``. The AMD primus-safe gateway accepts
    ``x-api-key`` natively, so the legacy auth-proxy on
    ``127.0.0.1:4002`` is no longer in the loop.
    """

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

    Reads (in priority order):

    * ``HYPERLOOM_REPORT_LLM_BACKEND`` — ``openai`` / ``anthropic`` /
      ``none``. Defaults to ``none`` so the dump-report CLI is safe to
      run on a stripped-down debug pod without network access.
    * ``HYPERLOOM_REPORT_MODEL`` — model id; falls back to
      ``claude-sonnet-4-5``.
    * ``HYPERLOOM_REPORT_MAX_TOKENS`` — int; defaults to 1024.
    * ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` for the OpenAI adapter,
      ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` for the Anthropic
      one. Missing required vars returns ``None`` (deterministic-only
      report).

    Returns:
        Any | None: A configured :class:`OpenAIHttpClient` or
            :class:`AnthropicHttpClient`, or ``None`` when the backend is
            disabled, unknown, or its required environment variables are unset.
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
