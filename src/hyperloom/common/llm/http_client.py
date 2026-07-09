# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Minimal OpenAI-compatible / Anthropic Messages-API POST-and-parse helpers.

Each function issues exactly one synchronous HTTP request and returns the
completion text (or raises :class:`LLMClientError`). This is the shared
"protocol skeleton" that ``hyperloom.inference_optimizer``'s report-narrative
client builds on (``breakdown/reporters/llm_client.py``); it is deliberately
NOT an SDK replacement — role backends that need streaming, tool-calling, or
retry policies (``orchestrator/roles/{claude,codex,critic_agent}.py``,
``orchestrator/scoring/proposal_scorer.py``) continue to use the ``openai`` /
``claude_agent_sdk`` packages directly (tree-reform.MD §12.2: role-specific
agentic behavior stays in ``orchestrator/roles/``).

Callers own credential resolution (which env var, what fallback order); this
module only accepts an already-resolved ``base_url``/``api_key`` pair.

``httpx`` is imported lazily so importing this module never pays the cost for
callers that end up not needing it (mirrors the prior per-call-site lazy
import convention in ``breakdown/reporters/llm_client.py``).
"""

from __future__ import annotations

import json

# Matches the previous per-call-site default in
# ``breakdown/reporters/llm_client.py`` (``timeout_sec: float = 60.0``).
DEFAULT_HTTP_TIMEOUT_SEC = 60.0


class LLMClientError(RuntimeError):
    """Wraps any client-side HTTP/parsing error so callers can degrade gracefully."""


def call_openai_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
    temperature: float = 0.2,
) -> str:
    """POST one OpenAI-compatible chat-completions request; return the reply text.

    Args:
        base_url: API base URL (``/chat/completions`` is appended).
        api_key: Bearer token for the ``Authorization`` header.
        model: Model id to request.
        system: System prompt content.
        user: User prompt content.
        max_output_tokens: ``max_tokens`` cap on the response.
        timeout_sec: Request timeout in seconds.
        temperature: Sampling temperature.

    Returns:
        The content of the first choice's message (``""`` if empty).

    Raises:
        LLMClientError: If the HTTP request fails or the response shape is
            unexpected.
    """
    import httpx  # local import (see module docstring)

    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }
    try:
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            content=json.dumps(body),
            timeout=timeout_sec,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError(f"openai chat-completions failed: {exc}") from exc
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError(f"unexpected openai response shape: {data!r}") from exc


def call_anthropic_messages(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
    anthropic_version: str = "2023-06-01",
) -> str:
    """POST one Anthropic Messages-API request; return the concatenated text.

    Args:
        base_url: API base URL (``/v1/messages`` is appended).
        api_key: Value sent in the ``x-api-key`` header.
        model: Model id to request.
        system: System prompt content.
        user: User prompt content.
        max_output_tokens: ``max_tokens`` cap on the response.
        timeout_sec: Request timeout in seconds.
        anthropic_version: ``anthropic-version`` header value.

    Returns:
        The joined text of all ``text`` content blocks in the response.

    Raises:
        LLMClientError: If the HTTP request fails or the response shape is
            unexpected.
    """
    import httpx

    url = base_url.rstrip("/") + "/v1/messages"
    body = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        r = httpx.post(
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": anthropic_version,
                "Content-Type": "application/json",
            },
            content=json.dumps(body),
            timeout=timeout_sec,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError(f"anthropic messages failed: {exc}") from exc
    try:
        blocks = data.get("content") or []
        text_parts: list[str] = [b.get("text") or "" for b in blocks if b.get("type") == "text"]
        return "".join(text_parts)
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError(f"unexpected anthropic response shape: {data!r}") from exc


__all__: list[str] = [
    "DEFAULT_HTTP_TIMEOUT_SEC",
    "LLMClientError",
    "call_openai_chat_completions",
    "call_anthropic_messages",
]
