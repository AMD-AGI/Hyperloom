# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CodexBackend

Codex roles talk to GPT-style models via the OpenAI SDK; they're
**no-tools by default**, so the intent transport is JSON-in-text:

    {"intents": [{"intent_type": "...", "payload": {...}}]}

    The model is told (via prompt suffix) to wrap its reply in a JSON
    envelope. CodexBackend extracts the envelope (handles fenced
    ``json`` blocks or bare JSON), validates with the same
    ``validate_envelope`` the Claude path uses, and returns the same
    :class:`BackendTurnResult` shape. The Coordinator never has to know
    which backend produced the intents.

Authentication:

* `OPENAI_BASE_URL` (or `ANTHROPIC_BASE_URL`) — gateway endpoint
* `ANTHROPIC_AUTH_TOKEN` (or `OPENAI_API_KEY`) — auth token
  The AMD gateway serves both Claude AND OpenAI models from the same URL,
  hence we accept the ANTHROPIC_* env vars too. OPENAI_BASE_URL is the
  canonical (install.sh agrees); ANTHROPIC_BASE_URL is kept as a legacy
  fallback.

Test seam:

* `client_factory: Callable[[], openai.AsyncOpenAI]` — replace the SDK
  client so unit tests don't need real credentials or network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ...protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from .base import BackendError, BackendTurnResult, parse_call_timeout_env


_OUTPUT_INSTRUCTIONS = """
==== OUTPUT FORMAT (REQUIRED) ====
Reply with EXACTLY ONE JSON object that matches this envelope schema:

{
  "intents": [
    { "intent_type": "<one of send_message|propose_action|review_verdict|"
                     "alert|update_state|update_persona|ask_question|"
                     "answer>",
      "payload": { /* per-intent fields */ } }
  ]
}

Rules:
- Wrap the JSON in a ```json fenced block (the parser accepts bare JSON
  too, but the fenced form is preferred — it makes parser fallback
  unambiguous).
- Free text outside the JSON is ignored.
- ALWAYS emit at least one intent. If you have nothing to say, emit
  {"intent_type": "send_message", "payload": {"topic": "heartbeat",
  "body_md": "ok"}}.

For Critic specifically: when reviewing a proposal, emit
  {"intent_type": "review_verdict", "payload": {
      "target_proposal_msg_id": "<the proposal's msg_id from the inbox>",
      "verdict": "approve" | "reject" | "redirect" | "advise" | "needs_review",
      "reasoning": "<short, explicit reasoning>",
      "kb_evidence": "<kb_id if any, optional>"
  }}
==== END OUTPUT FORMAT ====
""".strip()


# Match a fenced ```json ... ``` block (preferred), falling back to a bare
# top-level {...}. We compile both up-front; runtime cost is negligible.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*?\"intents\".*\})", re.DOTALL)


def _extract_envelope(text: str) -> dict | None:
    """Pull the first valid JSON envelope out of a model reply."""
    if not text:
        return None
    # Prefer fenced — least ambiguous.
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "intents" in data:
            return data
    # Fall back to bare JSON containing "intents". We try greedy and
    # progressively shorten the string from the right until json.loads
    # accepts it — handles trailing prose without a fence.
    for m in _BARE_JSON_RE.finditer(text):
        candidate = m.group(1)
        for end in range(len(candidate), 0, -1):
            try:
                data = json.loads(candidate[:end])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "intents" in data:
                return data
            break  # parsed but wrong shape; don't keep shrinking
    return None


@dataclass
class CodexBackend:
    """Production Codex backend. Implements :class:`Backend`."""

    model: str = "gpt-5.4"
    api_key_env: str = "ANTHROPIC_AUTH_TOKEN"  # AMD proxy; accepts OPENAI too
    # Canonical LiteLLM env (install.sh agrees). When two base-URL envs
    # coexist, OPENAI_BASE_URL wins; ANTHROPIC_BASE_URL is the legacy
    # fallback.
    base_url_env: str = "OPENAI_BASE_URL"
    max_completion_tokens: int = 2000
    name: str = "codex"
    # Wall-clock cap for one ``run()`` call. Mirrors ClaudeBackend's
    # ``call_timeout_s``: the AsyncOpenAI client honours per-request
    # timeouts internally but a stalled gateway can still block the
    # ``await create(...)`` for the full TCP timeout. Bounding it at
    # asyncio level guarantees the orchestrator reactor never sits idle
    # past this budget.
    #
    # Env-var override ``INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC`` mirrors
    # the claude backend knob; same rationale (heavy critic / kernel prompts
    # may exceed 120 s on the AMD gateway under load).
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC",
            default=120.0,
        )
    )

    # Test seam — set to bypass real OpenAI client construction.
    client_factory: Callable[[], Any] | None = None

    calls: list[dict[str, Any]] = field(default_factory=list)
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client_factory is not None:
            self._client = self.client_factory()
            return
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise BackendError(
                "openai SDK not installed; run `pip install openai>=1.50`"
            ) from exc

        api_key = (
            os.environ.get(self.api_key_env)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise BackendError(
                f"{self.api_key_env} not set in env (CodexBackend cannot auth)"
            )
        base_url = (
            os.environ.get(self.base_url_env)
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
        )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,  # ignored — single API call per turn
    ) -> BackendTurnResult:
        full_prompt = f"{prompt}\n\n{_OUTPUT_INSTRUCTIONS}"
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_prompt})

        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                ),
                timeout=self.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise BackendError(
                f"Codex API call timed out after {self.call_timeout_s:.0f}s "
                "(likely upstream proxy stall)"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"Codex API call failed: {exc!r}") from exc

        choice = resp.choices[0]
        text = (choice.message.content or "")
        finish = getattr(choice, "finish_reason", None)
        # Token usage: the OpenAI chat-completions response carries a
        # ``usage`` object (prompt_tokens / completion_tokens). Map it
        # onto the SAME metadata keys ClaudeBackend uses so
        # Coordinator's accumulator stays backend-agnostic. OpenAI has
        # no prompt-cache split, so the two cache_* counters are 0.
        usage = getattr(resp, "usage", None)
        input_tokens = self._safe_int(getattr(usage, "prompt_tokens", None))
        output_tokens = self._safe_int(getattr(usage, "completion_tokens", None))
        self.calls.append({
            "prompt_chars": len(full_prompt),
            "reply_chars": len(text),
            "finish_reason": finish,
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        })

        envelope = _extract_envelope(text)
        if envelope is None:
            raise NoIntentEmitted(
                f"codex reply contained no parseable JSON envelope "
                f"(reply_chars={len(text)}, finish={finish})"
            )
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(
                f"codex envelope invalid: {exc}"
            ) from exc
        return BackendTurnResult(
            intents=intents,
            raw_text=text,
            metadata={
                "model": self.model,
                "finish_reason": finish,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Coerce a usage value to int, defaulting to 0 on None / bad type.

        Mirrors ``ClaudeBackend._safe_int`` so both backends report
        identically-shaped token counts on metadata.
        """
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0


__all__ = ["CodexBackend", "_extract_envelope"]
