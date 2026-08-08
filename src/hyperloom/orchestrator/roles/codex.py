# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CodexBackend — GPT-style models via the OpenAI SDK.

No-tools by default, so the intent transport is a JSON-in-text envelope
(``{"intents": [...]}``) validated with the same ``validate_envelope`` the
Claude path uses. Credentials come from the OpenAI side only
(``OPENAI_BASE_URL`` + ``OPENAI_API_KEY``); an Anthropic-only deployment fails to
construct this backend.

Optional web search: when ``HYPERLOOM_CODEX_WEB_SEARCH`` is enabled, every turn
uses the OpenAI **Responses API** with the built-in server-side ``web_search``
tool instead of ``chat.completions``. The search resolves server-side in one
call; the model's final ``output_text`` still carries the ``{"intents": [...]}``
envelope, so the intent-transport contract is unchanged.

Test seam: ``client_factory`` replaces the SDK client so unit tests need no
real credentials or network.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from hyperloom.common.env import env_bool, env_str
from hyperloom.common.llm_config import (
    LLMConfigError,
    achat_completion,
    apply_reasoning_effort,
    aresponse,
    get_async_openai_client,
)
from hyperloom.common.jsonio import extract_first_json_with_key
from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from .agent_role import DEFAULT_CODEX_MODEL
from .base import (
    BackendError,
    BackendTurnResult,
    LLMCallFailed,
    build_chat_messages,
    parse_call_timeout_env,
    safe_int,
)


_OUTPUT_INSTRUCTIONS = """
==== OUTPUT FORMAT (REQUIRED) ====
Reply with EXACTLY ONE JSON object that matches this envelope schema:

{
  "intents": [
    { "intent_type": "<one of send_message|propose_action|review_verdict|"
                     "alert|update_state>",
      "payload": { /* per-intent fields */ } }
  ]
}

Rules:
- Wrap the JSON in a ```json fenced block (the parser accepts bare JSON
  too, but the fenced form is preferred — it makes parser fallback
  unambiguous).
- Free text outside the JSON is ignored.
- Put only NEW information in payload bodies; do not restate context already
  in SharedState, your inbox, or analysis.md. Keep length proportional to substance.
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


# Bare top-level {...} fallback carrying "intents" (fenced case handled by helper).
_BARE_JSON_RE = re.compile(r"(\{.*?\"intents\".*\})", re.DOTALL)


def _extract_envelope(text: str) -> dict | None:
    """Pull the first valid ``{"intents": ...}`` envelope out of a model reply."""
    return extract_first_json_with_key(text, "intents", _BARE_JSON_RE)


@dataclass
class CodexBackend:
    """Production Codex backend. Implements :class:`Backend`."""

    model: str = DEFAULT_CODEX_MODEL
    # Prefer the OpenAI-side key/URL; ANTHROPIC_* vars are accepted as fallbacks.
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    max_completion_tokens: int = 2000
    name: str = "codex"
    # Wall-clock cap for one ``run()`` call. Env override:
    # ``INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC``.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC",
            default=120.0,
        )
    )

    # Web search: when enabled, each turn uses the OpenAI Responses API with the
    # built-in server-side ``web_search`` tool. Env: ``HYPERLOOM_CODEX_WEB_SEARCH``.
    web_search: bool = field(default_factory=lambda: env_bool("HYPERLOOM_CODEX_WEB_SEARCH", False))
    # ``low`` | ``medium`` | ``high`` — passed through as the web_search tool's
    # ``search_context_size``. Env: ``HYPERLOOM_CODEX_WEB_SEARCH_CONTEXT_SIZE``.
    web_search_context_size: str = field(
        default_factory=lambda: env_str("HYPERLOOM_CODEX_WEB_SEARCH_CONTEXT_SIZE", "medium")
    )

    # Test seam — set to bypass real OpenAI client construction.
    client_factory: Callable[[], Any] | None = None

    calls: list[dict[str, Any]] = field(default_factory=list)
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Construct the OpenAI client (or use the test factory).

        When ``client_factory`` is set it builds the client directly (test
        seam). Otherwise it asks the shared LLM gateway for an async client
        keyed on the configured env vars.

        Raises:
            BackendError: If the ``openai`` SDK is not installed or no API key
                is found in the environment.
        """
        if self.client_factory is not None:
            self._client = self.client_factory()
            return
        try:
            self._client = get_async_openai_client(
                api_key_env=self.api_key_env,
                base_url_env=self.base_url_env,
            )
        except LLMConfigError as exc:
            raise BackendError(str(exc).replace("OpenAI-compatible client", "CodexBackend")) from exc

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,  # ignored — single API call per turn
    ) -> BackendTurnResult:
        """Run one turn via a single API call and parse intents.

        Appends the JSON-envelope output instructions to ``prompt``, issues one
        bounded request -- ``chat.completions`` by default, or the Responses API
        with the built-in server-side ``web_search`` tool when ``web_search`` is
        enabled -- extracts and validates the returned envelope, and records
        call telemetry.

        Args:
            prompt (str): The composed turn prompt.
            system_prompt (str | None): Optional system prompt sent as the
                leading system message.
            tools (list[str] | None): Unused; Codex roles are no-tools by
                default.
            max_turns (int): Ignored; one API call is made per turn.

        Returns:
            BackendTurnResult: The validated intents plus raw reply text and
            model/finish metadata.

        Raises:
            BackendError: If the API call times out or otherwise fails.
            NoIntentEmitted: If the reply has no parseable envelope or the
                envelope fails intent validation.
        """
        full_prompt = f"{prompt}\n\n{_OUTPUT_INSTRUCTIONS}"
        if self.web_search:
            text, finish, input_tokens, output_tokens, extra_meta = await self._run_responses(
                system_prompt, full_prompt
            )
        else:
            text, finish, input_tokens, output_tokens, extra_meta = await self._run_chat(system_prompt, full_prompt)
        self.calls.append(
            {
                "prompt_chars": len(full_prompt),
                "reply_chars": len(text),
                "finish_reason": finish,
                "model": self.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        )

        envelope = _extract_envelope(text)
        if envelope is None:
            raise NoIntentEmitted(
                f"codex reply contained no parseable JSON envelope (reply_chars={len(text)}, finish={finish})"
            )
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"codex envelope invalid: {exc}") from exc
        metadata: dict[str, Any] = {
            "model": self.model,
            "finish_reason": finish,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            # Full conversation text for conversations.jsonl, handed up for
            # the caller to persist.
            "prompt": full_prompt,
            "response": text,
        }
        metadata.update(extra_meta)
        return BackendTurnResult(intents=intents, raw_text=text, metadata=metadata)

    # ------------------------------------------------------------------
    async def _run_chat(
        self,
        system_prompt: str | None,
        full_prompt: str,
    ) -> tuple[str, Any, int, int, dict[str, Any]]:
        """Single ``chat.completions`` call (the default, no-tools path).

        Returns:
            ``(text, finish_reason, input_tokens, output_tokens, extra_metadata)``.
            ``extra_metadata`` is empty for this path.
        """
        messages = build_chat_messages(system_prompt, full_prompt)
        create_params = apply_reasoning_effort(
            {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.max_completion_tokens,
            }
        )
        try:
            result = await asyncio.wait_for(
                achat_completion(self._client, **create_params),
                timeout=self.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise LLMCallFailed(
                f"Codex API call timed out after {self.call_timeout_s:.0f}s (likely upstream proxy stall)"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMCallFailed(f"Codex API call failed: {exc!r}") from exc

        # Map OpenAI usage onto the SAME metadata keys ClaudeBackend uses so the
        # Coordinator's accumulator stays backend-agnostic; cache_* counters are 0.
        input_tokens = safe_int(getattr(result.usage, "prompt_tokens", None))
        output_tokens = safe_int(getattr(result.usage, "completion_tokens", None))
        return result.text, result.finish_reason, input_tokens, output_tokens, {}

    # ------------------------------------------------------------------
    async def _run_responses(
        self,
        system_prompt: str | None,
        full_prompt: str,
    ) -> tuple[str, Any, int, int, dict[str, Any]]:
        """Single Responses API call with the built-in ``web_search`` tool.

        The web search resolves server-side within this one call, so no
        client-side tool loop is needed. The final ``output_text`` still carries
        the intent envelope. Returns
        ``(text, status, input_tokens, output_tokens, extra_metadata)`` where
        ``extra_metadata`` may carry ``web_search_citations``.
        """
        tool_spec: dict[str, Any] = {"type": "web_search"}
        ctx = (self.web_search_context_size or "").strip().lower()
        if ctx in {"low", "medium", "high"}:
            tool_spec["search_context_size"] = ctx
        params: dict[str, Any] = {
            "model": self.model,
            "input": full_prompt,
            "tools": [tool_spec],
            "max_output_tokens": self.max_completion_tokens,
        }
        if system_prompt:
            params["instructions"] = system_prompt
        # The Responses API expresses reasoning effort as ``reasoning={"effort": ...}``
        # (not the chat-completions ``reasoning_effort`` field); translate via the
        # shared env-vocabulary helper so the same knob applies on both paths.
        _eff = apply_reasoning_effort({})
        if "reasoning_effort" in _eff:
            params["reasoning"] = {"effort": _eff["reasoning_effort"]}
        try:
            result = await asyncio.wait_for(
                aresponse(self._client, **params),
                timeout=self.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise LLMCallFailed(
                f"Codex Responses API call timed out after {self.call_timeout_s:.0f}s (likely upstream proxy stall)"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMCallFailed(f"Codex Responses API call failed: {exc!r}") from exc

        extra_meta: dict[str, Any] = {}
        if result.citations:
            extra_meta["web_search_citations"] = result.citations
        return result.text, result.status, result.input_tokens, result.output_tokens, extra_meta


__all__ = ["CodexBackend", "_extract_envelope"]
