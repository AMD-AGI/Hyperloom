"""CodexBackend — drives no-tools Codex roles via the OpenAI Chat
Completions API (DESIGN §5.1.1 / §10.5.5).

The Codex transport is much simpler than Claude's:

- No tools. The model reply is plain text.
- The reply MUST contain a single ``validated_json_output`` JSON envelope
  (see :func:`parse_codex_validated_json`). Every Critic / Sage system
  prompt already says so; this backend just appends a final reminder
  block in case the role prompt got dropped.
- One repair attempt on parse failure, using
  :func:`build_repair_prompt` (DESIGN §10.5.5).

Wire-protocol compatibility
---------------------------

Anything OpenAI-compatible works:

- ``api.openai.com`` directly with a real OPENAI_API_KEY.
- An OpenAI-compatible proxy (Azure / Foundry / a corporate gateway like
  the AMD primus-safe LLM proxy) — set ``OPENAI_BASE_URL`` (or pass
  ``base_url=...``).

Self-signed proxy certs are common in corp deployments. Pass
``verify_ssl=False`` (or set
``INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL=0``) to skip TLS verification.
We build a dedicated ``httpx.AsyncClient`` for that case so the rest of
the process is unaffected.

Test seam
---------

Construct ``CodexBackend(client=fake)`` where ``fake`` is anything with
``.chat.completions.create(...)`` returning an awaitable producing a
``Choices``-shaped object. See ``tests/test_codex_backend.py``.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..intent_parser import (
    Intent,
    IntentValidationError,
    NoIntentEmitted,
    build_repair_prompt,
    parse_codex_validated_json,
)
from .base import Backend, BackendError


_OUTPUT_INSTRUCTIONS = """
==== OUTPUT FORMAT (REQUIRED) ====
Reply with EXACTLY ONE JSON object inside a fenced ``validated_json_output``
block. No commentary outside the block — anything else is discarded.

```validated_json_output
{
  "intents": [
    { "intent_type": "<see table>", "payload": { /* see table */ } }
  ]
}
```

Allowed ``intent_type`` values + REQUIRED payload fields (additional
optional fields are fine — only the required ones are validated):

  send_message    -> { topic, body_md? }
  ask_question    -> { topic, question }
  answer          -> { in_reply_to, answer }
  alert           -> { severity, summary, detail? }
  objection       -> { target_msg_id, reason, severity? }
  vote            -> { target_msg_id, vote }
  propose_action  -> { action_name, predicted_gain_pct, params? }
  delegate        -> { action_name, params?, idempotency_key? }
  update_state    -> { changes }
  update_persona  -> { body_md }

Rules:
- The ``intents`` array MUST contain at least one element.
- ``topic`` for send_message MUST be one of: heartbeat, observation,
  decision, alert, question, answer, objection, vote, proposal,
  reflection_tick, rca_finding, kb_synthesis, kb_recall, event,
  graceful_stop. Use ``observation`` for free-form notes; use
  ``rca_finding`` for post-mortem hypotheses.
- If you have nothing useful to say, emit a single send_message with
  payload ``{"topic": "heartbeat", "body_md": "no-op tick"}``.
- Free text outside the fenced block is ignored.
==== END OUTPUT FORMAT ====
""".strip()


# ---------------------------------------------------------------------------
def _import_openai_sdk() -> Any:
    """Return the imported ``openai`` module or raise ``BackendError``."""
    try:
        return importlib.import_module("openai")
    except ImportError as exc:
        raise BackendError(
            "openai SDK not installed; run `pip install openai>=1.50`."
        ) from exc


def _maybe_import_httpx() -> Any | None:
    try:
        return importlib.import_module("httpx")
    except ImportError:  # pragma: no cover - httpx ships with openai
        return None


def _resolve_verify_ssl(explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(
        "INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL", ""
    ).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


# ---------------------------------------------------------------------------
@dataclass
class CodexBackend(Backend):
    """Production backend for no-tools Codex roles (Critic / Sage).

    Args:
        model: Model id, e.g. ``"gpt-5.4"``. Falls back to
            ``OPENAI_MODEL`` env, then to ``"gpt-5.4"`` literal.
        api_key_env: Env var holding the API key. Default
            ``"OPENAI_API_KEY"``.
        base_url: API base URL override (or read from ``OPENAI_BASE_URL``).
        verify_ssl: Skip TLS cert verification when ``False`` (or via
            ``INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL=0``).
        request_timeout_s: Per-request timeout passed to httpx + the SDK.
        max_completion_tokens: Cap on response tokens. ``None`` lets the
            proxy / model pick a default (recommended for gpt-5 family
            models that ignore ``max_tokens``).
        repair_attempts: How many extra calls we'll try on parse failure
            (uses :func:`build_repair_prompt`). 0 = no repair.
        client: Test seam — replace the AsyncOpenAI client. Anything with
            ``.chat.completions.create(model, messages, ...)`` returning an
            awaitable resolving to an object with
            ``.choices[0].message.content``.
        sdk_module: Test seam — replace the imported openai module so
            tests don't pull the SDK in.
    """

    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    verify_ssl: bool | None = None
    request_timeout_s: float = 120.0
    max_completion_tokens: int | None = None
    temperature: float | None = None
    repair_attempts: int = 1

    client: Any | None = None
    sdk_module: Any | None = None
    _http_client: Any | None = field(default=None, init=False, repr=False)

    calls: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.client is not None:
            # Test path or pre-built client — nothing else to do.
            return
        sdk = self.sdk_module or _import_openai_sdk()
        self.sdk_module = sdk

        if not os.environ.get(self.api_key_env):
            self.calls.append(
                {"warn": f"{self.api_key_env} not set in env"}
            )
        api_key = os.environ.get(self.api_key_env, "") or "EMPTY"

        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL") or None
        verify = _resolve_verify_ssl(self.verify_ssl)

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        if not verify:
            httpx_mod = _maybe_import_httpx()
            if httpx_mod is not None:
                self._http_client = httpx_mod.AsyncClient(
                    verify=False,
                    timeout=self.request_timeout_s,
                )
                client_kwargs["http_client"] = self._http_client

        async_client_cls = getattr(sdk, "AsyncOpenAI", None)
        if async_client_cls is None:  # pragma: no cover - very old SDK
            raise BackendError(
                "openai SDK has no AsyncOpenAI; upgrade to openai>=1.0."
            )
        try:
            self.client = async_client_cls(**client_kwargs)
        except Exception as exc:  # noqa: BLE001
            # Construction failure is deferred to ``.run()`` so callers that
            # only want to introspect the backend (CLI smoke tests, etc.)
            # can still build it. The real error surfaces at first call.
            self.calls.append(
                {"warn": f"AsyncOpenAI construction failed: {exc!r}"}
            )
            self.client = None

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        agent_name: str,
        allowed_tools: Sequence[str] = (),
        max_turns: int = 0,
        extra: dict | None = None,
    ) -> list[Intent]:
        """Codex is no-tools by definition — ``allowed_tools`` /
        ``max_turns`` are accepted for interface parity with
        :class:`ClaudeBackend` and ignored here.

        ``extra`` is purely metadata (role name, task_id, ...) — used for
        per-call logging only; never forwarded to the SDK.
        """
        full_prompt = self._compose_prompt(prompt)
        last_error: Exception | None = None
        for attempt in range(self.repair_attempts + 1):
            current = (
                full_prompt
                if attempt == 0
                else build_repair_prompt(
                    full_prompt,
                    last_error,
                    fenced_label="validated_json_output",
                )
            )
            text = await self._chat_complete(current)
            self.calls.append(
                {
                    "agent": agent_name,
                    "attempt": attempt,
                    "text_chars": len(text),
                    "extra": dict(extra or {}),
                }
            )
            try:
                return parse_codex_validated_json(text)
            except (IntentValidationError, NoIntentEmitted) as exc:
                last_error = exc
                if attempt == self.repair_attempts:
                    raise BackendError(
                        f"CodexBackend: failed to parse intents after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
        # unreachable
        raise BackendError("CodexBackend: exhausted retries unexpectedly")

    # ------------------------------------------------------------------
    def _compose_prompt(self, prompt: str) -> str:
        return f"{prompt.rstrip()}\n\n{_OUTPUT_INSTRUCTIONS}\n"

    async def _chat_complete(self, prompt: str) -> str:
        if self.client is None:
            raise BackendError(
                "CodexBackend: AsyncOpenAI client is not initialised. "
                "Check that ``openai`` is installed and "
                f"{self.api_key_env} / OPENAI_BASE_URL are set."
            )
        kwargs: dict[str, Any] = {
            "model": self.model or "gpt-5.4",
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.request_timeout_s,
        }
        if self.max_completion_tokens is not None:
            # Newer reasoning models use ``max_completion_tokens``; classical
            # chat models accept ``max_tokens``. We send both keys when set
            # so either dialect works at the proxy.
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        try:
            resp = await self.client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK + httpx raise many shapes
            raise BackendError(
                f"CodexBackend: SDK call failed: {exc}"
            ) from exc

        try:
            content = resp.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as exc:
            raise BackendError(
                f"CodexBackend: unexpected response shape: {exc}"
            ) from exc
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # OpenAI sometimes returns a list of content parts (vision-style);
        # join the .text fields.
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    chunks.append(t)
            else:
                t = getattr(part, "text", None)
                if isinstance(t, str):
                    chunks.append(t)
        return "".join(chunks)

    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """Best-effort cleanup of the dedicated httpx client (if we own it)."""
        http = self._http_client
        if http is None:
            return
        try:
            await http.aclose()
        except Exception:  # noqa: BLE001 — best-effort
            pass


__all__ = ["CodexBackend"]
