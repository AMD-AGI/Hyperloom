# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agentic Codex SDK backend for in-process specialist dispatch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.codex_session import (
    CodexSessionError,
    resolve_codex_sandbox_mode,
    run_codex_turn,
)
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.jsonio import extract_first_json_with_key
from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)

from .agent_role import DEFAULT_CODEX_MODEL
from .base import BackendError, BackendTurnResult, LLMCallFailed, parse_call_timeout_env, safe_int


_BARE_INTENTS_RE = re.compile(r'(\{.*?"intents".*\})', re.DOTALL)
_SPECIALIST_OUTPUT_INSTRUCTIONS = """
## Codex specialist output transport
The specialist protocol's `emit_intent` name describes the logical transport;
this Codex session does not expose that Claude MCP tool. Return the intent in
your final response as exactly one JSON object:

{
  "intents": [
    {
      "intent_type": "specialist_done",
      "payload": {
        "gap_canonical_id": "...",
        "domain": "...",
        "proposal_set": [],
        "empty": true,
        "summary": "..."
      }
    }
  ]
}

The payload must include every field required by the specialist prompt. Do not
put prose outside the JSON object.
""".strip()


@dataclass
class CodexAgentBackend:
    """Run a specialist through the Codex Agent SDK and validate its intent."""

    model: str = DEFAULT_CODEX_MODEL
    cwd: Path = field(default_factory=Path.cwd)
    writable_roots: tuple[Path, ...] = ()
    sandbox_mode: str = ""
    codex_bin: str = ""
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC",
            default=600.0,
        )
    )
    env: dict[str, str] | None = None
    name: str = "codex-agent"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize and secure the session-private runtime root."""
        self.cwd = Path(self.cwd).expanduser().resolve()
        try:
            self.cwd.mkdir(parents=True, mode=0o700, exist_ok=True)
            self.cwd.chmod(0o700)
        except OSError as exc:
            raise BackendError(f"cannot prepare Codex specialist cwd {self.cwd}: {exc}") from exc
        roots = tuple(Path(root).expanduser().resolve() for root in self.writable_roots)
        if self.cwd not in roots:
            roots = (self.cwd, *roots)
        self.writable_roots = roots

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        disallowed_tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Run one agentic Codex turn and parse its structured intent envelope.

        Args:
            prompt: The composed turn prompt; sent as the user turn.
            system_prompt: Sent as thread developer instructions alongside the
                specialist output contract.
            disallowed_tools: Unused. Codex containment is the sandbox preset;
                the Claude tool names have no counterpart here.
            max_turns: Unused. The SDK owns the agent loop within one turn.

        Returns:
            BackendTurnResult: The validated intents plus raw reply text and
            model/usage metadata.
        """
        try:
            resolved_sandbox = resolve_codex_sandbox_mode(
                sandbox_mode=self.sandbox_mode,
                env=self.env,
            )
        except CodexSessionError as exc:
            raise BackendError(redact_secret_values(str(exc))) from exc
        if resolved_sandbox == "read-only":
            raise BackendError(
                "Codex sandbox mode 'read-only' cannot run a specialist that must produce writable task results"
            )

        developer_instructions = "\n\n".join(
            part for part in ((system_prompt or "").strip(), _SPECIALIST_OUTPUT_INSTRUCTIONS) if part
        )
        user_prompt = prompt
        compatibility_prefix = f"{system_prompt}\n---\n" if system_prompt else ""
        if compatibility_prefix and user_prompt.startswith(compatibility_prefix):
            user_prompt = user_prompt[len(compatibility_prefix) :]
        try:
            sdk_result = await run_codex_turn(
                prompt=user_prompt,
                developer_instructions=developer_instructions,
                cwd=self.cwd,
                model=self.model,
                timeout_sec=self.call_timeout_s,
                writable_roots=self.writable_roots,
                sandbox_mode=resolved_sandbox,
                codex_bin=self.codex_bin,
                env=self.env,
            )
        except CodexSessionError as exc:
            raise LLMCallFailed(f"Codex Agent SDK specialist turn failed: {redact_secret_values(str(exc))}") from exc
        if sdk_result.error:
            raise LLMCallFailed("Codex Agent SDK specialist turn failed: " + redact_secret_values(sdk_result.error))

        envelope = extract_first_json_with_key(
            sdk_result.text,
            "intents",
            _BARE_INTENTS_RE,
            last=True,
        )
        if envelope is None:
            raise NoIntentEmitted("Codex Agent SDK specialist reply contained no parseable intents envelope")
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"Codex specialist intent envelope is invalid: {exc}") from exc
        if not intents:
            raise NoIntentEmitted("Codex specialist intent envelope contained no intents")

        usage = dict(sdk_result.usage or {})
        input_tokens = safe_int(usage.get("input_tokens"))
        output_tokens = safe_int(usage.get("output_tokens"))
        cache_read_tokens = safe_int(usage.get("cache_read_input_tokens"))
        reasoning_tokens = safe_int(usage.get("reasoning_output_tokens"))
        metadata: dict[str, Any] = {
            "model": self.model,
            "thread_id": sdk_result.thread_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            "error": "",
            "prompt": user_prompt,
            "response": sdk_result.text,
        }
        self.calls.append(
            {
                "model": self.model,
                "prompt_chars": len(user_prompt),
                "reply_chars": len(sdk_result.text),
                "intents": len(intents),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_tokens,
                "reasoning_output_tokens": reasoning_tokens,
                "thread_id": sdk_result.thread_id,
            }
        )
        return BackendTurnResult(
            intents=intents,
            raw_text=sdk_result.text,
            metadata=metadata,
        )


__all__ = ["CodexAgentBackend"]
