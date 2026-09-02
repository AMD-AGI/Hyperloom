# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HermesBackend — Coordinator intent transport through Hermes Agent."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import is_secret_shaped_env_name, redact_secret_values
from hyperloom.common.hermes_runtime import resolve_hermes_executable
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from ..trace.llm_trace import new_call_id
from .base import BackendError, BackendTurnResult, LLMCallFailed, parse_call_timeout_env
from .codex import build_output_instructions, decode_intent_envelope


def _json_reply(text: str) -> str:
    """Return the one JSON object from a Hermes final response."""

    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        json.loads(value)
        return value
    except ValueError:
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            candidate = value[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except ValueError:
                pass
    raise NoIntentEmitted(f"hermes reply did not contain one JSON object (chars={len(value)})")


@dataclass
class HermesBackend:
    """Stateless Hermes implementation of the Coordinator Backend protocol."""

    allowed_intents: frozenset[IntentType]
    model: str = "gpt-5.6-sol"
    cwd: Path = field(default_factory=Path.cwd)
    profile: str = "hyperloomfaithful"
    inference_provider: str = "openai-codex"
    hermes_bin: str = ""
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_HERMES_CALL_TIMEOUT_SEC",
            default=600.0,
        )
    )
    env: dict[str, str] | None = None

    name: str = "hermes"
    calls: list[dict[str, Any]] = field(default_factory=list)
    conversational = False

    def __post_init__(self) -> None:
        if not self.allowed_intents:
            raise BackendError("HermesBackend requires the emitting role's allowed_intents")
        self.cwd = Path(self.cwd).expanduser().resolve()
        self.cwd.mkdir(parents=True, exist_ok=True)
        executable = resolve_hermes_executable(self.hermes_bin)
        if not executable:
            raise BackendError("Hermes executable not found")
        self.hermes_bin = executable

    @property
    def needs_seed(self) -> bool:
        return True

    def needs_seed_for(self, _system_prompt: str | None) -> bool:
        return True

    def reset_conversation(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
        allow_no_intent: bool = False,
    ) -> BackendTurnResult:
        del tools, max_turns
        parts = [(system_prompt or "").strip()]
        if not allow_no_intent:
            parts.append(build_output_instructions(self.allowed_intents))
        parts.append(prompt)
        full_prompt = "\n\n".join(part for part in parts if part)
        argv = [
            self.hermes_bin,
            "--profile",
            self.profile,
            "--provider",
            self.inference_provider,
            "--model",
            self.model,
            "--safe-mode",
            "--toolsets",
            "todo",
            "-z",
            full_prompt,
        ]
        child_env = dict(os.environ)
        child_env.update(self.env or {})

        def _invoke() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                cwd=self.cwd,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=self.call_timeout_s,
                check=False,
            )

        try:
            completed = await asyncio.to_thread(_invoke)
        except subprocess.TimeoutExpired as exc:
            raise LLMCallFailed(f"Hermes turn timed out after {self.call_timeout_s:g}s") from exc
        except OSError as exc:
            raise BackendError(f"Hermes turn could not start: {exc}") from exc
        if completed.returncode != 0:
            detail = redact_secret_values((completed.stderr or "").strip()[-1000:])
            for key, secret in child_env.items():
                if secret and is_secret_shaped_env_name(key):
                    detail = detail.replace(secret, "[REDACTED]")
            raise LLMCallFailed(f"Hermes turn exited rc={completed.returncode}: {detail}")

        raw_text = completed.stdout or ""
        metadata: dict[str, Any] = {
            "model": self.model,
            "provider": self.inference_provider,
            "profile": self.profile,
            "call_id": new_call_id(),
            "prompt": prompt,
            "response": raw_text,
        }
        self.calls.append(
            {
                "model": self.model,
                "prompt_chars": len(prompt),
                "reply_chars": len(raw_text),
            }
        )
        if allow_no_intent:
            return BackendTurnResult(intents=[], raw_text=raw_text, metadata=metadata)
        envelope = decode_intent_envelope(_json_reply(raw_text))
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"hermes envelope invalid: {exc}") from exc
        return BackendTurnResult(intents=intents, raw_text=raw_text, metadata=metadata)


__all__ = ["HermesBackend"]
