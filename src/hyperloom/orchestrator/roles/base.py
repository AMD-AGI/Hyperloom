# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend protocol — what the Coordinator needs from any LLM provider."""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from hyperloom.inference_optimizer.protocol.intent import Intent
from hyperloom_common.env import EnvValueError, env_float

log = logging.getLogger(__name__)


def parse_call_timeout_env(env_name: str, *, default: float) -> float:
    """Read a per-call wall-clock timeout in seconds from ``env_name``; ``default`` when it is unset or blank.

    Raises:
        EnvValueError: The variable is set to anything but a positive finite number.
    """
    value = env_float(env_name, default)
    if value <= 0 or not math.isfinite(value):
        raise EnvValueError(f"{env_name}={value!r} is not a positive finite number")
    return value


def build_chat_messages(system_prompt: str | None, user_content: str) -> list[dict[str, Any]]:
    """Assemble an OpenAI-style chat ``messages`` list."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


class BackendError(RuntimeError):
    """Backend invocation failed (network, schema, etc.)."""


class LLMCallFailed(BackendError):
    """The model request itself failed — no usable response came back."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential-backoff policy for transient LLM call failures."""

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    multiplier: float = 2.0
    jitter_s: float = 0.5

    @classmethod
    def from_env(
        cls,
        prefix: str = "INFERENCE_OPTIMIZER_LLM_RETRY",
    ) -> "RetryPolicy":
        """Build a policy from ``<prefix>_{ATTEMPTS,BASE_S,MAX_S,MULT,JITTER_S}``.

        Raises:
            EnvValueError: One of them is set to anything but a non-negative finite number.
        """
        d = cls()

        def _num(suffix: str, default: float) -> float:
            name = f"{prefix}_{suffix}"
            value = env_float(name, default)
            if value < 0 or not math.isfinite(value):
                raise EnvValueError(f"{name}={value!r} is not a non-negative finite number")
            return value

        return cls(
            max_attempts=max(1, int(_num("ATTEMPTS", d.max_attempts))),
            base_delay_s=_num("BASE_S", d.base_delay_s),
            max_delay_s=_num("MAX_S", d.max_delay_s),
            multiplier=_num("MULT", d.multiplier),
            jitter_s=_num("JITTER_S", d.jitter_s),
        )

    def delay_for(self, attempt: int) -> float:
        """Backoff delay (seconds) before retry ``attempt`` (1-based prior attempt)."""
        raw = self.base_delay_s * (self.multiplier ** max(0, attempt - 1))
        capped = min(self.max_delay_s, raw)
        if self.jitter_s > 0:
            capped += random.uniform(0.0, self.jitter_s)
        return max(0.0, capped)


async def retry_with_backoff(
    fn: Callable[[], Awaitable[Any]],
    *,
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Await ``fn()`` with bounded exponential-backoff retry on ``retry_on``."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except retry_on as exc:
            if attempt >= policy.max_attempts:
                raise
            delay = policy.delay_for(attempt)
            log.warning(
                "LLM call failed (attempt %d/%d: %r); retrying in %.2fs",
                attempt,
                policy.max_attempts,
                exc,
                delay,
            )
            await sleep(delay)


@dataclass
class BackendTurnResult:
    """One turn's output from a backend."""

    intents: list[Intent] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Async LLM backend protocol used by the Coordinator reactor loop."""

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> BackendTurnResult:
        """Run one logical turn for the given prompt and return its intents."""


__all__ = [
    "Backend",
    "BackendError",
    "BackendTurnResult",
    "LLMCallFailed",
    "RetryPolicy",
    "build_chat_messages",
    "parse_call_timeout_env",
    "retry_with_backoff",
]
