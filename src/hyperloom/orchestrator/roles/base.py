# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend protocol — what the Coordinator needs from any LLM provider."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from hyperloom.inference_optimizer.protocol.intent import Intent

log = logging.getLogger(__name__)


def parse_call_timeout_env(env_name: str, *, default: float) -> float:
    """Read a per-call wall-clock timeout from ``env_name``, default on miss/error."""
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "%s=%r is not a float; using default %.1fs",
            env_name,
            raw,
            default,
        )
        return default
    if value <= 0 or not math.isfinite(value):
        log.warning(
            "%s=%r is not a positive finite number; using default %.1fs",
            env_name,
            raw,
            default,
        )
        return default
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
        """Build a policy from ``<prefix>_{ATTEMPTS,BASE_S,MAX_S,MULT,JITTER_S}``."""
        d = cls()

        def _num(suffix: str, default: float, *, cast: Callable[[float], Any]):
            """Read ``<prefix>_<suffix>`` as a number, falling back to ``default``."""
            raw = os.environ.get(f"{prefix}_{suffix}")
            if raw is None or not raw.strip():
                return default
            try:
                val = float(raw)
            except ValueError:
                log.warning("%s_%s=%r invalid; using %s", prefix, suffix, raw, default)
                return default
            if not math.isfinite(val) or val < 0:
                log.warning("%s_%s=%r invalid; using %s", prefix, suffix, raw, default)
                return default
            return cast(val)

        return cls(
            max_attempts=max(1, int(_num("ATTEMPTS", d.max_attempts, cast=int))),
            base_delay_s=float(_num("BASE_S", d.base_delay_s, cast=float)),
            max_delay_s=float(_num("MAX_S", d.max_delay_s, cast=float)),
            multiplier=float(_num("MULT", d.multiplier, cast=float)),
            jitter_s=float(_num("JITTER_S", d.jitter_s, cast=float)),
        )

    def delay_for(self, attempt: int) -> float:
        """Backoff delay (seconds) before retry ``attempt`` (1-based prior attempt)."""
        raw = self.base_delay_s * (self.multiplier ** max(0, attempt - 1))
        capped = min(self.max_delay_s, raw)
        if self.jitter_s > 0:
            capped += random.uniform(0.0, self.jitter_s)
        return max(0.0, capped)

    def worst_case_backoff_sec(self) -> float:
        """Longest total backoff a full run of attempts can sleep through."""
        return max(0, self.max_attempts - 1) * (self.max_delay_s + max(0.0, self.jitter_s))


#: Multiple of a backend's own worst-case chain used as the caller's ceiling. The
#: caller's cap is a backstop for a backend that did not honour its own budget,
#: so it has to sit clear of every turn that budget still calls healthy.
TURN_BUDGET_HEADROOM: float = 2.0


def derive_turn_budget_sec(
    per_attempt_sec: Callable[[int], float],
    policy: RetryPolicy | None = None,
) -> float:
    """Wall-clock ceiling for one turn, read off the backend's own budget.

    Args:
        per_attempt_sec: Seconds attempt ``n`` (1-based) is allowed to spend.
        policy: The retry policy wrapped around those attempts, or ``None`` when
            the backend makes a single attempt.

    Returns:
        float: The longest turn the backend's configuration still calls healthy,
        with :data:`TURN_BUDGET_HEADROOM` on top.
    """
    attempts = 1 if policy is None else max(1, policy.max_attempts)
    work = sum(per_attempt_sec(n) for n in range(1, attempts + 1))
    backoff = 0.0 if policy is None else policy.worst_case_backoff_sec()
    return (work + backoff) * TURN_BUDGET_HEADROOM


async def retry_with_backoff(
    fn: Callable[[], Awaitable[Any]],
    *,
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Any:
    """Await ``fn()`` with bounded exponential-backoff retry on ``retry_on``."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except retry_on as exc:  # type: ignore[misc]
            if attempt >= policy.max_attempts:
                raise
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                try:
                    on_retry(attempt, exc, delay)
                except Exception:  # noqa: BLE001 — telemetry callback never fatal
                    pass
            log.warning(
                "LLM call failed (attempt %d/%d: %r); retrying in %.2fs",
                attempt,
                policy.max_attempts,
                exc,
                delay,
            )
            await sleep(delay)


def safe_int(value: Any) -> int:
    """Coerce a possibly-missing usage value to a non-negative int."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


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
        max_turns: int = 1,
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
    "safe_int",
]
