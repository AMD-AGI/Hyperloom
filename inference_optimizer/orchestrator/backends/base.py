# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Backend protocol — what the Coordinator needs from any LLM provider.

Concrete implementations return a :class:`BackendTurnResult` carrying the
turn's intents; the Coordinator handles validation, PolicyGate, persistence.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from ...protocol.intent import Intent

log = logging.getLogger(__name__)


def parse_call_timeout_env(env_name: str, *, default: float) -> float:
    """Read a per-call wall-clock timeout from ``env_name``, default on miss/error.

    Returns ``default`` (logging a WARNING) when the env var is unset, empty,
    or not a positive finite float — a malformed knob must not be fatal.
    """
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "%s=%r is not a float; using default %.1fs",
            env_name, raw, default,
        )
        return default
    if value <= 0 or not math.isfinite(value):
        log.warning(
            "%s=%r is not a positive finite number; using default %.1fs",
            env_name, raw, default,
        )
        return default
    return value


class BackendError(RuntimeError):
    """Backend invocation failed (network, schema, etc.)."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential-backoff policy for transient LLM call failures (R6).

    A multi-day single-session run cannot afford to surface a transient gateway
    stall / 5xx / connection blip as a hard turn failure. ``max_attempts`` counts
    the FIRST try (so ``1`` disables retry). Delay grows
    ``base_delay_s * multiplier**(attempt-1)`` capped at ``max_delay_s`` plus up
    to ``jitter_s`` of randomization.
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    multiplier: float = 2.0
    jitter_s: float = 0.5

    @classmethod
    def from_env(
        cls, prefix: str = "INFERENCE_OPTIMIZER_LLM_RETRY",
    ) -> "RetryPolicy":
        """Build a policy from ``<prefix>_{ATTEMPTS,BASE_S,MAX_S,MULT,JITTER_S}``.

        Malformed values fall back to the dataclass default for that field.
        ``<prefix>_ATTEMPTS=1`` (or ``0``) disables retry.
        """
        d = cls()

        def _num(suffix: str, default: float, *, cast: Callable[[float], Any]):
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


async def retry_with_backoff(
    fn: Callable[[], Awaitable[Any]],
    *,
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Any:
    """Await ``fn()`` with bounded exponential-backoff retry on ``retry_on``.

    Re-raises the last exception once ``policy.max_attempts`` is exhausted. The
    ``sleep`` seam keeps tests deterministic (inject a no-op / fake clock).
    """
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
                attempt, policy.max_attempts, exc, delay,
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
    """Async LLM backend protocol used by the Coordinator reactor loop.

    Backends are stateful but each ``run`` invocation is one logical turn.
    """

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Run one logical turn for the given prompt and return its intents.

        Args:
            prompt (str): The user/turn prompt to send to the backend.
            system_prompt (str | None): Optional system prompt establishing the
                backend's role and rules for this turn.
            tools (list[str] | None): Optional list of tool names the backend is
                allowed to use this turn.
            max_turns (int): Maximum number of internal agentic sub-turns the
                backend may take to produce its result.

        Returns:
            BackendTurnResult: The intents emitted this turn plus any raw text
            and metadata.
        """


__all__ = [
    "Backend",
    "BackendError",
    "BackendTurnResult",
    "RetryPolicy",
    "parse_call_timeout_env",
    "retry_with_backoff",
]
